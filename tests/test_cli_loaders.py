import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import fixture, needs_soffice
from form2json import loaders, ocr
from form2json.cli import main

ROOT = Path(loaders.__file__).parents[1]          # katalog zawierający pakiet form2json


def _run(*args):
    return main([*map(str, args), "--no-vlm", "--ocr", "none"])


def test_rebuild_accepts_absolute_glob(tmp_path):
    out = tmp_path / "o"
    assert _run(fixture("f1.pdf"), "-o", out) == 0
    assert main(["--rebuild", str(out / "*.fields.json"), "-o", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "f1.pdf.form.json").exists()


@pytest.mark.parametrize("parent", ["form2json", "wyniki2025"])
def test_inputs_under_excluded_ancestor_names_are_processed(tmp_path, parent):
    d = tmp_path / parent / "in"
    d.mkdir(parents=True)
    shutil.copy(fixture("f1.pdf"), d)
    assert _run(d, "-o", tmp_path / "o") == 0
    assert (tmp_path / "o" / "f1.pdf.form.json").exists()


def test_output_name_collision_is_reported(tmp_path):
    for sub, src in (("a", "f1.pdf"), ("b", "f5.PDF")):
        (tmp_path / "in" / sub).mkdir(parents=True)
        shutil.copy(fixture(src), tmp_path / "in" / sub / "wniosek.pdf")
    assert _run(tmp_path / "in", "-o", tmp_path / "o") == 1


def test_no_inputs_and_bad_dpi(tmp_path):
    assert _run(tmp_path, "-o", tmp_path / "o") == 2
    with pytest.raises(SystemExit):
        _run(fixture("f1.pdf"), "-o", tmp_path / "o", "--dpi", "0")


def test_decompression_bomb_rejected(tmp_path):
    p = tmp_path / "bomb.png"
    cv2.imwrite(str(p), np.zeros((9000, 10000), np.uint8))          # 90 MP, ~90 KB
    code = ("import sys; from pathlib import Path; from form2json.loaders import imread_unicode\n"
            "try: imread_unicode(Path(sys.argv[1]))\nexcept ValueError: sys.exit(3)")
    r = subprocess.run([sys.executable, "-c", code, str(p)], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 3


@needs_soffice
def test_libreoffice_profile_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    src = tmp_path / "in.docx"
    shutil.copy(fixture("f5.docx"), src)
    loaders.office_to_pdf(src, tmp_path)
    assert not list(tmp_path.glob("lo_profile_*"))


def test_ocr_keeps_low_confidence_leaders(monkeypatch):
    t = ocr.TesseractBackend.__new__(ocr.TesseractBackend)
    t.lang, t.config, t.target_width = "pol", "", 100

    class Fake:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(*a, **k):
            return {"text": ["Nazwisko", "...........", "x"], "conf": ["90", "0", "5"],
                    "left": [0, 50, 90], "top": [0, 0, 0], "width": [40, 40, 5], "height": [10, 10, 10]}

    t.t = Fake
    assert [w.text for w in t(np.full((20, 100, 3), 255, np.uint8))] == ["Nazwisko", "..........."]
