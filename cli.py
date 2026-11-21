"""CLI:  python -m form2json <pliki/katalogi...> -o out/ [--ocr paddle] [--vlm]

Dla każdego pliku zapisuje:
  <nazwa>.<ext>.fields.json  - ogólny JSON (opis pola -> rodzaj pola, sekcja, bbox, opcje)
  <nazwa>.<ext>.form.json    - docelowy FormDefinition
  <nazwa>.<ext>.debug.png    - (z --debug) podgląd wykrytych pól na obrazie
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .build import to_form_definition
from .loaders import IMAGE_EXT, OFFICE_EXT
from .pipeline import extract

SUPPORTED = IMAGE_EXT | OFFICE_EXT | {".pdf"}


def iter_inputs(paths: list[str], exclude: Path | None = None):
    ex = exclude.resolve() if exclude else None
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED
                              and not (ex and ex in q.resolve().parents))
        elif p.suffix.lower() in SUPPORTED:
            yield p


def draw_debug(path: Path, generic: dict, out: Path, dpi: int, ocr) -> None:
    import cv2
    from .loaders import load_document
    doc = load_document(path, dpi=dpi, ocr=ocr)
    colors = {"checkbox": (0, 160, 0), "comb": (200, 0, 200), "box": (0, 0, 255), "dots": (255, 120, 0),
              "underline": (0, 140, 255), "captioned": (160, 0, 0)}
    for p in doc.pages:
        img = p.image.copy()
        for f in generic["fields"]:
            if f.get("page", 0) != p.index or "bbox" not in f:
                continue
            c = colors.get(f.get("detector", "").split("+")[0].split("-")[0], (0, 0, 0))
            x0, y0, x1, y1 = f["bbox"]
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
            if "label_bbox" in f:
                a = f["label_bbox"]
                cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (180, 180, 180), 1)
                cv2.line(img, (a[2], (a[1] + a[3]) // 2), (x0, (y0 + y1) // 2), (180, 180, 180), 1)
            cv2.putText(img, f["type"], (x0 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(out.with_name(out.name.replace(".debug.png", f".p{p.index + 1}.debug.png"))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="form2json")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--ocr", default="none", help="none|paddle|paddle-cpu|tesseract (obrazy i skany)")
    ap.add_argument("--vlm", action="store_true", help="weryfikacja przez VLM na DGX (config/dgx.toml)")
    ap.add_argument("--no-vlm", action="store_true", help="wyłącz VLM nawet gdy enabled=true w configu")
    ap.add_argument("--dgx-config", default=None, help="ścieżka do pliku konfiguracji DGX (domyślnie config/dgx.toml)")
    ap.add_argument("--tabs", default="top-section", choices=["top-section", "single"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(argv)

    from .ocr import get_ocr
    ocr = get_ocr(a.ocr)
    from .llm import VLMRefiner, load_config
    cfg = load_config(a.dgx_config)
    vlm = VLMRefiner(cfg) if (a.vlm or cfg["enabled"]) and not a.no_vlm else None
    if vlm:
        print(f"VLM: {vlm.model} @ {vlm.base}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for p in iter_inputs(a.inputs, out):
        name = p.name
        try:
            g = extract(p, ocr=ocr, vlm=vlm, dpi=a.dpi)
            (out / f"{name}.fields.json").write_text(json.dumps(g, ensure_ascii=False, indent=2), "utf-8")
            fd = to_form_definition(g, a.tabs)
            (out / f"{name}.form.json").write_text(json.dumps(fd, ensure_ascii=False, indent=2), "utf-8")
            if a.debug:
                draw_debug(p, g, out / f"{name}.debug.png", a.dpi, ocr)
            print(f"[OK] {name}: {len(g['fields'])} pól ({', '.join(g['text_source'])})")
        except Exception as e:
            rc = 1
            print(f"[ERR] {name}: {e}", file=sys.stderr)
            traceback.print_exc()
    return rc


if __name__ == "__main__":
    sys.exit(main())
