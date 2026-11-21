"""CLI:  python -m form2json <pliki/katalogi...> -o out/ [--ocr tesseract] [--no-vlm]

Dla każdego pliku zapisuje:
  <nazwa>.<ext>.fields.json  - ogólny JSON (opis pola -> rodzaj pola, sekcja, bbox, opcje)
  <nazwa>.<ext>.form.json    - docelowy FormDefinition
  <nazwa>.<ext>.debug.png    - (z --debug) podgląd wykrytych pól na obrazie
"""
from __future__ import annotations

import argparse
import glob
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
        if p.is_dir():           # wykluczenia tylko wewnątrz podanego katalogu, nie w jego przodkach
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED
                              and not (ex and ex in q.resolve().parents)
                              and "wizualizacja" not in q.name.lower()      # gotowe wizualizacje, nie formularze
                              and not any(x.name in ("form2json", ".venv") or x.name.startswith("wyniki")
                                          for x in q.relative_to(p).parents))
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


def _dpi(v: str) -> int:
    if not 72 <= int(v) <= 600:
        raise argparse.ArgumentTypeError("dpi musi być w zakresie 72-600")
    return int(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="form2json")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--ocr", default="auto", help="auto|tesseract|paddle|paddle-cpu|none (obrazy i skany)")
    ap.add_argument("--vlm", action="store_true", help="wymuś VLM (lokalny Ollama na GPU, config/local.toml)")
    ap.add_argument("--no-vlm", action="store_true", help="bez VLM - tylko geometria + OCR (CPU)")
    ap.add_argument("--vlm-model", default=None, help="np. qwen3-vl:8b, qwen3-vl:30b")
    ap.add_argument("--rebuild", action="store_true", help="wejście: *.fields.json - przebuduj tylko *.form.json")
    ap.add_argument("--config", default=None, help="plik konfiguracji VLM (domyślnie config/local.toml)")
    ap.add_argument("--tabs", default="auto", choices=["auto", "single"])
    ap.add_argument("--dpi", type=_dpi, default=150)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(argv)
    if a.rebuild:
        a.no_vlm, a.ocr = True, "none"

    from .ocr import get_ocr
    ocr = get_ocr(a.ocr)
    from .llm import VLMRefiner, load_config
    cfg = load_config(a.config)
    vlm = None
    if (a.vlm or cfg["enabled"]) and not a.no_vlm:
        vlm = VLMRefiner(cfg, model=a.vlm_model)
        if vlm.available():
            print(f"VLM: {vlm.model} @ {vlm.base} ({vlm.backend})")
        else:
            print(f"VLM niedostępny ({vlm.base}) - tryb CPU: geometria + OCR", file=sys.stderr)
            if a.vlm:
                return 2
            vlm = None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    # --rebuild: tylko ponowne zbudowanie *.form.json z istniejących *.fields.json (bez OCR/VLM)
    inputs = sorted(Path(x) for pat in a.inputs for x in (glob.glob(pat) if glob.has_magic(pat) else [pat])) \
        if a.rebuild else list(iter_inputs(a.inputs, out))
    if not inputs:
        print("Brak plików wejściowych o obsługiwanych rozszerzeniach", file=sys.stderr)
        return 2
    seen: dict[str, Path] = {}
    for p in inputs:
        name = p.name.removesuffix(".fields.json") if a.rebuild else p.name
        if name in seen:          # ta sama nazwa w dwóch katalogach nadpisałaby wyniki
            print(f"[ERR] {p}: nazwa wyniku koliduje z {seen[name]} - pomijam", file=sys.stderr)
            rc = 1
            continue
        seen[name] = p
        try:
            if a.rebuild:
                from .pipeline import reanalyze
                g = reanalyze(json.loads(Path(p).read_text("utf-8")))
            else:
                g = extract(p, ocr=ocr, vlm=vlm, dpi=a.dpi)
            (out / f"{name}.fields.json").write_text(json.dumps(g, ensure_ascii=False, indent=2), "utf-8")
            fd = to_form_definition(g, a.tabs)
            (out / f"{name}.form.json").write_text(json.dumps(fd, ensure_ascii=False, indent=2), "utf-8")
            if a.debug:
                draw_debug(p, g, out / f"{name}.debug.png", a.dpi, ocr)
            from .validate import validate
            errs, warns = validate(fd)
            (out / f"{name}.simple.json").write_text(json.dumps(
                {"title": g["title"], "pola": g["simple"]}, ensure_ascii=False, indent=2), "utf-8")
            print(f"[OK] {name}: {len(g['fields'])} pól ({', '.join(g['text_source'])}); "
                  f"walidacja: {len(errs)} błędów, {len(warns)} ostrzeżeń")
            for e in errs:
                print(f"     BŁĄD: {e}", file=sys.stderr)
        except Exception as e:
            rc = 1
            print(f"[ERR] {name}: {e}", file=sys.stderr)
            traceback.print_exc()
    return rc


if __name__ == "__main__":
    sys.exit(main())
