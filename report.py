"""Raport zbiorczy dla katalogu wyników:  python -m form2json.report wyniki [--ref <katalog z konfiguracja_form_*.json>]

Dla każdego pliku: liczba pól wykrytych / komponentów w FormDefinition, wynik walidacji, a dla formularzy
z plikiem referencyjnym - pokrycie (recall/precision) i zgodność typów. Zapisuje wyniki/RAPORT.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .compare import compare, fields
from .typing_rules import norm
from .validate import validate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=".")
    a = ap.parse_args(argv)
    out = Path(a.dir)
    refs = {}
    for r in Path(a.ref).rglob("konfiguracja_form_*.json"):
        d = json.loads(r.read_text("utf-8"))
        refs[norm(d["title"])] = d
    rows, by_form = [], defaultdict(list)
    for p in sorted(out.glob("*.form.json")):
        fd = json.loads(p.read_text("utf-8"))
        gp = p.with_name(p.name.replace(".form.json", ".fields.json"))
        g = json.loads(gp.read_text("utf-8")) if gp.exists() else {"fields": [], "text_source": ["?"]}
        e, w = validate(fd)
        name = p.name.removesuffix(".form.json")
        stem, ext = (name.rsplit(".", 1) + [""])[:2]
        n_comp = len(fields(fd))
        ref = next((v for k, v in refs.items() if k in norm(stem) or norm(stem) in k), None)
        cmp = compare(fd, ref) if ref else None
        rows.append((stem, ext.lower(), len(g["fields"]), n_comp, len(e), cmp, ",".join(g["text_source"])))
        by_form[stem].append(len(g["fields"]))
    lines = ["# Raport konwersji formularzy", "",
             "| formularz | format | źródło tekstu | pola (ogólny JSON) | komponenty FormDefinition | błędy walidacji "
             "| recall vs ref | recall: przedmiot wniosku | precision vs ref | zgodność typów |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for stem, ext, nf, nc, ne, cmp, src in rows:
        c = (f"{cmp['recall']:.0%} | {cmp['subject_recall']:.0%} | {cmp['precision']:.0%} | "
             f"{cmp['type_accuracy']:.0%}" if cmp else "– | – | – | –")
        lines.append(f"| {stem[:60]} | {ext} | {src} | {nf} | {nc} | {ne} | {c} |")
    ok = sum(1 for r in rows if r[4] == 0)
    lines += ["", f"**Poprawnych (0 błędów walidacji): {ok}/{len(rows)}**", "",
              "## Spójność między formatami (liczba pól ogólnego JSON)", "", "| formularz | min | max | formaty |",
              "|---|---|---|---|"]
    for stem, ns in by_form.items():
        lines.append(f"| {stem[:60]} | {min(ns)} | {max(ns)} | {len(ns)} |")
    (out / "RAPORT.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
