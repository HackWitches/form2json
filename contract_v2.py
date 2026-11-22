"""Wynik form2json (*.fields.json) -> "kontrakt v2" z service, żeby rysować go skryptem
cyfryzacja-formularzy/dev/visualize_output.py (ten sam podgląd dla obu rozwiązań).

    python -m form2json.contract_v2 wyniki_vlm --service ../cyfryzacja-formularzy/service [--out wyniki_vlm/kontrakt]

Kontrakt cytuje numery linii (L001...) z service/app/ingest/lines.py; pole dostaje linie, które przecina jego ramka
lub ramka opisu (bez ramki: linia z opisem). Tylko pliki NN.pdf (visualize_output.py bierze PDF
z annotations/forms/NN_*.json).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from .build import TYPE_MAP
from .loaders import DEFAULT_DPI

# role pokryte przez bazę formularza w service (strona Dane, Kontakt, FINISH) -> "pominiete" z powodem, jak w service
POMIN = {"applicant_name": "dane bazy", "applicant_address": "dane bazy", "applicant_id": "dane bazy",
         "correspondence_address": "dane bazy", "contact": "dane bazy", "place_date": "data złożenia",
         "company": "dane bazy", "signature": "podpis", "office_use": "dla urzędu"}
PT = 72 / DEFAULT_DPI  # piksele rastra (150 dpi) -> punkty PDF


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("ł", "l").replace("Ł", "L")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "pole"


def cited(f: dict, lines: list[dict]) -> list[str]:
    """Linie, które przecina ramka pola lub opisu (obie zmniejszone o 1 pt, żeby nie łapać sąsiadów).
    Pole bez ramki (VLM nie dopasował go do geometrii): linia strony z największą częścią słów opisu."""
    boxes = [[v * PT for v in b] for b in (f.get("bbox"), f.get("label_bbox")) if b]
    page = [ln for ln in lines if ln["page"] == f.get("page", 0) + 1]
    if boxes:
        hit = lambda a, b: min(a[2], b[2]) - max(a[0], b[0]) > 1 and min(a[3], b[3]) - max(a[1], b[1]) > 1  # noqa: E731
        return [ln["id"] for ln in page if any(hit(ln["bbox"], b) for b in boxes)]
    stems = lambda t: {w[:5] for w in slug(t).split("_") if len(w) > 2}  # noqa: E731  "rachunku" ~ "rachunek"
    words = stems(f["label"])
    score = lambda ln: len(words & stems(ln["text"])) / len(words) if words else 0  # noqa: E731
    best = max(page, key=score, default=None)
    return [best["id"]] if best and score(best) >= 0.5 else []  # ponytail: przybliżenie, tylko do podglądu


def convert(g: dict, lines: list[dict]) -> dict:
    tech: dict[str, str] = {}  # opis -> techName (warunki wskazują pole po opisie)
    used: set[str] = set()
    secs: dict[str, list] = {}
    skips = []
    for f in g["fields"]:
        ids = cited(f, lines)
        if f.get("role") in POMIN:
            skips.append({"linie": ids, "powod": POMIN[f["role"]]})
            continue
        name = slug(f["label"])
        base, k = name, 1
        while name in used:  # ten sam opis kilka razy (np. "tak"/"nie" w kolejnych pytaniach)
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        tech.setdefault(f["label"], name)
        sec = secs.setdefault(f.get("section") or "Przedmiot wniosku", [])
        sec.append({"wzorzec": "POLE", "techName": name, "tytul": f["label"],
                    "typ": TYPE_MAP.get(f["type"], "TEXT_FIELD"), "wymagane": bool(f.get("required")),
                    "opcje": f.get("options", []), "linie": ids, "_dep": (f.get("depends_on"), f.get("depends_value"))})
    for fs in secs.values():  # warunki dopiero gdy znane są wszystkie techName
        for p in fs:
            dep, val = p.pop("_dep")
            p["warunek"] = {"pole": tech[dep], "wartosc": val} if dep in tech else None
    return {
        "tytul": g.get("title", ""),
        "adresat": {"nazwa": g.get("addressee") or "", "rodzaj": ""},
        "warianty": {"podmiot": bool(g.get("entities", {}).get("company")), "pelnomocnik": True},
        "strony": [{"tytul": "Przedmiot wniosku", "warunek": None,  # ponytail: jedna strona; podział robi build.py
                    "sekcje": [{"tytul": s, "techName": slug(s), "powtarzalna": False, "warunek": None, "pola": fs}
                               for s, fs in secs.items()]}] if secs else [],
        "pominiete": skips,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path, help="katalog z NN.pdf.fields.json")
    ap.add_argument("--service", type=Path, required=True, help="katalog cyfryzacja-formularzy/service")
    ap.add_argument("--out", type=Path, help="domyślnie <results>/kontrakt")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.service.resolve()))
    from app.ingest.lines import lines  # service: ta sama numeracja linii co w visualize_output.py
    out = a.out or a.results / "kontrakt"
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(a.results.glob("[0-9][0-9].pdf.fields.json")):
        g = json.loads(p.read_text("utf-8"))
        c = convert(g, lines(g["source"]))
        (out / f"{p.name[:2]}.json").write_text(json.dumps(c, ensure_ascii=False, indent=2), "utf-8")
        n = sum(len(s["pola"]) for st in c["strony"] for s in st["sekcje"])
        print(f"[OK] {p.name} -> {p.name[:2]}.json: {n} pól, {len(c['pominiete'])} pominiętych")
    return 0


if __name__ == "__main__":
    sys.exit(main())
