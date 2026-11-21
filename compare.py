"""Porównanie wygenerowanego FormDefinition z plikiem referencyjnym (konfiguracja_form_*.json).

    python -m form2json.compare wynik.form.json referencja.json

Liczy pokrycie pól (po znormalizowanej etykiecie + typie komponentu) oraz zgodność typów.
"""
from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher

from .typing_rules import norm

STOP = {"jesli", "jezeli", "ktorego", "dotyczy", "wybierz", "mamy", "zrobic", "plik", "pole", "przez"}


def fields(fd: dict, subject_only: bool = False) -> list[tuple[str, str, str]]:
    """subject_only: tylko strony przedmiotu wniosku (bez 'Dane', 'Kontakt...' i FINISH)."""
    out = []
    for t in fd["tabs"]:
        if subject_only and (t["type"] == "FINISH" or t is fd["tabs"][0] or t is fd["tabs"][-2]):
            continue
        for q in t["questions"]:
            for g in q["field"].get("repeatingSectionGrid", []):
                f = g["field"]
                out.append((norm(f.get("title") or f.get("techName", "")), f["type"], q["field"]["title"]))
    return out


def compare(ours: dict, ref: dict) -> dict:
    a, b = fields(ours), fields(ref)
    used, hits, type_ok = set(), [], 0
    for lab, typ, sec in b:
        best, score = None, 0.0
        for i, (l2, t2, s2) in enumerate(a):
            if i in used:
                continue
            s = SequenceMatcher(None, lab, l2).ratio() + (0.15 if norm(sec) == norm(s2) else 0)
            if s > score:
                best, score = i, s
        if best is not None and score >= 0.75:
            used.add(best)
            hits.append((lab, a[best][0], typ, a[best][1]))
            type_ok += typ == a[best][1]
    return {"ref_fields": len(b), "our_fields": len(a), "matched": len(hits),
            "recall": round(len(hits) / max(1, len(b)), 3), "precision": round(len(hits) / max(1, len(a)), 3),
            "type_accuracy": round(type_ok / max(1, len(hits)), 3),
            "subject_recall": _subject_recall(ours, ref),
            "missing": [f"{l} ({t}) [{s}]" for l, t, s in b if not any(h[0] == l for h in hits)]}


def _subject_recall(ours: dict, ref: dict) -> float:
    a, b = fields(ours, True), fields(ref, True)
    used, hit = set(), 0
    stems = lambda s: {w[:5] for w in re.findall(r"[a-z0-9]+", s) if len(w) > 3 and w not in STOP}

    def sim(x: str, y: str) -> float:      # podobieństwo napisów lub wspólne rdzenie słów (PL: odmiana)
        sx, sy = stems(x), stems(y)
        ov = len(sx & sy) / max(1, min(len(sx), len(sy))) if sx and sy else 0.0
        return max(SequenceMatcher(None, x, y).ratio(), 0.6 * ov + 0.2 * (ov > 0))

    for lab, _, _ in b:
        cand = [(sim(lab, l2), i) for i, (l2, _, _) in enumerate(a) if i not in used]
        if cand and max(cand)[0] >= 0.6:
            used.add(max(cand)[1])
            hit += 1
    return round(hit / max(1, len(b)), 3)


if __name__ == "__main__":
    r = compare(json.load(open(sys.argv[1], encoding="utf-8")), json.load(open(sys.argv[2], encoding="utf-8")))
    print(json.dumps(r, ensure_ascii=False, indent=2))
