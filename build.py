"""Ogólny JSON pól -> docelowy FormDefinition (tabs / questions / flow).

Konwencje przejęte z plików referencyjnych (konfiguracja_form_*.json):
  * każda zakładka (tab) zawiera SECTION-y; pola siedzą w `repeatingSectionGrid`
  * siatka ma 3 kolumny, każde pole zajmuje 1 wiersz x 3 kolumny
  * wysokość sekcji = liczba pól + 1
  * ostatnia zakładka typu FINISH z podglądem wniosku
  * flow.links łączy zakładki po kolei
"""
from __future__ import annotations

import re
import uuid
from collections import OrderedDict

from .pipeline import strip_numbering
from .typing_rules import norm

# rodzaj pola -> typ komponentu w docelowym systemie
TYPE_MAP = {
    "string": "TEXT_FIELD", "email": "TEXT_FIELD", "phone": "TEXT_FIELD", "date": "TEXT_FIELD",
    "address": "TEXT_FIELD", "postal_code": "TEXT_FIELD", "pesel": "TEXT_FIELD", "nip": "TEXT_FIELD",
    "regon": "TEXT_FIELD", "iban": "TEXT_FIELD",
    "number": "NUMBER", "textarea": "TEXTAREA",
    "radio": "RADIO", "checkbox": "CHECKBOX", "file": "FILE_UPLOADER",
}
# podpowiedzi formatu pokazywane jako tooltip
TOOLTIP = {
    "date": "Format: DD-MM-RRRR", "postal_code": "Format: 00-000", "pesel": "11 cyfr",
    "nip": "10 cyfr", "regon": "9 lub 14 cyfr", "iban": "26 cyfr numeru rachunku",
    "email": "np. jan.kowalski@example.pl", "phone": "np. 600 700 800",
}
SKIP_TYPES = {"signature"}                       # podpis zastępuje podpis elektroniczny
SKIP_SECTIONS = re.compile(r"instrukcj|pouczeni|podstawa prawna|wypelnia urzad|adnotacj", re.I)


def tech(s: str) -> str:
    s = norm(s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:60] or "pole"


class _Names:
    def __init__(self):
        self.used: set[str] = set()

    def __call__(self, s: str) -> str:
        base = tech(s)
        n, k = base, 2
        while n in self.used:
            n, k = f"{base}_{k}", k + 1
        self.used.add(n)
        return n


def _field(f: dict, order: int, names: _Names) -> dict:
    t = f.get("type", "string")
    out = {
        "id": str(uuid.uuid4()),
        "type": TYPE_MAP.get(t, "TEXT_FIELD"),
        "techName": names(f["label"]),
        "title": f["label"],
        "description": "",
        "options": [{"id": o, "name": o, "techName": tech(o)} for o in f.get("options", [])],
        "validation": [],
        "readOnly": False,
        "repeatingSectionGrid": [],
        "states": [],
        "questionOrder": order,
        "tooltipMessage": f.get("hint") or TOOLTIP.get(t, ""),
    }
    if f.get("prefilled"):
        out["defaultValue"] = f["prefilled"]
    return out


def to_form_definition(generic: dict, tabs_mode: str = "top-section") -> dict:
    names = _Names()
    fields = [f for f in generic["fields"] if f.get("type") not in SKIP_TYPES]

    # --- grupowanie: zakładka -> sekcja -> pola
    tabs: "OrderedDict[str, OrderedDict[str, list[dict]]]" = OrderedDict()
    for f in fields:
        path = [p for p in (f.get("section") or "").split(" > ") if p]
        if any(SKIP_SECTIONS.search(norm(p)) for p in path):
            continue
        top = strip_numbering(path[0]) if path else "Dane"
        leaf = strip_numbering(path[-1]) if path else "Dane"
        tab = top if tabs_mode == "top-section" else "Dane"
        tabs.setdefault(tab, OrderedDict()).setdefault(leaf, []).append(f)

    out_tabs = []
    for tab_title, secs in tabs.items():
        questions, y = [], 0
        for sec_title, fs in secs.items():
            grid = [{"x": 0, "y": i, "rows": 1, "cols": 3, "field": _field(f, i, names)}
                    for i, f in enumerate(fs)]
            rows = len(grid) + 1
            questions.append({
                "x": 0, "y": y, "rows": rows, "cols": 3,
                "field": {
                    "id": str(uuid.uuid4()), "type": "SECTION", "techName": names(sec_title),
                    "title": sec_title, "description": "<html><head></head><body></body></html>",
                    "options": [], "validation": [], "readOnly": False,
                    "repeatingSectionGrid": grid, "states": [], "questionOrder": len(questions),
                    "tooltipMessage": "",
                },
            })
            y += rows
        out_tabs.append({"id": str(uuid.uuid4()), "title": tab_title, "type": "MAIN", "questions": questions})

    # zgoda RODO + strona końcowa (jak w materiałach referencyjnych)
    finish = {"id": str(uuid.uuid4()), "title": "Strona końcowa", "type": "FINISH", "questions": [{
        "x": 0, "y": 0, "rows": 2, "cols": 3,
        "field": {"id": str(uuid.uuid4()), "type": "SECTION", "techName": names("podglad_uzupelnionego_wniosku"),
                  "title": "Podgląd uzupełnionego wniosku", "description": "<html><head></head><body></body></html>",
                  "options": [], "validation": [], "readOnly": False, "states": [], "questionOrder": 0,
                  "tooltipMessage": "",
                  "repeatingSectionGrid": [{"x": 0, "y": 0, "rows": 1, "cols": 3, "field": {
                      "id": str(uuid.uuid4()), "type": "TEXT_BLOCK", "techName": names("podglad"),
                      "title": "", "description": "", "options": [], "validation": [], "readOnly": False,
                      "repeatingSectionGrid": [], "states": [], "questionOrder": 0, "tooltipMessage": ""}}]},
    }]}
    out_tabs.append(finish)

    links = [{"id": str(uuid.uuid4()), "source": a["id"], "target": b["id"]}
             for a, b in zip(out_tabs, out_tabs[1:])]
    return {"title": generic.get("title", ""), "tabs": out_tabs, "flow": {"links": links}}
