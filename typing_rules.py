"""Reguły określania rodzaju pola (deterministyczne, PL).

Kolejność: opis pola > liczba kratek > kształt ramki > kontekst sekcji.
Wynik VLM (jeśli używany) i tak przechodzi przez te reguły jako walidacja.
"""
from __future__ import annotations

import re
import unicodedata


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c)).replace("ł", "l")
    return re.sub(r"\s+", " ", s)


RULES: list[tuple[str, str, str | None]] = [
    # (regex na znormalizowanym opisie, typ, część adresu)
    (r"e-?mail|poczty elektronicznej|adres elektroniczn", "email", None),
    (r"telefon|\btel\b|nr tel|komork", "phone", None),
    (r"\bpesel\b", "pesel", None),
    (r"\bnip\b", "nip", None),
    (r"\bregon\b", "regon", None),
    (r"rachunk|\biban\b|nr konta|numer konta|konta bankowego", "iban", None),
    (r"kod pocztowy|kod\b", "postal_code", "postal_code"),
    (r"podpis", "signature", None),
    (r"\bdata\b|\bdnia\b|dzien|dd-mm|rrrr|data urodzenia|urodzon|w terminie|\bod$|\bdo$", "date", None),
    (r"miejscowos|miasto\b", "address", "city"),
    (r"\bulica\b|\bul\.?\b", "address", "street"),
    (r"nr domu|numer domu|nr budynku|numer budynku|nr posesji", "address", "building"),
    (r"nr lokalu|numer lokalu|nr mieszkania", "address", "apartment"),
    (r"\bpoczta\b", "address", "post_office"),
    (r"\bgmina\b", "address", "commune"),
    (r"\bpowiat\b", "address", "county"),
    (r"wojewodztw", "address", "voivodeship"),
    (r"\bkraj\b", "address", "country"),
    (r"\badres\b(?!.*(e-?mail|poczty elektr))", "address", "full"),
    (r"kwot[ayi]|kwoci|wysokosc (oplaty|kwoty|nadplaty)|\bzl\b|liczba|ilosc|powierzchni|\bm2\b|\bha\b|"     # "m²" po norm() (NFKD) to "m2"
     r"\bwiek\b|rok\b|\bsztuk", "number", None),
    (r"uzasadnienie|opis\b|uwagi|wyjasnieni|informacje dodatkowe|cel ", "textarea", None),
]


def infer_type(label: str, section: str, kind: str, n_cells: int = 0, parts: list[int] | None = None,
               rel_height: float = 1.0, hint: str = "") -> tuple[str, str | None]:
    lab = norm(label)
    parts = parts or []

    if kind == "checkbox":
        return "checkbox", None

    # kratki - liczba znaków bywa bardziej wiarygodna niż opis
    if parts == [2, 2, 4] or parts == [4, 2, 2]:
        return "date", None
    if parts == [2, 3]:
        return "postal_code", "postal_code"

    for rx, typ, part in RULES:
        if re.search(rx, lab):
            if typ == "textarea" and rel_height < 1.8 and kind not in ("box", "captioned"):
                typ = "string"
            return typ, part

    if re.search(r"(DD|RRRR)", hint):
        return "date", None
    if n_cells == 11:
        return "pesel", None
    if n_cells == 10:
        return "nip", None
    if n_cells == 26:
        return "iban", None
    if n_cells and n_cells <= 4:
        return "number", None

    sec = norm(section)
    if "zalacznik" in sec:
        return "file", None
    if rel_height >= 2.6 and kind in ("box", "captioned"):
        return "textarea", None
    return "string", None


# Uproszczony "rodzaj pola" (o który pytano wprost): checkbox/string/number/email/address/...
SIMPLE_TYPE = {
    "string": "string", "textarea": "string", "signature": "string",
    "number": "number", "pesel": "number", "nip": "number", "regon": "number", "iban": "number",
    "email": "email", "phone": "phone", "date": "date",
    "address": "address", "postal_code": "address",
    "checkbox": "checkbox", "radio": "checkbox", "file": "checkbox",
}
