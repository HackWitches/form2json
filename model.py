"""Wspólne struktury danych pipeline'u.

Wszystkie współrzędne są w pikselach strony zrenderowanej w `Page.dpi`
(układ: lewy górny róg = 0,0).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def contains(self, o: "Box", tol: float = 2) -> bool:
        return (o.x0 >= self.x0 - tol and o.y0 >= self.y0 - tol
                and o.x1 <= self.x1 + tol and o.y1 <= self.y1 + tol)

    def v_overlap(self, o: "Box") -> float:
        """Część wysokości mniejszego boxa pokrywająca się z drugim (0..1)."""
        inter = min(self.y1, o.y1) - max(self.y0, o.y0)
        return max(0.0, inter) / max(1.0, min(self.h, o.h))

    def h_overlap(self, o: "Box") -> float:
        inter = min(self.x1, o.x1) - max(self.x0, o.x0)
        return max(0.0, inter) / max(1.0, min(self.w, o.w))

    def union(self, o: "Box") -> "Box":
        return Box(min(self.x0, o.x0), min(self.y0, o.y0),
                   max(self.x1, o.x1), max(self.y1, o.y1))

    def as_list(self) -> list[int]:
        return [round(self.x0), round(self.y0), round(self.x1), round(self.y1)]


@dataclass
class Word:
    """Słowo/fragment tekstu z pozycją (z warstwy tekstowej PDF lub z OCR)."""
    text: str
    box: Box
    bold: bool = False
    size: float = 0.0          # wysokość fontu w px
    conf: float = 1.0


@dataclass
class Page:
    index: int
    image: np.ndarray          # BGR, uint8
    dpi: int
    words: list[Word] = field(default_factory=list)
    text_source: str = "none"  # "pdf" | "ocr:<backend>" | "none"

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


@dataclass
class Document:
    path: str
    pages: list[Page]
    native_text: str = ""      # tekst natywny (docx/xlsx/ods) jako kontekst dla LLM
    title: str = ""


# --------------------------------------------------------------------------
# Wynik pośredni: ogólne pary klucz -> rodzaj pola
# --------------------------------------------------------------------------

FIELD_TYPES = (
    "string", "textarea", "number", "email", "phone", "date",
    "address", "postal_code", "pesel", "nip", "regon", "iban",
    "checkbox", "radio", "signature", "file",
)


@dataclass
class FieldCandidate:
    label: str                         # "Opis pola"
    type: str = "string"               # "Rodzaj pola"
    section: str = ""                  # ścieżka sekcji, np. "2. Dane > 2.1. Adres"
    page: int = 0
    bbox: Optional[list[int]] = None   # obszar do wypełnienia
    label_bbox: Optional[list[int]] = None
    options: list[str] = field(default_factory=list)  # dla radio/checkbox-group
    max_length: Optional[int] = None   # liczba kratek w polu "grzebieniowym"
    address_part: Optional[str] = None  # city/street/building/apartment/postal_code/...
    hint: str = ""                     # np. "DD-MM-RRRR"
    prefilled: str = ""                # wartość już wpisana w formularzu
    question: str = ""                 # dla checkboxa: pytanie, do którego należy opcja
    detector: str = ""                 # box | comb | checkbox | underline | dots | table_cell | vlm
    confidence: float = 0.5
    role: str = ""                     # subject | applicant_name | applicant_address | contact | ... (semantic.py)
    required: Optional[bool] = None    # obligatoryjność (None = ustal regułami)
    depends_on: str = ""               # label pola, od którego zależy widoczność (warunek SHOW)
    depends_value: str = ""            # wartość/opcja w polu nadrzędnym

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, "", [])}
