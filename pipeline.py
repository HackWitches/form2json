"""Plik wejściowy -> ogólny JSON z parami "opis pola" -> "rodzaj pola"."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .detect import NUMBERING_RE, FormDetector, Phrase
from .loaders import load_document
from .model import Document, FieldCandidate
from .typing_rules import SIMPLE_TYPE, norm

MULTI_RE = re.compile(r"mozna zaznaczyc (kilka|wiele)|wielokrotn|zaznacz wszystkie|kilka odpowiedzi")


def strip_numbering(s: str) -> str:
    s = re.sub(r"^\s*((\d{1,2}(\.\d{1,2})*\.?)|([IVX]{1,4}\.)|([A-H]\.))\s+", "", s)
    s = re.sub(r"\s*\(.*$", "", s)                 # "(postaw x w wybranym polu)"
    return s.strip(" :")


def guess_title(doc: Document, dets: list[FormDetector]) -> str:
    if dets and dets[0].words:
        d = dets[0]
        top = [Phrase(ln) for ln in d.lines if ln[0].box.cy < 0.45 * d.page.width * 1.414]
        top = [p for p in top if len(p.text) > 8 and not NUMBERING_RE.match(p.text)]
        if top:
            best = max(top, key=lambda p: (p.size, p.bold))
            if best.size > 1.2 * d.line_h:
                return best.text
    return Path(doc.path).stem


def group_checkboxes(fields: list[FieldCandidate], line_h_by_page: dict[int, float]) -> list[FieldCandidate]:
    out: list[FieldCandidate] = []
    i = 0
    taken: set[int] = set()
    while i < len(fields):
        f = fields[i]
        if f.type != "checkbox":
            out.append(f)
            i += 1
            continue
        if id(f) in taken:
            i += 1
            continue
        grp = [f]
        j = i + 1
        lh = line_h_by_page.get(f.page, 20)
        if f.question:
            # opcje tego samego pytania (mogą być przeplecione polami "inline", np. "□ tatuaż ……")
            for g in fields[i + 1:]:
                if g.page != f.page or g.bbox[1] - grp[-1].bbox[3] > 3.5 * lh:
                    break
                if g.type == "checkbox" and g.question == f.question:
                    grp.append(g)
            taken.update(id(g) for g in grp)
            # pole-widmo z opisem pytania (komórka "Płeć" uznana za pole) - usuń
            out[:] = [o for o in out if not (o.label == f.question and o.page == f.page
                                             and abs(o.bbox[1] - f.bbox[1]) < 2 * lh)]
            j = i + 1
        else:
            while j < len(fields) and fields[j].type == "checkbox" and not fields[j].question \
                    and fields[j].section == f.section and fields[j].page == f.page \
                    and fields[j].bbox[1] - grp[-1].bbox[3] < 3.5 * lh:
                grp.append(fields[j])
                j += 1
        sec = norm(f.section)
        if "zalacznik" in sec:
            for g in grp:
                g.type = "file"
                out.append(g)
        elif len(grp) == 1:
            out.append(f)
        else:
            leaf = f.section.split(" > ")[-1] if f.section else ""
            q = f.question or strip_numbering(leaf) or "Wybierz opcję"
            sec = sec + " " + norm(q)
            bbox = grp[0].bbox[:]
            for g in grp[1:]:
                bbox = [min(bbox[0], g.bbox[0]), min(bbox[1], g.bbox[1]),
                        max(bbox[2], g.bbox[2]), max(bbox[3], g.bbox[3])]
            out.append(FieldCandidate(
                label=q, type="checkbox" if MULTI_RE.search(sec) else "radio",
                section=f.section, page=f.page, bbox=bbox,
                options=[g.label for g in grp], detector="checkbox-group",
                confidence=min(g.confidence for g in grp)))
        i = j
    return out


def merge_continuations(fields: list[FieldCandidate], line_h_by_page: dict[int, float]) -> list[FieldCandidate]:
    """Kolejne linie wykropkowania/linie do pisania z tym samym opisem -> jedno pole wieloliniowe."""
    out: list[FieldCandidate] = []
    for f in fields:
        p = out[-1] if out else None
        lh = line_h_by_page.get(f.page, 20)
        if p and p.page == f.page and p.label == f.label and p.detector in ("dots", "underline")                 and f.detector in ("dots", "underline") and 0 <= f.bbox[1] - p.bbox[3] < 2.5 * lh                 and f.bbox[0] <= p.bbox[0] + 5:
            p.bbox = [min(p.bbox[0], f.bbox[0]), p.bbox[1], max(p.bbox[2], f.bbox[2]), f.bbox[3]]
            if p.type == "string":
                p.type = "textarea"
            continue
        out.append(f)
    return out


def extract(path: str | Path, ocr=None, vlm=None, dpi: int = 150) -> dict:
    doc = load_document(path, dpi=dpi, ocr=ocr)
    fields: list[FieldCandidate] = []
    dets: list[FormDetector] = []
    sections: list[str] = []
    carry: str = ""
    for page in doc.pages:
        det = FormDetector(page)
        dets.append(det)
        fs, heads = det.detect()
        # sekcja kontynuowana z poprzedniej strony
        for f in fs:
            if carry and (not f.section or not heads or f.bbox[1] < heads[0].box.y0):
                f.section = carry if not f.section else f.section
        if fs:
            carry = fs[-1].section or carry
        fields.extend(fs)
        sections.extend(h.text for h in heads)

    lh = {d.page.index: d.line_h for d in dets}
    fields = merge_continuations(fields, lh)
    fields = group_checkboxes(fields, lh)
    title = guess_title(doc, dets)

    if vlm is not None:
        title, fields = vlm.refine([p.image for p in doc.pages], fields, doc.native_text, title)

    return {
        "source": str(path),
        "title": title,
        "pages": len(doc.pages),
        "text_source": sorted({p.text_source for p in doc.pages}),
        "sections": sections,
        "fields": [f.to_dict() for f in fields],
        # Widok uproszczony: "Opis pola" -> "Rodzaj pola"
        "simple": [{"Opis pola": f.label, "Rodzaj pola": SIMPLE_TYPE.get(f.type, f.type),
                    "Rodzaj szczegółowy": f.type,
                    **({"Opcje": f.options} if f.options else {}),
                    **({"Sekcja": f.section} if f.section else {})} for f in fields],
    }
