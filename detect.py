"""Detekcja pól formularza na podstawie geometrii (bez uczenia modeli).

Pracuje na rastrze strony + słowach z pozycjami, więc działa identycznie dla
PDF, DOCX/XLSX/ODS (po konwersji do PDF) i skanów/zdjęć (po OCR).

Wykrywane miejsca do wpisania danych:
  * box        - pusta prostokątna ramka (pole tekstowe / textarea)
  * comb       - rząd kratek na pojedyncze znaki (NIP, PESEL, kod, data)
  * checkbox   - mały kwadrat z opisem obok
  * table_cell - pusta komórka tabeli (opis z lewej lub nagłówek kolumny)
  * captioned  - komórka z małym opisem u góry i pustym miejscem pod nim
  * underline  - pozioma linia do pisania "na linii"
  * dots       - wykropkowanie "........" (z warstwy tekstowej lub z rastra)

Dla każdego miejsca szukany jest opis (label): w środku > z lewej > nad > pod.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .model import Box, FieldCandidate, Page, Word
from .typing_rules import infer_type

LEADER_RE = re.compile(r"[.…·_]{4,}")
NUMBERING_RE = re.compile(
    r"^\s*((\d{1,2}(\.\d{1,2})*\.?)|([IVX]{1,4}\.)"        # 2.  2.1.  IV.
    r"|([A-H](\.\d{1,2})+\.?)|([A-H]\.)"                   # B.1  A.
    r"|([A-H](?=\s+[A-ZĄĆĘŁŃÓŚŹŻ]{3,})))\s+\S")            # "A DANE IDENTYFIKACYJNE"
ROWNUM_RE = re.compile(r"^\s*\d{1,3}[.)]?\s*$")
CAPTION_RE = re.compile(r"^\s*[\(/].{1,80}[\)/]\s*$")   # "(imię i nazwisko)", "/ podpis /"
PUNCT_ONLY = re.compile(r"^[\s`'\".,:;_\-–—…·*()\[\]]*$")


# =========================================================== struktury pomocnicze
def clean_label(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(\s\d{1,2}|\*+)+$", "", s).strip()      # odnośniki przypisów: "ulica 1", "PESEL *"
    if s.count(")") > s.count("("):                       # "nr telefonu/adres e-mail)"
        s = s.replace(")", "", s.count(")") - s.count("("))
    if s.count("(") > s.count(")"):
        s += ")"
    return s.strip(" :/,;-")


@dataclass
class Phrase:
    words: list[Word]
    cell: int = -1                      # indeks komórki, w której leży fraza (-1 = poza)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    @property
    def box(self) -> Box:
        b = self.words[0].box
        for w in self.words[1:]:
            b = b.union(w.box)
        return b

    @property
    def bold(self) -> bool:
        return 2 * sum(w.bold for w in self.words) > len(self.words)

    @property
    def size(self) -> float:
        return float(np.median([w.size or w.box.h for w in self.words]))


@dataclass
class Region:
    """Miejsce do wpisania danych."""
    box: Box
    kind: str
    n_cells: int = 0
    parts: list[int] = field(default_factory=list)   # liczby kratek w kolejnych grupach (np. 2-2-4)
    cell: int = -1


# =========================================================== geometria rastra

class Geometry:
    def __init__(self, page: Page):
        self.page = page
        u = page.dpi / 150.0
        self.u = u
        g = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
        # binarizacja odporna na skany (adaptive) + Otsu dla czystych renderów
        if page.text_source == "pdf":
            # render wektorowy: cienkie linie bywają jasnoszare -> stały, wysoki próg
            _, self.bin = cv2.threshold(g, 215, 255, cv2.THRESH_BINARY_INV)
        else:
            _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            adap = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 15)
            self.bin = cv2.bitwise_or(otsu, adap) if page.text_source.startswith("ocr") else otsu

        k = max(8, int(10 * u))
        self.hmask = cv2.morphologyEx(self.bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        self.vmask = cv2.morphologyEx(self.bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
        self.cells = self._cells()
        self.hlines = self._hlines()
        self.dotted = self._dotted_lines()

    # ------------------------------------------------------------ komórki/ramki
    def _cells(self) -> list[Box]:
        grid = cv2.bitwise_or(self.hmask, self.vmask)
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
        inv = cv2.bitwise_not(grid)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        H, W = inv.shape
        out = []
        mn = max(7, int(9 * self.u))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < mn or h < mn:
                continue
            if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                continue                                    # tło strony
            if w > 0.97 * W and h > 0.5 * H:
                continue
            if area / float(w * h) < 0.82:                   # nie-prostokątne (pierścienie, obszary między tabelami)
                continue
            pad = 2
            out.append(Box(x - pad, y - pad, x + w + pad, y + h + pad))
        return out

    # ------------------------------------------------------------ linie poziome
    def _hlines(self) -> list[tuple[Box, float]]:
        """Samodzielne linie poziome (nie będące krawędzią ramki): (box, grubość)."""
        n, lab, stats, _ = cv2.connectedComponentsWithStats(self.hmask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < 50 * self.u or h > 12 * self.u:
                continue
            b = Box(x, y, x + w, y + h)
            # krawędź istniejącej komórki?
            edge = any(abs(c.y0 - b.y0) < 6 * self.u or abs(c.y1 - b.y1) < 6 * self.u
                       for c in self.cells if c.h_overlap(b) > 0.5 and c.x0 - 4 <= b.x0 + 4 and b.x1 - 4 <= c.x1 + 4
                       and c.w > 0.5 * b.w)
            if edge:
                continue
            # pionowe kreski stykające się z linią -> to część tabeli, nie "linia do pisania"
            vv = self.vmask[max(0, y - int(15 * self.u)):y + h + 1, x:x + w]
            if vv.any() and w < 0.95 * self.page.width:
                cols = np.where(vv.any(axis=0))[0]
                if len(cols) and (cols.min() < 6 or cols.max() > w - 6):
                    continue
            out.append((b, area / float(w)))
        return out

    # ------------------------------------------------------------ wykropkowania z rastra
    def _dotted_lines(self) -> list[Box]:
        n, lab, stats, cent = cv2.connectedComponentsWithStats(self.bin, connectivity=8)
        dmax = max(2, int(4 * self.u))
        dots = [(cent[i][0], cent[i][1]) for i in range(1, n)
                if stats[i][2] <= dmax and stats[i][3] <= dmax and stats[i][4] >= 1]
        if len(dots) < 8:
            return []
        dots.sort(key=lambda p: (round(p[1] / (3 * self.u)), p[0]))
        rows: dict[int, list[tuple[float, float]]] = {}
        for x, y in dots:
            rows.setdefault(int(round(y / (3 * self.u))), []).append((x, y))
        out = []
        gap = 14 * self.u
        for pts in rows.values():
            pts.sort()
            run = [pts[0]]
            for p in pts[1:] + [(1e9, 0)]:
                if p[0] - run[-1][0] <= gap:
                    run.append(p)
                else:
                    if len(run) >= 8 and run[-1][0] - run[0][0] > 50 * self.u:
                        ys = [q[1] for q in run]
                        out.append(Box(run[0][0], min(ys) - 2, run[-1][0], max(ys) + 2))
                    run = [p]
        return out

    def ink_ratio(self, b: Box) -> float:
        x0, y0, x1, y1 = (int(v) for v in (b.x0 + 4, b.y0 + 4, b.x1 - 4, b.y1 - 4))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(self.bin[y0:y1, x0:x1].mean() / 255.0)


# =========================================================== tekst -> frazy

def build_lines(words: list[Word]) -> list[list[Word]]:
    ws = sorted(words, key=lambda w: (w.box.cy, w.box.x0))
    lines: list[list[Word]] = []
    for w in ws:
        for ln in reversed(lines[-4:]):
            ref = ln[-1].box
            if abs(ref.cy - w.box.cy) < 0.45 * max(4, min(ref.h, w.box.h)):
                ln.append(w)
                break
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w.box.x0)
    lines.sort(key=lambda l: l[0].box.cy)
    return lines


def cell_of(b: Box, cells: list[Box]) -> int:
    best, area = -1, 1e18
    for i, c in enumerate(cells):
        if c.x0 - 1 <= b.cx <= c.x1 + 1 and c.y0 - 1 <= b.cy <= c.y1 + 1:
            a = c.w * c.h
            if a < area:
                best, area = i, a
    return best


def build_phrases(lines: list[list[Word]], cells: list[Box], u: float) -> list[Phrase]:
    phrases = []
    for ln in lines:
        cur = Phrase([ln[0]], cell_of(ln[0].box, cells))
        for w in ln[1:]:
            c = cell_of(w.box, cells)
            gap = w.box.x0 - cur.words[-1].box.x1
            lim = max(12 * u, 1.1 * max(w.box.h, cur.words[-1].box.h))
            if gap <= lim and c == cur.cell and not LEADER_RE.fullmatch(w.text) \
                    and not LEADER_RE.fullmatch(cur.words[-1].text) \
                    and w.text != "□" and cur.words[-1].text != "□":
                cur.words.append(w)
            else:
                phrases.append(cur)
                cur = Phrase([w], c)
        phrases.append(cur)
    return phrases


# =========================================================== detektor

class FormDetector:
    def __init__(self, page: Page):
        self.page = page
        self.geo = Geometry(page)
        self.u = self.geo.u
        words = [w for w in page.words if w.text.strip()]
        self.words = self._split_leaders(words)
        self.lines = build_lines(self.words)
        self.phrases = build_phrases(self.lines, self.geo.cells, self.u)
        self.line_h = float(np.median([w.box.h for w in self.words])) if self.words else 14 * self.u

    # ------------------------------------------------------------ wykropkowania w tekście
    @staticmethod
    def _split_leaders(words: list[Word]) -> list[Word]:
        """'Nazwisko:.........' -> ['Nazwisko:', '.........'] (podział proporcjonalny)."""
        out = []
        for w in words:
            m = LEADER_RE.search(w.text)
            if not m or LEADER_RE.fullmatch(w.text):
                out.append(w)
                continue
            n = len(w.text)
            pieces = [(0, m.start()), (m.start(), m.end()), (m.end(), n)]
            for a, b in pieces:
                t = w.text[a:b]
                if t.strip():
                    x0 = w.box.x0 + w.box.w * a / n
                    x1 = w.box.x0 + w.box.w * b / n
                    out.append(Word(t, Box(x0, w.box.y0, x1, w.box.y1), w.bold, w.size, w.conf))
        return out

    # ------------------------------------------------------------ pomocnicze
    def words_in(self, b: Box) -> list[Word]:
        return [w for w in self.words if b.x0 - 1 <= w.box.cx <= b.x1 + 1 and b.y0 - 1 <= w.box.cy <= b.y1 + 1]

    def is_empty(self, b: Box) -> bool:
        ws = self.words_in(b)
        return all(PUNCT_ONLY.match(w.text) for w in ws)

    def _is_heading(self, p: Phrase) -> bool:
        t = p.text
        if len(t) < 3 or len(t) > 160:
            return False
        if not NUMBERING_RE.match(t):
            return False
        head_bold = all(w.bold for w in p.words[:2])
        letters = [c for c in t if c.isalpha()]
        upper = len(letters) >= 6 and sum(c.isupper() for c in letters) > 0.85 * len(letters)
        return head_bold or upper or p.size > 1.15 * self.line_h

    # ------------------------------------------------------------ 1. regiony
    def regions(self) -> list[Region]:
        g, u = self.geo, self.u
        regs: list[Region] = []
        small_lo, small_hi = 9 * u, 62 * u

        cells = list(enumerate(g.cells))
        smalls = [(i, c) for i, c in cells
                  if small_lo <= c.w <= small_hi and small_lo <= c.h <= small_hi
                  and 0.55 <= c.w / c.h <= 1.8 and self.is_empty(c)]
        used = set()
        # --- grupy kratek (comb): sąsiadujące poziomo małe komórki o tej samej wysokości
        smalls.sort(key=lambda t: (round(t[1].cy / (6 * u)), t[1].x0))
        groups: list[list[tuple[int, Box]]] = []
        for i, c in smalls:
            if groups:
                last = groups[-1][-1][1]
                if abs(last.cy - c.cy) < 4 * u and abs(last.h - c.h) < 5 * u and -6 * u <= c.x0 - last.x1 < 10 * u:
                    groups[-1].append((i, c))
                    continue
            groups.append([(i, c)])

        combs, singles = [], []
        for grp in groups:
            (combs if len(grp) >= 2 else singles).append(grp)

        # łączenie grup kratek przedzielonych "-" "/" (data DD-MM-RRRR, kod 00-000)
        combs.sort(key=lambda gr: (round(gr[0][1].cy / (6 * u)), gr[0][1].x0))
        merged: list[list[list[tuple[int, Box]]]] = []
        for grp in combs:
            if merged:
                prev = merged[-1][-1]
                pb, cb = prev[-1][1], grp[0][1]
                gapbox = Box(pb.x1, min(pb.y0, cb.y0), cb.x0, max(pb.y1, cb.y1))
                between = self.words_in(gapbox)
                if abs(pb.cy - cb.cy) < 5 * u and 0 < cb.x0 - pb.x1 < 70 * u and \
                        all(w.text.strip() in "-–/." for w in between):
                    merged[-1].append(grp)
                    continue
            merged.append([grp])
        comb_regs: list[Region] = []
        for parts in merged:
            b = parts[0][0][1]
            for grp in parts:
                for i, c in grp:
                    b = b.union(c)
                    used.add(i)
            r = Region(b, "comb", n_cells=sum(len(p) for p in parts), parts=[len(p) for p in parts])
            prev = comb_regs[-1] if comb_regs else None
            if prev and len(r.parts) == 1 and abs(prev.box.x0 - b.x0) < 6 * u and abs(prev.box.x1 - b.x1) < 6 * u                     and 0 <= b.y0 - prev.box.y1 < 0.9 * self.line_h                     and not [p for p in self.phrases if prev.box.y1 - 2 <= p.box.cy <= b.y0 + 2
                             and p.box.h_overlap(b) > 0.2]:
                prev.box = prev.box.union(b)
                prev.n_cells += r.n_cells
                prev.parts = [prev.n_cells]
                prev.kind = "comb"
                prev.rows = getattr(prev, "rows", 1) + 1
                continue
            comb_regs.append(r)
        regs.extend(comb_regs)

        # --- checkboxy: pojedyncze, w miarę kwadratowe kratki
        for grp in singles:
            i, c = grp[0]
            if 0.7 <= c.w / c.h <= 1.4 and c.w <= 50 * u:
                regs.append(Region(c, "checkbox", cell=i))
                used.add(i)

        # --- checkboxy będące znakami tekstu ("□", "❑")
        for w in self.words:
            if w.text == "□":
                regs.append(Region(w.box, "checkbox"))

        # --- pozostałe komórki
        for i, c in cells:
            if i in used or c.h < 12 * u or c.w < 40 * u or c.h > 8 * c.w:
                continue
            ws = [w for w in self.words_in(c) if not PUNCT_ONLY.match(w.text)]
            if any(w.text == "□" for w in ws):
                continue                                  # komórka z checkboxami-znakami
            if not ws:
                regs.append(Region(c, "box", cell=i))
                continue
            # opis wewnątrz komórki + puste miejsce pod nim (formularze tabelaryczne)
            tb = ws[0].box
            for w in ws[1:]:
                tb = tb.union(w.box)
            free_below = c.y1 - tb.y1
            free_right = c.x1 - tb.x1
            if free_below > max(1.6 * self.line_h, 0.45 * c.h) and tb.y0 - c.y0 < 0.4 * c.h:
                regs.append(Region(Box(c.x0, tb.y1 + 2, c.x1, c.y1), "captioned", cell=i))
            elif free_right > max(70 * u, 0.3 * c.w) and c.h < 3 * self.line_h \
                    and len(ws) <= 6 and not ws[-1].text.endswith(".") \
                    and not LEADER_RE.search(" ".join(w.text for w in ws)):
                regs.append(Region(Box(tb.x1 + 4, c.y0, c.x1, c.y1), "captioned", cell=i))

        # --- linie do pisania i wykropkowania z rastra
        for b, thick in g.hlines:
            if thick > 3.2 * u and b.w > 0.5 * self.page.width:
                continue                                  # separator sekcji
            above = [w for w in self.words if w.box.h_overlap(b) > 0.3 and 0 <= b.y0 - w.box.y1 < 0.35 * self.line_h]
            if above and sum(w.box.w for w in above) > 0.4 * b.w:
                continue                                  # podkreślony tekst / link
            regs.append(Region(Box(b.x0, b.y0 - 1.3 * self.line_h, b.x1, b.y1), "underline"))
        for b in g.dotted:
            regs.append(Region(Box(b.x0, b.y0 - 1.2 * self.line_h, b.x1, b.y1 + 2), "dots_raster"))

        # --- wykropkowania z warstwy tekstowej / OCR
        for w in self.words:
            if LEADER_RE.fullmatch(w.text):
                regs.append(Region(Box(w.box.x0, w.box.y0 - 0.3 * self.line_h, w.box.x1, w.box.y1), "dots"))

        return self._dedupe(regs)

    @staticmethod
    def _dedupe(regs: list[Region]) -> list[Region]:
        prio = {"comb": 0, "checkbox": 0, "box": 1, "captioned": 1, "dots": 2, "dots_raster": 3, "underline": 3}
        regs.sort(key=lambda r: prio.get(r.kind, 9))
        out: list[Region] = []
        for r in regs:
            if any(o.box.v_overlap(r.box) > 0.5 and o.box.h_overlap(r.box) > 0.5 for o in out):
                continue
            out.append(r)
        # sklej sąsiadujące odcinki kropek w tej samej linii
        out.sort(key=lambda r: (round(r.box.cy / 8), r.box.x0))
        res: list[Region] = []
        for r in out:
            if r.kind == "dots_raster":
                r.kind = "dots"
            if res and r.kind == "dots" == res[-1].kind and abs(res[-1].box.cy - r.box.cy) < 6 \
                    and 0 <= r.box.x0 - res[-1].box.x1 < 15:
                res[-1].box = res[-1].box.union(r.box)
                continue
            res.append(r)
        return res

    # ------------------------------------------------------------ 2. opisy (labels)
    def _ok(self, p: Phrase) -> bool:
        return id(p) not in self._head_ids and not PUNCT_ONLY.match(p.text) \
            and not LEADER_RE.fullmatch(p.text) and p.text != "□"

    def _label_left(self, r: Region) -> list[Phrase]:
        b = r.box
        cands = [p for p in self.phrases
                 if self._ok(p) and p.box.x1 <= b.x0 + max(6 * self.u, 0.5 * self.line_h) and b.x0 - p.box.x1 < 0.55 * self.page.width
                 and (p.box.v_overlap(b) > 0.45 or b.y0 - 2 <= p.box.cy <= b.y1 + 2)
                 and not LEADER_RE.fullmatch(p.text) and not PUNCT_ONLY.match(p.text)]
        if not cands:
            return []
        best = max(cands, key=lambda p: p.box.x1)
        # inny region pomiędzy opisem a polem -> to nie nasz opis
        for o in self._regions:
            if o is not r and o.box.x0 >= best.box.x1 - 2 and o.box.x1 <= b.x0 + 2 and o.box.v_overlap(b) > 0.5:
                return []
        # opis wieloliniowy wyrównany do prawej ("Nazwa organu,\n w którym zapłacono opłatę")
        group = [best]
        for p in self.phrases:
            if p is best or p.cell != best.cell:
                continue
            if abs(p.box.x1 - best.box.x1) < 25 * self.u and p.box.x1 <= b.x0 + 6:
                top, bot = min(q.box.y0 for q in group), max(q.box.y1 for q in group)
                if -0.5 * self.line_h <= top - p.box.y1 < 0.8 * self.line_h or -0.5 * self.line_h <= p.box.y0 - bot < 0.8 * self.line_h:
                    if b.y0 - 1.2 * self.line_h <= p.box.cy <= b.y1 + 1.2 * self.line_h:
                        group.append(p)
        return sorted(group, key=lambda p: p.box.y0)

    def _label_above(self, r: Region) -> list[Phrase]:
        b = r.box
        cands = [p for p in self.phrases
                 if self._ok(p) and not CAPTION_RE.match(p.text) and p.box.y1 <= b.y0 + 3 and p.box.h_overlap(b) > 0.3
                 and p.box.x0 >= b.x0 - 30 * self.u and not PUNCT_ONLY.match(p.text)
                 and not LEADER_RE.fullmatch(p.text)]
        if not cands:
            return []
        best = max(cands, key=lambda p: p.box.y1)
        gap = b.y0 - best.box.y1
        if gap < 1.6 * self.line_h:
            return [best]
        # nagłówek kolumny tabeli: pomiędzy tylko puste komórki tej samej kolumny
        if best.cell >= 0:
            hc = self.geo.cells[best.cell]
            if abs(hc.x0 - b.x0) < 12 * self.u and abs(hc.x1 - b.x1) < 12 * self.u:
                return [best]
        return []

    def _label_below(self, r: Region) -> list[Phrase]:
        b = r.box
        cands = [p for p in self.phrases
                 if self._ok(p) and 0 <= p.box.y0 - b.y1 < 1.3 * self.line_h and p.box.h_overlap(b) > 0.5
                 and (CAPTION_RE.match(p.text) or p.text.startswith("("))]
        return [min(cands, key=lambda p: p.box.y0)] if cands else []

    def _label_right(self, r: Region) -> list[Phrase]:
        b = r.box
        cands = [p for p in self.phrases
                 if self._ok(p) and p.box.x0 >= b.x1 - 3 and p.box.x0 - b.x1 < 60 * self.u and p.box.v_overlap(b) > 0.4]
        return [min(cands, key=lambda p: p.box.x0)] if cands else []

    def _checkbox_question(self, r: Region, right: list[Phrase]) -> str:
        """Pytanie, do którego należy checkbox: opis z lewej w tym samym wierszu
        (np. "Płeć  □ pies  □ suka"), dziedziczone przez kolejne opcje w wierszu
        oraz przez opcje ułożone pionowo pod spodem."""
        b, u = r.box, self.u
        boxes = [o for o in self._regions if o.kind == "checkbox" and o is not r]
        left = [p for p in self.phrases if self._ok(p) and p.box.x1 <= b.x0 + 3 and p.box.v_overlap(b) > 0.4]
        if left:
            lp = max(left, key=lambda p: p.box.x1)
            owner = [o for o in boxes if o.box.v_overlap(b) > 0.5 and -3 <= lp.box.x0 - o.box.x1 < 60 * u]
            if owner:
                return self._q_of.get(id(owner[0]), "")
            return clean_label(lp.text)
        # opcja w pionowej liście: dziedzicz po checkboxie bezpośrednio wyżej
        above = [o for o in boxes if abs(o.box.x0 - b.x0) < 8 * u and 0 < b.y0 - o.box.y1 < 2.5 * self.line_h]
        if above:
            return self._q_of.get(id(max(above, key=lambda o: o.box.y0)), "")
        return ""

    def _label_inside(self, r: Region) -> list[Phrase]:
        c = self.geo.cells[r.cell]
        return [p for p in self.phrases if c.contains(p.box, 3) and not r.box.contains(p.box, 0)]

    # ------------------------------------------------------------ 3. sekcje
    def headings(self) -> list[Phrase]:
        hs = []
        for ln in self.lines:
            ph = Phrase(ln)
            if self._is_heading(ph):
                hs.append(ph)
        return hs

    # ------------------------------------------------------------ główna metoda
    def detect(self) -> tuple[list[FieldCandidate], list[Phrase]]:
        self._regions = self.regions()
        heads = self.headings()
        self._head_ids = {id(p) for p in self.phrases
                          if any(h.box.v_overlap(p.box) > 0.6 and h.box.h_overlap(p.box) > 0.6 for h in heads)}
        fields: list[FieldCandidate] = []
        q_of: dict[int, str] = {}
        self._q_of = q_of
        for r in self._regions:
            lab: list[Phrase] = []
            hint = ""
            question = ""
            if r.kind == "checkbox":
                right = self._label_right(r)
                lab = right or self._label_left(r)
                question = self._checkbox_question(r, right) if right else ""
                q_of[id(r)] = question
            elif r.kind == "captioned":
                lab = self._label_inside(r)
            else:
                if r.kind in ("dots", "underline"):
                    below = self._label_below(r)
                    lab = below if below and CAPTION_RE.match(below[0].text) else []
                lab = lab or self._label_left(r)
                if lab and ROWNUM_RE.match(" ".join(p.text for p in lab)):
                    rown = " ".join(p.text for p in lab).strip(" .)")
                    head = self._label_above(r)
                    lab = head
                    hint = f"wiersz {rown}"
                if not lab:
                    lab = self._label_above(r) or self._label_below(r)
            label = " ".join(p.text for p in lab).strip(" :/()")
            label = clean_label(label)
            if not label and r.kind in ("checkbox", "underline", "dots"):
                continue
            # podpowiedź formatu pod polem, np. "DD-MM-RRRR"
            for p in self.phrases:
                if 0 <= p.box.y0 - r.box.y1 < 1.4 * self.line_h and p.box.h_overlap(r.box) > 0.3 \
                        and re.search(r"(DD|RRRR|YYYY|MM)", p.text):
                    hint = re.sub(r"^[^A-Za-z]+", "", p.text)
            # instrukcje i nagłówki to nie pola
            if r.kind in ("box", "captioned") and any(h.box.v_overlap(r.box) > 0.6 and h.box.x0 >= r.box.x0 - 2
                                                      for h in heads):
                continue
            fc = FieldCandidate(
                label=label, page=self.page.index, bbox=r.box.as_list(),
                label_bbox=lab[0].box.union(lab[-1].box).as_list() if lab else None,
                detector=r.kind, hint=hint,
                max_length=r.n_cells or None,
                confidence={"comb": .9, "checkbox": .85, "box": .8, "captioned": .75,
                            "table_cell": .7, "dots": .7, "underline": .6}.get(r.kind, .5),
            )
            fc.question = question
            fc._region = r  # type: ignore[attr-defined]
            fields.append(fc)

        self._assign_sections(fields, heads)
        for f in fields:
            if not f.label:
                leaf = f.section.split(" > ")[-1] if f.section else ""
                f.label = re.sub(r"^\s*[\dIVX.]+\s+", "", re.sub(r"\s*\(.*$", "", leaf)).strip() or "Pole"
                f.confidence = 0.4
        for f in fields:
            r = f._region  # type: ignore[attr-defined]
            f.type, f.address_part = infer_type(f.label, f.section, r.kind, r.n_cells, r.parts,
                                                r.box.h / max(1.0, self.line_h), f.hint)
            del f._region  # type: ignore[attr-defined]
        return fields, heads

    def _assign_sections(self, fields: list[FieldCandidate], heads: list[Phrase]) -> None:
        stack: list[tuple[int, str]] = []
        events = [(h.box.y0, 0, h) for h in heads] + [(f.bbox[1], 1, f) for f in fields]
        events.sort(key=lambda e: (e[0], e[1]))
        for y, kind, obj in events:
            if kind == 0:
                t = re.sub(r"\s+", " ", obj.text)
                num = t.split()[0].rstrip(".")
                depth = num.count(".") + 1 if re.match(r"^\d", num) else 1
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                stack.append((depth, t))
            else:
                obj.section = " > ".join(s for _, s in stack)


def detect_page(page: Page) -> tuple[list[FieldCandidate], list[str]]:
    det = FormDetector(page)
    fields, heads = det.detect()
    return fields, [h.text for h in heads]
