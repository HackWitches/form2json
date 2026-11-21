"""Wczytywanie dowolnego formatu -> Document (obrazy stron + słowa z pozycjami).

Strategia "jeden wspólny tor":
  * PDF                -> render (PyMuPDF) + warstwa tekstowa (dokładne pozycje, pogrubienia)
  * DOC/DOCX/ODT/ODS/XLS/XLSX -> LibreOffice headless -> PDF -> jak wyżej
                          (+ tekst natywny jako dodatkowy kontekst dla LLM)
  * JPG/PNG/TIFF       -> obraz + OCR (PaddleOCR / Tesseract / docTR)
  * PDF bez tekstu (skan) -> render + OCR

Dzięki temu detekcja geometrii (ramki, kratki, checkboxy, linie) działa zawsze
na rastrze, niezależnie od formatu wejściowego.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .model import Box, Document, Page, Word

# znaki używane jako checkboxy w dokumentach Word (także z fontów Wingdings w obszarze PUA)
CHECKBOX_CHARS = "□☐❑❒▢■☑☒"
PUA_CHECKBOX = {0xF06F: "□", 0xF0A8: "□", 0xF071: "❑", 0xF072: "❒", 0xF0FE: "☑", 0xF0FD: "☒"}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".jpe", ".tif", ".tiff", ".bmp", ".webp"}
OFFICE_EXT = {".doc", ".docx", ".odt", ".rtf", ".ods", ".xls", ".xlsx", ".xlsm"}
DEFAULT_DPI = 150


# ---------------------------------------------------------------- utils

def imread_unicode(path: str | Path) -> np.ndarray:
    """cv2.imread nie obsługuje polskich znaków w ścieżce na Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nie mogę odczytać obrazu: {path}")
    return img


def find_soffice() -> str | None:
    for name in ("soffice", "libreoffice"):
        p = shutil.which(name)
        if p:
            return p
    for p in (r"C:\Program Files\LibreOffice\program\soffice.exe",
              r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
              "/usr/bin/soffice", "/usr/lib/libreoffice/program/soffice",
              "/Applications/LibreOffice.app/Contents/MacOS/soffice"):
        if os.path.exists(p):
            return p
    return None


def office_to_pdf(path: Path, outdir: Path) -> Path:
    soffice = find_soffice()
    if not soffice:
        raise RuntimeError(
            "Brak LibreOffice (soffice). Zainstaluj go, aby przetwarzać "
            "DOC/DOCX/ODS/XLSX (np. `apt install libreoffice-core libreoffice-writer libreoffice-calc`).")
    # osobny profil -> można uruchamiać równolegle
    profile = Path(tempfile.mkdtemp(prefix="lo_profile_"))
    cmd = [soffice, f"-env:UserInstallation={profile.as_uri()}", "--headless",
           "--convert-to", "pdf", "--outdir", str(outdir), str(path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    out = outdir / (path.stem + ".pdf")
    if not out.exists():
        raise RuntimeError(f"LibreOffice nie wygenerował PDF dla {path}")
    return out


# ---------------------------------------------------------------- native text

def native_text(path: Path) -> str:
    """Tekst natywny (z zachowaniem struktury tabel) - kontekst pomocniczy dla VLM."""
    ext = path.suffix.lower()
    try:
        if ext == ".docx":
            import docx  # python-docx
            d = docx.Document(str(path))
            out = [p.text for p in d.paragraphs if p.text.strip()]
            for ti, t in enumerate(d.tables):
                out.append(f"[TABELA {ti + 1}]")
                for r in t.rows:
                    cells = []
                    for c in r.cells:
                        txt = c.text.strip().replace("\n", " ")
                        if not cells or cells[-1] != txt:  # scalone komórki się powtarzają
                            cells.append(txt)
                    out.append(" | ".join(cells))
            return "\n".join(out)
        if ext in (".xlsx", ".xlsm"):
            import openpyxl
            wb = openpyxl.load_workbook(str(path), data_only=True)
            out = []
            for ws in wb.worksheets:
                out.append(f"[ARKUSZ {ws.title}]")
                for row in ws.iter_rows():
                    vals = [f"{c.coordinate}={c.value}" for c in row if c.value not in (None, "")]
                    if vals:
                        out.append(" | ".join(vals))
            return "\n".join(out)
        if ext == ".ods":
            from odf.opendocument import load
            from odf import text as odftext, table as odftable
            from odf.element import Element
            doc = load(str(path))
            out = []

            def txt(el: Element) -> str:
                return "".join(str(n) for n in el.childNodes if n.nodeType == n.TEXT_NODE) + \
                    "".join(txt(n) for n in el.childNodes if n.nodeType == n.ELEMENT_NODE)

            for row in doc.spreadsheet.getElementsByType(odftable.TableRow):
                vals = [txt(c).strip() for c in row.getElementsByType(odftable.TableCell)]
                vals = [v for v in vals if v]
                if vals:
                    out.append(" | ".join(vals))
            return "\n".join(out)
    except Exception as e:  # tekst natywny jest tylko pomocniczy
        return f"[native text error: {e}]"
    return ""


# ---------------------------------------------------------------- PDF

def _pdf_words(page, scale: float) -> list[Word]:
    """Słowa z warstwy tekstowej PDF, z informacją o pogrubieniu i rozmiarze."""
    words: list[Word] = []
    raw = page.get_text("rawdict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bold = bool(span.get("flags", 0) & 16) or "bold" in span.get("font", "").lower()
                size = span.get("size", 0) * scale
                cur, bx = "", None
                leader = "._…·"
                for ch in span.get("chars", []) + [None]:
                    c = ch["c"] if ch else " "
                    if 0xE000 <= ord(c) <= 0xF8FF:  # symbole z fontów Wingdings/Symbol (strzałki, ptaszki)
                        c = PUA_CHECKBOX.get(ord(c), " ")
                    if c in CHECKBOX_CHARS:         # checkbox-znak jako osobne "słowo"
                        if cur.strip():
                            words.append(Word(cur, bx, bold=bold, size=size))
                        x0, y0, x1, y1 = (v * scale for v in ch["bbox"])
                        words.append(Word("□", Box(x0, y0, x1, y1), size=size))
                        cur, bx = "", None
                        continue
                    # granica słowo|wykropkowanie ("dnia……") liczona dokładnie po znakach
                    is_lead, prev_lead = c in leader, bool(cur) and cur[-1] in leader
                    if cur and not c.isspace() and is_lead != prev_lead and                             (len(cur) >= 4 if prev_lead else c in "…_"):
                        words.append(Word(cur, bx, bold=bold, size=size))
                        cur, bx = "", None
                    if c.isspace():
                        if cur.strip():
                            words.append(Word(cur, bx, bold=bold, size=size))
                        cur, bx = "", None
                        continue
                    x0, y0, x1, y1 = (v * scale for v in ch["bbox"])
                    b = Box(x0, y0, x1, y1)
                    bx = b if bx is None else bx.union(b)
                    cur += c
    return words


def load_pdf(path: Path, dpi: int, ocr=None) -> list[Page]:
    import fitz  # PyMuPDF
    pages = []
    scale = dpi / 72.0
    with fitz.open(str(path)) as doc:
        for i, p in enumerate(doc):
            pix = p.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
            words = _pdf_words(p, scale)
            src = "pdf"
            if len(words) < 5 and ocr is not None:      # skan w PDF
                words, src = ocr(img), f"ocr:{ocr.name}"
            pages.append(Page(i, img, dpi, words, src))
    return pages


# ---------------------------------------------------------------- obrazy

def deskew(img: np.ndarray) -> np.ndarray:
    """Prostowanie skanu/zdjęcia na podstawie długich linii poziomych."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=200,
                            minLineLength=img.shape[1] // 4, maxLineGap=10)
    if lines is None:
        return img
    angles = []
    for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        if abs(a) < 10:
            angles.append(a)
    if not angles:
        return img
    a = float(np.median(angles))
    if abs(a) < 0.1:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def split_pages(img: np.ndarray) -> list[np.ndarray]:
    """Obraz z kilkoma stronami sklejonymi w pionie -> strony A4 (cięcie w najjaśniejszym wierszu)."""
    h, w = img.shape[:2]
    step = int(round(w * 1.414))
    if h <= step * 1.3:
        return [img]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (gray < 200).sum(axis=1)
    cuts, y = [0], 0
    while h - y > step * 1.3:
        lo, hi = y + int(step * 0.9), min(h - 1, y + int(step * 1.1))
        c = lo + int(np.argmin(ink[lo:hi]))
        cuts.append(c)
        y = c
    cuts.append(h)
    return [img[a:b] for a, b in zip(cuts, cuts[1:]) if b - a > 50]


def load_image(path: Path, ocr) -> list[Page]:
    img = deskew(imread_unicode(path))
    # Szacujemy DPI zakładając szerokość A4 (8.27 cala) - tylko do progów geometrii.
    dpi = max(72, int(round(img.shape[1] / 8.27)))
    if ocr is None:
        raise RuntimeError("Obraz wymaga OCR - podaj --ocr paddle|tesseract|doctr")
    return [Page(i, p, dpi, ocr(p), f"ocr:{ocr.name}") for i, p in enumerate(split_pages(img))]


# ---------------------------------------------------------------- wejście

def load_document(path: str | Path, dpi: int = DEFAULT_DPI, ocr=None) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = load_pdf(path, dpi, ocr)
        return Document(str(path), pages)
    if ext in IMAGE_EXT:
        return Document(str(path), load_image(path, ocr))
    if ext in OFFICE_EXT:
        with tempfile.TemporaryDirectory(prefix="form2json_") as tmp:
            # LibreOffice źle znosi nietypowe nazwy -> kopia pod prostą nazwą
            src = Path(tmp) / ("input" + ext)
            shutil.copyfile(path, src)
            pdf = office_to_pdf(src, Path(tmp))
            pages = load_pdf(pdf, dpi, ocr)
        return Document(str(path), pages, native_text=native_text(path))
    raise ValueError(f"Nieobsługiwany format: {ext}")
