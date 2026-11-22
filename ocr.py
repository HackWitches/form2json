"""Wymienne silniki OCR (wszystkie lokalne / self-hosted).

* paddle    - PaddleOCR 3.x (PP-OCRv5), opcjonalnie GPU (CUDA, RTX 4090)
* tesseract - Tesseract 5 + `pol.traineddata`, CPU - prosty fallback

Każdy backend zwraca listę `Word` w pikselach obrazu wejściowego.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from .detect import LEADER_RE
from .model import Box, Word


def _split_line(text: str, box: Box, conf: float) -> list[Word]:
    """OCR zwraca linie - dzielimy je na słowa proporcjonalnie do liczby znaków."""
    tokens = text.split(" ")
    n = max(1, len(text))
    out, pos = [], 0
    for t in tokens:
        if t:
            x0 = box.x0 + box.w * pos / n
            x1 = box.x0 + box.w * (pos + len(t)) / n
            out.append(Word(t, Box(x0, box.y0, x1, box.y1), size=box.h * 0.8, conf=conf))
        pos += len(t) + 1
    return out


class PaddleOCRBackend:
    name = "paddle"

    def __init__(self, lang: str = "pl", device: str = "gpu"):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            lang=lang, device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=4000, text_det_limit_type="max",
        )

    def __call__(self, img: np.ndarray) -> list[Word]:
        words: list[Word] = []
        # bardzo wysokie obrazy (kilka stron sklejonych) tniemy na kawałki
        h, w = img.shape[:2]
        step = int(w * 1.5)
        for y in range(0, h, step):
            tile = img[y:min(h, y + step + 60)]
            for res in self.ocr.predict(tile):
                for t, b, s in zip(res["rec_texts"], res["rec_boxes"], res["rec_scores"]):
                    if not t.strip():
                        continue
                    bx = Box(float(b[0]), float(b[1]) + y, float(b[2]), float(b[3]) + y)
                    # odrzuć duplikaty z zakładki między kafelkami
                    if any(abs(o.box.cy - bx.cy) < 4 and abs(o.box.x0 - bx.x0) < 6 for o in words):
                        continue
                    words.extend(_split_line(t, bx, float(s)))
        return words


TESSDATA_LOCAL = Path(__file__).with_name("tessdata")      # modele trzymane w projekcie (przenośne)


def find_tesseract() -> str | None:
    p = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if p:
        return p
    for c in ("C:/Program Files/Tesseract-OCR/tesseract.exe",
              os.path.expandvars("%LOCALAPPDATA%/Programs/Tesseract-OCR/tesseract.exe"),
              "/usr/bin/tesseract"):
        if os.path.exists(c):
            return c
    return None


class TesseractBackend:
    """Tesseract 5 (LSTM). Modele: form2json/tessdata/pol.traineddata (tessdata_best).

    Obraz jest powiększany do ~300 DPI (A4 = 2480 px szerokości) - przy 150 DPI
    Tesseract gubi polskie znaki diakrytyczne i małe czcionki.
    """
    name = "tesseract"

    def __init__(self, lang: str | None = None, psm: int = 11, target_width: int = 2480):
        import pytesseract
        cmd = find_tesseract()
        if not cmd:
            raise RuntimeError("Nie znaleziono tesseract (zainstaluj go lub ustaw TESSERACT_CMD)")
        pytesseract.pytesseract.tesseract_cmd = cmd
        self.t = pytesseract
        tessdir = os.environ.get("TESSDATA_DIR") or (str(TESSDATA_LOCAL) if (TESSDATA_LOCAL / "pol.traineddata").exists() else "")
        self.lang = lang or os.environ.get("FORM2JSON_OCR_LANG", "pol")
        self.config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        if tessdir:   # przez zmienną środowiskową - cudzysłowy w --tessdata-dir psują się na Windows
            os.environ["TESSDATA_PREFIX"] = tessdir
        self.target_width = target_width

    def __call__(self, img: np.ndarray) -> list[Word]:
        import cv2
        h, w = img.shape[:2]
        k = min(3.0, max(1.0, self.target_width / float(w)))
        big = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_CUBIC) if k > 1.05 else img
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        d = self.t.image_to_data(gray, lang=self.lang, config=self.config, output_type=self.t.Output.DICT)
        out = []
        for i, txt in enumerate(d["text"]):
            # wykropkowania "......" Tesseract zwraca z pewnością 0-30, a to z nich powstają pola "dots"
            if txt.strip() and (float(d["conf"][i]) > 30 or LEADER_RE.search(txt)):
                x, y, ww, hh = (d[c][i] / k for c in ("left", "top", "width", "height"))
                out.append(Word(txt, Box(x, y, x + ww, y + hh), size=hh, conf=float(d["conf"][i]) / 100))
        return out


def get_ocr(name: str | None):
    if name == "auto":
        return TesseractBackend() if find_tesseract() else None
    if not name or name == "none":
        return None
    if name == "paddle":
        return PaddleOCRBackend()
    if name == "paddle-cpu":
        return PaddleOCRBackend(device="cpu")
    if name == "tesseract":
        return TesseractBackend()
    raise ValueError(f"Nieznany OCR: {name}")
