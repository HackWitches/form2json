"""Weryfikacja/uzupełnienie pól przez lokalny model wizyjny (VLM) na RTX 4090.

Wszystko działa lokalnie, bez zewnętrznych usług:
  * ollama  (domyślnie) - `ollama serve` + `ollama pull qwen3-vl:8b`; natywne API /api/chat,
              wyjście wymuszone schematem JSON (`format`), akceleracja CUDA na RTX 4090
  * openai  - dowolny LOKALNY serwer zgodny z OpenAI (llama.cpp server, LM Studio) na localhost

Modele (bez douczania), mieszczące się w 24 GB VRAM:
  * qwen3-vl:8b-instruct        - szybki (~6 GB VRAM), domyślny
  * qwen3-vl:30b-a3b-instruct   - MoE A3B (Q4, ~20 GB VRAM), dokładniejszy
  (warianty bez '-instruct' to modele 'thinking' - wolne i zużywają limit tokenów na rozumowanie)
  * qwen2.5vl:7b  - alternatywa

VLM dostaje stronę (obraz) + kandydatów z detektora geometrii + tekst strony
i zwraca poprawioną listę pól z rolą (dane wnioskodawcy / adres / przedmiot wniosku / ...),
obligatoryjnością i zależnościami (warunki SHOW). Bez GPU pipeline działa bez VLM (CPU).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import tomllib
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import requests

from .model import FIELD_TYPES, FieldCandidate

ROLES = (
    "subject",            # przedmiot wniosku (pola specyficzne dla formularza)
    "applicant_name",     # imię / nazwisko / nazwa wnioskodawcy
    "applicant_id",       # PESEL / NIP / REGON / KRS / dowód wnioskodawcy
    "applicant_address",  # adres zamieszkania / siedziby wnioskodawcy
    "correspondence_address",
    "contact",            # telefon / e-mail
    "company",            # dane firmy / podmiotu
    "proxy",              # pełnomocnik
    "attachment",         # załącznik
    "statement",          # oświadczenie / zgoda do zaznaczenia
    "signature",          # podpis
    "place_date",         # miejscowość i data złożenia
    "office_use",         # wypełnia urząd / nr sprawy / pieczęć
)

FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "type": {"type": "string", "enum": list(FIELD_TYPES)},
        "role": {"type": "string", "enum": list(ROLES)},
        "section": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "address_part": {"type": "string"},
        "hint": {"type": "string"},
        "required": {"type": "boolean"},
        "depends_on": {"type": "string"},
        "depends_value": {"type": "string"},
    },
    "required": ["label", "type", "role", "section", "required"],
}
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "addressee": {"type": "string"},
        "fields": {"type": "array", "items": FIELD_SCHEMA},
    },
    "required": ["title", "fields"],
}

SYSTEM = """Jesteś ekspertem od cyfryzacji polskich formularzy urzędowych.
Analizujesz obraz strony formularza i wypisujesz WSZYSTKIE miejsca, w które wnioskodawca wpisuje dane
lub zaznacza wybór: puste ramki, kratki na znaki, checkboxy (□), linie i wykropkowania (.....) do pisania,
puste komórki tabel. Pomijaj nagłówki, pouczenia, podstawę prawną i tekst ciągły.
Dla każdego pola podaj:
- label: krótki opis pola po polsku, jak w formularzu, bez kropek, numeracji i dwukropka
  (np. "Imię i nazwisko", "Kod pocztowy", "Rasa psa"),
- type: string | textarea | number | email | phone | date | address | postal_code | pesel | nip |
  regon | iban | checkbox | radio | signature | file,
- role: subject (przedmiot wniosku), applicant_name, applicant_id, applicant_address,
  correspondence_address, contact (telefon, e-mail), company (dane firmy/podmiotu), proxy (pełnomocnik),
  attachment (załącznik), statement (oświadczenie/zgoda do zaznaczenia), signature, place_date
  (miejscowość i data wypełnienia), office_use (wypełnia urząd, nr sprawy, pieczęć),
- section: nazwa sekcji formularza, w której jest pole (np. "Dane wnioskodawcy"),
- required: true jeśli pole jest obowiązkowe (false gdy "opcjonalnie", "jeśli dotyczy", "nieobowiązkowe"),
- options: dla radio (wybór jednej opcji) lub grupy checkboxów - lista opcji,
- address_part: city | street | building | apartment | postal_code | post_office | commune | county |
  voivodeship | country | full,
- depends_on + depends_value: jeżeli pole wypełnia się tylko po wybraniu opcji w innym polu
  (np. "jeżeli tak, podaj...", "inne - jakie?"), podaj label tamtego pola i wartość opcji.
ZASADY:
1. label ma być krótką nazwą pola (2-6 słów), zrozumiałą bez kontekstu. Fragmenty zdań przeredaguj, np.
   "...wnoszę o stwierdzenie nadpłaty z tytułu ....." -> "Tytuł nadpłaty"; "w kwocie ....." -> "Kwota nadpłaty";
   "w miejscowości ....." -> "Miejscowość"; "Wrocław, dnia ....." -> "Data" (role place_date).
2. Grupę checkboxów lub numerowanych wariantów (1, 2, a), b)), z których wybiera się jeden, zwróć jako
   JEDNO pole typu radio: label = pytanie (np. "Sposób rozliczenia nadpłaty"), options = krótkie nazwy wariantów.
   Jeśli przy wariancie jest miejsce do wpisania (np. numer rachunku), dodaj osobne pole z depends_on = label
   pola radio i depends_value = nazwa wariantu.
3. Kilka linii wykropkowanych pod jednym opisem to jedno pole textarea. Kwoty, liczby, powierzchnie -> number.
4. Załączniki wymienione na liście (do dołączenia) -> type file, role attachment.
5. Oświadczenia do zaznaczenia ("Oświadczam, że...", "Wyrażam zgodę...") -> role statement, label = treść.
6. Nie dubluj pól. Pomijaj pola wypełniane przez urząd tylko oznaczając je role office_use.
Zachowaj kolejność czytania. Odpowiadaj wyłącznie JSON-em zgodnym ze schematem."""


def _encode(img: np.ndarray, max_side: int = 1600) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _tiles(img: np.ndarray) -> list[np.ndarray]:
    """Wysokie obrazy (kilka stron w jednym PNG) dzielimy na fragmenty ~A4 z zakładką."""
    h, w = img.shape[:2]
    step = int(w * 1.414)
    if h <= step * 1.15:
        return [img]
    return [img[y:min(h, y + step + int(0.05 * step))] for y in range(0, h, step)]


DEFAULT_CONFIG = Path(__file__).with_name("config") / "local.toml"


def load_config(path: str | Path | None = None) -> dict:
    """Wczytuje config/local.toml; zmienne FORM2JSON_VLM_URL/_MODEL/_BACKEND mają pierwszeństwo."""
    p = Path(path) if path else DEFAULT_CONFIG
    cfg: dict = {}
    if p.exists():
        with open(p, "rb") as f:
            cfg = tomllib.load(f).get("vlm", {})
    cfg.setdefault("enabled", False)
    cfg.setdefault("backend", "ollama")
    cfg.setdefault("base_url", "http://localhost:11434")
    cfg.setdefault("model", "qwen3-vl:8b-instruct")
    cfg.setdefault("api_key_env", "FORM2JSON_VLM_KEY")
    cfg.setdefault("verify_tls", True)
    cfg.setdefault("ca_bundle", "")
    cfg.setdefault("timeout_s", 600)
    cfg.setdefault("max_tokens", 8000)
    cfg.setdefault("temperature", 0.0)
    cfg.setdefault("max_image_side", 1600)
    cfg.setdefault("num_ctx", 16384)
    cfg.setdefault("use_candidates", False)
    cfg["backend"] = os.environ.get("FORM2JSON_VLM_BACKEND", cfg["backend"])
    cfg["base_url"] = os.environ.get("FORM2JSON_VLM_URL", cfg["base_url"])
    cfg["model"] = os.environ.get("FORM2JSON_VLM_MODEL", cfg["model"])
    cfg["api_key"] = os.environ.get(cfg["api_key_env"], "EMPTY")
    return cfg


class VLMRefiner:
    def __init__(self, config: dict | None = None, **overrides):
        cfg = dict(config or load_config())
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        self.cfg = cfg
        self.backend = cfg["backend"]
        self.base = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.key = cfg.get("api_key", "EMPTY")
        self.verify = cfg["ca_bundle"] or bool(cfg["verify_tls"])

    # ------------------------------------------------------------ transport
    def _chat(self, images: list[np.ndarray], text: str) -> dict:
        b64 = [_encode(i, self.cfg["max_image_side"]) for i in images]
        if self.backend == "ollama":
            body = {
                "model": self.model, "stream": False, "think": False, "format": SCHEMA,
                "options": {"temperature": self.cfg["temperature"], "num_ctx": self.cfg["num_ctx"],
                            "num_predict": self.cfg["max_tokens"]},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": text, "images": b64}],
            }
            r = requests.post(f"{self.base}/api/chat", json=body, timeout=self.cfg["timeout_s"])
            r.raise_for_status()
            txt = r.json()["message"]["content"]
        else:  # lokalny serwer zgodny z OpenAI (llama.cpp / LM Studio)
            base = self.base if self.base.endswith("/v1") else self.base + "/v1"
            content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in b64]
            content.append({"type": "text", "text": text})
            body = {
                "model": self.model, "temperature": self.cfg["temperature"], "max_tokens": self.cfg["max_tokens"],
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "form_fields", "schema": SCHEMA}},
            }
            r = requests.post(f"{base}/chat/completions", json=body, timeout=self.cfg["timeout_s"],
                              headers={"Authorization": f"Bearer {self.key}"}, verify=self.verify)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
        return json.loads(txt)

    def available(self) -> bool:
        try:
            u = f"{self.base}/api/tags" if self.backend == "ollama" else \
                (self.base if self.base.endswith("/v1") else self.base + "/v1") + "/models"
            return requests.get(u, timeout=5, verify=self.verify).ok
        except Exception:
            return False

    # ------------------------------------------------------------ refine
    def refine(self, pages, cands: list[FieldCandidate], native_text: str,
               title_guess: str) -> tuple[str, list[FieldCandidate], str]:
        """Zapytanie per strona (mniejsze modele lepiej radzą sobie z jedną stroną)."""
        res: list[FieldCandidate] = []
        self.raw: list[dict] = []          # surowe odpowiedzi modelu (zapisywane w *.fields.json)
        title, addressee = title_guess, ""
        for page in pages:
            pc = [c for c in cands if c.page == page.index]
            page_text = " ".join(w.text for w in page.words)[:6000]
            for ti, tile in enumerate(_tiles(page.image)):
                if self.cfg.get("use_candidates"):
                    cj = [{"label": c.label, "type": c.type, "section": c.section,
                           **({"options": c.options} if c.options else {})} for c in pc]
                    hint = (f"Kandydaci znalezieni automatycznie przez analizę geometrii (tylko podpowiedź: mają "
                            f"ucięte opisy, duplikaty i braki - przeredaguj etykiety zgodnie z ZASADAMI, uzupełnij "
                            f"brakujące pola, usuń fałszywe, połącz warianty w radio):\n"
                            f"{json.dumps(cj, ensure_ascii=False)}\n\n")
                else:
                    # małe modele "kopiują" etykiety kandydatów - podajemy tylko liczbę wykrytych miejsc
                    hint = (f"Analiza geometrii (ramki, kratki, wykropkowania) znalazła na tej stronie ok. "
                            f"{len(pc)} miejsc do wypełnienia - nie pomiń żadnego.\n\n")
                prompt = (f"Tytuł formularza (wstępnie): {title_guess}\nStrona {page.index + 1}.\n\n" + hint
                          + (f"Tekst strony (warstwa tekstowa/OCR, pomocniczo):\n{page_text}\n\n" if page_text else "")
                          + "Zwróć kompletną listę pól tej strony w zadanym schemacie JSON.")
                t0 = time.time()
                try:
                    out = self._chat([tile], prompt)
                except Exception as e:  # VLM jest opcjonalny - w razie błędu zostają kandydaci z geometrii
                    print(f"  [VLM] strona {page.index + 1}: błąd {e} - używam wyniku geometrii", file=sys.stderr)
                    res.extend(pc)
                    continue
                print(f"  [VLM] strona {page.index + 1}.{ti + 1}: {len(out.get('fields', []))} pól, "
                      f"{time.time() - t0:.1f}s", file=sys.stderr)
                self.raw.append({"page": page.index, "tile": ti, "response": out})
                if page.index == 0 and ti == 0:
                    title = out.get("title") or title
                    addressee = out.get("addressee") or ""
                res.extend(self._bind(out.get("fields", []), pc, page.index))
        return title, res, addressee

    @staticmethod
    def _bind(fields: list[dict], cands: list[FieldCandidate], page: int) -> list[FieldCandidate]:
        res = []
        for f in fields:
            lab = (f.get("label") or "").strip(" .:…")
            if not lab:
                continue
            fc = FieldCandidate(label=lab, type=f.get("type", "string"), section=f.get("section", ""),
                                options=[o for o in (f.get("options") or []) if o.strip()],
                                address_part=f.get("address_part") or None, hint=f.get("hint", ""),
                                page=page, detector="vlm", confidence=0.8)
            fc.role = f.get("role") or ""
            fc.required = f.get("required")
            if f.get("depends_on"):
                fc.depends_on, fc.depends_value = f["depends_on"], f.get("depends_value", "")
            best, score = None, 0.0
            for c in cands:
                if getattr(c, "_taken", False):
                    continue
                s = SequenceMatcher(None, c.label.lower(), lab.lower()).ratio()
                if s > score:
                    best, score = c, s
            if best is not None and score >= 0.72:
                best._taken = True  # type: ignore[attr-defined]
                fc.bbox, fc.label_bbox = best.bbox, best.label_bbox
                fc.max_length = best.max_length
                fc.detector = f"{best.detector}+vlm"
                fc.confidence = 0.95
            res.append(fc)
        return res
