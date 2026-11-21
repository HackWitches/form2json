# form2json — konwersja formularzy do JSON (self-hosted)

Wejście: `PDF, DOC, DOCX, ODS, XLSX, JPG/JPEG, PNG` → wyjście dla każdego pliku:

| plik | zawartość |
|---|---|
| `*.fields.json` | **ogólny JSON**: lista pól `label` (opis pola) → `type` (rodzaj pola) + sekcja, opcje, bbox, liczba kratek, część adresu, pewność; dodatkowo widok `simple`: `{"Opis pola": …, "Rodzaj pola": …}` |
| `*.form.json` | docelowy **FormDefinition** (`tabs` → `SECTION` → `repeatingSectionGrid`, `flow.links`) |
| `*.pN.debug.png` | (opcja `--debug`) podgląd: wykryte pola, ich typy i powiązane opisy |

## Architektura

```
 plik ──► loaders ──► strony (raster + słowa z pozycjami) ──► detect (geometria) ──► typing_rules
           │  PDF: PyMuPDF (warstwa tekstowa, pogrubienia)                     │
           │  DOC/DOCX/ODS/XLSX: LibreOffice headless → PDF                    ▼
           │  JPG/PNG/skan: deskew + OCR (PaddleOCR GPU / Tesseract)     kandydaci pól
           ▼                                                                   │
      tekst natywny (python-docx / openpyxl / odfpy) ─────────────► [opcjonalnie] VLM na DGX
                                                                    (Qwen3-VL przez vLLM, JSON-schema)
                                                                               ▼
                                                     *.fields.json ──► build ──► *.form.json
```

1. **Jeden tor dla wszystkich formatów** – wszystko sprowadzamy do „obraz strony + słowa z bbox”.
   Formaty natywne mają dokładny tekst (bez błędów OCR), obrazy idą przez OCR.
2. **Detekcja geometrii (OpenCV, bez uczenia)** – morfologia wyciąga linie poziome/pionowe, a obszary
   zamknięte liniami dają ramki/komórki. Klasyfikacja: `box`, `comb` (kratki na znaki, łączone przez
   „-” w datę/kod), `checkbox`, `captioned` (opis w komórce + puste miejsce), `underline`,
   `dots` (wykropkowania z tekstu oraz z rastra).
3. **Wiązanie opisu z polem** – w środku > z lewej (także opisy wieloliniowe) > nad (także nagłówek kolumny
   tabeli) > pod (podpis „(imię i nazwisko)” pod wykropkowaniem); dla checkboxa opis z prawej.
4. **Sekcje** – numerowane, pogrubione nagłówki (`2.`, `2.1.`) → ścieżka sekcji.
5. **Rodzaj pola** – reguły PL (e-mail, telefon, PESEL, NIP, IBAN, kod, data, elementy adresu, kwoty,
   uzasadnienie) + liczba kratek (2-2-4 = data, 2-3 = kod, 11 = PESEL, 10 = NIP, 26 = IBAN) +
   wysokość ramki (textarea). Grupy checkboxów → `radio` z `options`; w „Załącznikach” → `file`.
6. **VLM (opcjonalnie, `--vlm`)** – lokalny model wizyjny dostaje obraz + kandydatów i zwraca poprawioną
   listę (dodaje brakujące, usuwa fałszywe, nadaje sekcje/opcje). Wyjście wymuszone schematem JSON
   (guided decoding vLLM). Pola z VLM dowiązywane są do bbox z geometrii.

## Uruchomienie

```bash
pip install -r requirements.txt pytesseract     # + LibreOffice i Tesseract w systemie
python -m form2json "../" -o ../wyniki --debug                     # wszystkie formaty, CPU
python -m form2json "../" -o ../wyniki --ocr tesseract             # obrazy przez Tesseract (CPU)
python -m form2json "../" -o ../wyniki --ocr paddle --vlm          # DGX: PaddleOCR GPU + VLM
```

## Konfiguracja

* **DGX / VLM** – `config/dgx.toml` (adres vLLM, model, timeouty, TLS). Klucz API w zmiennej
  środowiskowej wskazanej przez `api_key_env` (domyślnie `FORM2JSON_VLM_KEY`).
  VLM włącza flaga `--vlm` albo `enabled = true` w pliku; `--dgx-config` wskazuje inny plik.
* **Tesseract** – modele w `tessdata/` (projekt, przenośne): `pol.traineddata` z *tessdata_best*
  + `osd`/`eng` skopiowane z instalacji. Zmienne: `TESSERACT_CMD`, `TESSDATA_DIR`, `FORM2JSON_OCR_LANG`
  (domyślnie `pol`). Obraz jest powiększany do ~300 DPI przed OCR.
* **LibreOffice** – wykrywany automatycznie (`soffice` w PATH lub `C:\Program Files\LibreOffice`).
