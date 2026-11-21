# form2json — konwersja formularzy papierowych do JSON (on-premise, RTX 4090 / CPU)

Wejście: `PDF, DOC, DOCX, ODS, XLSX, JPG/JPEG, PNG`. Dla każdego pliku powstają:

| plik | zawartość |
|---|---|
| `*.simple.json` | **ogólny JSON klucz → wartość**: `Opis pola`, `Rodzaj pola` (string / number / email / phone / date / address / checkbox), rodzaj szczegółowy (pesel, nip, iban, postal_code, textarea, radio, file…), rola, obligatoryjność, opcje, zależność |
| `*.fields.json` | pełny wynik pośredni: pola + bbox, sekcje, adresat, wykryte podmioty, surowa odpowiedź VLM |
| `*.form.json` | docelowy **FormDefinition** (`tabs` → `SECTION` → `repeatingSectionGrid`, `flow.links`), zgodny ze schematem `importowanyPlikJsonSchemat.json` i z regułami technicznymi |

Wszystko działa lokalnie: bez zewnętrznych API, bez trenowania modeli.

## Architektura

```
plik ─► loaders ─► strony: raster + słowa z pozycjami ─► detect (OpenCV) ─► kandydaci pól
         PDF: PyMuPDF (warstwa tekstowa)                  ramki, kratki, checkboxy,
         DOC/DOCX/ODS/XLSX: LibreOffice → PDF             linie, wykropkowania + opisy
         JPG/PNG: prostowanie + Tesseract (pol)                     │
                                                                     ▼
                                     VLM na RTX 4090 (Ollama, qwen3-vl:8b-instruct)
                                     obraz strony + tekst → pola, rola, wymagalność,
                                     zależności (JSON wymuszony schematem); bbox z geometrii
                                                                     │
                                   reguły PL (typing_rules): PESEL, NIP, REGON, IBAN, kod,
                                   data, kwoty, adres; semantic: rola, wymagalność, SHOW
                                                                     │
               *.simple.json / *.fields.json ─► build (standard formularzy) ─► *.form.json ─► validate
```

1. **Jeden tor dla wszystkich formatów**: każdy plik staje się „obrazem strony + słowami z bbox”.
   Formaty z tekstem (PDF, Office) mają dokładny tekst, obrazy idą przez OCR.
2. **Geometria (OpenCV, bez uczenia)**: linie, ramki, kratki (2-2-4 = data, 2-3 = kod, 11 = PESEL…),
   checkboxy, wykropkowania. Opis pola szukany w środku, z lewej, nad i pod polem.
3. **VLM (GPU)**: lokalny Qwen3-VL czyta stronę i zwraca krótkie, przeredagowane etykiety
   („w kwocie ……” → „Kwota nadpłaty”), łączy warianty w `radio`, wykrywa pola warunkowe
   („jeżeli tak, podaj…”), nadaje rolę (dane wnioskodawcy, adres, przedmiot wniosku, załącznik,
   oświadczenie, podpis, pole urzędu). Wynik jest walidowany regułami. Bez GPU ten krok jest pomijany.
4. **Builder zgodny ze standardem** („Standardy i dobre praktyki”, „Reguły techniczne”):
   * strona **Dane**: Urząd/Organ (adresat wykryty z dokumentu, tylko do odczytu), Podmiot wniosku,
     Twoje dane (dane inicjalne `MLPZ.*`, `EXTERNAL_SYSTEM`, tylko do odczytu), Adres zamieszkania
     (Kraj ze słownikiem `Kraje`, `TERYT_4G_DOWODOWY`, warianty PL i zagranica przez jsonata `SHOW`),
     warunkowo: Dane innej osoby, Dane podmiotu, Adres siedziby, Pełnomocnictwo, Opłata skarbowa (v1, v2)
   * strony **przedmiotu wniosku**: pola z formularza; `date` → `DATEPICKER`, kwoty → `NUMBER`,
     opisy → `TEXTAREA`, wybór jednej opcji → `RADIO` (>5 opcji → `SELECT`, >15 z wyszukiwarką),
     wielokrotny → `CHECKBOX`, załączniki → `FILE_UPLOADER`, powtarzane wiersze tabel → `REPEATING_SECTION`
   * strona **Kontakt i zgody**: telefon/e-mail (`MLPZ`, walidator `EMAIL`), oświadczenia z formularza
     → `CONSENT_SECTION`, zgoda RODO
   * strona **FINISH**: wyłącznie `INFO_SECTION` (podgląd)
   * walidatory: `REQUIRED`, `EMAIL`, `PESEL`, `NIP`, `REGON`, `ZIP_CODE`, `SERIAL_NUMBER_ID_CARD`
   * pomijane: podpis (zastępuje go podpis elektroniczny), „miejscowość, data”, pola „wypełnia urząd”
5. **Walidator** (`validate.py`): schemat JSON + reguły techniczne (≥2 strony, FINISH na końcu i tylko
   INFO_SECTION, pola w sekcjach, brak zagnieżdżeń, unikalne id i techName, opcje albo słownik,
   poprawne odwołania jsonata, osiągalność stron w `flow.links`).

## Instalacja (Windows, RTX 4090)

```powershell
python -m venv .venv-form2json
.venv-form2json\Scripts\python -m pip install -r form2json\requirements.txt
winget install UB-Mannheim.TesseractOCR
winget install TheDocumentFoundation.LibreOffice
winget install Ollama.Ollama
ollama pull qwen3-vl:8b-instruct
```

Używaj wariantów `-instruct`. Domyślne tagi `qwen3-vl:8b` i `qwen3-vl:30b` to modele „thinking”:
zużywają limit tokenów na rozumowanie i zwracają pustą odpowiedź.

## Uruchomienie

Najprościej: ustaw ścieżki w `form2json/runtime.config` i uruchom bez parametrów
(w PyCharm/IntelliJ: *Script path* = `form2json\run.py`, bez parametrów i bez katalogu roboczego):

```powershell
python form2json\run.py            # albo: python -m form2json   (z katalogu Formularze-Wypakowane)
python form2json\run.py --fast     # prostszy, szybszy model (profile = fast)
```

`runtime.config`:

| sekcja / klucz | znaczenie |
|---|---|
| `[paths] input`, `output` | katalog lub pliki wejściowe (kilka rozdzielonych `;`), katalog wyników |
| `[model] vlm_enabled` | `true` = VLM na GPU, `false` = tylko geometria + OCR (CPU) |
| `[model] profile` | `default` (**qwen3-vl:8b-instruct**, model domyślny) / `fast` (qwen3-vl:4b-instruct) / `accurate` (qwen3-vl:30b-a3b-instruct) |
| `[performance] file_workers`, `vlm_parallel`, `server_parallel` | pliki naraz, strony jednego pliku naraz do VLM, zapytania liczone jednocześnie przez serwer Ollama (1 = po kolei) |
| `[processing] ocr`, `tabs`, `dpi`, `debug` | jak parametry `--ocr`, `--tabs`, `--dpi`, `--debug` |
| `[report] enabled`, `ref_dir` | generowanie `RAPORT.md` w katalogu wyników i katalog z plikami `konfiguracja_form_*.json` |
| `[timing] enabled` | czas pliku i czas łączny w linii `[OK]` + podsumowanie (pliki, strony, pola, czas łączny, średnio na plik i na stronę) |

Parametry z linii poleceń mają pierwszeństwo przed `runtime.config` (`--no-report`, `--no-timing`,
`--profile`, `--vlm-model`, `--runtime-config inny.config`...). Jeśli model wybranego profilu nie jest pobrany
w Ollamie, program wypisze polecenie `ollama pull` i użyje modelu domyślnego.

### Przyspieszenie (RTX 4090, CUDA)

Ollama już liczy model na GPU (CUDA, cały model w VRAM), ale domyślnie obsługuje jedno zapytanie naraz
i bez flash attention. Przyspieszenie (~3× przy tej samej dokładności) działa tak samo na każdym systemie,
bez ustawiania niczego w systemie:

* **lokalnie (Windows / Linux / macOS)**: `[ollama] manage = auto` w `runtime.config` - form2json
  (`ollama_server.py`) sam uruchamia `ollama serve` na porcie 11435 z `OLLAMA_FLASH_ATTENTION=1`
  i `OLLAMA_NUM_PARALLEL=4`, w razie potrzeby pobiera model przez API i zamyka serwer po zakończeniu.
  Wymaga tylko zainstalowanej Ollamy; modele są współdzielone z systemową Ollamą.
* **Docker** (`deploy/docker-compose.yml`): kontener `ollama/ollama` na GPU z tymi samymi zmiennymi
  + kontener form2json (LibreOffice, Tesseract `pol`). Na innym komputerze wystarczy Docker z obsługą GPU
  NVIDIA (Linux: NVIDIA Container Toolkit, Windows: Docker Desktop + WSL2):

```powershell
cd form2json
$env:INPUT_DIR = "C:\formularze"; $env:OUTPUT_DIR = "C:\wyniki"
docker compose -f deploy/docker-compose.yml up --build --abort-on-container-exit form2json
```

Pierwsze uruchomienie w Dockerze pobiera model (~6 GB) do wolumenu `ollama`.
`vlm_parallel` (strony naraz) i `file_workers` (pliki naraz) w `runtime.config` działają w obu trybach.

Pomiar, 5 plików / 22 strony, w tym 2 z referencją:

| ustawienie | czas | s/stronę | recall (śr.) |
|---|---|---|---|
| poprzednio (Ollama domyślnie, strony po kolei) | 143 s | 6,5 | 89% |
| + `OLLAMA_FLASH_ATTENTION=1` | 99 s | 4,5 | 89% |
| + `OLLAMA_NUM_PARALLEL=4`, `vlm_parallel = 4` | 70 s | 3,2 | 89% |
| obraz strony 1280 px zamiast 1600 (odrzucone) | 59 s | 2,7 | 88%, ale recall przedmiotu wniosku spada do 43% na skanie JPG |

4 pliki z `test-kilku-plikow` (36 stron): 3 min 56 s → 1 min 21 s (`file_workers = 2`), dokładność bez zmian.
Bez zmiennych w Ollamie `vlm_parallel` nic nie psuje, ale też nie przyspiesza (zapytania czekają w kolejce).

Pozostałe tryby:

```powershell
# wszystkie formularze w katalogu (GPU, jeśli Ollama działa; inaczej automatycznie CPU)
python -m form2json . -o wyniki

# tylko CPU (Ryzen), bez VLM
python -m form2json . -o wyniki --no-vlm

# większy model (MoE, ~20 GB VRAM)
python -m form2json plik.pdf -o wyniki --vlm-model qwen3-vl:30b-a3b-instruct

# po zmianie reguł: przebuduj FormDefinition z zapisanych *.fields.json (bez OCR i VLM, sekundy)
python -m form2json --rebuild "wyniki/*.fields.json" -o wyniki

# raport zbiorczy (walidacja, recall względem referencji, spójność formatów) -> wyniki/RAPORT.md
python -m form2json.report wyniki --ref .

# podgląd, walidacja, porównanie z referencją
python -m form2json.show wyniki\plik.pdf.form.json
python -m form2json.validate wyniki\*.form.json
python -m form2json.compare wyniki\plik.pdf.form.json konfiguracja_form_....json
```

Konfiguracja VLM: `config/local.toml` (model, adres Ollamy, rozmiar obrazu, kontekst). Zmienne
`FORM2JSON_VLM_URL` i `FORM2JSON_VLM_MODEL` nadpisują wartości z pliku. Tesseract: modele w
`tessdata/` (`pol` z tessdata_best), zmienne `TESSERACT_CMD`, `TESSDATA_DIR`, `FORM2JSON_OCR_LANG`.

## Moduły

| moduł | rola |
|---|---|
| `loaders.py` | wczytanie formatu → strony (raster + słowa); LibreOffice headless dla Office |
| `ocr.py` | Tesseract (CPU, domyślnie), opcjonalnie PaddleOCR |
| `detect.py` | detekcja geometrii pól i wiązanie opisów |
| `typing_rules.py` | rodzaj pola z opisu / liczby kratek / kształtu |
| `llm.py` | lokalny VLM (Ollama) z wymuszonym schematem JSON |
| `semantic.py` | role, obligatoryjność, zależności, adresat, typ wnioskodawcy |
| `build.py` | FormDefinition zgodny ze standardem i regułami |
| `validate.py` | walidacja schematu i reguł |
| `compare.py`, `show.py` | ocena względem referencji, podgląd |

## Wyniki na materiałach konkursowych (RTX 4090, qwen3-vl:8b-instruct)

* 9 formularzy × wszystkie dostępne formaty = **69 plików, 69/69 FormDefinition bez błędów walidacji**
  (schemat JSON + reguły techniczne). Pełna tabela: `wyniki/RAPORT.md`.
* Porównanie z plikami referencyjnymi (`python -m form2json.compare`):

| formularz | recall (cały formularz) | recall: przedmiot wniosku | zgodność typów |
|---|---|---|---|
| Wniosek o ustalenie numeru porządkowego nieruchomości | 91% | 100% | 98% |
| Wniosek o stwierdzenie nadpłaty podatku | 84% | 71% | 98% |

  Wynik jest jednakowy dla PDF / DOC / DOCX / ODS / XLSX / JPG / PNG tego samego formularza.
* Czas: ~10–20 s na stronę (LibreOffice + OCR + VLM). Bez GPU (`--no-vlm`) kilka sekund na plik,
  ale z gorszymi etykietami pól („w kwocie” zamiast „Kwota nadpłaty”).
* Zabezpieczenia: wysokie skany (kilka stron w jednym obrazie) są dzielone na strony A4; ucięty JSON z
  modelu jest odzyskiwany do ostatniego pełnego pola; powtórzenia pól z „pętli” modelu są usuwane;
  role z VLM są weryfikowane regułami (np. „Termin wynajęcia od” nie jest datą złożenia wniosku).
