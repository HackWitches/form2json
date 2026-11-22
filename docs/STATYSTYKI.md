# form2json vs service na lokalnym modelu (Spark, qwen3-vl:30b-a3b-instruct)

Stan: 22.11.2026, gałąź `fix/code-review` (PR #1) przed scaleniem z `master` (równoległy VLM nie zmienia wyników, tylko czas).
Ten sam model dla obu rozwiązań: Ollama na DGX Spark, `qwen3-vl:30b-a3b-instruct` (GGUF Q4), przez tunel `127.0.0.1:18001`.
Wyniki `service` pochodzą z `cyfryzacja-formularzy/temp/` (Claude) i `temp/ollama/` (Ollama, bez formularza 07).

## Wniosek

Na tym samym lokalnym modelu `service` jest trochę lepszy od `form2json`: więcej znalezionych pól i wyraźnie więcej poprawnych warunków, przy podobnej liczbie pól bez błędów.
Przewaga z `service` + Claude (88% / 72%) bierze się z modelu, którego regulamin nie dopuszcza.
`form2json` daje identyczny wynik dla PDF, DOCX, JPG i PNG; tego nie zmierzono dla `service` z Ollamą (w `temp/ollama` są tylko PDF).
Oba rozwiązania z lokalnym modelem źle ustalają warianty (podmiot, pełnomocnik); bez VLM `form2json` ustala je regułą poprawnie.

## Miara 1: `eval/porownaj.py` (anotacje zespołu, przedmiot wniosku)

Suma dla formularzy wspólnych dla wszystkich wariantów (01-06, 08, 09; 137 oczekiwanych pól):

| Wariant | Znalezione | Bez błędów | Nadmiarowe | Warunki poprawne | Warianty źle |
|---|---|---|---|---|---|
| service + Claude (niedozwolony) | 121 (88%) | 99 (72%) | 13 | 54/57 | 0/8 |
| **service + Ollama** | **97 (71%)** | **41 (30%)** | 43 | **16/42** | 4/8 |
| **form2json + VLM** | **86 (63%)** | **38 (28%)** | 69 | **5/40** | 2/8 |
| form2json bez VLM | 68 (50%) | 23 (17%) | 81 | 0/23 | 1/8 |

Uwaga: anotacje powstały pod kontrakt `service`, więc ta miara go faworyzuje.
`form2json` jest liczony przez adapter `python -m form2json.contract_v2` (pola wnioskodawcy idą do "pominiete: dane bazy", tak jak w `service`).

Per formularz (znalezione / oczekiwane, pola bez błędów, warunki):

| Form. | service + Claude | service + Ollama | form2json + VLM | form2json bez VLM |
|---|---|---|---|---|
| 01 | 4/4, 4 czystych, war. 0/0 | 4/4, 3 czystych, war. 0/0, warianty źle | 4/4, 3 czystych, war. 0/0, warianty źle | 3/4, 2 czystych, war. 0/0 |
| 02 | 7/7, 6 czystych, war. 2/2 | 7/7, 5 czystych, war. 2/2, warianty źle | 6/7, 6 czystych, war. 2/2 | 5/7, 4 czystych, war. 0/1 |
| 03 | 5/6, 3 czystych, war. 1/1 | 5/6, 0 czystych, war. 1/1, warianty źle | 2/6, 1 czystych, war. 1/1 | 2/6, 0 czystych, war. 0/1 |
| 04 | 10/10, 8 czystych, war. 0/2 | 4/10, 3 czystych, war. 0/0 | 7/10, 4 czystych, war. 0/0 | 6/10, 4 czystych, war. 0/0 |
| 05 | 7/7, 6 czystych, war. 2/2 | 6/7, 4 czystych, war. 0/1 | 5/7, 4 czystych, war. 1/1 | 5/7, 2 czystych, war. 0/1 |
| 06 | 22/23, 16 czystych, war. 2/2 | 21/23, 11 czystych, war. 0/2 | 15/23, 8 czystych, war. 0/0 | 13/23, 7 czystych, war. 0/0 |
| 07 | 16/17, 13 czystych, war. 1/2 | - | 10/17, 1 czystych, war. 0/2 | 9/17, 0 czystych, war. 0/3 |
| 08 | 45/57, 40 czystych, war. 36/36 | 32/57, 12 czystych, war. 11/24 | 32/57, 9 czystych, war. 1/27 | 19/57, 1 czystych, war. 0/11 |
| 09 | 21/23, 16 czystych, war. 11/12 | 18/23, 3 czystych, war. 2/12, warianty źle | 15/23, 3 czystych, war. 0/9, warianty źle | 15/23, 3 czystych, war. 0/9, warianty źle |

## Miara 2: `python -m form2json.compare` (oficjalne referencje COI, cały formularz)

| Wariant | Form. 1: recall / precyzja | Form. 5: recall / precyzja |
|---|---|---|
| service + Claude (niedozwolony) | 94% / 96% | 93% / 91% |
| service + Ollama | 66% / 76% | 91% / 76% |
| form2json + VLM | 62% / 87% | 84% / 77% |
| form2json bez VLM | 89% / 87% | 84% / 79% |

Formularz 1 z lokalnym modelem traci około 25 punktów w obu rozwiązaniach z jednego powodu: model zamienia "imię i nazwisko/nazwa wnioskodawcy" na "Imię i nazwisko", więc wariant "firma" znika razem z sekcją podmiotu.
`form2json` bez VLM liczy wariant z surowych opisów (`semantic.detect_entities`) i tego błędu nie ma.

## Spójność formatów (form2json, compare.py)

| Plik | bez VLM: recall / przedmiot | VLM: recall / przedmiot |
|---|---|---|
| 01 PDF, DOCX | 89% / 75% | 62% / 100% |
| 01 JPG, PNG | 87% / 50% | 62% / 100% |
| 05 PDF, DOCX | 84% / 57% | 84% / 71% |
| 05 JPG | 82% / 29% | 84% / 71% |
| 05 PNG | 84% / 43% | 84% / 71% |

Z VLM wynik jest taki sam dla każdego formatu tego samego formularza, a przedmiot wniosku ze skanu jest tak dobry jak z PDF.

## Czas

`form2json` + VLM: 41 stron w 567 s, mediana 12,3 s na stronę (pierwsza strona 143 s przez ładowanie modelu).
`service` + Ollama: 79-200 s na formularz (`temp/ollama.*.log`).

## Wizualizacje

`docs/wizualizacje/NN_wynik.pdf`: form2json + VLM (Spark), formularze 01-09, każda strona papieru z naniesionymi polami plus strona podsumowania.
Rysowane skryptem `cyfryzacja-formularzy/dev/visualize_output.py`, tym samym co wyniki `service`, więc oba rozwiązania można oglądać obok siebie.
Wariant bez VLM i `service` + Ollama nie są w repozytorium; odtwarza się je komendami z README (sekcja "Wizualizacja").

Pola, które VLM dodał bez ramki z geometrii, są przypięte do linii po tekście opisu, więc ich położenie jest przybliżone.

## Jak odtworzyć

Z katalogu nadrzędnego, w którym leżą obok siebie `form2json` i `cyfryzacja-formularzy`:

```bash
ssh -N -L 18001:127.0.0.1:11434 mk@100.105.35.36    # tunel do Ollamy na Sparku
FORM2JSON_VLM_URL=http://127.0.0.1:18001 python -m form2json --vlm --vlm-model qwen3-vl:30b-a3b-instruct \
  --ollama never -o form2json/wyniki_vlm <katalog z plikami 01.pdf ... 09.pdf>
python -m form2json.contract_v2 form2json/wyniki_vlm --service cyfryzacja-formularzy/service
cd cyfryzacja-formularzy && python3 eval/porownaj.py ../form2json/wyniki_vlm/kontrakt
service/.venv/bin/python dev/visualize_output.py 01 02 03 04 05 06 07 08 09 \
  --in ../form2json/wyniki_vlm/kontrakt --out ../form2json/wyniki_vlm/viz
```
