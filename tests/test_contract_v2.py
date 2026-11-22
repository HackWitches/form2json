from form2json.contract_v2 import convert

# ramki w pikselach 150 dpi; linie w punktach PDF (x 72/150), strony linii od 1
LINES = [{"id": "L001", "page": 1, "bbox": [10, 10, 200, 20], "text": "a"},
         {"id": "L002", "page": 1, "bbox": [10, 30, 200, 40], "text": "b"},
         {"id": "L003", "page": 2, "bbox": [10, 10, 200, 20], "text": "tak"}]


def px(*pt):
    return [round(v * 150 / 72) for v in pt]


def test_convert_cites_lines_dedupes_names_and_skips_base_roles():
    g = {"title": "Wniosek", "addressee": "Wójt", "entities": {"company": True}, "fields": [
        {"label": "Imię", "type": "string", "role": "applicant_name", "page": 0, "bbox": px(20, 12, 50, 18)},
        {"label": "Czy psy?", "type": "radio", "role": "subject", "page": 0, "bbox": px(20, 32, 50, 38),
         "options": ["tak", "nie"]},
        {"label": "tak", "type": "string", "role": "subject", "page": 1, "bbox": px(20, 12, 50, 18),
         "depends_on": "Czy psy?", "depends_value": "tak"},
        {"label": "tak", "type": "string", "role": "subject", "page": 1},
    ]}
    c = convert(g, LINES)
    assert c["pominiete"] == [{"linie": ["L001"], "powod": "dane bazy"}]
    pola = c["strony"][0]["sekcje"][0]["pola"]
    assert [p["techName"] for p in pola] == ["czy_psy", "tak", "tak_2"]
    assert [p["linie"] for p in pola] == [["L002"], ["L003"], ["L003"]]  # ostatnie bez ramki: po tekście
    assert pola[1]["warunek"] == {"pole": "czy_psy", "wartosc": "tak"} and pola[0]["warunek"] is None
    assert c["warianty"] == {"podmiot": True, "pelnomocnik": True}


def test_field_without_box_is_matched_by_label_text():
    lines = [{"id": "L001", "page": 1, "bbox": [0, 0, 1, 1], "text": "Wnioskodawca:"},
             {"id": "L002", "page": 1, "bbox": [0, 0, 1, 1], "text": "2) zwrócić przelewem na wskazany rachunek"}]
    g = {"fields": [{"label": "Numer rachunku", "type": "iban", "role": "subject", "page": 0},
                    {"label": "Oświadczenie o prawdzie", "type": "checkbox", "role": "statement", "page": 0}]}
    pola = convert(g, lines)["strony"][0]["sekcje"][0]["pola"]
    assert [p["linie"] for p in pola] == [["L002"], []]
