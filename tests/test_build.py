import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import all_fields, f, generic, queries
from form2json.build import FormBuilder, clean_title, subject_field, to_form_definition
from form2json.validate import validate


def test_html_from_document_is_escaped():
    fd = to_form_definition(generic([f("Uwagi", hint="<img src=x onerror=alert(1)>")], title="Wniosek <b>&</b>"))
    descs = " ".join(x["description"] for x in all_fields(fd))
    assert "<img" not in descs and "&lt;img src=x onerror=alert(1)&gt;" in descs
    assert "<b>&</b>" not in descs


def test_label_validators_use_word_boundaries_and_at_most_one_format():
    b = FormBuilder("t")
    assert subject_field(b, {"label": "Numer PESEL dziecka", "type": "string"})["validation"] == ["REQUIRED", "PESEL"]
    assert subject_field(b, {"label": "Numer PESEL lub NIP", "type": "pesel"})["validation"] == ["REQUIRED", "PESEL"]
    assert "NIP" not in subject_field(b, {"label": "Opis lokalizacji (np. Snipe)", "type": "string"})["validation"]


def test_acronyms_keep_case():
    assert [clean_title(x) for x in ("NIP", "PESEL", "UZASADNIENIE")] == ["NIP", "PESEL", "Uzasadnienie"]


def test_serial_validator_not_duplicated():
    fd = to_form_definition(generic([f("Seria i numer dowodu osobistego", role="applicant_id")]))
    v = next(x["validation"] for x in all_fields(fd) if "dowodu" in x["title"].lower())
    assert v.count("SERIAL_NUMBER_ID_CARD") == 1


def _apostrophe_form():
    sec = "I. Dane dziecka ('opcjonalnie')"
    return generic([f("Czy posiada", "radio", sec, options=['Tak "zawsze"', "Nie"]),
                    f("Szczegóły", section=sec, depends_on="Czy posiada", depends_value='Tak "zawsze"'),
                    *[f(f"Pole A{i}", section=sec) for i in range(3)],
                    *[f(f"Pole C{i}", section="III. Inne") for i in range(8)]])


def test_jsonata_literals_escaped_and_validated():
    fd = to_form_definition(_apostrophe_form())
    assert validate(fd)[0] == []
    assert any('"Tak \\"zawsze\\""' in q for q in queries(fd))


@pytest.mark.skipif(not shutil.which("node"), reason="brak node")
def test_jsonata_queries_parse_with_reference_implementation():
    jsdir = Path(os.environ.get("F2J_JSONATA_DIR", "."))     # katalog z `npm install jsonata@2.0.6`
    if not (jsdir / "node_modules" / "jsonata").exists():
        pytest.skip("brak pakietu jsonata")
    qs = list(queries(to_form_definition(_apostrophe_form())))
    js = ('const j=require("jsonata");const bad=[];for(const q of JSON.parse(require("fs").readFileSync(0,"utf8")))'
          '{try{j(q)}catch(e){bad.push(q)}};console.log(JSON.stringify(bad))')
    out = subprocess.run(["node", "-e", js], input=json.dumps(qs), capture_output=True, text=True, cwd=jsdir, check=True)
    assert json.loads(out.stdout) == []


def test_tab_titles_unique():
    fs = [f(f"Pole A{i}", section="A. DANE > 1. Nieruchomość") for i in range(4)] + \
         [f(f"Pole B{i}", section="B. CEL") for i in range(8)]
    fd = to_form_definition(generic(fs))
    titles = [t["title"] for t in fd["tabs"]]
    assert len(titles) == len(set(titles)) and validate(fd)[0] == []


def test_depends_value_mapped_to_option_or_dropped():
    fs = [f("Czy posiada psa", "radio", "Psy", options=["Tak", "Nie"]),
          f("Rasa", section="Psy", depends_on="Czy posiada psa", depends_value="tak, posiadam"),
          f("Imię psa", section="Psy", depends_on="Czy posiada psa", depends_value="tak")]
    fd = to_form_definition(generic(fs))
    by = {x["title"]: x for x in all_fields(fd)}
    assert by["Imię psa"]["states"][0]["query"].startswith('("Tak" in')
    assert by["Rasa"]["states"] == [] and "REQUIRED" not in by["Rasa"]["validation"]
    assert validate(fd)[0] == []


def test_repeated_table_rows_become_repeating_section_only():
    rows = [f(lab, "number", "Tabela") for _ in range(3) for lab in ("Długość", "Szerokość")]
    fd = to_form_definition(generic([f("Nazwa inwestycji", section="Tabela"), *rows, f("Uwagi", section="Tabela")]))
    secs = [(q["field"]["type"], [g["field"]["title"] for g in q["field"]["repeatingSectionGrid"]])
            for t in fd["tabs"] if t["title"] == "Informacje o wniosku" for q in t["questions"]]
    assert secs == [("SECTION", ["Nazwa inwestycji"]), ("REPEATING_SECTION", ["Długość", "Szerokość"]),
                    ("SECTION", ["Uwagi"])]
