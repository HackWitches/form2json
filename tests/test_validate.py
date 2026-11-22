import copy
import json

import pytest

from conftest import REFS, f, generic
from form2json.build import to_form_definition
from form2json.validate import validate


def _ok_form():
    return to_form_definition(generic([f("Czy posiada psa", "radio", "Psy", options=["Tak", "Nie"]),
                                       f("Rasa", section="Psy", depends_on="Czy posiada psa", depends_value="Tak")]))


def test_generated_form_is_valid():
    assert validate(_ok_form())[0] == []


def test_empty_object_is_rejected_by_schema():
    assert any("'tabs' is a required property" in e for e in validate({})[0])


def test_nested_field_garbage_is_rejected_by_schema():
    fd = _ok_form()
    fld = fd["tabs"][0]["questions"][0]["field"]["repeatingSectionGrid"][0]["field"]
    fld["validation"], fld["foo"] = "REQUIRED", 1
    errs = validate(fd)[0]
    assert any("validation" in e for e in errs) and any("foo" in e for e in errs)


def test_duplicate_tab_title_is_error():
    fd = _ok_form()
    fd["tabs"][1]["title"] = fd["tabs"][0]["title"]
    assert any("zduplikowany tytuł strony" in e for e in validate(fd)[0])


def _rasa(fd):
    return next(g["field"] for t in fd["tabs"] for q in t["questions"] for g in q["field"]["repeatingSectionGrid"]
                if g["field"]["title"] == "Rasa")


def test_condition_value_must_be_an_option():
    fd = _ok_form()
    st = _rasa(fd)["states"][0]
    st["query"] = st["query"].replace('"Tak"', '"tak, posiadam"')
    assert any("nie jest opcją" in e for e in validate(fd)[0])


def test_broken_jsonata_is_reported():
    fd = copy.deepcopy(_ok_form())
    st = _rasa(fd)["states"][0]
    st["query"] = st["query"].replace("'Informacje o wniosku'", "'Dane dziecka ('opcjonalnie')'")
    assert any("nierozpoznana JSONata" in w for w in validate(fd)[1])


def test_repeating_section_condition_is_recognized():
    fd = _ok_form()
    st = _rasa(fd)["states"][0]
    st["query"] = st["query"].replace(".field.repeatingSectionGrid['czy_posiada_psa' in field.techName].field.__userAnswer",
                                      ".field.__userAnswer[$repeatingSectionIndex]['czy_posiada_psa' in fieldTechName].answers")
    e, w = validate(fd)
    assert e == [] and not any("nierozpoznana" in x for x in w)


def test_references_pass_new_checks():
    refs = sorted(REFS.glob("*/konfiguracja_form_*.json"))
    if not refs:
        pytest.skip("brak referencji COI")
    for r in refs:
        errs = validate(json.loads(r.read_text("utf-8")))[0]
        # znane odstępstwa referencji od reguł: validation = {} i TEXT_BLOCK na stronie FINISH
        assert [e for e in errs if "validation musi być tablicą" not in e and "FINISH" not in e] == [], r.name
