import re
from pathlib import Path

import pytest

from conftest import fixture
from form2json import pipeline
from form2json.pipeline import extract, reanalyze
from form2json.semantic import ADDR_LINE, find_addressee
from form2json.typing_rules import norm

PKG = Path(pipeline.__file__).parent


def test_no_control_characters_in_sources():
    """\\b zapisane jako znak backspace (0x08) psuje regexy bez błędu składni."""
    for p in PKG.glob("*.py"):
        bad = [i for i, line in enumerate(p.read_text("utf-8").splitlines(), 1)
               if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", line)]
        assert bad == [], f"{p.name}: {bad}"


def test_title_regex_word_boundary():
    assert pipeline.TITLE_RE.match(norm("Wniosek o stwierdzenie nadpłaty"))
    assert not pipeline.TITLE_RE.match(norm("Wnioskodawca"))


def test_addressee_skips_address_lines():
    assert ADDR_LINE.search(norm("Urząd Skarbowy ul. Długa"))
    assert not ADDR_LINE.search(norm("Wójt Gminy Wydział."))       # "al." tylko jako osobne słowo
    assert find_addressee(["Burmistrz Miasta ul. Długa", "Wójt Gminy Elbląg"]) == "Wójt Gminy Elbląg"


class StubVLM:
    model, raw = "stub", []

    def refine(self, pages, fields, native_text, title):
        return "Wniosek o ustalenie numeru porządkowego", fields, ""


@pytest.mark.parametrize("name, start", [("f1.pdf", "Wniosek o ustalenie"), ("f5.PDF", "Wniosek o stwierdzenie")])
def test_title_from_document_not_filename(name, start):
    assert extract(fixture(name))["title"].startswith(start)


def test_vlm_title_used_when_geometry_title_is_not_a_form_title(monkeypatch):
    monkeypatch.setattr(pipeline, "guess_title", lambda doc, dets: "f1")
    assert extract(fixture("f1.pdf"), vlm=StubVLM())["title"] == "Wniosek o ustalenie numeru porządkowego"


def test_reanalyze_reapplies_rules_for_geometry_fields():
    g = {"fields": [{"label": "Uwagi (opcjonalnie)", "type": "textarea", "detector": "box", "role": "subject",
                     "required": True},
                    {"label": "Telefon kontaktowy", "type": "phone", "detector": "box", "role": "subject",
                     "required": True},
                    {"label": "Rasa psa", "type": "string", "detector": "vlm", "role": "subject", "required": True}]}
    out = {x["label"]: x for x in reanalyze(g)["fields"]}
    assert out["Uwagi (opcjonalnie)"]["required"] is False
    assert out["Telefon kontaktowy"]["role"] == "contact"
    assert out["Rasa psa"]["required"] is True                    # decyzja VLM zostaje
