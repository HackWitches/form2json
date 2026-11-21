import os
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(os.environ.get("F2J_FIXTURES", Path(__file__).with_name("fixtures")))
REFS = Path(os.environ.get("F2J_REFS", FIXTURES / "refs"))


def fixture(name: str) -> Path:
    p = FIXTURES / name
    if not p.exists():
        pytest.skip(f"brak pliku testowego {p}")
    return p


def f(label, type="string", section="", role="subject", **kw):
    return {"label": label, "type": type, "section": section, "role": role, **kw}


def generic(fields, title="Wniosek o test", **kw):
    return {"title": title, "fields": fields, "entities": {"company": False, "proxy": False}, **kw}


def all_fields(fd):
    return [g["field"] for t in fd["tabs"] for q in t["questions"] for g in q["field"]["repeatingSectionGrid"]]


def queries(fd):
    for t in fd["tabs"]:
        for q in t["questions"]:
            for x in [q["field"], *[g["field"] for g in q["field"]["repeatingSectionGrid"]]]:
                for st in x["states"]:
                    yield st["query"]


needs_soffice = pytest.mark.skipif(not shutil.which("soffice"), reason="brak LibreOffice")
