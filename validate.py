"""Walidacja FormDefinition: schemat JSON (importowanyPlikJsonSchemat) + reguły techniczne.

    python -m form2json.validate wyniki/*.form.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("config") / "importowanyPlikJsonSchemat.json"
SECTION_TYPES = {"SECTION", "REPEATING_SECTION"}
FIELD_TYPES = {"TEXT_FIELD", "CHECKBOX", "NUMBER", "CONSENT_SECTION", "SELECT", "TEXTAREA", "FILE_UPLOADER",
               "RADIO", "DATEPICKER", "TEXT_BLOCK", "INFO_SECTION"}
INPUT_TYPES = FIELD_TYPES - {"TEXT_BLOCK", "INFO_SECTION"}
VALIDATORS = {"REQUIRED", "EMAIL", "PESEL", "NIP", "REGON", "ZIP_CODE", "EPUAP", "ADE", "SERIAL_NUMBER_ID_CARD"}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
TECH_RE = re.compile(r"^[a-z0-9_]+$")


def _schema_errors(fd: dict) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return []
    if not SCHEMA_PATH.exists():
        return []
    schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    return [f"schema: {'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(fd)]


def validate(fd: dict) -> tuple[list[str], list[str]]:
    E, W = _schema_errors(fd), []
    title = fd.get("title", "")
    if not title or len(title) > 255:
        E.append("tytuł formularza pusty lub dłuższy niż 255 znaków")
    tabs = fd.get("tabs", [])
    if len(tabs) < 2:
        E.append("formularz musi mieć co najmniej 2 strony")
    if tabs and tabs[-1].get("type") != "FINISH":
        E.append("ostatnia strona musi mieć typ FINISH")
    ids, techs = set(), set()
    all_tech: set[str] = set()

    def check_id(i: str, what: str):
        if i in ids:
            E.append(f"zduplikowane id: {what}")
        ids.add(i)
        if not UUID_RE.match(i or ""):
            W.append(f"id nie jest UUID: {what}")

    for ti, t in enumerate(tabs):
        check_id(t.get("id", ""), f"tab {t.get('title')}")
        ttype = t.get("type")
        if ttype not in ("MAIN", "FINISH"):
            E.append(f"strona '{t.get('title')}': typ {ttype} (dozwolone MAIN/FINISH)")
        if ttype == "FINISH" and ti != len(tabs) - 1:
            E.append("strona FINISH musi być ostatnia")
        n_fields = 0
        for q in t.get("questions", []):
            sec = q["field"]
            check_id(sec.get("id", ""), f"sekcja {sec.get('title')}")
            if sec.get("type") not in SECTION_TYPES:
                E.append(f"strona '{t.get('title')}': pole '{sec.get('title')}' ({sec.get('type')}) poza sekcją")
                continue
            if not sec.get("title") or not sec.get("techName"):
                E.append(f"sekcja bez title/techName na stronie '{t.get('title')}'")
            for g in sec.get("repeatingSectionGrid", []):
                f = g["field"]
                n_fields += 1
                check_id(f.get("id", ""), f"pole {f.get('techName')}")
                ft, tn = f.get("type"), f.get("techName", "")
                where = f"'{t.get('title')}' > '{sec.get('title')}' > {tn}"
                if ft in SECTION_TYPES:
                    E.append(f"{where}: zagnieżdżona sekcja")
                elif ft not in FIELD_TYPES:
                    E.append(f"{where}: nieznany typ {ft}")
                if not tn:
                    E.append(f"{where}: brak techName")
                elif not TECH_RE.match(tn):
                    W.append(f"{where}: techName zawiera znaki spoza [a-z0-9_]")
                if tn in techs:
                    E.append(f"{where}: zduplikowany techName")
                techs.add(tn)
                all_tech.add(tn)
                if ttype == "FINISH" and ft != "INFO_SECTION":
                    E.append(f"{where}: na stronie FINISH dozwolone tylko INFO_SECTION (jest {ft})")
                if not isinstance(f.get("validation", []), list):
                    E.append(f"{where}: validation musi być tablicą")
                for v in f.get("validation", []):
                    if v not in VALIDATORS:
                        W.append(f"{where}: nieznany walidator {v}")
                if ft in ("RADIO", "CHECKBOX", "SELECT") and not f.get("options") and not f.get("dictionaryName"):
                    E.append(f"{where}: {ft} bez opcji i bez słownika")
                if ft == "CONSENT_SECTION" and not (f.get("consentSectionConfig") or {}).get("consents"):
                    E.append(f"{where}: CONSENT_SECTION bez zgód")
                if ft == "INFO_SECTION" and not f.get("description"):
                    E.append(f"{where}: INFO_SECTION wymaga description")
            all_tech.add(sec.get("techName", ""))
        if n_fields == 0:
            E.append(f"strona '{t.get('title')}' nie zawiera żadnego pola")

    # stany jsonata - odwołania do istniejących stron/techName
    titles = {t.get("title") for t in tabs}
    for t in tabs:
        for q in t.get("questions", []):
            objs = [q["field"]] + [g["field"] for g in q["field"].get("repeatingSectionGrid", [])]
            for f in objs:
                for st in f.get("states", []) or []:
                    if st.get("type") not in ("SHOW", "READ_ONLY"):
                        E.append(f"{f.get('techName')}: stan {st.get('type')} (dozwolone SHOW/READ_ONLY)")
                    qq = st.get("query", "")
                    for tab_ref in re.findall(r"tabs\['([^']+)' in title\]", qq):
                        if tab_ref not in titles:
                            E.append(f"{f.get('techName')}: jsonata odwołuje się do nieistniejącej strony '{tab_ref}'")
                    for ref in re.findall(r"\['([a-z0-9_]+)' in field\.techName\]", qq):
                        if ref not in all_tech:
                            E.append(f"{f.get('techName')}: jsonata odwołuje się do nieistniejącego pola '{ref}'")

    # przepływ stron
    links = fd.get("flow", {}).get("links", [])
    tab_ids = [t["id"] for t in tabs]
    reach = {tab_ids[0]} if tab_ids else set()
    changed = True
    while changed:
        changed = False
        for ln in links:
            if ln.get("source") in reach and ln.get("target") not in reach:
                reach.add(ln["target"])
                changed = True
    for ln in links:
        if ln.get("source") not in tab_ids or ln.get("target") not in tab_ids:
            E.append(f"flow.links {ln.get('id')}: odwołanie do nieistniejącej strony")
    if set(tab_ids) - reach:
        E.append("nie wszystkie strony są osiągalne przez flow.links")
    return E, W


def main(argv=None) -> int:
    rc = 0
    for p in argv or sys.argv[1:]:
        fd = json.loads(Path(p).read_text("utf-8"))
        e, w = validate(fd)
        print(f"{'OK ' if not e else 'ERR'} {p}: {len(e)} błędów, {len(w)} ostrzeżeń")
        for x in e:
            print("   BŁĄD:", x)
        for x in w[:10]:
            print("   uwaga:", x)
        rc |= bool(e)
    return rc


if __name__ == "__main__":
    sys.exit(main())
