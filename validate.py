"""Walidacja FormDefinition: schemat JSON (importowanyPlikJsonSchemat) + reguły techniczne.

    python -m form2json.validate wyniki/*.form.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import jsonschema

CONFIG = Path(__file__).with_name("config")
# oficjalny schemat (bez zmian) + ścisły schemat form2json: required, additionalProperties, rekurencja po siatce
SCHEMAS = [json.loads((CONFIG / n).read_text("utf-8"))
           for n in ("importowanyPlikJsonSchemat.json", "formDefinition.strict.schema.json")]
SECTION_TYPES = {"SECTION", "REPEATING_SECTION"}
FIELD_TYPES = {"TEXT_FIELD", "CHECKBOX", "NUMBER", "CONSENT_SECTION", "SELECT", "TEXTAREA", "FILE_UPLOADER",
               "RADIO", "DATEPICKER", "TEXT_BLOCK", "INFO_SECTION"}
INPUT_TYPES = FIELD_TYPES - {"TEXT_BLOCK", "INFO_SECTION"}
VALIDATORS = {"REQUIRED", "EMAIL", "PESEL", "NIP", "REGON", "ZIP_CODE", "EPUAP", "ADE", "SERIAL_NUMBER_ID_CARD"}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
TECH_RE = re.compile(r"^[a-z0-9_]+$")


# warunek SHOW: wartość jako literał JSON, nazwy w apostrofach; pole zwykłe albo w sekcji powtarzającej się
ATOM_RE = re.compile(r'(?:\$not)?\(("(?:[^"\\]|\\.)*") in tabs\[\'([^\'\\]+)\' in title\]\.questions\[\'([a-z0-9_]+)\' '
                     r"in field\.techName\]\.field\.(?:repeatingSectionGrid\['([a-z0-9_]+)' in field\.techName\]"
                     r"\.field\.__userAnswer|__userAnswer\[\$repeatingSectionIndex\]\['([a-z0-9_]+)' in fieldTechName\]"
                     r"\.answers)\)")


def _schema_errors(fd: dict) -> list[str]:
    return [f"schema: {'/'.join(map(str, e.path))}: {e.message}"
            for s in SCHEMAS for e in jsonschema.Draft202012Validator(s).iter_errors(fd)]


def _query_errors(q: str, tabs: list[dict]) -> tuple[list[str], list[str]]:
    """(błędy, ostrzeżenia): odwołania do strony, sekcji, pola i opcji; nieznany kształt zapytania -> ostrzeżenie."""
    rest = ATOM_RE.sub("", q)
    warns = [f"nierozpoznana JSONata (nie sprawdzono): {q[:120]}"] \
        if re.sub(r"[()\s]|\b(and|or)\b", "", rest) or rest.count("(") != rest.count(")") else []
    errs = []
    for lit, tab, sec, fld, rfld in ATOM_RE.findall(q):
        value, fld = json.loads(lit), fld or rfld
        ts = [t for t in tabs if t.get("title") == tab]
        secs = [x["field"] for t in ts for x in t.get("questions", []) if x["field"].get("techName") == sec]
        fs = [g["field"] for s_ in secs for g in s_.get("repeatingSectionGrid", []) if g["field"].get("techName") == fld]
        if len(ts) != 1 or not secs or not fs:
            errs.append(f"JSONata: brak pola {tab!r} > {sec} > {fld}")
        elif fs[0].get("options") and not fs[0].get("dictionaryName") \
                and value not in [o.get("name") for o in fs[0]["options"]]:
            errs.append(f"JSONata: wartość {value!r} nie jest opcją pola {fld}")
    return errs, warns


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
    tab_titles = [t.get("title") for t in tabs]
    for dup in sorted({x for x in tab_titles if tab_titles.count(x) > 1}):
        E.append(f"zduplikowany tytuł strony '{dup}' (JSONata adresuje strony po tytule)")
    ids, techs = set(), set()

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
        if n_fields == 0:
            E.append(f"strona '{t.get('title')}' nie zawiera żadnego pola")

    # stany jsonata - składnia i odwołania do istniejących stron/sekcji/pól/opcji
    for t in tabs:
        for q in t.get("questions", []):
            objs = [q["field"]] + [g["field"] for g in q["field"].get("repeatingSectionGrid", [])]
            for f in objs:
                for st in f.get("states", []) or []:
                    if st.get("type") not in ("SHOW", "READ_ONLY"):
                        E.append(f"{f.get('techName')}: stan {st.get('type')} (dozwolone SHOW/READ_ONLY)")
                    qe, qw = _query_errors(st.get("query", ""), tabs)
                    E += [f"{f.get('techName')}: {e}" for e in qe]
                    W += [f"{f.get('techName')}: {w}" for w in qw]

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
