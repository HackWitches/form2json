"""Semantyka pól: rola pola w formularzu, obligatoryjność, zależności (warunki SHOW).

Rola decyduje, czy pole z formularza papierowego trafia do predefiniowanej sekcji
standardu (Twoje dane, Adres zamieszkania, Dane do kontaktu, Zgody...), czy do części
"przedmiot wniosku". Reguły są deterministyczne; jeśli VLM nadał rolę - reguły jej nie nadpisują
(poza oczywistymi przypadkami, np. podpis).
"""
from __future__ import annotations

import re

from .model import FieldCandidate
from .typing_rules import norm

OFFICE_RE = re.compile(r"wypelnia (urzad|organ|pracownik)|nr sprawy|znak sprawy|pieczec|adnotacj|data wplywu|"
                       r"potwierdzenie (przyjecia|wplywu)|data i podpis (pracownika|przyjmujacego)|"
                       r"\bdecyzj[ai] nr\b|miejsce na|wypelnia (organ|urzad)|kancelari|pracownik")
APPLICANT_SEC_RE = re.compile(r"wnioskodaw|skladajac|dane identyfikacyjne|dane osob|zglaszajac|podatnik|"
                              r"dane (wlasciciela|inwestora)|wlasciciel|deklarujac|zobowiazan|dane wnioskodawcy|"
                              r"twoje dane|dane podmiotu|dane skladajacego")
CORR_RE = re.compile(r"korespondenc|do doreczen")
SUBJECT_QUALIFIER = re.compile(r"kandydat|psa\b|psow|dziecka|ucznia|zawodnik|sportowc|kierownik|wykonawc|"
                               r"inwestycj|robot|nieruchomosc|dzialk|budynk|lokalu uzytk|pojazd|zmarl|"
                               r"opiekun|rodzic|swiadk|uczestnik|najemc|dzierzawc|trenera|klubu|pelnomocnik")
STATEMENT_RE = re.compile(r"^(\W*)(oswiadczam|wyrazam zgode|zapoznalem|zapoznalam|zapoznalem\(am\)|zobowiazuje|"
                          r"jestem swiadom|przyjmuje do wiadomosci|potwierdzam|akceptuje|zgadzam sie)")
PLACE_DATE_RE = re.compile(r"^(.{0,40},\s*)?(dnia|data)$|miejscowosc,? (i )?data|data (wypelnienia|zlozenia|sporzadzenia)|"
                           r"^data i miejsce|^miejsce i data")
OPTIONAL_RE = re.compile(r"opcjonaln|nieobowiazk|jesli (dotyczy|posiada)|jezeli (dotyczy|posiada)|o ile posiada|"
                         r"\bfakultatywn|w przypadku posiadania")
CONDITION_RE = re.compile(r"^(\W*)(jezeli|jesli|w przypadku|gdy |inne|inny|inna|jakie|jaki|jaka|podac jak|"
                          r"prosze podac|wskaz)")
NAME_RE = re.compile(r"\bimie\b|\bimiona\b|nazwisko|nazwa (wnioskodawcy|pelna|firmy|podmiotu)|^wnioskodawca$|"
                     r"imie i nazwisko|nazwisko i imie")
COMPANY_RE = re.compile(r"\bnip\b|\bregon\b|\bkrs\b|nazwa (firmy|podmiotu|pelna|instytucji|organizacji)|"
                        r"/\s*nazwa|nazwa\s*/|siedzib|osob(y|a) prawn|przedsiebiorc|jednostk(i|a) organizac")
ID_RE = re.compile(r"\bpesel\b|\bnip\b|\bregon\b|\bkrs\b|dowod|seria i numer|nr dokumentu|paszport")


def assign_roles(fields: list[FieldCandidate]) -> list[FieldCandidate]:
    """Nadaje role polom bez roli (np. bez VLM)."""
    for f in fields:
        lab, sec = norm(f.label), norm(f.section)
        if f.type == "signature" or re.search(r"\bpodpis", lab):
            f.role = "signature"
            continue
        if f.role:
            continue
        if OFFICE_RE.search(lab) or OFFICE_RE.search(sec):
            f.role = "office_use"
        elif f.type == "file" or "zalacznik" in sec:
            f.role = "attachment"
        elif f.type == "date" and PLACE_DATE_RE.search(lab.strip(" .:")):
            f.role = "place_date"
        elif f.type == "checkbox" and not f.options and STATEMENT_RE.search(lab):
            f.role = "statement"
        elif SUBJECT_QUALIFIER.search(lab) or (SUBJECT_QUALIFIER.search(sec) and not APPLICANT_SEC_RE.search(sec)):
            f.role = "subject"
        elif f.type in ("email", "phone"):
            f.role = "contact"
        elif f.type in ("address", "postal_code") and (CORR_RE.search(lab) or CORR_RE.search(sec)):
            f.role = "correspondence_address"
        elif f.type in ("address", "postal_code") and (APPLICANT_SEC_RE.search(sec) or not sec
                                                       or re.search(r"zamieszk|zameldow|siedzib", lab)):
            f.role = "applicant_address" if not re.search(r"miejsc|lokalizac|polozen", lab) or \
                re.search(r"zamieszk|zameldow|siedzib", lab) else "subject"
        elif NAME_RE.search(lab) and (APPLICANT_SEC_RE.search(sec) or not sec):
            f.role = "applicant_name"
        elif ID_RE.search(lab) and (APPLICANT_SEC_RE.search(sec) or not sec):
            f.role = "applicant_id"
        else:
            f.role = "subject"
    return fields


def assign_required(fields: list[FieldCandidate]) -> None:
    for f in fields:
        if f.required is not None:
            continue
        lab = norm(f.label + " " + f.hint)
        f.required = not (OPTIONAL_RE.search(lab) or f.role in ("contact",) or f.depends_on
                          or CONDITION_RE.search(norm(f.label)))


def assign_dependencies(fields: list[FieldCandidate]) -> None:
    """Pola warunkowe bez VLM: 'jeżeli tak, ...' po polu Tak/Nie, pole o nazwie opcji po radio."""
    prev: FieldCandidate | None = None
    for f in fields:
        if f.depends_on:
            prev = f if f.options else prev
            continue
        lab = norm(f.label)
        if prev is not None and prev.options and f.role == prev.role and f.section == prev.section:
            opts = {norm(o).strip(" .:…"): o for o in prev.options}
            if lab.strip(" .:…") in opts:
                f.depends_on, f.depends_value = prev.label, opts[lab.strip(" .:…")]
                f.label = f"{f.label} - podaj szczegóły" if len(f.label) < 25 else f.label
            elif CONDITION_RE.search(lab):
                yes = next((o for o in prev.options if norm(o).strip() in ("tak", "tak.")), None)
                other = next((o for o in prev.options if re.match(r"(inne|inny|inna)\b", norm(o))), None)
                pick = yes if re.search(r"jezeli|jesli|w przypadku|gdy", lab) and yes else (other or yes)
                if pick:
                    f.depends_on, f.depends_value = prev.label, pick
        if f.options and f.type in ("radio", "checkbox", "select"):
            prev = f
    return None


def detect_entities(fields: list[FieldCandidate], text: str) -> dict:
    """Kto może złożyć wniosek (wpływa na sekcję 'Podmiot wniosku')."""
    t = norm(text)
    labels = " ".join(norm(f.label) for f in fields if f.role != "subject")
    company = bool(COMPANY_RE.search(labels)) or bool(re.search(r"nazwa (firmy|podmiotu)|osob(y|a) prawn|"
                                                                  r"\bkrs\b|siedziby", t))
    proxy = bool(re.search(r"pelnomocni", t))
    return {"company": company, "proxy": proxy}


ADDRESSEE_RE = re.compile(r"^\s*(do\s+)?(burmistrz|wojt|prezydent|starost|marszalek|urzad|zarzad|dyrektor|"
                          r"rada|komendant|kierownik|naczelnik|przewodnicz)", re.I)


def find_addressee(lines: list[str]) -> str:
    """Adresat pisma, np. 'Burmistrz Miasta i Gminy Młynary' (zwykle w prawym górnym rogu)."""
    best = ""
    for i, ln in enumerate(lines[:60]):
        s = ln.strip(" .:")
        if ADDRESSEE_RE.match(norm(s)) and 6 < len(s) < 120 and not re.search(r"wnios|deklarac", norm(s)):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            # nazwa organu bywa w 2 liniach: "Burmistrz" / "Miasta i Gminy X"
            if nxt and re.match(r"^(miasta|gminy|powiatu|wojewodztwa|m\.|w |we )", norm(nxt)) and len(s) < 40:
                s = f"{s} {nxt.strip(' .:')}"
            s = re.sub(r"^do\s+", "", s, flags=re.I)
            return s[0].upper() + s[1:]
    return best
