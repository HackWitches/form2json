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


ROLES_OK = {"subject", "applicant_name", "applicant_id", "applicant_address", "correspondence_address",
            "contact", "company", "proxy", "attachment", "statement", "signature", "place_date", "office_use"}


def assign_roles(fields: list[FieldCandidate]) -> list[FieldCandidate]:
    """Nadaje role polom bez roli (np. bez VLM) i poprawia oczywiste pomyłki ról z VLM."""
    seen_addr_sec: set[str] = set()
    for f in fields:
        if f.role == "applicant_address" and f.type == "address" and (f.address_part in (None, "full")):
            if not seen_addr_sec:
                seen_addr_sec.add(norm(f.section))
        lab, sec = norm(f.label), norm(f.section)
        if f.type == "signature" or re.search(r"\bpodpis", lab):
            f.role = "signature"
            continue
        if f.role:   # rola z VLM - sanityzacja oczywistych pomyłek
            if f.role == "statement" and not (f.type == "checkbox" or STATEMENT_RE.search(lab)):
                f.role = ""
            elif f.role == "contact" and f.type not in ("email", "phone"):
                f.role = ""
            elif f.role in ("applicant_name", "applicant_id", "applicant_address") and (
                    SUBJECT_QUALIFIER.search(lab)
                    or (sec and SUBJECT_QUALIFIER.search(sec) and not APPLICANT_SEC_RE.search(sec))
                    or (f.role == "applicant_address" and seen_addr_sec and sec and sec not in seen_addr_sec
                        and not CORR_RE.search(sec) and not APPLICANT_SEC_RE.search(sec))):
                f.role = "subject"     # np. "Miejscowość" w sekcji "Dane nieruchomości"
            elif f.role not in ROLES_OK:
                f.role = ""
            if f.role:
                if f.role == "applicant_address":
                    seen_addr_sec.add(sec)
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
                                                       or re.search(r"adres (zamieszk|zameldow|siedzib)", sec)
                                                       or re.search(r"zamieszk|zameldow|siedzib", lab)):
            f.role = "applicant_address" if not re.search(r"miejsc(e|a|u)\b|lokalizac|polozen", lab) or \
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
        lab = norm(f.label + " " + f.hint)
        if f.depends_on and not OPTIONAL_RE.search(lab):
            f.required = True           # pole warunkowe jest wymagane, gdy jest widoczne
        if f.required is not None:
            continue
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


ADDRESSEE_RE = re.compile(r"(?<![\w])(Burmistrz|Wójt|WÓJT|BURMISTRZ|Prezydent|PREZYDENT|Starosta|STAROSTA|"
                          r"Marszałek|Urząd|URZĄD|Zarząd|Dyrektor|Komendant|Kierownik|Naczelnik|Rada|RADA)"
                          r"(?![\wąćęłńóśźż])(.*)$")
ADDR_LINE = re.compile(r"\d|ul\.|al\.|^\s*$|wnios|deklarac|zarzadz|regulamin|instrukcj")


def _tc(s: str) -> str:
    return " ".join(w if not w.isupper() or len(w) < 3 else w.capitalize() for w in s.split())


def find_addressee(lines: list[str]) -> str:
    """Adresat pisma, np. 'Wójt Gminy Elbląg', 'Urząd Gminy Ulan-Majorat' (zwykle u góry strony)."""
    for i, ln in enumerate(lines[:80]):
        m = ADDRESSEE_RE.search(ln)
        if not m:
            continue
        s = (m.group(1) + m.group(2)).strip(" .:,")
        s = re.split(r"\s{3,}", s)[0]
        if len(s) > 90 or ADDR_LINE.search(norm(s)) or re.search(r"[.…]{3}", s):
            continue
        # nazwa organu bywa w 2 liniach: "Urząd Gminy" / "Ulan-Majorat", "Urząd Miejski Wrocławia" / "Wydział ..."
        nxt = lines[i + 1].strip(" .:,") if i + 1 < len(lines) else ""
        if nxt and len(s) < 45 and len(nxt) < 45 and not ADDR_LINE.search(norm(nxt))                 and not re.search(r"[.…(/]", nxt) and nxt[:1].isupper():
            s = f"{s}, {nxt}" if norm(nxt).startswith("wydzial") else f"{s} {nxt}"
        return _tc(s)
    return ""
