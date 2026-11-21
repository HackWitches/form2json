"""Ogólny JSON pól -> docelowy FormDefinition (tabs / questions / flow).

Zgodność z "Reguły techniczne konfiguracji formularzy" + "Standardy i dobre praktyki":
  * >= 2 strony: MAIN ... MAIN, ostatnia FINISH (tylko INFO_SECTION, bez pól do wprowadzania)
  * każde pole w SECTION / REPEATING_SECTION, sekcje nie są zagnieżdżane
  * każde pole ma obligatoryjność: "REQUIRED" w `validation` (brak = pole opcjonalne)
  * walidatory: EMAIL, PESEL, NIP, REGON, ZIP_CODE, SERIAL_NUMBER_ID_CARD
  * strony połączone `flow.links`, zależności pól jako `states` (jsonata SHOW)
  * sekcje predefiniowane: Urząd/Organ, Podmiot wniosku, Twoje dane (MLPZ, tylko do odczytu),
    Adres (Kraj/TERYT ze słownikami, warianty PL / zagranica), Dane innej osoby, Dane podmiotu,
    Pełnomocnictwo, Opłata skarbowa, Dane do kontaktu, Oświadczenia (CONSENT_SECTION), Zgoda
  * dane inicjalne: defaultValue "MLPZ.name" + defaultValueType "EXTERNAL_SYSTEM"
"""
from __future__ import annotations

import json
import re
import uuid
from collections import OrderedDict
from difflib import SequenceMatcher
from html import escape

from .typing_rules import norm

EMPTY_HTML = "<html><head></head><body></body></html>"
TYPE_MAP = {
    "string": "TEXT_FIELD", "email": "TEXT_FIELD", "phone": "TEXT_FIELD", "address": "TEXT_FIELD",
    "postal_code": "TEXT_FIELD", "pesel": "TEXT_FIELD", "nip": "TEXT_FIELD", "regon": "TEXT_FIELD",
    "iban": "TEXT_FIELD", "date": "DATEPICKER", "number": "NUMBER", "textarea": "TEXTAREA",
    "radio": "RADIO", "checkbox": "CHECKBOX", "select": "SELECT", "file": "FILE_UPLOADER",
}
VALIDATOR = {"email": "EMAIL", "pesel": "PESEL", "nip": "NIP", "regon": "REGON", "postal_code": "ZIP_CODE"}
TOOLTIP = {"postal_code": "Format: 00-000", "pesel": "11 cyfr", "nip": "10 cyfr", "regon": "9 lub 14 cyfr",
           "iban": "26 cyfr numeru rachunku", "phone": "Np. 600 700 800"}
INPUT_TYPES = {"TEXT_FIELD", "TEXTAREA", "NUMBER", "DATEPICKER", "RADIO", "CHECKBOX", "SELECT",
               "FILE_UPLOADER", "CONSENT_SECTION"}
STANDARD_ROLES = {"applicant_name", "applicant_id", "applicant_address", "correspondence_address",
                  "contact", "company", "proxy", "signature", "place_date", "office_use"}

OPLATA_HTML = ("<html><head></head><body><p>Opłata skarbowa za pełnomocnictwo wynosi <strong>17 zł</strong>.</p>"
               "<p>Przelej ją na konto urzędu miasta lub gminy właściwego dla adresu zamieszkania osoby, która "
               "udzieliła pełnomocnictwa. Jeśli adres nie jest w Polsce, opłatę przelej na konto Urzędu Miasta "
               "Stołecznego Warszawy – Centrum Obsługi Podatnika.</p><p>W tytule przelewu wpisz „{title} – opłata "
               "skarbowa za pełnomocnictwo”.</p><p>Dodaj plik z potwierdzeniem przelewu. Znajdziesz go na stronie "
               "lub w aplikacji swojego banku.</p></body></html>")
RODO_HTML = ("<html><head></head><body><p>Wyrażam zgodę na przetwarzanie moich danych osobowych zgodnie z "
             "<a href=\"#klauzula-informacyjna\" target=\"_blank\" rel=\"noopener noreferrer\">klauzulą "
             "informacyjną</a></p></body></html>")


def tech(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")
    return s[:60].strip("_") or "pole"


def html(s: str) -> str:
    """Tekst (z dokumentu lub VLM) w otoczce HTML; znaki <, >, & są escapowane."""
    return f"<html><head></head><body><p>{escape(s, quote=False)}</p></body></html>" if s else ""


def tab_title(s: str) -> str:
    """Tytuł strony trafia do JSONaty w apostrofach, a JSONata nie ma sekwencji \\' - zamieniamy ' na ’."""
    return s.replace("'", "\u2019").replace("\\", "/")


class _Names:
    def __init__(self):
        self.used: set[str] = set()

    def __call__(self, s: str) -> str:
        base = tech(s)
        n, k = base, 2
        while n in self.used:
            n, k = f"{base}_{k}", k + 1
        self.used.add(n)
        return n


LEMMA = {"miejscowosci": "Miejscowość", "ulicy": "Ulica", "obrebie": "Obręb", "gminie": "Gmina",
         "kwocie": "Kwota", "terminie": "Termin", "dniu": "Data", "dnia": "Data", "powiecie": "Powiat",
         "wojewodztwie": "Województwo", "celu": "Cel", "zwiazku z": "Uzasadnienie", "wysokosci": "Wysokość"}
ACRONYMS = {"nip", "pesel", "regon", "krs", "iban", "epuap", "ceidg", "teryt"}     # "NIP" nie staje się "Nip"
PREP_RE = re.compile(r"^(w|we|przy|na|z|ze|do|od|o|po|dla|w zwiazku z)\s+(.+)$")


def clean_title(s: str) -> str:
    m = PREP_RE.match(norm(s).strip())
    if m and m.group(2).strip(" .:") in LEMMA:
        return LEMMA[m.group(2).strip(" .:")]
    s = re.sub(r"^\s*(za[łl][ąa]cznik\s*(nr)?\s*\d+\s*[:.)\-–]\s*|w za[łl][ąa]czeniu:?\s*(\d+\)\s*)?|\d+\)\s*)",
               "", s, flags=re.I)
    s = re.sub(r"^\s*((\d{1,2}(\.\d{1,2})*\.?)|([IVX]{1,4}\.)|([A-H](\.\d)*\.?))\s+", "", s)
    s = re.sub(r"\s*\((?:postaw|zaznacz|należy|wypełnić|dot\.|opcjonaln)[^)]*\)?", "", s, flags=re.I)
    s = re.sub(r"[\s.…_:*]+$", "", s.strip())
    s = s.capitalize() if s.isupper() and norm(s) not in ACRONYMS else s[:1].upper() + s[1:]
    return s.strip()[:255]


# ====================================================================== builder
class FormBuilder:
    def __init__(self, title: str):
        self.title = title
        self.names = _Names()
        self.tabs: list[dict] = []

    # ------------------------------------------------------------ elementy
    def field(self, ftype: str, title: str, tech_name: str | None = None, *, required: bool = True,
              validators: list[str] | None = None, options: list[str] | None = None, description: str = "",
              read_only: bool = False, default=None, default_external: bool = False, dictionary: str = "",
              states: list[dict] | None = None, tooltip: str = "", **extra) -> dict:
        val = (["REQUIRED"] if required and ftype not in ("TEXT_BLOCK", "INFO_SECTION") else []) + \
              [v for v in (validators or []) if v]
        f = {
            "id": str(uuid.uuid4()), "type": ftype, "techName": self.names(tech_name or title), "title": title,
            "description": description,
            "options": [{"id": str(uuid.uuid4()), "name": o, "techName": tech(o)} for o in (options or [])],
            "validation": list(dict.fromkeys(val)), "readOnly": read_only, "repeatingSectionGrid": [],
            "states": states or [], "questionOrder": 0, "tooltipMessage": tooltip,
        }
        if dictionary:
            f["dictionaryName"] = dictionary
        if default is not None:
            f["defaultValue"] = default
        if default_external:
            f["defaultValueType"] = "EXTERNAL_SYSTEM"
        f.update(extra)
        return f

    def section(self, title: str, fields: list[dict], *, tech_name: str | None = None, description: str = "",
                states: list[dict] | None = None, repeating: bool = False) -> dict:
        for i, f in enumerate(fields):
            f["questionOrder"] = i
        return {
            "id": str(uuid.uuid4()), "type": "REPEATING_SECTION" if repeating else "SECTION",
            "techName": self.names(tech_name or title), "title": title, "description": description or EMPTY_HTML,
            "options": [], "validation": [], "readOnly": False,
            "repeatingSectionGrid": [{"x": 0, "y": i, "rows": 1, "cols": 3, "field": f} for i, f in enumerate(fields)],
            "states": states or [], "questionOrder": 0, "tooltipMessage": "",
        }

    def tab(self, title: str, sections: list[dict], ttype: str = "MAIN") -> dict:
        sections = [s for s in sections if s and s["repeatingSectionGrid"]]
        questions, y = [], 0
        for i, s in enumerate(sections):
            s["questionOrder"] = i
            rows = len(s["repeatingSectionGrid"]) + 1
            questions.append({"x": 0, "y": y, "rows": rows, "cols": 3, "field": s})
            y += rows
        t = {"id": str(uuid.uuid4()), "title": tab_title(title), "type": ttype, "questions": questions}
        if questions:
            self.tabs.append(t)
        return t

    # ------------------------------------------------------------ jsonata
    @staticmethod
    def q_answer(tab: str, sec: str, fld: str, value: str, negate: bool = False) -> str:
        base = (f"{json.dumps(value, ensure_ascii=False)} in tabs['{tab_title(tab)}' in title]"
                f".questions['{sec}' in field.techName]"
                f".field.repeatingSectionGrid['{fld}' in field.techName].field.__userAnswer")
        return f"$not({base})" if negate else f"({base})"

    @staticmethod
    def show(*queries: str, op: str = "and") -> list[dict]:
        return [{"type": "SHOW", "query": f" {op} ".join(queries)}]

    def build(self) -> dict:
        links = [{"id": str(uuid.uuid4()), "source": a["id"], "target": b["id"]}
                 for a, b in zip(self.tabs, self.tabs[1:])]
        return {"title": self.title, "tabs": self.tabs, "flow": {"links": links}}


# ====================================================================== sekcje standardowe
def address_section(b: FormBuilder, tab: str, title: str, suffix: str, *, mlpz: bool,
                    states: list[dict] | None = None) -> dict:
    sec_tech = b.names(tech(title))
    b.names.used.discard(sec_tech)          # section() zarejestruje ją ponownie
    sfx = f"_{suffix}" if suffix else ""
    kraj = f"kraj{sfx}"
    pl = b.show(b.q_answer(tab, sec_tech, kraj, "Polska"))
    zagr = b.show(b.q_answer(tab, sec_tech, kraj, "Polska", negate=True))
    m = (lambda a: dict(default=f"MLPZ.{a}", default_external=True)) if mlpz else (lambda a: {})
    fields = [
        b.field("SELECT", "Kraj", kraj, dictionary="Kraje", default=["Polska"], searchEnabled=True,
                multiLevelTree=False),
        b.field("TEXT_FIELD", "Miejscowość", f"miejscowosc{sfx}_zagranica", states=zagr),
        b.field("TEXT_FIELD", "Kod pocztowy", f"kod_pocztowy{sfx}_zagranica", states=zagr),
        b.field("TEXT_FIELD", "Ulica", f"ulica{sfx}_zagranica", states=zagr),
        b.field("SELECT", "Województwo, powiat, gmina, miejscowość", f"wojewodztwo{sfx}",
                dictionary="TERYT_4G_DOWODOWY", default=[], searchEnabled=True, multiLevelTree=True, states=pl),
        b.field("TEXT_FIELD", "Kod pocztowy", f"kod_pocztowy{sfx}", validators=["ZIP_CODE"], states=pl,
                tooltip="Format: 00-000", **m("postCode")),
        b.field("TEXT_FIELD", "Ulica", f"ulica{sfx}", states=pl, **m("street")),
        b.field("TEXT_FIELD", "Numer budynku", f"numer_budynku{sfx}", **m("building")),
        b.field("TEXT_FIELD", "Numer lokalu", f"numer_lokalu{sfx}", required=False, **m("apartment")),
    ]
    return b.section(title, fields, tech_name=sec_tech, states=states)


def subject_field(b: FormBuilder, f: dict) -> dict:
    t = f.get("type", "string")
    opts = f.get("options") or []
    if t == "radio" and len(opts) > 5:
        t = "select"
    lab = clean_title(f["label"]) or "Pole"
    if t in ("radio", "checkbox", "select") and not opts:
        if t == "select":
            t = "string"
        else:                       # pojedynczy checkbox "tak/nie" -> CHECKBOX z jedną opcją
            t, opts = "checkbox", [lab]
    ftype = TYPE_MAP.get(t, "TEXT_FIELD")
    desc = html(f["hint"]) if f.get("hint") and t != "date" else ""
    extra = {"searchEnabled": len(opts) > 15} if ftype == "SELECT" else {}
    if ftype == "SELECT":
        extra["defaultValue"] = []
    out = b.field(ftype, lab, required=bool(f.get("required", True)), validators=[VALIDATOR.get(t, "")],
                  options=opts if ftype in ("RADIO", "CHECKBOX", "SELECT") else None, description=desc,
                  tooltip=TOOLTIP.get(t, ""), **extra)
    if f.get("prefilled"):
        out["defaultValue"] = f["prefilled"]
    nl = norm(lab)
    if ftype == "TEXT_FIELD":     # walidatory z treści etykiety (najwyżej jeden walidator formatu)
        for rx, v in ((r"epuap", "EPUAP"), (r"e-?doreczen", "ADE"), (r"dowod(u)? osobist|seria i numer dowodu",
                                                                       "SERIAL_NUMBER_ID_CARD"),
                      (r"\bpesel\b", "PESEL"), (r"\bnip\b", "NIP"), (r"\bregon\b", "REGON"),
                      (r"e-?mail", "EMAIL"), (r"kod pocztowy", "ZIP_CODE")):
            if re.search(rx, nl) and not set(out["validation"]) - {"REQUIRED"}:
                out["validation"].append(v)
    return out


def collapse_repeats(fs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Ten sam układ pól powtórzony w tabeli (np. długość/szerokość/razem x5) -> (pola przed, wzorzec wiersza, pola po)."""
    labs = [norm(f["label"]) for f in fs]
    n = len(labs)
    for k in range(1, n // 2 + 1):
        for start in range(0, n - 2 * k + 1):
            pat = labs[start:start + k]
            r = 1
            while labs[start + r * k:start + (r + 1) * k] == pat:
                r += 1
            if r >= 2 and r * k >= 4 and set(pat) != {"pole"}:    # "Pole" = pole bez rozpoznanej etykiety
                return fs[:start], fs[start:start + k], fs[start + r * k:]
    return fs, [], []


def _dedupe(fs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for f in fs:
        key = (norm(f["label"]), f.get("type"), tuple(f.get("options") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# ====================================================================== główna funkcja
def to_form_definition(generic: dict, tabs_mode: str = "auto") -> dict:
    fields = generic["fields"]
    title = clean_title(generic.get("title") or "Wniosek")
    b = FormBuilder(title)
    ent = generic.get("entities", {"company": False, "proxy": True})
    roles = {f.get("role", "subject") for f in fields}

    # ------------------------------------------------ strona 1: Dane
    T1 = "Dane"
    addressee = generic.get("addressee") or ""
    organ = bool(re.match(r"(burmistrz|wojt|prezydent|starosta|marszalek|dyrektor|rada|komendant|kierownik)",
                          norm(addressee)))
    s_urzad = b.section(
        "Organ, do którego jest składany wniosek" if organ else "Urząd, do którego jest składany wniosek",
        [b.field("TEXT_FIELD", "Nazwa organu" if organ else "Nazwa urzędu", read_only=True,
                 default=addressee or "Urząd")])

    opts = ["Swoim", "Innej osoby"] + (["Osoby prawnej lub jednostki organizacyjnej"] if ent["company"] else [])
    S_POD, F_KTO, F_JAKO = "podmiot_wniosku", "wybierz_w_czyim_imieniu_wnioskujesz", "wybierz_jako_kto_wnioskujesz"
    OPR = "Osoby prawnej lub jednostki organizacyjnej"
    q = lambda v: b.q_answer(T1, S_POD, F_KTO, v)
    pod_fields = [b.field("RADIO", "Wybierz, w czyim imieniu wnioskujesz", F_KTO, options=opts)]
    if ent["company"]:
        pod_fields.append(b.field("RADIO", "Wybierz, jako kto wnioskujesz", F_JAKO,
                                  options=["Pełnomocnik", "Reprezentant"], states=b.show(q(OPR))))
    s_pod = b.section("Podmiot wniosku", pod_fields, tech_name=S_POD)

    extra_me = []
    for f in fields:          # dane wnioskodawcy spoza standardu (np. dowód osobisty, data urodzenia)
        if f.get("role") in ("applicant_name", "applicant_id"):
            lab = norm(f["label"])
            if re.search(r"\bimie\b|nazwisk|pesel|^wnioskodawca$|nazwa|\bnip\b|\bregon\b|\bkrs\b", lab):
                continue
            ff = subject_field(b, f)
            if re.search(r"dowod|seria", lab) and "SERIAL_NUMBER_ID_CARD" not in ff["validation"]:
                ff["validation"].append("SERIAL_NUMBER_ID_CARD")
            extra_me.append(ff)
    s_me = b.section("Twoje dane", [
        b.field("TEXT_FIELD", "Imię", "imie", read_only=True, default="MLPZ.name", default_external=True),
        b.field("TEXT_FIELD", "Nazwisko", "nazwisko", read_only=True, default="MLPZ.surname", default_external=True),
        b.field("TEXT_FIELD", "Numer PESEL", "numer_pesel", validators=["PESEL"], read_only=True,
                default="MLPZ.pesel", default_external=True),
        *_dedupe_fields(extra_me)])
    sections1 = [s_urzad, s_pod, s_me,
                 address_section(b, T1, "Adres zamieszkania", "", mlpz=True)]
    if "correspondence_address" in roles:
        sec = b.section("Adres do korespondencji", [
            b.field("RADIO", "Czy adres do korespondencji jest inny niż adres zamieszkania?",
                    "inny_adres_do_korespondencji", options=["Tak", "Nie"])])
        sections1.append(sec)
        sections1.append(address_section(
            b, T1, "Adres korespondencyjny", "korespondencja", mlpz=False,
            states=b.show(b.q_answer(T1, sec["techName"], "inny_adres_do_korespondencji", "Tak"))))
    if ent["company"]:
        st = b.show(q(OPR))
        sections1.append(b.section("Dane podmiotu", [
            b.field("TEXT_FIELD", "Nazwa"), b.field("TEXT_FIELD", "NIP", validators=["NIP"], tooltip="10 cyfr"),
            b.field("TEXT_FIELD", "REGON", validators=["REGON"], tooltip="9 lub 14 cyfr"),
            b.field("TEXT_FIELD", "Numer KRS", required=False, tooltip="Wpisz 10 cyfr")], states=st))
        sections1.append(address_section(b, T1, "Adres siedziby", "siedziba", mlpz=False, states=st))
        sections1.append(b.section(
            "Potwierdzenie, że jesteś reprezentantem",
            [b.field("FILE_UPLOADER", "Plik z odpisem danych")],
            description=html("Dodaj dokument potwierdzający, że możesz reprezentować podmiot."),
            states=b.show(q(OPR), b.q_answer(T1, S_POD, F_JAKO, "Reprezentant"))))
    st_inna = b.show(q("Innej osoby"))
    sections1.append(b.section("Dane innej osoby", [
        b.field("TEXT_FIELD", "Imię", "imie_inna_osoba"), b.field("TEXT_FIELD", "Nazwisko", "nazwisko_inna_osoba"),
        b.field("TEXT_FIELD", "Numer PESEL", "numer_pesel_inna_osoba", validators=["PESEL"], tooltip="11 cyfr")],
        states=st_inna))
    sections1.append(address_section(b, T1, "Adres zamieszkania innej osoby", "inna_osoba", mlpz=False,
                                     states=st_inna))
    pel_q = q("Innej osoby") + (f" or ({q(OPR)} and {b.q_answer(T1, S_POD, F_JAKO, 'Pełnomocnik')})"
                                if ent["company"] else "")
    sections1.append(b.section("Pełnomocnictwo", [b.field("FILE_UPLOADER", "Plik dokumentu z pełnomocnictwem")],
                               description=html("Dodaj dokument potwierdzający, że jesteś pełnomocnikiem."),
                               states=[{"type": "SHOW", "query": pel_q}]))
    sec_op_tech = "oplata_skarbowa_pelnomocnictwo"
    op_q = b.show(b.q_answer(T1, sec_op_tech, "czy_zlozenie_pelnomocnictwa_wymaga_oplaty_skarbowej", "Tak"))
    sections1.append(b.section("Opłata skarbowa", [
        b.field("RADIO", "Czy złożenie pełnomocnictwa wymaga opłaty skarbowej?", options=["Tak", "Nie"]),
        b.field("TEXT_BLOCK", "", "oplata_info", required=False, description=OPLATA_HTML.format(title=escape(title)),
                states=op_q),
        b.field("FILE_UPLOADER", "Plik z potwierdzeniem przelewu", states=op_q)],
        tech_name=sec_op_tech, states=st_inna,
        description=html("Nie musisz płacić za złożenie pełnomocnictwa wystawionego przez najbliższą rodzinę: "
                         "małżonka, dzieci, rodziców, dziadków, wnuków, rodzeństwo.")))
    if ent["company"]:      # Opłata skarbowa (wersja 2): pełnomocnik osoby prawnej
        sections1.append(b.section("Opłata skarbowa za pełnomocnictwo", [
            b.field("TEXT_BLOCK", "", "oplata_info_podmiot", required=False,
                    description=OPLATA_HTML.format(title=escape(title)).replace("zamieszkania osoby, która udzieliła",
                                                                        "siedziby podmiotu, który udzielił")),
            b.field("FILE_UPLOADER", "Plik z potwierdzeniem przelewu", "plik_z_potwierdzeniem_przelewu_podmiot")],
            states=b.show(q(OPR), b.q_answer(T1, S_POD, F_JAKO, "Pełnomocnik"))))
    b.tab(T1, sections1)

    # ------------------------------------------------ strony przedmiotu wniosku
    subj = [f for f in fields if f.get("role", "subject") == "subject"]
    atts = [f for f in fields if f.get("role") == "attachment"
            and not re.fullmatch(r"(zalacznik|zalaczniki|inne|w zalaczeniu|tak|nie)\W*(nr\s*)?\d*\W*",
                                 norm(f["label"]).strip())]
    if len(atts) < len([f for f in fields if f.get("role") == "attachment"]):
        atts.append({"label": "Inne załączniki", "type": "file", "required": False})
    groups: "OrderedDict[str, OrderedDict[str, list[dict]]]" = OrderedDict()
    for f in subj:
        path = [clean_title(p) for p in (f.get("section") or "").split(" > ") if clean_title(p)]
        top = path[0] if path else ""
        leaf = path[-1] if path else ""
        groups.setdefault(top, OrderedDict()).setdefault(leaf, []).append(f)
    # osobne strony tylko dla dużych, wyraźnie wydzielonych części formularza (np. I., II., III.)
    sizes = [sum(len(v) for v in secs.values()) for secs in groups.values()]
    multi_tab = tabs_mode != "single" and 2 <= len(groups) <= 5 and min(sizes) >= 4 and sum(sizes) >= 12
    # kilka wniosków w jednym formularzu ("I. Wniosek o..., II. Wniosek o...") -> wybór rodzaju wniosku
    # i sekcje warunkowe (jsonata SHOW) zamiast osobnych, zawsze widocznych stron
    variants = [t for t in groups if re.match(r"(wniosek|deklaracja|zgloszenie|wyroznienie|nagroda)", norm(t))]
    variant_mode = multi_tab and len(variants) >= 2
    subj_tabs: list[tuple[str, list[tuple]]] = []
    if variant_mode:
        subj_tabs.append(("Informacje o wniosku", [((leaf if leaf != top else "Szczegóły wniosku"), fs,
                                                    top if top in variants else "")
                                                   for top, secs in groups.items() for leaf, fs in secs.items()]))
        multi_tab = False
    elif multi_tab:
        for top, secs in groups.items():
            subj_tabs.append((top or "Informacje o wniosku", [(k, v, "") for k, v in secs.items()]))
    else:
        subj_tabs.append(("Informacje o wniosku", [(leaf or top, fs, "") for top, secs in groups.items()
                                                    for leaf, fs in secs.items()]))
    built_subject = []
    used_titles = {T1, "Kontakt i zgody", "Strona końcowa"}
    for ti, (raw_title, secs) in enumerate(subj_tabs):
        tab_t = base_t = tab_title(raw_title[:80])
        k = 2
        while tab_t in used_titles:          # JSONata adresuje strony po tytule - tytuły muszą być unikalne
            tab_t, k = f"{base_t} ({k})", k + 1
        used_titles.add(tab_t)
        sections = []
        merged: "OrderedDict[tuple, list[dict]]" = OrderedDict()
        for sec_title, fs, variant in secs:
            if not sec_title or re.search(r"wnioskodaw|skladajac|twoje dane", norm(sec_title)):
                sec_title = "Szczegóły wniosku"        # pola przedmiotu wniosku w sekcji "Dane wnioskodawcy"
            merged.setdefault((variant, sec_title), []).extend(fs)
        if variant_mode:
            vs = list(dict.fromkeys(v for v, _ in merged if v))
            vsec = b.section("Rodzaj wniosku", [b.field("RADIO", "Wybierz rodzaj wniosku", "wybierz_rodzaj_wniosku",
                                                        options=vs)], tech_name="rodzaj_wniosku")
            sections.append((vsec, []))
        for (variant, sec_title), fs in merged.items():
            title_s = sec_title if not variant else (variant if sec_title == "Szczegóły wniosku"
                                                     else f"{sec_title} ({variant})")
            st = b.show(b.q_answer(tab_t, "rodzaj_wniosku", "wybierz_rodzaj_wniosku", variant)) if variant else None
            # najpierw wykryj powtórzenia, potem usuń duplikaty (odwrotnie _dedupe zjada powtórzone wiersze)
            for part, rep in zip(collapse_repeats(fs), (False, True, False)):
                part = _dedupe(part)
                if part:
                    sec = b.section(title_s[:120], [subject_field(b, f) for f in part], repeating=rep, states=st)
                    sections.append((sec, part))
        # zależności (SHOW) w obrębie strony
        by_label = {}
        for sec, fs in sections:
            for g, f in zip(sec["repeatingSectionGrid"], fs):
                by_label[norm(f["label"])] = (sec, g["field"])
        for sec, fs in sections:
            if sec["type"] == "REPEATING_SECTION":
                continue
            for g, f in zip(sec["repeatingSectionGrid"], fs):
                dep = f.get("depends_on")
                if dep and norm(dep) in by_label:
                    psec, pf = by_label[norm(dep)]
                    if psec["type"] == "SECTION" and pf is not g["field"]:
                        value = _option(pf, f.get("depends_value") or "Tak")
                        if value is None:    # warunek nigdy by nie zaszedł - pole zostaje widoczne, ale nie wymagane
                            g["field"]["validation"] = [v for v in g["field"]["validation"] if v != "REQUIRED"]
                            continue
                        g["field"]["states"] = b.show(b.q_answer(tab_t, psec["techName"], pf["techName"], value))
        if ti == len(subj_tabs) - 1 and atts:
            sections.append((b.section("Załączniki", [
                b.field("FILE_UPLOADER", clean_title(a["label"]) or "Załącznik",
                        required=bool(a.get("required", False)))
                for a in _dedupe(atts)], description=html("Dodaj wymagane dokumenty.")), []))
        built_subject.append(b.tab(tab_t, [s for s, _ in sections]))
    if not any(t["questions"] for t in built_subject):
        b.tab("Informacje o wniosku", [b.section("Szczegóły wniosku", [
            b.field("TEXTAREA", "Treść wniosku")])])

    # ------------------------------------------------ kontakt i zgody
    statements = _dedupe([f for f in fields if f.get("role") == "statement"])
    kontakt = [b.section("Dane do kontaktu", [
        b.field("TEXT_FIELD", "Numer telefonu", "numer_telefonu", required=False, default="MLPZ.phoneNumber",
                default_external=True),
        b.field("TEXT_FIELD", "Adres e-mail", "adres_e_mail", required=False, validators=["EMAIL"],
                default="MLPZ.email", default_external=True)],
        description=html("Wpisanie danych do kontaktu może przyspieszyć załatwienie sprawy. Urzędnik będzie mógł "
                         "skontaktować się z Tobą w przypadku pytań lub wątpliwości. Jeśli widzisz uzupełnione "
                         "dane, sprawdź, czy są poprawne. Możesz je zmienić lub usunąć."))]
    if statements:
        kontakt.append(b.section("Oświadczenie" if len(statements) == 1 else "Oświadczenia", [b.field(
            "CONSENT_SECTION", "", "oswiadczenia", required=False, selectAllEnabled=len(statements) > 1,
            consentSectionConfig={"consents": [{"description": html(clean_title(s["label"])),
                                                "required": bool(s.get("required", True))} for s in statements],
                                  "required": True, "files": []})]))
    kontakt.append(b.section("Zgoda", [b.field(
        "CONSENT_SECTION", "", "zgoda", required=False,
        consentSectionConfig={"consents": [{"description": RODO_HTML, "required": True}],
                              "required": True, "files": []})], tech_name="zgoda_sekcja"))
    b.tab("Kontakt i zgody", kontakt)

    # ------------------------------------------------ strona końcowa (FINISH): tylko INFO_SECTION
    b.tab("Strona końcowa", [b.section("Podgląd uzupełnionego wniosku", [b.field(
        "INFO_SECTION", "", "podglad", required=False,
        description=html("Zanim przejdziesz do wyboru sposobu podpisu, sprawdź dane na podglądzie. "
                         "Jeśli zobaczysz błędy, wróć do edycji."))])], ttype="FINISH")
    return b.build()


def _option(pf: dict, want: str) -> str | None:
    """Nazwa opcji pola sterującego odpowiadająca wartości z VLM/reguł (JSONata porównuje tekst dokładnie)."""
    names = [o["name"] for o in pf.get("options", [])]
    exact = [n for n in names if norm(n).strip(" .") == norm(want).strip(" .")]
    if exact:
        return exact[0]
    best = max(names, key=lambda n: SequenceMatcher(None, norm(n), norm(want)).ratio(), default=None)
    return best if best and SequenceMatcher(None, norm(best), norm(want)).ratio() >= 0.8 else None


def _dedupe_fields(fs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for f in fs:
        if norm(f["title"]) not in seen:
            seen.add(norm(f["title"]))
            out.append(f)
    return out
