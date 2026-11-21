"""Podgląd struktury FormDefinition / ogólnego JSON w konsoli:  python -m form2json.show plik.form.json"""
import json
import sys


def show_form(d: dict) -> None:
    print("TYTUŁ:", d["title"])
    for t in d["tabs"]:
        print(f"[{t['type']}] {t['title']}")
        for q in t["questions"]:
            s = q["field"]
            st = " (warunkowa)" if s.get("states") else ""
            print(f"   {s['type']}: {s['title']}{st}")
            for g in s.get("repeatingSectionGrid", []):
                f = g["field"]
                extra = []
                if f.get("validation"):
                    extra.append(",".join(f["validation"]))
                if f.get("options"):
                    extra.append("opcje=" + "|".join(o["name"] for o in f["options"])[:80])
                if f.get("dictionaryName"):
                    extra.append("słownik=" + f["dictionaryName"])
                if f.get("defaultValueType"):
                    extra.append("dane=" + str(f.get("defaultValue")))
                if f.get("states"):
                    extra.append("SHOW")
                if f["type"] == "CONSENT_SECTION":
                    extra.append(f"{len(f['consentSectionConfig']['consents'])} zgód")
                print(f"      - {f['type']:<15} {f['title'][:60]:<60} {' '.join(extra)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = json.load(open(p, encoding="utf-8"))
        if "tabs" in d:
            show_form(d)
        else:
            for x in d.get("pola") or d.get("simple"):
                print(f" - {x['Opis pola'][:60]:<60} {x['Rodzaj pola']:<9} {x.get('Rola','')}")
