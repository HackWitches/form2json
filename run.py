"""Uruchomienie form2json jako zwykłego skryptu (bez `python -m`), np. przyciskiem RUN w PyCharm/IntelliJ:

    python C:\\Hackathon2026\\Formularze-Wypakowane\\form2json\\run.py [parametry]

Bez parametrów wszystkie ustawienia pochodzą z runtime.config (obok tego pliku).
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    pkg_dir = Path(__file__).resolve().parent
    # katalog pakietu zamieniamy na katalog nadrzędny: `import form2json` działa niezależnie od
    # katalogu roboczego, a moduły pakietu (model.py, ocr.py...) nie zasłaniają bibliotek
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != pkg_dir]
    sys.path.insert(0, str(pkg_dir.parent))
    from form2json.cli import main
    sys.exit(main())
