"""Punkt wejścia form2json.

Uruchomienie (ustawienia z runtime.config, parametry z linii poleceń mają pierwszeństwo):
    python form2json\\run.py                       # z katalogu Formularze-Wypakowane, bez parametrów
    python -m form2json                            # to samo jako moduł
    python -m form2json plik.pdf katalog\\ -o out --fast --no-report

Dla każdego pliku zapisuje:
  <nazwa>.<ext>.simple.json  - ogólny JSON (opis pola -> rodzaj pola)
  <nazwa>.<ext>.fields.json  - wynik pośredni (pola, bbox, sekcje, surowa odpowiedź VLM)
  <nazwa>.<ext>.form.json    - docelowy FormDefinition
  <nazwa>.<ext>.pN.debug.png - (debug = true) podgląd wykrytych pól na obrazie

Kolejne kroki main(): parse_args -> resolve_settings -> setup_ocr -> setup_vlm -> collect_inputs
-> process_files (process_file dla każdego pliku) -> print_summary -> make_report.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .build import to_form_definition
from .loaders import IMAGE_EXT, OFFICE_EXT
from .pipeline import extract
from .runtime import Runtime, load_runtime

SUPPORTED = IMAGE_EXT | OFFICE_EXT | {".pdf"}
_LOCK = threading.Lock()


# ---------------------------------------------------------------- ustawienia

@dataclass
class Settings:
    inputs: list[str]
    out: Path
    ocr: str
    tabs: str
    dpi: int
    debug: bool
    rebuild: bool
    use_vlm: bool
    force_vlm: bool
    model: str | None
    fallback_model: str | None
    vlm_config: str | None
    vlm_parallel: int
    file_workers: int
    report: bool
    ref_dir: str
    timing: bool
    runtime_source: str = ""
    ollama: object = None           # OllamaSettings z runtime.config


@dataclass
class FileResult:
    name: str
    ok: bool
    pages: int = 0
    fields: int = 0
    seconds: float = 0.0


@dataclass
class RunStats:
    started: float = field(default_factory=time.perf_counter)
    results: list[FileResult] = field(default_factory=list)
    finished: int = 0

    def done(self) -> int:
        """Numer kolejny ukończonego pliku (bezpieczne dla wątków)."""
        with _LOCK:
            self.finished += 1
            return self.finished

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="form2json", description="Formularze PDF/DOC/DOCX/ODS/XLSX/JPG/PNG -> JSON")
    ap.add_argument("inputs", nargs="*", help="pliki/katalogi (domyślnie [paths] input z runtime.config)")
    ap.add_argument("-o", "--out", default=None, help="katalog wyników (domyślnie [paths] output)")
    ap.add_argument("--runtime-config", default=None, help="plik runtime.config (domyślnie obok cli.py)")
    ap.add_argument("--ocr", default=None, help="auto|tesseract|paddle|paddle-cpu|none (obrazy i skany)")
    ap.add_argument("--vlm", action="store_true", help="wymuś VLM (błąd, jeśli Ollama nie odpowiada)")
    ap.add_argument("--no-vlm", action="store_true", help="bez VLM - tylko geometria + OCR (CPU)")
    ap.add_argument("--profile", choices=["default", "fast", "accurate"], default=None,
                    help="model z runtime.config: default | fast | accurate")
    ap.add_argument("--fast", action="store_const", const="fast", dest="profile", help="= --profile fast")
    ap.add_argument("--vlm-model", default=None, help="konkretny tag modelu Ollama, np. qwen3-vl:8b-instruct")
    ap.add_argument("--rebuild", action="store_true", help="wejście: *.fields.json - przebuduj tylko *.form.json")
    ap.add_argument("--config", default=None, help="plik konfiguracji VLM (domyślnie config/local.toml)")
    ap.add_argument("--tabs", default=None, choices=["auto", "single"])
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--vlm-parallel", type=int, default=None, help="ile stron naraz do VLM")
    ap.add_argument("--workers", type=int, default=None, help="ile plików przetwarzać naraz")
    ap.add_argument("--ollama", choices=["auto", "always", "never"], default=None,
                    help="własny serwer Ollama z przyspieszeniem (domyślnie [ollama] manage)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--report", action=argparse.BooleanOptionalAction, default=None,
                    help="generuj RAPORT.md (domyślnie [report] enabled)")
    ap.add_argument("--ref", default=None, help="katalog z konfiguracja_form_*.json dla raportu")
    ap.add_argument("--timing", action=argparse.BooleanOptionalAction, default=None,
                    help="czas na plik + podsumowanie (domyślnie [timing] enabled)")
    return ap.parse_args(argv)


def resolve_settings(a: argparse.Namespace, rt: Runtime) -> Settings:
    """Linia poleceń > runtime.config > wartości domyślne."""
    inputs = a.inputs or rt.inputs
    if not inputs:
        raise SystemExit("Brak plików wejściowych: podaj je w linii poleceń albo w runtime.config ([paths] input).")
    use_vlm = not (a.no_vlm or a.rebuild) and (a.vlm or rt.vlm_enabled)
    if a.vlm_model:
        model = a.vlm_model
    elif a.profile:
        model = rt.models[a.profile]
    else:
        model = rt.model if rt.source else None      # bez runtime.config: model z config/local.toml
    report = rt.report if a.report is None else a.report
    return Settings(
        inputs=list(inputs), out=Path(a.out or rt.output),
        ocr="none" if a.rebuild else (a.ocr or rt.ocr), tabs=a.tabs or rt.tabs, dpi=a.dpi or rt.dpi,
        debug=a.debug or rt.debug, rebuild=a.rebuild, use_vlm=use_vlm, force_vlm=a.vlm,
        model=model, fallback_model=rt.models["default"], vlm_config=a.config,
        vlm_parallel=max(1, a.vlm_parallel or rt.vlm_parallel), file_workers=max(1, a.workers or rt.file_workers),
        report=bool(report),
        ref_dir=a.ref or rt.ref_dir or _first_dir(inputs),
        timing=rt.timing if a.timing is None else a.timing, runtime_source=rt.source,
        ollama=_ollama_settings(a, rt))


def _ollama_settings(a: argparse.Namespace, rt: Runtime):
    o = rt.ollama
    if a.ollama:
        o.manage = a.ollama
    return o


def _first_dir(inputs: list[str]) -> str:
    p = Path(inputs[0])
    return str(p if p.is_dir() else p.parent)


def print_settings(s: Settings) -> None:
    print(f"Konfiguracja: {s.runtime_source or 'domyślna (brak runtime.config)'}")
    print(f"  wejście: {'; '.join(s.inputs)}")
    print(f"  wyniki:  {s.out}")
    print(f"  raport:  {'tak' if s.report else 'nie'}   pomiar czasu: {'tak' if s.timing else 'nie'}   "
          f"równolegle: {s.file_workers} plik(i), {s.vlm_parallel} str. do VLM, "
          f"{s.ollama.num_parallel} zapytań w serwerze")


# ---------------------------------------------------------------- przygotowanie

def setup_ocr(s: Settings):
    from .ocr import get_ocr
    return get_ocr(s.ocr)


def _installed_models(vlm) -> set[str] | None:
    """Lista modeli w Ollamie (None, gdy backend nie jest Ollamą albo nie da się sprawdzić)."""
    if vlm.backend != "ollama":
        return None
    try:
        import requests
        r = requests.get(f"{vlm.base}/api/tags", timeout=5)
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return None


def setup_vlm(s: Settings):
    """Zwraca (vlm | None, kod błędu). Bez Ollamy - tryb CPU (geometria + OCR)."""
    if not s.use_vlm:
        print("VLM: wyłączony - tryb CPU: geometria + OCR")
        return None, 0
    from .llm import VLMRefiner, load_config
    cfg = load_config(s.vlm_config)
    cfg["parallel"] = s.vlm_parallel
    if cfg["backend"] == "ollama":
        from .ollama_server import ensure_model, prepare
        try:
            cfg["base_url"], how = prepare(cfg["base_url"], s.ollama)
            print(f"Ollama: {how}")
        except Exception as e:
            print(f"Nie udało się uruchomić własnego serwera Ollama ({e}) - używam {cfg['base_url']}",
                  file=sys.stderr)
        if s.ollama.auto_pull:
            ensure_model(cfg["base_url"].rstrip("/"), s.model or cfg["model"])
    vlm = VLMRefiner(cfg, model=s.model)
    if not vlm.available():
        print(f"VLM niedostępny ({vlm.base}) - tryb CPU: geometria + OCR", file=sys.stderr)
        return None, (2 if s.force_vlm else 0)
    models = _installed_models(vlm)
    if models is not None and vlm.model not in models:
        print(f"Model {vlm.model} nie jest pobrany (ollama pull {vlm.model}).", file=sys.stderr)
        if s.fallback_model in models and s.fallback_model != vlm.model:
            print(f"  -> używam modelu domyślnego {s.fallback_model}", file=sys.stderr)
            vlm = VLMRefiner(cfg, model=s.fallback_model)
        else:
            print("  -> tryb CPU: geometria + OCR", file=sys.stderr)
            return None, (2 if s.force_vlm else 0)
    print(f"VLM: {vlm.model} @ {vlm.base} ({vlm.backend})")
    return vlm, 0


def iter_inputs(paths: list[str], exclude: Path | None = None):
    ex = exclude.resolve() if exclude else None
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED
                              and not (ex and ex in q.resolve().parents)
                              and "wizualizacja" not in q.name.lower()      # gotowe wizualizacje, nie formularze
                              and not any(x.name in ("form2json", ".venv") or x.name.startswith("wyniki")
                                          for x in q.parents))
        elif p.suffix.lower() in SUPPORTED:
            yield p


def collect_inputs(s: Settings) -> list[Path]:
    s.out.mkdir(parents=True, exist_ok=True)
    if s.rebuild:      # --rebuild: istniejące *.fields.json (bez OCR/VLM)
        files = sorted(Path(x) for pat in s.inputs for x in (Path().glob(pat) if "*" in pat else [pat]))
    else:
        files = list(iter_inputs(s.inputs, s.out))
    print(f"Plików do rozpoznania: {len(files)}")
    return files


# ---------------------------------------------------------------- przetwarzanie

def draw_debug(path: Path, generic: dict, out: Path, dpi: int, ocr) -> None:
    import cv2
    from .loaders import load_document
    doc = load_document(path, dpi=dpi, ocr=ocr)
    colors = {"checkbox": (0, 160, 0), "comb": (200, 0, 200), "box": (0, 0, 255), "dots": (255, 120, 0),
              "underline": (0, 140, 255), "captioned": (160, 0, 0)}
    for p in doc.pages:
        img = p.image.copy()
        for f in generic["fields"]:
            if f.get("page", 0) != p.index or "bbox" not in f:
                continue
            c = colors.get(f.get("detector", "").split("+")[0].split("-")[0], (0, 0, 0))
            x0, y0, x1, y1 = f["bbox"]
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
            if "label_bbox" in f:
                a = f["label_bbox"]
                cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (180, 180, 180), 1)
                cv2.line(img, (a[2], (a[1] + a[3]) // 2), (x0, (y0 + y1) // 2), (180, 180, 180), 1)
            cv2.putText(img, f["type"], (x0 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(out.with_name(out.name.replace(".debug.png", f".p{p.index + 1}.debug.png"))))


def recognize(p: Path, s: Settings, ocr, vlm) -> dict:
    """Ogólny JSON pól z pliku (albo z zapisanego *.fields.json przy --rebuild)."""
    if s.rebuild:
        from .pipeline import reanalyze
        return reanalyze(json.loads(p.read_text("utf-8")))
    return extract(p, ocr=ocr, vlm=vlm, dpi=s.dpi)


def save_outputs(p: Path, name: str, g: dict, s: Settings, ocr) -> tuple[list, list]:
    """Zapis *.fields.json, *.form.json, *.simple.json; zwraca (błędy, ostrzeżenia) walidacji."""
    from .validate import validate
    (s.out / f"{name}.fields.json").write_text(json.dumps(g, ensure_ascii=False, indent=2), "utf-8")
    fd = to_form_definition(g, s.tabs)
    (s.out / f"{name}.form.json").write_text(json.dumps(fd, ensure_ascii=False, indent=2), "utf-8")
    if s.debug and not s.rebuild:
        draw_debug(p, g, s.out / f"{name}.debug.png", s.dpi, ocr)
    errs, warns = validate(fd)
    (s.out / f"{name}.simple.json").write_text(json.dumps(
        {"title": g["title"], "pola": g["simple"]}, ensure_ascii=False, indent=2), "utf-8")
    return errs, warns


def fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f} s"
    m, s_ = divmod(sec, 60)
    if m < 60:
        return f"{int(m)} min {s_:04.1f} s"
    h, m = divmod(m, 60)
    return f"{int(h)} h {int(m):02d} min {int(s_):02d} s"


def process_file(n: int, p: Path, s: Settings, ocr, vlm, stats: RunStats) -> FileResult:
    name = p.name.removesuffix(".fields.json") if s.rebuild else p.name
    t0 = time.perf_counter()
    try:
        g = recognize(p, s, ocr, vlm)
        errs, warns = save_outputs(p, name, g, s, ocr)
        res = FileResult(name, True, int(g.get("pages") or 0), len(g["fields"]), time.perf_counter() - t0)
        i = stats.done()
        msg = (f"[OK] [{i}/{n}] {name}: {res.fields} pól, {res.pages} str. ({', '.join(g['text_source'])}); "
               f"walidacja: {len(errs)} błędów, {len(warns)} ostrzeżeń")
        if s.timing:
            msg += f" | plik: {fmt_time(res.seconds)} | łącznie: {fmt_time(stats.elapsed)}"
        print(msg, flush=True)
        for e in errs:
            print(f"     BŁĄD: {e}", file=sys.stderr)
    except Exception as e:
        res = FileResult(name, False, seconds=time.perf_counter() - t0)
        i = stats.done()
        msg = f"[ERR] [{i}/{n}] {name}: {e}"
        if s.timing:
            msg += f" | plik: {fmt_time(res.seconds)} | łącznie: {fmt_time(stats.elapsed)}"
        print(msg, file=sys.stderr, flush=True)
        traceback.print_exc()
    return res


def process_files(files: list[Path], s: Settings, ocr, vlm) -> RunStats:
    stats = RunStats()              # zegar startuje razem z rozpoznawaniem pierwszego pliku
    if s.file_workers > 1 and len(files) > 1:
        # kilka plików naraz: LibreOffice/OCR (CPU) jednego pliku nakłada się z VLM (GPU) innego
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=s.file_workers) as ex:
            stats.results = list(ex.map(lambda p: process_file(len(files), p, s, ocr, vlm, stats), files))
    else:
        stats.results = [process_file(len(files), p, s, ocr, vlm, stats) for p in files]
    return stats


# ---------------------------------------------------------------- podsumowanie i raport

def summarize(stats: RunStats) -> dict:
    ok = [r for r in stats.results if r.ok]
    total = stats.elapsed
    pages = sum(r.pages for r in ok)
    return {"files": len(stats.results), "files_ok": len(ok), "files_err": len(stats.results) - len(ok),
            "pages": pages, "fields": sum(r.fields for r in ok), "seconds": total,
            "per_file": total / len(stats.results) if stats.results else 0.0,
            "per_page": total / pages if pages else 0.0}


def print_summary(sm: dict) -> None:
    print("\n=== Podsumowanie ===")
    print(f"Rozpoznane pliki:      {sm['files_ok']}/{sm['files']}"
          + (f"  (błędy: {sm['files_err']})" if sm['files_err'] else ""))
    print(f"Rozpoznane strony:     {sm['pages']}")
    print(f"Wykryte pola do wpisania danych: {sm['fields']}")
    print(f"Czas łącznie:          {fmt_time(sm['seconds'])}")
    print(f"Średnio na plik:       {fmt_time(sm['per_file'])}")
    print(f"Średnio na stronę:     {fmt_time(sm['per_page'])}")


def make_report(s: Settings, sm: dict | None) -> None:
    from .report import generate_report
    path = generate_report(s.out, s.ref_dir, summary=sm, echo=False)
    print(f"\nRaport: {path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = resolve_settings(args, load_runtime(args.runtime_config))
    print_settings(settings)
    ocr = setup_ocr(settings)
    vlm, rc = setup_vlm(settings)
    if rc:
        return rc
    files = collect_inputs(settings)
    stats = process_files(files, settings, ocr, vlm)
    sm = summarize(stats)
    if settings.timing:
        print_summary(sm)
    if settings.report:
        make_report(settings, sm if settings.timing else None)
    return 0 if sm["files_err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
