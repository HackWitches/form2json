"""Wczytanie runtime.config (INI): ścieżki, wybór modelu, raport, pomiar czasu."""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RUNTIME = Path(__file__).with_name("runtime.config")
DEFAULT_MODEL = "qwen3-vl:8b-instruct"
FAST_MODEL = "qwen3-vl:4b-instruct"
ACCURATE_MODEL = "qwen3-vl:30b-a3b-instruct"


@dataclass
class Runtime:
    inputs: list[str] = field(default_factory=list)
    output: str = "out"
    vlm_enabled: bool = True
    profile: str = "default"
    models: dict = field(default_factory=lambda: {"default": DEFAULT_MODEL, "fast": FAST_MODEL,
                                                  "accurate": ACCURATE_MODEL})
    ocr: str = "auto"
    tabs: str = "auto"
    dpi: int = 150
    debug: bool = False
    report: bool = False
    ref_dir: str = ""
    timing: bool = False
    source: str = ""            # z jakiego pliku wczytano (puste = wartości domyślne)

    @property
    def model(self) -> str:
        return self.models.get(self.profile) or self.models["default"]


def load_runtime(path: str | Path | None = None) -> Runtime:
    p = Path(path) if path else DEFAULT_RUNTIME
    rt = Runtime()
    if not p.exists():
        if path:
            raise FileNotFoundError(f"Brak pliku konfiguracji: {p}")
        return rt
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=None)
    cp.read(p, encoding="utf-8")
    base = p.resolve().parent

    def path_of(s: str) -> str:
        s = s.strip().strip('"')
        return str((base / s).resolve()) if s and not Path(s).is_absolute() else s

    g = lambda sec, key, d="": cp.get(sec, key, fallback=d).strip()
    b = lambda sec, key, d: cp.getboolean(sec, key, fallback=d)
    rt.inputs = [path_of(x) for x in g("paths", "input").split(";") if x.strip()]
    rt.output = path_of(g("paths", "output", rt.output))
    rt.vlm_enabled = b("model", "vlm_enabled", rt.vlm_enabled)
    rt.profile = g("model", "profile", rt.profile).lower() or "default"
    for k in rt.models:
        rt.models[k] = g("model", f"{k}_model", rt.models[k]) or rt.models[k]
    if rt.profile not in rt.models:
        raise ValueError(f"runtime.config: profile = {rt.profile!r}, dozwolone: {', '.join(rt.models)}")
    rt.ocr = g("processing", "ocr", rt.ocr) or rt.ocr
    rt.tabs = g("processing", "tabs", rt.tabs) or rt.tabs
    rt.dpi = cp.getint("processing", "dpi", fallback=rt.dpi)
    rt.debug = b("processing", "debug", rt.debug)
    rt.report = b("report", "enabled", rt.report)
    rt.ref_dir = path_of(g("report", "ref_dir"))
    rt.timing = b("timing", "enabled", rt.timing)
    rt.source = str(p)
    return rt
