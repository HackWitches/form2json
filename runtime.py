"""Wczytanie runtime.config (INI): ścieżki, wybór modelu, raport, pomiar czasu."""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

from .ollama_server import OllamaSettings

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
    vlm_parallel: int = 4       # ile stron jednego pliku naraz do VLM (wymaga OLLAMA_NUM_PARALLEL >= tej liczby)
    file_workers: int = 2       # ile plików przetwarzać naraz (LibreOffice/OCR na CPU nakładają się z GPU)
    ocr: str = "auto"
    tabs: str = "auto"
    dpi: int = 150
    debug: bool = False
    report: bool = False
    ref_dir: str = ""
    timing: bool = False
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
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
    def num(key: str, old_sec: tuple[str, str], default: int) -> int:
        """[performance] key; starsze pliki miały te klucze w [model]/[processing]/[ollama]."""
        v = cp.getint("performance", key, fallback=None)
        return max(1, v if v is not None else cp.getint(old_sec[0], old_sec[1], fallback=default))

    rt.file_workers = num("file_workers", ("processing", "file_workers"), rt.file_workers)
    rt.vlm_parallel = num("vlm_parallel", ("model", "vlm_parallel"), rt.vlm_parallel)
    rt.ocr = g("processing", "ocr", rt.ocr) or rt.ocr
    rt.tabs = g("processing", "tabs", rt.tabs) or rt.tabs
    rt.dpi = cp.getint("processing", "dpi", fallback=rt.dpi)
    rt.debug = b("processing", "debug", rt.debug)
    rt.report = b("report", "enabled", rt.report)
    rt.ref_dir = path_of(g("report", "ref_dir"))
    rt.timing = b("timing", "enabled", rt.timing)
    o = rt.ollama
    o.manage = (g("ollama", "manage", o.manage) or o.manage).lower()
    if o.manage not in ("auto", "always", "never"):
        raise ValueError(f"runtime.config: [ollama] manage = {o.manage!r}, dozwolone: auto, always, never")
    o.port = cp.getint("ollama", "port", fallback=o.port)
    o.flash_attention = b("ollama", "flash_attention", o.flash_attention)
    o.num_parallel = num("server_parallel", ("ollama", "num_parallel"), rt.vlm_parallel)
    o.keep_alive = g("ollama", "keep_alive", o.keep_alive) or o.keep_alive
    o.auto_pull = b("ollama", "auto_pull", o.auto_pull)
    rt.source = str(p)
    return rt
