"""Serwer Ollama z ustawieniami przyspieszającymi - niezależnie od systemu operacyjnego.

Ollama domyślnie obsługuje jedno zapytanie naraz i bez flash attention. Zamiast ustawiać zmienne
środowiskowe w systemie (inaczej w Windows, Linux, Docker), form2json sam uruchamia `ollama serve`
jako proces potomny z właściwymi zmiennymi, na osobnym porcie, i zamyka go po zakończeniu pracy.
Modele są brane z tego samego katalogu co systemowa Ollama (~/.ollama/models), bez ponownego pobierania.

Tryby ([ollama] manage w runtime.config):
  auto   - jeśli adres VLM jest lokalny i jest program `ollama`: własny serwer z ustawieniami;
           w przeciwnym razie (np. Docker: http://ollama:11434) używa adresu z konfiguracji
  always - zawsze własny serwer (błąd, jeśli brak programu `ollama`)
  never  - zawsze adres z konfiguracji (serwer skonfigurowany samodzielnie)

W Dockerze serwer Ollama działa w osobnym kontenerze z tymi samymi zmiennymi (deploy/docker-compose.yml).
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


@dataclass
class OllamaSettings:
    manage: str = "auto"            # auto | always | never
    port: int = 11435               # port własnego serwera (systemowa Ollama zwykle 11434)
    flash_attention: bool = True
    num_parallel: int = 4
    keep_alive: str = "30m"         # jak długo model zostaje w VRAM po ostatnim zapytaniu
    auto_pull: bool = True          # pobierz model, jeśli go nie ma


def find_ollama() -> str | None:
    p = shutil.which("ollama")
    if p:
        return p
    cands = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
             Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama"),
             Path("/Applications/Ollama.app/Contents/Resources/ollama")]
    return next((str(c) for c in cands if c.is_file()), None)


def is_up(base: str, timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{base}/api/version", timeout=timeout).ok
    except Exception:
        return False


def server_env(s: OllamaSettings) -> dict:
    """Zmienne środowiskowe serwera Ollama (te same w procesie potomnym i w docker-compose)."""
    return {"OLLAMA_FLASH_ATTENTION": "1" if s.flash_attention else "0",
            "OLLAMA_NUM_PARALLEL": str(s.num_parallel),
            "OLLAMA_KEEP_ALIVE": s.keep_alive}


class ManagedOllama:
    """`ollama serve` uruchomiony przez form2json; zamykany przy wyjściu z programu."""

    def __init__(self, exe: str, s: OllamaSettings):
        self.exe, self.s = exe, s
        self.base = f"http://127.0.0.1:{s.port}"
        self.proc: subprocess.Popen | None = None

    def start(self, wait_s: float = 60) -> str:
        if is_up(self.base):            # już działa (np. z poprzedniego uruchomienia) - używamy go
            return self.base
        env = {**os.environ, **server_env(self.s), "OLLAMA_HOST": f"127.0.0.1:{self.s.port}"}
        log = open(Path(os.environ.get("TEMP", "/tmp")) / "form2json_ollama.log", "ab")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen([self.exe, "serve"], env=env, stdout=log, stderr=log,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        atexit.register(self.stop)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ollama serve zakończył się (kod {self.proc.returncode}), log: {log.name}")
            if is_up(self.base):
                return self.base
            time.sleep(0.5)
        raise RuntimeError(f"ollama serve nie odpowiada na {self.base} po {wait_s:.0f} s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def ensure_model(base: str, model: str) -> bool:
    """Pobiera model przez API Ollamy, jeśli go brak (postęp w konsoli). Zwraca True, gdy model jest."""
    try:
        names = {m["name"] for m in requests.get(f"{base}/api/tags", timeout=10).json().get("models", [])}
    except Exception:
        return False
    if model in names:
        return True
    print(f"Pobieranie modelu {model} (jednorazowo)...", flush=True)
    last = ""
    with requests.post(f"{base}/api/pull", json={"name": model, "stream": True}, stream=True, timeout=None) as r:
        for line in r.iter_lines():
            if not line:
                continue
            j = json.loads(line)
            if "error" in j:
                print(f"  błąd: {j['error']}", file=sys.stderr)
                return False
            st = j.get("status", "")
            if j.get("total"):
                st = f"{st} {j.get('completed', 0) / j['total']:.0%}"
            if st != last:
                print(f"\r  {st:<60}", end="", flush=True)
                last = st
    print()
    return True


_MANAGED: ManagedOllama | None = None


def prepare(base_url: str, s: OllamaSettings) -> tuple[str, str]:
    """Zwraca (adres serwera do użycia, opis). Uruchamia własny serwer, jeśli tak wynika z ustawień."""
    global _MANAGED
    local = (urlparse(base_url).hostname or "") in LOCAL_HOSTS
    if s.manage == "never" or (s.manage == "auto" and not local):
        return base_url, "serwer z konfiguracji"
    exe = find_ollama()
    if not exe:
        if s.manage == "always":
            raise RuntimeError("manage = always, ale nie znaleziono programu `ollama`")
        return base_url, "serwer z konfiguracji (brak programu ollama)"
    _MANAGED = ManagedOllama(exe, s)
    base = _MANAGED.start()
    env = server_env(s)
    return base, (f"własny serwer (flash attention={env['OLLAMA_FLASH_ATTENTION']}, "
                  f"równoległe zapytania={env['OLLAMA_NUM_PARALLEL']})")
