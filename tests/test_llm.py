import json

import numpy as np
import pytest

from form2json import llm
from form2json.model import FieldCandidate, Page


def _vlm(chat, **cfg):
    v = llm.VLMRefiner(llm.load_config(None) | cfg)
    v._chat = chat
    return v


def _field(label, **kw):
    return {"label": label, "type": "string", "role": "subject", "section": "", "required": True, **kw}


def test_tile_error_falls_back_to_geometry_for_whole_page():
    page = Page(0, np.full((3000, 1000, 3), 255, np.uint8), 150)        # 3 kafelki
    cands = [FieldCandidate(label=f"Pole {i}", page=0, bbox=[0, i, 1, i + 1], detector="box") for i in range(3)]
    calls = []

    def chat(images, text):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError("timeout")
        return {"title": "Wniosek", "fields": [_field("Pole 0")]}

    _, res, _ = _vlm(chat).refine([page], cands, "", "x")
    assert [r.label for r in res] == ["Pole 0", "Pole 1", "Pole 2"]


def test_invalid_vlm_response_does_not_crash_file(monkeypatch):
    class Resp:                          # serwer zgodny z OpenAI, który nie wymusił schematu
        def raise_for_status(self):
            pass

        def json(self):
            out = {"title": "x", "fields": [_field("Płeć", type="radio", options=[1, 2])]}
            return {"choices": [{"message": {"content": json.dumps(out)}}]}

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp())
    v = llm.VLMRefiner(llm.load_config(None) | {"backend": "openai"})
    page = Page(0, np.full((100, 100, 3), 255, np.uint8), 150)
    cand = FieldCandidate(label="Płeć", page=0, bbox=[0, 0, 1, 1], detector="checkbox-group")
    _, res, _ = v.refine([page], [cand], "", "x")
    assert res == [cand]


def test_tiles_have_no_sliver():
    img = np.zeros((int(1000 * 1.414 * 2.02), 1000, 3), np.uint8)
    assert min(t.shape[0] for t in llm._tiles(img)) > 0.3 * 1414


def test_remote_vlm_rejected_unless_allowed():
    with pytest.raises(ValueError):
        llm.VLMRefiner(llm.load_config(None) | {"base_url": "https://api.example.com"})
    llm.VLMRefiner(llm.load_config(None) | {"base_url": "https://api.example.com", "allow_remote": True})
    llm.VLMRefiner(llm.load_config(None) | {"base_url": "http://192.168.1.10:11434"})


def test_parse_partial_recovers_complete_fields():
    txt = '{"title": "W", "fields": [{"label": "A", "type": "string"}, {"label": "B", "ty'
    assert [x["label"] for x in llm.parse_partial(txt)["fields"]] == ["A"]


def test_docker_service_name_resolving_to_private_ip_is_local(monkeypatch):
    # deploy/docker-compose.yml: FORM2JSON_VLM_URL=http://ollama:11434 (sieć Dockera, adres prywatny)
    addrs = {"ollama": "172.18.0.2", "evil.example": "93.184.216.34"}
    monkeypatch.setattr(llm.socket, "getaddrinfo", lambda h, *_: [(2, 1, 6, "", (addrs[h], 0))])
    assert llm._is_local("http://ollama:11434")
    assert not llm._is_local("http://evil.example:11434")
