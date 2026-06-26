# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import httpx

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


dummy_logger_module.get_logger = lambda _name=None: _DummyLogger()
sys.modules["src.common.logger"] = dummy_logger_module

from plugins.nai_draw_plugin.core.clients.danbooru_online_client import DanbooruOnlineClient


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_danbooru_online_client_ignores_proxy_env_for_health_search_and_related(monkeypatch) -> None:
    client_kwargs: list[dict[str, object]] = []
    calls: list[tuple[str, str, object]] = []

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            client_kwargs.append(dict(kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            calls.append(("GET", url, None))
            return _FakeResponse({"status": "ok", "loaded": True})

        async def post(self, url: str, json=None):
            calls.append(("POST", url, json))
            if url.endswith("/search"):
                return _FakeResponse({"results": [{"tag": "hatsune_miku"}]})
            return _FakeResponse({"results": [{"tag": "twintails"}]})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:7890")

    client = DanbooruOnlineClient(base_url="https://example.com/api", timeout=12.0)

    assert asyncio.run(client.health_check()) is True
    assert asyncio.run(client.search("初音未来")) == {"results": [{"tag": "hatsune_miku"}]}
    assert asyncio.run(client.related(["hatsune_miku"])) == [{"tag": "twintails"}]

    assert calls == [
        ("GET", "https://example.com/api/health", None),
        ("POST", "https://example.com/api/search", {
            "query": "初音未来",
            "top_k": 5,
            "limit": 80,
            "popularity_weight": 0.15,
            "show_nsfw": False,
            "use_segmentation": True,
        }),
        ("POST", "https://example.com/api/related", {
            "tags": ["hatsune_miku"],
            "limit": 50,
            "show_nsfw": False,
        }),
    ]
    assert client_kwargs == [
        {"timeout": 12.0, "trust_env": False},
        {"timeout": 12.0, "trust_env": False},
        {"timeout": 12.0, "trust_env": False},
    ]
