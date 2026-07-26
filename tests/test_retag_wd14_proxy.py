# -*- coding: utf-8 -*-
"""WD14 客户端代理配置与连接逻辑的单元测试。

主要验证 ``proxy`` 参数是否被正确透传到 ``gradio_client.Client`` 的
``httpx_kwargs``；不真的访问网络。
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.retag import wd14_client as wd14_module
from plugins.nai_draw_plugin.core.retag.wd14_client import WD14Client
from plugins.nai_draw_plugin.core.services.blocking_io_runner import BlockingIORunner


class _StubGradioClient:
    """伪 Client：只记录最近一次构造时收到的参数。"""

    last_kwargs: Dict[str, Any] = {}
    last_args: tuple = ()
    raise_on_first: bool = False
    first_called: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _StubGradioClient.last_args = args
        _StubGradioClient.last_kwargs = dict(kwargs)
        if _StubGradioClient.raise_on_first and not _StubGradioClient.first_called:
            _StubGradioClient.first_called = True
            raise TypeError("got unexpected keyword argument 'proxy'")


def _reset_stub() -> None:
    _StubGradioClient.last_kwargs = {}
    _StubGradioClient.last_args = ()
    _StubGradioClient.raise_on_first = False
    _StubGradioClient.first_called = False


async def _run_inline(function, *args, **_kwargs):
    return function(*args)


def test_proxy_passed_into_httpx_kwargs() -> None:
    """配置了 proxy 时应该出现在 httpx_kwargs 里。"""
    _reset_stub()
    client = WD14Client(
        spaces_config=[{"name": "demo/space", "type": "danbooru_v4", "api": "/_fn_submit"}],
        proxy="http://127.0.0.1:7890",
        run_blocking=_run_inline,
    )
    with patch.object(wd14_module, "Client", _StubGradioClient, create=True), patch.object(
        wd14_module, "GRADIO_AVAILABLE", True
    ):
        result = client._get_or_create_client("demo/space")
    assert isinstance(result, _StubGradioClient)
    httpx_kwargs = _StubGradioClient.last_kwargs.get("httpx_kwargs", {})
    assert httpx_kwargs.get("proxy") == "http://127.0.0.1:7890"
    assert httpx_kwargs.get("timeout") == client.timeout


def test_empty_proxy_does_not_pollute_httpx_kwargs() -> None:
    """没配 proxy 时 httpx_kwargs 不应包含 proxy 键，避免覆盖 httpx 默认行为。"""
    _reset_stub()
    client = WD14Client(
        spaces_config=[{"name": "x/y", "type": "pixai", "api": "/predict_image"}],
        run_blocking=_run_inline,
    )
    with patch.object(wd14_module, "Client", _StubGradioClient, create=True), patch.object(
        wd14_module, "GRADIO_AVAILABLE", True
    ):
        client._get_or_create_client("x/y")
    httpx_kwargs = _StubGradioClient.last_kwargs.get("httpx_kwargs", {})
    assert "proxy" not in httpx_kwargs


def test_blank_proxy_string_treated_as_unset() -> None:
    """空白字符串等价于没设代理。"""
    _reset_stub()
    client = WD14Client(
        spaces_config=[{"name": "x/y", "type": "pixai", "api": "/predict_image"}],
        proxy="   ",
        run_blocking=_run_inline,
    )
    assert client.proxy is None


def test_proxy_fallback_when_httpx_rejects_proxy_kwarg() -> None:
    """httpx 旧版不识别 proxy= 时回退用 proxies=。"""
    _reset_stub()
    _StubGradioClient.raise_on_first = True
    client = WD14Client(
        spaces_config=[{"name": "demo/space", "type": "danbooru_v4", "api": "/_fn_submit"}],
        proxy="http://127.0.0.1:7890",
        run_blocking=_run_inline,
    )
    with patch.object(wd14_module, "Client", _StubGradioClient, create=True), patch.object(
        wd14_module, "GRADIO_AVAILABLE", True
    ):
        result = client._get_or_create_client("demo/space")
    assert isinstance(result, _StubGradioClient)
    httpx_kwargs = _StubGradioClient.last_kwargs.get("httpx_kwargs", {})
    assert httpx_kwargs.get("proxies") == "http://127.0.0.1:7890"


def test_tag_with_space_routes_all_blocking_calls_through_injected_runner() -> None:
    calls: list[str] = []

    async def recording_runner(function, *args, **_kwargs):
        calls.append(function.__name__)
        return function(*args)

    client = WD14Client(
        spaces_config=[{"name": "x/y", "type": "pixai", "api": "/predict_image"}],
        run_blocking=recording_runner,
    )
    client._get_or_create_client = lambda _space_name: object()
    client._predict_space_with_retry = lambda *_args: [
        "",
        "",
        "",
        "",
        {},
        {"feature_scores": {"1girl": 0.95}},
    ]

    result = asyncio.run(
        client._tag_with_space(
            image_base64="aW1hZ2U=",
            threshold=0.35,
            character_threshold=0.8,
            space_info={"name": "x/y", "type": "pixai", "api": "/predict_image"},
        )
    )

    assert [tag["label"] for tag in result["tags"]] == ["1girl"]
    assert calls == ["<lambda>", "<lambda>"]


def test_cancellation_waits_for_prediction_before_removing_temporary_image() -> None:
    runner = BlockingIORunner(thread_name_prefix="test-wd14-io")
    client = WD14Client(
        spaces_config=[{"name": "x/y", "type": "pixai", "api": "/predict_image"}],
        run_blocking=runner.run,
    )
    prediction_started = threading.Event()
    release_prediction = threading.Event()
    observed_paths: list[Path] = []
    client._get_or_create_client = lambda _space_name: object()

    def blocking_prediction(_client, image_path, *_args):
        path = Path(image_path)
        observed_paths.append(path)
        prediction_started.set()
        assert release_prediction.wait(timeout=2.0)
        assert path.exists()
        return []

    client._predict_space_with_retry = blocking_prediction

    async def scenario() -> None:
        task = asyncio.create_task(
            client._tag_with_space(
                image_base64="aW1hZ2U=",
                threshold=0.35,
                character_threshold=0.8,
                space_info={"name": "x/y", "type": "pixai", "api": "/predict_image"},
            )
        )
        while not prediction_started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert observed_paths[0].exists()
        release_prediction.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not observed_paths[0].exists()

    try:
        asyncio.run(scenario())
    finally:
        release_prediction.set()
        runner.close()
