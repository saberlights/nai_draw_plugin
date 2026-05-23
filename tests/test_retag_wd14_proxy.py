# -*- coding: utf-8 -*-
"""WD14 客户端代理配置与连接逻辑的单元测试。

主要验证 ``proxy`` 参数是否被正确透传到 ``gradio_client.Client`` 的
``httpx_kwargs``；不真的访问网络。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.retag import wd14_client as wd14_module
from plugins.nai_draw_plugin.core.retag.wd14_client import WD14Client


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


def test_proxy_passed_into_httpx_kwargs() -> None:
    """配置了 proxy 时应该出现在 httpx_kwargs 里。"""
    _reset_stub()
    client = WD14Client(
        spaces_config=[{"name": "demo/space", "type": "danbooru_v4", "api": "/_fn_submit"}],
        proxy="http://127.0.0.1:7890",
    )
    with patch.object(wd14_module, "Client", _StubGradioClient), patch.object(
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
    client = WD14Client(spaces_config=[{"name": "x/y", "type": "pixai", "api": "/predict_image"}])
    with patch.object(wd14_module, "Client", _StubGradioClient), patch.object(
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
    )
    assert client.proxy is None


def test_proxy_fallback_when_httpx_rejects_proxy_kwarg() -> None:
    """httpx 旧版不识别 proxy= 时回退用 proxies=。"""
    _reset_stub()
    _StubGradioClient.raise_on_first = True
    client = WD14Client(
        spaces_config=[{"name": "demo/space", "type": "danbooru_v4", "api": "/_fn_submit"}],
        proxy="http://127.0.0.1:7890",
    )
    with patch.object(wd14_module, "Client", _StubGradioClient), patch.object(
        wd14_module, "GRADIO_AVAILABLE", True
    ):
        result = client._get_or_create_client("demo/space")
    assert isinstance(result, _StubGradioClient)
    httpx_kwargs = _StubGradioClient.last_kwargs.get("httpx_kwargs", {})
    assert httpx_kwargs.get("proxies") == "http://127.0.0.1:7890"


def test_tag_with_space_runs_client_construction_in_executor() -> None:
    """gradio_client.Client(...) 同步阻塞 ≈ 12s，必须经 executor，否则 event loop 被冻。

    用源码扫描代替运行时断言：直接验证 `_tag_with_space` 函数体里调用 `_get_or_create_client`
    时套了 `run_in_executor`，避免后续重构者下意识把它改回同步调用。
    """
    import inspect

    source = inspect.getsource(WD14Client._tag_with_space)
    # 必须存在 run_in_executor 调用，且其参数包含 _get_or_create_client
    assert "run_in_executor" in source, "_tag_with_space 必须经 executor 调用 Client(...)"
    assert "_get_or_create_client" in source
    # 反向防回归：不允许出现裸的同步调用 self._get_or_create_client(...) 直接拿返回值
    bad_pattern = "client = self._get_or_create_client(space_name)"
    assert bad_pattern not in source, (
        "检测到同步调用 _get_or_create_client；这会冻结 event loop 12s 量级，"
        "请改回 await loop.run_in_executor(None, self._get_or_create_client, space_name)"
    )
