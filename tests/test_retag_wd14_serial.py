# -*- coding: utf-8 -*-
"""WD14 串行轮询行为验证。

关键不变量：
  * 前一个 Space 成功后，**不再请求**后续 Space（不浪费 HF 三家流量）。
  * 前一个 Space 失败后，才继续下一个。
  * 全失败时抛 WD14ClientError，detail 是最后一次错误。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.retag.wd14_client import WD14Client, WD14ClientError


def _make_spaces(n: int) -> list[Dict[str, Any]]:
    return [
        {"name": f"demo/space_{i}", "type": "pixai", "api": "/predict_image"}
        for i in range(1, n + 1)
    ]


def _build_client(spaces: list[Dict[str, Any]]) -> WD14Client:
    client = WD14Client(spaces_config=spaces, max_retries=1, retry_delay=0.01)
    client._gradio_available = True
    return client


def test_serial_polling_stops_at_first_success() -> None:
    """前一个 Space 成功后，绝不应再请求后续 Space。"""
    spaces = _make_spaces(3)
    client = _build_client(spaces)
    call_order: list[str] = []

    async def fake_tag_with_space(self: WD14Client, *, space_info: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
        name = space_info["name"]
        call_order.append(name)
        # 第一个就成功
        return {"tags": [{"label": "1girl", "score": 0.99}]}

    with patch.object(WD14Client, "_tag_with_space", new=fake_tag_with_space):
        result = asyncio.run(
            client.tag_image(image_base64="x" * 100, threshold=0.3, character_threshold=0.8)
        )

    assert result["tags"][0]["label"] == "1girl"
    assert call_order == ["demo/space_1"], (
        f"轮询模式下第一个成功后不应继续，实际调用了：{call_order}"
    )
    assert client.current_space_name == "demo/space_1"


def test_serial_polling_falls_through_on_failure() -> None:
    """前一个 Space 失败时，才打扰下一个。"""
    spaces = _make_spaces(3)
    client = _build_client(spaces)
    call_order: list[str] = []

    async def fake_tag_with_space(self: WD14Client, *, space_info: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
        name = space_info["name"]
        call_order.append(name)
        if name in ("demo/space_1", "demo/space_2"):
            raise WD14ClientError(f"{name} 模拟失败")
        return {"tags": [{"label": "smile", "score": 0.7}]}

    with patch.object(WD14Client, "_tag_with_space", new=fake_tag_with_space):
        result = asyncio.run(
            client.tag_image(image_base64="x" * 100, threshold=0.3, character_threshold=0.8)
        )

    assert call_order == ["demo/space_1", "demo/space_2", "demo/space_3"]
    assert result["tags"][0]["label"] == "smile"
    assert client.current_space_name == "demo/space_3"


def test_all_failed_raises_with_last_error() -> None:
    spaces = _make_spaces(3)
    client = _build_client(spaces)

    async def fake_tag_with_space(self: WD14Client, *, space_info: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
        raise WD14ClientError(f"{space_info['name']} 挂了")

    with patch.object(WD14Client, "_tag_with_space", new=fake_tag_with_space):
        try:
            asyncio.run(
                client.tag_image(image_base64="x" * 100, threshold=0.3, character_threshold=0.8)
            )
        except WD14ClientError as exc:
            # 最后一次错误应当来自最后一个 Space
            assert "demo/space_3" in str(exc), f"detail 应含最后一次失败信息，实际：{exc}"
        else:  # pragma: no cover
            raise AssertionError("全部失败时必须抛 WD14ClientError")


def test_overall_timeout_stops_polling() -> None:
    """每个 Space 都很慢、总耗时超 command_timeout 时不应无限串。"""
    spaces = _make_spaces(3)
    client = _build_client(spaces)
    # 把整体超时压得很低，让 wait_for 在第一个 Space 内就触发超时
    client.command_timeout = 0.2

    call_order: list[str] = []

    async def slow_tag_with_space(self: WD14Client, *, space_info: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
        call_order.append(space_info["name"])
        await asyncio.sleep(0.5)
        return {"tags": []}

    with patch.object(WD14Client, "_tag_with_space", new=slow_tag_with_space):
        try:
            asyncio.run(
                client.tag_image(image_base64="x" * 100, threshold=0.3, character_threshold=0.8)
            )
        except WD14ClientError as exc:
            assert "超时" in str(exc) or "timeout" in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError("应抛超时错误")

    # 命中超时后不应该把剩余 Space 全部串完
    assert len(call_order) < 3, (
        f"整体超时后应停止后续轮询，实际跑了 {len(call_order)} 个 Space"
    )
