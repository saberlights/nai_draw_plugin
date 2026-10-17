"""反推耗时不能占住命令 RPC；结果与异常仍须送回原会话。"""

from types import SimpleNamespace
from typing import Any
import asyncio
import base64
import random

import pytest

from . import test_retag_config_layout as _stub_bootstrap  # noqa: F401
from plugins.nai_draw_plugin import plugin as plugin_module
from plugins.nai_draw_plugin.core.retag.reverser import ReverseResult
from plugins.nai_draw_plugin.core.services.background_task_supervisor import BackgroundTaskSupervisor
from plugins.nai_draw_plugin.plugin import NaiPicPlugin


class _RecordingSend:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.finished = asyncio.Event()

    async def text(self, text: str, stream_id: str, **kwargs: Any) -> bool:
        self.messages.append((stream_id, text))
        if text.startswith(("✅", "❌")):
            self.finished.set()
        return True


def _build_plugin(reverse: Any) -> tuple[NaiPicPlugin, _RecordingSend]:
    plugin = object.__new__(NaiPicPlugin)
    send = _RecordingSend()
    plugin.ctx = SimpleNamespace(send=send)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._reverse_service = SimpleNamespace(reverse=reverse)
    plugin._image_cache_service = SimpleNamespace(
        resolve_image_base64=lambda **_kwargs: base64.b64encode(b"test-image").decode("ascii")
    )
    plugin.get_plugin_config_data = lambda: {"retag": {"enabled": True}}
    return plugin, send


@pytest.mark.parametrize("scheduling_seed", [0, 7, 42])
def test_retag_command_returns_before_slow_reverse_finishes(scheduling_seed: int) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def reverse(image_bytes: bytes, **kwargs: Any) -> ReverseResult:
            calls.append(image_bytes)
            started.set()
            await release.wait()
            for _ in range(random.Random(scheduling_seed).randrange(10)):
                await asyncio.sleep(0)
            return ReverseResult(source="wd14", prompt="1girl, smile", tags=["1girl", "smile"])

        plugin, send = _build_plugin(reverse)
        try:
            # 用短 RPC 截止时间复现宿主的 60s 上限；后端只在命令返回后放行。
            result = await asyncio.wait_for(
                plugin.handle_nai_retag_command(stream_id="retag-rpc", user_id="u1"),
                timeout=0.1,
            )
            assert result == (True, "已开始反推", True)
            await asyncio.wait_for(started.wait(), timeout=1)
            assert send.finished.is_set() is False
            assert calls == [b"test-image"]
            release.set()
            await asyncio.wait_for(send.finished.wait(), timeout=1)
            assert send.messages == [
                ("retag-rpc", "🔍 正在反推 tag，请稍候..."),
                (
                    "retag-rpc",
                    "✅ 反推完成（🔍 WD14 在线 Space，2 个 tag）\n\n"
                    "1girl, smile\n\n💡 可直接用于 /nai0 <prompt>",
                ),
            ]
        finally:
            release.set()
            await plugin._background_tasks.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("unexpected", [False, True])
def test_retag_background_failure_is_reported_to_original_stream(unexpected: bool) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def reverse(image_bytes: bytes, **kwargs: Any) -> ReverseResult:
            await release.wait()
            if unexpected:
                raise RuntimeError("backend unavailable")
            return ReverseResult(source="failed", prompt="", detail="WD14 调用超时")

        plugin, send = _build_plugin(reverse)
        try:
            result = await asyncio.wait_for(
                plugin.handle_nai_retag_command(stream_id="retag-failure", user_id="u1"),
                timeout=0.1,
            )
            assert result == (True, "已开始反推", True)
            release.set()
            await asyncio.wait_for(send.finished.wait(), timeout=1)
            errors = [(stream, text) for stream, text in send.messages if text.startswith("❌")]
            assert len(errors) == 1
            assert errors[0][0] == "retag-failure"
            assert ("backend unavailable" if unexpected else "WD14 调用超时") in errors[0][1]
            assert all(not text.startswith("✅") for _, text in send.messages)
        finally:
            release.set()
            await plugin._background_tasks.shutdown()

    asyncio.run(scenario())


def test_retag_command_rejects_new_work_after_shutdown() -> None:
    async def scenario() -> None:
        calls = []

        async def reverse(image_bytes: bytes, **kwargs: Any) -> ReverseResult:
            calls.append(image_bytes)
            return ReverseResult(source="wd14", prompt="smile", tags=["smile"])

        plugin, send = _build_plugin(reverse)
        await plugin._background_tasks.shutdown()
        result = await plugin.handle_nai_retag_command(stream_id="retag-closed", user_id="u1")
        assert result == (False, "插件正在卸载，无法启动反推", True)
        assert calls == []
        assert send.messages == []

    asyncio.run(scenario())
