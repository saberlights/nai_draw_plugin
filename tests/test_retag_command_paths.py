# -*- coding: utf-8 -*-
"""`/nai 反推` 与 `/nai 反推路径` 两条命令的行为测试。

锁的是命令层做的三件事：路径优先级（单次覆盖 > 会话粘性 > 配置默认）、
把会话 NSFW 开关透给视觉模型、以及按结果形态给出正确的回执文案。

复用兄弟测试里的 maibot_sdk / src.* stub 链（import 顺序敏感，必须先 import 它）。
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image as PILImage

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

# 先跑 stub 装配，再 import plugin（与 test_retag_config_layout 同一套）
from plugins.nai_draw_plugin.tests import test_retag_config_layout as _stub_bootstrap  # noqa: F401,E402

from plugins.nai_draw_plugin.core.retag.reverser import (  # noqa: E402
    KIND_NATURAL_LANGUAGE,
    KIND_TAGS,
    ReverseResult,
)
from plugins.nai_draw_plugin.core.services.background_task_supervisor import (  # noqa: E402
    BackgroundTaskSupervisor,
)
from plugins.nai_draw_plugin.core.services.session_state import session_state  # noqa: E402
from plugins.nai_draw_plugin import plugin as plugin_module  # noqa: E402
from plugins.nai_draw_plugin.plugin import NaiPicPlugin  # noqa: E402


class _RecordingSend:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def text(self, text: str, stream_id: str, storage_message: bool = True) -> bool:
        del stream_id, storage_message
        self.texts.append(text)
        return True

    @property
    def joined(self) -> str:
        return "\n".join(self.texts)


class _RecordingReverseService:
    """记录 reverse() 收到的 path / sfw，并按脚本返回结果。"""

    def __init__(self, result: ReverseResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def reverse(self, image_bytes: bytes, *, path: str = "auto", sfw: bool = True):
        self.calls.append({"path": path, "sfw": sfw, "size": len(image_bytes)})
        return self.result


def _png_base64() -> str:
    buffer = io.BytesIO()
    PILImage.new("RGB", (64, 64), (10, 20, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_plugin(
    *,
    result: ReverseResult,
    retag_config: dict[str, Any] | None = None,
) -> tuple[NaiPicPlugin, _RecordingSend, _RecordingReverseService]:
    plugin = object.__new__(NaiPicPlugin)
    send = _RecordingSend()
    reverse_service = _RecordingReverseService(result)

    plugin.ctx = types.SimpleNamespace(send=send)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._reverse_service = reverse_service
    plugin._image_cache_service = types.SimpleNamespace(
        resolve_image_base64=lambda **_kwargs: _png_base64()
    )
    plugin.get_plugin_config_data = lambda: {"retag": {"enabled": True, **(retag_config or {})}}
    return plugin, send, reverse_service


_TAG_RESULT = ReverseResult(
    source="wd14",
    prompt="1girl, smile",
    tags=["1girl", "smile"],
    kind=KIND_TAGS,
)

_CAPTION = (
    "A young woman sits on a rooftop railing, her ash-blonde hair swept to the left "
    "by the wind while her half-lidded grey eyes glance down and away from the viewer."
)
_LLM_RESULT = ReverseResult(
    source="llm",
    prompt=_CAPTION,
    tags=[],
    kind=KIND_NATURAL_LANGUAGE,
)


@pytest.fixture(autouse=True)
def _clear_session():
    session_state.clear_session_state("stream", "s-retag-cmd")
    yield
    session_state.clear_session_state("stream", "s-retag-cmd")


# ── 路径优先级 ────────────────────────────────────────────────────────────


def test_retag_uses_config_default_path_when_nothing_selected() -> None:
    plugin, _send, svc = _build_plugin(
        result=_TAG_RESULT,
        retag_config={"default_path": "wd14"},
    )

    asyncio.run(plugin._run_retag(stream_id="s-retag-cmd", user_id="u1"))

    assert svc.calls[0]["path"] == "wd14"


def test_sticky_session_path_beats_config_default() -> None:
    plugin, _send, svc = _build_plugin(
        result=_LLM_RESULT,
        retag_config={"default_path": "wd14"},
    )
    session_state.set_retag_path("stream", "s-retag-cmd", "llm")

    asyncio.run(plugin._run_retag(stream_id="s-retag-cmd", user_id="u1"))

    assert svc.calls[0]["path"] == "llm"


def test_one_shot_override_beats_sticky_and_leaves_it_untouched() -> None:
    """`/nai 反推 meta` 只对本次生效，粘性选定不能被顺手改掉。"""
    plugin, _send, svc = _build_plugin(
        result=_TAG_RESULT,
        retag_config={"default_path": "auto"},
    )
    session_state.set_retag_path("stream", "s-retag-cmd", "llm")

    asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="meta")
    )

    assert svc.calls[0]["path"] == "meta"
    assert (
        session_state.get_retag_path("stream", "s-retag-cmd", lambda *_a, **_k: "auto")
        == "llm"
    )


def test_illegal_one_shot_path_is_rejected_before_touching_the_backend() -> None:
    plugin, send, svc = _build_plugin(result=_TAG_RESULT)

    ok, _reason, _handled = asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="vlm")
    )

    assert ok is False
    assert svc.calls == []
    assert "vlm" in send.joined
    assert "/nai 反推路径 llm" in send.joined, "错误提示要顺带列出可选值"


def test_disabled_retag_short_circuits() -> None:
    plugin, send, svc = _build_plugin(result=_TAG_RESULT, retag_config={"enabled": False})

    ok, _reason, _handled = asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1")
    )

    assert ok is False
    assert svc.calls == []
    assert "已在配置中关闭" in send.joined


# ── NSFW 开关透传 ─────────────────────────────────────────────────────────


def test_nsfw_filter_state_is_forwarded_as_sfw_flag() -> None:
    plugin, _send, svc = _build_plugin(result=_LLM_RESULT)
    session_state.set_nsfw_filter_enabled("stream", "s-retag-cmd", True)

    asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="llm")
    )
    assert svc.calls[0]["sfw"] is True

    session_state.set_nsfw_filter_enabled("stream", "s-retag-cmd", False)
    asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="llm")
    )
    assert svc.calls[1]["sfw"] is False


# ── 回执文案按结果形态分支 ────────────────────────────────────────────────


def test_tag_result_reports_tag_count() -> None:
    plugin, send, _svc = _build_plugin(result=_TAG_RESULT)

    asyncio.run(plugin._run_retag(stream_id="s-retag-cmd", user_id="u1"))

    assert "2 个 tag" in send.joined
    assert "1girl, smile" in send.joined
    assert "词）" not in send.joined


def test_natural_language_result_reports_word_count_not_tags() -> None:
    """对一段 20+ 词的描述说"0 个 tag"是错的，用户会以为反推失败。"""
    plugin, send, _svc = _build_plugin(result=_LLM_RESULT)

    asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="llm")
    )

    assert f"{len(_CAPTION.split())} 词" in send.joined
    assert "个 tag" not in send.joined
    assert _CAPTION in send.joined


def test_natural_language_result_warns_about_tag_only_fields() -> None:
    """纯自然语言表达不了 rating / 人数 / 视角，回执必须说清楚，否则用户以为反推漏了。"""
    plugin, send, _svc = _build_plugin(result=_LLM_RESULT)

    asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="llm")
    )

    assert "rating / 人数 / 视角" in send.joined
    assert "V5" in send.joined


def test_llm_path_announces_its_longer_wait() -> None:
    plugin, send, _svc = _build_plugin(result=_LLM_RESULT)

    asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="llm")
    )

    assert "视觉模型" in send.texts[0]


def test_failure_message_names_the_path_and_the_reason() -> None:
    plugin, send, _svc = _build_plugin(
        result=ReverseResult(source="failed", prompt="", detail="视觉模型调用超时（120s）")
    )

    ok, _reason, _handled = asyncio.run(
        plugin._run_retag(stream_id="s-retag-cmd", user_id="u1", path_override="llm")
    )

    assert ok is False
    assert "路径 llm" in send.joined
    assert "视觉模型调用超时（120s）" in send.joined


# ── /nai 反推路径 ─────────────────────────────────────────────────────────


def test_path_switch_without_argument_shows_current_and_options() -> None:
    plugin, send, _svc = _build_plugin(
        result=_TAG_RESULT,
        retag_config={"default_path": "meta"},
    )

    ok, _reason, _handled = asyncio.run(
        plugin._run_retag_path_switch(stream_id="s-retag-cmd", user_id="u1", path_value="")
    )

    assert ok is True
    assert "当前反推路径：meta" in send.joined
    for path in ("auto", "meta", "wd14", "llm"):
        assert f"/nai 反推路径 {path}" in send.joined


def test_path_switch_persists_selection_for_the_session() -> None:
    plugin, send, _svc = _build_plugin(result=_LLM_RESULT)

    ok, _reason, _handled = asyncio.run(
        plugin._run_retag_path_switch(
            stream_id="s-retag-cmd", user_id="u1", path_value="LLM"
        )
    )

    assert ok is True
    assert "llm" in send.joined
    assert (
        session_state.get_retag_path("stream", "s-retag-cmd", lambda *_a, **_k: "auto")
        == "llm"
    )


def test_path_switch_rejects_unknown_value() -> None:
    plugin, send, _svc = _build_plugin(result=_TAG_RESULT)
    session_state.set_retag_path("stream", "s-retag-cmd", "meta")

    ok, _reason, _handled = asyncio.run(
        plugin._run_retag_path_switch(
            stream_id="s-retag-cmd", user_id="u1", path_value="wd_14"
        )
    )

    assert ok is False
    assert "wd_14" in send.joined
    # 切换失败不能把原有选定改坏
    assert (
        session_state.get_retag_path("stream", "s-retag-cmd", lambda *_a, **_k: "auto")
        == "meta"
    )
