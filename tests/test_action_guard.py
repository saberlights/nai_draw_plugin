"""Action Guard 单测：覆盖用户原话取词 + 关键词分级 + reasoning fallback。"""

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))


# ---- 上游依赖打桩（与现有 test_sdk_runtime_tag_retriever 保持一致） ----
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


def _get_logger(_name=None):
    return _DummyLogger()


dummy_logger_module.get_logger = _get_logger
sys.modules["src.common.logger"] = dummy_logger_module

src_package = types.ModuleType("src")
src_package.__path__ = [os.path.join(MAIBOT_ROOT, "src")]
sys.modules.setdefault("src", src_package)

src_config_package = types.ModuleType("src.config")
src_config_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "config")]
sys.modules.setdefault("src.config", src_config_package)

config_module = types.ModuleType("src.config.config")
config_module.global_config = types.SimpleNamespace(
    bot=types.SimpleNamespace(qq_account="999", platforms=[])
)
config_module.model_config = types.SimpleNamespace(
    model_task_config=types.SimpleNamespace(embedding=None)
)
sys.modules["src.config.config"] = config_module

model_configs_module = types.ModuleType("src.config.model_configs")
model_configs_module.TaskConfig = type("TaskConfig", (), {})
sys.modules["src.config.model_configs"] = model_configs_module

src_llm_models_package = types.ModuleType("src.llm_models")
src_llm_models_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "llm_models")]
sys.modules.setdefault("src.llm_models", src_llm_models_package)

utils_model_module = types.ModuleType("src.llm_models.utils_model")


class _DummyLLMOrchestrator:
    def __init__(self, *args, **kwargs):
        self.model_for_task = None
        self.model_usage = {}


utils_model_module.LLMOrchestrator = _DummyLLMOrchestrator
sys.modules["src.llm_models.utils_model"] = utils_model_module

src_services_module = types.ModuleType("src.services")
src_services_module.llm_service = types.SimpleNamespace()
sys.modules["src.services"] = src_services_module

tag_retriever_module = types.ModuleType("plugins.nai_draw_plugin.core.services.tag_retriever")
tag_retriever_module.get_tag_retriever = lambda **_kwargs: None
sys.modules.setdefault("plugins.nai_draw_plugin.core.services.tag_retriever", tag_retriever_module)

# 与 test_sdk_runtime_tag_retriever 一致：截断 core.mixins 包的 __init__，
# 避免间接加载 auto_recall_mixin → src.chat 的重链路
mixins_package = types.ModuleType("plugins.nai_draw_plugin.core.mixins")
mixins_package.__path__ = [os.path.join(MAIBOT_ROOT, "plugins", "nai_draw_plugin", "core", "mixins")]
sys.modules.setdefault("plugins.nai_draw_plugin.core.mixins", mixins_package)

from plugins.nai_draw_plugin import sdk_runtime as sdk_runtime_module
from plugins.nai_draw_plugin.core.services.session_state import session_state
from plugins.nai_draw_plugin.sdk_runtime import (
    NaiInvocation,
    _reasoning_implies_explicit_request,
)


def _build_invocation(*, stream_id: str = "test-stream") -> NaiInvocation:
    invocation = object.__new__(NaiInvocation)
    invocation.plugin_config = {
        "action_guard": {
            "enabled": True,
            "explicit_request_min_interval_seconds": 45,
            "proactive_min_interval_seconds": 240,
            "weak_negative_ttl_seconds": 60,
        },
        "auto_draw_on_reply": {
            "min_interval_seconds": 180,
        },
    }
    invocation.stream_id = stream_id
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    return invocation


def _reset_interval_state(stream_id: str) -> None:
    session_state._last_action_image_sent_at.pop(stream_id, None)
    session_state._last_auto_draw_sent_at.pop(stream_id, None)


def _wrap_aged(text_fetcher):
    """把"返回字符串"的旧式 fake 包装成 ``_fetch_last_user_text_with_age`` 兼容的形式。

    旧测试都不关心 staleness，所以 age 用 None 表示未知；走到弱否定时 None 会被保守
    阻断，但旧测试场景里没有弱否定单独命中（要么 explicit、要么 strong 否定）。
    """

    async def _aged(self, *, lookback: int = 6):
        text = await text_fetcher(self, lookback=lookback)
        return text, None

    return _aged


# ==================== _reasoning_implies_explicit_request ====================


def test_reasoning_fallback_recognizes_user_request_phrases() -> None:
    assert _reasoning_implies_explicit_request("用户要求看一张自拍") is True
    assert _reasoning_implies_explicit_request("对方想看你今天的穿搭") is True
    assert _reasoning_implies_explicit_request("用户让我画初音未来") is True
    assert _reasoning_implies_explicit_request("用户追图，要求再来一张") is True


def test_reasoning_fallback_ignores_pure_self_description() -> None:
    # 纯 bot 自身视角的视觉描述不应升级到 explicit
    assert _reasoning_implies_explicit_request("我正坐在窗边，光线很柔和，配图比文字更自然") is False
    assert _reasoning_implies_explicit_request("当前场景适合用一张配图带过") is False
    assert _reasoning_implies_explicit_request("") is False


# ==================== _assess_action_trigger ====================


def test_guard_uses_user_text_explicit_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)

    async def fake_fetch(self, *, lookback: int = 6):
        return "再来一张自拍"

    monkeypatch.setattr(NaiInvocation, "_fetch_last_user_text_with_age", _wrap_aged(fake_fetch))

    result = asyncio.run(invocation._assess_action_trigger(reasoning="bot 觉得该配图"))

    assert result["should_generate"] is True
    assert result["category"] == "explicit"
    assert result["explicit_request"] is True
    assert result["signal_source"] == "user_text"


def test_guard_uses_user_text_neutral_falls_to_proactive(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)

    async def fake_fetch(self, *, lookback: int = 6):
        return "今天天气真不错"

    monkeypatch.setattr(NaiInvocation, "_fetch_last_user_text_with_age", _wrap_aged(fake_fetch))

    result = asyncio.run(invocation._assess_action_trigger(reasoning="用户要求看图"))

    # 即便 reasoning 里写了"用户要求"，只要拿到了原话且原话不含强信号，仍按 proactive
    assert result["category"] == "proactive"
    assert result["signal_source"] == "user_text"


def test_guard_negative_intent_in_user_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)

    async def fake_fetch(self, *, lookback: int = 6):
        return "别给我画图，文字回复就行"

    monkeypatch.setattr(NaiInvocation, "_fetch_last_user_text_with_age", _wrap_aged(fake_fetch))

    result = asyncio.run(invocation._assess_action_trigger(reasoning=""))

    assert result["should_generate"] is False
    assert result["category"] == "blocked"


def test_guard_falls_back_to_reasoning_when_user_text_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)

    async def fake_fetch(self, *, lookback: int = 6):
        return ""

    monkeypatch.setattr(NaiInvocation, "_fetch_last_user_text_with_age", _wrap_aged(fake_fetch))

    result = asyncio.run(invocation._assess_action_trigger(reasoning="用户要求看你的穿搭"))

    assert result["category"] == "explicit"
    assert result["signal_source"] == "reasoning"


def test_guard_no_signal_anywhere_defaults_to_proactive(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)

    async def fake_fetch(self, *, lookback: int = 6):
        return ""

    monkeypatch.setattr(NaiInvocation, "_fetch_last_user_text_with_age", _wrap_aged(fake_fetch))

    result = asyncio.run(invocation._assess_action_trigger(reasoning="此刻配图比文字更自然"))

    assert result["category"] == "proactive"
    assert result["explicit_request"] is False


def test_guard_interval_block_after_recent_send(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation(stream_id="recent-send-stream")
    _reset_interval_state(invocation.stream_id)
    # 模拟刚刚发过图：1 秒前
    session_state.set_last_action_image_sent_at(invocation.stream_id, sent_at=__import__("time").time() - 1.0)

    async def fake_fetch(self, *, lookback: int = 6):
        return "再来一张"  # explicit 信号

    monkeypatch.setattr(NaiInvocation, "_fetch_last_user_text_with_age", _wrap_aged(fake_fetch))

    result = asyncio.run(invocation._assess_action_trigger(reasoning=""))

    assert result["category"] == "explicit"
    assert result["should_generate"] is False
    assert "等待" in result["detail"]


# ==================== _fetch_last_user_text ====================


def test_fetch_last_user_text_skips_bot_and_images(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()

    async def fake_recent(self, *, limit: int = 120, hours: float = 24.0):
        return [
            {"user_id": "999", "processed_plain_text": "[NAI图片]"},  # bot 自己发的图
            {"user_id": "999", "processed_plain_text": "我刚发了张图"},  # bot 自己的文字
            {"user_id": "user-1", "processed_plain_text": "[图片消息]"},  # 用户发的是图
            {"user_id": "user-1", "processed_plain_text": "再来一张呀"},  # 用户原话
            {"user_id": "user-1", "processed_plain_text": "前面更早的消息"},
        ]

    monkeypatch.setattr(NaiInvocation, "_find_recent_messages", fake_recent)
    monkeypatch.setattr(NaiInvocation, "_get_target_platform", lambda self: "qq")
    # 显式打桩 bot_account，避免 sibling 测试改写 global_config 后串扰
    monkeypatch.setattr(sdk_runtime_module, "_resolve_bot_account", lambda platform: "999")

    text = asyncio.run(invocation._fetch_last_user_text())
    assert text == "再来一张呀"


def test_fetch_last_user_text_returns_empty_when_no_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()

    async def fake_recent(self, *, limit: int = 120, hours: float = 24.0):
        return [
            {"user_id": "999", "processed_plain_text": "[NAI图片]"},
            {"user_id": "999", "processed_plain_text": "我又发了一张"},
        ]

    monkeypatch.setattr(NaiInvocation, "_find_recent_messages", fake_recent)
    monkeypatch.setattr(NaiInvocation, "_get_target_platform", lambda self: "qq")
    monkeypatch.setattr(sdk_runtime_module, "_resolve_bot_account", lambda platform: "999")

    text = asyncio.run(invocation._fetch_last_user_text())
    assert text == ""


# ==================== strong / weak negative + staleness ====================


def _patch_user_text_with_age(monkeypatch: pytest.MonkeyPatch, text: str, age: float | None) -> None:
    async def fake(self, *, lookback: int = 6):
        return text, age

    monkeypatch.setattr(NaiInvocation, "_fetch_last_user_text_with_age", fake)


def test_guard_strong_negative_blocks_even_if_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)
    _patch_user_text_with_age(monkeypatch, "别画了", age=3600.0)  # 一小时前说的"别画了"

    result = asyncio.run(invocation._assess_action_trigger(reasoning=""))
    assert result["should_generate"] is False
    assert result["category"] == "blocked"


def test_guard_weak_negative_blocks_when_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)
    _patch_user_text_with_age(monkeypatch, "用文字给我讲", age=10.0)

    result = asyncio.run(invocation._assess_action_trigger(reasoning=""))
    assert result["should_generate"] is False
    assert result["category"] == "blocked"
    assert "文字" in result["detail"]


def test_guard_weak_negative_passes_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)
    # TTL 默认 60s，这里 age=120s 已过期
    _patch_user_text_with_age(monkeypatch, "用文字给我讲", age=120.0)

    result = asyncio.run(invocation._assess_action_trigger(reasoning=""))
    assert result["should_generate"] is True  # 弱否定失效，但用户原话又不是 explicit
    assert result["category"] == "proactive"


def test_guard_weak_negative_age_unknown_blocks_conservatively(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation()
    _reset_interval_state(invocation.stream_id)
    _patch_user_text_with_age(monkeypatch, "文字就行", age=None)

    result = asyncio.run(invocation._assess_action_trigger(reasoning=""))
    # age 未知时按"未过期"保守阻断
    assert result["should_generate"] is False
    assert result["category"] == "blocked"


# ==================== auto_draw 间隔门：独立计时，max(action, auto_draw) ====================


def test_auto_draw_interval_first_time_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation(stream_id="auto-first")
    _reset_interval_state(invocation.stream_id)

    can_send, _detail = invocation._check_action_image_interval("auto_draw")
    assert can_send is True


def test_auto_draw_blocked_when_explicit_recent(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation(stream_id="auto-after-explicit")
    _reset_interval_state(invocation.stream_id)
    import time as _time
    # explicit 路径 30s 前发过图：auto_draw 间隔 180s 应该还阻塞
    session_state.set_last_action_image_sent_at(invocation.stream_id, _time.time() - 30.0)

    can_send, detail = invocation._check_action_image_interval("auto_draw")
    assert can_send is False
    assert "等待" in detail


def test_auto_draw_does_not_block_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation(stream_id="explicit-after-auto")
    _reset_interval_state(invocation.stream_id)
    import time as _time
    # auto_draw 1s 前发过图：explicit 路径应该不受 auto_draw 影响
    session_state.set_last_auto_draw_sent_at(invocation.stream_id, _time.time() - 1.0)

    can_send, _detail = invocation._check_action_image_interval("explicit")
    assert can_send is True  # explicit 只看 last_action_image_sent_at


# ==================== _inject_self_image_hint ====================


def test_inject_self_image_hint_adds_prefix_when_no_persona() -> None:
    out = sdk_runtime_module._inject_self_image_hint("窗边 慵懒 近景", mode="portrait")
    assert out.startswith("一女")
    assert "肖像照" in out
    assert "窗边" in out


def test_inject_self_image_hint_keeps_existing_persona() -> None:
    out = sdk_runtime_module._inject_self_image_hint("一女 自拍 沙发", mode="selfie")
    # 不应该重复堆叠"一女"
    assert out.count("一女") == 1


def test_inject_self_image_hint_for_scene_mode() -> None:
    out = sdk_runtime_module._inject_self_image_hint("便利店 收银台", mode="scene")
    assert "生活照" in out
    assert "便利店" in out


def test_inject_self_image_hint_empty_description() -> None:
    out = sdk_runtime_module._inject_self_image_hint("", mode="portrait")
    assert "肖像照" in out
    assert "一女" in out

