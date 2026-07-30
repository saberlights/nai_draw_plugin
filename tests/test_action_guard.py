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
tag_retriever_module.reset_tag_retriever = lambda: None
sys.modules.setdefault("plugins.nai_draw_plugin.core.services.tag_retriever", tag_retriever_module)

mixins_package = types.ModuleType("plugins.nai_draw_plugin.core.mixins")
mixins_package.__path__ = [os.path.join(MAIBOT_ROOT, "plugins", "nai_draw_plugin", "core", "mixins")]
sys.modules.setdefault("plugins.nai_draw_plugin.core.mixins", mixins_package)

from plugins.nai_draw_plugin import sdk_runtime as sdk_runtime_module  # noqa: E402
from plugins.nai_draw_plugin.core.services.generation_admission_policy import (  # noqa: E402
    reasoning_implies_explicit_request,
)
from plugins.nai_draw_plugin.core.services.visual_continuity import (  # noqa: E402
    StableVisualTags,
)
from plugins.nai_draw_plugin.sdk_runtime import NaiInvocation  # noqa: E402


def _build_invocation(*, stream_id: str = "test-stream") -> NaiInvocation:
    invocation = object.__new__(NaiInvocation)
    invocation.stream_id = stream_id
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    invocation._pending_visual_continuity = None
    return invocation


# ==================== reasoning_implies_explicit_request ====================


def test_reasoning_fallback_recognizes_user_request_phrases() -> None:
    assert reasoning_implies_explicit_request("用户要求看一张自拍") is True
    assert reasoning_implies_explicit_request("对方想看你今天的穿搭") is True
    assert reasoning_implies_explicit_request("用户让我画初音未来") is True
    assert reasoning_implies_explicit_request("用户追图，要求再来一张") is True


def test_reasoning_fallback_ignores_pure_self_description() -> None:
    # 纯 bot 自身视角的视觉描述不应升级到 explicit
    assert reasoning_implies_explicit_request("我正坐在窗边，光线很柔和，配图比文字更自然") is False
    assert reasoning_implies_explicit_request("当前场景适合用一张配图带过") is False
    assert reasoning_implies_explicit_request("") is False


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


# ==================== _inject_self_image_hint ====================


def test_inject_self_image_hint_adds_prefix_when_no_persona() -> None:
    out = sdk_runtime_module._inject_self_image_hint("窗边 慵懒", mode="portrait")
    assert out.startswith("一女")
    assert "肖像照" in out
    assert "窗边" in out
    assert not out.endswith("近景")


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
    assert "近景" not in out


def test_action_third_person_bot_scene_still_injects_configured_appearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation = _build_invocation()
    invocation.action_data = {
        "subject_and_pov": "一女 第三视角",
        "action": "坐在窗边读书",
        "emotion": "放松",
        "scene_delta": "",
        "framing": "中景",
    }
    invocation.reasoning = ""
    processed_prompts: list[str] = []
    prompt_kwargs: list[dict[str, object]] = []

    async def allowed() -> bool:
        return True

    async def approved_trigger(**_kwargs):
        return types.SimpleNamespace(
            should_generate=True,
            category="explicit",
            detail="允许",
            signal_source="test",
            signal_text="test",
        )

    async def generated_prompt(*_args, **kwargs):
        prompt_kwargs.append(kwargs)
        return "rating:general, solo, 1girl, reading, medium shot", None

    def process_selfie(description: str, *_args, **_kwargs) -> str:
        processed_prompts.append(description)
        return f"{description}, configured bot appearance"

    async def generated_image(**_kwargs):
        return True, "image-data"

    async def sent_image(*_args, **_kwargs):
        return True, "sent"

    invocation.ensure_user_not_blacklisted = allowed
    invocation.ensure_generation_permission = allowed
    invocation._assess_action_trigger = approved_trigger
    invocation._is_action_guard_enabled = lambda: True
    invocation._generate_prompt_with_llm = generated_prompt
    invocation._process_selfie_prompt = process_selfie
    invocation._sanitize_prompt_for_sfw_mode = lambda text: text
    invocation._sanitize_structured_for_sfw_mode = lambda payload: payload
    invocation._select_send_payload = lambda prompt, payload: (prompt, None)
    invocation._is_prompt_show_enabled = lambda: False
    invocation._get_model_config = lambda **_kwargs: {
        "base_url": "https://example.invalid",
        "nai_size": "竖图",
    }
    invocation._send_image_result = sent_image
    invocation.get_config = lambda key, default=None: {
        "prompt_generator.enforce_tag_order": False,
        "prompt_generator.inherit_ttl": 3_600,
        "components.enable_debug_info": False,
    }.get(key, default)
    invocation.api_client = types.SimpleNamespace(generate_image=generated_image)
    monkeypatch.setattr(
        sdk_runtime_module.session_state,
        "set_last_selfie_context",
        lambda *_args, **_kwargs: None,
    )

    ok, detail = asyncio.run(invocation.handle_action())

    assert (ok, detail) == (True, "sent")
    assert processed_prompts == [
        "rating:general, solo, 1girl, reading, medium shot"
    ]
    assert prompt_kwargs[0]["use_visual_continuity"] is True


def test_action_stops_when_prompt_generation_fails_instead_of_sending_selfie_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tag LLM 被拒绝后不得把中文清空成仅含 selfie_prompt_add 的请求。"""

    invocation = _build_invocation()
    invocation.action_data = {
        "subject_and_pov": "一女 第三视角",
        "action": "侧坐在折叠椅上，一只手撑着后腰",
        "emotion": "倦怠",
        "scene_delta": "淡青色丝绸旗袍，肉色丝袜，复古黑色皮鞋；凌乱片场",
        "framing": "中景",
    }
    invocation.reasoning = ""
    api_called = False
    sent_texts: list[str] = []

    async def allowed() -> bool:
        return True

    async def approved_trigger(**_kwargs):
        return types.SimpleNamespace(
            should_generate=True,
            category="explicit",
            detail="允许",
            signal_source="test",
            signal_text="test",
        )

    async def failed_prompt(*_args, **_kwargs):
        return None

    async def generated_image(**_kwargs):
        nonlocal api_called
        api_called = True
        return True, "image-data"

    async def sent_image(*_args, **_kwargs):
        return True, "sent"

    async def sent_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    invocation.ensure_user_not_blacklisted = allowed
    invocation.ensure_generation_permission = allowed
    invocation._assess_action_trigger = approved_trigger
    invocation._is_action_guard_enabled = lambda: True
    invocation._generate_prompt_with_llm = failed_prompt
    invocation._process_selfie_prompt = lambda *_args, **_kwargs: "configured bot appearance"
    invocation._sanitize_prompt_for_sfw_mode = lambda text: text
    invocation._sanitize_structured_for_sfw_mode = lambda payload: payload
    invocation._select_send_payload = lambda prompt, payload: (prompt, None)
    invocation._is_prompt_show_enabled = lambda: False
    invocation._get_model_config = lambda **_kwargs: {
        "base_url": "https://example.invalid",
        "nai_size": "竖图",
    }
    invocation.get_config = lambda key, default=None: {
        "prompt_generator.enforce_tag_order": False,
        "components.enable_debug_info": False,
    }.get(key, default)
    invocation.api_client = types.SimpleNamespace(generate_image=generated_image)
    invocation._send_image_result = sent_image
    invocation.send_text = sent_text
    monkeypatch.setattr(
        sdk_runtime_module.session_state,
        "set_last_selfie_context",
        lambda *_args, **_kwargs: None,
    )

    ok, detail = asyncio.run(invocation.handle_action())

    assert ok is False
    assert "提示词生成" in detail
    assert api_called is False
    assert sent_texts == ["提示词生成器开小差了，请直接告诉我想画什么，或者稍后再试一次~"]


@pytest.mark.parametrize(
    ("send_success", "expected_commit_count"),
    ((True, 1), (False, 0)),
)
def test_action_commits_visual_continuity_only_after_image_delivery(
    monkeypatch: pytest.MonkeyPatch,
    send_success: bool,
    expected_commit_count: int,
) -> None:
    invocation = _build_invocation()
    invocation.action_data = {
        "subject_and_pov": "一女 第三视角",
        "action": "站在片场中央",
        "emotion": "平静",
        "scene_delta": "淡青色丝绸旗袍；片场固定灯架和黑色遮光布",
        "framing": "全身",
    }
    invocation.reasoning = ""
    candidate = StableVisualTags(
        outfit=("pale cyan silk qipao", "high side slit", "black leather shoes"),
        environment=("film set", "light stand", "black blackout cloth"),
        outfit_key="pale_cyan_qipao_set",
        environment_key="film_set",
    )
    committed: list[StableVisualTags] = []

    async def allowed() -> bool:
        return True

    async def approved_trigger(**_kwargs):
        return types.SimpleNamespace(
            should_generate=True,
            category="explicit",
            detail="允许",
            signal_source="test",
            signal_text="test",
        )

    async def generated_prompt(*_args, **kwargs):
        assert kwargs["stable_change_text"] == invocation.action_data["scene_delta"]
        invocation._pending_visual_continuity = candidate
        return "rating:general, solo, 1girl, pale cyan silk qipao, film set", None

    async def generated_image(**_kwargs):
        return True, "image-data"

    async def sent_image(*_args, **_kwargs):
        return send_success, "sent" if send_success else "send failed"

    invocation.ensure_user_not_blacklisted = allowed
    invocation.ensure_generation_permission = allowed
    invocation._assess_action_trigger = approved_trigger
    invocation._is_action_guard_enabled = lambda: True
    invocation._generate_prompt_with_llm = generated_prompt
    invocation._process_selfie_prompt = lambda text, *_args, **_kwargs: text
    invocation._sanitize_prompt_for_sfw_mode = lambda text: text
    invocation._sanitize_structured_for_sfw_mode = lambda payload: payload
    invocation._select_send_payload = lambda prompt, payload: (prompt, None)
    invocation._is_prompt_show_enabled = lambda: False
    invocation._get_model_config = lambda **_kwargs: {
        "base_url": "https://example.invalid",
        "nai_size": "竖图",
    }
    invocation.get_config = lambda key, default=None: {
        "prompt_generator.enforce_tag_order": False,
        "prompt_generator.inherit_ttl": 3_600,
        "components.enable_debug_info": False,
    }.get(key, default)
    invocation.api_client = types.SimpleNamespace(generate_image=generated_image)
    invocation._send_image_result = sent_image
    monkeypatch.setattr(
        sdk_runtime_module.session_state,
        "set_visual_continuity",
        lambda _stream_id, stable: committed.append(stable),
    )
    monkeypatch.setattr(
        sdk_runtime_module.session_state,
        "set_last_selfie_context",
        lambda *_args, **_kwargs: None,
    )

    ok, _detail = asyncio.run(invocation.handle_action())

    assert ok is send_success
    assert committed == [candidate] * expected_commit_count
