import asyncio
import base64
import os
import sys
import types
from pathlib import Path

import pytest

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
config_module.global_config = types.SimpleNamespace()
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

mixins_package = types.ModuleType("plugins.nai_draw_plugin.core.mixins")
mixins_package.__path__ = [os.path.join(MAIBOT_ROOT, "plugins", "nai_draw_plugin", "core", "mixins")]
sys.modules.setdefault("plugins.nai_draw_plugin.core.mixins", mixins_package)

tag_retriever_module = types.ModuleType("plugins.nai_draw_plugin.core.services.tag_retriever")
tag_retriever_module.get_tag_retriever = lambda **_kwargs: None
sys.modules.setdefault("plugins.nai_draw_plugin.core.services.tag_retriever", tag_retriever_module)

from plugins.nai_draw_plugin.core.services import tag_candidate_resolver as resolver_module
from plugins.nai_draw_plugin.core.tag_retriever_display import build_tag_retriever_display_message
from plugins.nai_draw_plugin.core.tag_retriever_display import build_tag_retriever_display_messages
from plugins.nai_draw_plugin.sdk_runtime import NaiInvocation
from plugins.nai_draw_plugin import sdk_runtime as sdk_runtime_module


class _FakeOnlineRetriever:
    def __init__(self, results: dict[str, list[dict[str, object]]], formatted: str) -> None:
        self.results = results
        self.formatted = formatted
        self.queries: list[str] = []

    async def retrieve(self, query: str) -> dict[str, list[dict[str, object]]]:
        self.queries.append(query)
        return self.results

    def format_candidates(self, results: dict[str, list[dict[str, object]]]) -> str:
        assert results == self.results
        return self.formatted


class _FakeLocalRetriever:
    def __init__(self, results: list[dict[str, object]], formatted: str) -> None:
        self.results = results
        self.formatted = formatted
        self.calls: list[tuple[str, int, float]] = []

    async def retrieve(self, query: str, top_k: int, min_score: float) -> list[dict[str, object]]:
        self.calls.append((query, top_k, min_score))
        return self.results

    def format_candidates(self, results: list[dict[str, object]]) -> str:
        assert results == self.results
        return self.formatted


def _build_invocation(tag_retriever_config: dict[str, object]) -> NaiInvocation:
    invocation = object.__new__(NaiInvocation)
    invocation.plugin_config = {"tag_retriever": tag_retriever_config}
    invocation.log_prefix = "test"
    return invocation


def _build_image_send_invocation() -> NaiInvocation:
    invocation = object.__new__(NaiInvocation)
    invocation.plugin_config = {
        "model": {
            "base_url": "https://std.loliyc.com",
            "nai_proxy_mode": "auto",
            "nai_request_timeout": 321.0,
        }
    }
    invocation.log_prefix = "test"
    invocation.stream_id = "stream-1"
    invocation.user_id = "user-1"
    invocation.source = "command"
    invocation._last_send_timestamp = None
    invocation.api_client = types.SimpleNamespace()

    # 本组用例只验证发图行为；跳过识图回写由 test_skip_self_vlm.py 覆盖，
    # 这里置空避免触达图片库 / 真实 DB。
    async def _noop_register(_image_base64: str, _description: str) -> None:
        return None

    invocation._register_self_image_as_processed = _noop_register
    return invocation


def test_retrieve_tag_candidates_uses_online_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = _build_invocation(
        {
            "enabled": True,
            "mode": "online",
            "api_url": "https://example.com/api",
            "timeout": 12.0,
            "search_limit": 11,
            "search_top_k": 4,
            "related_limit": 7,
            "related_seed_count": 3,
            "show_nsfw": False,
            "popularity_weight": 0.2,
        }
    )
    online_retriever = _FakeOnlineRetriever(
        {
            "search": [{"tag": "hatsune_miku"}],
            "related": [{"tag": "twintails"}],
        },
        "<online>",
    )
    local_retriever = _FakeLocalRetriever(
        [{"cn": "初音未来", "tag": "hatsune_miku", "score": 0.95}],
        "<local>",
    )
    captured_kwargs: dict[str, object] = {}

    def fake_get_online_retriever(**kwargs: object) -> _FakeOnlineRetriever:
        captured_kwargs.update(kwargs)
        return online_retriever

    def fake_get_tag_retriever(**kwargs: object) -> _FakeLocalRetriever:
        return local_retriever

    monkeypatch.setattr(resolver_module, "get_online_retriever", fake_get_online_retriever)
    monkeypatch.setattr(resolver_module, "get_tag_retriever", fake_get_tag_retriever)

    result = asyncio.run(invocation._retrieve_tag_candidates("画一张初音未来"))

    assert result == "<online>"
    assert online_retriever.queries == ["画一张初音未来"]
    assert local_retriever.calls == []
    assert captured_kwargs == {
        "enabled": True,
        "base_url": "https://example.com/api",
        "timeout": 12.0,
        "search_limit": 11,
        "search_top_k": 4,
        "related_limit": 7,
        "related_seed_count": 3,
        "show_nsfw": False,
        "popularity_weight": 0.2,
    }


def test_retrieve_tag_candidates_falls_back_to_local_when_online_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation = _build_invocation(
        {
            "enabled": True,
            "mode": "online",
            "top_k": 6,
            "min_score": 0.45,
        }
    )
    online_retriever = _FakeOnlineRetriever({"search": [], "related": []}, "<online-empty>")
    local_results = [{"cn": "初音未来", "tag": "hatsune_miku", "score": 0.93}]
    local_retriever = _FakeLocalRetriever(local_results, "<local>")

    def fake_get_online_retriever(**kwargs: object) -> _FakeOnlineRetriever:
        return online_retriever

    def fake_get_tag_retriever(**kwargs: object) -> _FakeLocalRetriever:
        return local_retriever

    monkeypatch.setattr(resolver_module, "get_online_retriever", fake_get_online_retriever)
    monkeypatch.setattr(resolver_module, "get_tag_retriever", fake_get_tag_retriever)

    result = asyncio.run(invocation._retrieve_tag_candidates("画一张猫耳少女"))

    assert result == "<local>"
    assert online_retriever.queries == ["画一张猫耳少女"]
    assert local_retriever.calls == [("画一张猫耳少女", 6, 0.45)]


def test_build_tag_retriever_display_message_strips_internal_wrapper() -> None:
    message = build_tag_retriever_display_message(
        "<tag_candidates>\n"
        "## 语义匹配（与用户描述直接相关，优先选用）\n"
        "- 初音未来 → hatsune_miku [character] (相关度 0.95)\n"
        "\n"
        "## 共现推荐（与上述标签在真实画作中经常搭配出现）\n"
        "- 双马尾 → twintails [general] (共现度 0.82)\n"
        "</tag_candidates>"
    )

    assert message == (
        "🔎 Danbooru Tag 检索结果:\n"
        "## 语义匹配（与用户描述直接相关，优先选用）\n"
        "- 初音未来 → hatsune_miku [character] (相关度 0.95)\n"
        "## 共现推荐（与上述标签在真实画作中经常搭配出现）\n"
        "- 双马尾 → twintails [general] (共现度 0.82)"
    )


def test_build_tag_retriever_display_messages_splits_long_output() -> None:
    messages = build_tag_retriever_display_messages(
        "<tag_candidates>\n"
        "## 语义匹配（与用户描述直接相关，优先选用）\n"
        + "\n".join(f"- 标签{i:02d} → tag_{i:02d} [general] (相关度 0.95)" for i in range(1, 25))
        + "\n</tag_candidates>",
        max_chars=180,
    )

    assert len(messages) > 1
    assert messages[0].startswith("🔎 Danbooru Tag 检索结果 (1/")
    assert messages[-1].startswith(f"🔎 Danbooru Tag 检索结果 ({len(messages)}/{len(messages)}):")
    assert all(len(message) <= 180 for message in messages)
    assert "标签01" in messages[0]
    assert "标签24" in messages[-1]


def test_generate_prompt_with_llm_shows_tag_candidates_when_enabled() -> None:
    invocation = object.__new__(NaiInvocation)
    invocation.stream_id = "stream-tag-show"
    invocation.group_id = ""
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    sent_texts: list[tuple[str, bool]] = []

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append((text, storage_message))
        return True

    async def fake_retrieve_tag_candidates(_request_text: str) -> str:
        return (
            "<tag_candidates>\n"
            "## 语义匹配（与用户描述直接相关，优先选用）\n"
            "- 初音未来 → hatsune_miku [character] (相关度 0.95)\n"
            "</tag_candidates>"
        )

    async def fake_request_llm_text(*_args, **_kwargs) -> str:
        return "1girl, hatsune_miku"

    invocation.send_text = fake_send_text
    invocation.get_config = lambda _key, default=None: default
    invocation._get_prompt_generator_config = lambda: {"output_format": "text"}
    invocation._render_generator_prompt = lambda *_args, **_kwargs: "header\n<<TAG_CANDIDATES>>"
    invocation._retrieve_tag_candidates = fake_retrieve_tag_candidates
    invocation._request_llm_text = fake_request_llm_text
    invocation._cleanup_llm_prompt = lambda response: response
    invocation._is_tag_retriever_show_enabled = lambda: True

    result = asyncio.run(
        invocation._generate_prompt_with_llm(
            "画一张初音未来",
            allow_inherit=False,
        )
    )

    assert result == ("1girl, hatsune_miku", None)
    assert sent_texts == [
        (
            "🔎 Danbooru Tag 检索结果:\n"
            "## 语义匹配（与用户描述直接相关，优先选用）\n"
            "- 初音未来 → hatsune_miku [character] (相关度 0.95)",
            False,
        )
    ]


def test_generate_prompt_with_llm_splits_tag_candidates_output_when_long() -> None:
    invocation = object.__new__(NaiInvocation)
    invocation.stream_id = "stream-tag-show-long"
    invocation.group_id = ""
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    sent_texts: list[tuple[str, bool]] = []

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append((text, storage_message))
        return True

    async def fake_retrieve_tag_candidates(_request_text: str) -> str:
        return (
            "<tag_candidates>\n"
            "## 语义匹配（与用户描述直接相关，优先选用）\n"
            + "\n".join(f"- 标签{i:02d} → tag_{i:02d} [general] (相关度 0.95)" for i in range(1, 25))
            + "\n</tag_candidates>"
        )

    async def fake_request_llm_text(*_args, **_kwargs) -> str:
        return "1girl, tag_01"

    invocation.send_text = fake_send_text
    invocation.get_config = lambda _key, default=None: default
    invocation._get_prompt_generator_config = lambda: {"output_format": "text"}
    invocation._render_generator_prompt = lambda *_args, **_kwargs: "header\n<<TAG_CANDIDATES>>"
    invocation._retrieve_tag_candidates = fake_retrieve_tag_candidates
    invocation._request_llm_text = fake_request_llm_text
    invocation._cleanup_llm_prompt = lambda response: response
    invocation._is_tag_retriever_show_enabled = lambda: True

    result = asyncio.run(
        invocation._generate_prompt_with_llm(
            "画一张初音未来",
            allow_inherit=False,
        )
    )

    assert result == ("1girl, tag_01", None)
    assert len(sent_texts) > 1
    assert sent_texts[0][0].startswith("🔎 Danbooru Tag 检索结果 (1/")
    assert sent_texts[-1][0].startswith(f"🔎 Danbooru Tag 检索结果 ({len(sent_texts)}/{len(sent_texts)}):")
    assert sent_texts[0][1] is False
    assert all(storage_message is False for _, storage_message in sent_texts)


def test_generate_prompt_with_llm_retries_smaller_tag_chunks_when_send_fails() -> None:
    invocation = object.__new__(NaiInvocation)
    invocation.stream_id = "stream-tag-show-retry"
    invocation.group_id = ""
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    sent_texts: list[tuple[str, bool]] = []

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        if len(text) > 90:
            return False
        sent_texts.append((text, storage_message))
        return True

    async def fake_retrieve_tag_candidates(_request_text: str) -> str:
        return (
            "<tag_candidates>\n"
            "## 语义匹配（与用户描述直接相关，优先选用）\n"
            + "\n".join(f"- 标签{i:02d} → tag_{i:02d} [general] (相关度 0.95)" for i in range(1, 12))
            + "\n</tag_candidates>"
        )

    async def fake_request_llm_text(*_args, **_kwargs) -> str:
        return "1girl, tag_01"

    invocation.send_text = fake_send_text
    invocation.get_config = lambda _key, default=None: default
    invocation._get_prompt_generator_config = lambda: {"output_format": "text"}
    invocation._render_generator_prompt = lambda *_args, **_kwargs: "header\n<<TAG_CANDIDATES>>"
    invocation._retrieve_tag_candidates = fake_retrieve_tag_candidates
    invocation._request_llm_text = fake_request_llm_text
    invocation._cleanup_llm_prompt = lambda response: response
    invocation._is_tag_retriever_show_enabled = lambda: True

    result = asyncio.run(
        invocation._generate_prompt_with_llm(
            "画一张初音未来",
            allow_inherit=False,
        )
    )

    assert result == ("1girl, tag_01", None)
    assert sent_texts
    assert all(len(text) <= 90 for text, _ in sent_texts)
    assert all(storage_message is False for _, storage_message in sent_texts)


def test_generate_prompt_with_llm_shows_empty_tag_retriever_notice_when_no_candidates() -> None:
    invocation = object.__new__(NaiInvocation)
    invocation.stream_id = "stream-tag-show-empty"
    invocation.group_id = ""
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    sent_texts: list[tuple[str, bool]] = []

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append((text, storage_message))
        return True

    async def fake_retrieve_tag_candidates(_request_text: str) -> str:
        return ""

    async def fake_request_llm_text(*_args, **_kwargs) -> str:
        return "1girl, tag_01"

    invocation.send_text = fake_send_text
    invocation.get_config = lambda key, default=None: (
        {"mode": "online"} if key == "tag_retriever" else default
    )
    invocation._get_prompt_generator_config = lambda: {"output_format": "text"}
    invocation._render_generator_prompt = lambda *_args, **_kwargs: "header\n<<TAG_CANDIDATES>>"
    invocation._retrieve_tag_candidates = fake_retrieve_tag_candidates
    invocation._request_llm_text = fake_request_llm_text
    invocation._cleanup_llm_prompt = lambda response: response
    invocation._is_tag_retriever_show_enabled = lambda: True

    result = asyncio.run(
        invocation._generate_prompt_with_llm(
            "画一张初音未来",
            allow_inherit=False,
        )
    )

    assert result == ("1girl, tag_01", None)
    assert sent_texts == [
        (
            "🔎 Danbooru Tag 检索结果:\n⚠️ 未检索到候选标签（mode=online）",
            False,
        )
    ]


def test_send_image_result_downloads_generation_url_then_sends_base64_for_unknown_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未知平台 + generation URL：跳过直发，下载后以 base64 image 段直发。"""
    invocation = _build_image_send_invocation()
    send_calls: list[tuple[str, str, str]] = []
    sent_texts: list[str] = []
    remember_calls: list[tuple[str, float]] = []
    session_marks: list[tuple[str, float]] = []
    schedule_calls: list[bool] = []

    async def fake_send_custom(
        message_type: str,
        content: str,
        *,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        send_calls.append((message_type, content, display_message))
        return True

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    async def fake_download(_url: str) -> str | None:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"

    async def fake_schedule() -> None:
        schedule_calls.append(True)

    monkeypatch.setattr(
        sdk_runtime_module,
        "remember_pending_plugin_image_send",
        lambda stream_id, send_timestamp: remember_calls.append((stream_id, send_timestamp)),
    )
    monkeypatch.setattr(
        sdk_runtime_module,
        "discard_pending_plugin_image_send",
        lambda *_args, **_kwargs: pytest.fail("unexpected pending-image discard"),
    )
    monkeypatch.setattr(
        sdk_runtime_module.session_state,
        "set_last_action_image_sent_at",
        lambda stream_id, send_timestamp: session_marks.append((stream_id, send_timestamp)),
    )

    invocation.send_custom = fake_send_custom
    invocation.send_text = fake_send_text
    invocation._get_target_platform = lambda: ""
    invocation._download_remote_image_as_base64 = fake_download
    invocation._schedule_auto_recall = fake_schedule
    invocation._build_image_display_message = lambda _desc="": "[nai-image]"

    result = asyncio.run(invocation._send_image_result("https://std.loliyc.com/generate?tag=test", "test"))

    assert result == (True, "图片生成成功", True)
    assert send_calls == [("image", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB", "[nai-image]")]
    assert sent_texts == []
    assert len(remember_calls) == 1
    assert len(session_marks) == 1
    assert schedule_calls == [True]


def test_send_image_result_downloads_generation_url_for_qq_after_direct_url_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QQ 平台 + generation URL：QQ 直发 URL 失败后下载并以 base64 image 段重发。"""
    invocation = _build_image_send_invocation()
    send_calls: list[tuple[str, str, str]] = []
    sent_texts: list[str] = []
    session_marks: list[tuple[str, float]] = []
    schedule_calls: list[bool] = []

    async def fake_send_custom(
        message_type: str,
        content: str,
        *,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        send_calls.append((message_type, content, display_message))
        # QQ 远程 URL 直发失败，触发下载回退
        if message_type == "imageurl":
            return False
        return True

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    async def fake_download(_url: str) -> str | None:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"

    async def fake_schedule() -> None:
        schedule_calls.append(True)

    monkeypatch.setattr(
        sdk_runtime_module,
        "remember_pending_plugin_image_send",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sdk_runtime_module,
        "discard_pending_plugin_image_send",
        lambda *_args, **_kwargs: pytest.fail("unexpected pending-image discard"),
    )
    monkeypatch.setattr(
        sdk_runtime_module.session_state,
        "set_last_action_image_sent_at",
        lambda stream_id, send_timestamp: session_marks.append((stream_id, send_timestamp)),
    )

    invocation.send_custom = fake_send_custom
    invocation.send_text = fake_send_text
    invocation._get_target_platform = lambda: "qq"
    invocation._download_remote_image_as_base64 = fake_download
    invocation._schedule_auto_recall = fake_schedule
    invocation._build_image_display_message = lambda _desc="": "[nai-image]"

    result = asyncio.run(invocation._send_image_result("https://std.loliyc.com/generate?tag=test", "test"))

    assert result == (True, "图片生成成功", True)
    assert send_calls == [
        ("imageurl", "https://std.loliyc.com/generate?tag=test", "[nai-image]"),
        ("image", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB", "[nai-image]"),
    ]
    assert len(session_marks) == 1
    assert schedule_calls == [True]
    assert sent_texts == []


def test_send_image_result_falls_back_to_base64_after_remote_url_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通 CDN URL：直发抛异常 → 下载 → base64 image 段重发。"""
    invocation = _build_image_send_invocation()
    send_calls: list[tuple[str, str]] = []
    session_marks: list[tuple[str, float]] = []

    async def fake_send_custom(
        message_type: str,
        content: str,
        *,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        send_calls.append((message_type, content))
        if content.startswith("https://"):
            raise RuntimeError("napcat send failed")
        return True

    async def fake_send_text(_text: str, storage_message: bool = True) -> bool:
        return True

    async def fake_download(_url: str) -> str | None:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAC"

    async def fake_schedule() -> None:
        return None

    monkeypatch.setattr(sdk_runtime_module, "remember_pending_plugin_image_send", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sdk_runtime_module,
        "discard_pending_plugin_image_send",
        lambda *_args, **_kwargs: pytest.fail("unexpected pending-image discard"),
    )
    monkeypatch.setattr(
        sdk_runtime_module.session_state,
        "set_last_action_image_sent_at",
        lambda stream_id, send_timestamp: session_marks.append((stream_id, send_timestamp)),
    )

    invocation.send_custom = fake_send_custom
    invocation.send_text = fake_send_text
    invocation._get_target_platform = lambda: ""
    invocation._download_remote_image_as_base64 = fake_download
    invocation._schedule_auto_recall = fake_schedule
    invocation._build_image_display_message = lambda _desc="": "[nai-image]"

    result = asyncio.run(invocation._send_image_result("https://cdn.example.com/images/test.png", "test"))

    assert result == (True, "图片生成成功", True)
    assert send_calls == [
        ("imageurl", "https://cdn.example.com/images/test.png"),
        ("image", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAC"),
    ]
    assert len(session_marks) == 1


def test_manual_recall_skips_stale_images_without_attempting_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation = _build_image_send_invocation()
    invocation.stream_id = "stream-1"
    invocation.get_config = lambda key, default=None: 3600 if key == "auto_recall.manual_max_age_seconds" else default

    sent_texts: list[str] = []

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    async def fake_find_last_plugin_image_row(*_args, **kwargs):
        exclude_ids = set(kwargs.get("exclude_message_ids") or set())
        if "old-image-id" in exclude_ids:
            return None
        return {
            "message_id": "old-image-id",
            "timestamp": "2024-05-11 12:00:00",
            "is_picture": 1,
        }

    async def fake_try_recall_message(_message_id: str) -> bool:
        pytest.fail("stale image should not trigger recall attempt")

    monkeypatch.setattr(sdk_runtime_module, "_find_last_plugin_image_row", fake_find_last_plugin_image_row)
    monkeypatch.setattr(sdk_runtime_module.time, "time", lambda: 1746954000.0)

    invocation.send_text = fake_send_text
    invocation._try_recall_message = fake_try_recall_message

    result = asyncio.run(invocation._do_manual_recall())

    assert result == (False, "找不到可撤回的消息", True)
    assert sent_texts == ["❌ 找不到近期可撤回的图片（图片可能已超过平台撤回时限）"]


def test_send_base64_image_result_sends_image_segment_directly() -> None:
    """_send_base64_image_result 始终以 image 段直发 base64，不依赖本地文件路径。"""
    invocation = _build_image_send_invocation()
    send_calls: list[tuple[str, str, str]] = []

    async def fake_send_custom(
        message_type: str,
        content: str,
        *,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        send_calls.append((message_type, content, display_message))
        return True

    invocation.send_custom = fake_send_custom

    result = asyncio.run(invocation._send_base64_image_result("iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB", "[nai-image]"))

    assert result is True
    assert send_calls == [("image", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB", "[nai-image]")]


def test_send_base64_image_result_propagates_send_failure() -> None:
    """send_custom 返回 False 时函数直接返回 False，不再尝试任何 file:// 二次发送。"""
    invocation = _build_image_send_invocation()
    send_calls: list[tuple[str, str]] = []

    async def fake_send_custom(
        message_type: str,
        content: str,
        *,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        send_calls.append((message_type, content))
        return False

    invocation.send_custom = fake_send_custom

    result = asyncio.run(invocation._send_base64_image_result("iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB", "[nai-image]"))

    assert result is False
    assert send_calls == [("image", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")]


def test_download_remote_image_as_base64_skips_generation_request_url() -> None:
    invocation = _build_image_send_invocation()

    async def fail_send_request_with_retry(*_args, **_kwargs):
        pytest.fail("generation request URL 不应再次发起下载请求")

    invocation.api_client = types.SimpleNamespace(_send_request_with_retry=fail_send_request_with_retry)

    result = asyncio.run(invocation._download_remote_image_as_base64("https://cdn.example.com/generate?tag=test"))

    assert result is None
