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

from plugins.nai_draw_plugin.core.services import danbooru_online_retriever as online_retriever_module
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
    invocation._last_send_timestamp = None
    invocation.api_client = types.SimpleNamespace()
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

    monkeypatch.setattr(online_retriever_module, "get_online_retriever", fake_get_online_retriever)
    monkeypatch.setattr(sdk_runtime_module, "get_tag_retriever", fake_get_tag_retriever)

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

    monkeypatch.setattr(online_retriever_module, "get_online_retriever", fake_get_online_retriever)
    monkeypatch.setattr(sdk_runtime_module, "get_tag_retriever", fake_get_tag_retriever)

    result = asyncio.run(invocation._retrieve_tag_candidates("画一张猫耳少女"))

    assert result == "<local>"
    assert online_retriever.queries == ["画一张猫耳少女"]
    assert local_retriever.calls == [("画一张猫耳少女", 6, 0.45)]


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
