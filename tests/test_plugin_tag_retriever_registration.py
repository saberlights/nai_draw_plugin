import sys
import types
import asyncio
from pathlib import Path
from typing import Any

import pytest

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

maibot_sdk_stub = types.ModuleType("maibot_sdk")
maibot_sdk_stub.Action = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.Command = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.HookHandler = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.MaiBotPlugin = type("MaiBotPlugin", (), {})
sys.modules.setdefault("maibot_sdk", maibot_sdk_stub)

maibot_sdk_types_stub = types.ModuleType("maibot_sdk.types")
maibot_sdk_types_stub.ActivationType = type("ActivationType", (), {"ALWAYS": "ALWAYS"})
maibot_sdk_types_stub.HookMode = type("HookMode", (), {"OBSERVE": "OBSERVE"})
maibot_sdk_types_stub.HookOrder = type("HookOrder", (), {"EARLY": "EARLY", "NORMAL": "NORMAL", "LATE": "LATE"})
sys.modules.setdefault("maibot_sdk.types", maibot_sdk_types_stub)

src_config_package = types.ModuleType("src.config")
src_config_package.__path__ = [str(MAIBOT_ROOT / "src" / "config")]
sys.modules.setdefault("src.config", src_config_package)

src_chat_package = types.ModuleType("src.chat")
src_chat_package.__path__ = [str(MAIBOT_ROOT / "src" / "chat")]
sys.modules.setdefault("src.chat", src_chat_package)

src_chat_utils_package = types.ModuleType("src.chat.utils")
src_chat_utils_package.__path__ = [str(MAIBOT_ROOT / "src" / "chat" / "utils")]
sys.modules.setdefault("src.chat.utils", src_chat_utils_package)

chat_utils_module = types.ModuleType("src.chat.utils.utils")
chat_utils_module.parse_platform_accounts = lambda platforms: {}
sys.modules["src.chat.utils.utils"] = chat_utils_module

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
src_llm_models_package.__path__ = [str(MAIBOT_ROOT / "src" / "llm_models")]
sys.modules.setdefault("src.llm_models", src_llm_models_package)

src_common_data_models_package = types.ModuleType("src.common.data_models")
src_common_data_models_package.__path__ = [str(MAIBOT_ROOT / "src" / "common" / "data_models")]
sys.modules.setdefault("src.common.data_models", src_common_data_models_package)

llm_service_data_models_module = types.ModuleType("src.common.data_models.llm_service_data_models")
llm_service_data_models_module.LLMGenerationOptions = type("LLMGenerationOptions", (), {})
llm_service_data_models_module.LLMImageOptions = type("LLMImageOptions", (), {})
sys.modules["src.common.data_models.llm_service_data_models"] = llm_service_data_models_module

utils_model_module = types.ModuleType("src.llm_models.utils_model")


class _DummyLLMOrchestrator:
    def __init__(self, *args, **kwargs):
        self.model_for_task = None
        self.model_usage = {}


utils_model_module.LLMOrchestrator = _DummyLLMOrchestrator
sys.modules["src.llm_models.utils_model"] = utils_model_module

src_services_module = types.ModuleType("src.services")
src_services_module.__path__ = [str(MAIBOT_ROOT / "src" / "services")]
src_services_module.llm_service = types.SimpleNamespace()
sys.modules["src.services"] = src_services_module

embedding_service_module = types.ModuleType("src.services.embedding_service")
embedding_service_module.EmbeddingServiceClient = type("EmbeddingServiceClient", (), {})
sys.modules["src.services.embedding_service"] = embedding_service_module

llm_service_module = types.ModuleType("src.services.llm_service")
llm_service_module.LLMServiceClient = type("LLMServiceClient", (), {})
llm_service_module.resolve_task_name = lambda preferred_task_name="": preferred_task_name or "default"
llm_service_module.resolve_task_name_from_model_config = (
    lambda model_config, preferred_task_name="": preferred_task_name or "default"
)
sys.modules["src.services.llm_service"] = llm_service_module

from plugins.nai_draw_plugin import plugin as plugin_module
from plugins.nai_draw_plugin.plugin import NaiPicPlugin


def test_refresh_runtime_singletons_uses_online_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.get_plugin_config_data = lambda: {
        "tag_retriever": {
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
    }
    calls: dict[str, object] = {}

    def fake_reset_tag_retriever() -> None:
        calls["reset_tag"] = True

    def fake_reset_online_retriever() -> None:
        calls["reset_online"] = True

    def fake_get_online_retriever(**kwargs: object) -> None:
        calls["online_kwargs"] = kwargs

    def fake_get_tag_retriever(**kwargs: object) -> None:
        calls["local_kwargs"] = kwargs

    monkeypatch.setattr(plugin_module, "reset_tag_retriever", fake_reset_tag_retriever)
    monkeypatch.setattr(plugin_module, "get_tag_retriever", fake_get_tag_retriever)
    monkeypatch.setattr(
        plugin_module,
        "_load_online_retriever_api",
        lambda: (fake_get_online_retriever, fake_reset_online_retriever),
    )

    plugin._refresh_runtime_singletons()

    assert calls["reset_tag"] is True
    assert calls["reset_online"] is True
    assert "local_kwargs" not in calls
    assert calls["online_kwargs"] == {
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


def test_refresh_runtime_singletons_uses_local_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.get_plugin_config_data = lambda: {
        "tag_retriever": {
            "enabled": True,
            "mode": "local",
            "top_k": 42,
            "min_score": 0.55,
        }
    }
    calls: dict[str, object] = {}

    def fake_reset_tag_retriever() -> None:
        calls["reset_tag"] = True

    def fake_get_tag_retriever(**kwargs: object) -> None:
        calls["local_kwargs"] = kwargs

    monkeypatch.setattr(plugin_module, "reset_tag_retriever", fake_reset_tag_retriever)
    monkeypatch.setattr(plugin_module, "get_tag_retriever", fake_get_tag_retriever)
    monkeypatch.setattr(plugin_module, "_load_online_retriever_api", lambda: None)

    plugin._refresh_runtime_singletons()

    assert calls["reset_tag"] is True
    assert calls["local_kwargs"] == {
        "enabled": True,
        "top_k": 42,
        "min_score": 0.55,
    }


class _DummySend:
    def __init__(self) -> None:
        self.text_calls: list[tuple[str, str, bool]] = []

    async def text(self, text: str, stream_id: str, storage_message: bool = True) -> bool:
        self.text_calls.append((text, stream_id, storage_message))
        return True


class _DummyInvocation:
    async def ensure_generation_permission(self) -> bool:
        return True

    async def handle_nai_draw(self, description: str) -> tuple[bool, str | None, bool]:
        return True, description, True

    async def handle_nai0_draw(self, tags: str) -> tuple[bool, str | None, bool]:
        return True, tags, True


def test_handle_nai_draw_allows_multiple_commands_in_same_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())

    invocation = _DummyInvocation()

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        return invocation

    started_coroutines: list[object] = []

    def fake_run_invocation_in_background(coroutine: object) -> None:
        started_coroutines.append(coroutine)
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)
    monkeypatch.setattr(plugin, "_run_invocation_in_background", fake_run_invocation_in_background)

    async def _run() -> tuple[tuple[bool, str | None, bool], tuple[bool, str | None, bool]]:
        first = await plugin.handle_nai_draw(
            stream_id="stream-1",
            matched_groups={"description": "初音未来"},
        )
        second = await plugin.handle_nai_draw(
            stream_id="stream-1",
            matched_groups={"description": "初音未来"},
        )
        return first, second

    first, second = asyncio.run(_run())

    assert first == (True, "已开始生成图片", True)
    assert second == (True, "已开始生成图片", True)
    assert len(started_coroutines) == 2
    assert plugin.ctx.send.text_calls == [
        ("收到，正在生成图片，请稍候...", "stream-1", False),
        ("收到，正在生成图片，请稍候...", "stream-1", False),
    ]


def test_handle_nai0_draw_allows_multiple_commands_in_same_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())

    invocation = _DummyInvocation()

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        return invocation

    started_coroutines: list[object] = []

    def fake_run_invocation_in_background(coroutine: object) -> None:
        started_coroutines.append(coroutine)
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)
    monkeypatch.setattr(plugin, "_run_invocation_in_background", fake_run_invocation_in_background)

    async def _run() -> tuple[tuple[bool, str | None, bool], tuple[bool, str | None, bool]]:
        first = await plugin.handle_nai_0_draw(
            stream_id="stream-2",
            matched_groups={"tags": "1girl, hatsune miku"},
        )
        second = await plugin.handle_nai_0_draw(
            stream_id="stream-2",
            matched_groups={"tags": "1girl, hatsune miku"},
        )
        return first, second

    first, second = asyncio.run(_run())

    assert first == (True, "已开始生成图片", True)
    assert second == (True, "已开始生成图片", True)
    assert len(started_coroutines) == 2
    assert plugin.ctx.send.text_calls == [
        ("收到，正在生成图片，请稍候...", "stream-2", False),
        ("收到，正在生成图片，请稍候...", "stream-2", False),
    ]


def test_start_image_generation_in_background_still_blocks_duplicate_action_stream() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._tasks = set()
    stream_id = "stream-action-guard"
    session_state = plugin_module.session_state
    session_state.clear_pending_image_generation(stream_id)

    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_generation() -> None:
        started.set()
        await release.wait()

    async def _run() -> tuple[bool, bool]:
        first = plugin._start_image_generation_in_background(stream_id, lambda: fake_generation())
        second = plugin._start_image_generation_in_background(stream_id, lambda: fake_generation())
        assert session_state.get_pending_image_generation_started_at(stream_id) is not None
        await started.wait()
        release.set()
        await asyncio.sleep(0)
        return first, second

    first, second = asyncio.run(_run())

    assert first is True
    assert second is False
    assert session_state.get_pending_image_generation_started_at(stream_id) is None
