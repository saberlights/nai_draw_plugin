import sys
import types
import asyncio
from pathlib import Path
from typing import Any
from weakref import WeakSet

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
from plugins.nai_draw_plugin.core.services.background_task_supervisor import (
    BackgroundTaskSupervisor,
)
from plugins.nai_draw_plugin.core.services.generation_admission_policy import (
    AdmissionDecision,
)
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


def test_refresh_retag_runtime_injects_wd14_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = object.__new__(NaiPicPlugin)

    async def run_wd14(function, *args, **kwargs):
        return function(*args)

    plugin._wd14_io = types.SimpleNamespace(run=run_wd14)
    plugin._image_cache_service = types.SimpleNamespace(update_config=lambda **_kwargs: None)
    plugin._reverse_service = types.SimpleNamespace(
        update_wd14_client=lambda client: captured.setdefault("client", client),
        update_wd14_thresholds=lambda **_kwargs: None,
    )
    plugin.get_plugin_config_data = lambda: {"retag": {"wd14_enabled": True}}
    captured: dict[str, object] = {}

    class _CapturingWD14Client:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr(plugin_module, "WD14Client", _CapturingWD14Client)

    plugin._refresh_retag_runtime()

    assert captured["client"].__class__ is _CapturingWD14Client
    assert captured["kwargs"]["run_blocking"] is run_wd14


def test_reply_hook_uses_admission_description_for_background_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin_config = {
        "auto_draw_on_reply": {"enabled": True, "score_threshold": 0.6},
        "action_guard": {"enabled": True},
    }
    policy_calls: list[dict[str, object]] = []
    create_calls: list[tuple[str, dict[str, str], str]] = []
    start_calls: list[tuple[str, str, object]] = []

    class _Policy:
        def evaluate_reply_candidate(self, **kwargs: object) -> AdmissionDecision:
            policy_calls.append(kwargs)
            return AdmissionDecision(
                should_generate=True,
                category="auto_draw",
                detail="准入",
                seed_description="一女 自拍 窗边",
            )

    class _Invocation:
        async def handle_auto_draw_from_reply(
            self,
            description: str,
            *,
            reply_context_text: str,
        ) -> None:
            start_calls.append((description, reply_context_text, self))

    invocation = _Invocation()
    plugin._generation_admission_policy = _Policy()

    async def fake_load_config() -> dict[str, object]:
        return plugin_config

    async def fake_create_invocation(
        stream_id: str,
        *,
        action_data: dict[str, str],
        source: str,
    ) -> _Invocation:
        create_calls.append((stream_id, action_data, source))
        return invocation

    def fake_start(
        stream_id: str,
        job_factory,
        *,
        invocation: object,
        name: str,
    ) -> bool:
        assert stream_id == "stream-1"
        assert name == "nai-reply-auto-draw"
        asyncio.get_running_loop().create_task(job_factory())
        return True

    monkeypatch.setattr(plugin, "_load_plugin_config_data", fake_load_config)
    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)
    monkeypatch.setattr(plugin, "_start_image_generation_in_background", fake_start)

    async def scenario() -> None:
        await plugin.handle_replyer_after_response_for_auto_draw(
            session_id=" stream-1 ",
            response=" 我刚洗完澡靠在窗边发呆，有点累 ",
        )
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert policy_calls == [
        {
            "stream_id": "stream-1",
            "reply_text": "我刚洗完澡靠在窗边发呆，有点累",
            "config": plugin_config,
        }
    ]
    assert create_calls == [
        (
            "stream-1",
            {"description": "一女 自拍 窗边"},
            "reply_auto_draw",
        )
    ]
    assert start_calls == [
        ("一女 自拍 窗边", "我刚洗完澡靠在窗边发呆，有点累", invocation)
    ]


def test_reply_hook_rejection_does_not_create_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    create_calls = 0

    class _Policy:
        def evaluate_reply_candidate(self, **kwargs: object) -> AdmissionDecision:
            return AdmissionDecision(False, "blocked", "reply 未达到阈值")

    plugin._generation_admission_policy = _Policy()

    async def fake_load_config() -> dict[str, object]:
        return {"auto_draw_on_reply": {"enabled": True}}

    async def fake_create_invocation(*args: object, **kwargs: object) -> None:
        nonlocal create_calls
        create_calls += 1

    monkeypatch.setattr(plugin, "_load_plugin_config_data", fake_load_config)
    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)

    asyncio.run(
        plugin.handle_replyer_after_response_for_auto_draw(
            session_id="stream-1",
            response="我刚洗完澡靠在窗边发呆，有点累",
        )
    )

    assert create_calls == 0


def test_reply_hook_config_load_is_cancelled_before_invocation_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._generation_admission_policy = types.SimpleNamespace()
    load_started = asyncio.Event()
    create_calls = 0

    async def load_config() -> dict[str, object]:
        load_started.set()
        await asyncio.Event().wait()
        return {}

    async def create_invocation(*_args: object, **_kwargs: object) -> None:
        nonlocal create_calls
        create_calls += 1

    monkeypatch.setattr(plugin, "_load_plugin_config_data", load_config)
    monkeypatch.setattr(plugin, "_create_invocation", create_invocation)

    async def scenario() -> None:
        hook = asyncio.create_task(
            plugin.handle_replyer_after_response_for_auto_draw(
                session_id="stream-unload-reply",
                response="我刚洗完澡靠在窗边发呆，有点累",
            )
        )
        await load_started.wait()
        await plugin._background_tasks.shutdown()
        assert hook.cancelled()

    asyncio.run(scenario())

    assert create_calls == 0


class _DummySend:
    def __init__(self) -> None:
        self.text_calls: list[tuple[str, str, bool]] = []

    async def text(self, text: str, stream_id: str, storage_message: bool = True) -> bool:
        self.text_calls.append((text, stream_id, storage_message))
        return True


class _FailingSend(_DummySend):
    async def text(self, text: str, stream_id: str, storage_message: bool = True) -> bool:
        raise RuntimeError("ack failed")


class _RejectedSend(_DummySend):
    async def text(self, text: str, stream_id: str, storage_message: bool = True) -> bool:
        self.text_calls.append((text, stream_id, storage_message))
        return False


class _BlockingSend(_DummySend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def text(self, text: str, stream_id: str, storage_message: bool = True) -> bool:
        self.started.set()
        await asyncio.Event().wait()
        return True


class _DummyInvocation:
    def __init__(self, *, generation_allowed: bool = True) -> None:
        self.draw_calls: list[str] = []
        self.nai0_calls: list[str] = []
        self.close_calls = 0
        self.generation_allowed = generation_allowed

    async def ensure_generation_permission(self) -> bool:
        return self.generation_allowed

    async def ensure_user_not_blacklisted(self) -> bool:
        return True

    async def preflight_action_guard(self) -> AdmissionDecision | None:
        return None

    async def handle_nai_draw(self, description: str) -> tuple[bool, str | None, bool]:
        self.draw_calls.append(description)
        return True, description, True

    async def handle_nai0_draw(self, tags: str) -> tuple[bool, str | None, bool]:
        self.nai0_calls.append(tags)
        return True, tags, True

    async def handle_action(self) -> tuple[bool, str]:
        raise RuntimeError("unexpected action failure")

    async def handle_admin_command(
        self,
        action: str,
        param: str,
    ) -> tuple[bool, str | None, bool]:
        return True, f"{action}:{param}", True

    def close(self) -> None:
        self.close_calls += 1


class _BlockingPreflightInvocation(_DummyInvocation):
    def __init__(self) -> None:
        super().__init__()
        self.preflight_started = asyncio.Event()

    async def preflight_action_guard(self) -> AdmissionDecision | None:
        self.preflight_started.set()
        await asyncio.Event().wait()
        return None


def test_handle_nai_draw_allows_multiple_commands_in_same_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()

    invocation = _DummyInvocation()

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        return invocation

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)

    async def _run() -> tuple[tuple[bool, str | None, bool], tuple[bool, str | None, bool]]:
        first = await plugin.handle_nai_draw(
            stream_id="stream-1",
            matched_groups={"description": "初音未来"},
        )
        second = await plugin.handle_nai_draw(
            stream_id="stream-1",
            matched_groups={"description": "初音未来"},
        )
        await asyncio.sleep(0)
        await plugin._background_tasks.shutdown()
        return first, second

    first, second = asyncio.run(_run())

    assert first == (True, "已开始生成图片", True)
    assert second == (True, "已开始生成图片", True)
    assert invocation.draw_calls == ["初音未来", "初音未来"]
    assert invocation.close_calls == 2
    assert plugin.ctx.send.text_calls == [
        ("收到，正在生成图片，请稍候...", "stream-1", False),
        ("收到，正在生成图片，请稍候...", "stream-1", False),
    ]


def test_handle_nai0_draw_allows_multiple_commands_in_same_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()

    invocation = _DummyInvocation()

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        return invocation

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)

    async def _run() -> tuple[tuple[bool, str | None, bool], tuple[bool, str | None, bool]]:
        first = await plugin.handle_nai_0_draw(
            stream_id="stream-2",
            matched_groups={"tags": "1girl, hatsune miku"},
        )
        second = await plugin.handle_nai_0_draw(
            stream_id="stream-2",
            matched_groups={"tags": "1girl, hatsune miku"},
        )
        await asyncio.sleep(0)
        await plugin._background_tasks.shutdown()
        return first, second

    first, second = asyncio.run(_run())

    assert first == (True, "已开始生成图片", True)
    assert second == (True, "已开始生成图片", True)
    assert invocation.nai0_calls == ["1girl, hatsune miku", "1girl, hatsune miku"]
    assert invocation.close_calls == 2
    assert plugin.ctx.send.text_calls == [
        ("收到，正在生成图片，请稍候...", "stream-2", False),
        ("收到，正在生成图片，请稍候...", "stream-2", False),
    ]


def test_start_image_generation_in_background_still_blocks_duplicate_action_stream() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
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


def test_duplicate_generation_closes_rejected_invocation_without_starting_it() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()
    stream_id = "stream-duplicate-close"
    session_state = plugin_module.session_state
    session_state.clear_pending_image_generation(stream_id)
    first_owner = session_state.acquire_pending_image_generation(stream_id)
    job_calls = 0

    async def generation() -> None:
        nonlocal job_calls
        job_calls += 1

    try:
        started = plugin._start_image_generation_in_background(
            stream_id,
            generation,
            invocation=invocation,
        )
    finally:
        assert first_owner is not None
        session_state.release_pending_image_generation(stream_id, first_owner)

    assert started is False
    assert job_calls == 0
    assert invocation.close_calls == 1


def test_generation_permission_rejection_closes_invocation_without_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation(generation_allowed=False)

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        plugin._active_invocations.add(invocation)
        return invocation

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)

    result = asyncio.run(
        plugin.handle_nai_draw(
            stream_id="stream-permission-rejected",
            matched_groups={"description": "初音未来"},
        )
    )

    assert result == (False, "没有权限", True)
    assert plugin.ctx.send.text_calls == []
    assert invocation.draw_calls == []
    assert invocation.close_calls == 1
    assert list(plugin._active_invocations) == []


def test_foreground_command_uses_managed_invocation_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        plugin._active_invocations.add(invocation)
        return invocation

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)

    result = asyncio.run(
        plugin.handle_nai_admin_control_command(
            stream_id="stream-admin",
            matched_groups={"action": "size", "param": "1024x1024"},
        )
    )

    assert result == (True, "size:1024x1024", True)
    assert invocation.close_calls == 1
    assert list(plugin._active_invocations) == []


def test_command_background_failure_reports_once_and_closes_invocation() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()

    async def fail_generation() -> None:
        raise RuntimeError("unexpected failure")

    async def scenario() -> bool:
        started = await plugin._start_command_image_generation(
            "stream-failure",
            fail_generation,
            invocation=invocation,
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await plugin._background_tasks.shutdown()
        return started

    assert asyncio.run(scenario()) is True
    assert plugin.ctx.send.text_calls == [
        ("收到，正在生成图片，请稍候...", "stream-failure", False),
        ("图片生成任务意外中断，请稍后重试。", "stream-failure", False),
    ]
    assert invocation.close_calls == 1


def test_command_ack_failure_does_not_start_job_and_closes_invocation() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_FailingSend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()
    job_calls = 0

    async def generation() -> None:
        nonlocal job_calls
        job_calls += 1

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="ack failed"):
            await plugin._start_command_image_generation(
                "stream-ack-failure",
                generation,
                invocation=invocation,
            )

    asyncio.run(scenario())

    assert job_calls == 0
    assert invocation.close_calls == 1


def test_command_rejected_ack_does_not_start_job_and_closes_invocation() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_RejectedSend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()
    job_calls = 0

    async def generation() -> None:
        nonlocal job_calls
        job_calls += 1

    started = asyncio.run(
        plugin._start_command_image_generation(
            "stream-ack-rejected",
            generation,
            invocation=invocation,
        )
    )

    assert started is False
    assert job_calls == 0
    assert invocation.close_calls == 1


def test_command_ack_is_cancelled_by_shutdown_before_job_can_start() -> None:
    plugin = object.__new__(NaiPicPlugin)
    blocking_send = _BlockingSend()
    plugin.ctx = types.SimpleNamespace(send=blocking_send)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()
    job_calls = 0

    async def generation() -> None:
        nonlocal job_calls
        job_calls += 1

    async def scenario() -> bool:
        submission = asyncio.create_task(
            plugin._start_command_image_generation(
                "stream-ack-shutdown",
                generation,
                invocation=invocation,
            )
        )
        await blocking_send.started.wait()
        await plugin._background_tasks.shutdown()
        return await submission

    assert asyncio.run(scenario()) is False
    assert job_calls == 0
    assert invocation.close_calls == 1


def test_action_background_failure_reports_once_and_releases_invocation() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()
    stream_id = "stream-action-failure"
    plugin_module.session_state.clear_pending_image_generation(stream_id)

    async def fake_create_invocation(*_args: Any, **_kwargs: Any) -> _DummyInvocation:
        plugin._active_invocations.add(invocation)
        return invocation

    plugin._create_invocation = fake_create_invocation

    async def scenario() -> tuple[bool, str]:
        result = await plugin.handle_nai_web_draw(stream_id=stream_id)
        await asyncio.sleep(0)
        await plugin._background_tasks.shutdown()
        return result

    result = asyncio.run(scenario())

    assert result[0] is True
    assert plugin.ctx.send.text_calls == [
        ("图片生成任务意外中断，请稍后重试。", stream_id, False)
    ]
    assert invocation.close_calls == 1
    assert list(plugin._active_invocations) == []
    assert plugin_module.session_state.get_pending_image_generation_started_at(stream_id) is None


def test_action_preflight_is_cancelled_by_shutdown_and_closes_invocation() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin.ctx = types.SimpleNamespace(send=_DummySend())
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _BlockingPreflightInvocation()

    async def create_invocation(*_args: object, **_kwargs: object) -> _DummyInvocation:
        plugin._active_invocations.add(invocation)
        return invocation

    plugin._create_invocation = create_invocation

    async def scenario() -> None:
        action = asyncio.create_task(
            plugin.handle_nai_web_draw(stream_id="stream-preflight-shutdown")
        )
        await invocation.preflight_started.wait()
        await plugin._background_tasks.shutdown()
        assert action.cancelled()

    asyncio.run(scenario())

    assert invocation.close_calls == 1
    assert list(plugin._active_invocations) == []


def test_create_invocation_rechecks_shutdown_after_config_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    constructor_calls = 0

    async def load_config() -> dict[str, object]:
        load_started.set()
        await release_load.wait()
        return {}

    class _Invocation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal constructor_calls
            constructor_calls += 1

    monkeypatch.setattr(plugin, "_load_plugin_config_data", load_config)
    monkeypatch.setattr(plugin_module, "NaiInvocation", _Invocation)

    async def scenario() -> None:
        creation = asyncio.create_task(plugin._create_invocation("stream-config-race"))
        await load_started.wait()
        await plugin._background_tasks.shutdown()
        release_load.set()
        with pytest.raises(RuntimeError, match="正在卸载"):
            await creation

    asyncio.run(scenario())

    assert constructor_calls == 0
    assert list(plugin._active_invocations) == []


def test_unload_cancels_generation_releases_lease_and_closes_invocation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._wd14_io = types.SimpleNamespace(close=lambda: None)
    plugin._http_io = types.SimpleNamespace(close=lambda: None)
    plugin._blocking_io = types.SimpleNamespace(close=lambda: None)
    plugin._active_invocations = WeakSet()
    plugin._image_cache_service = types.SimpleNamespace(clear=lambda: None)
    admission_clear_calls: list[bool] = []
    plugin._generation_admission_policy = types.SimpleNamespace(
        clear=lambda: admission_clear_calls.append(True)
    )
    monkeypatch.setattr(plugin, "_refresh_runtime_singletons", lambda **_kwargs: None)
    invocation = _DummyInvocation()
    plugin._active_invocations.add(invocation)
    stream_id = "stream-unload"
    session_state = plugin_module.session_state
    session_state.clear_pending_image_generation(stream_id)
    started = asyncio.Event()

    async def generation() -> None:
        started.set()
        await asyncio.Event().wait()

    async def scenario() -> None:
        assert plugin._start_image_generation_in_background(
            stream_id,
            generation,
            invocation=invocation,
        ) is True
        await started.wait()
        await plugin.on_unload()

    asyncio.run(scenario())

    assert session_state.get_pending_image_generation_started_at(stream_id) is None
    assert invocation.close_calls == 1
    assert admission_clear_calls == [True]


def test_closing_supervisor_rejects_generation_and_releases_acquired_lease() -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocation = _DummyInvocation()
    stream_id = "stream-closing"
    session_state = plugin_module.session_state
    session_state.clear_pending_image_generation(stream_id)
    job_calls = 0

    async def generation() -> None:
        nonlocal job_calls
        job_calls += 1

    async def scenario() -> bool:
        await plugin._background_tasks.shutdown()
        return plugin._start_image_generation_in_background(
            stream_id,
            generation,
            invocation=invocation,
        )

    assert asyncio.run(scenario()) is False
    assert job_calls == 0
    assert invocation.close_calls == 1
    assert session_state.get_pending_image_generation_started_at(stream_id) is None


def test_foreground_invocation_closes_after_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._active_invocations = WeakSet()
    invocations = [_DummyInvocation(), _DummyInvocation(), _DummyInvocation()]

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        invocation = invocations.pop(0)
        plugin._active_invocations.add(invocation)
        return invocation

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)

    async def succeed(invocation: _DummyInvocation) -> str:
        return "success"

    async def reject(invocation: _DummyInvocation) -> tuple[bool, str]:
        return False, "rejected"

    async def fail(invocation: _DummyInvocation) -> str:
        raise RuntimeError("foreground failure")

    first_invocation, second_invocation, third_invocation = invocations

    async def scenario() -> tuple[str, tuple[bool, str]]:
        result = await plugin._run_foreground_invocation("stream-success", succeed)
        rejected = await plugin._run_foreground_invocation("stream-rejected", reject)
        with pytest.raises(RuntimeError, match="foreground failure"):
            await plugin._run_foreground_invocation("stream-failure", fail)
        return result, rejected

    assert asyncio.run(scenario()) == ("success", (False, "rejected"))
    assert first_invocation.close_calls == 1
    assert second_invocation.close_calls == 1
    assert third_invocation.close_calls == 1
    assert list(plugin._active_invocations) == []


def test_unload_cancels_foreground_before_closing_http_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._blocking_io = types.SimpleNamespace(close=lambda: None)
    plugin._active_invocations = WeakSet()
    plugin._image_cache_service = types.SimpleNamespace(clear=lambda: None)
    plugin._generation_admission_policy = types.SimpleNamespace(clear=lambda: None)
    events: list[str] = []
    plugin._wd14_io = types.SimpleNamespace(close=lambda: events.append("wd14-close"))
    plugin._http_io = types.SimpleNamespace(close=lambda: events.append("http-close"))
    monkeypatch.setattr(plugin, "_refresh_runtime_singletons", lambda **_kwargs: None)
    invocation = _DummyInvocation()
    started = asyncio.Event()

    async def fake_create_invocation(*args: Any, **kwargs: Any) -> _DummyInvocation:
        plugin._active_invocations.add(invocation)
        return invocation

    async def foreground_operation(_invocation: _DummyInvocation) -> None:
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            events.append("foreground-finalize")

    monkeypatch.setattr(plugin, "_create_invocation", fake_create_invocation)

    async def scenario() -> None:
        task = asyncio.create_task(
            plugin._run_foreground_invocation("stream-foreground", foreground_operation)
        )
        await started.wait()
        await plugin.on_unload()
        assert task.cancelled()

    asyncio.run(scenario())

    assert events == ["foreground-finalize", "wd14-close", "http-close"]
    assert invocation.close_calls == 1
    assert list(plugin._active_invocations) == []


def test_unload_cancels_retag_before_closing_wd14_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = object.__new__(NaiPicPlugin)
    plugin._background_tasks = BackgroundTaskSupervisor(logger=plugin_module.logger)
    plugin._blocking_io = types.SimpleNamespace(close=lambda: None)
    plugin._http_io = types.SimpleNamespace(close=lambda: None)
    plugin._active_invocations = WeakSet()
    plugin._image_cache_service = types.SimpleNamespace(clear=lambda: None)
    plugin._generation_admission_policy = types.SimpleNamespace(clear=lambda: None)
    events: list[str] = []
    plugin._wd14_io = types.SimpleNamespace(close=lambda: events.append("wd14-close"))
    monkeypatch.setattr(plugin, "_refresh_runtime_singletons", lambda **_kwargs: None)
    started = asyncio.Event()

    async def blocking_retag(**_kwargs: Any) -> tuple[bool, str | None, bool]:
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            events.append("retag-finalize")

    monkeypatch.setattr(plugin, "_run_retag", blocking_retag)

    async def scenario() -> None:
        task = asyncio.create_task(
            plugin.handle_nai_retag_command(stream_id="stream-retag", user_id="user-retag")
        )
        await started.wait()
        await plugin.on_unload()
        assert task.cancelled()

    asyncio.run(scenario())

    assert events == ["retag-finalize", "wd14-close"]
