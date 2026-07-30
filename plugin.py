from typing import Any, Awaitable, Callable, TypeVar
from weakref import WeakSet

import inspect
import os
import re

from maibot_sdk import Action, Command, HookHandler, MaiBotPlugin
from maibot_sdk.types import ActivationType, HookMode, HookOrder
from src.common.logger import get_logger

from .core.constants import NAI_PIC_IMAGE_DISPLAY_MARKER
from .core.plugin_config import PLUGIN_CONFIG
from .core.retag import ImageCacheService, ReverseService, WD14Client
from .core.reply_command_text import normalize_reply_command_text
from .core.services.background_task_supervisor import BackgroundTaskSupervisor
from .core.services.blocking_io_runner import BlockingIORunner
from .core.services.generation_admission_policy import GenerationAdmissionPolicy
from .core.services.session_state import session_state
from .core.services.tag_retriever import get_tag_retriever, reset_tag_retriever
from .runtime_recall import (
    attach_plugin_image_marker_to_message,
    remember_sent_plugin_image_message,
    reset_runtime_recall_tracking_state,
)
from .sdk_runtime import NaiInvocation


logger = get_logger("nai_draw_plugin")
InvocationResultT = TypeVar("InvocationResultT")


def _load_online_retriever_api() -> tuple[Any, Any] | None:
    """按需加载在线检索器，避免本地模式在缺依赖时阻塞插件注册。"""
    try:
        from .core.services.danbooru_online_retriever import get_online_retriever, reset_online_retriever
    except Exception:
        return None
    return get_online_retriever, reset_online_retriever


class NaiPicPlugin(MaiBotPlugin):
    """同步 nai_pic_plugin 业务逻辑的 NovelAI NewAPI 网关图片生成插件。"""

    # 插件基本信息
    plugin_name = PLUGIN_CONFIG.plugin_name
    plugin_version = PLUGIN_CONFIG.plugin_version
    plugin_author = PLUGIN_CONFIG.plugin_author
    enable_plugin = True
    dependencies: list[str] = []
    python_dependencies: list[str] = ["httpx", "requests"]
    config_file_name = "config.toml"

    # MaiBot SDK 通过这些类属性发现配置；定义与渲染由同一个配置 Module 驱动。
    config_file_header = PLUGIN_CONFIG.config_file_header
    config_section_order = PLUGIN_CONFIG.config_section_order
    config_section_group_headers = PLUGIN_CONFIG.config_section_group_headers
    config_hidden_fields = PLUGIN_CONFIG.config_hidden_fields
    config_schema = PLUGIN_CONFIG.config_schema

    def get_default_config(self) -> dict[str, Any]:
        """返回 Runner 首次启动所需的默认配置。"""
        return PLUGIN_CONFIG.default_config()

    def get_webui_config_schema(
        self,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> dict[str, Any]:
        """返回 WebUI 可渲染的配置 Schema。"""
        return PLUGIN_CONFIG.webui_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )

    def __init__(self) -> None:
        """初始化插件实例。"""
        super().__init__()
        self._background_tasks = BackgroundTaskSupervisor(logger=logger)
        self._blocking_io = BlockingIORunner(thread_name_prefix="nai-plugin-io")
        self._http_io = BlockingIORunner(
            thread_name_prefix="nai-http-io",
            max_workers=4,
        )
        self._wd14_io = BlockingIORunner(thread_name_prefix="nai-wd14-io")
        self._active_invocations: WeakSet[NaiInvocation] = WeakSet()
        self._generation_admission_policy = GenerationAdmissionPolicy(
            state=session_state,
            logger=logger,
        )
        # 反推链路：图片缓存与编排服务都在 __init__ 阶段就准备好，避免 HookHandler 在配置加载前触发时 NoneError
        self._image_cache_service: ImageCacheService = ImageCacheService()
        self._reverse_service: ReverseService = ReverseService(wd14_client=None)

    async def on_load(self) -> None:
        """处理插件加载。"""
        self._refresh_runtime_singletons()
        self._refresh_retag_runtime()
        # 主程序 _save_plugin_config 在整文件重写时不会把 ConfigField.description 渲染成注释。
        # 在 on_load 兜底回填一次，保留用户已写入的值，仅在文件里完全没有注释时触发，
        # 避免覆盖用户手写注释。
        try:
            self._regenerate_config_with_comments_if_needed()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"config 注释回填失败（已忽略）：{exc!r}")

    async def on_unload(self) -> None:
        """处理插件卸载。"""
        await self._background_tasks.shutdown()
        self._wd14_io.close()
        self._http_io.close()
        self._blocking_io.close()
        for invocation in list(self._active_invocations):
            invocation.close()
        reset_runtime_recall_tracking_state()
        self._image_cache_service.clear()
        self._refresh_runtime_singletons(reset_only=True)

    async def on_config_update(
        self,
        scope: str | dict[str, object],
        config_data: dict[str, object] | str | None = None,
        version: str = "",
    ) -> None:
        """处理配置热更新。

        兼容两种调用形式：
        1. 新版 Runner：``on_config_update(scope, config_data, version)``
        2. 旧版 SDK：``on_config_update(config_data, version)``
        """
        if isinstance(scope, dict):
            _scope = "self"
        else:
            _scope = scope

        if _scope == "self":
            self._refresh_runtime_singletons()
            self._refresh_retag_runtime()

    def _refresh_runtime_singletons(self, *, reset_only: bool = False) -> None:
        """刷新插件级单例缓存，保证配置热更新后新调用使用最新参数。"""
        online_retriever_api = _load_online_retriever_api()
        reset_tag_retriever()
        if online_retriever_api is not None:
            _, reset_online_retriever = online_retriever_api
            reset_online_retriever()
        if reset_only:
            return

        plugin_config = self.get_plugin_config_data()
        tag_retriever_config = plugin_config.get("tag_retriever")
        if not isinstance(tag_retriever_config, dict):
            return
        if not tag_retriever_config.get("enabled", False):
            return

        mode = str(tag_retriever_config.get("mode", "local") or "local").strip().lower()
        if mode == "online":
            if online_retriever_api is None:
                return
            get_online_retriever, _ = online_retriever_api
            get_online_retriever(
                enabled=True,
                base_url=tag_retriever_config.get("api_url", "https://sakizuki-danboorusearch.hf.space/api"),
                timeout=tag_retriever_config.get("timeout", 90.0),
                search_limit=tag_retriever_config.get("search_limit", 30),
                search_top_k=tag_retriever_config.get("search_top_k", 5),
                related_limit=tag_retriever_config.get("related_limit", 20),
                related_seed_count=tag_retriever_config.get("related_seed_count", 8),
                show_nsfw=tag_retriever_config.get("show_nsfw", True),
                popularity_weight=tag_retriever_config.get("popularity_weight", 0.15),
            )
            return

        get_tag_retriever(
            enabled=True,
            top_k=tag_retriever_config.get("top_k", 50),
            min_score=tag_retriever_config.get("min_score", 0.6),
        )

    def _refresh_retag_runtime(self) -> None:
        """刷新反推链路的运行时单例（图缓存 TTL、WD14 客户端）。"""
        plugin_config = self.get_plugin_config_data()
        retag_config = plugin_config.get("retag") if isinstance(plugin_config, dict) else None
        if not isinstance(retag_config, dict):
            retag_config = {}

        self._image_cache_service.update_config(
            cache_ttl_seconds=float(retag_config.get("cache_ttl_seconds", 3600) or 3600),
            per_stream_capacity=int(retag_config.get("image_cache_per_stream", 20) or 20),
        )

        wd14_enabled = bool(retag_config.get("wd14_enabled", True))
        wd14_threshold = float(retag_config.get("wd14_threshold", 0.35) or 0.35)
        wd14_character_threshold = float(retag_config.get("wd14_character_threshold", 0.8) or 0.8)

        if wd14_enabled:
            spaces_raw = retag_config.get("wd14_spaces")
            spaces_config: list[dict[str, str]] = []
            if isinstance(spaces_raw, list):
                for item in spaces_raw:
                    if isinstance(item, dict) and item.get("name") and item.get("type") and item.get("api"):
                        spaces_config.append(
                            {
                                "name": str(item["name"]),
                                "type": str(item["type"]),
                                "api": str(item["api"]),
                            }
                        )
            wd14_client = WD14Client(
                model=str(retag_config.get("wd14_model", "SmilingWolf/wd-eva02-large-tagger-v3")),
                timeout=float(retag_config.get("wd14_request_timeout", 20.0) or 20.0),
                max_retries=int(retag_config.get("wd14_max_retries", 1) or 1),
                retry_delay=float(retag_config.get("wd14_retry_delay", 0.5) or 0.5),
                spaces_config=spaces_config or None,
                proxy=str(retag_config.get("wd14_proxy", "") or "").strip() or None,
                run_blocking=self._wd14_io.run,
            )
        else:
            wd14_client = None

        self._reverse_service.update_wd14_client(wd14_client)
        self._reverse_service.update_wd14_thresholds(
            threshold=wd14_threshold,
            character_threshold=wd14_character_threshold,
            enabled=wd14_enabled,
        )

    def _close_invocation(self, invocation: NaiInvocation) -> None:
        """关闭一次 Invocation 并从卸载兜底集合移除。"""
        try:
            invocation.close()
        finally:
            self._active_invocations.discard(invocation)

    def _run_invocation_in_background(
        self,
        coroutine_factory: Any,
        *,
        name: str = "nai-invocation",
        invocation: NaiInvocation | None = None,
        on_failure: Any = None,
    ) -> bool:
        """在后台执行一次耗时调用，避免命令 / 工具 RPC 超时。"""
        task = self._background_tasks.start(
            coroutine_factory,
            name=name,
            on_failure=on_failure,
            finalize=(
                lambda: self._close_invocation(invocation)
                if invocation is not None
                else None
            ),
        )
        if task is None and invocation is not None:
            self._close_invocation(invocation)
        return task is not None

    @HookHandler(
        "send_service.after_build_message",
        name="nai_draw_plugin_mark_recall_image",
        description="为本插件图片消息补充撤回标记",
    )
    async def handle_send_service_after_build_message(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在消息发送前写入撤回识别标记。"""
        if not isinstance(message, dict):
            return {"action": "continue"}

        if not attach_plugin_image_marker_to_message(message, NAI_PIC_IMAGE_DISPLAY_MARKER):
            return {"action": "continue"}

        updated_kwargs = dict(kwargs)
        updated_kwargs["message"] = message
        return {"action": "continue", "modified_kwargs": updated_kwargs}

    @HookHandler(
        "send_service.after_send",
        name="nai_draw_plugin_track_recall_image",
        description="记录本插件已成功发送的图片消息ID",
        mode=HookMode.OBSERVE,
    )
    async def handle_send_service_after_send(
        self,
        message: dict[str, Any] | None = None,
        sent: bool = False,
        **kwargs: Any,
    ) -> None:
        """在消息成功发送后记录可撤回的最终消息 ID。"""
        del kwargs

        if not sent or not isinstance(message, dict):
            return None

        if remember_sent_plugin_image_message(message, NAI_PIC_IMAGE_DISPLAY_MARKER):
            self._image_cache_service.cache_inbound_message(message)
        return None

    @HookHandler(
        "chat.receive.before_process",
        name="nai_draw_plugin_retag_receive_image_cache",
        description="缓存入站图片消息，供 /nai 反推 解析引用回复",
        order=HookOrder.EARLY,
    )
    async def handle_retag_receive_before_process(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """监听所有入站消息，把带图的存到 ImageCacheService。"""
        del kwargs
        if isinstance(message, dict):
            self._image_cache_service.cache_inbound_message(message)
        return {"action": "continue"}

    @HookHandler(
        "chat.receive.after_process",
        name="nai_draw_plugin_normalize_reply_command_text",
        description="避免引用回复里的历史 /nai 命令被 MaiBot 后处理文本再次触发",
        order=HookOrder.EARLY,
    )
    async def handle_reply_command_text_after_process(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在命令匹配前把 reply 引用内容从本次命令文本里剥离。"""
        if not isinstance(message, dict):
            return {"action": "continue"}

        normalized_text = normalize_reply_command_text(message)
        if normalized_text is None:
            return {"action": "continue"}

        updated_message = dict(message)
        updated_message["processed_plain_text"] = normalized_text
        updated_kwargs = dict(kwargs)
        updated_kwargs["message"] = updated_message
        return {"action": "continue", "modified_kwargs": updated_kwargs}

    @HookHandler(
        "chat.command.before_execute",
        name="nai_draw_plugin_retag_command_message_cache",
        description="在需要引用图的命令（反推 / i2i / vibe存 / ref存）执行前缓存当前命令消息（保留 reply 信息）",
        order=HookOrder.EARLY,
    )
    async def handle_retag_command_before_execute(
        self,
        message: dict[str, Any] | None = None,
        command_name: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """仅在需要引用图的命令触发前生效，其它命令直接放行。

        /nai vibe 与 /nai ref 已迁移到命名图库，不再走引用图，所以从这个集合里拿掉了。"""
        del kwargs
        if command_name in {
            "nai_retag_command",
            "nai_i2i_command",
            "nai_vibe_save_command",
            "nai_ref_save_command",
        } and isinstance(message, dict):
            self._image_cache_service.remember_command_message(message)
        return {"action": "continue"}

    def _start_image_generation_in_background(
        self,
        stream_id: str,
        coroutine_factory: Any,
        *,
        invocation: NaiInvocation | None = None,
        name: str = "nai-image-generation",
        on_failure: Any = None,
    ) -> bool:
        """在后台启动图片生成任务，并阻止同会话重复启动。"""
        if not stream_id:
            return self._run_invocation_in_background(
                coroutine_factory,
                name=name,
                invocation=invocation,
                on_failure=on_failure,
            )

        pending_owner = session_state.acquire_pending_image_generation(stream_id)
        if pending_owner is None:
            if invocation is not None:
                self._close_invocation(invocation)
            return False

        def _finalize() -> None:
            try:
                if invocation is not None:
                    self._close_invocation(invocation)
            finally:
                session_state.release_pending_image_generation(stream_id, pending_owner)

        task = self._background_tasks.start(
            coroutine_factory,
            name=name,
            on_failure=on_failure,
            finalize=_finalize,
        )
        if task is None:
            _finalize()
            return False
        return True

    async def _start_command_image_generation(
        self,
        stream_id: str,
        coroutine_factory: Any,
        *,
        invocation: NaiInvocation,
    ) -> bool:
        """后台执行显式生图命令，允许同会话内并发处理多个用户请求。"""
        async def _acknowledge() -> bool:
            if not stream_id:
                return True
            return bool(
                await self.ctx.send.text(
                    "收到，正在生成图片，请稍候...",
                    stream_id,
                    storage_message=False,
                )
            )

        return await self._background_tasks.submit(
            coroutine_factory,
            before_start=_acknowledge,
            name="nai-command-generation",
            on_failure=(
                lambda _exc: self.ctx.send.text(
                    "图片生成任务意外中断，请稍后重试。",
                    stream_id,
                    storage_message=False,
                )
                if stream_id
                else None
            ),
            finalize=lambda: self._close_invocation(invocation),
        )

    async def _run_retag(self, *, stream_id: str, user_id: str) -> tuple[bool, str | None, bool]:
        """执行 `/nai 反推`：取目标图 → 反推 → 把结果发回会话。"""
        plugin_config = self.get_plugin_config_data()
        retag_config = plugin_config.get("retag") if isinstance(plugin_config, dict) else None
        if not isinstance(retag_config, dict) or not retag_config.get("enabled", True):
            await self.ctx.send.text("❌ /nai 反推 已在配置中关闭", stream_id, storage_message=False)
            return False, "反推未启用", True

        image_base64 = self._image_cache_service.resolve_image_base64(
            stream_id=stream_id,
            user_id=user_id,
        )
        if not image_base64:
            await self.ctx.send.text(
                "❌ 未找到图片\n请引用回复一张图后发送 /nai 反推，或在同一条消息内发图加命令",
                stream_id,
                storage_message=False,
            )
            return False, "未找到图片", True

        try:
            import base64 as _base64
            payload = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
            image_bytes = _base64.b64decode(payload)
        except Exception as exc:
            await self.ctx.send.text(f"❌ 图片解码失败: {exc}", stream_id, storage_message=False)
            return False, "图片解码失败", True

        await self.ctx.send.text("🔍 正在反推 tag，请稍候...", stream_id, storage_message=False)

        result = await self._reverse_service.reverse(image_bytes)
        if result.source == "failed" or not result.prompt:
            await self.ctx.send.text(
                "❌ 反推失败：" + (result.detail or "未知原因") + "\n（仅 PNG 元数据命中或 WD14 可用时才能拿到 tag）",
                stream_id,
                storage_message=False,
            )
            return False, "反推失败", True

        source_label = {
            "metadata": "📦 PNG 元数据",
            "wd14": "🔍 WD14 在线 Space",
        }.get(result.source, result.source)

        await self.ctx.send.text(
            f"✅ 反推完成（{source_label}，{len(result.tags)} 个 tag）\n\n{result.prompt}\n\n💡 可直接用于 /nai0 <prompt>",
            stream_id,
        )
        return True, "反推成功", True

    def _config_path(self) -> str:
        """返回当前插件实例对应的配置文件路径。"""
        plugin_file = inspect.getfile(self.__class__)
        return os.path.join(os.path.dirname(plugin_file), self.config_file_name)

    def _regenerate_config_with_comments_if_needed(self) -> None:
        """迁移无注释配置，使 Schema 说明进入 TOML。"""
        PLUGIN_CONFIG.regenerate_comments_if_needed(self._config_path())

    async def _load_plugin_config_data(self) -> dict[str, Any]:
        """合并本地配置与宿主运行时覆盖值。"""
        local_config = PLUGIN_CONFIG.load_local(self._config_path())
        runtime_config = await self.ctx.config.get_all()
        return PLUGIN_CONFIG.merge(local_config, runtime_config)

    async def _create_invocation(
        self,
        stream_id: str,
        *,
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        action_data: dict[str, Any] | None = None,
        reasoning: str = "",
        text: str = "",
        source: str = "command",
    ) -> NaiInvocation:
        """构造一次命令或 Action 调用的运行上下文。"""
        if self._background_tasks.is_closing:
            raise RuntimeError("插件正在卸载，拒绝创建新的调用上下文")
        plugin_config = await self._load_plugin_config_data()
        if self._background_tasks.is_closing:
            raise RuntimeError("插件正在卸载，拒绝创建新的调用上下文")
        invocation = NaiInvocation(
            self,
            plugin_config,
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            action_data=action_data,
            reasoning=reasoning,
            text=text,
            source=source,
        )
        self._active_invocations.add(invocation)
        return invocation

    async def _run_foreground_invocation(
        self,
        stream_id: str,
        operation: Callable[[NaiInvocation], Awaitable[InvocationResultT]],
        **invocation_kwargs: Any,
    ) -> InvocationResultT:
        """执行前台 Invocation，并在任意终态释放其资源。"""
        async def _execute() -> InvocationResultT:
            invocation = await self._create_invocation(stream_id, **invocation_kwargs)
            try:
                return await operation(invocation)
            finally:
                self._close_invocation(invocation)

        return await self._background_tasks.run(
            _execute,
            name="nai-foreground-invocation",
        )

    @Command(
        "nai_admin_control_command",
        description="NAI 管理命令：/nai <st|sp|set|art|size|ban|unban|banlist|help>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+(?P<action>st|sp|set|art|size|ban|unban|banlist|help)(?:\s+(?P<param>.+))?$",
    )
    async def handle_nai_admin_control_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai st|sp|set|art|size|help`。"""
        del kwargs
        action = str((matched_groups or {}).get("action", "") or "").strip()
        param = str((matched_groups or {}).get("param", "") or "").strip()
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_admin_command(action, param),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    @Command(
        "nai_recall_control_command",
        description="NAI 自动撤回控制命令：/nai <on|off>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+(?P<action>on|off)$",
    )
    async def handle_nai_recall_control_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai on|off`。"""
        del kwargs
        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_recall_switch(action),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    @Command(
        "nai_nsfw_control_command",
        description="NSFW 内容过滤控制命令：/nai nsfw <on|off>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+nsfw(?:\s+(?P<action>on|off))?$",
    )
    async def handle_nai_nsfw_control_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai nsfw`。"""
        del kwargs
        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_nsfw_command(action),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    @Command(
        "nai_manual_recall_command",
        description="手动撤回图片：/nai 撤回",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+撤回(?:\s+.*)?$",
    )
    async def handle_nai_manual_recall_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai 撤回`。"""
        del kwargs
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.manual_recall(),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    @Command(
        "nai_retag_command",
        description="图片反推：/nai 反推（PNG 元数据 → WD14 兜底，只输出正向 prompt）",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+反推(?:\s+.*)?$",
    )
    async def handle_nai_retag_command(
        self,
        stream_id: str = "",
        user_id: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai 反推`。

        反推链路全部走插件内单例，命令本身不接 Invocation。
        """
        del kwargs
        return await self._background_tasks.run(
            lambda: self._run_retag(stream_id=stream_id, user_id=user_id),
            name="nai-retag-foreground",
        )

    @Command(
        "nai_draw",
        description="使用自然语言描述生成图片；/nai 随机 [角色] 或 /nai 随机自拍 [角色] 可生成开放题材随机色图",
        # negative lookahead 排除所有 /nai 子命令；随机命令同时兼容 `/nai随机` 的紧凑写法；
        # vibe/ref 后面可接 CJK 后缀（存/图库/删/选），
        # 所以用 ``(?:\b|[一-鿿])`` 覆盖空格后置和中文后缀两种情形，避免 ``vibe存`` 被
        # 通用命令吞掉（vibe\b 在 latin→CJK 边界不成立）
        pattern=r"^(?:.*，说：\s*)?/nai(?:\s+|(?=随机(?:自拍)?(?:\s|$)))(?!on$|off$|st$|sp$|set\b|art\b|size\b|ban\b|unban\b|banlist\b|help\b|pt\s|tag\s|nsfw\b|models$|i2i\b|ref(?:\b|[一-鿿])|vibe(?:\b|[一-鿿])|撤回(?:\s|$)|反推(?:\s|$))(?P<description>[\s\S]+)$",
    )
    async def handle_nai_draw(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        text: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai`。"""
        del kwargs
        description = str((matched_groups or {}).get("description", "") or "").strip()

        async def _prepare() -> tuple[bool, str | None, bool]:
            invocation = await self._create_invocation(
                stream_id,
                group_id=group_id,
                user_id=user_id,
                matched_groups=matched_groups,
                text=text,
            )
            try:
                if not await invocation.ensure_generation_permission():
                    self._close_invocation(invocation)
                    return False, "没有权限", True
            except BaseException:
                self._close_invocation(invocation)
                raise
            if not await self._start_command_image_generation(
                stream_id,
                lambda: invocation.handle_nai_draw(description),
                invocation=invocation,
            ):
                return False, "", True
            return True, "已开始生成图片", True

        return await self._background_tasks.run(_prepare, name="nai-draw-submission")

    @Command(
        "nai_0_draw",
        description="直接使用英文标签生成图片",
        # 排除 /nai0 vibe / /nai0 ref 子命令；与 /nai 主命令对齐用 CJK 边界覆盖
        pattern=r"^(?:.*，说：\s*)?/nai0\s+(?!vibe(?:\b|[一-鿿])|ref(?:\b|[一-鿿]))(?P<tags>[\s\S]+)$",
    )
    async def handle_nai_0_draw(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        text: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai0`。"""
        del kwargs
        tags = str((matched_groups or {}).get("tags", "") or "").strip()

        async def _prepare() -> tuple[bool, str | None, bool]:
            invocation = await self._create_invocation(
                stream_id,
                group_id=group_id,
                user_id=user_id,
                matched_groups=matched_groups,
                text=text,
            )
            try:
                if not await invocation.ensure_generation_permission():
                    self._close_invocation(invocation)
                    return False, "没有权限", True
            except BaseException:
                self._close_invocation(invocation)
                raise
            if not await self._start_command_image_generation(
                stream_id,
                lambda: invocation.handle_nai0_draw(tags),
                invocation=invocation,
            ):
                return False, "", True
            return True, "已开始生成图片", True

        return await self._background_tasks.run(_prepare, name="nai0-draw-submission")

    @Command(
        "nai_0_vibe_command",
        description="Vibe Transfer（直发英文 tags 不过 LLM）：/nai0 vibe [@<名字1> [@<名字2>...]] <英文 tags>",
        # 与 /nai vibe 同结构：可选 @<名字>... 单次覆盖，否则用 /nai vibe选 的粘性选定；
        # tags 直接当 prompt 送 NAI（跳过 LLM 翻译，对照 /nai0 的纯英文 tag 习惯）
        pattern=r"^(?:.*，说：\s*)?/nai0\s+vibe\s+(?P<at_names>(?:@\S+\s+)*)(?P<tags>[\s\S]+)$",
    )
    async def handle_nai0_vibe_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_draw_raw_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_0_ref_command",
        description="角色参考（直发英文 tags 不过 LLM）：/nai0 ref [@<名字>] <英文 tags>",
        # ref 固定 1 张参考图，pattern 与 vibe 对齐多 @ 段；store 层 set_selection 上限管 ≤1
        pattern=r"^(?:.*，说：\s*)?/nai0\s+ref\s+(?P<at_names>(?:@\S+\s+)*)(?P<tags>[\s\S]+)$",
    )
    async def handle_nai0_ref_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_draw_raw_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_prompt_show_command",
        description="NAI 提示词显示控制命令：/nai pt <on|off>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+pt\s+(?P<action>on|off)$",
    )
    async def handle_nai_prompt_show_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai pt on|off`。"""
        del kwargs
        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_prompt_show_command(action),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    @Command(
        "nai_tag_retriever_show_command",
        description="Danbooru 检索结果显示控制命令：/nai tag <on|off>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+tag\s+(?P<action>on|off)$",
    )
    async def handle_nai_tag_retriever_show_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai tag on|off`。"""
        del kwargs
        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_tag_retriever_show_command(action),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    @Command(
        "nai_models_command",
        description="拉取 NewAPI 网关实时可用模型列表：/nai models",
        pattern=r"^(?:.*，说：\s*)?/nai\s+models$",
    )
    async def handle_nai_models_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai models`。"""
        del kwargs
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_models_command(),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    @Command(
        "nai_i2i_command",
        description="图生图：/nai i2i <描述>（需引用一张图）",
        # 只跳过完整的引用/图片等方括号组件或旧转述前缀；不能从引用内容里扫描历史 /nai。
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+i2i\s+(?P<description>[\s\S]+)$",
    )
    async def handle_nai_i2i_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai i2i <描述>`：取引用图执行 NewAPI §20.1 i2i 图生图。"""
        del kwargs
        return await self._run_image_to_image_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            mode="i2i",
        )

    @Command(
        "nai_ref_command",
        description="角色参考：/nai ref [@<名字>] <描述>（用图库里的角色参考图，仅 V4.5 模型）",
        # 宽松前缀，同 nai_i2i_command 注释；可选 @<名字>... 单次覆盖，否则用 /nai ref选 的粘性选定
        # ref 最多 1 张：pattern 允许多个 @<名字> 透传，store 层做硬上限校验给统一错误提示
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+ref\s+(?P<at_names>(?:@\S+\s+)*)(?P<description>[\s\S]+)$",
    )
    async def handle_nai_ref_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai ref [@<名字>] <描述>`：从角色参考图库取图执行 NewAPI §20.4。"""
        del kwargs
        return await self._run_named_reference_draw_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_vibe_command",
        description="Vibe Transfer：/nai vibe [@<名字1> [@<名字2>...]] <描述>（用图库里的 vibe 图，最多 4 张）",
        # 宽松前缀，同 nai_i2i_command 注释；可选 @<名字>... 单次覆盖，否则用 /nai vibe选 的粘性选定
        # at_names 用 (?:@\S+\s+)* 整体捕获 0~N 个 @ 前缀，命令层 re.findall 拆解；
        # vibe 最多 4 张走 store 层硬限制，超 4 走统一错误提示
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+vibe\s+(?P<at_names>(?:@\S+\s+)*)(?P<description>[\s\S]+)$",
    )
    async def handle_nai_vibe_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai vibe [@<名字>] <描述>`：从 vibe 图库取图执行 NewAPI §20.3。"""
        del kwargs
        return await self._run_named_reference_draw_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    # ── 命名图库：存 / 图库 / 删 / 选（vibe + ref 8 条对称命令） ──────────

    @Command(
        "nai_vibe_save_command",
        description="把引用回复的图存入 vibe 图库：/nai vibe存 <名字>",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+vibe存\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_vibe_save_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_save_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_vibe_list_command",
        description="列出 vibe 图库的所有命名图：/nai vibe图库",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+vibe图库\s*$",
    )
    async def handle_nai_vibe_list_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_list_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_vibe_delete_command",
        description="从 vibe 图库删除一张命名图：/nai vibe删 <名字>",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+vibe删\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_vibe_delete_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_delete_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_vibe_select_command",
        description="把本会话的默认 vibe 图设为 1~4 张命名图：/nai vibe选 <名字1> [<名字2>...]",
        # 1 ~ N 个名字，空格分隔；store 层会做 vibe ≤ 4 的硬限制
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+vibe选\s+(?P<names>\S+(?:\s+\S+)*)\s*$",
    )
    async def handle_nai_vibe_select_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_select_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_vibe_clear_command",
        description="一键清空 vibe 图库（当前用户）：/nai vibe清空",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+vibe清空\s*$",
    )
    async def handle_nai_vibe_clear_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_clear_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_ref_save_command",
        description="把引用回复的图存入角色参考图库：/nai ref存 <名字>",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+ref存\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_ref_save_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_save_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_list_command",
        description="列出角色参考图库的所有命名图：/nai ref图库",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+ref图库\s*$",
    )
    async def handle_nai_ref_list_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_list_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_delete_command",
        description="从角色参考图库删除一张命名图：/nai ref删 <名字>",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+ref删\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_ref_delete_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_delete_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_select_command",
        description="把本会话的默认角色参考图设为某张命名图：/nai ref选 <名字>",
        # ref 固定最多 1 张，pattern 与 vibe 选保持一致捕获 names 组；store 层若收到 >1 会拒
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+ref选\s+(?P<names>\S+(?:\s+\S+)*)\s*$",
    )
    async def handle_nai_ref_select_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_select_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_clear_command",
        description="一键清空角色参考图库（当前用户）：/nai ref清空",
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+ref清空\s*$",
    )
    async def handle_nai_ref_clear_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_clear_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_type_command",
        description="切换本会话角色参考类型：/nai ref类型 <character|style|both>",
        # 无参时打印当前态；both 是 character&style 的友好别名
        pattern=r"^(?:(?:\[[^\]]*\]\s*)|(?:[^/\n]*，说：\s*))*/nai\s+ref类型(?:\s+(?P<value>\S+))?\s*$",
    )
    async def handle_nai_ref_type_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        value = str((matched_groups or {}).get("value", "") or "").strip()
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_ref_type_command(value),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    async def _run_image_to_image_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        mode: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai i2i 的引用图链路（ref 已迁移到命名图库，不再共享此路径）。"""
        async def _prepare() -> tuple[bool, str | None, bool]:
            description = str((matched_groups or {}).get("description", "") or "").strip()
            image_base64 = self._image_cache_service.resolve_image_base64(
                stream_id=stream_id,
                user_id=user_id,
            )
            if not image_base64:
                await self.ctx.send.text(
                    "❌ 未找到参考图\n请引用回复一张图后再发送 /nai i2i，或在同一条消息内附图加命令",
                    stream_id,
                    storage_message=False,
                )
                return False, "未找到图片", True

            invocation = await self._create_invocation(
                stream_id,
                group_id=group_id,
                user_id=user_id,
                matched_groups=matched_groups,
            )
            try:
                if not await invocation.ensure_generation_permission():
                    self._close_invocation(invocation)
                    return False, "没有权限", True
            except BaseException:
                self._close_invocation(invocation)
                raise

            if not await self._start_command_image_generation(
                stream_id,
                lambda: invocation.handle_image_to_image_draw(
                    description, image_base64=image_base64, mode=mode
                ),
                invocation=invocation,
            ):
                return False, "", True
            return True, "已开始生成图片", True

        return await self._background_tasks.run(_prepare, name="nai-i2i-submission")

    # ── 命名图库 helper（vibe / ref 共用骨架，scope 决定走哪个库） ──────

    async def _run_named_reference_draw_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe / /nai ref 共用：从图库取图（@<名字>... 或粘性选定），背后投递。

        命令 pattern 用 ``(?P<at_names>(?:@\\S+\\s+)*)`` 把 0~N 个 ``@<名字>`` 整体捕获，
        这里 ``re.findall`` 拆成 List[str] 透传给 invocation；空列表退化成 None 走粘性选定。
        """
        async def _prepare() -> tuple[bool, str | None, bool]:
            description = str((matched_groups or {}).get("description", "") or "").strip()
            at_names_str = str((matched_groups or {}).get("at_names", "") or "")
            explicit_names = re.findall(r"@(\S+)", at_names_str) or None

            invocation = await self._create_invocation(
                stream_id,
                group_id=group_id,
                user_id=user_id,
                matched_groups=matched_groups,
            )
            try:
                if not await invocation._ensure_named_reference_admin(scope=scope, action="draw"):
                    self._close_invocation(invocation)
                    return False, "没有管理员权限", True
                if not await invocation.ensure_generation_permission():
                    self._close_invocation(invocation)
                    return False, "没有权限", True
            except BaseException:
                self._close_invocation(invocation)
                raise

            if not await self._start_command_image_generation(
                stream_id,
                lambda: invocation.handle_named_reference_draw(
                    scope=scope,
                    description=description,
                    explicit_names=explicit_names,
                ),
                invocation=invocation,
            ):
                return False, "", True
            return True, "已开始生成图片", True

        return await self._background_tasks.run(
            _prepare,
            name=f"nai-{scope}-draw-submission",
        )

    async def _run_named_reference_save_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe存 / /nai ref存：取引用图存入对应命名图库。"""
        name = str((matched_groups or {}).get("name", "") or "").strip()

        async def _operation(invocation: NaiInvocation) -> tuple[bool, str | None, bool]:
            # 先过管理员鉴权再做图片查找，避免非管理员收到“未找到参考图”误导提示。
            if not await invocation._ensure_named_reference_admin(scope=scope, action="save"):
                return False, "没有管理员权限", True

            image_base64 = self._image_cache_service.resolve_image_base64(
                stream_id=stream_id,
                user_id=user_id,
            )
            if not image_base64:
                scope_cmd = "vibe存" if scope == "vibe" else "ref存"
                await self.ctx.send.text(
                    f"❌ 未找到参考图\n请引用回复一张图后再发送 /nai {scope_cmd} <名字>，"
                    "或在同一条消息内附图加命令",
                    stream_id,
                    storage_message=False,
                )
                return False, "未找到图片", True

            return await invocation.handle_named_reference_save(
                scope=scope,
                name=name,
                image_base64=image_base64,
            )

        return await self._run_foreground_invocation(
            stream_id,
            _operation,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    async def _run_named_reference_list_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe图库 / /nai ref图库。"""
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_named_reference_list(scope=scope),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    async def _run_named_reference_delete_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe删 / /nai ref删。"""
        name = str((matched_groups or {}).get("name", "") or "").strip()
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_named_reference_delete(
                scope=scope,
                name=name,
            ),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    async def _run_named_reference_select_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe选 / /nai ref选：把"空格分隔的多名字"拆成 List[str] 透给 invocation。

        vibe / ref 的 pattern 都用 ``(?P<names>\\S+(?:\\s+\\S+)*)`` 捕获 1~N 个 token，
        store 层会按 scope 的上限（vibe 4 / ref 1）做硬校验，错误统一冒泡。
        """
        names_str = str((matched_groups or {}).get("names", "") or "").strip()
        names = [token for token in names_str.split() if token]
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_named_reference_select(
                scope=scope,
                names=names,
            ),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    async def _run_named_reference_clear_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe清空 / /nai ref清空：一键清空当前用户该 scope 的全部图 + 选定。"""
        return await self._run_foreground_invocation(
            stream_id,
            lambda invocation: invocation.handle_named_reference_clear_all(scope=scope),
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )

    async def _run_named_reference_draw_raw_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai0 vibe / /nai0 ref：用图库里的图 + 用户给的英文 tags，跳过 LLM 翻译。

        与 /nai vibe / /nai ref 的区别仅在于 raw_prompt 透传 — description 同 raw_prompt
        以满足下游空检查；store 层选定 / @<名字...> 单次覆盖、controlnet / character_references
        组装等逻辑全部复用 handle_named_reference_draw 已有路径。
        """
        async def _prepare() -> tuple[bool, str | None, bool]:
            raw_tags = str((matched_groups or {}).get("tags", "") or "").strip()
            at_names_str = str((matched_groups or {}).get("at_names", "") or "")
            explicit_names = re.findall(r"@(\S+)", at_names_str) or None

            invocation = await self._create_invocation(
                stream_id,
                group_id=group_id,
                user_id=user_id,
                matched_groups=matched_groups,
            )
            try:
                if not await invocation._ensure_named_reference_admin(scope=scope, action="draw"):
                    self._close_invocation(invocation)
                    return False, "没有管理员权限", True
                if not await invocation.ensure_generation_permission():
                    self._close_invocation(invocation)
                    return False, "没有权限", True
            except BaseException:
                self._close_invocation(invocation)
                raise

            if not await self._start_command_image_generation(
                stream_id,
                lambda: invocation.handle_named_reference_draw(
                    scope=scope,
                    description=raw_tags,
                    explicit_names=explicit_names,
                    raw_prompt=raw_tags,
                ),
                invocation=invocation,
            ):
                return False, "", True
            return True, "已开始生成图片", True

        return await self._background_tasks.run(
            _prepare,
            name=f"nai0-{scope}-draw-submission",
        )

    @Action(
        "nai_web_draw",
        description=(
            "生成图片/照片/自拍/场景图。"
            "核心是把当前聊天情景视觉化；Bot 出镜不等于自拍，"
            "应根据语境选择自拍、第三视角、POV、远景或纯环境等合适画面。"
            "【参数填写优先级】先把语义拆到正确字段，再调用工具；每个字段只填自己的信息，"
            "禁止把一整段中文画面描述复制到 scene_delta。"
            "action=动作/姿态，emotion=表情/情绪，framing=景别，subject_and_pov=人数与视角，"
            "scene_delta=仅稳定服装和固定环境，dynamic_scene=临时物件/倒影/天气/时间/光线。"
            "既可以响应用户明确的看图请求，也可以在 bot 自己说出视觉自指/进入情感互动节点时主动跟一张图。"
            "【调用语义 - 重要】本 Action 是 fire-and-forget 异步任务："
            "调用成功只代表'图片任务已提交后台'，图片由插件自行通过会话发送，"
            "不会出现在本次 tool_result 的 content 里。"
            "因此：调用本 Action 后，禁止再调用 send_image / 引用本次 call_id 的 media_index，"
            "也禁止调用 wait 等待图片——图片到时会自行送达，按文字正常推进对话即可。"
        ),
        activation_type=ActivationType.ALWAYS,
        parallel_action=True,
        action_parameters={
            # 六个结构化字段：每个字段只承担一类信息，强制 Planner 分维度思考，
            # 避免一锅炖成关键词堆砌。下游会按字段顺序拼成单行 request；若 Planner
            # 兼容性原因只填了 description，则按整段兜底使用。
            "subject_and_pov": (
                "【只填主体与视角】格式：'一女' / '一男一女' / '两女'，可加一个视角。"
                "对方观看 bot=POV；bot 明确举手机/前置相机=自拍；旁观叙事=第三视角。"
                "不要在这里写动作、表情、服装或地点。"
                "【主体身份】本字段需要区分三种情况："
                "(a) bot 自己出镜（包括 bot cos 某角色，出镜的还是 bot）→ 正常写 '一女' 等；"
                "(b) 画一个具体的二次元角色 / 用户点名的非 bot 角色（如'画一张初音未来'）→ "
                "必须在主体前加 token '画指定角色'，例如 '画指定角色 一女 第三视角'。"
                "(c) 画面只有环境或物件、bot 不出镜 → 必须写 token 'Bot不出镜'，"
                "例如 'Bot不出镜 纯环境'。"
                "这些 token 用于告知后端本轮不是 bot 出镜，禁止叠加 bot 外貌锚点。"
                "判断标准：用户/Planner 明确写了具体角色名（初音未来、蕾姆、芙兰朵露等）"
                "或作品角色的 → (b)；纯环境/物件 → (c)；"
                "'画一张自己'/cosplay/泛指人物 → (a)。"
            ),
            "action": (
                "【只填动作与姿态】必须用用户原话/reasoning 里的动词，禁止软化；不要写表情、情绪、"
                "服装、地点或镜头。"
                "如'揉胸'写'揉胸'、不要写'轻捧'；'骑'写'骑乘'、不要写'坐在身上'。"
                "纯静态画面可写'站立'。例如：'坐着，揉腰'，不要写'坐着，揉腰，嘟嘴'。"
            ),
            "emotion": (
                "【只填表情与情绪】必须贴 reasoning 里的当前心境，不要写动作、服装、地点或镜头，"
                "不要默认套'迷离咬唇'。"
                "示例：'不情愿 害羞'、'撒娇 期待'、'紧张 微微低头'、'慵懒 半眯眼'。"
                "无明显情绪可留空。"
            ),
            "scene_delta": (
                "【兼容旧字段】稳定服装与固定环境的综合描述。新调用优先使用 outfit_change / "
                "environment_change 分别声明变化；本轮没有换装/换地点时留空。"
                "若使用本字段，只写稳定设计，不写动作、表情、姿态、镜头或临时散落物。"
            ),
            "outfit_change": (
                "【Planner 先判定服装变化，只填枚举】unchanged=服装没变（默认）；clear=本轮不可见；"
                "switch=换回之前穿过的某套（可写 switch:<服装库 key>，key 见上一次画图回执的"
                "'视觉连续性状态'；不记得 key 就只写 switch，把口语描述放进 outfit_new_look）；"
                "replace=聊天中明确建立了新服装。"
                "unchanged 时下游逐字复用上一轮服装 Tag，不重新翻译；"
                "不要因为动作或构图变化而 replace；拿不准就写 unchanged。"
            ),
            "outfit_new_look": (
                "【仅 outfit_change=replace/switch 时填】replace：新服装的完整中文描述，"
                "具体保留用户说出的款式、颜色、材质和可见细节；"
                "switch：想切回哪套的口语描述（如'之前那套白色连衣裙'）。其余情况留空。"
            ),
            "environment_change": (
                "【Planner 先判定环境变化，只填枚举】unchanged=地点没变（默认）；clear=纯人物无背景；"
                "switch=回到之前出现过的地点（可写 switch:<环境库 key>；不记得 key 就只写 switch，"
                "把口语描述放进 environment_new_look）；replace=聊天中明确换了新地点。"
                "unchanged 时下游逐字复用上一轮环境 Tag；"
                "动态灯光、天气、烟雾、倒影和临时散落物放 dynamic_scene，不算环境变化。"
            ),
            "environment_new_look": (
                "【仅 environment_change=replace/switch 时填】replace：新固定环境的完整中文描述"
                "（空间布局、固定物件、材质、配色）；switch：想回到哪个地点的口语描述。其余情况留空。"
            ),
            "dynamic_scene": (
                "*【只填动态场景】* 本轮短暂变化的临时物件、倒影、天气、时间、灯光、烟雾、"
                "凌乱物品和即时氛围；不要写服装、固定家具、人物名、动作、表情或景别。"
                "例如：'镜中倒影，夜间暖色灯光，化妆桌上散落的刷子和口红'。"
                "这些内容不进入稳定环境卡片，每轮可以变化。"
            ),
            "framing": (
                "【只填景别/构图】只写一个景别或构图词：近景/特写/中景/全身/胸部以上/俯视/仰视/侧面。"
                "第三视角、POV、自拍只写在 subject_and_pov；不要在这里重复视角，也不要写动作、表情或场景。"
            ),
            "description": (
                "兜底字段，正常留空。"
                "只有当本轮内容无法拆进上面 6 个字段时，才在这里写一行完整关键词串。"
                "格式：人数 + 视角 + 动作 + 情绪 + 稳定服装/环境 + 动态场景 + 构图；禁写外貌锚点和画质词。"
                ),
            "size": "图片尺寸（默认从配置获取）",
        },
        # 用星号把最容易串字段的规则提升到 Planner 注意力最高的区域，并给出可直接模仿的拆分样例。
        action_require=[
            "*【字段边界是硬约束，先拆分再调用】*",
            "错误示例：action='坐着 揉腰 嘟嘴'、framing='中景 第三视角'、scene_delta='夜场后台化妆间……她揉腰……丝袜……'。",
            "正确示例：subject_and_pov='一女 第三视角'；action='坐着，揉腰'；emotion='羞恼，脸红，嘟嘴，疲惫'；",
            "framing='中景'；scene_delta='亮银色弹力亮片面料的高腰修身短裙，窄版剪裁，黑色细腰带，半透明黑色丝袜；"
            "后台化妆间固定镜台、整面镜子、木质抽屉柜和米白墙面'；",
            "dynamic_scene='镜中倒影，夜间暖色灯光，化妆桌上散落刷子和口红'。",
            "*scene_delta 不得包含动作/表情/倒影/临时物件；framing 不得包含 POV/第三视角；嘟嘴必须在 emotion。*",
            "*必须由 Planner 先分别决定 outfit_change 和 environment_change（只填枚举，新内容写进 "
            "outfit_new_look / environment_new_look）；Tag LLM 不得猜测稳定区是否变化。*",
            "可以触发的典型时机：",
            "1. 用户明确要求看图/画图/发图/自拍/肖像/再来一张",
            "2. 用户明确想看 bot 本人的样子、穿搭、状态、某个身体/服饰视觉重点",
            "3. bot 这一轮要回复的话里包含自身姿态、穿着、动作、所处场景的视觉描写"
            "（例：我刚洗完澡靠在窗边、今天穿了新裙子、在便利店买东西、慵懒地躺在床上）"
            "——这种时机配一张图比纯文字更自然，可以主动跟一张",
            "4. 用户分享情绪、晚安、回家了、到家了、想你了、撒娇等亲密互动节点，"
            "bot 想以一张近照/自拍作为情感回应",
            "不触发：纯知识问答、技术讨论、bot 这一轮明显走理性解释/代码/列点风格的回复，"
            "或者用户明确拒绝出图（'不要画''别画图'）",
            "节奏建议：自然搭图，不刷屏。短间隔内连发要克制；但 bot 自己开口提到视觉细节时不要犹豫——"
            "比起'刚发过图，先不发'，更应该判断'这句话本身配图是否自然'。",
        ],
        associated_types=["text"],
    )
    async def handle_nai_web_draw(
        self,
        stream_id: str = "",
        user_id: str = "",
        group_id: str = "",
        action_data: dict[str, Any] | None = None,
        reasoning: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """处理自动生图 Action。"""
        del kwargs

        async def _prepare() -> tuple[bool, str]:
            invocation = await self._create_invocation(
                stream_id,
                user_id=user_id,
                group_id=group_id,
                action_data=action_data,
                reasoning=reasoning,
                source="action",
            )
            try:
                if not await invocation.ensure_user_not_blacklisted():
                    self._close_invocation(invocation)
                    return False, "黑名单用户"

                guard_state = await invocation.preflight_action_guard()
                if guard_state is not None and not guard_state.should_generate:
                    self._close_invocation(invocation)
                    return False, guard_state.detail
            except BaseException:
                self._close_invocation(invocation)
                raise

            if not self._start_image_generation_in_background(
                stream_id,
                invocation.handle_action,
                invocation=invocation,
                name="nai-action-generation",
                on_failure=(
                    lambda _exc: self.ctx.send.text(
                        "图片生成任务意外中断，请稍后重试。",
                        stream_id,
                        storage_message=False,
                    )
                    if stream_id
                    else None
                ),
            ):
                return False, (
                    "同会话已有图片任务在后台进行中，本轮跳过出图、按文字回复推进；"
                    "请不要调用 send_image 或 wait，正在生成的那张图会自行送达"
                )
            return True, (
                "图片任务已提交后台，图片由插件异步发送到会话，本次 tool_result 不包含 image 内容；"
                "请不要调用 send_image 引用本次 call_id，也不要 wait，按文字正常推进对话即可"
                + invocation.render_visual_state_for_planner()
            )

        return await self._background_tasks.run(_prepare, name="nai-action-submission")


def create_plugin():
    """创建新版 SDK 插件实例。"""
    return NaiPicPlugin()
