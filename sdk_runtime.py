"""NAI Low 插件新版 SDK 运行辅助。

将旧版命令与 Action 的主要业务逻辑迁移到新版 `MaiBotPlugin` 调用方式。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any, Dict, List, Optional

import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from src.common.logger import get_logger

from .runtime_recall import (
    discard_pending_plugin_image_send,
    normalize_db_timestamp,
    remember_pending_plugin_image_send,
)

from .core.clients.nai_web_client import NaiWebClient
from .core.constants import NAI_PIC_IMAGE_DISPLAY_MARKER
from .core.mixins.model_config_mixin import ModelConfigMixin
from .core.rules.selfie_rules import (
    detect_bot_self_image_intent,
    detect_selfie_from_output,
    merge_selfie_prompt,
)
from .core.services.generation_admission_policy import AdmissionDecision
from .core.services.llm_text_generator import MaiBotLLMTextGenerator
from .core.services.prompt_generation_workflow import PromptGenerationWorkflow
from .core.services.random_scene_planner import RandomScenePlanner
from .core.services.recall_workflow import RecallWorkflow
from .core.services.session_state import session_state
from .core.services.user_blacklist import user_blacklist
from .core.services.named_reference_store import (
    CapacityExceededError as _NamedRefCapacityExceededError,
    InvalidImageError as _NamedRefInvalidImageError,
    InvalidNameError as _NamedRefInvalidNameError,
    OWNER_GROUP as _NAMED_OWNER_GROUP,
    OWNER_USER as _NAMED_OWNER_USER,
    SCOPE_REF as _NAMED_SCOPE_REF,
    SCOPE_VIBE as _NAMED_SCOPE_VIBE,
    get_named_reference_store,
    max_selection_for_scope as _max_selection_for_scope,
)
from .core.utils.action_payload import (
    STRUCTURED_DESCRIPTION_FIELDS,
    compose_description_from_action_payload,
    is_named_character_intent,
)
from .core.utils.display_message_helper import build_action_image_display_message
from .core.utils.help_renderer import HELP_FALLBACK_TEXT as _HELP_FALLBACK_TEXT
from .core.utils.image_meta import (
    normalize_image_base64 as _normalize_image_for_payload,
    read_image_dimensions as _read_image_dimensions,
)
from .core.utils.prompt_postprocessor import (
    normalize_characters_order,
    normalize_prompt_order,
    remove_selfie_appearance_tags,
    sanitize_sfw_characters,
    sanitize_sfw_prompt,
    strip_cjk_and_fullwidth,
    strip_cjk_and_fullwidth_from_characters,
    user_mentions_appearance,
)
from .core.utils.random_scene_description import parse_random_scene_request

logger = get_logger("nai_draw_plugin")


def _scope_label(scope: str) -> str:
    """把 ``vibe`` / ``ref`` 翻译成 user-facing 中文标签，供命名图库的提示文本使用。"""
    if scope == _NAMED_SCOPE_VIBE:
        return "Vibe"
    if scope == _NAMED_SCOPE_REF:
        return "角色参考"
    return scope


def _get_nested_config_value(config_data: dict[str, Any], key: str, default: Any = None) -> Any:
    """从插件配置中读取点分路径。"""
    current: Any = config_data
    for part in str(key or "").split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _extract_message_field(message: Any, field: str) -> Any:
    """兼容字典消息的字段读取。"""
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def _text_looks_like_image(text: Any) -> bool:
    """判断文本是否像图片消息。"""
    if not isinstance(text, str):
        return False
    normalized = text.strip()
    if not normalized:
        return False
    return normalized.startswith(("[图片", "[NAI图片", "[image", "[imageurl", "[picid", "picid:"))


def _looks_like_generation_request_url(url: Any) -> bool:
    """识别误被当成图片直链的生成接口 URL。"""
    if not isinstance(url, str):
        return False

    normalized = url.strip()
    if not normalized.startswith(("http://", "https://")):
        return False

    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return False

    path = parsed.path.rstrip("/").lower()
    if not path.endswith("/generate"):
        return False

    query = parsed.query.lower()
    return any(
        token in query
        for token in (
            "tag=",
            "model=",
            "negative=",
            "artist=",
            "token=",
            "sampler=",
            "steps=",
            "cfg=",
            "scale=",
            "size=",
        )
    )


def _is_image_message(message: Any) -> bool:
    """判断消息是否为图片。"""
    if isinstance(message, dict):
        if message.get("is_picid") or message.get("is_picture"):
            return True
        segment = message.get("message_segment")
        if isinstance(segment, dict):
            segment_type = segment.get("type")
            if segment_type in {"image", "imageurl"}:
                return True
            if segment_type == "seglist":
                for child in segment.get("data") or []:
                    if isinstance(child, dict) and child.get("type") in {"image", "imageurl"}:
                        return True
        for key in ("processed_plain_text", "display_message", "raw_message"):
            if _text_looks_like_image(message.get(key)):
                return True
        return False

    if getattr(message, "is_picid", False) or getattr(message, "is_picture", False):
        return True
    for key in ("processed_plain_text", "display_message", "raw_message"):
        if _text_looks_like_image(getattr(message, key, None)):
            return True
    return False


def _row_age_seconds(row: Any) -> float | None:
    """根据消息行的 timestamp 字段返回距今秒数；解析失败返回 None。"""
    if isinstance(row, dict):
        raw_ts = row.get("timestamp")
    else:
        raw_ts = getattr(row, "timestamp", None)
    normalized = normalize_db_timestamp(raw_ts)
    if normalized is None:
        return None
    return max(0.0, time.time() - float(normalized))


# 主动出图时往 description 前置的自指标签。这里只声明 bot 本人出镜意图，
# 景别和视角交由提示词规则及用户请求决定，避免把全身请求预先压成近景。
_SELF_IMAGE_HINT_BY_MODE: dict[str, str] = {
    "selfie": "一女 自拍",
    "portrait": "一女 肖像照",
    "scene": "一女 生活照",
}


def _inject_self_image_hint(description: str, *, mode: str) -> str:
    """把对应模式的 self-image 标签拼到 description 前面，避免后续 LLM 改写丢失意图。

    已经包含人数（"一女" "一男一女"等）则不重复加。
    """
    hint = _SELF_IMAGE_HINT_BY_MODE.get(mode, _SELF_IMAGE_HINT_BY_MODE["portrait"])
    desc = (description or "").strip()
    if not desc:
        return hint
    # 若 description 已经写了人数前缀（一女/二女/一男一女/两女 等），不要重复堆叠
    leading_persona_pattern = re.compile(r"^(?:一|二|两|三|1|2|3)(?:女|男|男一女|女一男)\b")
    if leading_persona_pattern.match(desc):
        # 仅在 hint 的"非人数部分"还没出现在 desc 时追加
        hint_tail = " ".join(hint.split()[1:]).strip()
        if hint_tail and hint_tail not in desc:
            return f"{desc} {hint_tail}"
        return desc
    return f"{hint} {desc}"


def _extract_message_sender_id(message: Any) -> str:
    """从消息行（dict 或对象）中提取发送者 user_id。"""
    if isinstance(message, dict):
        direct = message.get("user_id")
        if direct:
            return str(direct)
        for nested_key in ("user_info", "message_info"):
            nested = message.get(nested_key)
            if isinstance(nested, dict):
                # message_info 自己可能再嵌一层 user_info
                if nested_key == "message_info":
                    mi_user_info = nested.get("user_info")
                    if isinstance(mi_user_info, dict):
                        user_id = mi_user_info.get("user_id")
                        if user_id:
                            return str(user_id)
                else:
                    user_id = nested.get("user_id")
                    if user_id:
                        return str(user_id)
        return ""

    user_info_obj = getattr(message, "user_info", None)
    user_id_obj = getattr(user_info_obj, "user_id", None) if user_info_obj else None
    if user_id_obj:
        return str(user_id_obj)
    legacy = getattr(message, "user_id", None)
    return str(legacy) if legacy else ""


def _resolve_bot_account(platform: str) -> str:
    """读取当前 bot 的账号 ID，用于把 bot 自己发的消息排除。

    保持最小依赖：直接读 ``global_config.bot``，QQ 用 ``qq_account``，其他平台
    优先用 ``platforms`` 映射，否则回落到 ``qq_account``。重型的 platform 解析
    工具不在这里调用，避免拖入额外模块。
    """
    try:
        from src.config.config import global_config  # 延迟导入，避免测试时拖入重模块
    except Exception:
        return ""
    bot_config = getattr(global_config, "bot", None)
    if not bot_config:
        return ""

    qq_account = str(getattr(bot_config, "qq_account", "") or "").strip()
    telegram_account = str(getattr(bot_config, "telegram_account", "") or "").strip()
    platform_key = (platform or "").strip().lower()

    # 从 platforms 配置中提取（结构可能是 list[dict] 也可能已被解析为映射）
    platforms_raw = getattr(bot_config, "platforms", None) or []
    if isinstance(platforms_raw, dict):
        for k, v in platforms_raw.items():
            if str(k).strip().lower() == platform_key and v:
                return str(v).strip()
    elif isinstance(platforms_raw, list):
        for item in platforms_raw:
            if isinstance(item, dict):
                name = str(item.get("platform") or item.get("name") or "").strip().lower()
                account = item.get("account") or item.get("id") or item.get("user_id")
                if name == platform_key and account:
                    return str(account).strip()

    if platform_key in {"qq"} and qq_account:
        return qq_account
    if platform_key in {"telegram", "tg"} and telegram_account:
        return telegram_account
    return qq_account


class NaiInvocation(ModelConfigMixin):
    """一次命令或 Action 调用的上下文封装。"""

    def __init__(
        self,
        plugin: Any,
        plugin_config: dict[str, Any],
        stream_id: str,
        *,
        group_id: str = "",
        user_id: str = "",
        matched_groups: Optional[dict[str, str]] = None,
        action_data: Optional[dict[str, Any]] = None,
        reasoning: str = "",
        text: str = "",
        source: str = "command",
    ) -> None:
        self.plugin = plugin
        self.ctx = plugin.ctx
        self.plugin_config = plugin_config
        self.stream_id = str(stream_id or "")
        self.group_id = str(group_id or "")
        self.user_id = str(user_id or "")
        self.matched_groups = matched_groups or {}
        self.action_data = action_data or {}
        self.reasoning = str(reasoning or "")
        self.text = str(text or "")
        self.source = source
        self.log_prefix = "nai_draw_plugin"
        self._generation_admission_policy = plugin._generation_admission_policy
        self.api_client = NaiWebClient(
            log_prefix=self.log_prefix,
            run_blocking=self.plugin._http_io.run,
        )
        text_generator = MaiBotLLMTextGenerator(self.log_prefix)
        self._prompt_generation_workflow = PromptGenerationWorkflow(
            config=self.plugin_config,
            stream_id=self.stream_id,
            text_generator=text_generator,
            send_text=self.send_text,
            show_tag_candidates=self._is_tag_retriever_show_enabled(),
            log_prefix=self.log_prefix,
        )
        random_scene_config = self.get_config("random_scene", {})
        self._random_scene_planner = RandomScenePlanner(
            config=random_scene_config if isinstance(random_scene_config, dict) else {},
            text_generator=text_generator,
            log_prefix=self.log_prefix,
        )
        self._recall_workflow = RecallWorkflow(
            config=self.plugin_config,
            stream_id=self.stream_id,
            context=self.ctx,
            send_text=self.send_text,
            start_task=self.plugin._background_tasks.start,
            log_prefix=self.log_prefix,
        )
        self._last_send_timestamp: float | None = None
        # Action Guard 评估缓存：主路径同步预检后，后台 handle_action 复用结果，避免重复读消息库
        self._cached_action_trigger_assessment: AdmissionDecision | None = None

    def close(self) -> None:
        """释放当前调用持有的可关闭资源。"""
        self.api_client.close()

    def get_config(self, key: str, default: Any = None) -> Any:
        """兼容旧逻辑的同步配置读取接口。"""
        return _get_nested_config_value(self.plugin_config, key, default)

    def _get_chat_identity(self) -> tuple[str, str, str]:
        """返回兼容旧状态管理的会话标识。

        新版 SDK Command/Action 目前不会直接注入平台信息，这里统一使用
        `stream` 作为逻辑平台，并用 `stream_id` 作为会话主键。
        """
        chat_id = self.stream_id or self.user_id
        return "stream", chat_id, self.user_id

    def _get_target_platform(self) -> str:
        """读取当前发送目标的平台标识。"""
        if not self.stream_id:
            return ""

        try:
            from src.chat.message_receive.chat_manager import chat_manager

            session = chat_manager.get_existing_session_by_session_id(self.stream_id)
            if session is None:
                session = chat_manager.get_session_by_session_id(self.stream_id)
        except Exception as exc:
            logger.debug("%s 读取目标平台失败: %r", self.log_prefix, exc)
            return ""

        return str(getattr(session, "platform", "") or "").strip().lower()

    async def send_text(self, text: str, storage_message: bool = True) -> bool:
        """发送文本。"""
        if not self.stream_id:
            return False
        return bool(await self.ctx.send.text(text, self.stream_id, storage_message=storage_message))

    async def send_custom(
        self,
        message_type: str,
        content: Any,
        *,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        """发送自定义消息。"""
        if not self.stream_id:
            return False
        return bool(
            await self.ctx.send.custom(
                message_type,
                content,
                self.stream_id,
                display_message=display_message,
                storage_message=storage_message,
            )
        )

    async def send_command(
        self,
        command: str,
        data: dict[str, Any],
        *,
        display_message: str = "",
        storage_message: bool = True,
    ) -> Any:
        """发送平台命令。"""
        if not self.stream_id:
            return False
        return await self.ctx.send.command(
            command,
            self.stream_id,
            data=data,
            display_message=display_message,
            storage_message=storage_message,
        )

    @property
    def action_name(self) -> str:
        """兼容旧 Action 的名称访问。"""
        return "nai_web_draw"

    def _build_image_display_message(self, description: str = "") -> str:
        """构造可供撤回逻辑识别的展示文案。"""
        readable = build_action_image_display_message(description)
        return f"{NAI_PIC_IMAGE_DISPLAY_MARKER} {readable}".strip()

    def _chat_type_text(self) -> str:
        """返回用户可读的聊天类型。"""
        return "群聊" if self.group_id else "私聊"

    def _named_reference_owner(self) -> tuple[str, str]:
        """命名图库（vibe / ref）的归属维度：群聊共享群图库，私聊按 user 隔离。

        返回 ``(owner_kind, owner_id)``：群聊 ``("group", group_id)``，
        私聊 ``("user", user_id)``。这样同一群聊里所有成员共用一份图库，
        修复了"群里每个人各存各的图"的历史 bug。
        """
        if self.group_id:
            return _NAMED_OWNER_GROUP, self.group_id
        return _NAMED_OWNER_USER, self.user_id

    def _check_user_permission(self) -> bool:
        """检查当前用户是否有权限触发生图。"""
        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            return True
        if not user_id:
            return True
        return session_state.check_user_permission(platform, chat_id, user_id, self.get_config)

    async def ensure_generation_permission(self) -> bool:
        """检查当前用户是否有权限使用生图能力，并在失败时返回提示。"""
        if not await self.ensure_user_not_blacklisted():
            return False

        if self._check_user_permission():
            return True

        await self.send_text(
            "❌ 当前会话已开启管理员模式，仅管理员可以使用 NAI 生图功能",
            storage_message=False,
        )
        return False

    async def ensure_user_not_blacklisted(self) -> bool:
        """检查当前用户是否被插件黑名单封禁。"""
        if not self.user_id:
            return True
        if not user_blacklist.is_blacklisted(self.user_id):
            return True

        await self.send_text(
            "❌ 你已被加入 NAI 插件黑名单，无法使用本插件任何功能",
            storage_message=False,
        )
        return False

    def _is_prompt_show_enabled(self) -> bool:
        """检查是否开启提示词显示。"""
        platform, chat_id, _ = self._get_chat_identity()
        if not chat_id:
            return False
        return session_state.is_prompt_show_enabled(platform, chat_id, self.get_config)

    def _is_tag_retriever_show_enabled(self) -> bool:
        """检查是否开启 Danbooru 检索结果显示。"""
        platform, chat_id, _ = self._get_chat_identity()
        if not chat_id:
            return False
        return session_state.is_tag_retriever_show_enabled(platform, chat_id, self.get_config)

    def _sanitize_prompt_for_sfw_mode(self, prompt: str) -> str:
        """LLM 翻译完到送 API 之间的最后清洗钩子：

        1. 启用 NSFW 过滤时进一步剔除擦边/色情标签（受 stream 级开关控制）；
        2. **无条件**剔除 LLM 残留的 CJK 字符与全角符号——NewAPI §8 明确要求
           prompt / negative_prompt 必须英文，含 CJK 一律 400。SFW 不开也必须清。
        """
        if not prompt:
            return prompt
        if session_state.is_nsfw_filter_enabled("stream", self.stream_id, self.get_config):
            prompt = sanitize_sfw_prompt(prompt)
        return strip_cjk_and_fullwidth(prompt)

    def _sanitize_structured_for_sfw_mode(
        self,
        structured: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """送 API 前多角色 payload 的最后清洗钩子。

        与 _sanitize_prompt_for_sfw_mode 行为对称：先按 SFW 开关做擦边过滤，再做无条件
        CJK + 全角清洗。任一步骤后角色数 < 2 都返回 None 触发字符串降级，避免把
        "只剩 1 个角色"的不合规 payload 硬送上游。
        """
        if not structured:
            return None

        global_text = structured.get("global_text", "")
        characters = structured.get("characters") or []

        if session_state.is_nsfw_filter_enabled("stream", self.stream_id, self.get_config):
            global_text, characters = sanitize_sfw_characters(global_text, characters)
            if len(characters) < 2:
                logger.info(
                    f"{self.log_prefix} SFW 过滤后多角色 payload 剩余 {len(characters)} 项，"
                    "降级回单字符串路径"
                )
                return None

        global_text, characters = strip_cjk_and_fullwidth_from_characters(
            global_text, characters
        )
        if len(characters) < 2:
            logger.info(
                f"{self.log_prefix} CJK 清洗后多角色 payload 剩余 {len(characters)} 项，"
                "降级回单字符串路径"
            )
            return None
        return {**structured, "global_text": global_text, "characters": characters}

    def _normalize_structured_order(
        self,
        structured: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """按 normalize_prompt_order 规则整理多角色 payload。"""
        if not structured:
            return None
        new_global, new_chars = normalize_characters_order(
            structured.get("global_text", ""),
            structured.get("characters") or [],
        )
        return {**structured, "global_text": new_global, "characters": new_chars}

    @staticmethod
    def _select_send_payload(
        prompt: str,
        structured: Optional[dict[str, Any]],
    ) -> tuple[str, Optional[List[Dict[str, Any]]]]:
        """根据是否存在合法结构化 payload，返回送往 generate_image 的 (prompt, characters)。

        结构化路径下 ``prompt`` 用 ``global_text`` 单段；字符串路径下沿用拍平后的字符串，
        ``characters`` 为 ``None``。
        """
        if structured and len(structured.get("characters") or []) >= 2:
            return structured.get("global_text", "") or prompt, list(structured["characters"])
        return prompt, None

    async def _find_recent_messages(self, limit: int = 120, hours: float = 24.0) -> list[Any]:
        """读取当前会话最近消息。

        优先通过 database.query 直接查库（避免 message.get_recent 的 datetime 序列化问题），
        失败时回退到 message.get_recent。
        """
        if not self.stream_id:
            logger.debug("%s _find_recent_messages: stream_id 为空", self.log_prefix)
            return []

        # 方式1: 直接查数据库，绕过 _serialize_messages 的 datetime bug
        try:
            db_result = await self.ctx.call_capability(
                "database.query",
                model_name="Messages",
                query_type="get",
                filters={"session_id": self.stream_id},
                order_by=["-timestamp"],
                limit=limit,
            )
            if isinstance(db_result, dict) and db_result.get("success"):
                rows = db_result.get("result")
                if isinstance(rows, list) and rows:
                    logger.debug("%s 通过 database.query 获取到 %d 条消息", self.log_prefix, len(rows))
                    return rows
        except Exception as exc:
            logger.debug("%s database.query 方式获取消息失败: %r", self.log_prefix, exc)

        # 方式2: 回退到 message.get_recent（可能因 datetime 序列化失败）
        try:
            result = await self.ctx.call_capability(
                "message.get_recent",
                chat_id=self.stream_id,
                limit=limit,
                hours=hours,
                filter_mai=False,
            )
        except Exception as exc:
            logger.warning("%s 获取最近消息失败（可能是序列化问题）: %r", self.log_prefix, exc)
            return []
        if isinstance(result, dict):
            if not result.get("success", True):
                logger.warning("%s 获取最近消息返回失败: %s", self.log_prefix, result.get("error", "未知"))
            messages = result.get("messages")
            if isinstance(messages, list):
                return messages
        if isinstance(result, list):
            return result
        return []

    async def _fetch_last_user_text(self, *, lookback: int = 6) -> str:
        """从最近消息中取一条真实用户原话，供 Action Guard 关键词分级。"""
        text, _ = await self._fetch_last_user_text_with_age(lookback=lookback)
        return text

    async def _fetch_last_user_text_with_age(
        self,
        *,
        lookback: int = 6,
    ) -> tuple[str, float | None]:
        """同 ``_fetch_last_user_text``，但额外返回消息距今的秒数（None 表示未知）。

        Action 入口拿到的 action_data["description"] 是 Planner LLM 生成的关键词串，
        不是用户原话。这里回查消息库，跳过 bot 自己的消息与图片消息，取最新一条
        用户文本及其发生时间，供调用方做弱否定关键词的 staleness 判断。
        """
        if not self.stream_id:
            return "", None

        platform = self._get_target_platform()
        bot_account = _resolve_bot_account(platform)

        rows = await self._find_recent_messages(limit=max(2, lookback) * 3, hours=2.0)
        if not rows:
            return "", None

        for row in rows:
            if _is_image_message(row):
                continue
            sender_id = _extract_message_sender_id(row)
            if bot_account and sender_id and sender_id == bot_account:
                continue
            for key in ("processed_plain_text", "display_message", "raw_message"):
                value = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        return text, _row_age_seconds(row)
        return "", None

    async def _schedule_auto_recall(self) -> None:
        """把图片发送结果适配到召回工作流。"""
        platform, chat_id, _ = self._get_chat_identity()
        enabled = bool(
            chat_id and session_state.is_recall_enabled(platform, chat_id, self.get_config)
        )
        await self._recall_workflow.schedule_auto_recall(
            enabled=enabled,
            send_timestamp=self._last_send_timestamp,
        )

    async def _download_remote_image_as_base64(self, url: str) -> str | None:
        """下载远程图片并转为 Base64。"""
        if _looks_like_generation_request_url(url):
            logger.warning("%s 远程图片URL仍是生成接口，停止自动补拉以避免重复扣费", self.log_prefix)
            return None

        try:
            model_config = self._get_model_config()
            if not isinstance(model_config, dict):
                model_config = {}
            content = await self.api_client.download_image_bytes(url, model_config)
        except requests.RequestException as exc:
            logger.error("%s 下载远程图片失败: %r", self.log_prefix, exc, exc_info=True)
            return None
        except Exception as exc:
            logger.error("%s 下载远程图片异常: %r", self.log_prefix, exc, exc_info=True)
            return None

        if not content:
            return None

        return base64.b64encode(content).decode("utf-8")

    async def _send_base64_image_result(
        self, image_base64: str, display_message: str, *, image_description: str = ""
    ) -> bool:
        """以 Base64 image 段直发图片到平台。

        maim_message + napcat 协议原生支持 base64 image segment，napcat 自行落盘
        后再投递；插件无需也不应当依赖 ``file://`` 本地路径——一旦 napcat 与本插件
        不在同一文件系统（如 napcat 跑在容器内），``file://`` 引用就无法被读取。
        """
        sent = await self.send_custom(
            "image",
            image_base64,
            display_message=display_message,
        )
        # 命令 / 自动跟图发出的图：发送（已 await 入库）后立刻在图片库标记“已识别”，
        # 让 MaiBot 后续不再对这张图触发 VLM 识图；action（麦麦主动画图）不在此列。
        if self._skip_self_vlm():
            await self._register_self_image_as_processed(image_base64, image_description)
        return sent

    def _skip_self_vlm(self) -> bool:
        """是否跳过 MaiBot 对“本插件这次发出的图”的 VLM 识图。

        仅 ``action``（LLM 工具调用 handle_action）保留识图，让麦麦能“看见”自己在对话里
        主动画的图；命令（/nai 等）与自动跟图（reply_auto_draw）都是用户点单或配图的产物，
        描述已知、无需再过 VLM，一律跳过。
        """
        return self.source != "action"

    async def _register_self_image_as_processed(self, image_base64: str, description: str) -> None:
        """把本插件刚发出的图片在 MaiBot 图片库标记为“已识别”，从源头省掉 VLM 识图。

        根因：插件经 ``send.custom("image", ...)`` 发图时，宿主构造的 ImageComponent 不带
        content，落库时 ``vlm_processed=False``；此后任何对该图的 ``get_image_description``
        都会触发后台 VLM。这里在发送（send_service 已 await 写库）后，按宿主同款
        ``sha256(原始字节)`` 定位该图记录，写入已知描述并置 ``vlm_processed=True``，使后续
        识图直接命中缓存、不再调 VLM。``description`` 为空时兜底为占位，确保命中缓存所需的
        “描述非空”条件成立。

        尽力而为：图未入库（imageurl 直发、storage_message=False 等）时查不到记录直接跳过；
        本回写属于发送主链外的优化边路，图片此刻已送达，故异常只记日志、不向上抛。
        """
        payload = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
        try:
            image_bytes = base64.b64decode(payload)
        except (binascii.Error, ValueError) as exc:
            logger.warning("%s 跳过识图回写终止：图片 Base64 解码失败 %r", self.log_prefix, exc)
            return
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        final_description = description.strip() or "[由 NovelAI 生成的图片]"

        def _mark_processed() -> bool:
            from src.chat.image_system.image_manager import image_manager

            record = image_manager.get_image_from_db(image_hash)
            if record is None:
                return False
            record.description = final_description
            record.vlm_processed = True
            return image_manager.update_image_description(record)

        try:
            updated = await self.plugin._blocking_io.run(_mark_processed)
        except Exception as exc:
            logger.warning("%s 跳过识图回写失败: hash=%s error=%r", self.log_prefix, image_hash[:12], exc)
            return

        if updated:
            logger.info(
                "%s 已将图片标记为跳过识图（source=%s）: hash=%s",
                self.log_prefix, self.source, image_hash[:12],
            )
        else:
            logger.debug(
                "%s 跳过识图回写未命中图片库记录（可能未入库，已忽略）: hash=%s",
                self.log_prefix, image_hash[:12],
            )

    async def _send_help_image(self) -> bool:
        """直接发送随插件打包的帮助图。

        图片由开发者运行 ``python -m plugins.nai_draw_plugin.core.utils.help_renderer``
        预渲染到 ``assets/help.png``，运行时不再启动 chromium、不依赖系统中文字体；
        文件缺失或读取失败均返回 False，由调用方回退到纯文本帮助。
        """
        help_image_path = Path(__file__).resolve().parent / "assets" / "help.png"
        if not help_image_path.is_file():
            logger.warning(
                "%s 帮助图缺失（%s），回退文字版", self.log_prefix, help_image_path
            )
            return False
        try:
            raw = help_image_path.read_bytes()
        except OSError as exc:
            logger.warning("%s 读取帮助图失败，回退文字: %r", self.log_prefix, exc)
            return False
        image_base64 = base64.b64encode(raw).decode("ascii")
        return await self._send_base64_image_result(image_base64, "📖 NovelAI 画图插件帮助")

    async def _send_image_url_with_fallback(
        self, image_url: str, display_message: str, *, image_description: str = ""
    ) -> bool:
        """优先发送远程图片 URL，失败时回退为本地下载再发送 Base64。"""
        target_platform = self._get_target_platform()
        if target_platform == "qq":
            try:
                if await self.send_custom(
                    "imageurl",
                    image_url,
                    display_message=display_message,
                ):
                    return True
                logger.warning("%s QQ 远程图片 URL 发送失败，回退为 Base64", self.log_prefix)
            except Exception as exc:
                logger.warning("%s QQ 远程图片 URL 发送异常，回退为 Base64: %r", self.log_prefix, exc)

        elif _looks_like_generation_request_url(image_url):
            logger.warning(
                "%s 远程图片 URL 看起来像生成接口，跳过直接外发，改为本地下载",
                self.log_prefix,
            )
        else:
            try:
                if await self.send_custom(
                    "imageurl",
                    image_url,
                    display_message=display_message,
                ):
                    return True
                logger.warning("%s 远程图片 URL 发送失败，回退为 Base64", self.log_prefix)
            except Exception as exc:
                logger.warning("%s 远程图片 URL 发送异常，回退为 Base64: %r", self.log_prefix, exc)

        image_base64 = await self._download_remote_image_as_base64(image_url)
        if not image_base64:
            return False

        logger.info("%s 远程图片 URL 已回退为 Base64 发送", self.log_prefix)
        return await self._send_base64_image_result(
            image_base64, display_message, image_description=image_description
        )

    async def manual_recall(self) -> tuple[bool, str | None, bool]:
        """执行 `/nai 撤回`。"""
        logger.info("%s [手动撤回] 收到撤回请求, stream_id=%s", self.log_prefix, self.stream_id)
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True
        try:
            return await self._recall_workflow.execute_manual_recall()
        except Exception as exc:
            logger.error("%s [手动撤回] 未预期异常: %r", self.log_prefix, exc, exc_info=True)
            try:
                await self.send_text("❌ 撤回过程出现内部错误", storage_message=False)
            except Exception as send_exc:
                logger.warning("%s [手动撤回] 内部错误提示发送失败: %r", self.log_prefix, send_exc)
            return False, "撤回内部错误", True

    async def _send_image_result(
        self,
        result: str,
        description: str = "",
        *,
        track_as_auto_draw: bool = False,
    ) -> tuple[bool, str | None, bool]:
        """发送图片结果。

        Args:
            track_as_auto_draw: 若为 True，把这次发送计入 auto_draw 独立间隔门，
                不刷新 explicit/proactive 共用的最近出图时间——这样 reply hook
                自动跟图不会冻结后续用户的明确出图请求。
        """
        final_image_data = self._process_api_response(result)
        if not final_image_data:
            await self.send_text("API 返回了无效的数据")
            return False, "数据格式错误", True

        display_message = self._build_image_display_message(description)
        self._last_send_timestamp = time.time()

        try:
            if final_image_data.startswith(("http://", "https://")):
                remember_pending_plugin_image_send(self.stream_id, self._last_send_timestamp)
                send_ok = await self._send_image_url_with_fallback(
                    final_image_data,
                    display_message,
                    image_description=description,
                )
            elif final_image_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                remember_pending_plugin_image_send(self.stream_id, self._last_send_timestamp)
                send_ok = await self._send_base64_image_result(
                    final_image_data,
                    display_message,
                    image_description=description,
                )
            else:
                await self.send_text("API 返回了无法识别的图片格式")
                return False, "数据格式错误", True
        except Exception as exc:
            discard_pending_plugin_image_send(self.stream_id, self._last_send_timestamp)
            logger.error("%s 图片发送失败: %r", self.log_prefix, exc, exc_info=True)
            await self.send_text(f"图片发送失败: {str(exc)[:100]}")
            return False, "发送失败", True

        if not send_ok:
            logger.warning(
                "%s 图片发送返回失败，可能是适配器超时（图片可能仍在后台发送中）",
                self.log_prefix,
            )
            # NapCat 适配器在处理大图片时可能超时（30s），但图片会在后台继续发送
            # 保留 pending 记录以支持后续撤回，假定发送成功继续后续流程

        self._generation_admission_policy.record_success(
            stream_id=self.stream_id,
            category="auto_draw" if track_as_auto_draw else "action",
            sent_at=self._last_send_timestamp,
        )
        await self._schedule_auto_recall()
        return True, "图片生成成功", True

    def _process_api_response(self, result: str) -> Optional[str]:
        """归一化 API 返回。"""
        if not result:
            return None
        if result.startswith(("http://", "https://")):
            return result
        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            return result
        if "," in result and result.startswith("data:image"):
            return result.split(",", 1)[1]
        return result

    def _process_selfie_prompt(
        self,
        description: str,
        raw_request: str = "",
        *,
        include_selfie_prompt_add: bool = True,
        log_changes: bool = True,
    ) -> str:
        """处理自拍模式的附加提示词。"""
        model_config = self._get_model_config(is_selfie=True)
        selfie_prompt_add = model_config.get("selfie_prompt_add", "") if model_config else ""

        policy = str(self.get_config("prompt_generator.selfie_appearance_policy", "auto") or "auto").strip().lower()
        user_specified = user_mentions_appearance(raw_request)
        original_description = description

        # auto 模式：先移除 LLM 随机外貌，再合并 selfie_prompt_add（允许配置的固定外貌）
        if policy == "auto" and not user_specified:
            description = remove_selfie_appearance_tags(description)

        if include_selfie_prompt_add and selfie_prompt_add:
            description = merge_selfie_prompt(description, selfie_prompt_add)

        # never 模式：合并 selfie_prompt_add 后再移除所有外貌（包括配置中的固定外貌）
        if policy == "never" and not user_specified:
            description = remove_selfie_appearance_tags(description)

        if log_changes and description != original_description:
            logger.debug(
                "%s 自拍提示词后处理已生效：policy=%s user_specified=%s",
                self.log_prefix,
                policy,
                user_specified,
            )

        return description

    async def _generate_prompt_with_llm(
        self,
        request_text: str,
        *,
        allow_inherit: bool,
        include_custom_system_prompt: bool = True,
        reply_context_text: str = "",
        reasoning_context_text: str = "",
    ) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
        result = await self._prompt_generation_workflow.generate(
            request_text,
            allow_inherit=allow_inherit,
            include_custom_system_prompt=include_custom_system_prompt,
            reply_context_text=reply_context_text,
            reasoning_context_text=reasoning_context_text,
        )
        if result is None:
            return None
        return result.text, result.structured

    async def _generate_random_description(
        self,
        *,
        selfie: bool = False,
        character: str = "",
    ) -> str | None:
        return await self._random_scene_planner.generate(
            selfie=selfie,
            character=character,
        )

    async def handle_nai_draw(self, description: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai`。"""
        try:
            if not await self.ensure_generation_permission():
                return False, "没有权限", True

            description = str(description or "").strip()
            if not description:
                await self.send_text("请输入你想画的内容，例如：/nai 画一张初音未来")
                return False, "未提供描述", True

            raw_description = description
            is_random_request, is_random_selfie, random_character = parse_random_scene_request(
                raw_description
            )
            if is_random_request:
                description = await self._generate_random_description(
                    selfie=is_random_selfie,
                    character=random_character,
                ) or ""
                if not description:
                    await self.send_text("随机场景生成失败，请稍后再试~")
                    return False, "随机生成失败", True

            llm_result = await self._generate_prompt_with_llm(
                description,
                allow_inherit=False,
                # NSFW 模板路径会自动注入 custom_prompt.system_prompt；SFW 模板由内部门控跳过
                include_custom_system_prompt=True,
            )
            if not llm_result:
                await self.send_text("提示词生成失败，请稍后再试~")
                return False, "提示词生成失败", True
            generated_prompt, structured_payload = llm_result

            # 治根：只在用户原话明确想看 bot 本人时走 selfie 后处理。
            # 旧实现 detect_selfie_from_output 会把 LLM 用作 framing 的 `portrait photo`
            # / `full body portrait` 误判成 "bot 本人图片"，导致 `/nai 中野二乃，
            # 展示身材` 这类点名二次元角色的请求被注入 bot 默认外貌，把角色洗成 bot 自己。
            # 随机场景的内容可能恰好包含“自拍”，但那只是镜头形式，不代表用户要画 bot 本人。
            # 只有无指定角色的 `/nai 随机自拍` 保留旧的 bot 自拍后处理；普通命令仍按用户原话判定。
            is_selfie = (
                (is_random_selfie and not random_character)
                if is_random_request
                else detect_bot_self_image_intent(raw_description)
            )
            selfie_base_prompt = generated_prompt
            if is_selfie:
                generated_prompt = self._process_selfie_prompt(
                    generated_prompt,
                    description,
                    include_selfie_prompt_add=True,
                    log_changes=True,
                )
                # 自拍场景目前一律按单字符串路径处理（_process_selfie_prompt 只作用于字符串）
                structured_payload = None

            if self.get_config("prompt_generator.enforce_tag_order", False):
                generated_prompt = normalize_prompt_order(generated_prompt)
                structured_payload = self._normalize_structured_order(structured_payload)

            generated_prompt = self._sanitize_prompt_for_sfw_mode(generated_prompt)
            structured_payload = self._sanitize_structured_for_sfw_mode(structured_payload)

            if self._is_prompt_show_enabled():
                show_prompt = generated_prompt
                header = "📝 提示词:"
                if is_selfie and self.get_config("prompt_show.hide_selfie_prompt_add", False):
                    # 重新处理以隐藏 selfie_prompt_add，但仍需应用外貌过滤
                    show_prompt = self._process_selfie_prompt(
                        selfie_base_prompt,
                        description,
                        include_selfie_prompt_add=False,
                        log_changes=False,
                    )
                    header = "📝 提示词(已隐藏自拍补充):"
                elif is_selfie:
                    # 即使不隐藏 selfie_prompt_add，显示的也应该是过滤后的版本
                    show_prompt = generated_prompt
                show_prompt = self._sanitize_prompt_for_sfw_mode(show_prompt)
                await self.send_text(f"{header}\n{show_prompt}", storage_message=False)

            model_config = self._get_model_config(is_selfie=is_selfie)
            if not model_config or not model_config.get("base_url"):
                await self.send_text("NovelAI 配置错误，请检查配置文件")
                return False, "配置错误", True

            image_size = model_config.get("nai_size") or model_config.get("default_size", "")
            enable_debug = bool(self.get_config("components.enable_debug_info", False))
            if enable_debug:
                await self.send_text("正在生成图片，请稍候...")

            request_prompt, request_characters = self._select_send_payload(
                generated_prompt, structured_payload
            )
            success, result = await self.api_client.generate_image(
                prompt=request_prompt,
                model_config=model_config,
                size=image_size,
                characters=request_characters,
            )

            if not success:
                await self.send_text(f"生成图片失败：{result}")
                return False, f"生成失败: {result}", True

            send_result = await self._send_image_result(result, description)
            if send_result[0] and enable_debug:
                await self.send_text("图片生成完成！")
            return send_result
        except Exception as exc:
            logger.error("%s /nai 命令执行异常: %r", self.log_prefix, exc, exc_info=True)
            await self.send_text(f"执行失败：{str(exc)[:100]}")
            return False, f"执行失败: {exc}", True

    async def handle_image_to_image_draw(
        self,
        description: str,
        *,
        image_base64: str,
        mode: str,
        strength: Optional[float] = None,
        fidelity: Optional[float] = None,
        type_value: Optional[str] = None,
        raw_prompt: Optional[str] = None,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai i2i` 与 `/nai ref` 共享的图生图流程。

        Args:
            mode: "i2i" 走文档 §20.1 图生图；"ref" 走文档 §20.4 角色参考。
            strength: i2i / ref 的整体强度；缺省让网关用默认值。
            fidelity: ref 专属，主参考强度。
            type_value: ref 专属，``character`` / ``style`` / ``character&style``。
            raw_prompt: 不为 None 时跳过 LLM 翻译，``/nai0 ref`` 路径使用。
        """
        if mode not in {"i2i", "ref"}:
            await self.send_text(f"❌ 不支持的图生图模式：{mode!r}")
            return False, f"模式不支持: {mode}", True

        return await self._run_image_pipeline(
            description=description,
            image_base64=image_base64,
            mode=mode,
            strength=strength,
            fidelity=fidelity,
            type_value=type_value,
            raw_prompt=raw_prompt,
        )

    async def handle_nai_vibe_draw(
        self,
        description: str,
        *,
        image_base64_list: List[str],
        info_extracted: Optional[float] = None,
        strength: Optional[float] = None,
        raw_prompt: Optional[str] = None,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai vibe`：1~4 张 Vibe Transfer（文档 §20.3 / §20.3.2）。

        image_base64_list 支持 1~4 张参考图，全部命中本地 vibe cache 时整个请求
        免 1 anlas 流量附加费（§20.3.2）；部分命中只省命中那几张的编码成本。

        raw_prompt 不为 None 时跳过 LLM 翻译，``/nai0 vibe`` 路径使用。
        """
        return await self._run_image_pipeline(
            description=description,
            image_base64=None,
            vibe_images_base64=list(image_base64_list or []),
            mode="vibe",
            info_extracted=info_extracted,
            strength=strength,
            raw_prompt=raw_prompt,
        )

    # ====== 命名图库 (vibe / ref 共用骨架) ======

    async def _ensure_named_reference_admin(self, *, scope: str, action: str) -> bool:
        """命名图库命令的管理员鉴权（与 /nai nsfw 同套 ``is_admin_user`` 判定）。

        ``scope=="ref"`` 时全部 action 仅限管理员；``scope=="vibe"`` 时仅 ``draw``
        放开给普通用户，其余（save / select / list / delete / clear）仅限管理员。
        返回 True 放行；返回 False 表示已对用户发送拒绝提示，调用方应立即结束命令。
        """
        if scope == "ref":
            scope_label = "角色参考"
        elif scope == "vibe" and action != "draw":
            scope_label = "Vibe"
        else:
            return True

        _, _, user_id = self._get_chat_identity()
        if session_state.is_admin_user(user_id, self.get_config):
            return True

        await self.send_text(
            f"❌ 只有管理员可以使用 {scope_label} 图库相关命令",
            storage_message=False,
        )
        return False

    async def handle_named_reference_save(
        self,
        *,
        scope: str,
        name: str,
        image_base64: str,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai (vibe|ref)存 <名字>`：把引用回复的图入库。

        scope: ``vibe`` / ``ref``，决定图落到哪个图库。"""
        if not await self._ensure_named_reference_admin(scope=scope, action="save"):
            return False, "没有管理员权限", True
        if not await self.ensure_generation_permission():
            return False, "没有权限", True

        scope_label = _scope_label(scope)
        normalized = _normalize_image_for_payload(image_base64)
        if not normalized:
            await self.send_text(f"❌ 未解析到图片，请引用回复一张图后再发送 /nai {scope}存 <名字>")
            return False, "未找到图片", True

        try:
            image_bytes = base64.b64decode(normalized, validate=False)
        except (ValueError, TypeError) as exc:
            await self.send_text(f"❌ 参考图 base64 解码失败: {exc}")
            return False, "图片解码失败", True

        store = get_named_reference_store()
        owner_kind, owner_id = self._named_reference_owner()
        try:
            ref = store.save(
                scope=scope,
                owner_kind=owner_kind,
                owner_id=owner_id,
                name=name,
                image_bytes=image_bytes,
            )
        except _NamedRefInvalidNameError as exc:
            await self.send_text(f"❌ 名字不合规：{exc}")
            return False, "名字不合规", True
        except _NamedRefInvalidImageError as exc:
            await self.send_text(f"❌ 图片不合规：{exc}")
            return False, "图片不合规", True
        except _NamedRefCapacityExceededError as exc:
            await self.send_text(f"❌ {exc}")
            return False, "图库已满", True

        # 小图友好提示：协议层引用回复经常给 thumb，提示用户下次直接附图能存到原图
        warn_suffix = ""
        if ref.width < 256 or ref.height < 256:
            warn_suffix = (
                f"\n⚠️ 存入尺寸 {ref.width}x{ref.height} 偏小，疑似平台缩略图\n"
                "下次想存高清原图请把图作为命令的同条消息附件发出（不要走引用回复）"
            )
        await self.send_text(
            f"✅ 已入 {scope_label} 图库：{name}\n"
            f"   格式 {ref.image_format.upper()}，{ref.width}x{ref.height}，{ref.byte_size / 1024:.1f}KB"
            + warn_suffix
        )
        return True, "已入库", True

    async def handle_named_reference_list(
        self,
        *,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai (vibe|ref)图库`：列出当前归属（群聊→该群 / 私聊→个人）的命名图。"""
        if not await self._ensure_named_reference_admin(scope=scope, action="list"):
            return False, "没有管理员权限", True
        scope_label = _scope_label(scope)
        store = get_named_reference_store()
        owner_kind, owner_id = self._named_reference_owner()
        entries = store.list(scope=scope, owner_kind=owner_kind, owner_id=owner_id)
        if not entries:
            await self.send_text(
                f"📂 {scope_label} 图库还是空的\n"
                f"先用 `/nai {scope}存 <名字>` 把引用回复的图入库吧"
            )
            return True, "空库", True

        # 标出"当前选定"项（list），方便用户知道下一条裸命令会用哪几张
        selected_list = store.get_selection(
            scope=scope,
            owner_kind=owner_kind,
            owner_id=owner_id,
            stream_id=self.stream_id,
        )
        selected_set = set(selected_list)
        lines = [f"📂 {scope_label} 图库（{len(entries)} 张）"]
        for ref in entries:
            marker = "★ " if ref.name in selected_set else "  "
            lines.append(
                f"{marker}{ref.name}（{ref.image_format.upper()} "
                f"{ref.width}x{ref.height}，{ref.byte_size / 1024:.1f}KB）"
            )
        if selected_list:
            lines.append(
                f"\n当前会话选定（{len(selected_list)} 张）：{' / '.join(selected_list)}"
                f"（裸命令 /nai {scope} <描述> 会一起用）"
            )
        else:
            max_count = _max_selection_for_scope(scope)
            if max_count > 1:
                lines.append(
                    f"\n本会话未选定，可用 /nai {scope}选 <名字1> [<名字2>...]"
                    f" 设置默认图（最多 {max_count} 张）"
                )
            else:
                lines.append(f"\n本会话未选定，可用 /nai {scope}选 <名字> 设置默认图")
        await self.send_text("\n".join(lines))
        return True, "已列出图库", True

    async def handle_named_reference_delete(
        self,
        *,
        scope: str,
        name: str,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai (vibe|ref)删 <名字>`。"""
        if not await self._ensure_named_reference_admin(scope=scope, action="delete"):
            return False, "没有管理员权限", True
        scope_label = _scope_label(scope)
        store = get_named_reference_store()
        owner_kind, owner_id = self._named_reference_owner()
        try:
            ok = store.delete(scope=scope, owner_kind=owner_kind, owner_id=owner_id, name=name)
        except _NamedRefInvalidNameError as exc:
            await self.send_text(f"❌ 名字不合规：{exc}")
            return False, "名字不合规", True
        if not ok:
            await self.send_text(f"⚠️ {scope_label} 图库里没有 {name}")
            return False, "未找到命名图", True
        await self.send_text(f"🗑 已删除 {scope_label} 图库的 {name}")
        return True, "已删除", True

    async def handle_named_reference_clear_all(
        self,
        *,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai (vibe|ref)清空`：一键删除当前归属该 scope 的全部图 + 选定。

        语义是"清空当前归属的整个 {scope} 图库"。群聊里归属是该群本身，会清空整个
        群的共享图库；私聊里归属是 user_id，按个人隔离。跨 stream 生效（store 层不
        区分 stream，按 owner 隔离）。返回实际删除的张数，便于用户确认。
        """
        if not await self._ensure_named_reference_admin(scope=scope, action="clear"):
            return False, "没有管理员权限", True
        scope_label = _scope_label(scope)
        store = get_named_reference_store()
        owner_kind, owner_id = self._named_reference_owner()
        try:
            deleted = store.clear_all(scope=scope, owner_kind=owner_kind, owner_id=owner_id)
        except Exception as exc:
            logger.error(
                "%s /nai %s清空 执行异常: %r", self.log_prefix, scope, exc, exc_info=True
            )
            await self.send_text(f"❌ 清空 {scope_label} 图库失败：{str(exc)[:100]}")
            return False, "清空失败", True

        if deleted == 0:
            await self.send_text(f"📂 {scope_label} 图库本来就是空的，没有可删除的图")
            return True, "图库为空", True

        await self.send_text(
            f"🧹 已清空 {scope_label} 图库共 {deleted} 张图，"
            f"本会话的 {scope_label} 选定也已重置；想用先 /nai {scope}存 <名字> 再来。"
        )
        return True, "已清空", True

    async def handle_named_reference_select(
        self,
        *,
        scope: str,
        names: List[str],
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai (vibe|ref)选 <名字> [<名字>...]`：把当前会话的粘性选定指向若干张图。

        vibe 接受 1~4 张（§20.3），ref 接受 1 张（§20.4）。"""
        if not await self._ensure_named_reference_admin(scope=scope, action="select"):
            return False, "没有管理员权限", True
        scope_label = _scope_label(scope)
        store = get_named_reference_store()
        owner_kind, owner_id = self._named_reference_owner()
        max_count = _max_selection_for_scope(scope)
        if not names:
            await self.send_text(f"❌ 请至少给一张名字：/nai {scope}选 <名字>")
            return False, "名字为空", True
        if len(names) > max_count:
            await self.send_text(
                f"❌ {scope_label} 最多同时选 {max_count} 张参考图，本次给了 {len(names)} 张"
            )
            return False, "超过上限", True
        try:
            store.set_selection(
                scope=scope,
                owner_kind=owner_kind,
                owner_id=owner_id,
                stream_id=self.stream_id,
                names=names,
            )
        except _NamedRefInvalidNameError as exc:
            await self.send_text(f"❌ 名字不合规：{exc}")
            return False, "名字不合规", True
        except KeyError as exc:
            await self.send_text(
                f"❌ {scope_label} 图库里 {exc.args[0] if exc.args else '某张图'} 不存在\n"
                f"用 /nai {scope}图库 查看现有命名图"
            )
            return False, "未找到命名图", True
        except ValueError as exc:
            await self.send_text(f"❌ {exc}")
            return False, "选定参数非法", True
        names_str = " / ".join(names)
        await self.send_text(
            f"✅ 已把本会话的 {scope_label} 默认图设为：{names_str}（共 {len(names)} 张）\n"
            f"之后 /nai {scope} <描述> 会一并用这些；想换图请重新 /nai {scope}选 <名字...>"
        )
        return True, "已设置选定", True

    async def handle_named_reference_draw(
        self,
        *,
        scope: str,
        description: str,
        explicit_names: Optional[List[str]] = None,
        raw_prompt: Optional[str] = None,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai vibe <描述>` / `/nai ref <描述>` 与 `/nai0 vibe` / `/nai0 ref`。

        explicit_names 来自命令里的 ``@<名字> @<名字>...`` 单次指定（vibe 最多 4 张，
        ref 最多 1 张）；为空时回退到本会话的粘性选定列表。两者都没有则报错并指引
        用户如何入库 / 选定。

        raw_prompt 不为 None 时跳过 LLM 翻译，直接用作 prompt（``/nai0`` 路径）；
        description 仍作为请求文本沿用 sanity 检查（空时报错）。
        """
        if not await self._ensure_named_reference_admin(scope=scope, action="draw"):
            return False, "没有管理员权限", True
        scope_label = _scope_label(scope)
        store = get_named_reference_store()
        owner_kind, owner_id = self._named_reference_owner()
        max_count = _max_selection_for_scope(scope)

        chosen_names: List[str] = []
        if explicit_names:
            if len(explicit_names) > max_count:
                await self.send_text(
                    f"❌ {scope_label} 单次最多用 {max_count} 张参考图，本次收到 {len(explicit_names)} 张"
                )
                return False, "超过单次上限", True
            chosen_names = list(explicit_names)
        else:
            chosen_names = store.get_selection(
                scope=scope,
                owner_kind=owner_kind,
                owner_id=owner_id,
                stream_id=self.stream_id,
            )
            if not chosen_names:
                await self.send_text(
                    f"❌ 还未在本会话选定 {scope_label} 图\n"
                    f"先 /nai {scope}存 <名字> 入库，再 /nai {scope}选 <名字...>；"
                    f"或单次用 /nai {scope} @<名字>... <描述>"
                )
                return False, "未选定命名图", True

        # 逐张取图字节；vibe 多图时任何一张拿不到都整体报错（保留剩余的不进入后续）
        images_bytes: List[bytes] = []
        for name in chosen_names:
            try:
                image_bytes = store.get(
                    scope=scope,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                    name=name,
                )
            except _NamedRefInvalidNameError as exc:
                await self.send_text(f"❌ 名字不合规：{exc}")
                return False, "名字不合规", True
            if image_bytes is None:
                # 没在 @<名字> 指定中：可能粘性选定指向已删图，清掉选定
                if not explicit_names:
                    store.clear_selection(
                        scope=scope,
                        owner_kind=owner_kind,
                        owner_id=owner_id,
                        stream_id=self.stream_id,
                    )
                    await self.send_text(
                        f"❌ 选定列表里的 {name} 已不在 {scope_label} 图库（可能已被删），"
                        f"已自动清除整段选定；请 /nai {scope}选 <名字...> 重新选择"
                    )
                    return False, "选定的图已不存在", True
                await self.send_text(
                    f"❌ {scope_label} 图库里没有 {name}\n"
                    f"用 /nai {scope}图库 查看，或 /nai {scope}存 {name} 先入库"
                )
                return False, "未找到命名图", True
            images_bytes.append(image_bytes)

        images_base64 = [base64.b64encode(b).decode("ascii") for b in images_bytes]
        logger.info(
            "%s /nai %s 取命名图：names=%s, 共 %d 张，合计字节=%.1fKB",
            self.log_prefix,
            scope,
            chosen_names,
            len(images_bytes),
            sum(len(b) for b in images_bytes) / 1024.0,
        )

        if scope == _NAMED_SCOPE_VIBE:
            return await self.handle_nai_vibe_draw(
                description,
                image_base64_list=images_base64,
                raw_prompt=raw_prompt,
            )
        # ref 固定 1 张：store 层 set_selection 已限制 ≤1，这里再兜底取第一张
        return await self.handle_image_to_image_draw(
            description,
            image_base64=images_base64[0],
            mode="ref",
            raw_prompt=raw_prompt,
        )

    def _read_clamped_float_config(
        self,
        path: str,
        default: float,
        lo: float,
        hi: float,
    ) -> float:
        """读取 float 配置并夹到 ``[lo, hi]``；非法值回退默认。

        i2i / vibe / ref 几个 NewAPI §20.x 参数（strength / noise / fidelity /
        info_extracted / overall_strength）都需要范围保护，避免用户写飞值后让
        服务端打 400。
        """
        try:
            value = float(self.get_config(path, default))
        except (TypeError, ValueError):
            return float(default)
        if value < lo:
            return float(lo)
        if value > hi:
            return float(hi)
        return value

    async def _run_image_pipeline(
        self,
        *,
        description: str,
        image_base64: Optional[str] = None,
        vibe_images_base64: Optional[List[str]] = None,
        mode: str,
        strength: Optional[float] = None,
        fidelity: Optional[float] = None,
        type_value: Optional[str] = None,
        info_extracted: Optional[float] = None,
        raw_prompt: Optional[str] = None,
    ) -> tuple[bool, str | None, bool]:
        """共享 i2i / ref / vibe 三条命令的"取参考图 → LLM → 发请求"主干。

        - i2i / ref：用 image_base64 单图
        - vibe：用 vibe_images_base64 列表（1~4 张），逐张组装到 controlnet.images[]
        - raw_prompt：不为 None 时跳过 LLM 翻译，直接当 prompt 用（``/nai0 vibe`` / ``/nai0 ref`` 路径）
        """
        try:
            if not await self.ensure_generation_permission():
                return False, "没有权限", True

            description = str(description or "").strip()
            if not description:
                example = {
                    "i2i": "/nai i2i 改成森林背景",
                    "ref": "/nai ref 站在街道，看向镜头",
                    "vibe": "/nai vibe 都市夜景，霓虹氛围",
                }.get(mode, "/nai i2i 改成森林背景")
                await self.send_text(f"请输入你想画的内容，例如：{example}")
                return False, "未提供描述", True

            # 按模式收集 normalized image(s)：vibe 走 list，其余走 single
            normalized_images: List[str] = []
            if mode == "vibe":
                source_list = list(vibe_images_base64 or [])
                if not source_list:
                    await self.send_text("❌ vibe 模式需要至少一张参考图")
                    return False, "未找到图片", True
                for raw in source_list:
                    n = _normalize_image_for_payload(raw)
                    if not n:
                        await self.send_text("❌ 有一张参考图解析失败，请检查图库内容")
                        return False, "图片解析失败", True
                    normalized_images.append(n)
            else:
                normalized_image = _normalize_image_for_payload(image_base64 or "")
                if not normalized_image:
                    await self.send_text("❌ 未能解析参考图，请引用回复一张图后再发命令")
                    return False, "未找到图片", True
                normalized_images = [normalized_image]

            # 临时 debug：把拿到的图实际尺寸 + 字节大小 log 出来，
            # 用来诊断"原图 vs 协议 thumb"的来源问题；后续根因解决（bot 出图原图缓存）落地后可删
            for idx, n_img in enumerate(normalized_images):
                _probe_dims = _read_image_dimensions(n_img)
                _probe_byte_len = (len(n_img) * 3) // 4
                logger.info(
                    "%s /nai %s 取到参考图[%d/%d]：dims=%s, 估算字节=%.1fKB (base64 长度=%d)",
                    self.log_prefix,
                    mode,
                    idx + 1,
                    len(normalized_images),
                    _probe_dims,
                    _probe_byte_len / 1024.0,
                    len(n_img),
                )

            # /nai0 路径：raw_prompt 给定时跳过 LLM 翻译，直接当 prompt 用。
            # raw_prompt 仍走 sanitize 链以剔除 CJK / SFW 违规，避免上游 §8 直接 400。
            if raw_prompt is not None:
                generated_prompt = str(raw_prompt or "").strip()
                if not generated_prompt:
                    await self.send_text("❌ /nai0 路径下英文 tags 不能为空")
                    return False, "未提供 tags", True
                structured_payload: Optional[Dict[str, Any]] = None
            else:
                llm_result = await self._generate_prompt_with_llm(
                    description,
                    allow_inherit=False,
                    include_custom_system_prompt=True,
                )
                if not llm_result:
                    await self.send_text("提示词生成失败，请稍后再试~")
                    return False, "提示词生成失败", True
                generated_prompt, structured_payload = llm_result

            if self.get_config("prompt_generator.enforce_tag_order", False):
                generated_prompt = normalize_prompt_order(generated_prompt)
                structured_payload = self._normalize_structured_order(structured_payload)

            # vibe 与 /nai 文字命令对齐：用户文本里点名要 bot 自拍 / 肖像时注入 bot 外貌
            # 与 selfie_prompt_add。raw_prompt（/nai0 vibe）路径用户已显式给 tags，跳过；
            # ref / i2i 不注入——会把指定参考图洗成 bot 外貌（见 _run_image_pipeline 主路径
            # `is_selfie=False` 历史注释）。
            is_selfie = (
                mode == "vibe"
                and raw_prompt is None
                and detect_bot_self_image_intent(description)
            )
            selfie_base_prompt = generated_prompt
            if is_selfie:
                generated_prompt = self._process_selfie_prompt(
                    generated_prompt,
                    description,
                    include_selfie_prompt_add=True,
                    log_changes=True,
                )
                # 自拍场景目前一律按单字符串路径处理（_process_selfie_prompt 只作用于字符串）
                structured_payload = None

            generated_prompt = self._sanitize_prompt_for_sfw_mode(generated_prompt)
            structured_payload = self._sanitize_structured_for_sfw_mode(structured_payload)

            if raw_prompt is None and self._is_prompt_show_enabled():
                show_prompt = generated_prompt
                header = "📝 提示词:"
                if is_selfie and self.get_config("prompt_show.hide_selfie_prompt_add", False):
                    show_prompt = self._process_selfie_prompt(
                        selfie_base_prompt,
                        description,
                        include_selfie_prompt_add=False,
                        log_changes=False,
                    )
                    header = "📝 提示词(已隐藏自拍补充):"
                show_prompt = self._sanitize_prompt_for_sfw_mode(show_prompt)
                await self.send_text(f"{header}\n{show_prompt}", storage_message=False)

            model_config = self._get_model_config(is_selfie=is_selfie)
            if not model_config or not model_config.get("base_url"):
                await self.send_text("NovelAI 配置错误，请检查配置文件")
                return False, "配置错误", True

            image_size: Any = model_config.get("nai_size") or model_config.get("default_size", "")
            i2i_payload: Optional[dict[str, Any]] = None
            controlnet_payload: Optional[dict[str, Any]] = None
            character_references_payload: Optional[list[dict[str, Any]]] = None

            if mode == "i2i":
                normalized_image = normalized_images[0]
                dims = _read_image_dimensions(normalized_image)
                if dims is None:
                    # 文档 §20.1 要求 image 宽高严格等于外层 size；解析不出尺寸就一定送不出合规请求，
                    # 不能静默走默认 size 让上游打 400（曾经的 bug：image 188x188 被默认 size=832x1216
                    # 配走，服务端报 REQUEST_VALIDATION_ERROR 用户一脸懵）
                    await self.send_text(
                        "❌ 无法解析参考图尺寸：可能是缩略图、损坏或不受支持的格式\n"
                        "NewAPI i2i 要求图片宽高必须严格等于输出 size。请直接把原图作为命令的同条消息发出来\n"
                        "（PNG/JPEG/WebP，不要回复引用，部分平台引用回复只会给低分辨率缩略图）"
                    )
                    return False, "图片尺寸解析失败", True
                width, height = dims
                if width % 64 != 0 or height % 64 != 0:
                    await self.send_text(
                        f"❌ 参考图尺寸 {width}x{height} 不是 64 的倍数，"
                        "NewAPI i2i 要求宽高必须 64 整除；请先裁/缩到合规尺寸再发"
                    )
                    return False, "尺寸不合规", True
                if width < 256 or height < 256:
                    # 256 以下基本就是缩略图，硬送即使形式合规出图也是糊的
                    await self.send_text(
                        f"❌ 参考图尺寸 {width}x{height} 过小（< 256），疑似缩略图\n"
                        "请直接把原图作为命令的同条消息发出来，避免走引用回复拿到缩略图"
                    )
                    return False, "参考图过小", True
                image_size = [width, height]
                # §20.1：strength / noise 都从 config 读取，命令调用方未显式覆盖时用 config 默认
                effective_strength = (
                    strength
                    if strength is not None
                    else self._read_clamped_float_config("i2i.strength", 0.7, 0.01, 0.99)
                )
                effective_noise = self._read_clamped_float_config("i2i.noise", 0.0, 0.0, 0.99)
                i2i_payload = {
                    "image": normalized_image,
                    "strength": effective_strength,
                    "noise": effective_noise,
                }
            elif mode == "ref":
                normalized_image = normalized_images[0]
                # §20.4：type / fidelity / strength 三项都走 config 默认，调用方可覆盖；
                # type 还支持本会话的运行时切换（/nai ref类型 ...）
                ref_platform, ref_chat_id, _ = self._get_chat_identity()
                explicit_type = (type_value or "").strip()
                effective_type = (
                    explicit_type
                    if explicit_type
                    else session_state.get_character_reference_type(
                        ref_platform, ref_chat_id, self.get_config
                    )
                )
                effective_fidelity = (
                    fidelity
                    if fidelity is not None
                    else self._read_clamped_float_config("character_reference.fidelity", 1.0, 0.0, 1.0)
                )
                effective_strength = (
                    strength
                    if strength is not None
                    else self._read_clamped_float_config("character_reference.strength", 1.0, 0.0, 1.0)
                )
                ref_entry: dict[str, Any] = {
                    "image": normalized_image,
                    "type": effective_type,
                    "fidelity": effective_fidelity,
                    "strength": effective_strength,
                }
                character_references_payload = [ref_entry]
            elif mode == "vibe":
                # §20.3：controlnet.images[] 最多 4 张，逐张组装 image+info_extracted+strength；
                # 顶层 controlnet.strength 走 [vibe].overall_strength
                effective_info = (
                    info_extracted
                    if info_extracted is not None
                    else self._read_clamped_float_config("vibe.info_extracted", 0.7, 0.01, 1.0)
                )
                effective_per_img_strength = (
                    strength
                    if strength is not None
                    else self._read_clamped_float_config("vibe.reference_strength", 0.6, 0.01, 1.0)
                )
                effective_overall_strength = self._read_clamped_float_config(
                    "vibe.overall_strength", 1.0, 0.0, 1.0
                )
                vibe_entries: List[Dict[str, Any]] = []
                for n_img in normalized_images:
                    vibe_entries.append({
                        "image": n_img,
                        "info_extracted": effective_info,
                        "strength": effective_per_img_strength,
                    })
                controlnet_payload = {
                    "images": vibe_entries,
                    "strength": effective_overall_strength,
                }

            enable_debug = bool(self.get_config("components.enable_debug_info", False))
            if enable_debug:
                await self.send_text("正在生成图片，请稍候...")

            request_prompt, request_characters = self._select_send_payload(
                generated_prompt, structured_payload
            )
            success, result = await self.api_client.generate_image(
                prompt=request_prompt,
                model_config=model_config,
                size=image_size,
                characters=request_characters,
                i2i_payload=i2i_payload,
                controlnet_payload=controlnet_payload,
                character_references_payload=character_references_payload,
            )

            if not success:
                await self.send_text(f"生成图片失败：{result}")
                return False, f"生成失败: {result}", True

            send_result = await self._send_image_result(result, description)
            if send_result[0] and enable_debug:
                await self.send_text("图片生成完成！")
            return send_result
        except Exception as exc:
            logger.error(
                "%s /nai %s 命令执行异常: %r", self.log_prefix, mode, exc, exc_info=True
            )
            await self.send_text(f"执行失败：{str(exc)[:100]}")
            return False, f"执行失败: {exc}", True

    async def handle_nai0_draw(self, tags: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai0`。"""
        try:
            if not await self.ensure_generation_permission():
                return False, "没有权限", True

            tags = str(tags or "").strip()
            if not tags:
                await self.send_text("请输入英文标签，例如：/nai0 hatsune miku, smile")
                return False, "未提供标签", True

            model_config = self._get_model_config()
            if not model_config or not model_config.get("base_url"):
                await self.send_text("NovelAI 配置错误，请检查配置文件")
                return False, "配置错误", True

            image_size = model_config.get("nai_size") or model_config.get("default_size", "")
            enable_debug = bool(self.get_config("components.enable_debug_info", False))
            if enable_debug:
                await self.send_text("正在生成图片，请稍候...")

            success, result = await self.api_client.generate_image(
                prompt=tags,
                model_config=model_config,
                size=image_size,
            )

            if not success:
                await self.send_text(f"生成图片失败：{result}")
                return False, f"生成失败: {result}", True

            send_result = await self._send_image_result(result, tags)
            if send_result[0] and enable_debug:
                await self.send_text("图片生成完成！")
            return send_result
        except Exception as exc:
            logger.error("%s /nai0 命令执行异常: %r", self.log_prefix, exc, exc_info=True)
            await self.send_text(f"执行失败：{str(exc)[:100]}")
            return False, f"执行失败: {exc}", True

    # 结构化字段顺序固定为：主体视角 → 动作 → 情绪 → 场景增量 → 构图。
    # 这个顺序与 NAI tag 标准排序对齐，下游 prompt 模板里"tag 顺序"硬规则也基于此排序解析。
    # 实际取值与拼接逻辑见 core/utils/action_payload.py（提到独立模块方便单测）。
    _STRUCTURED_DESCRIPTION_FIELDS = STRUCTURED_DESCRIPTION_FIELDS

    def _compose_description_from_action_data(self) -> str:
        """把 Planner 拆分的 5 个结构化字段 + ``description`` 拼成单行 request 文本。

        细节见 ``compose_description_from_action_payload``：``description`` 字段含**独有的
        核心锚点**（角色名 / 服装款式 / 场景物件），不能因为结构化字段非空就丢——否则
        会导致"画一张初音未来"丢失"初音未来"，下游 LLM 只能猜场景。
        """
        return compose_description_from_action_payload(self.action_data)

    def _is_named_character_intent(self) -> bool:
        """Planner 是否声明"本轮画指定角色，非 bot 出镜"。

        命中后跳过 ``_inject_self_image_hint`` 与 ``_process_selfie_prompt``——这两步
        是为"bot 自己出镜"设计的兜底（注入肖像/自拍语义、把 bot 默认外貌锚点合进
        prompt 并删冲突发色/瞳色），对"用户/bot 点名画指定二次元角色"是有害注入：
        会把 ``初音未来`` 的绿色双马尾洗成 bot 自己的发色。
        """
        return is_named_character_intent(self.action_data)

    async def handle_action(self) -> tuple[bool, str]:
        """处理 `nai_web_draw` Action。"""
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户"
        if not await self.ensure_generation_permission():
            return False, "没有权限"

        description = self._compose_description_from_action_data()
        size = str(self.action_data.get("size", "") or "").strip()

        # Planner 极少数情况下不给 description，回落到 reasoning 仅作生图素材；
        # Action Guard 判定独立走真实用户原话，与这里的 fallback 无关。
        if not description:
            description = self.reasoning.strip()

        # raw_description 在后续自拍/外观策略里被当作"本轮请求文本"使用，需保留
        # LLM 改写前的版本（与最终 description 区分）。
        raw_description = description

        # "画指定角色" 短路：Planner 明确标记本轮主体不是 bot 时，跳过 self-image 注入与
        # selfie 后处理。这两步原本是给"bot 自己出镜"兜底的——会把"肖像照"塞进 description、
        # 把 bot 默认外貌锚点合进 prompt，对画指定角色（如初音未来）就是把角色洗成 bot。
        is_named_character = self._is_named_character_intent()

        trigger_assessment = await self._assess_action_trigger(reasoning=self.reasoning)
        if self._is_action_guard_enabled() and not trigger_assessment.should_generate:
            logger.info(
                "%s Action 出图已拦截: category=%s detail=%s signal=%s text=%s",
                self.log_prefix,
                trigger_assessment.category,
                trigger_assessment.detail,
                trigger_assessment.signal_source,
                trigger_assessment.signal_text,
            )
            return False, trigger_assessment.detail

        # 主动出图自动 self-image 增强：bot 自己想发图时，让出来的图更像"她给你看一眼自己"
        # 而不是"画了一张陌生女孩"。explicit 路径不动，保持用户原意。
        # 画指定角色路径不注入：本轮主体是指定角色而非 bot，加"肖像照"会把角色洗成 bot 肖像。
        if (
            trigger_assessment.category == "proactive"
            and bool(self.get_config("action_guard.proactive_self_image_boost", True))
            and description
            and not is_named_character
            and not detect_selfie_from_output(description)
        ):
            description = _inject_self_image_hint(description, mode="portrait")
            raw_description = description
            logger.debug("%s 主动出图已注入 self-image 提示: %s", self.log_prefix, description[:80])

        generated_prompt = await self._generate_prompt_with_llm(
            description,
            allow_inherit=True,
            include_custom_system_prompt=True,
            reasoning_context_text=self.reasoning,
        )
        structured_payload: Optional[Dict[str, Any]] = None
        if generated_prompt:
            description = generated_prompt[0].strip()
            structured_payload = generated_prompt[1]
        elif not description:
            await self.send_text("提示词生成器开小差了，请直接告诉我想画什么，或者稍后再试一次~")
            return False, "图片描述为空"

        is_selfie = (
            False
            if is_named_character
            else detect_bot_self_image_intent(raw_description)
        )
        selfie_base_prompt = description
        if is_selfie:
            description = self._process_selfie_prompt(
                description,
                raw_description,
                include_selfie_prompt_add=True,
                log_changes=True,
            )
            session_state.set_last_selfie_context(
                self.stream_id,
                description,
                raw_description,
                ttl=float(self.get_config("prompt_generator.inherit_ttl", 0) or 0),
            )
            structured_payload = None

        if self.get_config("prompt_generator.enforce_tag_order", False):
            description = normalize_prompt_order(description)
            structured_payload = self._normalize_structured_order(structured_payload)

        description = self._sanitize_prompt_for_sfw_mode(description)
        structured_payload = self._sanitize_structured_for_sfw_mode(structured_payload)

        if self._is_prompt_show_enabled():
            show_prompt = description
            header = "📝 提示词:"
            if is_selfie and self.get_config("prompt_show.hide_selfie_prompt_add", False):
                show_prompt = self._process_selfie_prompt(
                    selfie_base_prompt,
                    raw_description,
                    include_selfie_prompt_add=False,
                    log_changes=False,
                )
                header = "📝 提示词(已隐藏自拍补充):"
            show_prompt = self._sanitize_prompt_for_sfw_mode(show_prompt)
            await self.send_text(f"{header}\n{show_prompt}", storage_message=False)

        model_config = self._get_model_config(is_selfie=is_selfie)
        if not model_config or not model_config.get("base_url"):
            await self.send_text("抱歉，NAI low-level 网关地址未配置，无法提供服务。")
            return False, "模型配置无效"

        image_size = size or model_config.get("nai_size") or model_config.get("default_size", "")
        enable_debug = bool(self.get_config("components.enable_debug_info", False))
        if enable_debug:
            await self.send_text("收到！正在使用 NAI low-level 网关生成图片，请稍候...")

        request_prompt, request_characters = self._select_send_payload(
            description, structured_payload
        )
        try:
            success, result = await self.api_client.generate_image(
                prompt=request_prompt,
                model_config=model_config,
                size=image_size,
                characters=request_characters,
            )
        except Exception as exc:
            logger.error("%s Action 生图失败: %r", self.log_prefix, exc, exc_info=True)
            await self.send_text(f"图片生成服务遇到意外问题: {str(exc)[:100]}")
            return False, str(exc)

        if not success:
            await self.send_text(f"哎呀，生成图片时遇到问题：{result}")
            return False, str(result)

        send_result = await self._send_image_result(result, raw_description or description)
        if send_result[0] and enable_debug:
            await self.send_text("图片生成完成！")
        return send_result[0], send_result[1] or ""

    async def handle_auto_draw_from_reply(
        self,
        seed_description: str,
        *,
        reply_context_text: str = "",
    ) -> tuple[bool, str]:
        """reply 后置 hook 触发的自动跟图。

        与 handle_action 区别：
        - description 由 reply 评分模块拼好（``seed_description``），不依赖 Planner 写参数
        - guard 走 ``category="auto_draw"``，使用独立间隔门
        - 发送计入 ``last_auto_draw_sent_at``，不会冻结后续显式请求
        - 失败不发用户可见报错（OBSERVE hook 静默兜底）

        ``reply_context_text`` 是 bot 即将说出的回复原文：description 只是关键词拼接，LLM
        看不到 reply 的具体语境（"刚洗完澡"暗示的浴袍/湿发等）；这段原文会注入 prompt 模板，
        让生成的图与文匹配。
        """
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户"
        if not await self.ensure_generation_permission():
            return False, "没有权限"

        description = (seed_description or "").strip()
        if not description:
            return False, "空 description"

        # auto_draw 单独跑 guard：负向用户原话仍要兜底，间隔走 auto_draw 档
        guard_state = await self._assess_auto_draw_trigger()
        if self._is_action_guard_enabled() and not guard_state.should_generate:
            logger.info(
                "%s reply 自动跟图被拦截: detail=%s text=%s",
                self.log_prefix,
                guard_state.detail,
                guard_state.signal_text,
            )
            return False, guard_state.detail

        # 自动 self-image 增强：description 不含自拍/肖像/生活照标签时补一个
        if (
            bool(self.get_config("auto_draw_on_reply.self_image_boost", True))
            and not detect_selfie_from_output(description)
        ):
            description = _inject_self_image_hint(description, mode="portrait")

        raw_description = description

        generated_prompt = await self._generate_prompt_with_llm(
            description,
            allow_inherit=True,
            include_custom_system_prompt=True,
            reply_context_text=reply_context_text,
        )
        structured_payload: Optional[Dict[str, Any]] = None
        if generated_prompt:
            description = generated_prompt[0].strip()
            structured_payload = generated_prompt[1]
        elif not description:
            return False, "图片描述为空"

        # 肖像规则不再要求 portrait photo/candid photo；因此这里必须依赖 LLM 前已经
        # 注入的 bot 本人意图，不能反推最终标签，否则会漏掉自拍外貌串。
        is_selfie = detect_bot_self_image_intent(raw_description)
        if is_selfie:
            description = self._process_selfie_prompt(
                description,
                raw_description,
                include_selfie_prompt_add=True,
                log_changes=True,
            )
            session_state.set_last_selfie_context(
                self.stream_id,
                description,
                raw_description,
                ttl=float(self.get_config("prompt_generator.inherit_ttl", 0) or 0),
            )
            structured_payload = None

        if self.get_config("prompt_generator.enforce_tag_order", False):
            description = normalize_prompt_order(description)
            structured_payload = self._normalize_structured_order(structured_payload)

        description = self._sanitize_prompt_for_sfw_mode(description)
        structured_payload = self._sanitize_structured_for_sfw_mode(structured_payload)

        model_config = self._get_model_config(is_selfie=is_selfie)
        if not model_config or not model_config.get("base_url"):
            return False, "模型配置无效"

        image_size = model_config.get("nai_size") or model_config.get("default_size", "")

        request_prompt, request_characters = self._select_send_payload(
            description, structured_payload
        )
        try:
            success, result = await self.api_client.generate_image(
                prompt=request_prompt,
                model_config=model_config,
                size=image_size,
                characters=request_characters,
            )
        except Exception as exc:
            logger.error("%s reply 自动跟图生成失败: %r", self.log_prefix, exc, exc_info=True)
            return False, str(exc)

        if not success:
            logger.info("%s reply 自动跟图未成功: %s", self.log_prefix, result)
            return False, str(result)

        send_result = await self._send_image_result(
            result,
            raw_description or description,
            track_as_auto_draw=True,
        )
        return send_result[0], send_result[1] or ""

    async def _assess_auto_draw_trigger(self) -> AdmissionDecision:
        """读取最近用户原话并评估 reply 自动跟图。"""
        user_text, age_seconds = await self._fetch_last_user_text_with_age()
        return self._generation_admission_policy.evaluate_auto_draw(
            stream_id=self.stream_id,
            config=self.plugin_config,
            user_text=user_text,
            user_text_age_seconds=age_seconds,
        )

    def _is_action_guard_enabled(self) -> bool:
        """检查是否启用自动出图保护。"""
        return bool(self.get_config("action_guard.enabled", True))

    async def preflight_action_guard(self) -> AdmissionDecision | None:
        """Action Guard 同步预检：让 Planner 在 RPC 返回时就能拿到拦截原因。

        返回 None 表示 guard 未启用，调用方应放行；否则返回类型化准入结论。
        结果会缓存到 invocation 上，后台 ``handle_action`` 复用同一次评估，避免重复读消息库。
        """
        if not self._is_action_guard_enabled():
            return None
        return await self._assess_action_trigger(reasoning=self.reasoning)

    async def _assess_action_trigger(self, reasoning: str = "") -> AdmissionDecision:
        """Action Guard 评估入口；结果缓存供 handle_action 后台复用。"""
        if self._cached_action_trigger_assessment is not None:
            return self._cached_action_trigger_assessment
        user_text, age_seconds = await self._fetch_last_user_text_with_age()
        result = self._generation_admission_policy.evaluate_action(
            stream_id=self.stream_id,
            config=self.plugin_config,
            user_text=user_text,
            user_text_age_seconds=age_seconds,
            reasoning=reasoning,
        )
        self._cached_action_trigger_assessment = result
        return result

    async def handle_admin_command(self, action: str, param: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai st|sp|set|art|size|help`。"""
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        if action == "help":
            if await self._send_help_image():
                return True, "显示帮助信息", True
            # 渲染失败：回退到纯文本（与图片同源结构化数据，避免双份维护）
            await self.send_text(_HELP_FALLBACK_TEXT)
            return True, "显示帮助信息", True

        is_admin = session_state.is_admin_user(user_id, self.get_config)
        if action in {"st", "sp", "set", "ban", "unban", "banlist"} and not is_admin:
            if action == "set":
                await self.send_text("❌ 只有管理员可以切换生图模型", storage_message=False)
            elif action in {"ban", "unban", "banlist"}:
                await self.send_text("❌ 只有管理员可以管理黑名单", storage_message=False)
            else:
                await self.send_text("❌ 只有管理员可以开启/关闭管理员模式", storage_message=False)
            return False, "没有管理员权限", True

        if action in {"art", "size"} and session_state.is_admin_mode_enabled(platform, chat_id, self.get_config):
            if not is_admin:
                await self.send_text("❌ 当前会话已开启管理员模式，仅管理员可以修改 NAI 配置", storage_message=False)
                return False, "没有权限", True

        if action == "st":
            session_state.set_admin_mode(platform, chat_id, True)
            await self.send_text(
                f"✅ 已在{self._chat_type_text()}中开启 NAI 管理员模式\n"
                "🔒 现在所有 NAI 命令仅管理员可使用"
            )
            return True, "管理员模式已开启", True

        if action == "sp":
            session_state.set_admin_mode(platform, chat_id, False)
            await self.send_text(
                f"✅ 已在{self._chat_type_text()}中关闭 NAI 管理员模式\n"
                "🔓 现在所有人都可使用 NAI 命令"
            )
            return True, "管理员模式已关闭", True

        model_mappings = {
            "3": "nai-diffusion-3",
            "f3": "nai-diffusion-3-furry",
            "4c": "nai-diffusion-4-curated",
            "4": "nai-diffusion-4-full",
            "4.5c": "nai-diffusion-4-5-curated",
            "4.5": "nai-diffusion-4-5-full",
        }
        size_mappings = {
            "竖": "832x1216",
            "竖图": "832x1216",
            "横": "1216x832",
            "横图": "1216x832",
            "方": "1024x1024",
            "方图": "1024x1024",
            "h": "1216x832",
            "v": "832x1216",
            "s": "1024x1024",
        }

        if action == "banlist":
            blacklist_entries = user_blacklist.list_entries()
            if not blacklist_entries:
                await self.send_text("当前黑名单为空", storage_message=False)
                return True, "黑名单为空", True

            lines = ["当前黑名单用户："]
            for entry in blacklist_entries:
                suffix_parts = []
                if entry["created_at"]:
                    suffix_parts.append(f"添加时间: {entry['created_at']}")
                if entry["created_by"]:
                    suffix_parts.append(f"操作人: {entry['created_by']}")

                suffix = f"（{'，'.join(suffix_parts)}）" if suffix_parts else ""
                lines.append(f"- {entry['user_id']}{suffix}")

            await self.send_text("\n".join(lines), storage_message=False)
            return True, "显示黑名单列表", True

        if action in {"ban", "unban"}:
            target_user_id = self._extract_target_user_id(param)
            if not target_user_id:
                await self.send_text(
                    "❌ 请输入目标用户 ID，例如：/nai ban 123456789",
                    storage_message=False,
                )
                return False, "缺少目标用户 ID", True

            if target_user_id == user_id:
                await self.send_text("❌ 不允许将自己加入黑名单", storage_message=False)
                return False, "不允许拉黑自己", True

            if action == "ban":
                added = user_blacklist.add_user(target_user_id, operator_id=user_id)
                if not added:
                    await self.send_text(f"⚠️ 用户 {target_user_id} 已在黑名单中", storage_message=False)
                    return False, "用户已在黑名单中", True

                await self.send_text(
                    f"✅ 已将用户 {target_user_id} 加入黑名单\n"
                    "🔒 该用户现在无法使用本插件任何功能",
                    storage_message=False,
                )
                return True, "已加入黑名单", True

            removed = user_blacklist.remove_user(target_user_id)
            if not removed:
                await self.send_text(f"⚠️ 用户 {target_user_id} 不在黑名单中", storage_message=False)
                return False, "用户不在黑名单中", True

            await self.send_text(f"✅ 已将用户 {target_user_id} 移出黑名单", storage_message=False)
            return True, "已移出黑名单", True

        if action == "set":
            if not param:
                current_model = session_state.get_selected_model(platform, chat_id) or self.get_config(
                    "model.default_model",
                    "nai-diffusion-4-5-full",
                )
                await self.send_text(
                    f"当前模型: {current_model}\n\n"
                    "可用模型:\n"
                    "3 - nai-diffusion-3\n"
                    "f3 - nai-diffusion-3-furry\n"
                    "4c - nai-diffusion-4-curated\n"
                    "4 - nai-diffusion-4-full\n"
                    "4.5c - nai-diffusion-4-5-curated\n"
                    "4.5 - nai-diffusion-4-5-full"
                )
                return True, "显示模型列表", True

            if param not in model_mappings:
                await self.send_text("❌ 无效的模型代号，可用值：3 / f3 / 4c / 4 / 4.5c / 4.5")
                return False, "无效的模型代号", True

            model_name = model_mappings[param]
            session_state.set_selected_model(platform, chat_id, model_name)
            await self.send_text(f"✅ 已切换到模型: {model_name}")
            return True, f"已切换到模型 {model_name}", True

        if action == "art":
            current_model = session_state.get_selected_model(platform, chat_id) or self.get_config(
                "model.default_model",
                "nai-diffusion-4-5-full",
            )
            if "nai-diffusion-3" in current_model:
                config_section = "model_nai3"
            elif "nai-diffusion-4-5" in current_model:
                config_section = "model_nai4_5"
            elif "nai-diffusion-4" in current_model:
                config_section = "model_nai4"
            else:
                await self.send_text("❌ 当前模型不支持画师串切换")
                return False, "模型不支持画师串", True

            artist_presets_raw = self.get_config(f"{config_section}.artist_presets", [])
            artist_presets = session_state._parse_artist_presets(artist_presets_raw)
            if not artist_presets:
                await self.send_text("❌ 当前模型未配置画师串预设")
                return False, "未配置画师串", True

            if not param:
                current_index = session_state.get_effective_artist_index(platform, chat_id, current_model, self.get_config)
                lines = [
                    f"{'→ ' if index == current_index else '  '}{index}. {preset['name']}"
                    for index, preset in enumerate(artist_presets, 1)
                ]
                await self.send_text("\n".join(lines))
                return True, "显示画师串列表", True

            try:
                index = int(param)
            except ValueError:
                await self.send_text("❌ 画师串编号必须是数字")
                return False, "无效的画师串编号", True

            if index < 1 or index > len(artist_presets):
                await self.send_text(f"❌ 无效的画师串编号，可用范围：1-{len(artist_presets)}")
                return False, "无效的画师串编号", True

            session_state.set_selected_artist_index(platform, chat_id, index)
            await self.send_text(f"✅ 已切换到画师串 #{index}\n名称: {artist_presets[index - 1]['name']}")
            return True, f"已切换到画师串 #{index}", True

        if action == "size":
            if not param:
                current_size = session_state.get_selected_size(platform, chat_id) or self.get_config(
                    "model.default_size",
                    "832x1216",
                )
                await self.send_text(
                    f"当前尺寸: {current_size}\n\n"
                    "可用尺寸:\n"
                    "竖/v - 832x1216\n"
                    "横/h - 1216x832\n"
                    "方/s - 1024x1024"
                )
                return True, "显示尺寸列表", True

            if param not in size_mappings:
                await self.send_text("❌ 无效的尺寸代号，可用值：竖/v、横/h、方/s")
                return False, "无效的尺寸代号", True

            session_state.set_selected_size(platform, chat_id, size_mappings[param])
            await self.send_text(f"✅ 已切换到尺寸: {size_mappings[param]}")
            return True, f"已切换到尺寸 {size_mappings[param]}", True

        await self.send_text("使用 /nai help 查看帮助")
        return False, "未知操作", True

    async def handle_recall_switch(self, action: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai on|off`。"""
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        if not session_state.is_admin_user(user_id, self.get_config):
            await self.send_text("❌ 只有管理员可以使用自动撤回控制命令", storage_message=False)
            return False, "没有管理员权限", True

        allowed_groups = self.get_config("auto_recall.allowed_groups", [])
        if allowed_groups and f"{platform}:{chat_id}" not in allowed_groups:
            await self.send_text("❌ 当前会话没有使用自动撤回功能的权限")
            return False, "当前会话没有使用自动撤回功能的权限", True

        if action == "on":
            session_state.set_recall_enabled(platform, chat_id, True)
            delay_seconds = self.get_config("auto_recall.delay_seconds", 5)
            await self.send_text(
                f"✅ 已在{self._chat_type_text()}中开启 NAI 图片自动撤回功能\n"
                f"📝 图片将在发送后 {delay_seconds} 秒自动撤回"
            )
            return True, "自动撤回已开启", True

        session_state.set_recall_enabled(platform, chat_id, False)
        await self.send_text(f"✅ 已在{self._chat_type_text()}中关闭 NAI 图片自动撤回功能")
        return True, "自动撤回已关闭", True

    async def handle_nsfw_command(self, action: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai nsfw`。"""
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        if not session_state.is_admin_user(user_id, self.get_config):
            await self.send_text("❌ 只有管理员可以使用 NSFW 过滤控制命令", storage_message=False)
            return False, "没有管理员权限", True

        if not action:
            current_state = session_state.is_nsfw_filter_enabled(platform, chat_id, self.get_config)
            state_text = "已开启" if current_state else "已关闭"
            await self.send_text(
                f"当前 NSFW 过滤状态: {state_text}\n\n"
                "使用方法:\n"
                "/nai nsfw on - 开启 NSFW 内容过滤\n"
                "/nai nsfw off - 关闭 NSFW 内容过滤",
                storage_message=False,
            )
            return True, "显示 NSFW 过滤状态", True

        enabled = action == "on"
        session_state.set_nsfw_filter_enabled(platform, chat_id, enabled)
        state_text = "开启" if enabled else "关闭"
        await self.send_text(f"✅ 已在{self._chat_type_text()}中{state_text} NSFW 内容过滤", storage_message=False)
        return True, f"NSFW 过滤已{state_text}", True

    async def handle_ref_type_command(self, value: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai ref类型 <character|style|both>`：切换本会话 §20.4 type。

        管理员鉴权与 /nai ref 其它子命令一致（ref 全部仅限管理员）；
        ``both`` 是 ``character&style`` 的输入别名，归一后入 session_state。
        无参数时打印当前态与用法。
        """
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        if not session_state.is_admin_user(user_id, self.get_config):
            await self.send_text("❌ 只有管理员可以切换角色参考类型", storage_message=False)
            return False, "没有管理员权限", True

        current = session_state.get_character_reference_type(
            platform, chat_id, self.get_config
        )

        if not value:
            await self.send_text(
                f"当前角色参考类型: {current}\n\n"
                "使用方法:\n"
                "/nai ref类型 character - 仅提取角色\n"
                "/nai ref类型 style - 仅提取风格\n"
                "/nai ref类型 both - 角色 + 风格（character&style）",
                storage_message=False,
            )
            return True, "显示角色参考类型", True

        alias = value.strip().lower()
        normalized = "character&style" if alias == "both" else alias
        try:
            session_state.set_character_reference_type(platform, chat_id, normalized)
        except ValueError as exc:
            await self.send_text(
                f"❌ {exc}\n可填：character / style / both（=character&style）",
                storage_message=False,
            )
            return False, "类型不合规", True

        await self.send_text(
            f"✅ 已把本会话的角色参考类型切换为：{normalized}",
            storage_message=False,
        )
        return True, "已切换角色参考类型", True

    async def handle_prompt_show_command(self, action: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai pt on|off`。"""
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        if session_state.is_admin_mode_enabled(platform, chat_id, self.get_config):
            if not session_state.is_admin_user(user_id, self.get_config):
                await self.send_text("❌ 当前会话已开启管理员模式，仅管理员可以修改提示词显示设置", storage_message=False)
                return False, "没有权限", True

        enabled = action == "on"
        session_state.set_prompt_show_enabled(platform, chat_id, enabled)
        await self.send_text("✅ 已开启提示词显示" if enabled else "✅ 已关闭提示词显示")
        return True, "提示词显示状态已更新", True

    async def handle_tag_retriever_show_command(self, action: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai tag on|off`。"""
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        if session_state.is_admin_mode_enabled(platform, chat_id, self.get_config):
            if not session_state.is_admin_user(user_id, self.get_config):
                await self.send_text("❌ 当前会话已开启管理员模式，仅管理员可以修改 Danbooru 检索结果显示设置", storage_message=False)
                return False, "没有权限", True

        enabled = action == "on"
        session_state.set_tag_retriever_show_enabled(platform, chat_id, enabled)
        await self.send_text(
            "✅ 已开启 Danbooru 检索结果显示"
            if enabled
            else "✅ 已关闭 Danbooru 检索结果显示"
        )
        return True, "Danbooru 检索结果显示状态已更新", True

    async def handle_models_command(self) -> tuple[bool, str | None, bool]:
        """处理 `/nai models`：拉 ``GET /v1/models`` 展示网关实时模型列表，
        并与 ``[model].available_models`` 对比标注配置漂移。
        """
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        model_config = self.get_config("model", {}) or {}
        if not isinstance(model_config, dict) or not model_config.get("base_url"):
            await self.send_text("❌ NewAPI 网关 base_url 未配置")
            return False, "配置错误", True

        success, payload = await self.api_client.list_models(model_config)
        if not success:
            await self.send_text(f"❌ 获取模型列表失败：{payload}")
            return False, f"list_models 失败: {payload}", True

        remote_models: list[str] = payload if isinstance(payload, list) else []
        if not remote_models:
            await self.send_text("⚠️ 网关返回的模型列表为空")
            return True, "list_models 空结果", True

        configured = list(model_config.get("available_models") or [])
        current = str(model_config.get("default_model") or "").strip()

        configured_set = set(configured)
        remote_set = set(remote_models)
        missing_locally = [m for m in remote_models if m not in configured_set]
        missing_remotely = [m for m in configured if m not in remote_set]

        lines: list[str] = [f"🌐 NewAPI 返回 {len(remote_models)} 个模型："]
        for model_id in remote_models:
            marker = " ⭐(当前默认)" if model_id == current else ""
            local_marker = "" if model_id in configured_set else " 🆕(未列入 available_models)"
            lines.append(f"  • {model_id}{marker}{local_marker}")

        if missing_remotely:
            lines.append("")
            lines.append("⚠️ 以下模型在 available_models 配置里，但网关此次未返回：")
            for model_id in missing_remotely:
                lines.append(f"  • {model_id}")

        if missing_locally:
            lines.append("")
            lines.append("💡 上述带 🆕 的模型可加入 [model].available_models 后用 /nai set 切换。")

        await self.send_text("\n".join(lines))
        return True, f"列出 {len(remote_models)} 个模型", True

    @staticmethod
    def _extract_target_user_id(raw_value: str) -> str:
        """从命令参数中提取目标用户 ID。"""
        text = str(raw_value or "").strip()
        if not text:
            return ""

        for pattern in (
            r"(?:qq|user_id|uid)=(\d+)",
            r"<@!?(\d+)>",
            r"@(\d+)",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return text
