"""NAI Low 插件新版 SDK 运行辅助。

将旧版命令与 Action 的主要业务逻辑迁移到新版 `MaiBotPlugin` 调用方式。
"""

from __future__ import annotations

import base64
import json
import tomllib
from collections.abc import Callable
from uuid import uuid4
from typing import Any, Optional

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from aiohttp import ClientSession, ClientTimeout

from src.common.logger import get_logger
from src.config.model_configs import TaskConfig
from src.llm_models.utils_model import LLMOrchestrator
from src.services import llm_service

from .runtime_recall import (
    MANUAL_RECALL_TTL_SECONDS,
    discard_pending_plugin_image_send,
    extract_plugin_row_message_id,
    is_napcat_action_accepted,
    load_recent_plugin_image_rows,
    load_recent_session_image_rows,
    load_recent_tracked_plugin_image_rows,
    normalize_db_timestamp,
    prune_recent_ids,
    remember_pending_plugin_image_send,
    remember_recent_id,
    resolve_db_path,
    select_recent_plugin_image_row,
    wait_for_formal_message_id,
)

from .core.clients.nai_web_client import NaiWebClient
from .core.constants import NAI_PIC_IMAGE_DISPLAY_MARKER
from .core.mixins.model_config_mixin import ModelConfigMixin
from .core.rules.prompt_rules import PROMPT_GENERATOR_TEMPLATE, SFW_PROMPT_GENERATOR_TEMPLATE
from .core.rules.selfie_rules import (
    detect_explicit_image_request,
    detect_negative_image_intent,
    detect_negative_image_intent_strength,
    detect_selfie_from_output,
    get_selfie_hint,
    merge_selfie_prompt,
)
from .core.services.prompt_memory import render_previous_prompt_block
from .core.services.session_state import session_state
from .core.services.tag_retriever import get_tag_retriever
from .core.services.user_blacklist import user_blacklist
from .core.utils.display_message_helper import build_action_image_display_message
from .core.utils.prompt_output_parser import parse_prompt_from_structured_output
from .core.utils.prompt_postprocessor import (
    normalize_prompt_order,
    remove_selfie_appearance_tags,
    sanitize_sfw_prompt,
    user_mentions_appearance,
)
from .core.utils.random_scene_description import (
    get_random_scene_similarity_score,
    is_random_scene_too_similar,
    normalize_random_scene_description,
)

logger = get_logger("nai_draw_plugin")
_DB_PATH = resolve_db_path(__file__)
_RECENT_MANUAL_RECALL_IDS: dict[str, dict[str, float]] = {}
_NAPCAT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "MaiBot-Napcat-Adapter" / "config.toml"


def _load_napcat_server_config() -> dict[str, Any] | None:
    """读取 Napcat 连接配置，供撤回动作直连使用。"""
    if not _NAPCAT_CONFIG_PATH.is_file():
        return None

    try:
        with _NAPCAT_CONFIG_PATH.open("rb") as fp:
            config_data = tomllib.load(fp)
    except Exception as exc:
        logger.warning("[nai_low] 读取 Napcat 配置失败: %r", exc)
        return None

    server_config = config_data.get("napcat_server")
    if not isinstance(server_config, dict):
        return None

    host = str(server_config.get("host") or "").strip()
    port = server_config.get("port")
    token = str(server_config.get("token") or "").strip()
    timeout = server_config.get("action_timeout_sec", 15.0)

    try:
        normalized_port = int(port)
    except (TypeError, ValueError):
        return None

    try:
        action_timeout = max(1.0, float(timeout))
    except (TypeError, ValueError):
        action_timeout = 15.0

    if not host or normalized_port <= 0:
        return None

    return {
        "ws_url": f"ws://{host}:{normalized_port}",
        "token": token,
        "action_timeout_sec": action_timeout,
    }


class _PinnedTaskLLMOrchestrator(LLMOrchestrator):
    """仅在 nai_low 自定义模型调用中使用的固定模型调度器。"""

    def __init__(self, task_config: TaskConfig, request_type: str = "") -> None:
        self._pinned_task_config = task_config
        super().__init__(task_name="planner", request_type=request_type)

    def _get_task_config_or_raise(self) -> TaskConfig:
        return self._pinned_task_config

    def _refresh_task_config(self) -> TaskConfig:
        latest = self._pinned_task_config
        if latest is not self.model_for_task:
            self.model_for_task = latest
        if list(self.model_usage.keys()) != latest.model_list:
            self.model_usage = {model: self.model_usage.get(model, (0, 0, 0)) for model in latest.model_list}
        return self.model_for_task


async def _find_last_plugin_image_row(
    invocation: "NaiInvocation",
    *,
    limit: int = 120,
    target_send_timestamp: float | None = None,
    exclude_message_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """从宿主消息库中读取最近一条本插件图片消息。"""
    if not getattr(invocation, "stream_id", ""):
        return None

    tracked_rows = load_recent_tracked_plugin_image_rows(
        invocation.stream_id,
        limit=limit,
    )
    tracked_row = select_recent_plugin_image_row(
        tracked_rows,
        target_send_timestamp=target_send_timestamp,
        exclude_message_ids=exclude_message_ids,
    )
    if tracked_row is not None:
        return tracked_row

    marked_rows = load_recent_plugin_image_rows(
        _DB_PATH,
        invocation.stream_id,
        NAI_PIC_IMAGE_DISPLAY_MARKER,
        limit=limit,
    )
    marked_row = select_recent_plugin_image_row(
        marked_rows,
        target_send_timestamp=target_send_timestamp,
        exclude_message_ids=exclude_message_ids,
    )
    if marked_row is not None:
        return marked_row

    fallback_rows = load_recent_session_image_rows(
        _DB_PATH,
        invocation.stream_id,
        limit=limit,
    )
    return select_recent_plugin_image_row(
        fallback_rows,
        target_send_timestamp=target_send_timestamp,
        exclude_message_ids=exclude_message_ids,
    )


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


# Planner reasoning 中暗示"用户明确要求图片"的措辞。仅在拿不到用户原话时作为 fallback。
# 选词保守：只覆盖明显指向"用户/对方请求"的转述，避免把 bot 自身视觉描写误判成 explicit。
_REASONING_EXPLICIT_HINTS: tuple[str, ...] = (
    "用户要求", "用户想看", "用户想要", "用户希望", "用户让",
    "对方要求", "对方想看", "对方想要", "对方希望", "对方让",
    "他要求", "他想看", "她要求", "她想看",
    "明确要求", "明确想看", "明确请求",
    "要我画", "让我画", "叫我画", "要我发", "让我发",
    "追图", "继续画", "再画一张", "再来一张",
)


def _reasoning_implies_explicit_request(reasoning: str) -> bool:
    """fallback：从 Planner reasoning 里识别用户显式请求语义。"""
    if not reasoning:
        return False
    lowered = reasoning.lower()
    return any(hint in reasoning or hint.lower() in lowered for hint in _REASONING_EXPLICIT_HINTS)


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


# 主动出图时往 description 前置的自指标签。selfie 偏镜头近、肖像偏室内近景、
# 场景偏生活照——目的都是让生成的图像感觉像"bot 在分享自己"而不是"画了张陌生人"。
_SELF_IMAGE_HINT_BY_MODE: dict[str, str] = {
    "selfie": "一女 自拍 近景",
    "portrait": "一女 肖像照 近景",
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


def _render_reply_context_block(reply_context_text: str) -> str:
    """渲染 reply 后置跟图专用的"bot 即将说出的回复原文"语境块。

    Reply hook 链路里，description 是关键词拼接（"一女 自拍 近景 窗边"），LLM 看不到 bot
    实际要说的那句话。这个块把原文塞回 prompt，让 LLM 基于具体语境补全画面细节（衣着/光照/
    姿态），避免图与文脱节。其他链路（command / Planner Action）调用方传空字符串即可。
    """
    text = (reply_context_text or "").strip()
    if not text:
        return ""
    return (
        "<bot_reply_context>\n"
        "（这是 bot 本人这一轮即将说出去的回复原文。请基于这段语境扩展画面细节"
        "——衣着、姿态、光照、室内陈设等——让生成的图与文匹配，"
        "而不是仅看 user_request 的关键词。）\n"
        f"{text}\n"
        "</bot_reply_context>"
    )


def _render_reasoning_context_block(reasoning_context_text: str) -> str:
    """渲染 Planner Action 链路专用的"Planner reasoning"语境块。

    Action 链路里，``description`` / 5 个结构化字段都是 Planner 关键词化的二手语义；
    reasoning 才是原始动机和动词/情绪/关系。把 reasoning 塞回模板，让下游 LLM 在
    user_request 失真时能回到原意，避免动作被软化、情绪被默认套模板。
    其他入口（command / reply 自动跟图）传空字符串即可。
    """
    text = (reasoning_context_text or "").strip()
    if not text:
        return ""
    return (
        "<planner_reasoning>\n"
        "（Planner 本轮 reasoning。与 user_request 冲突时以本块为准："
        "动词保持原意，情绪贴 reasoning，不要默认套'迷离/陶醉'。）\n"
        f"{text}\n"
        "</planner_reasoning>"
    )


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

    _recent_random_scenes: list[str] = []
    _max_recent_scenes = 5
    _max_random_scene_attempts = 4
    _random_scene_repeat_threshold = 0.6

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
        self.api_client = NaiWebClient(self)
        self._last_send_timestamp: float | None = None
        # Action Guard 评估缓存：主路径同步预检后，后台 handle_action 复用结果，避免重复读消息库
        self._cached_action_trigger_assessment: dict[str, Any] | None = None

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

    def _sanitize_prompt_for_sfw_mode(self, prompt: str) -> str:
        """在启用 NSFW 过滤时进一步剔除擦边/色情标签。"""
        if not prompt:
            return prompt
        if not session_state.is_nsfw_filter_enabled("stream", self.stream_id, self.get_config):
            return prompt
        return sanitize_sfw_prompt(prompt)

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

    async def _find_last_plugin_image_message_id(
        self,
        *,
        limit: int = 120,
        target_send_timestamp: float | None = None,
        exclude_message_ids: Optional[set[str]] = None,
    ) -> str | None:
        """查找最近一条本插件发送的图片消息。"""
        try:
            row = await _find_last_plugin_image_row(
                self,
                limit=limit,
                target_send_timestamp=target_send_timestamp,
                exclude_message_ids=exclude_message_ids,
            )
        except Exception as exc:
            logger.warning("%s 读取本地消息库失败: %r", self.log_prefix, exc)
            row = None

        if row:
            message_id = extract_plugin_row_message_id(row)
            if message_id:
                return message_id

        return None

    def _get_recent_manual_recall_ids(self) -> set[str]:
        """获取当前会话最近已经尝试手动撤回过的消息 ID。"""
        return prune_recent_ids(
            _RECENT_MANUAL_RECALL_IDS,
            getattr(self, "stream_id", ""),
            ttl_seconds=MANUAL_RECALL_TTL_SECONDS,
        )

    def _remember_recent_manual_recall_id(self, message_id: str) -> None:
        """记录当前会话刚尝试手动撤回过的消息 ID。"""
        remember_recent_id(
            _RECENT_MANUAL_RECALL_IDS,
            getattr(self, "stream_id", ""),
            message_id,
        )

    def _get_manual_recall_max_age_seconds(self) -> float:
        """读取手动撤回允许命中的最老图片年龄。"""
        try:
            raw_value = self.get_config("auto_recall.manual_max_age_seconds", 3600)
        except Exception:
            raw_value = 3600

        try:
            max_age_seconds = float(raw_value)
        except (TypeError, ValueError):
            return 3600.0

        return max(0.0, max_age_seconds)

    async def _resolve_local_plugin_image_message_id(
        self,
        *,
        limit: int = 120,
        target_send_timestamp: float | None = None,
        exclude_message_ids: set[str] | None = None,
        initial_row: dict[str, Any] | None = None,
        id_wait_seconds: float | None = None,
    ) -> str | None:
        """围绕目标时间轮询本地消息库，等待占位 ID 变为正式 ID。"""
        if id_wait_seconds is None:
            try:
                id_wait_seconds = max(0.0, float(self.get_config("auto_recall.id_wait_seconds", 15) or 15))
            except (TypeError, ValueError):
                id_wait_seconds = 15.0
        else:
            id_wait_seconds = max(0.0, float(id_wait_seconds))

        async def _row_loader() -> dict[str, Any] | None:
            try:
                return await _find_last_plugin_image_row(
                    self,
                    limit=limit,
                    target_send_timestamp=target_send_timestamp,
                    exclude_message_ids=exclude_message_ids,
                )
            except Exception as exc:
                logger.warning("%s 轮询本地消息库失败: %r", self.log_prefix, exc)
                return None

        return await wait_for_formal_message_id(
            _row_loader,
            initial_row=initial_row,
            id_wait_seconds=id_wait_seconds,
        )

    async def _try_recall_message(self, message_id: str) -> bool:
        """优先使用 Napcat API 撤回消息。"""
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or not getattr(self, "stream_id", ""):
            return False

        async def _try_direct_napcat_action() -> bool:
            if not normalized_message_id.isdigit():
                logger.warning("%s 撤回失败：消息ID不是纯数字: %s", self.log_prefix, normalized_message_id)
                return False

            server_config = _load_napcat_server_config()
            if server_config is None:
                logger.warning("%s 未找到可用的 Napcat 连接配置，无法直连撤回", self.log_prefix)
                return False

            headers = {"Authorization": f"Bearer {server_config['token']}"} if server_config.get("token") else {}
            timeout = float(server_config.get("action_timeout_sec", 15.0))
            echo_id = uuid4().hex
            payload = {
                "action": "delete_msg",
                "params": {"message_id": int(normalized_message_id)},
                "echo": echo_id,
            }

            try:
                async with ClientSession(headers=headers, timeout=ClientTimeout(total=None, connect=10)) as session:
                    async with session.ws_connect(
                        str(server_config["ws_url"]),
                        heartbeat=None,
                    ) as ws:
                        await ws.send_str(json.dumps(payload, ensure_ascii=False))

                        deadline = asyncio.get_running_loop().time() + timeout
                        while True:
                            remaining = deadline - asyncio.get_running_loop().time()
                            if remaining <= 0:
                                raise TimeoutError(f"Napcat delete_msg 超时 ({timeout}s)")

                            message = await asyncio.wait_for(ws.receive(), timeout=remaining)
                            if message.type.name != "TEXT":
                                continue

                            response = json.loads(message.data)
                            if str(response.get("echo") or "").strip() != echo_id:
                                continue

                            logger.debug("%s 撤回(napcat-ws) 结果: %r", self.log_prefix, response)
                            return is_napcat_action_accepted(response)
            except Exception as exc:
                logger.warning("%s 撤回(napcat-ws) 失败: %r", self.log_prefix, exc)
                return False

        async def _capability_api_call(api_name: str, **api_args: Any) -> Any:
            api_proxy = getattr(self.ctx, "api", None)
            if api_proxy is not None and hasattr(api_proxy, "call"):
                return await api_proxy.call(api_name, **api_args)

            call_capability = getattr(self.ctx, "call_capability", None)
            if callable(call_capability):
                return await call_capability("api.call", api_name=api_name, args=api_args)

            raise AttributeError("当前上下文不支持 API 能力调用")

        async def _try_napcat_delete_api() -> bool:
            if not normalized_message_id.isdigit():
                logger.warning(
                    "%s 撤回失败：消息ID不是纯数字，无法调用 napcat 删除 API: %s",
                    self.log_prefix,
                    normalized_message_id,
                )
                return False

            try:
                result = await _capability_api_call(
                    "adapter.napcat.message.delete_msg",
                    message_id=int(normalized_message_id),
                )
                logger.debug("%s 撤回(napcat-api) 结果: %r", self.log_prefix, result)
                if not is_napcat_action_accepted(result):
                    return False
                return True
            except Exception as exc:
                logger.warning("%s 撤回(napcat-api) 失败: %r", self.log_prefix, exc)
                return False

        if await _try_direct_napcat_action():
            return True
        return await _try_napcat_delete_api()

    async def _schedule_auto_recall(self) -> None:
        """调度自动撤回。"""
        platform, chat_id, _ = self._get_chat_identity()
        if not chat_id:
            return
        if not session_state.is_recall_enabled(platform, chat_id, self.get_config):
            return

        try:
            delay_seconds = max(0.0, float(self.get_config("auto_recall.delay_seconds", 5) or 5))
        except (TypeError, ValueError):
            delay_seconds = 5.0
        try:
            id_wait_seconds = max(0.0, float(self.get_config("auto_recall.id_wait_seconds", 15) or 15))
        except (TypeError, ValueError):
            id_wait_seconds = 15.0

        target_send_timestamp = getattr(self, "_last_send_timestamp", None)

        async def _job() -> None:
            await asyncio.sleep(delay_seconds)
            message_id = await self._resolve_local_plugin_image_message_id(
                limit=120,
                target_send_timestamp=target_send_timestamp,
                id_wait_seconds=id_wait_seconds,
            )
            if not message_id:
                logger.warning("%s 自动撤回未命中消息", self.log_prefix)
                return
            success = await self._try_recall_message(message_id)
            if success:
                logger.info("%s 已自动撤回消息 %s", self.log_prefix, message_id)
            else:
                logger.warning("%s 自动撤回失败: %s", self.log_prefix, message_id)

        self.plugin._track_task(asyncio.create_task(_job()))

    async def _download_remote_image_as_base64(self, url: str) -> str | None:
        """下载远程图片并转为 Base64。"""
        if _looks_like_generation_request_url(url):
            logger.warning("%s 远程图片URL仍是生成接口，停止自动补拉以避免重复扣费", self.log_prefix)
            return None

        try:
            model_config = self._get_model_config()
            if not isinstance(model_config, dict):
                model_config = {}

            parsed_url = urlsplit(url)
            request_base_url = ""
            if parsed_url.scheme and parsed_url.netloc:
                request_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            elif isinstance(model_config.get("base_url"), str):
                request_base_url = str(model_config.get("base_url") or "").strip()

            request_headers = (
                NaiWebClient._build_request_headers(request_base_url)
                if request_base_url
                else dict(NaiWebClient._DEFAULT_REQUEST_HEADERS)
            )
            request_timeout = NaiWebClient._resolve_request_timeout(model_config)
            proxy_mode = NaiWebClient._resolve_proxy_mode(model_config)
            response = await self.api_client._send_request_with_retry(
                url,
                {},
                proxy_mode,
                request_timeout,
                request_headers,
            )
        except requests.RequestException as exc:
            logger.error("%s 下载远程图片失败: %r", self.log_prefix, exc, exc_info=True)
            return None
        except Exception as exc:
            logger.error("%s 下载远程图片异常: %r", self.log_prefix, exc, exc_info=True)
            return None

        if response.status_code != 200:
            logger.warning("%s 下载远程图片返回 HTTP %s", self.log_prefix, response.status_code)
            return None

        content_type = str(response.headers.get("content-type") or "").lower()
        if not content_type.startswith("image/"):
            response_text = NaiWebClient._get_response_text(response)
            if "application/json" in content_type or NaiWebClient._looks_like_html_response(
                content_type,
                response_text,
            ):
                logger.warning("%s 下载远程图片收到非图片响应: %s", self.log_prefix, content_type or "unknown")
                return None

        content = response.content
        if not content:
            logger.warning("%s 下载远程图片内容为空", self.log_prefix)
            return None

        return base64.b64encode(content).decode("utf-8")

    async def _send_base64_image_result(self, image_base64: str, display_message: str) -> bool:
        """以 Base64 image 段直发图片到平台。

        maim_message + napcat 协议原生支持 base64 image segment，napcat 自行落盘
        后再投递；插件无需也不应当依赖 ``file://`` 本地路径——一旦 napcat 与本插件
        不在同一文件系统（如 napcat 跑在容器内），``file://`` 引用就无法被读取。
        """
        return await self.send_custom(
            "image",
            image_base64,
            display_message=display_message,
        )

    async def _send_image_url_with_fallback(self, image_url: str, display_message: str) -> bool:
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
        return await self._send_base64_image_result(image_base64, display_message)

    async def manual_recall(self) -> tuple[bool, str | None, bool]:
        """执行 `/nai 撤回`。"""
        logger.info("%s [手动撤回] 收到撤回请求, stream_id=%s", self.log_prefix, self.stream_id)
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True
        try:
            return await self._do_manual_recall()
        except Exception as exc:
            logger.error("%s [手动撤回] 未预期异常: %r", self.log_prefix, exc, exc_info=True)
            try:
                await self.send_text("❌ 撤回过程出现内部错误", storage_message=False)
            except Exception:
                pass
            return False, "撤回内部错误", True

    async def _do_manual_recall(self) -> tuple[bool, str | None, bool]:
        """手动撤回的核心逻辑。"""
        recent_excludes = self._get_recent_manual_recall_ids()
        attempted_ids: set[str] = set(recent_excludes)
        max_attempts = 5
        max_age_seconds = self._get_manual_recall_max_age_seconds()
        current_time = time.time()
        skipped_stale_rows = False
        attempted_recall = False

        for _ in range(max_attempts):
            row = await _find_last_plugin_image_row(
                self,
                limit=300,
                exclude_message_ids=attempted_ids,
            )
            initial_message_id = extract_plugin_row_message_id(row)
            if not initial_message_id:
                break

            target_send_timestamp = normalize_db_timestamp(row.get("timestamp")) if row else None
            if (
                max_age_seconds > 0
                and target_send_timestamp is not None
                and current_time - target_send_timestamp > max_age_seconds
            ):
                skipped_stale_rows = True
                attempted_ids.add(initial_message_id)
                logger.info(
                    "%s [手动撤回] 跳过超出撤回窗口的图片: %s age=%.1fs",
                    self.log_prefix,
                    initial_message_id,
                    current_time - target_send_timestamp,
                )
                continue

            resolved_message_id = await self._resolve_local_plugin_image_message_id(
                limit=300,
                target_send_timestamp=target_send_timestamp,
                exclude_message_ids=attempted_ids,
                initial_row=row,
            )
            message_id = str(resolved_message_id or initial_message_id).strip()

            current_attempt_ids = {
                str(initial_message_id or "").strip(),
                str(message_id or "").strip(),
            }
            current_attempt_ids.discard("")
            attempted_ids.update(current_attempt_ids)

            logger.info("%s [手动撤回] 准备撤回消息: %s", self.log_prefix, message_id)
            attempted_recall = True
            success = await self._try_recall_message(message_id)
            if success:
                for recent_id in current_attempt_ids:
                    self._remember_recent_manual_recall_id(recent_id)
                await self.send_text("✅ 已撤回", storage_message=False)
                return True, "手动撤回成功", True

            logger.warning("%s [手动撤回] 撤回失败，尝试上一条图片", self.log_prefix)

        for recent_id in attempted_ids:
            self._remember_recent_manual_recall_id(recent_id)

        if attempted_ids == recent_excludes or (skipped_stale_rows and not attempted_recall):
            logger.info("%s [手动撤回] 未找到可撤回的图片消息", self.log_prefix)
            not_found_text = "❌ 找不到可撤回的图片（直接发送 /nai 撤回 即可按顺序撤回最近一张）"
            if skipped_stale_rows:
                not_found_text = "❌ 找不到近期可撤回的图片（图片可能已超过平台撤回时限）"
            await self.send_text(
                not_found_text,
                storage_message=False,
            )
            return False, "找不到可撤回的消息", True

        await self.send_text(
            "❌ 撤回失败（可能消息已被删除、超过撤回时限、或 bot 无权撤回）",
            storage_message=False,
        )
        return False, "手动撤回失败", True

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
                )
            elif final_image_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                remember_pending_plugin_image_send(self.stream_id, self._last_send_timestamp)
                send_ok = await self._send_base64_image_result(
                    final_image_data,
                    display_message,
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
            discard_pending_plugin_image_send(self.stream_id, self._last_send_timestamp)
            await self.send_text("图片发送失败")
            return False, "发送失败", True

        if track_as_auto_draw:
            session_state.set_last_auto_draw_sent_at(self.stream_id, self._last_send_timestamp)
        else:
            session_state.set_last_action_image_sent_at(self.stream_id, self._last_send_timestamp)
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

        if policy == "auto" and not user_specified:
            description = remove_selfie_appearance_tags(description)

        if include_selfie_prompt_add and selfie_prompt_add:
            description = merge_selfie_prompt(description, selfie_prompt_add)

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

    def _get_prompt_generator_config(self) -> dict[str, Any]:
        """返回提示词生成配置。"""
        config = self.get_config("prompt_generator", {})
        return config if isinstance(config, dict) else {}

    def _get_random_scene_config(self) -> dict[str, Any]:
        """返回随机场景配置。"""
        config = self.get_config("random_scene", {})
        return config if isinstance(config, dict) else {}

    def _resolve_task_name(self, preferred_name: str) -> str | None:
        """解析当前可用的任务名。"""
        models = llm_service.get_available_models()
        if not models:
            return None

        for candidate in [preferred_name, "planner", "replyer"]:
            normalized = str(candidate or "").strip()
            if normalized and normalized in models:
                return normalized

        return next(iter(models.keys()), None)

    async def _request_llm_text(
        self,
        prompt: str,
        *,
        request_type: str,
        generator_config: dict[str, Any],
        default_model_name: str,
        default_temperature: float,
        default_max_tokens: int,
    ) -> str | None:
        """统一发起文本生成请求。"""
        custom_model = generator_config.get("custom_model")
        temperature_raw = generator_config.get("temperature", default_temperature)
        max_tokens_raw = generator_config.get("max_tokens", default_max_tokens)

        try:
            temperature = float(temperature_raw)
        except (TypeError, ValueError):
            temperature = default_temperature

        try:
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError):
            max_tokens = default_max_tokens

        if isinstance(custom_model, dict) and custom_model.get("model_list"):
            try:
                model_list = custom_model.get("model_list", [])
                normalized_model_list = [str(item).strip() for item in (model_list if isinstance(model_list, list) else [model_list])]
                normalized_model_list = [item for item in normalized_model_list if item]
                if normalized_model_list:
                    pinned_task = TaskConfig(
                        model_list=normalized_model_list,
                        max_tokens=int(custom_model.get("max_tokens", max_tokens) or max_tokens),
                        temperature=float(custom_model.get("temperature", temperature) or temperature),
                        slow_threshold=float(custom_model.get("slow_threshold", 30.0) or 30.0),
                        selection_strategy="random",
                    )
                    orchestrator = _PinnedTaskLLMOrchestrator(pinned_task, request_type=request_type)
                    result = await orchestrator.generate_response_async(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    response_text = (result.response or "").strip()
                    if response_text:
                        return response_text
                    logger.warning("%s 自定义提示词模型返回空响应，回退到宿主任务模型", self.log_prefix)
            except Exception as exc:
                logger.warning(
                    "%s 自定义提示词模型调用失败，回退到宿主任务模型: %s",
                    self.log_prefix,
                    exc,
                )

        task_name = self._resolve_task_name(str(generator_config.get("model_name", "") or default_model_name))
        if not task_name:
            return None

        result = await llm_service.generate(
            llm_service.LLMServiceRequest(
                task_name=task_name,
                request_type=request_type,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not result.success or not result.completion.response:
            return None
        return result.completion.response.strip()

    def _render_generator_prompt(
        self,
        template: str,
        request_text: str,
        *,
        include_custom_system_prompt: bool = True,
        previous_prompt: str = "",
        previous_request: str = "",
        last_selfie_prompt: str = "",
        last_selfie_request: str = "",
        last_selfie_scene: str = "",
        last_selfie_anchor: Optional[dict[str, list[str]]] = None,
        reply_context_text: str = "",
        reasoning_context_text: str = "",
    ) -> str:
        """渲染提示词生成模板。"""
        custom_system_prompt = ""
        if include_custom_system_prompt:
            custom_system_prompt = str(self.get_config("custom_prompt.system_prompt", "") or "").strip()
        if custom_system_prompt:
            custom_system_prompt = custom_system_prompt + "\n\n"

        previous_block = render_previous_prompt_block(previous_prompt, previous_request)
        selfie_scene_context = self._build_selfie_scene_context(
            request_text,
            last_selfie_prompt=last_selfie_prompt,
            last_selfie_request=last_selfie_request,
            last_selfie_scene=last_selfie_scene,
            last_selfie_anchor=last_selfie_anchor,
        )
        reply_context_block = _render_reply_context_block(reply_context_text)
        reasoning_context_block = _render_reasoning_context_block(reasoning_context_text)
        prompt = template.replace("<<CUSTOM_SYSTEM_PROMPT>>", custom_system_prompt).strip()
        prompt = prompt.replace("<<PREVIOUS_PROMPT>>", previous_block).strip()
        prompt = prompt.replace("<<REPLY_CONTEXT>>", reply_context_block).strip()
        prompt = prompt.replace("<<REASONING_CONTEXT>>", reasoning_context_block).strip()
        prompt = prompt.replace("<<CURRENT_TIME_CONTEXT>>", self._build_current_time_context()).strip()
        prompt = prompt.replace("<<SELFIE_HINT>>", get_selfie_hint()).strip()
        prompt = prompt.replace("<<SELFIE_SCENE_CONTEXT>>", selfie_scene_context).strip()
        prompt = prompt.replace("<<USER_REQUEST>>", request_text.strip() or "N/A")
        return prompt

    async def _retrieve_tag_candidates(self, request_text: str) -> str:
        """执行 Danbooru tag 检索增强。"""
        try:
            retriever_config = self.get_config("tag_retriever", {}) or {}
            if not isinstance(retriever_config, dict) or not retriever_config.get("enabled", False):
                return ""

            mode = str(retriever_config.get("mode", "local") or "local").strip().lower()
            logger.info(f"{self.log_prefix} Tag 检索已启用，模式={mode}，query='{request_text[:30]}'")

            if mode == "online":
                return await self._retrieve_online_tag_candidates(request_text, retriever_config)
            return await self._retrieve_local_tag_candidates(request_text, retriever_config)
        except Exception as exc:
            logger.warning(f"{self.log_prefix} Tag 检索失败，已跳过: {exc}")
            return ""

    async def _retrieve_online_tag_candidates(
        self,
        request_text: str,
        retriever_config: dict[str, Any],
    ) -> str:
        """执行在线 Danbooru 检索，失败时回退到本地检索。"""
        try:
            from .core.services.danbooru_online_retriever import get_online_retriever
        except Exception as exc:
            logger.warning(f"{self.log_prefix} Tag 在线检索初始化失败，回退到本地检索: {exc}")
            return await self._retrieve_local_tag_candidates(request_text, retriever_config)

        retriever = get_online_retriever(
            enabled=True,
            base_url=retriever_config.get("api_url", "https://sakizuki-danboorusearch.hf.space/api"),
            timeout=retriever_config.get("timeout", 90.0),
            search_limit=retriever_config.get("search_limit", 30),
            search_top_k=retriever_config.get("search_top_k", 5),
            related_limit=retriever_config.get("related_limit", 20),
            related_seed_count=retriever_config.get("related_seed_count", 8),
            show_nsfw=retriever_config.get("show_nsfw", True),
            popularity_weight=retriever_config.get("popularity_weight", 0.15),
        )
        if not retriever:
            return ""

        results = await retriever.retrieve(query=request_text)
        search_count = len(results.get("search", []))
        related_count = len(results.get("related", []))
        if search_count == 0 and related_count == 0:
            logger.info(f"{self.log_prefix} Tag 在线检索无结果，回退到本地检索")
            return await self._retrieve_local_tag_candidates(request_text, retriever_config)

        logger.info(
            f"{self.log_prefix} Tag 在线检索命中："
            f"query='{request_text[:30]}' search={search_count} related={related_count}"
        )
        return retriever.format_candidates(results)

    async def _retrieve_local_tag_candidates(
        self,
        request_text: str,
        retriever_config: dict[str, Any],
    ) -> str:
        """执行本地 Danbooru 检索。"""
        retriever = get_tag_retriever(
            enabled=True,
            top_k=retriever_config.get("top_k", 20),
            min_score=retriever_config.get("min_score", 0.3),
        )
        if not retriever:
            return ""

        results = await retriever.retrieve(
            query=request_text,
            top_k=retriever_config.get("top_k", 20),
            min_score=retriever_config.get("min_score", 0.3),
        )
        if not results:
            return ""

        tag_list = ", ".join(f"{item['cn']}→{item['tag']}({item['score']})" for item in results)
        logger.info(f"{self.log_prefix} Tag 本地检索命中：{tag_list}")
        return retriever.format_candidates(results)

    async def _generate_prompt_with_llm(
        self,
        request_text: str,
        *,
        allow_inherit: bool,
        include_custom_system_prompt: bool = True,
        reply_context_text: str = "",
        reasoning_context_text: str = "",
    ) -> str | None:
        """将自然语言描述转换为提示词。

        ``reply_context_text`` 仅在 reply 后置自动跟图链路传入，作为 bot 即将说出的回复
        原文喂给 LLM；``reasoning_context_text`` 仅在 Planner Action 链路传入，作为本轮
        出图的原始动机/动词/情绪语义喂给 LLM。其他入口传空字符串即可，渲染时占位符会被
        消解为空。
        """
        request_text = str(request_text or "").strip()
        if not request_text:
            return None

        generator_config = self._get_prompt_generator_config()
        output_format = str(generator_config.get("output_format", "json") or "json").strip().lower()
        nsfw_enabled = session_state.is_nsfw_filter_enabled("stream", self.stream_id, self.get_config)

        if output_format == "json":
            from .core.rules.prompt_rules import PROMPT_GENERATOR_JSON_TEMPLATE, SFW_PROMPT_GENERATOR_JSON_TEMPLATE

            default_template = SFW_PROMPT_GENERATOR_JSON_TEMPLATE if nsfw_enabled else PROMPT_GENERATOR_JSON_TEMPLATE
        else:
            default_template = SFW_PROMPT_GENERATOR_TEMPLATE if nsfw_enabled else PROMPT_GENERATOR_TEMPLATE

        previous_prompt = ""
        previous_request = ""
        last_selfie_prompt = ""
        last_selfie_request = ""
        last_selfie_scene = ""
        last_selfie_anchor: dict[str, list[str]] = {}
        if allow_inherit and self.stream_id:
            inherit_ttl_raw = self.get_config("prompt_generator.inherit_ttl", 0)
            try:
                inherit_ttl = float(inherit_ttl_raw or 0)
            except (TypeError, ValueError):
                inherit_ttl = 0.0
            previous_prompt, previous_request = session_state.get_last_nai_context(self.stream_id, ttl=inherit_ttl)
            (
                last_selfie_prompt,
                last_selfie_request,
                last_selfie_scene,
                last_selfie_anchor,
            ) = session_state.get_last_selfie_context(self.stream_id, ttl=inherit_ttl)
            previous_prompt = previous_prompt or ""
            previous_request = previous_request or ""

        prompt_template = str(generator_config.get("prompt_template") or default_template)
        # 仅在 NSFW 模板（即未开启 NSFW 过滤）路径下注入 custom_prompt.system_prompt；
        # SFW 模板要保持安全输出，不能被破限词颠覆
        effective_include_custom_system_prompt = include_custom_system_prompt and not nsfw_enabled
        prompt = self._render_generator_prompt(
            prompt_template,
            request_text,
            include_custom_system_prompt=effective_include_custom_system_prompt,
            previous_prompt=previous_prompt if allow_inherit else "",
            previous_request=previous_request if allow_inherit else "",
            last_selfie_prompt=last_selfie_prompt if allow_inherit else "",
            last_selfie_request=last_selfie_request if allow_inherit else "",
            last_selfie_scene=last_selfie_scene if allow_inherit else "",
            last_selfie_anchor=last_selfie_anchor if allow_inherit else None,
            reply_context_text=reply_context_text,
            reasoning_context_text=reasoning_context_text,
        )
        tag_candidates = await self._retrieve_tag_candidates(request_text)
        prompt = prompt.replace("<<TAG_CANDIDATES>>", tag_candidates).strip()

        response = await self._request_llm_text(
            prompt,
            request_type="nai_draw_plugin.prompt_generator",
            generator_config=generator_config,
            default_model_name="planner",
            default_temperature=0.2,
            default_max_tokens=200,
        )
        if not response:
            return None

        cleaned_prompt = self._cleanup_llm_prompt(response)
        if not cleaned_prompt:
            return None

        if allow_inherit and self.stream_id:
            session_state.set_last_nai_context(self.stream_id, cleaned_prompt, request_text)
        return cleaned_prompt

    async def _generate_random_description(self, *, selfie: bool = False) -> str | None:
        """生成随机场景描述。"""
        random_config = self._get_random_scene_config()

        best_candidate: str | None = None
        best_score: float | None = None
        rejected_candidates: list[str] = []

        for attempt in range(self._max_random_scene_attempts):
            prompt = self._build_random_scene_prompt(selfie=selfie, rejected_candidates=rejected_candidates)
            response = await self._request_llm_text(
                prompt,
                request_type="nai_draw_plugin.random_scene",
                generator_config=random_config,
                default_model_name="planner",
                default_temperature=1.0,
                default_max_tokens=200,
            )
            if not response:
                continue

            lines = [line.strip() for line in response.splitlines() if line.strip()]
            if not lines:
                continue

            candidate = normalize_random_scene_description(lines[0])
            score = get_random_scene_similarity_score(candidate, self._recent_random_scenes)
            if not is_random_scene_too_similar(
                candidate,
                self._recent_random_scenes,
                threshold=self._random_scene_repeat_threshold,
            ):
                self._remember_random_scene(candidate)
                return candidate

            rejected_candidates.append(candidate)
            if best_score is None or score < best_score:
                best_candidate = candidate
                best_score = score
            logger.info("%s 随机场景过于相似，重试中: %.2f %s", self.log_prefix, score, candidate)

        if best_candidate:
            self._remember_random_scene(best_candidate)
        return best_candidate

    def _build_random_scene_prompt(
        self,
        *,
        selfie: bool = False,
        rejected_candidates: Optional[list[str]] = None,
    ) -> str:
        """构造随机场景提示。"""
        selfie_extra = ""
        if selfie:
            selfie_extra = (
                "\n\n额外要求（自拍模式）：\n"
                "- 必须明确是自拍，输出中包含“自拍”或具体自拍方式\n"
                "- 自拍内容同样要明确偏成人向\n"
                "- 自拍场景和方式也要多样化"
            )

        prompt = (
            "随机生成一个二次元 NSFW 场景，并用空格分隔的中文短标签描述它。\n\n"
            "要求：\n"
            "- 题材不限，强度不限，可以是任何成人向内容\n"
            "- 结果必须具体、可视化、适合转成 Danbooru 风格标签\n"
            "- 只输出 1 行，包含 6-10 个中文短标签\n"
            "- 标签尽量简短，使用明确视觉概念，不要写成句子\n"
            "- 标签尽量覆盖人数、人物构成、状态、互动、视角、场景\n"
            "- 不要和最近结果过于相似，尽量主动切换题材和画面类型"
            f"{selfie_extra}"
        )

        if self._recent_random_scenes:
            prompt += "\n\n以下是最近已生成过的内容，禁止与它们重复或相似：\n"
            prompt += "\n".join(self._recent_random_scenes)

        if rejected_candidates:
            prompt += "\n\n以下候选刚刚被判定为过于相似，禁止继续沿着这些方向小修小补：\n"
            prompt += "\n".join(rejected_candidates)

        # 与主提示词生成保持一致：仅在 NSFW 模式（未开启过滤）下把 custom_prompt.system_prompt 拼到最前
        nsfw_enabled = session_state.is_nsfw_filter_enabled("stream", self.stream_id, self.get_config)
        if not nsfw_enabled:
            custom_system_prompt = str(self.get_config("custom_prompt.system_prompt", "") or "").strip()
            if custom_system_prompt:
                prompt = f"{custom_system_prompt}\n\n{prompt}"

        return prompt

    @classmethod
    def _remember_random_scene(cls, result: str) -> None:
        """记录最近的随机场景。"""
        if not result:
            return
        cls._recent_random_scenes.append(result)
        if len(cls._recent_random_scenes) > cls._max_recent_scenes:
            cls._recent_random_scenes.pop(0)

    def _build_current_time_context(self) -> str:
        """构造当前时间上下文。"""
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 8:
            period = "清晨"
        elif 8 <= hour < 11:
            period = "上午"
        elif 11 <= hour < 14:
            period = "中午"
        elif 14 <= hour < 17:
            period = "下午"
        elif 17 <= hour < 19:
            period = "傍晚"
        elif 19 <= hour < 23:
            period = "夜晚"
        else:
            period = "深夜"
        return (
            "<current_time_context>\n"
            f"当前本地时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（{period}）。\n"
            "仅在用户未明确指定时，用于补全时间、光线和背景氛围。\n"
            "</current_time_context>"
        )

    def _build_selfie_scene_context(
        self,
        request_text: str,
        *,
        last_selfie_prompt: str = "",
        last_selfie_request: str = "",
        last_selfie_scene: str = "",
        last_selfie_anchor: Optional[dict[str, list[str]]] = None,
    ) -> str:
        """为 Action 的自拍/展示照连续发图构建 LLM 上下文。"""
        current_request = str(request_text or "").strip()
        previous_prompt = str(last_selfie_prompt or "").strip()
        if not self._is_likely_selfie_request(current_request, previous_prompt):
            return ""

        lines = [
            "<selfie_scene_context>",
            "这轮请求很可能属于 bot 本人自拍/展示照 的连续发图。",
            "若用户没有明确要求切换场景、换穿搭或改光线，默认延续上一轮的背景、穿搭、时间氛围与构图重点。",
            "服装连续性要尽量真实：若用户没有明确要求换衣服、换颜色、换材质、换风格，默认延续上一轮服装款式、主色、材质、袜子和鞋子的视觉设定，不要突然从白衣变黑衣，或从针织变皮衣。",
            "如果用户明确指定了本轮想看的重点（如黑丝、鞋子、腿部、全身穿搭、背景），优先保留该重点，并选择能看清它的构图。",
        ]
        if last_selfie_request:
            lines.append(f"上一轮用户请求：{last_selfie_request.strip()}")
        if previous_prompt:
            lines.append(f"上一轮自拍提示词：{previous_prompt}")
        lines.append("</selfie_scene_context>")
        return "\n".join(lines)

    def _is_likely_selfie_request(self, request_text: str, last_selfie_prompt: str = "") -> bool:
        """判断当前请求是否属于自拍/肖像/展示照连续请求。

        用于决定是否给 LLM 注入"自拍连续场景"上下文，并不影响 Action 是否触发。
        """
        text = str(request_text or "").strip()
        if not text:
            return False

        # 强信号：含画图/自拍/肖像/想看 bot/追图等关键词，统一走 selfie_rules
        if detect_explicit_image_request(text):
            return True

        # 隐式追图：仅在上一轮已是自拍/肖像时，识别少量未在显式关键词里的延续表达
        if last_selfie_prompt and detect_selfie_from_output(last_selfie_prompt):
            continuation_patterns = [
                r"继续", r"还是.*", r"来点不一样", r"换成.+", r"改成.+",
                r"换地方", r"同一个场景", r"同样背景",
            ]
            return any(re.search(pattern, text) for pattern in continuation_patterns)

        return False

    def _extract_selfie_anchor_data(self, prompt: str) -> dict[str, list[str]]:
        """自拍连续性不再使用结构化锚点，统一交给 LLM 自行判断。"""
        return {}

    def _format_selfie_anchor_summary(self, anchor_data: dict[str, list[str]]) -> str:
        """自拍连续性不再输出锚点摘要。"""
        return ""

    def _normalize_prompt_tags(self, prompt: str) -> list[str]:
        """将提示词切分并清洗为可分析标签。"""
        raw_tags = [segment.strip() for segment in prompt.replace("\n", ",").split(",") if segment.strip()]
        normalized_tags: list[str] = []
        for tag in raw_tags:
            cleaned = re.sub(r"^-?\d+(?:\.\d+)?::", "", tag.strip())
            cleaned = cleaned.replace("::", "")
            cleaned = cleaned.strip("{}[]() ")
            if cleaned:
                normalized_tags.append(cleaned.lower())
        return normalized_tags

    def _cleanup_llm_prompt(self, prompt: str) -> str:
        """清理 LLM 返回的提示词。"""
        if not prompt:
            return ""

        parsed_prompt = parse_prompt_from_structured_output(prompt)
        if parsed_prompt:
            return parsed_prompt

        cleaned = prompt.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
            if "\n" in cleaned:
                first_line, rest = cleaned.split("\n", 1)
                if first_line.strip().isalpha() and len(first_line.strip()) < 15:
                    cleaned = rest.strip()

        cleaned = re.sub(r"^\s*prompt\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("，", ", ")
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip("` \n")

    async def handle_nai_draw(self, description: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai`。"""
        try:
            if not await self.ensure_generation_permission():
                return False, "没有权限", True

            description = str(description or "").strip()
            if not description:
                await self.send_text("请输入你想画的内容，例如：/nai 画一张初音未来")
                return False, "未提供描述", True

            is_random_selfie = description in {"随机自拍", "random selfie"}
            if description in {"随机", "random", "rand"} or is_random_selfie:
                description = await self._generate_random_description(selfie=is_random_selfie) or ""
                if not description:
                    await self.send_text("随机场景生成失败，请稍后再试~")
                    return False, "随机生成失败", True

            generated_prompt = await self._generate_prompt_with_llm(
                description,
                allow_inherit=False,
                # NSFW 模板路径会自动注入 custom_prompt.system_prompt；SFW 模板由内部门控跳过
                include_custom_system_prompt=True,
            )
            if not generated_prompt:
                await self.send_text("提示词生成失败，请稍后再试~")
                return False, "提示词生成失败", True

            is_selfie = detect_selfie_from_output(generated_prompt)
            selfie_base_prompt = generated_prompt
            if is_selfie:
                generated_prompt = self._process_selfie_prompt(
                    generated_prompt,
                    description,
                    include_selfie_prompt_add=True,
                    log_changes=True,
                )

            if self.get_config("prompt_generator.enforce_tag_order", False):
                generated_prompt = normalize_prompt_order(generated_prompt)

            generated_prompt = self._sanitize_prompt_for_sfw_mode(generated_prompt)

            if self._is_prompt_show_enabled():
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

            image_size = model_config.get("nai_size") or model_config.get("default_size", "1024x1280")
            enable_debug = bool(self.get_config("components.enable_debug_info", False))
            if enable_debug:
                await self.send_text("正在生成图片，请稍候...")

            success, result = await self.api_client.generate_image(
                prompt=generated_prompt,
                model_config=model_config,
                size=image_size,
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

            image_size = model_config.get("nai_size") or model_config.get("default_size", "1024x1280")
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
    _STRUCTURED_DESCRIPTION_FIELDS = (
        "subject_and_pov",
        "action",
        "emotion",
        "scene_delta",
        "framing",
    )

    def _compose_description_from_action_data(self) -> str:
        """把 Planner 拆分的 5 个结构化字段拼成单行 request 文本。

        - 5 字段全空时回落到旧版 ``description`` 字段，保持对 Planner 偶发只填 description 的兼容。
        - 拼接时各字段之间用空格连接，与原 description 关键词串风格保持一致。
        - 任何一个字段非空都视为"走结构化路径"，此时忽略 description（防止 Planner 同时填 6 个字段导致重复）。
        """
        structured_parts: list[str] = []
        for key in self._STRUCTURED_DESCRIPTION_FIELDS:
            value = str(self.action_data.get(key, "") or "").strip()
            if value:
                structured_parts.append(value)
        if structured_parts:
            return " ".join(structured_parts)
        return str(self.action_data.get("description", "") or "").strip()

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

        trigger_assessment = await self._assess_action_trigger(reasoning=self.reasoning)
        if self._is_action_guard_enabled() and not trigger_assessment["should_generate"]:
            logger.info(
                "%s Action 出图已拦截: category=%s detail=%s signal=%s text=%s",
                self.log_prefix,
                trigger_assessment["category"],
                trigger_assessment["detail"],
                trigger_assessment.get("signal_source", ""),
                trigger_assessment.get("signal_text", ""),
            )
            return False, trigger_assessment["detail"]

        # 主动出图自动 self-image 增强：bot 自己想发图时，让出来的图更像"她给你看一眼自己"
        # 而不是"画了一张陌生女孩"。explicit 路径不动，保持用户原意。
        if (
            trigger_assessment["category"] == "proactive"
            and bool(self.get_config("action_guard.proactive_self_image_boost", True))
            and description
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
        if generated_prompt:
            description = generated_prompt.strip()
        elif not description:
            await self.send_text("提示词生成器开小差了，请直接告诉我想画什么，或者稍后再试一次~")
            return False, "图片描述为空"

        is_selfie = detect_selfie_from_output(description)
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
            )

        if self.get_config("prompt_generator.enforce_tag_order", False):
            description = normalize_prompt_order(description)

        description = self._sanitize_prompt_for_sfw_mode(description)

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

        try:
            success, result = await self.api_client.generate_image(
                prompt=description,
                model_config=model_config,
                size=image_size,
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
        if self._is_action_guard_enabled() and not guard_state["should_generate"]:
            logger.info(
                "%s reply 自动跟图被拦截: detail=%s text=%s",
                self.log_prefix,
                guard_state["detail"],
                guard_state.get("signal_text", ""),
            )
            return False, guard_state["detail"]

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
        if generated_prompt:
            description = generated_prompt.strip()
        elif not description:
            return False, "图片描述为空"

        is_selfie = detect_selfie_from_output(description)
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
            )

        if self.get_config("prompt_generator.enforce_tag_order", False):
            description = normalize_prompt_order(description)

        description = self._sanitize_prompt_for_sfw_mode(description)

        model_config = self._get_model_config(is_selfie=is_selfie)
        if not model_config or not model_config.get("base_url"):
            return False, "模型配置无效"

        image_size = model_config.get("nai_size") or model_config.get("default_size", "")

        try:
            success, result = await self.api_client.generate_image(
                prompt=description,
                model_config=model_config,
                size=image_size,
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

    async def _assess_auto_draw_trigger(self) -> dict[str, Any]:
        """auto_draw 用的 guard：保留负向用户原话兜底 + auto_draw 档间隔。"""
        user_text, age_seconds = await self._fetch_last_user_text_with_age()
        if user_text:
            negative_strength = detect_negative_image_intent_strength(user_text)
            if negative_strength == "strong":
                return {
                    "should_generate": False,
                    "detail": "用户明确表示不需要图片",
                    "signal_text": user_text[:120],
                }
            if negative_strength == "weak":
                weak_ttl = max(
                    0,
                    int(self.get_config("action_guard.weak_negative_ttl_seconds", 60) or 60),
                )
                if age_seconds is None or age_seconds <= weak_ttl:
                    return {
                        "should_generate": False,
                        "detail": "用户刚才偏好文字回复",
                        "signal_text": user_text[:120],
                    }
        can_send, detail = self._check_action_image_interval("auto_draw")
        return {
            "should_generate": can_send,
            "detail": detail,
            "signal_text": (user_text or "")[:120],
        }

    def _is_action_guard_enabled(self) -> bool:
        """检查是否启用自动出图保护。"""
        return bool(self.get_config("action_guard.enabled", True))

    async def preflight_action_guard(self) -> dict[str, Any] | None:
        """Action Guard 同步预检：让 Planner 在 RPC 返回时就能拿到拦截原因。

        返回 None 表示 guard 未启用，调用方应放行；返回 dict 表示 guard 结论，
        ``should_generate`` 为 False 时 ``detail`` 给出可透传给 Planner 的失败原因。
        结果会缓存到 invocation 上，后台 ``handle_action`` 复用同一次评估，避免重复读消息库。
        """
        if not self._is_action_guard_enabled():
            return None
        return await self._assess_action_trigger(reasoning=self.reasoning)

    async def _assess_action_trigger(self, reasoning: str = "") -> dict[str, Any]:
        """Action Guard 评估入口；结果缓存供 handle_action 后台复用。"""
        if self._cached_action_trigger_assessment is not None:
            return self._cached_action_trigger_assessment
        result = await self._compute_action_trigger_assessment(reasoning=reasoning)
        self._cached_action_trigger_assessment = result
        return result

    async def _compute_action_trigger_assessment(self, reasoning: str = "") -> dict[str, Any]:
        """评估当前 Action 是否真的适合出图，并应用频率保护。

        设计原则：
        - 信任 Planner 的语义判断：Planner 调了 Action 即"它认为该发图"，Guard 不再做白名单二次拦截
        - Guard 只负责两件事：
          ① 否定意图黑名单兜底（用户明确说"不要画"也调用了，按 Planner 误判处理）
          ② 频率保护，按"用户原话强度"分级 explicit / proactive 两档
        - 判定输入必须是用户原话，不能是 action_data["description"]（那是 LLM 生成的关键词）
          原话取不到时回落 Planner reasoning：reasoning 含"用户/对方/他说/明确/要求"等显式信号视为 explicit
        - 否定关键词区分强弱：strong（"不要画"）永久阻断；weak（"用文字"）仅在新鲜（< 60s）且
          是最近一条消息时阻断，避免 stale 偏好一直冻结。
        """
        user_text, age_seconds = await self._fetch_last_user_text_with_age()
        signal_source = "user_text"
        signal_text = user_text

        if user_text:
            negative_strength = detect_negative_image_intent_strength(user_text)
            if negative_strength == "strong":
                return {
                    "should_generate": False,
                    "explicit_request": False,
                    "category": "blocked",
                    "detail": "用户明确表示不需要图片",
                    "signal_source": "user_text",
                    "signal_text": user_text[:120],
                }
            if negative_strength == "weak":
                weak_ttl = max(
                    0,
                    int(self.get_config("action_guard.weak_negative_ttl_seconds", 60) or 60),
                )
                # age 未知时保守按"未过期"处理，仍然阻断；明确过期才放行
                if age_seconds is None or age_seconds <= weak_ttl:
                    return {
                        "should_generate": False,
                        "explicit_request": False,
                        "category": "blocked",
                        "detail": "用户刚才偏好文字回复",
                        "signal_source": "user_text",
                        "signal_text": user_text[:120],
                    }

        if user_text and detect_explicit_image_request(user_text):
            is_explicit = True
        elif user_text:
            is_explicit = False
        else:
            # 拿不到用户原话时退化到 Planner reasoning：仅当 reasoning 出现明确指向用户请求的措辞才升级到 explicit
            reasoning_text = str(reasoning or "").strip()
            signal_source = "reasoning" if reasoning_text else "none"
            signal_text = reasoning_text
            is_explicit = bool(reasoning_text) and _reasoning_implies_explicit_request(reasoning_text)

        category = "explicit" if is_explicit else "proactive"
        can_send, detail = self._check_action_image_interval(category)
        return {
            "should_generate": can_send,
            "explicit_request": is_explicit,
            "category": category,
            "detail": detail,
            "signal_source": signal_source,
            "signal_text": signal_text[:120],
        }

    def _check_action_image_interval(self, category: str) -> tuple[bool, str]:
        """检查自动出图间隔，避免短时间连续刷图。

        分档：
        - explicit:  用户原话明确要求看图/画图/自拍/追图 → 短间隔（默认 45s）
        - proactive: bot 主动判断需要配图 → 中间隔（默认 240s）
        - auto_draw: reply 后置 hook 自动跟图 → 独立间隔（默认 180s），同时尊重
          action_image_sent_at 与 auto_draw_sent_at 中较新的那次出图
        """
        # 三档间隔走 ``get_config`` 的 default 兜底，不再用 ``or X`` 二次兜底——
        # 否则用户显式配 0（"完全不节流"）会被当成 falsy 顶替成默认值
        explicit_interval = max(
            0,
            int(self.get_config("action_guard.explicit_request_min_interval_seconds", 5)),
        )
        proactive_interval = max(
            0,
            int(self.get_config("action_guard.proactive_min_interval_seconds", 10)),
        )
        auto_draw_interval = max(
            0,
            int(self.get_config("auto_draw_on_reply.min_interval_seconds", 15)),
        )

        last_action_at = session_state.get_last_action_image_sent_at(self.stream_id)
        last_auto_draw_at = session_state.get_last_auto_draw_sent_at(self.stream_id)

        if category == "auto_draw":
            # 自动跟图：尊重所有最近出图时间，取最近的一次做间隔判定
            effective_last = max(
                (ts for ts in (last_action_at, last_auto_draw_at) if ts is not None),
                default=None,
            )
            required_interval = auto_draw_interval
        elif category == "explicit":
            effective_last = last_action_at
            required_interval = explicit_interval
        else:
            effective_last = last_action_at
            required_interval = proactive_interval

        if effective_last is None:
            return True, "首次出图"

        elapsed = max(0.0, time.time() - effective_last)
        if elapsed >= required_interval:
            return True, "触发条件满足"

        remaining_seconds = int(required_interval - elapsed + 0.999)
        logger.debug(
            "%s Action 出图节流命中: category=%s required=%ds remaining=%ds",
            self.log_prefix, category, required_interval, remaining_seconds,
        )
        # 给 Planner 的 detail 不含具体秒数、不出现"等待"字样：之前写成"还需等待约 X 秒"
        # 会被 Planner LLM 直接联想到主程序的 wait 工具，进而调 wait(seconds=120) 把整个
        # 对话循环锁死。这里改成只描述"本轮跳过 + 走文字"，并明确禁止 wait。
        return False, "图片节流中，本轮跳过出图、直接用文字回复推进；插件会自行解除冷却，请不要使用 wait 工具"

    async def handle_admin_command(self, action: str, param: str) -> tuple[bool, str | None, bool]:
        """处理 `/nai st|sp|set|art|size|help`。"""
        if not await self.ensure_user_not_blacklisted():
            return False, "黑名单用户", True

        platform, chat_id, user_id = self._get_chat_identity()
        if not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        if action == "help":
            help_text = """📖 NovelAI 图片生成插件命令帮助

【生图命令】
/nai <描述> - 使用自然语言生成图片
  示例：/nai 画一张初音未来
/nai 随机 - 随机生成一张 NSFW 图片
/nai 随机自拍 - 随机生成一张 NSFW 自拍图片
/nai0 <英文标签> - 直接使用英文标签生成图片
  示例：/nai0 1girl, hatsune miku, smile
💡 描述含「自拍/镜子/合照/手机拍」走自拍路径；
   含「肖像/生活照/立绘/portrait」走肖像路径；
   其它走普通画图。

【模型 / 尺寸 / 画师】（所有人可用，会话级，重启回退默认）
/nai set [代号] - 查看/切换模型
  代号：3=V3, f3=Furry V3, 4=V4, 4.5=V4.5 Full, 4.5p=V4.5 Preview
/nai size [代号] - 查看/切换尺寸
  代号：竖/v、横/h、方/s
/nai art [编号] - 查看/切换画师风格预设

【自动撤回】（仅管理员）
/nai on - 开启图片自动撤回
/nai off - 关闭图片自动撤回

【手动撤回】
/nai 撤回 - 撤回最近一张本插件发送的图片

【提示词显示】
/nai pt on/off - 开关提示词显示

【NSFW 过滤】（会话级）
/nai nsfw - 查看当前状态
/nai nsfw on/off - 开关 NSFW 过滤

【图片反推】
/nai 反推 - 回引图片或同消息发图，把图反推成 Danbooru tag
  · PNG 原图（NAI/SD 元数据）→ 秒级精确还原 prompt
  · 非原图 → 走 WD14 在线 Space 兜底（耗时 20~60s，只输出正向）

【管理员功能】
/nai st - 开启管理员模式
/nai sp - 关闭管理员模式
/nai ban <用户ID> - 拉黑指定用户
/nai unban <用户ID> - 取消拉黑指定用户
/nai banlist - 查看黑名单

【其它】
/nai help - 显示本帮助"""
            await self.send_text(help_text)
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
            "4": "nai-diffusion-4-full",
            "4.5": "nai-diffusion-4-5-full",
            "4.5p": "nai-diffusion-4-5-curated-anlas-0",
            "4.5-preview": "nai-diffusion-4-5-curated-anlas-0",
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
                    "4 - nai-diffusion-4-full\n"
                    "4.5 - nai-diffusion-4-5-full\n"
                    "4.5p - nai-diffusion-4-5-curated-anlas-0 (Preview)"
                )
                return True, "显示模型列表", True

            if param not in model_mappings:
                await self.send_text("❌ 无效的模型代号，可用值：3 / f3 / 4 / 4.5 / 4.5p")
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
