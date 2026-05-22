import asyncio
import time
from typing import Any, Dict, Optional

from src.chat.utils.utils import parse_platform_accounts
from src.common.logger import get_logger
from src.config.config import global_config

from ..constants import NAI_PIC_IMAGE_DISPLAY_MARKER
from ..utils.display_message_helper import is_nai_action_image_display_message

logger = get_logger("nai_draw_plugin")


def _get_bot_account_for_platform(platform: str) -> str:
    """根据平台获取机器人自身账号"""
    platform_key = (platform or "").strip().lower()
    bot_config = getattr(global_config, "bot", None)
    if not bot_config:
        return ""

    account_map_raw = parse_platform_accounts(getattr(bot_config, "platforms", []) or [])
    account_map = {
        k.strip().lower(): str(v).strip()
        for k, v in account_map_raw.items()
        if v
    }

    qq_account = str(getattr(bot_config, "qq_account", "") or "").strip()
    if qq_account:
        account_map.setdefault("qq", qq_account)

    telegram_account = str(getattr(bot_config, "telegram_account", "") or "").strip()
    if telegram_account:
        account_map.setdefault("telegram", telegram_account)
        account_map.setdefault("tg", telegram_account)

    if platform_key in account_map:
        return account_map[platform_key]

    return qq_account


def _extract_message_field(msg: Any, field: str):
    """兼容 DatabaseMessages 与 dict 的字段访问"""
    if isinstance(msg, dict):
        return msg.get(field)
    return getattr(msg, field, None)


def _extract_sender_user_id(msg: Any) -> str:
    """获取消息发送者ID（兼容 DatabaseMessages 与 dict）"""
    try:
        if isinstance(msg, dict):
            direct = msg.get("user_id")
            if direct:
                return str(direct)

            user_info = msg.get("user_info")
            if isinstance(user_info, dict):
                user_id = user_info.get("user_id")
                if user_id:
                    return str(user_id)

            message_info = msg.get("message_info")
            if isinstance(message_info, dict):
                mi_user_info = message_info.get("user_info")
                if isinstance(mi_user_info, dict):
                    user_id = mi_user_info.get("user_id")
                    if user_id:
                        return str(user_id)

            return ""

        user_info_obj = getattr(msg, "user_info", None)
        user_id_obj = getattr(user_info_obj, "user_id", None) if user_info_obj else None
        if user_id_obj:
            return str(user_id_obj)

        legacy = getattr(msg, "user_id", None)
        if legacy:
            return str(legacy)

        return ""
    except Exception:
        return ""


def _text_looks_like_image(text: str) -> bool:
    """判断文本是否像“图片消息本体”（避免把引用/回复内容误判为图片）"""
    if not isinstance(text, str):
        return False
    normalized = text.strip()
    if not normalized:
        return False
    return normalized.startswith(("[图片", "[NAI图片", "[image", "[imageurl", "[picid", "picid:"))


def _is_image_message(msg: Any) -> bool:
    """判断消息是否为bot发送的图片"""
    try:
        if isinstance(msg, dict):
            if msg.get("is_picid"):
                return True
            seg = msg.get("message_segment")
            if isinstance(seg, dict):
                seg_type = seg.get("type")
                if seg_type in {"image", "imageurl"}:
                    return True
                if seg_type == "seglist":
                    for child in seg.get("data") or []:
                        if isinstance(child, dict) and child.get("type") in {"image", "imageurl"}:
                            return True
            for key in ("processed_plain_text", "display_message", "raw_message"):
                text_val = msg.get(key)
                if isinstance(text_val, str) and _text_looks_like_image(text_val):
                    return True
            return False

        if getattr(msg, "is_picid", False):
            return True
        text_candidates = [
            getattr(msg, "processed_plain_text", None),
            getattr(msg, "display_message", None),
            getattr(msg, "raw_message", None),
        ]
        for text in text_candidates:
            if isinstance(text, str) and _text_looks_like_image(text):
                return True
        return False
    except Exception:
        return False


def _is_nai_draw_plugin_image_message(msg: Any) -> bool:
    """
    判断消息是否为“本插件发送的图片消息”。

    规则：
    - 必须是图片消息（image/imageurl/picid 等）
    - 且 display_message 中包含本插件旧标记，或使用 action 路径的可读前缀
    """
    try:
        display_message = _extract_message_field(msg, "display_message")
        if not isinstance(display_message, str):
            return False
        if (
            NAI_PIC_IMAGE_DISPLAY_MARKER not in display_message
            and not is_nai_action_image_display_message(display_message)
        ):
            return False
        return _is_image_message(msg)
    except Exception:
        return False


class AutoRecallMixin:
    """提供自动撤回相关的通用方法"""

    def _get_recall_context(self) -> Dict[str, Any]:
        """获取自动撤回所需的上下文信息"""
        # Command组件
        message = getattr(self, "message", None)
        if message and getattr(message, "message_info", None):
            message_info = message.message_info
            return {
                "platform": getattr(message_info, "platform", "") or "",
                "group_info": getattr(message_info, "group_info", None),
                "user_info": getattr(message_info, "user_info", None),
                "chat_stream": getattr(message, "chat_stream", None),
            }

        # Action组件
        action_message = getattr(self, "action_message", None)
        if action_message:
            chat_info = getattr(action_message, "chat_info", None)
            platform = getattr(chat_info, "platform", None) if chat_info else getattr(self, "platform", None)
            group_info = None
            user_info = None

            if chat_info:
                group_info = getattr(chat_info, "group_info", None)
                user_info = getattr(chat_info, "user_info", None)

            if group_info is None:
                group_info = getattr(action_message, "group_info", None)
            if user_info is None:
                user_info = getattr(action_message, "user_info", None)

            return {
                "platform": platform or "",
                "group_info": group_info,
                "user_info": user_info,
                "chat_stream": getattr(self, "chat_stream", None),
            }

        # 兜底
        return {
            "platform": getattr(self, "platform", "") or "",
            "group_info": None,
            "user_info": None,
            "chat_stream": getattr(self, "chat_stream", None),
        }

    async def _schedule_auto_recall(self, placeholder_message_id: Optional[str] = None):
        """计划自动撤回任务"""
        try:
            context = self._get_recall_context()
            platform = context.get("platform") or ""
            group_info = context.get("group_info")
            user_info = context.get("user_info")

            if group_info and getattr(group_info, "group_id", None):
                chat_id = str(getattr(group_info, "group_id"))
            elif user_info and getattr(user_info, "user_id", None):
                chat_id = str(getattr(user_info, "user_id"))
            else:
                logger.debug(f"{self.log_prefix} 无法识别聊天类型，跳过自动撤回")
                return

            if not self._is_auto_recall_enabled(platform, chat_id):
                logger.debug(f"{self.log_prefix} 会话未启用自动撤回")
                return

            delay_seconds = self.get_config("auto_recall.delay_seconds", 5)
            id_wait_seconds = max(0, self.get_config("auto_recall.id_wait_seconds", 15))
            poll_interval = min(1.0, max(0.2, id_wait_seconds / 10)) if id_wait_seconds else 0
            target_send_timestamp = getattr(self, "_last_send_timestamp", None)

            await asyncio.sleep(0.2)

            # 自动撤回通常发生在发送后很短时间内，窗口不用太大（避免大群高频下查询过重）
            message_id = await self._get_last_message_id(
                require_marker=True,
                hours=0.5,
                limit=80,
                target_send_timestamp=target_send_timestamp,
            )
            if not message_id:
                logger.warning(f"{self.log_prefix} 未能获取消息ID，无法自动撤回")
                return

            logger.info(f"{self.log_prefix} 计划在 {delay_seconds} 秒后撤回消息: {message_id}")
            initial_message_id = message_id

            async def _resolve_message_id(initial_id: Optional[str]) -> Optional[str]:
                candidate = initial_id
                if not candidate:
                    return None
                if not candidate.startswith("send_api_"):
                    return candidate
                if id_wait_seconds <= 0:
                    return candidate
                deadline = time.monotonic() + id_wait_seconds
                while time.monotonic() < deadline:
                    refreshed_id = await self._get_last_message_id(
                        require_marker=True,
                        hours=0.5,
                        limit=80,
                        target_send_timestamp=target_send_timestamp,
                    )
                    if refreshed_id and not refreshed_id.startswith("send_api_"):
                        logger.debug(f"{self.log_prefix} 占位ID替换为正式ID: {refreshed_id}")
                        return refreshed_id
                    await asyncio.sleep(poll_interval or 0.5)
                logger.debug(f"{self.log_prefix} 在限定时间内未获取正式ID，继续使用占位ID")
                return candidate

            async def _delayed_recall():
                await asyncio.sleep(delay_seconds)
                target_message_id = await _resolve_message_id(initial_message_id)
                if not target_message_id:
                    logger.warning(f"{self.log_prefix} 撤回失败：缺少消息ID")
                    return
                try:
                    success = await self._try_recall_message(target_message_id)
                    if success:
                        logger.info(f"{self.log_prefix} 消息 {target_message_id} 已成功撤回")
                    else:
                        logger.warning(f"{self.log_prefix} 消息 {target_message_id} 撤回失败")
                except Exception as exc:
                    logger.error(f"{self.log_prefix} 撤回消息时出错: {exc!r}")

            task = asyncio.create_task(_delayed_recall())
            if hasattr(self, "plugin") and hasattr(self.plugin, "_track_task"):
                self.plugin._track_task(task)
        except Exception as exc:
            logger.error(f"{self.log_prefix} 计划自动撤回失败: {exc!r}")

    def _is_auto_recall_enabled(self, platform: str, chat_id: str) -> bool:
        """由子类实现，用于判断当前会话是否启用了自动撤回"""
        raise NotImplementedError

    async def _get_last_message_id(
        self,
        hours: float = 24.0,
        limit: int = 120,
        require_marker: bool = False,
        target_send_timestamp: Optional[float] = None,
        exclude_message_ids: Optional[set[str]] = None,
    ) -> Optional[str]:
        """获取最后发送的消息ID（可选：仅限本插件图片）"""
        resolved_id, placeholder_id, _ = await self._get_last_message_candidate(
            hours=hours,
            limit=limit,
            require_marker=require_marker,
            target_send_timestamp=target_send_timestamp,
            exclude_message_ids=exclude_message_ids,
        )
        if resolved_id:
            return resolved_id
        return placeholder_id

    async def _get_last_message_candidate(
        self,
        hours: float = 24.0,
        limit: int = 120,
        require_marker: bool = False,
        target_send_timestamp: Optional[float] = None,
        exclude_message_ids: Optional[set[str]] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[float]]:
        """获取最后发送消息的候选结果，同时返回候选时间戳。"""
        try:
            logger.info(f"{self.log_prefix} 开始获取消息ID")

            context = self._get_recall_context()
            chat_stream = context.get("chat_stream")
            # 兼容 BotChatSession（session_id）和其他可能的 chat_stream 类型（stream_id）
            stream_id = (
                getattr(chat_stream, "stream_id", None)
                or getattr(chat_stream, "session_id", None)
            ) if chat_stream else None
            if not stream_id:
                logger.info(f"{self.log_prefix} 无法获取stream_id")
                return None, None, None

            platform = context.get("platform", "") or ""
            bot_account = _get_bot_account_for_platform(platform)
            send_timestamp = (
                target_send_timestamp
                if target_send_timestamp is not None
                else getattr(self, "_last_send_timestamp", None)
            )
            timestamp_tolerance = 0.2

            from src.plugin_system import message_api

            max_attempts = 5

            for attempt in range(max_attempts):
                msgs = message_api.get_recent_messages(
                    chat_id=str(stream_id),
                    hours=hours,
                    limit=limit,
                    limit_mode="latest",
                    filter_mai=False
                ) or []
                logger.debug(f"{self.log_prefix} 尝试{attempt + 1}/{max_attempts}，获取到 {len(msgs)} 条消息")

                resolved_id, placeholder_id, candidate_time = self._select_best_message_candidate(
                    msgs=msgs,
                    require_marker=require_marker,
                    bot_account=bot_account,
                    send_timestamp=send_timestamp,
                    timestamp_tolerance=timestamp_tolerance,
                    exclude_message_ids=exclude_message_ids,
                )
                if resolved_id:
                    logger.info(f"{self.log_prefix} 命中消息ID: {resolved_id}")
                    return resolved_id, None, candidate_time

                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.4)

            if placeholder_id:
                logger.warning(f"{self.log_prefix} 未获取到正式ID，使用占位ID: {placeholder_id}")
                return None, placeholder_id, candidate_time

            logger.warning(f"{self.log_prefix} 所有方法都未能获取消息ID")
            return None, None, None
        except Exception as exc:
            logger.error(f"{self.log_prefix} 获取消息ID失败: {exc!r}")
            return None, None, None

    def _select_best_message_id(
        self,
        msgs: list,
        require_marker: bool,
        bot_account: str,
        send_timestamp: Optional[float],
        timestamp_tolerance: float,
        exclude_message_ids: Optional[set[str]] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """从候选消息中挑选最适合本次撤回的消息ID。"""
        resolved_id, placeholder_id, _ = self._select_best_message_candidate(
            msgs=msgs,
            require_marker=require_marker,
            bot_account=bot_account,
            send_timestamp=send_timestamp,
            timestamp_tolerance=timestamp_tolerance,
            exclude_message_ids=exclude_message_ids,
        )
        return resolved_id, placeholder_id

    def _select_best_message_candidate(
        self,
        msgs: list,
        require_marker: bool,
        bot_account: str,
        send_timestamp: Optional[float],
        timestamp_tolerance: float,
        exclude_message_ids: Optional[set[str]] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[float]]:
        """从候选消息中挑选最适合本次撤回的消息ID，并返回候选时间戳。"""
        excluded_ids = {str(mid) for mid in (exclude_message_ids or set()) if mid}
        best_formal: Optional[tuple[tuple[float, float, float, float], str, Optional[float]]] = None
        best_placeholder: Optional[tuple[tuple[float, float, float, float], str, Optional[float]]] = None

        def _remember_candidate(
            message_id: str,
            msg_time_val: Optional[float],
            sort_key: tuple[float, float, float, float],
        ) -> None:
            nonlocal best_formal, best_placeholder
            candidate = (sort_key, message_id, msg_time_val)
            if message_id.startswith("send_api_"):
                if best_placeholder is None or sort_key > best_placeholder[0]:
                    best_placeholder = candidate
                return

            if best_formal is None or sort_key > best_formal[0]:
                best_formal = candidate

        for index, msg in enumerate(msgs):
            msg_is_plugin_image = _is_nai_draw_plugin_image_message(msg)
            msg_is_image = _is_image_message(msg)

            if require_marker:
                if not msg_is_plugin_image and not msg_is_image:
                    continue
            else:
                if not msg_is_image:
                    continue

            message_id = _extract_message_field(msg, "message_id")
            if not message_id:
                continue
            message_id = str(message_id)
            if message_id in excluded_ids:
                continue

            msg_user_id = _extract_sender_user_id(msg)
            if not require_marker and bot_account:
                if not msg_user_id or msg_user_id != bot_account:
                    continue

            if require_marker and not msg_is_plugin_image:
                if not bot_account or not msg_is_image or msg_user_id != bot_account:
                    continue

            msg_time = _extract_message_field(msg, "time")
            try:
                msg_time_val = float(msg_time) if msg_time is not None else None
            except (TypeError, ValueError):
                msg_time_val = None

            if send_timestamp is None:
                sort_key = (
                    msg_time_val if msg_time_val is not None else float("-inf"),
                    1.0 if not message_id.startswith("send_api_") else 0.0,
                    1.0 if msg_is_plugin_image else 0.0,
                    float(index),
                )
                _remember_candidate(message_id, msg_time_val, sort_key)
                continue

            if msg_time_val is not None and msg_time_val + timestamp_tolerance < send_timestamp:
                continue

            if require_marker and not msg_is_plugin_image:
                # 某些平台 echo 回写后的正式消息不会保留 display_message，
                # 这里允许在“同 bot + 同时间附近 + 图片消息”条件下继续命中正确目标。
                if msg_time_val is not None and abs(msg_time_val - send_timestamp) > 8.0:
                    continue

            sort_key = (
                1.0 if not message_id.startswith("send_api_") else 0.0,
                -abs(msg_time_val - send_timestamp) if msg_time_val is not None else float("-inf"),
                1.0 if msg_is_plugin_image else 0.0,
                msg_time_val if msg_time_val is not None else float("-inf"),
            )
            _remember_candidate(message_id, msg_time_val, sort_key)

        if send_timestamp is None and best_placeholder:
            if best_formal is None or best_placeholder[0] > best_formal[0]:
                return None, best_placeholder[1], best_placeholder[2]

        if best_formal:
            return best_formal[1], None, best_formal[2]

        if best_placeholder:
            return None, best_placeholder[1], best_placeholder[2]

        return None, None, None

    async def _try_recall_message(self, message_id: str) -> bool:
        """尝试撤回消息"""
        try:
            delete_commands = ["delete_msg", "DELETE_MSG", "recall_msg", "RECALL_MSG"]
            for cmd in delete_commands:
                try:
                    result = await self.send_command(
                        cmd,
                        {"message_id": str(message_id)},
                        display_message="",
                        storage_message=False
                    )
                    if isinstance(result, bool) and result:
                        return True
                    if isinstance(result, dict):
                        status = str(result.get("status", "")).lower()
                        if status in ("ok", "success") or result.get("retcode") == 0:
                            return True
                except Exception as exc:
                    logger.debug(f"{self.log_prefix} 尝试命令 {cmd} 失败: {exc!r}")
                    continue
            return False
        except Exception as exc:
            logger.error(f"{self.log_prefix} 撤回消息异常: {exc!r}")
            return False
