# -*- coding: utf-8 -*-
"""
/nai 撤回 命令：手动撤回 bot 发送的图片。

- 仅支持直接发送 /nai 撤回
- 每次按顺序撤回 bot 最近发送的一张图片
"""
import asyncio
import time
from typing import Optional, Tuple, Dict

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand

from ..mixins.auto_recall_mixin import AutoRecallMixin

logger = get_logger("nai_draw_plugin")


class NaiManualRecallCommand(BaseCommand, AutoRecallMixin):
    """发送 /nai 撤回，手动撤回 bot 发送的图片"""

    command_name = "nai_manual_recall_command"
    command_description = "手动撤回图片：/nai 撤回"
    command_pattern = r"(?:.*?)(?:/nai\s+撤回)(?:\s+.*)?$"
    _recent_manual_recall_ids: Dict[str, Dict[str, float]] = {}
    _RECENT_RECALL_TTL_SECONDS = 600

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        logger.info(f"{self.log_prefix} [手动撤回] 收到 /nai 撤回")

        try:
            resolved_id, placeholder_id, candidate_time = await self._get_last_message_candidate(
                require_marker=True,
                hours=24.0,
                limit=300,
                exclude_message_ids=self._get_recent_manual_recall_ids(),
            )
        except Exception as exc:
            logger.warning(f"{self.log_prefix} [手动撤回] 获取消息候选失败: {exc!r}")
            resolved_id, placeholder_id, candidate_time = None, None, None

        if resolved_id:
            return await self._do_recall(resolved_id, "最近图片")

        if placeholder_id:
            try:
                target_id = await self._resolve_latest_message_id(placeholder_id, target_send_timestamp=candidate_time)
            except Exception as exc:
                logger.warning(f"{self.log_prefix} [手动撤回] 解析消息ID失败: {exc!r}")
                target_id = placeholder_id
            return await self._do_recall(target_id, "最近图片")

        await self.send_text(
            "❌ 找不到可撤回的图片（直接发送 /nai 撤回 即可按顺序撤回最近一张）",
            storage_message=False,
        )
        return False, "找不到可撤回的消息", True

    async def _do_recall(self, message_id: str, source: str) -> Tuple[bool, Optional[str], bool]:
        """执行撤回并返回结果"""
        success = await self._try_recall_message(message_id)
        if success:
            self._remember_recent_manual_recall_id(message_id)
            logger.info(f"{self.log_prefix} [手动撤回] 消息 {message_id} 已撤回（{source}）")
            await self.send_text("✅ 已撤回", storage_message=False)
            return True, "手动撤回成功", True
        else:
            logger.warning(f"{self.log_prefix} [手动撤回] 消息 {message_id} 撤回失败（{source}）")
            await self.send_text(
                "❌ 撤回失败（可能消息已被删除、超过撤回时限、或 bot 无权撤回）",
                storage_message=False,
            )
            return False, "撤回失败", True

    def _get_recent_manual_recall_ids(self) -> set[str]:
        """获取当前会话近期已经尝试手动撤回过的消息ID，用于连续撤回时跳过旧目标。"""
        stream_id = str(getattr(getattr(self.message, "chat_stream", None), "stream_id", "") or "")
        if not stream_id:
            return set()

        now = time.monotonic()
        recent_map = self._recent_manual_recall_ids.get(stream_id, {})
        filtered = {
            message_id: ts
            for message_id, ts in recent_map.items()
            if now - ts <= self._RECENT_RECALL_TTL_SECONDS
        }
        if filtered:
            self._recent_manual_recall_ids[stream_id] = filtered
        else:
            self._recent_manual_recall_ids.pop(stream_id, None)
        return set(filtered.keys())

    def _remember_recent_manual_recall_id(self, message_id: str) -> None:
        """记录当前会话刚刚尝试撤回过的消息ID，避免下一次还命中同一条。"""
        stream_id = str(getattr(getattr(self.message, "chat_stream", None), "stream_id", "") or "")
        target_id = (message_id or "").strip()
        if not stream_id or not target_id:
            return

        recent_ids = self._recent_manual_recall_ids.setdefault(stream_id, {})
        recent_ids[target_id] = time.monotonic()

    async def _resolve_latest_message_id(self, message_id: str, target_send_timestamp: Optional[float] = None) -> str:
        """
        处理“撤回最近图片”场景下的临时 message_id（send_api_*）问题。

        message_id 可能是 send_api_*（发送时的临时ID），真实平台ID会通过 echo 异步回写。
        这里在限定时间内围绕目标时间轮询，等待其变为正式ID，避免回退命中更早的图片。
        """
        candidate = (message_id or "").strip()
        if not candidate.startswith("send_api_"):
            return candidate

        id_wait_seconds = max(0, self.get_config("auto_recall.id_wait_seconds", 15))
        if id_wait_seconds <= 0:
            return candidate

        poll_interval = min(1.0, max(0.2, id_wait_seconds / 10))
        deadline = time.monotonic() + id_wait_seconds
        while time.monotonic() < deadline:
            refreshed = await self._get_last_message_id(
                require_marker=True,
                hours=24.0,
                limit=300,
                target_send_timestamp=target_send_timestamp,
                exclude_message_ids=self._get_recent_manual_recall_ids(),
            )
            if refreshed and not refreshed.startswith("send_api_"):
                return refreshed
            await asyncio.sleep(poll_interval)
        return candidate

    # ---- AutoRecallMixin 要求的抽象方法 ----

    def _is_auto_recall_enabled(self, platform: str, chat_id: str) -> bool:
        return False
