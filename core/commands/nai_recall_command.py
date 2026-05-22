# -*- coding: utf-8 -*-
"""
/nai on|off 命令：自动撤回控制

所有状态通过 session_state 管理
"""
from typing import Tuple, Optional

from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger

from ..services.session_state import session_state

logger = get_logger("nai_draw_plugin")


class NaiRecallControlCommand(BaseCommand):
    """NovelAI 图片生成自动撤回控制命令"""

    command_name = "nai_recall_control_command"
    command_description = "NAI自动撤回控制命令：/nai <on|off>"
    command_pattern = r"(?:.*，说：\s*)?/nai\s+(?P<action>on|off)$"

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行自动撤回控制命令"""
        logger.info(f"{self.log_prefix} [自动撤回] /nai {self.matched_groups.get('action', '')}")

        action = self.matched_groups.get("action", "").strip()

        # 获取会话信息
        platform, chat_id, user_id, chat_type = self._get_session_info()
        if not platform or not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        # 权限检查：始终需要管理员权限
        if not session_state.is_admin_user(user_id, self.get_config):
            await self.send_text("❌ 只有管理员可以使用自动撤回控制命令", storage_message=False)
            return False, "没有管理员权限", True

        # 检查会话权限
        has_permission, permission_error = self._check_chat_permission(platform, chat_id)
        if not has_permission:
            await self.send_text(f"❌ {permission_error}")
            return False, permission_error, True

        if action == "on":
            session_state.set_recall_enabled(platform, chat_id, True)
            delay_seconds = self.get_config("auto_recall.delay_seconds", 5)
            await self.send_text(
                f"✅ 已在{chat_type}中开启NAI图片自动撤回功能\n"
                f"📝 图片将在发送后 {delay_seconds} 秒自动撤回\n"
                f"💡 使用 /nai off 可关闭此功能"
            )
            return True, "自动撤回已开启", True

        elif action == "off":
            session_state.set_recall_enabled(platform, chat_id, False)
            await self.send_text(
                f"✅ 已在{chat_type}中关闭NAI图片自动撤回功能\n"
                f"💡 使用 /nai on 可重新开启"
            )
            return True, "自动撤回已关闭", True

        else:
            await self.send_text(
                "使用方法：\n"
                "/nai on - 开启NAI图片自动撤回\n"
                "/nai off - 关闭NAI图片自动撤回"
            )
            return False, "无效的操作参数", True

    def _get_session_info(self) -> Tuple[str, str, str, str]:
        """获取会话信息"""
        if not self.message or not getattr(self.message, "message_info", None):
            return "", "", "", ""

        message_info = self.message.message_info
        platform = getattr(message_info, "platform", "") or ""
        group_info = getattr(message_info, "group_info", None)
        user_info = getattr(message_info, "user_info", None)

        if not user_info:
            return platform, "", "", ""

        user_id = str(getattr(user_info, "user_id", "") or "")

        if group_info and getattr(group_info, "group_id", None):
            chat_id = str(group_info.group_id)
            chat_type = "群聊"
        else:
            chat_id = str(getattr(user_info, "user_id", "") or "")
            chat_type = "私聊"

        return platform, chat_id, user_id, chat_type

    def _check_chat_permission(self, platform: str, chat_id: str) -> Tuple[bool, Optional[str]]:
        """检查会话是否有自动撤回权限"""
        current_chat_key = f"{platform}:{chat_id}"

        allowed_groups = self.get_config("auto_recall.allowed_groups", [])
        if not allowed_groups:
            logger.info(f"{self.log_prefix} [自动撤回] 未配置白名单，允许所有会话")
            return True, None

        if current_chat_key in allowed_groups:
            return True, None

        return False, "当前会话没有使用自动撤回功能的权限"

    # 兼容性方法
    @classmethod
    def is_recall_enabled(cls, platform: str, chat_id: str, get_config_func) -> bool:
        """兼容方法：检查是否启用自动撤回"""
        return session_state.is_recall_enabled(platform, chat_id, get_config_func)
