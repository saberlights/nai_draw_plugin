# -*- coding: utf-8 -*-
"""
/nai pt 命令：控制是否输出生成的提示词

所有状态通过 session_state 管理
"""
from typing import Tuple, Optional

from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger

from ..services.session_state import session_state

logger = get_logger("nai_draw_plugin")


class NaiPromptShowCommand(BaseCommand):
    """NovelAI 提示词显示控制命令"""

    command_name = "nai_prompt_show_command"
    command_description = "NAI提示词显示控制命令：/nai pt <on|off>"
    command_pattern = r"(?:.*，说：\s*)?/nai\s+pt\s+(?P<action>on|off)$"

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行提示词显示控制命令"""
        logger.info(f"{self.log_prefix} [提示词显示] /nai pt {self.matched_groups.get('action', '')}")

        action = self.matched_groups.get("action", "").strip()

        # 获取会话信息
        platform, chat_id, user_id, chat_type = self._get_session_info()
        if not platform or not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        # 权限检查：如果管理员模式开启，则需要管理员权限
        if session_state.is_admin_mode_enabled(platform, chat_id, self.get_config):
            if not session_state.is_admin_user(user_id, self.get_config):
                return False, "没有权限", True

        if action == "on":
            session_state.set_prompt_show_enabled(platform, chat_id, True)
            await self.send_text(f"✅ 已开启提示词显示")
            return True, "提示词显示已开启", True

        elif action == "off":
            session_state.set_prompt_show_enabled(platform, chat_id, False)
            await self.send_text(f"✅ 已关闭提示词显示")
            return True, "提示词显示已关闭", True

        else:
            await self.send_text("/nai pt on|off - 开启/关闭提示词显示")
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
        elif user_id:
            chat_id = user_id
            chat_type = "私聊"
        else:
            return platform, "", "", ""

        return platform, chat_id, user_id, chat_type

    # 兼容性方法
    @classmethod
    def is_prompt_show_enabled(cls, platform: str, chat_id: str, get_config_func) -> bool:
        """兼容方法：检查是否启用提示词显示"""
        return session_state.is_prompt_show_enabled(platform, chat_id, get_config_func)
