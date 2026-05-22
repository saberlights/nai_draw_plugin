# -*- coding: utf-8 -*-
"""
/nai nsfw 命令：NSFW内容过滤开关控制

所有状态通过 session_state 管理
"""
from typing import Tuple, Optional

from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger

from ..services.session_state import session_state

logger = get_logger("nai_draw_plugin")


class NaiNsfwControlCommand(BaseCommand):
    """NovelAI NSFW内容过滤控制命令"""

    command_name = "nai_nsfw_control_command"
    command_description = "NSFW内容过滤控制命令：/nai nsfw <on|off>"
    command_pattern = r"(?:.*，说：\s*)?/nai\s+nsfw(?:\s+(?P<action>on|off))?$"

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行NSFW过滤控制命令"""
        logger.info(f"{self.log_prefix} [NSFW过滤] /nai nsfw {self.matched_groups.get('action', '') or '查询'}")

        action = (self.matched_groups.get("action") or "").strip().lower()

        # 获取会话信息
        platform, chat_id, user_id, chat_type = self._get_session_info()
        if not platform or not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        # 权限检查：始终需要管理员权限
        if not session_state.is_admin_user(user_id, self.get_config):
            await self.send_text("❌ 只有管理员可以使用NSFW过滤控制命令", storage_message=False)
            return False, "没有管理员权限", True

        # 如果没有参数，显示当前状态
        if not action:
            current_state = session_state.is_nsfw_filter_enabled(platform, chat_id, self.get_config)
            state_text = "已开启" if current_state else "已关闭"
            await self.send_text(
                f"当前NSFW过滤状态: {state_text}\n\n"
                "使用方法:\n"
                "/nai nsfw on - 开启NSFW内容过滤（禁止生成NSFW）\n"
                "/nai nsfw off - 关闭NSFW内容过滤（允许生成NSFW）",
                storage_message=False
            )
            return True, "显示NSFW过滤状态", True

        if action == "on":
            session_state.set_nsfw_filter_enabled(platform, chat_id, True)
            await self.send_text(
                f"✅ 已在{chat_type}中开启NSFW内容过滤\n"
                f"🔒 生成的图片将避免包含成人内容\n"
                f"💡 使用 /nai nsfw off 可关闭过滤",
                storage_message=False
            )
            return True, "NSFW过滤已开启", True

        elif action == "off":
            session_state.set_nsfw_filter_enabled(platform, chat_id, False)
            await self.send_text(
                f"✅ 已在{chat_type}中关闭NSFW内容过滤\n"
                f"🔓 生成的图片将不受NSFW限制\n"
                f"💡 使用 /nai nsfw on 可重新开启",
                storage_message=False
            )
            return True, "NSFW过滤已关闭", True

        else:
            await self.send_text(
                "使用方法:\n"
                "/nai nsfw on - 开启NSFW内容过滤\n"
                "/nai nsfw off - 关闭NSFW内容过滤",
                storage_message=False
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
        elif user_id:
            chat_id = user_id
            chat_type = "私聊"
        else:
            return platform, "", "", ""

        return platform, chat_id, user_id, chat_type

    # 兼容性方法
    @classmethod
    def is_nsfw_filter_enabled(cls, platform: str, chat_id: str, get_config_func) -> bool:
        """兼容方法：检查是否启用NSFW过滤"""
        return session_state.is_nsfw_filter_enabled(platform, chat_id, get_config_func)
