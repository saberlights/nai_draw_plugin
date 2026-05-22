# -*- coding: utf-8 -*-
"""
NAI 管理员权限控制命令

处理以下命令：
- /nai st      : 开启管理员模式
- /nai sp      : 关闭管理员模式
- /nai set     : 模型切换
- /nai art     : 画师串切换
- /nai size    : 尺寸切换
- /nai help    : 帮助信息

所有状态通过 session_state 管理
"""
from typing import Tuple, Optional, List, Dict

from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger

from ..services.session_state import session_state

logger = get_logger("nai_draw_plugin")


class NaiAdminControlCommand(BaseCommand):
    """NAI 管理员模式控制命令"""

    # 模型映射表
    MODEL_MAPPINGS = {
        "3": "nai-diffusion-3",
        "f3": "nai-diffusion-3-furry",
        "4": "nai-diffusion-4-full",
        "4.5": "nai-diffusion-4-5-full",
        "4.5p": "nai-diffusion-4-5-curated-anlas-0",
        "4.5-preview": "nai-diffusion-4-5-curated-anlas-0",
    }

    # 尺寸映射表
    SIZE_MAPPINGS = {
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

    # Command 基本信息
    command_name = "nai_admin_control_command"
    command_description = "NAI管理员模式控制命令：/nai <st|sp|set|art|size|help>"
    command_pattern = r"(?:.*，说：\s*)?/nai\s+(?P<action>st|sp|set|art|size|help)(?:\s+(?P<param>.+))?$"

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行管理员模式控制命令"""
        logger.info(f"{self.log_prefix} [管理控制] /nai {self.matched_groups.get('action', '')}")

        action = self.matched_groups.get("action", "").strip()
        param = self.matched_groups.get("param", "").strip() if self.matched_groups.get("param") else ""

        # 获取会话信息
        platform, chat_id, user_id, chat_type = self._get_session_info()
        if not platform or not chat_id:
            await self.send_text("❌ 无法获取会话信息", storage_message=False)
            return False, "无法获取会话信息", True

        # help 命令对所有人开放
        if action == "help":
            return await self._handle_help()

        # 检查管理员权限
        is_admin = session_state.is_admin_user(user_id, self.get_config)

        # st/sp/set 操作始终需要管理员权限
        if action in ["st", "sp", "set"]:
            if not is_admin:
                if action == "set":
                    await self.send_text("❌ 只有管理员可以切换生图模型", storage_message=False)
                    return False, "没有管理员权限", True
                await self.send_text("❌ 只有管理员可以开启/关闭管理员模式", storage_message=False)
                return False, "没有管理员权限", True

        # art/size 操作根据管理员模式状态判断
        elif action in ["art", "size"]:
            if session_state.is_admin_mode_enabled(platform, chat_id, self.get_config):
                if not is_admin:
                    return False, "没有权限", True

        # 执行具体操作
        if action == "st":
            session_state.set_admin_mode(platform, chat_id, True)
            await self.send_text(
                f"✅ 已在{chat_type}中开启NAI管理员模式\n"
                f"🔒 现在所有NAI命令仅管理员可使用\n"
                f"💡 使用 /nai sp 可关闭此模式"
            )
            return True, "管理员模式已开启", True

        if action == "sp":
            session_state.set_admin_mode(platform, chat_id, False)
            await self.send_text(
                f"✅ 已在{chat_type}中关闭NAI管理员模式\n"
                f"🔓 现在所有人都可使用NAI命令\n"
                f"💡 使用 /nai st 可重新开启"
            )
            return True, "管理员模式已关闭", True

        if action == "set":
            return await self._handle_set_model(platform, chat_id, param)

        if action == "art":
            return await self._handle_set_artist(platform, chat_id, param)

        if action == "size":
            return await self._handle_set_size(platform, chat_id, param)

        # 未知操作
        await self.send_text(
            "使用方法：\n"
            "/nai st - 开启管理员模式\n"
            "/nai sp - 关闭管理员模式\n"
            "/nai set <模型> - 切换生图模型\n"
            "/nai art <编号> - 切换画师风格预设\n"
            "/nai size <尺寸> - 切换图片尺寸\n"
            "/nai help - 查看所有命令帮助"
        )
        return False, "无效的操作参数", True

    def _get_session_info(self) -> Tuple[str, str, str, str]:
        """
        获取会话信息

        Returns:
            (platform, chat_id, user_id, chat_type) 元组
        """
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

    async def _handle_help(self) -> Tuple[bool, Optional[str], bool]:
        """处理帮助命令"""
        help_text = """📖 NovelAI 图片生成插件命令帮助

【生图命令】
/nai <描述> - 使用自然语言生成图片
  示例：/nai 画一张初音未来
/nai 随机 - 随机生成一张NSFW图片
/nai 随机自拍 - 随机生成一张NSFW自拍图片
/nai0 <英文标签> - 直接使用英文标签生成图片
  示例：/nai0 1girl, hatsune miku, smile

【模型管理】（仅管理员可用）
/nai set - 查看当前模型和可用模型列表
/nai set <代号> - 切换生图模型
  可用模型：3=V3, f3=Furry V3, 4=V4, 4.5=V4.5 Full, 4.5p=V4.5 Preview
  示例：/nai set 4.5p

【画师风格预设】
/nai art - 查看当前画师串列表
/nai art <编号> - 切换画师风格预设
  示例：/nai art 2

【图片尺寸】
/nai size - 查看当前尺寸
/nai size <尺寸> - 切换图片尺寸
  可用尺寸：竖/v=竖图(832x1216), 横/h=横图(1216x832), 方/s=方图(1024x1024)
  示例：/nai size 横

【自动撤回】
/nai on - 开启图片自动撤回功能（仅管理员可用）
/nai off - 关闭图片自动撤回功能（仅管理员可用）

【手动撤回】
/nai 撤回 - 按顺序撤回本插件最近发送的一张图片（所有人可用）
  连续发送可依次从最新一张往前撤回

【提示词显示】
/nai pt on - 开启提示词显示
/nai pt off - 关闭提示词显示

【NSFW过滤】
/nai nsfw - 查看当前NSFW过滤状态
/nai nsfw on - 开启NSFW过滤
/nai nsfw off - 关闭NSFW过滤

【管理员功能】（仅管理员可用）
/nai st - 开启管理员模式
/nai sp - 关闭管理员模式

【其他】
/nai help - 显示此帮助信息

💡 提示：管理员模式开启后，所有命令仅管理员可用"""

        await self.send_text(help_text)
        return True, "显示帮助信息", True

    async def _handle_set_model(self, platform: str, chat_id: str, model_key: str) -> Tuple[bool, Optional[str], bool]:
        """处理模型切换命令"""
        if not model_key:
            current_model = session_state.get_selected_model(platform, chat_id)
            if current_model:
                current_display = f"当前模型: {current_model}"
            else:
                default_model = self.get_config("model.default_model", "nai-diffusion-4-5-full")
                current_display = f"当前使用默认模型: {default_model}"

            await self.send_text(
                f"{current_display}\n\n"
                "可用模型:\n"
                "3 - nai-diffusion-3\n"
                "f3 - nai-diffusion-3-furry\n"
                "4 - nai-diffusion-4-full\n"
                "4.5 - nai-diffusion-4-5-full\n"
                "4.5p - nai-diffusion-4-5-curated-anlas-0 (Preview)\n\n"
                "使用方法: /nai set <模型代号>"
            )
            return True, "显示模型列表", True

        if model_key not in self.MODEL_MAPPINGS:
            await self.send_text(
                f"❌ 无效的模型代号: {model_key}\n\n"
                "可用模型:\n"
                "3 - nai-diffusion-3\n"
                "f3 - nai-diffusion-3-furry\n"
                "4 - nai-diffusion-4-full\n"
                "4.5 - nai-diffusion-4-5-full\n"
                "4.5p - nai-diffusion-4-5-curated-anlas-0 (Preview)"
            )
            return False, "无效的模型代号", True

        model_name = self.MODEL_MAPPINGS[model_key]
        session_state.set_selected_model(platform, chat_id, model_name)

        await self.send_text(
            f"✅ 已切换到模型: {model_name}\n"
            f"代号: {model_key}"
        )
        return True, f"已切换到模型 {model_name}", True

    async def _handle_set_artist(self, platform: str, chat_id: str, preset_index: str) -> Tuple[bool, Optional[str], bool]:
        """处理画师串切换命令"""
        # 获取当前模型
        current_model = session_state.get_selected_model(platform, chat_id)
        if not current_model:
            current_model = self.get_config("model.default_model", "nai-diffusion-4-5-full")

        # 根据模型确定配置节
        config_section, model_display = self._get_model_config_section(current_model)
        if not config_section:
            await self.send_text("❌ 当前模型不支持画师串切换")
            return False, "模型不支持画师串", True

        # 获取画师串列表
        artist_presets_raw = self.get_config(f"{config_section}.artist_presets", [])
        if not artist_presets_raw:
            await self.send_text(f"❌ {model_display} 模型未配置画师串预设")
            return False, "未配置画师串", True

        artist_presets = session_state._parse_artist_presets(artist_presets_raw)

        # 显示列表
        if not preset_index:
            current_index = session_state.get_effective_artist_index(
                platform,
                chat_id,
                current_model,
                self.get_config,
            )
            preset_list = "\n".join([
                f"{'→ ' if i == current_index else '  '}{i}. {preset['name']}"
                for i, preset in enumerate(artist_presets, 1)
            ])

            await self.send_text(
                f"当前模型: {model_display}\n"
                f"当前画师串: #{current_index} - {artist_presets[current_index - 1]['name']}\n\n"
                f"可用画师串:\n{preset_list}\n\n"
                f"使用方法: /nai art <编号>"
            )
            return True, "显示画师串列表", True

        # 验证索引
        try:
            index = int(preset_index)
            if index < 1 or index > len(artist_presets):
                await self.send_text(
                    f"❌ 无效的画师串编号: {index}\n"
                    f"可用范围: 1-{len(artist_presets)}"
                )
                return False, "无效的画师串编号", True
        except ValueError:
            await self.send_text("❌ 画师串编号必须是数字")
            return False, "无效的画师串编号", True

        # 设置画师串
        session_state.set_selected_artist_index(platform, chat_id, index)
        selected_preset = artist_presets[index - 1]

        await self.send_text(
            f"✅ 已切换到画师串 #{index}\n"
            f"名称: {selected_preset['name']}\n"
            f"模型: {model_display}"
        )
        return True, f"已切换到画师串 #{index}", True

    async def _handle_set_size(self, platform: str, chat_id: str, size_key: str) -> Tuple[bool, Optional[str], bool]:
        """处理尺寸切换命令"""
        if not size_key:
            current_size = session_state.get_selected_size(platform, chat_id)
            if current_size:
                size_name = "自定义"
                for key, value in self.SIZE_MAPPINGS.items():
                    if value == current_size and key in ["竖图", "横图", "方图"]:
                        size_name = key
                        break
                current_display = f"当前尺寸: {size_name} ({current_size})"
            else:
                current_display = "当前使用默认配置尺寸"

            await self.send_text(
                f"{current_display}\n\n"
                "可用尺寸:\n"
                "竖/v - 竖图 (832x1216)\n"
                "横/h - 横图 (1216x832)\n"
                "方/s - 方图 (1024x1024)\n\n"
                "使用方法: /nai size <尺寸代号>"
            )
            return True, "显示尺寸列表", True

        if size_key not in self.SIZE_MAPPINGS:
            await self.send_text(
                f"❌ 无效的尺寸代号: {size_key}\n\n"
                "可用尺寸:\n"
                "竖/v - 竖图 (832x1216)\n"
                "横/h - 横图 (1216x832)\n"
                "方/s - 方图 (1024x1024)"
            )
            return False, "无效的尺寸代号", True

        size_value = self.SIZE_MAPPINGS[size_key]
        session_state.set_selected_size(platform, chat_id, size_value)

        size_names = {"832x1216": "竖图", "1216x832": "横图", "1024x1024": "方图"}
        size_display = size_names.get(size_value, size_value)

        await self.send_text(
            f"✅ 已切换到: {size_display}\n"
            f"尺寸: {size_value}"
        )
        return True, f"已切换到尺寸 {size_value}", True

    @staticmethod
    def _get_model_config_section(model_name: str) -> Tuple[Optional[str], str]:
        """根据模型名称获取配置节和显示名称"""
        if "nai-diffusion-3" in model_name:
            return "model_nai3", "NAI V3"
        elif "nai-diffusion-4-5" in model_name:
            return "model_nai4_5", "NAI V4.5"
        elif "nai-diffusion-4" in model_name:
            return "model_nai4", "NAI V4"
        return None, ""

    # ==================== 兼容性方法（委托给 session_state）====================

    @classmethod
    def is_admin_mode_enabled(cls, platform: str, chat_id: str, get_config_func) -> bool:
        """兼容方法：检查管理员模式是否启用"""
        return session_state.is_admin_mode_enabled(platform, chat_id, get_config_func)

    @classmethod
    def check_user_permission(cls, platform: str, chat_id: str, user_id: str, get_config_func) -> bool:
        """兼容方法：检查用户权限"""
        return session_state.check_user_permission(platform, chat_id, user_id, get_config_func)

    @classmethod
    def get_selected_model(cls, platform: str, chat_id: str, get_config_func) -> Optional[str]:
        """兼容方法：获取选定的模型"""
        return session_state.get_selected_model(platform, chat_id)

    @classmethod
    def get_selected_artist_preset(cls, platform: str, chat_id: str, model_name: str, get_config_func) -> Optional[str]:
        """兼容方法：获取选定的画师串"""
        return session_state.get_selected_artist_preset(platform, chat_id, model_name, get_config_func)

    @classmethod
    def get_selected_size(cls, platform: str, chat_id: str) -> Optional[str]:
        """兼容方法：获取选定的尺寸"""
        return session_state.get_selected_size(platform, chat_id)
