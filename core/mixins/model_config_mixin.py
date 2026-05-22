# -*- coding: utf-8 -*-
"""
模型配置混入

统一处理模型/画师串选择及版本配置合并逻辑
使用 session_state 获取会话级别的运行时状态
"""
from typing import Dict, Any, Tuple, Optional

from src.common.logger import get_logger

from ..services.session_state import session_state

logger = get_logger("nai_draw_plugin")


class ModelConfigMixin:
    """为命令和动作提供统一的模型配置解析逻辑"""

    @staticmethod
    def _merge_negative_prompts(base_negative: Any, extra_negative: Any) -> str:
        """将额外负面提示词追加到现有负面提示词后面。"""
        base_text = str(base_negative or "").strip()
        extra_text = str(extra_negative or "").strip()
        if base_text and extra_text:
            return f"{base_text}, {extra_text}"
        return base_text or extra_text

    def _get_model_config(
        self,
        is_selfie: Optional[bool] = None,
        *,
        apply_artist_preset: bool = True,
    ) -> Dict[str, Any]:
        """
        获取合并后的模型配置

        合并顺序：
        1. 基础配置（model.*）
        2. 版本特定配置（model_nai3/model_nai4/model_nai4_5）
        3. 会话级别覆盖（模型、画师串、尺寸）
        4. NSFW 过滤
        """
        base_config = self.get_config("model", {})  # type: ignore[attr-defined]
        if not base_config:
            logger.error(f"{self._log_prefix} 模型配置读取失败")
            return {}

        platform, chat_id, _ = self._get_chat_identity()

        # 获取模型名称（优先使用会话选择）
        model_name = base_config.get("default_model", "")
        if platform and chat_id:
            selected_model = session_state.get_selected_model(platform, chat_id)
            if selected_model:
                model_name = selected_model
                logger.debug(f"{self._log_prefix} 使用会话选定的模型: {selected_model}")

        # 获取版本特定配置
        version_config = self._get_version_config(model_name)

        # 合并配置
        merged_config = base_config.copy()
        if model_name:
            merged_config["default_model"] = model_name

        if version_config:
            for key, value in version_config.items():
                if key == "nai_extra_params":
                    # 合并 extra_params
                    base_extra = merged_config.get("nai_extra_params", {}) or {}
                    merged_extra = dict(base_extra)
                    merged_extra.update(value or {})
                    merged_config["nai_extra_params"] = merged_extra
                elif key == "artist_presets":
                    # 跳过画师串列表，后续单独处理
                    continue
                else:
                    merged_config[key] = value

        # 应用会话级别的画师串选择
        if apply_artist_preset and platform and chat_id and model_name:
            selected_artist = session_state.get_selected_artist_preset_config(
                platform, chat_id, model_name, self.get_config  # type: ignore[attr-defined]
            )
            if selected_artist:
                selected_prompt = str(selected_artist.get("prompt", "") or "")
                if selected_prompt:
                    merged_config["nai_artist_prompt"] = selected_prompt
                    logger.debug(f"{self._log_prefix} 使用会话选定的画师串: {selected_prompt[:50]}...")

                # 仅当预设里显式配置了非空负面提示词时才覆盖模型默认值；
                # 未配置、空字符串或纯空白都继续回退到模型段 negative_prompt_add。
                selected_negative = str(selected_artist.get("negative_prompt_add", "") or "").strip()
                if selected_negative:
                    merged_config["negative_prompt_add"] = selected_negative
                    logger.debug(f"{self._log_prefix} 使用画师预设专属负面提示词: {selected_negative[:50]}...")

        # 应用会话级别的尺寸选择
        if platform and chat_id:
            selected_size = session_state.get_selected_size(platform, chat_id)
            if selected_size:
                merged_config["nai_size"] = selected_size
                logger.debug(f"{self._log_prefix} 使用会话选定的尺寸: {selected_size}")

        # 自拍专属负面先合并：放到通用负面之前，确保自拍形象反向锁
        # （hair behind back / side parting / asymmetrical hair 等）享有高优先级
        # NAI 4/4.5 负面同样遵循"越前越高优先级"
        if is_selfie:
            selfie_negative = merged_config.get("selfie_negative_prompt_add", "")
            merged_config["negative_prompt_add"] = self._merge_negative_prompts(
                selfie_negative,
                merged_config.get("negative_prompt_add", ""),
            )
            if str(selfie_negative or "").strip():
                logger.debug(f"{self._log_prefix} 已追加自拍专属负面提示词")

        # 应用 NSFW 过滤（最后合并 → 拼到最前 → 永远享有最高优先级）
        if platform and chat_id:
            if session_state.is_nsfw_filter_enabled(
                platform, chat_id, self.get_config  # type: ignore[attr-defined]
            ):
                nsfw_tags = self.get_config("nsfw_filter.filter_tags", "{{{{{nsfw}}}}}")  # type: ignore[attr-defined]
                current_negative = merged_config.get("negative_prompt_add", "")
                merged_config["negative_prompt_add"] = self._merge_negative_prompts(
                    nsfw_tags,
                    current_negative,
                )
                logger.debug(f"{self._log_prefix} 已应用NSFW过滤: {nsfw_tags}")

        return merged_config

    def _get_version_config(self, model_name: str) -> Dict[str, Any]:
        """根据模型名称获取版本特定配置"""
        if not model_name:
            return {}

        if "nai-diffusion-3" in model_name:
            config_section = "model_nai3"
            logger.debug(f"{self._log_prefix} 检测到 NAI V3 模型，使用 {config_section} 配置")
        elif "nai-diffusion-4-5" in model_name:
            config_section = "model_nai4_5"
            logger.debug(f"{self._log_prefix} 检测到 NAI V4.5 模型，使用 {config_section} 配置")
        elif "nai-diffusion-4" in model_name:
            config_section = "model_nai4"
            logger.debug(f"{self._log_prefix} 检测到 NAI V4 模型，使用 {config_section} 配置")
        else:
            return {}

        return self.get_config(config_section, {})  # type: ignore[attr-defined]

    @property
    def _log_prefix(self) -> str:
        """获取日志前缀"""
        return getattr(self, "log_prefix", "nai_draw_plugin")

    def _get_chat_identity(self) -> Tuple[str, str, str]:
        """
        获取当前会话身份信息

        Returns:
            (platform, chat_id, user_id) 元组
        """
        message = getattr(self, "action_message", None) or getattr(self, "message", None)
        if not message:
            return "", "", ""

        # 优先从常规 MessageRecv（带 message_info）中获取
        info = getattr(message, "message_info", None)
        if info:
            platform = str(getattr(info, "platform", "") or "")
            group_info = getattr(info, "group_info", None)
            user_info = getattr(info, "user_info", None)

            chat_id = ""
            if group_info and getattr(group_info, "group_id", None):
                chat_id = str(group_info.group_id)
            elif user_info and getattr(user_info, "user_id", None):
                chat_id = str(user_info.user_id)

            user_id = str(getattr(user_info, "user_id", "") if user_info else "")
            return platform, chat_id, user_id

        # Planner 触发的 Action 通常只有 DatabaseMessages，从 chat_info 中获取
        chat_info = getattr(message, "chat_info", None)
        user_info = getattr(message, "user_info", None) or getattr(chat_info, "user_info", None)

        platform = str(getattr(chat_info, "platform", "") or "")

        chat_id = ""
        group_info = getattr(chat_info, "group_info", None) if chat_info else None
        if group_info and getattr(group_info, "group_id", None):
            chat_id = str(group_info.group_id)
        elif user_info and getattr(user_info, "user_id", None):
            chat_id = str(user_info.user_id)
        elif chat_info and getattr(chat_info, "stream_id", None):
            chat_id = str(chat_info.stream_id)

        user_id = str(getattr(user_info, "user_id", "") if user_info else "")

        return platform, chat_id, user_id
