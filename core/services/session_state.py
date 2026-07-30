# -*- coding: utf-8 -*-
"""
统一会话状态管理器

集中管理所有会话级别的运行时状态，包括：
- 管理员模式
- 模型选择
- 画师串选择
- 尺寸选择
- 自动撤回
- NSFW过滤
- 提示词显示
- Danbooru 检索结果显示

替代原来分散在各个 Command 类中的状态字典
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from src.common.logger import get_logger

from .nsfw_state_store import nsfw_state_store
from .session_preferences import SessionPreferences
from .transient_generation_state import TransientGenerationState
from .visual_continuity import StableVisualTags
from .visual_continuity_store import visual_continuity_store

logger = get_logger("nai_draw_plugin")


class SessionStateManager:
    """
    单例模式的会话状态管理器

    使用方式：
        from .services.session_state import session_state

        # 查询状态
        enabled = session_state.is_admin_mode_enabled(platform, chat_id, get_config)

        # 设置状态
        session_state.set_admin_mode(platform, chat_id, True)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        """初始化偏好与瞬态生成状态 Module。"""
        self._preferences = SessionPreferences()
        self._transient = TransientGenerationState()

    @staticmethod
    def _make_key(platform: str, chat_id: str) -> str:
        """生成会话唯一标识"""
        return f"{platform}:{chat_id}"

    # ==================== 管理员模式 ====================

    def is_admin_mode_enabled(
        self,
        platform: str,
        chat_id: str,
        get_config: Callable
    ) -> bool:
        """
        检查指定会话是否启用了管理员模式

        Args:
            platform: 平台标识
            chat_id: 会话ID（group_id 或 user_id）
            get_config: 获取配置的函数

        Returns:
            bool: 是否启用管理员模式
        """
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is not None and preference.admin_mode is not None:
            return preference.admin_mode
        return get_config("admin.default_admin_mode", False)

    def set_admin_mode(self, platform: str, chat_id: str, enabled: bool):
        """设置管理员模式"""
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, admin_mode=bool(enabled))
        logger.info(f"[nai_pic] 会话 {key} 管理员模式已{'开启' if enabled else '关闭'}")

    def check_user_permission(
        self,
        platform: str,
        chat_id: str,
        user_id: str,
        get_config: Callable
    ) -> bool:
        """
        检查用户是否有权限使用生图功能

        管理员模式关闭时：所有人都有权限
        管理员模式开启时：只有管理员有权限

        Args:
            platform: 平台标识
            chat_id: 会话ID
            user_id: 用户ID
            get_config: 获取配置的函数

        Returns:
            bool: 是否有权限
        """
        if not self.is_admin_mode_enabled(platform, chat_id, get_config):
            return True

        admin_users = self._get_admin_users(get_config)
        if not admin_users:
            # 未配置管理员列表时，管理员模式不生效（与 is_admin_user 语义保持一致）
            return True
        return str(user_id) in admin_users

    def is_admin_user(self, user_id: str, get_config: Callable) -> bool:
        """检查用户是否是管理员"""
        admin_users = self._get_admin_users(get_config)
        if not admin_users:
            # 未配置管理员列表时，默认允许所有人
            return True
        return str(user_id) in admin_users

    def _get_admin_users(self, get_config: Callable) -> List[str]:
        """获取标准化后的管理员 ID 列表。"""
        admin_users = get_config("admin.admin_users", [])
        if not isinstance(admin_users, list):
            return []
        return [str(user_id).strip() for user_id in admin_users if str(user_id).strip()]

    # ==================== 模型选择 ====================

    def get_selected_model(self, platform: str, chat_id: str) -> Optional[str]:
        """获取指定会话选定的模型"""
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        return preference.selected_model if preference is not None else None

    def set_selected_model(self, platform: str, chat_id: str, model: str):
        """设置模型"""
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, selected_model=model)
        logger.info(f"[nai_pic] 会话 {key} 已切换模型: {model}")

    # ==================== 画师串选择 ====================

    def get_selected_artist_index(self, platform: str, chat_id: str) -> int:
        """获取指定会话选定的画师串索引（从1开始）"""
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is None or preference.selected_artist_index is None:
            return 1
        return preference.selected_artist_index

    def get_effective_artist_index(
        self,
        platform: str,
        chat_id: str,
        model_name: str,
        get_config: Callable,
    ) -> int:
        """
        获取指定会话当前实际生效的画师串索引。

        若会话中未手动切换，则回退到配置中的 default_artist_preset。
        """
        config_section = self._get_artist_config_section(model_name)
        if not config_section:
            return 1

        artist_presets_raw = get_config(f"{config_section}.artist_presets", [])
        artist_presets = self._parse_artist_presets(artist_presets_raw)
        if not artist_presets:
            return 1

        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is not None and preference.selected_artist_index is not None:
            selected_index = preference.selected_artist_index
            return selected_index if 1 <= selected_index <= len(artist_presets) else 1

        return self._resolve_default_artist_index(config_section, artist_presets, get_config)

    def set_selected_artist_index(self, platform: str, chat_id: str, index: int):
        """设置画师串索引"""
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, selected_artist_index=index)
        logger.info(f"[nai_pic] 会话 {key} 已切换画师串: #{index}")

    def get_selected_artist_preset(
        self,
        platform: str,
        chat_id: str,
        model_name: str,
        get_config: Callable
    ) -> Optional[str]:
        """获取指定会话选定的画师串内容。"""
        selected_preset = self.get_selected_artist_preset_config(
            platform,
            chat_id,
            model_name,
            get_config,
        )
        if not selected_preset:
            return None
        return selected_preset.get("prompt")

    def get_selected_artist_preset_config(
        self,
        platform: str,
        chat_id: str,
        model_name: str,
        get_config: Callable,
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定会话当前选中的画师预设完整配置。

        Returns:
            统一格式的预设字典，至少包含 name / prompt，
            若配置了非空的 negative_prompt_add 也会一并返回。
        """
        config_section = self._get_artist_config_section(model_name)
        if not config_section:
            return None

        # 获取画师串列表
        artist_presets_raw = get_config(f"{config_section}.artist_presets", [])
        if not artist_presets_raw:
            return None

        # 解析画师串列表
        artist_presets = self._parse_artist_presets(artist_presets_raw)
        if not artist_presets:
            return None

        # 优先使用会话中手动切换的画师串
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is not None and preference.selected_artist_index is not None:
            selected_index = preference.selected_artist_index
        else:
            selected_index = self._resolve_default_artist_index(config_section, artist_presets, get_config)

        # 确保索引有效
        if 1 <= selected_index <= len(artist_presets):
            return artist_presets[selected_index - 1]
        return artist_presets[0] if artist_presets else None

    @staticmethod
    def _get_artist_config_section(model_name: str) -> Optional[str]:
        """根据模型名解析画师串配置节。"""
        if "nai-diffusion-3" in model_name:
            return "model_nai3"
        if "nai-diffusion-4-5" in model_name:
            return "model_nai4_5"
        if "nai-diffusion-4" in model_name:
            return "model_nai4"
        return None

    def _resolve_default_artist_index(
        self,
        config_section: str,
        artist_presets: List[Dict[str, str]],
        get_config: Callable,
    ) -> int:
        """解析配置中的默认画师串，支持序号或名称。"""
        default_value = get_config(f"{config_section}.default_artist_preset", "")
        if default_value is None:
            return 1

        if isinstance(default_value, int):
            return default_value if 1 <= default_value <= len(artist_presets) else 1

        default_text = str(default_value).strip()
        if not default_text:
            return 1

        if default_text.isdigit():
            index = int(default_text)
            return index if 1 <= index <= len(artist_presets) else 1

        for index, preset in enumerate(artist_presets, 1):
            if preset.get("name", "").strip() == default_text:
                return index

        logger.warning(f"[nai_pic] 默认画师串配置无效: {config_section}.default_artist_preset={default_text!r}，回退到第一个预设")
        return 1

    @staticmethod
    def _parse_artist_presets(presets_raw: List) -> List[Dict[str, Any]]:
        """
        解析画师串预设列表，兼容新旧格式

        新格式：[{"name": "风格名", "prompt": "画师串内容", "negative_prompt_add": "可选负面提示词"}, ...]
        旧格式：["画师串内容1", "画师串内容2", ...]

        Returns:
            统一返回 [{"name": "...", "prompt": "...", "negative_prompt_add": "..."}, ...]
        """
        if not presets_raw:
            return []

        result = []
        for i, preset in enumerate(presets_raw, 1):
            if isinstance(preset, dict):
                name = preset.get("name", f"画师串 {i}")
                prompt = preset.get("prompt", "")
                normalized_preset: Dict[str, Any] = {"name": name, "prompt": prompt}
                negative_prompt_add = str(preset.get("negative_prompt_add", "") or "").strip()
                if negative_prompt_add:
                    normalized_preset["negative_prompt_add"] = negative_prompt_add
                result.append(normalized_preset)
            elif isinstance(preset, str):
                preview = preset[:30] + "..." if len(preset) > 30 else preset
                result.append({"name": f"#{i} {preview}", "prompt": preset})
            else:
                logger.warning(f"[nai_pic] 跳过无效的画师串格式: {type(preset)}")
                continue

        return result

    # ==================== 尺寸选择 ====================

    def get_selected_size(self, platform: str, chat_id: str) -> Optional[str]:
        """获取指定会话选定的尺寸"""
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        return preference.selected_size if preference is not None else None

    def set_selected_size(self, platform: str, chat_id: str, size: str):
        """设置尺寸"""
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, selected_size=size)
        logger.info(f"[nai_pic] 会话 {key} 已切换尺寸: {size}")

    # ==================== 自动撤回 ====================

    def is_recall_enabled(
        self,
        platform: str,
        chat_id: str,
        get_config: Callable
    ) -> bool:
        """检查是否启用自动撤回"""
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is not None and preference.recall_enabled is not None:
            return preference.recall_enabled
        return get_config("auto_recall.enabled", False)

    def set_recall_enabled(self, platform: str, chat_id: str, enabled: bool):
        """设置自动撤回"""
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, recall_enabled=bool(enabled))
        logger.info(f"[nai_pic] 会话 {key} 自动撤回已{'开启' if enabled else '关闭'}")

    # ==================== NSFW过滤 ====================

    def is_nsfw_filter_enabled(
        self,
        platform: str,
        chat_id: str,
        get_config: Callable
    ) -> bool:
        """检查是否启用NSFW过滤。

        优先级：持久化 store（含跨重启状态） > 配置默认。
        store 命中即返回，让重启后的实例继续沿用上次会话的开关。
        """
        persisted = nsfw_state_store.get(platform, chat_id)
        if persisted is not None:
            return persisted
        return get_config("nsfw_filter.enabled", False)

    def set_nsfw_filter_enabled(self, platform: str, chat_id: str, enabled: bool):
        """设置NSFW过滤并落盘，跨重启保留。"""
        nsfw_state_store.set(platform, chat_id, enabled)
        key = self._make_key(platform, chat_id)
        logger.info(f"[nai_pic] 会话 {key} NSFW过滤已{'开启' if enabled else '关闭'}")

    # ==================== 提示词显示 ====================

    def is_prompt_show_enabled(
        self,
        platform: str,
        chat_id: str,
        get_config: Callable
    ) -> bool:
        """检查是否启用提示词显示"""
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is not None and preference.prompt_show_enabled is not None:
            return preference.prompt_show_enabled
        default_enabled = get_config("prompt_show.enabled", None)
        if default_enabled is not None:
            return bool(default_enabled)

        # 兼容旧配置：历史版本可能使用 prompt_generator.show_prompt
        return bool(get_config("prompt_generator.show_prompt", False))

    def set_prompt_show_enabled(self, platform: str, chat_id: str, enabled: bool):
        """设置提示词显示"""
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, prompt_show_enabled=bool(enabled))
        logger.info(f"[nai_pic] 会话 {key} 提示词显示已{'开启' if enabled else '关闭'}")

    def is_tag_retriever_show_enabled(
        self,
        platform: str,
        chat_id: str,
        get_config: Callable,
    ) -> bool:
        """检查是否启用 Danbooru 检索结果显示。"""
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is not None and preference.tag_retriever_show_enabled is not None:
            return preference.tag_retriever_show_enabled
        return bool(get_config("tag_retriever.show_result", False))

    def set_tag_retriever_show_enabled(self, platform: str, chat_id: str, enabled: bool) -> None:
        """设置 Danbooru 检索结果显示。"""
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, tag_retriever_show_enabled=bool(enabled))
        logger.info(f"[nai_pic] 会话 {key} Danbooru 检索结果显示已{'开启' if enabled else '关闭'}")

    # ==================== 角色参考提取目标（/nai ref） ====================

    # NewAPI §20.4 character_references[i].type 取值；前端命令值（包括 ``both`` 别名）
    # 在 plugin 层归一后再透到 store
    _ALLOWED_CHARACTER_REFERENCE_TYPES = ("character", "style", "character&style")

    def get_character_reference_type(
        self,
        platform: str,
        chat_id: str,
        get_config: Callable,
    ) -> str:
        """读取本会话的 character_references.type，缺省回退 config / API 默认。"""
        key = self._make_key(platform, chat_id)
        preference = self._preferences.get(key)
        if preference is not None and preference.character_reference_type is not None:
            return preference.character_reference_type
        raw = str(get_config("character_reference.type", "character&style") or "").strip()
        if raw not in self._ALLOWED_CHARACTER_REFERENCE_TYPES:
            return "character&style"
        return raw

    def set_character_reference_type(self, platform: str, chat_id: str, type_value: str) -> None:
        """设置本会话的 character_references.type；非法值 raises ValueError。"""
        normalized = (type_value or "").strip()
        if normalized not in self._ALLOWED_CHARACTER_REFERENCE_TYPES:
            raise ValueError(
                f"无效的 character_references.type：{type_value!r}；只允许 "
                f"{self._ALLOWED_CHARACTER_REFERENCE_TYPES}"
            )
        key = self._make_key(platform, chat_id)
        self._preferences.update(key, character_reference_type=normalized)
        logger.info(f"[nai_pic] 会话 {key} 角色参考类型已切换: {normalized}")

    # ==================== 调试/管理 ====================

    def get_session_state_summary(self, platform: str, chat_id: str) -> Dict[str, Any]:
        """获取指定会话的状态摘要（用于调试）"""
        key = self._make_key(platform, chat_id)
        return {
            "key": key,
            **self._preferences.summary(key),
            "nsfw_filter": nsfw_state_store.get(platform, chat_id),
        }

    def clear_session_state(self, platform: str, chat_id: str):
        """清除指定会话的所有状态（含 NSFW 持久化条目）。"""
        key = self._make_key(platform, chat_id)
        self._preferences.clear(key)
        nsfw_state_store.clear(platform, chat_id)
        logger.info(f"[nai_pic] 会话 {key} 状态已清除")

    def clear_transient_generation_state(self, chat_stream_id: str) -> None:
        """清除一个聊天流的生成上下文、冷却、pending 与视觉连续性状态。"""
        self._transient.clear_session(chat_stream_id)
        visual_continuity_store.clear(chat_stream_id)

    # ==================== 上一轮提示词（Action 专用） ====================

    def get_last_nai_context(
        self, chat_stream_id: str, ttl: float = 0
    ) -> Tuple[Optional[str], Optional[str]]:
        """获取指定聊天流的上一轮 LLM 提示词及用户请求。

        Args:
            chat_stream_id: 聊天流 ID
            ttl: 有效时间（秒），>0 时检查过期；过期则删除并返回 (None, None)

        Returns:
            (prompt, request)；无数据或已过期时返回 (None, None)
        """
        return self._transient.get_last_nai_context(chat_stream_id, ttl)

    def set_last_nai_context(
        self,
        chat_stream_id: str,
        prompt: str,
        request: str = "",
        ttl: float = 0,
    ) -> None:
        """设置指定聊天流的上一轮 LLM 提示词及用户请求。

        自动附带当前时间戳。
        """
        self._transient.set_last_nai_context(chat_stream_id, prompt, request, ttl)

    # ==================== Bot 情景图视觉连续性 ====================

    def get_visual_continuity(
        self,
        chat_stream_id: str,
        ttl: float = 0,
    ) -> Optional[StableVisualTags]:
        """获取服装与环境的稳定 NovelAI Tag（持久化存储，跨重启保留）。

        ``ttl`` 只约束当前服装/环境；卡片库不过期，始终可供 switch 回切。
        """
        return visual_continuity_store.get(chat_stream_id, ttl)

    def set_visual_continuity(
        self,
        chat_stream_id: str,
        stable: StableVisualTags,
    ) -> None:
        """保存已经生成过、后续需要逐字复用的稳定 NovelAI Tag 并落盘。"""
        visual_continuity_store.set(chat_stream_id, stable)

    # ==================== 上一轮自拍场景（Action 自拍专用） ====================

    def get_last_selfie_context(
        self, chat_stream_id: str, ttl: float = 0
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, List[str]]]:
        """获取指定聊天流的上一轮自拍提示词、请求、场景摘要与结构化锚点。"""
        return self._transient.get_last_selfie_context(chat_stream_id, ttl)

    def set_last_selfie_context(
        self,
        chat_stream_id: str,
        prompt: str,
        request: str = "",
        scene_summary: str = "",
        anchor_data: Optional[Dict[str, List[str]]] = None,
        ttl: float = 0,
    ) -> None:
        """设置指定聊天流的上一轮自拍提示词、请求、场景摘要与结构化锚点。"""
        self._transient.set_last_selfie_context(
            chat_stream_id,
            prompt,
            request,
            scene_summary,
            anchor_data,
            ttl,
        )

    # ==================== Action 最近出图时间 ====================

    def get_last_action_image_sent_at(self, chat_stream_id: str) -> Optional[float]:
        """获取指定聊天流最近一次自动出图成功发送时间。"""
        return self._transient.get_last_action_image_sent_at(chat_stream_id)

    def set_last_action_image_sent_at(self, chat_stream_id: str, sent_at: Optional[float] = None) -> None:
        """记录指定聊天流最近一次自动出图成功发送时间。"""
        self._transient.set_last_action_image_sent_at(chat_stream_id, sent_at)

    # ==================== 图片生成中状态 ====================

    def get_pending_image_generation_started_at(self, chat_stream_id: str) -> Optional[float]:
        """获取指定聊天流当前生成中的图片任务开始时间。"""
        return self._transient.get_pending_image_generation_started_at(chat_stream_id)

    def acquire_pending_image_generation(self, chat_stream_id: str) -> Optional[str]:
        """原子获取当前聊天流的生成 lease。"""
        return self._transient.acquire_pending_image_generation(chat_stream_id)

    def release_pending_image_generation(self, chat_stream_id: str, owner: str) -> bool:
        """仅由 lease owner 清除当前聊天流的 pending 状态。"""
        return self._transient.release_pending_image_generation(chat_stream_id, owner)

    def set_pending_image_generation(self, chat_stream_id: str, started_at: Optional[float] = None) -> None:
        """标记指定聊天流存在进行中的图片任务。"""
        self._transient.set_pending_image_generation(chat_stream_id, started_at)

    def clear_pending_image_generation(self, chat_stream_id: str) -> None:
        """清除指定聊天流的图片生成中状态。"""
        self._transient.clear_pending_image_generation(chat_stream_id)


# 全局单例实例
session_state = SessionStateManager()
