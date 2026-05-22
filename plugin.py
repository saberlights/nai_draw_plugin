from typing import Any, List
from weakref import WeakSet

import asyncio
import inspect
import os
import tomllib

from maibot_sdk import Action, Command, HookHandler, MaiBotPlugin
from maibot_sdk.types import ActivationType, HookMode

from src.core.config_types import ConfigField

from .core.constants import NAI_PIC_IMAGE_DISPLAY_MARKER
from .core.services.session_state import session_state
from .core.services.tag_retriever import get_tag_retriever, reset_tag_retriever
from .runtime_recall import (
    attach_plugin_image_marker_to_message,
    remember_sent_plugin_image_message,
    reset_runtime_recall_tracking_state,
)
from .sdk_runtime import NaiInvocation


def _merge_config_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置，优先使用运行时覆盖值。"""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_online_retriever_api() -> tuple[Any, Any] | None:
    """按需加载在线检索器，避免本地模式在缺依赖时阻塞插件注册。"""
    try:
        from .core.services.danbooru_online_retriever import get_online_retriever, reset_online_retriever
    except Exception:
        return None
    return get_online_retriever, reset_online_retriever


class NaiPicPlugin(MaiBotPlugin):
    """同步 nai_pic_plugin 业务逻辑的 NovelAI NewAPI 网关图片生成插件。"""

    # 插件基本信息
    plugin_name = "nai_draw_plugin"
    plugin_version = "1.2.1"
    plugin_author = "Rabbit"
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[str] = ["httpx", "requests"]
    config_file_name = "config.toml"

    # 配置节描述
    config_section_descriptions = {
        "plugin": "插件基本配置",
        "model": "NovelAI Web 请求连接与默认模型配置",
        "prompt_generator": "提示词生成配置（/nai）",
        "prompt_generator.custom_model": "提示词生成自定义模型配置",
        "random_scene": "随机场景生成配置（/nai 随机）",
        "random_scene.custom_model": "随机场景生成自定义模型配置",
        "tagger": "图片打标配置（/打标）",
        "tagger.custom_model": "图片打标自定义模型配置",
        "components": "组件配置",
        "prompt_show": "提示词显示配置",
        "nsfw_filter": "NSFW 内容过滤配置",
        "auto_recall": "自动撤回配置",
        "action_guard": "自动出图触发保护配置",
        "admin": "管理员权限配置",
        "tag_retriever": "Danbooru Tag 检索增强配置",
        "custom_prompt": "自定义系统提示词配置",
        "model_nai4_5": "NovelAI V4.5 模型专用配置（nai-diffusion-4-5-full 等最新模型）",
        "model_nai4": "NovelAI V4 模型专用配置（nai-diffusion-4-curated、nai-diffusion-4-full 等）",
        "model_nai3": "NovelAI V3 模型专用配置（nai-diffusion-3 和 nai-diffusion-3-furry）",
    }

    # 配置Schema
    config_schema = {
        "plugin": {
            "name": ConfigField(
                type=str,
                default="nai_draw_plugin",
                description="NovelAI Web 图片生成插件",
                required=True
            ),
            "config_version": ConfigField(
                type=str,
                default="1.3.0",
                description="插件配置版本号"
            ),
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用插件"
            )
        },
        "model": {
            "name": ConfigField(
                type=str,
                default="NovelAI NewAPI Gateway",
                description="模型显示名称"
            ),
            "base_url": ConfigField(
                type=str,
                default="https://api.tuercha.com",
                description="NewAPI 兼容网关基础地址",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="",
                description="NewAPI 鉴权密钥（OpenAI 风格 Bearer Token）",
                required=False
            ),
            "available_models": ConfigField(
                type=list,
                default=[
                    "nai-diffusion-3",
                    "nai-diffusion-3-furry",
                    "nai-diffusion-4-curated",
                    "nai-diffusion-4-full",
                    "nai-diffusion-4-5-curated",
                    "nai-diffusion-4-5-full",
                ],
                description="可用的 NovelAI 模型列表"
            ),
            "default_model": ConfigField(
                type=str,
                default="nai-diffusion-4-5-full",
                description="当前使用的模型名称（从 available_models 中选择）"
            ),
            "nai_endpoint": ConfigField(
                type=str,
                default="/v1/chat/completions",
                description="NewAPI 兼容生图端点路径"
            ),
            "nai_request_timeout": ConfigField(
                type=float,
                default=600.0,
                description="NewAPI 生图请求超时（秒）"
            ),
            "nai_proxy_mode": ConfigField(
                type=str,
                default="auto",
                description="插件代理模式：auto=先继承环境代理，代理失败后回退直连；inherit=始终继承环境代理；direct=始终直连"
            ),
            "nai_max_tokens": ConfigField(
                type=int,
                default=100000,
                description="单次绘图允许消耗的 token 预算（1 Anlas = 10000 tokens，推荐 100000=10 Anlas）"
            ),
        },
        "model_nai3": {
            "artist_presets": ConfigField(
                type=list,
                default=[
                    {"name": "示例风格1", "prompt": "artist:example1, artist:example2, year 2023"},
                    {"name": "示例风格2", "prompt": "artist:example3, artist:example4, year 2024"}
                ],
                description="NAI V3 画师风格预设列表（可配置多个），每个预设包含 name（显示名称）、prompt（画师串内容），可选填写 negative_prompt_add（该预设专属负面提示词）"
            ),
            "default_artist_preset": ConfigField(
                type=str,
                default="",
                description="NAI V3 默认画师风格预设，支持填写预设名称或序号；留空时默认使用第一个预设"
            ),
            "nai_artist_prompt": ConfigField(
                type=str,
                default="",
                description="NAI V3 专用画师风格提示词（可选，优先级低于 artist_presets）"
            ),
            "nai_size": ConfigField(
                type=str,
                default="竖图",
                description="NAI V3 专用图片尺寸"
            ),
            "sampler": ConfigField(
                type=str,
                default="k_euler_ancestral",
                description="NAI V3 专用采样器"
            ),
            "num_inference_steps": ConfigField(
                type=int,
                default=28,
                description="NAI V3 专用推理步数"
            ),
            "guidance_scale": ConfigField(
                type=float,
                default=5.0,
                description="NAI V3 专用指导强度"
            ),
            "seed": ConfigField(
                type=int,
                default=-1,
                description="NAI V3 专用随机种子；小于 0 时每次随机"
            ),
            "quality_toggle": ConfigField(
                type=bool,
                default=True,
                description="NAI V3 low-level qualityToggle"
            ),
            "auto_smea": ConfigField(
                type=bool,
                default=False,
                description="NAI V3 low-level autoSmea"
            ),
            "image_format": ConfigField(
                type=str,
                default="png",
                description="NAI V3 返回图片格式"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1280",
                description="NAI V3 专用默认尺寸"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V3 专用自动添加的提示词后缀"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V3 专用负面提示词"
            ),
            "selfie_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V3 专用自拍模式提示词"
            ),
            "selfie_negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V3 专用自拍模式负面提示词，会追加到 negative_prompt_add 后面"
            ),
            "nai_extra_params": ConfigField(
                type=dict,
                default={},
                description="NAI V3 low-level 原始 parameters 透传字段。可在此自定义任意底层参数；其中 width/height、steps、n_samples、qualityToggle 通常会影响点数，参考图相关字段可能也会影响点数。"
            )
        },
        "model_nai4": {
            "artist_presets": ConfigField(
                type=list,
                default=[
                    {"name": "风格组合1", "prompt": "1.2::artist1::, 1.0::artist2::, 0.9::artist3::"},
                    {"name": "风格组合2", "prompt": "1.5::artist4::, 1.0::artist5::, 0.8::artist6::"}
                ],
                description="NAI V4 画师风格预设列表（可配置多个），每个预设包含 name（显示名称）、prompt（画师串内容），可选填写 negative_prompt_add（该预设专属负面提示词）"
            ),
            "default_artist_preset": ConfigField(
                type=str,
                default="",
                description="NAI V4 默认画师风格预设，支持填写预设名称或序号；留空时默认使用第一个预设"
            ),
            "nai_artist_prompt": ConfigField(
                type=str,
                default="",
                description="NAI V4 专用画师风格提示词（可选，优先级低于 artist_presets）"
            ),
            "nai_size": ConfigField(
                type=str,
                default="竖图",
                description="NAI V4 专用图片尺寸"
            ),
            "sampler": ConfigField(
                type=str,
                default="k_euler_ancestral",
                description="NAI V4 专用采样器"
            ),
            "num_inference_steps": ConfigField(
                type=int,
                default=28,
                description="NAI V4 专用推理步数"
            ),
            "guidance_scale": ConfigField(
                type=float,
                default=5.0,
                description="NAI V4 专用指导强度"
            ),
            "seed": ConfigField(
                type=int,
                default=-1,
                description="NAI V4 专用随机种子；小于 0 时每次随机"
            ),
            "quality_toggle": ConfigField(
                type=bool,
                default=True,
                description="NAI V4 low-level qualityToggle"
            ),
            "auto_smea": ConfigField(
                type=bool,
                default=False,
                description="NAI V4 low-level autoSmea"
            ),
            "image_format": ConfigField(
                type=str,
                default="png",
                description="NAI V4 返回图片格式"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1280",
                description="NAI V4 专用默认尺寸"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4 专用自动添加的提示词后缀"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4 专用负面提示词"
            ),
            "selfie_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4 专用自拍模式提示词"
            ),
            "selfie_negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4 专用自拍模式负面提示词，会追加到 negative_prompt_add 后面"
            ),
            "nai_extra_params": ConfigField(
                type=dict,
                default={},
                description="NAI V4 low-level 原始 parameters 透传字段。可在此自定义任意底层参数；其中 width/height、steps、n_samples、qualityToggle 通常会影响点数，参考图相关字段可能也会影响点数。"
            )
        },
        "model_nai4_5": {
            "artist_presets": ConfigField(
                type=list,
                default=[
                    {"name": "风格示例1", "prompt": "1.2::artist:example1::, 1.0::artist:example2::, 0.8::artist:example3::"},
                    {"name": "风格示例2", "prompt": "1.5::artist:example4::, 1.3::artist:example5::"}
                ],
                description="NAI V4.5 画师风格预设列表（可配置多个），每个预设包含 name（显示名称）、prompt（画师串内容），可选填写 negative_prompt_add（该预设专属负面提示词）"
            ),
            "default_artist_preset": ConfigField(
                type=str,
                default="",
                description="NAI V4.5 默认画师风格预设，支持填写预设名称或序号；留空时默认使用第一个预设"
            ),
            "nai_artist_prompt": ConfigField(
                type=str,
                default="",
                description="NAI V4.5 专用画师风格提示词（可选，优先级低于 artist_presets）"
            ),
            "nai_size": ConfigField(
                type=str,
                default="竖图",
                description="NAI V4.5 专用图片尺寸"
            ),
            "sampler": ConfigField(
                type=str,
                default="k_euler_ancestral",
                description="NAI V4.5 专用采样器"
            ),
            "num_inference_steps": ConfigField(
                type=int,
                default=28,
                description="NAI V4.5 专用推理步数"
            ),
            "guidance_scale": ConfigField(
                type=float,
                default=5.0,
                description="NAI V4.5 专用指导强度"
            ),
            "seed": ConfigField(
                type=int,
                default=-1,
                description="NAI V4.5 专用随机种子；小于 0 时每次随机"
            ),
            "quality_toggle": ConfigField(
                type=bool,
                default=True,
                description="NAI V4.5 low-level qualityToggle"
            ),
            "auto_smea": ConfigField(
                type=bool,
                default=False,
                description="NAI V4.5 low-level autoSmea"
            ),
            "image_format": ConfigField(
                type=str,
                default="png",
                description="NAI V4.5 返回图片格式"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1280",
                description="NAI V4.5 专用默认尺寸"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4.5 专用自动添加的提示词后缀"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4.5 专用负面提示词"
            ),
            "selfie_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4.5 专用自拍模式提示词"
            ),
            "selfie_negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="NAI V4.5 专用自拍模式负面提示词，会追加到 negative_prompt_add 后面"
            ),
            "nai_extra_params": ConfigField(
                type=dict,
                default={},
                description="NAI V4.5 low-level 原始 parameters 透传字段。可在此自定义任意底层参数；其中 width/height、steps、n_samples、qualityToggle 通常会影响点数，参考图相关字段可能也会影响点数。"
            )
        },
        "components": {
            "enable_debug_info": ConfigField(
                type=bool,
                default=False,
                description="是否显示调试信息"
            ),
        },
        "auto_recall": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用自动撤回"
            ),
            "delay_seconds": ConfigField(
                type=int,
                default=5,
                description="撤回延迟时间（秒）"
            ),
            "id_wait_seconds": ConfigField(
                type=int,
                default=15,
                description="等待正式消息ID的最长时间（秒）"
            ),
            "manual_max_age_seconds": ConfigField(
                type=int,
                default=3600,
                description="手动撤回允许命中的最老图片年龄（秒）；超出时直接视为不可撤回，避免反复命中老图"
            ),
            "allowed_groups": ConfigField(
                type=list,
                default=[],
                description="允许使用自动撤回功能的会话白名单（格式：platform:chat_id）"
            )
        },
        "admin": {
            "admin_users": ConfigField(
                type=list,
                default=[],
                description="管理员用户ID列表（字符串格式），管理员可以使用 /nai st/sp 命令控制管理员模式"
            ),
            "default_admin_mode": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用管理员模式（开启后仅管理员可使用 /nai 生图命令）"
            )
        },
        "prompt_show": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用提示词显示（使用 /nai pt on|off 可在运行时切换）"
            ),
            "hide_selfie_prompt_add": ConfigField(
                type=bool,
                default=False,
                description="提示词显示时是否隐藏配置文件中的自拍补充提示词（selfie_prompt_add）。仅影响展示，不影响实际生图。"
            )
        },
        "nsfw_filter": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用NSFW内容过滤（使用 /nai nsfw on|off 可在运行时切换）"
            ),
            "filter_tags": ConfigField(
                type=str,
                default="{{{{{nsfw}}}}}",
                description="NSFW过滤标签（高权重），当启用过滤时自动添加到负面提示词"
            )
        },
        "prompt_generator": {
            "model_name": ConfigField(
                type=str,
                default="",
                description="提示词生成使用的LLM模型代号，留空则自动选择"
            ),
            "output_format": ConfigField(
                type=str,
                default="json",
                description="提示词生成输出格式：json=结构化输出（默认，支持多人分段与意图元数据），text=纯提示词"
            ),
            "selfie_appearance_policy": ConfigField(
                type=str,
                default="auto",
                description="自拍外貌标签策略：auto=仅在用户未指定外貌时移除LLM随机发色/发型/瞳色（尽量保留配置中的自拍特征），never=始终移除（除非用户明确指定），keep=不移除"
            ),
            "enforce_tag_order": ConfigField(
                type=bool,
                default=False,
                description="是否对最终提示词做轻量排序（人数/视角前置、year后置），降低顺序混乱"
            ),
            "temperature": ConfigField(
                type=float,
                default=0.2,
                description="提示词生成LLM的温度设置"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=500,
                description="提示词生成LLM响应的最大token"
            ),
            "prompt_template": ConfigField(
                type=str,
                default="",
                description="自定义提示词生成模板，支持<<USER_REQUEST>>、<<SELFIE_HINT>>、<<CURRENT_TIME_CONTEXT>>、<<SELFIE_SCENE_CONTEXT>>占位符"
            ),
            "inherit_ttl": ConfigField(
                type=int,
                default=3600,
                description="上一轮提示词继承的有效时间（秒），超过后不再继承。默认3600（1小时），0=永不过期"
            ),
            "custom_model": ConfigField(
                type=dict,
                default={
                    "model_list": [],
                    "max_tokens": 500,
                    "temperature": 0.2,
                    "slow_threshold": 30.0
                },
                description="自定义模型配置（可选），model_list 中的模型名称必须是系统 model_config 中已定义的模型"
            )
        },
        "action_guard": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用 nai_web_draw Action 的触发保护：① 否定意图兜底（用户说'不要画'仍调用时拦截）② 频率分级保护"
            ),
            "explicit_request_min_interval_seconds": ConfigField(
                type=int,
                default=45,
                description="用户原话含明确画图/自拍/肖像/追图等强信号时的最小间隔（秒）"
            ),
            "proactive_min_interval_seconds": ConfigField(
                type=int,
                default=600,
                description="用户原话未含强信号、由 bot 主动判断要发图时的最小间隔（秒）；显著高于显式档以避免闲聊刷图"
            ),
        },
        "random_scene": {
            "temperature": ConfigField(
                type=float,
                default=1.0,
                description="随机场景生成LLM的温度设置"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=200,
                description="随机场景生成LLM响应的最大token"
            ),
            "custom_model": ConfigField(
                type=dict,
                default={
                    "model_list": [],
                    "max_tokens": 200,
                    "temperature": 1.0,
                    "slow_threshold": 30.0
                },
                description="随机场景生成自定义模型配置（可选），model_list 中的模型名称必须是系统 model_config 中已定义的模型"
            ),
        },
        "tagger": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用 /打标 命令"
            ),
            "model_task": ConfigField(
                type=str,
                default="vlm",
                description="打标使用的模型任务名（对应 model_config.model_task_config.<name>，默认 vlm）"
            ),
            "custom_model": ConfigField(
                type=dict,
                default={
                    "model_list": [],
                    "max_tokens": 1024,
                    "temperature": 0.2,
                    "slow_threshold": 30.0
                },
                description=(
                    "打标专用自定义模型配置（可选）。"
                    "当 model_list 非空时将优先使用该配置，完全独立于 model_task。"
                    "若未显式设置 tagger.max_tokens/tagger.temperature，将默认采用这里的同名值。"
                    "注意：所选模型必须支持图像输入。"
                )
            ),
            "temperature": ConfigField(
                type=float,
                default=0.2,
                description="打标模型温度（越低越稳定）"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=1200,
                description="打标模型最大输出 token"
            ),
        },
        "custom_prompt": {
            "system_prompt": ConfigField(
                type=str,
                default="",
                description="自定义系统提示词，会添加到 LLM 提示词规则的最前面，可用于自定义额外指导或规则"
            ),
        },
        "tag_retriever": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用 Danbooru Tag 检索增强"
            ),
            "mode": ConfigField(
                type=str,
                default="online",
                description="检索模式：online = 远程 DanbooruSearchOnline API，local = 本地 embedding"
            ),
            "api_url": ConfigField(
                type=str,
                default="https://sakizuki-danboorusearch.hf.space/api",
                description="DanbooruSearchOnline API 地址"
            ),
            "timeout": ConfigField(
                type=float,
                default=90.0,
                description="在线检索请求超时（秒）"
            ),
            "search_limit": ConfigField(
                type=int,
                default=30,
                description="在线 /search 返回标签上限"
            ),
            "search_top_k": ConfigField(
                type=int,
                default=5,
                description="在线 /search 每个分词段召回数"
            ),
            "related_limit": ConfigField(
                type=int,
                default=20,
                description="在线 /related 返回推荐上限"
            ),
            "related_seed_count": ConfigField(
                type=int,
                default=8,
                description="在线共现推荐使用的种子标签数量"
            ),
            "show_nsfw": ConfigField(
                type=bool,
                default=True,
                description="在线检索是否允许返回 NSFW 标签"
            ),
            "popularity_weight": ConfigField(
                type=float,
                default=0.15,
                description="在线检索标签热度权重"
            ),
            "top_k": ConfigField(
                type=int,
                default=50,
                description="本地检索返回的候选 tag 数量"
            ),
            "min_score": ConfigField(
                type=float,
                default=0.6,
                description="本地检索最低相似度阈值（低于此分数的不返回）"
            ),
        },
    }

    def __init__(self) -> None:
        """初始化插件实例。"""
        super().__init__()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._active_invocations: WeakSet[NaiInvocation] = WeakSet()

    async def on_load(self) -> None:
        """处理插件加载。"""
        self._refresh_runtime_singletons()

    async def on_unload(self) -> None:
        """处理插件卸载。"""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for invocation in list(self._active_invocations):
            invocation.close()
        reset_runtime_recall_tracking_state()
        self._refresh_runtime_singletons(reset_only=True)

    async def on_config_update(
        self,
        scope: str | dict[str, object],
        config_data: dict[str, object] | str | None = None,
        version: str = "",
    ) -> None:
        """处理配置热更新。

        兼容两种调用形式：
        1. 新版 Runner：``on_config_update(scope, config_data, version)``
        2. 旧版 SDK：``on_config_update(config_data, version)``
        """
        if isinstance(scope, dict):
            _scope = "self"
            _config_data = scope
            _version = str(config_data or version or "")
        else:
            _scope = scope
            _config_data = config_data if isinstance(config_data, dict) else {}
            _version = version

        del _config_data
        del _version

        if _scope == "self":
            self._refresh_runtime_singletons()

    def _refresh_runtime_singletons(self, *, reset_only: bool = False) -> None:
        """刷新插件级单例缓存，保证配置热更新后新调用使用最新参数。"""
        online_retriever_api = _load_online_retriever_api()
        reset_tag_retriever()
        if online_retriever_api is not None:
            _, reset_online_retriever = online_retriever_api
            reset_online_retriever()
        if reset_only:
            return

        plugin_config = self.get_plugin_config_data()
        tag_retriever_config = plugin_config.get("tag_retriever")
        if not isinstance(tag_retriever_config, dict):
            return
        if not tag_retriever_config.get("enabled", False):
            return

        mode = str(tag_retriever_config.get("mode", "local") or "local").strip().lower()
        if mode == "online":
            if online_retriever_api is None:
                return
            get_online_retriever, _ = online_retriever_api
            get_online_retriever(
                enabled=True,
                base_url=tag_retriever_config.get("api_url", "https://sakizuki-danboorusearch.hf.space/api"),
                timeout=tag_retriever_config.get("timeout", 90.0),
                search_limit=tag_retriever_config.get("search_limit", 30),
                search_top_k=tag_retriever_config.get("search_top_k", 5),
                related_limit=tag_retriever_config.get("related_limit", 20),
                related_seed_count=tag_retriever_config.get("related_seed_count", 8),
                show_nsfw=tag_retriever_config.get("show_nsfw", True),
                popularity_weight=tag_retriever_config.get("popularity_weight", 0.15),
            )
            return

        get_tag_retriever(
            enabled=True,
            top_k=tag_retriever_config.get("top_k", 50),
            min_score=tag_retriever_config.get("min_score", 0.6),
        )

    def _track_task(self, task: asyncio.Task[Any]) -> None:
        """跟踪后台任务，便于插件卸载时统一清理。"""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _run_invocation_in_background(
        self,
        coroutine: asyncio.Future[Any] | asyncio.Task[Any] | Any,
    ) -> None:
        """在后台执行一次耗时调用，避免命令 / 工具 RPC 超时。"""

        async def _runner() -> None:
            try:
                await coroutine
            except Exception:
                # 具体报错已经在 invocation 内部记录，这里只兜底避免任务未处理异常。
                return

        self._track_task(asyncio.create_task(_runner()))

    @HookHandler(
        "send_service.after_build_message",
        name="nai_draw_plugin_mark_recall_image",
        description="为本插件图片消息补充撤回标记",
    )
    async def handle_send_service_after_build_message(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在消息发送前写入撤回识别标记。"""
        if not isinstance(message, dict):
            return {"action": "continue"}

        if not attach_plugin_image_marker_to_message(message, NAI_PIC_IMAGE_DISPLAY_MARKER):
            return {"action": "continue"}

        updated_kwargs = dict(kwargs)
        updated_kwargs["message"] = message
        return {"action": "continue", "modified_kwargs": updated_kwargs}

    @HookHandler(
        "send_service.after_send",
        name="nai_draw_plugin_track_recall_image",
        description="记录本插件已成功发送的图片消息ID",
        mode=HookMode.OBSERVE,
    )
    async def handle_send_service_after_send(
        self,
        message: dict[str, Any] | None = None,
        sent: bool = False,
        **kwargs: Any,
    ) -> None:
        """在消息成功发送后记录可撤回的最终消息 ID。"""
        del kwargs

        if not sent or not isinstance(message, dict):
            return None

        remember_sent_plugin_image_message(message, NAI_PIC_IMAGE_DISPLAY_MARKER)
        return None

    def _is_image_generation_pending(self, stream_id: str) -> bool:
        """检查当前会话是否已有进行中的图片任务。"""
        return bool(stream_id and session_state.get_pending_image_generation_started_at(stream_id) is not None)

    def _start_image_generation_in_background(
        self,
        stream_id: str,
        coroutine_factory: Any,
    ) -> bool:
        """在后台启动图片生成任务，并阻止同会话重复启动。"""
        if not stream_id:
            self._run_invocation_in_background(coroutine_factory())
            return True

        if self._is_image_generation_pending(stream_id):
            return False

        session_state.set_pending_image_generation(stream_id)

        async def _runner() -> None:
            try:
                await coroutine_factory()
            except Exception:
                return
            finally:
                session_state.clear_pending_image_generation(stream_id)

        self._track_task(asyncio.create_task(_runner()))
        return True

    async def _start_command_image_generation(
        self,
        stream_id: str,
        coroutine_factory: Any,
    ) -> bool:
        """后台执行显式生图命令，允许同会话内并发处理多个用户请求。"""
        self._run_invocation_in_background(coroutine_factory())

        if stream_id:
            await self.ctx.send.text("收到，正在生成图片，请稍候...", stream_id, storage_message=False)
        return True

    def _load_local_plugin_config(self) -> dict[str, Any]:
        """回退读取当前插件目录下的 `config.toml`。"""
        plugin_file = inspect.getfile(self.__class__)
        config_path = os.path.join(os.path.dirname(plugin_file), "config.toml")
        if not os.path.isfile(config_path):
            return {}

        try:
            with open(config_path, "rb") as config_file:
                config_data = tomllib.load(config_file)
            return config_data if isinstance(config_data, dict) else {}
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    async def _load_plugin_config_data(self) -> dict[str, Any]:
        """优先读取宿主提供的插件配置，不存在时回退本地文件。"""
        local_config = self._load_local_plugin_config()
        runtime_config = await self.ctx.config.get_all()
        if not isinstance(runtime_config, dict):
            return local_config
        return _merge_config_dicts(local_config, runtime_config)

    async def _create_invocation(
        self,
        stream_id: str,
        *,
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        action_data: dict[str, Any] | None = None,
        reasoning: str = "",
        text: str = "",
        source: str = "command",
    ) -> NaiInvocation:
        """构造一次命令或 Action 调用的运行上下文。"""
        plugin_config = await self._load_plugin_config_data()
        invocation = NaiInvocation(
            self,
            plugin_config,
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            action_data=action_data,
            reasoning=reasoning,
            text=text,
            source=source,
        )
        self._active_invocations.add(invocation)
        return invocation

    @Command(
        "nai_admin_control_command",
        description="NAI 管理命令：/nai <st|sp|set|art|size|ban|unban|banlist|help>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+(?P<action>st|sp|set|art|size|ban|unban|banlist|help)(?:\s+(?P<param>.+))?$",
    )
    async def handle_nai_admin_control_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai st|sp|set|art|size|help`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        action = str((matched_groups or {}).get("action", "") or "").strip()
        param = str((matched_groups or {}).get("param", "") or "").strip()
        return await invocation.handle_admin_command(action, param)

    @Command(
        "nai_recall_control_command",
        description="NAI 自动撤回控制命令：/nai <on|off>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+(?P<action>on|off)$",
    )
    async def handle_nai_recall_control_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai on|off`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        return await invocation.handle_recall_switch(action)

    @Command(
        "nai_nsfw_control_command",
        description="NSFW 内容过滤控制命令：/nai nsfw <on|off>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+nsfw(?:\s+(?P<action>on|off))?$",
    )
    async def handle_nai_nsfw_control_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai nsfw`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        return await invocation.handle_nsfw_command(action)

    @Command(
        "nai_manual_recall_command",
        description="手动撤回图片：/nai 撤回",
        pattern=r"^(?:.*?)(?:/nai\s+撤回)(?:\s+.*)?$",
    )
    async def handle_nai_manual_recall_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai 撤回`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        return await invocation.manual_recall()

    @Command(
        "nai_draw",
        description="使用自然语言描述生成图片",
        pattern=r"^(?:.*，说：\s*)?/nai\s+(?!on$|off$|st$|sp$|set\b|art\b|artgen\b|artr$|artfix\b|size\b|ban\b|unban\b|banlist\b|help\b|pt\s|nsfw\b|撤回(?:\s|$))(?P<description>[\s\S]+)$",
    )
    async def handle_nai_draw(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        text: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            text=text,
        )
        description = str((matched_groups or {}).get("description", "") or "").strip()
        if not await invocation.ensure_generation_permission():
            return False, "没有权限", True
        if not await self._start_command_image_generation(
            stream_id,
            lambda: invocation.handle_nai_draw(description),
        ):
            return False, "", True
        return True, "已开始生成图片", True

    @Command(
        "nai_0_draw",
        description="直接使用英文标签生成图片",
        pattern=r"^(?:.*，说：\s*)?/nai0\s+(?P<tags>[\s\S]+)$",
    )
    async def handle_nai_0_draw(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        text: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai0`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            text=text,
        )
        tags = str((matched_groups or {}).get("tags", "") or "").strip()
        if not await invocation.ensure_generation_permission():
            return False, "没有权限", True
        if not await self._start_command_image_generation(
            stream_id,
            lambda: invocation.handle_nai0_draw(tags),
        ):
            return False, "", True
        return True, "已开始生成图片", True

    @Command(
        "nai_prompt_show_command",
        description="NAI 提示词显示控制命令：/nai pt <on|off>",
        pattern=r"^(?:.*，说：\s*)?/nai\s+pt\s+(?P<action>on|off)$",
    )
    async def handle_nai_prompt_show_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai pt on|off`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        return await invocation.handle_prompt_show_command(action)

    @Action(
        "nai_web_draw",
        description=(
            "生成图片/照片/自拍/场景图。"
            "可以根据语境发送 bot 本人的自拍、非自拍肖像照，或符合对话场景的图片。"
        ),
        activation_type=ActivationType.ALWAYS,
        parallel_action=True,
        action_parameters={
            "description": (
                "【必须】先输出人数（如'一女''一男一女''两女'），再输出画面关键词。"
                "关键词用空格分隔，只输出有视觉意义的核心词，禁止输出完整句子和虚词。"
                "【意图对齐规则】description 必须优先符合用户这轮要求和当前上下文连续意图；"
                "如果用户要看的是自拍、本人照片、穿搭、状态、环境、工作现场或上一张图的延续，就按那个意图写，"
                "不要擅自改成别的题材、别的关注点、别的穿搭主题。"
                "服装、场景、构图可以补全，但必须服务于用户当前想看的重点；"
                "即使用户没有明确写出衣服，也应该根据场景主动补出合理的服装款式、颜色、必要时的材质/质感细节，"
                "让 description 对后续 tag 检索足够具体，而不是只写空泛的'衣服''穿搭''常服'。"
                "但这种补全必须自然贴合场景与人物状态，不要每次都固定成同一种穿搭。"
                "如果上下文是在延续上一张图，就优先延续同一人物状态、场景和视觉重点，而不是突然换成固定套路。"
                "尤其是连续发图时，若用户没有明确要求换衣服、换颜色、换材质、换风格，就默认沿用上一张的服装款式、主色和材质，不要自己把白衣换成黑衣。"
                "【图片类型规则】先判断这轮更适合 自拍、非自拍肖像/生活照，还是普通画图。"
                "只有用户明确想要自拍、镜拍、前置、拍给他看时，才必须包含'自拍'二字；"
                "如果只是想看bot本人的样子、穿搭、状态，也可以输出不带'自拍'的肖像照/生活照描述；"
                "如果重点是当前环境、手头在做的事、某个视觉场景，也可以输出更合适的场景图描述。"
                "例如：用户说'发张自拍'→'一女 自拍'；"
                "用户说'想看看你长什么样'→'一女 肖像照 正脸 近景'；"
                "用户说'看看你穿黑丝的样子'→'一女 全身 黑丝 室内生活照'；"
                "用户只说'看看你现在的样子'→ 可以补成'一女 室内生活照 靠窗 慵懒 近景'这类更具体但符合场景的描述；"
                "用户说'画初音未来穿泳装'→'一女 初音未来 泳装 海边'（这是画图，不是自拍）。"
            ),
            "size": "图片尺寸（默认从配置获取）",
        },
        action_require=[
            "满足以下任一条件时触发：",
            "1. 用户明确要求看图/画图/发图/自拍/肖像/再来一张",
            "2. 用户明确想看 bot 本人的样子、穿搭、状态、某个身体/服饰视觉重点",
            "3. bot 自己在角色扮演里产生了强视觉语境（描述自身当前穿搭、所在场景、刚做的事、手头的画面），"
            "用配图比纯文字更自然时，可以主动发一张",
            "不触发：知识问答、技术讨论、对图片的评价或询问、单纯活跃气氛、撩拨",
            "频率约束：主动发图比用户显式要求更克制；刚发过图就不要再主动发，除非用户明确继续追图",
        ],
        associated_types=["text"],
    )
    async def handle_nai_web_draw(
        self,
        stream_id: str = "",
        action_data: dict[str, Any] | None = None,
        reasoning: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """处理自动生图 Action。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            action_data=action_data,
            reasoning=reasoning,
            source="action",
        )
        if not await invocation.ensure_user_not_blacklisted():
            return False, "黑名单用户"
        if not self._start_image_generation_in_background(stream_id, invocation.handle_action):
            return False, ""
        return True, "已开始生成图片"


def create_plugin():
    """创建新版 SDK 插件实例。"""
    return NaiPicPlugin()
