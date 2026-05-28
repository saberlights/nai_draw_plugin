from typing import Any, List
from weakref import WeakSet

import asyncio
import inspect
import os
import re
import tomllib

import tomlkit

from maibot_sdk import Action, Command, HookHandler, MaiBotPlugin
from maibot_sdk.types import ActivationType, HookMode, HookOrder

from src.core.config_types import ConfigField

from .core.constants import NAI_PIC_IMAGE_DISPLAY_MARKER
from .core.retag import ImageCacheService, ReverseService, WD14Client
from .core.rules.reply_auto_draw import (
    compose_description_from_reply,
    score_reply_for_auto_draw,
)
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


_CONFIG_VALUE_MISSING = object()


def _resolve_existing_config_value(
    existing_doc: Any,
    section: str,
    field: str,
    default: Any,
) -> Any:
    """读 existing_doc 里的字段值，缺则用 default。

    existing_doc 可能是 tomlkit 的 Document/Table，也可能是普通 dict；都用 ``get``
    访问。tomlkit 包装过的值通过 ``unwrap()`` 还原成 Python 原生类型，避免重写时
    把内部对象写进新文档。
    """
    if existing_doc is None:
        return default
    section_value: Any
    try:
        section_value = existing_doc.get(section, _CONFIG_VALUE_MISSING)
    except Exception:
        return default
    if section_value is _CONFIG_VALUE_MISSING:
        return default
    try:
        raw = section_value.get(field, _CONFIG_VALUE_MISSING)
    except Exception:
        return default
    if raw is _CONFIG_VALUE_MISSING:
        return default
    return raw.unwrap() if hasattr(raw, "unwrap") else raw


def _dump_scalar_kv(key: str, value: Any) -> str:
    """用 tomlkit 序列化单个 key=value 行，确保字符串转义、数字格式等正确。"""
    import tomlkit as _tomlkit
    try:
        snippet = _tomlkit.dumps({key: value}).rstrip("\n")
    except Exception:
        # 兜底：value 不被 tomlkit 接受时，转字符串重试
        snippet = _tomlkit.dumps({key: str(value)}).rstrip("\n")
    return snippet


def _is_array_of_tables(value: Any) -> bool:
    """判断 list 是否为'数组表'（list of dict）形态，需要渲染成 [[..]] 块。"""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, dict) for item in value)
    )


def _render_subtable(qualified_name: str, value: dict[str, Any]) -> str:
    """渲染 [section.sub] 子表。嵌套 dict 递归处理，scalar 先输出。"""
    if not isinstance(value, dict):
        return ""
    lines: list[str] = [f"[{qualified_name}]"]
    scalar_items: list[tuple[str, Any]] = []
    nested_dicts: list[tuple[str, dict]] = []
    nested_aots: list[tuple[str, list]] = []
    for k, v in value.items():
        if isinstance(v, dict):
            nested_dicts.append((k, v))
        elif _is_array_of_tables(v):
            nested_aots.append((k, v))
        else:
            scalar_items.append((k, v))
    for k, v in scalar_items:
        lines.append(_dump_scalar_kv(k, v))
    for k, v in nested_dicts:
        lines.append("")
        lines.append(_render_subtable(f"{qualified_name}.{k}", v))
    for k, v in nested_aots:
        lines.append("")
        lines.append(_render_array_of_tables(f"{qualified_name}.{k}", v))
    return "\n".join(lines)


def _render_array_of_tables(qualified_name: str, items: list[Any]) -> str:
    """渲染 [[section.field]] 数组表。每个元素是 dict。"""
    blocks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        block_lines: list[str] = [f"[[{qualified_name}]]"]
        for k, v in item.items():
            if isinstance(v, dict):
                block_lines.append("")
                block_lines.append(_render_subtable(f"{qualified_name}.{k}", v))
            elif _is_array_of_tables(v):
                block_lines.append("")
                block_lines.append(_render_array_of_tables(f"{qualified_name}.{k}", v))
            else:
                block_lines.append(_dump_scalar_kv(k, v))
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def _render_section_with_comments(
    *,
    section_name: str,
    fields: dict[str, Any],
    section_desc: Any,
    existing_doc: Any,
) -> str:
    """按 schema 顺序渲染一个 section：scalar 字段优先（带注释），dict / 数组表在末尾。"""
    lines: list[str] = []
    section_desc_text = section_desc.strip() if isinstance(section_desc, str) else ""
    if section_desc_text:
        lines.append(f"# {section_desc_text}")
    lines.append(f"[{section_name}]")

    scalar_fields: list[tuple[str, ConfigField, Any]] = []
    dict_fields: list[tuple[str, ConfigField, dict]] = []
    aot_fields: list[tuple[str, ConfigField, list]] = []

    for field_name, field_def in fields.items():
        if not isinstance(field_def, ConfigField):
            continue
        value = _resolve_existing_config_value(
            existing_doc, section_name, field_name, field_def.default
        )
        if isinstance(value, dict):
            dict_fields.append((field_name, field_def, value))
        elif _is_array_of_tables(value):
            aot_fields.append((field_name, field_def, value))
        else:
            scalar_fields.append((field_name, field_def, value))

    for fname, fdef, fvalue in scalar_fields:
        desc = (fdef.description or "").strip()
        if desc:
            lines.append(f"# {desc}")
        lines.append(_dump_scalar_kv(fname, fvalue))

    for fname, fdef, fvalue in dict_fields:
        desc = (fdef.description or "").strip()
        lines.append("")
        if desc:
            lines.append(f"# {desc}")
        lines.append(_render_subtable(f"{section_name}.{fname}", fvalue))

    for fname, fdef, fvalue in aot_fields:
        desc = (fdef.description or "").strip()
        lines.append("")
        if desc:
            lines.append(f"# {desc}")
        lines.append(_render_array_of_tables(f"{section_name}.{fname}", fvalue))

    return "\n".join(lines)


def _format_comment_block(text: str) -> str:
    """把一段可能多行的字符串渲染成 ``# ...`` 注释块；空行渲染为单独的 ``#``。

    传入文本里以 ``#`` 开头的行原样保留（允许在 group header 里手写 ``# ----- xxx -----``
    这种已经带 ``#`` 的样式，但当前调用方都没这么写）。
    """
    if not isinstance(text, str):
        return ""
    rendered: list[str] = []
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if not stripped:
            rendered.append("#")
        elif stripped.lstrip().startswith("#"):
            rendered.append(stripped)
        else:
            rendered.append(f"# {stripped}")
    return "\n".join(rendered)


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
    plugin_version = "1.8.0"
    plugin_author = "saberlight"
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[str] = ["httpx", "requests"]
    config_file_name = "config.toml"

    # 配置文件顶部说明，渲染时挂在所有 section 之前（写 config.toml 时按行加 # 前缀）。
    config_file_header = (
        "nai_draw_plugin - 配置文件\n"
        "与 nai_pic_plugin 共享同一套业务逻辑，底层请求改为 NewAPI 兼容 OpenAI 协议\n"
        "（POST /v1/chat/completions，绘图参数以 JSON 字符串塞入 messages[0].content）。\n"
        "支持 NAI 格式提示词（大括号权重），仅支持文生图。\n"
        "\n"
        "建议按这个顺序改：\n"
        "1. [plugin] 是否启用插件\n"
        "2. [model] NewAPI 地址 / 密钥 / 默认生图模型\n"
        "3. [prompt_generator] 提示词生成模型\n"
        "4. [model_nai4_5] 当前默认模型（V4.5）的专属参数\n"
        "5. 其他功能按需开启"
    )

    # section 渲染顺序；schema 字典本身的顺序与历史代码相关，渲染另走这套清单，
    # 保证配置文件读起来从'要先改的'到'通常不动的'。未列出的 section 走 schema 字典原顺序。
    config_section_order = [
        "plugin",
        "model",
        "prompt_generator",
        "action_guard",
        "auto_draw_on_reply",
        "random_scene",
        "components",
        "prompt_show",
        "nsfw_filter",
        "auto_recall",
        "admin",
        "tag_retriever",
        "retag",
        "custom_prompt",
        "model_nai4_5",
        "model_nai4",
        "model_nai3",
    ]

    # 大段分隔符；key 是 section 名，value 是渲染在该 section 之前的多行注释块
    # （每行自动加 # 前缀，空行渲染为 #）。仅在该 section 处开启一个新组，组内
    # 其它 section 直接跟在后面，不再插入分隔符。
    config_section_group_headers = {
        "plugin": "========== 基础开关 ==========",
        "model": "========== NewAPI 兼容网关连接与默认模型 ==========",
        "prompt_generator": "========== 提示词生成（/nai） ==========",
        "action_guard": "========== 自动出图触发保护 ==========",
        "random_scene": "========== 随机场景生成（/nai 随机） ==========\n未配置的项会回退到 [prompt_generator]",
        "components": "========== 功能开关 ==========",
        "retag": (
            "========== 图片反推（/nai 反推） ==========\n"
            "PNG 元数据可命中 → 直接读 prompt；不可命中 → 用 WD14 在线 Space 兜底（需安装 gradio_client）。\n"
            "只输出正向 prompt，不返回负面。"
        ),
        "custom_prompt": (
            "========== 自定义系统提示词 ==========\n"
            "这段通常不需要频繁修改；保留在文件末尾，避免影响日常配置体验。"
        ),
        "model_nai4_5": (
            "========== 生图模型专属配置 ==========\n"
            "下面三段会按当前模型自动选用。\n"
            "你当前默认模型是 V4.5，所以优先看 [model_nai4_5]。\n"
            "\n"
            "----- NAI V4.5（当前默认模型） -----"
        ),
        "model_nai4": "----- NAI V4 -----",
        "model_nai3": "----- NAI V3 / V3 Furry -----",
    }

    # 配置节描述（兼容老逻辑用，新渲染不会再把它单独输出为 section 上方注释；
    # 仅为 schema 内联 dict 字段做兜底，避免删后老代码崩）。
    config_section_descriptions: dict[str, str] = {}

    # 不渲染到 config.toml 的字段（schema 仍保留以便高级用户手动覆盖；默认值在代码层走兜底）。
    # 结构：{section_name: {field_name, ...}}
    config_hidden_fields: dict[str, set[str]] = {
        # WD14 Space 列表用户基本改不动（要清楚 type/api 协议）；默认 3 个 Space 内置在
        # WD14Client.DEFAULT_SPACES，留空配置即用默认，碍眼又易写错故不渲染。
        "retag": {"wd14_spaces"},
    }

    # 配置Schema
    config_schema = {
        "plugin": {
            "name": ConfigField(
                type=str,
                default="nai_draw_plugin",
                description="插件标识；可填任意字符串，通常不需要修改",
                required=True
            ),
            "config_version": ConfigField(
                type=str,
                default="1.6.0",
                description="插件配置版本号；由插件自行维护，请勿手动修改"
            ),
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用插件；可填 true / false"
            )
        },
        "model": {
            "name": ConfigField(
                type=str,
                default="NovelAI NewAPI Gateway",
                description="网关显示名称；可填任意字符串，仅用于日志/展示"
            ),
            "base_url": ConfigField(
                type=str,
                default="https://api.tuercha.com",
                description="NewAPI 兼容网关基础地址；可填 https://xxx 格式 URL，必填，由服务提供方给出",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="",
                description="NewAPI 鉴权密钥；可填以 sk- 开头的 OpenAI 风格 Bearer Token，由服务提供方给出",
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
                description="可用模型列表；填字符串数组，每项需与服务方 /v1/models 返回的 id 一致，供 /nai set 切换"
            ),
            "default_model": ConfigField(
                type=str,
                default="nai-diffusion-4-5-full",
                description="默认生图模型；可填 available_models 中任意一项，作为新会话的初始模型"
            ),
            "nai_request_timeout": ConfigField(
                type=float,
                default=600.0,
                description="生图请求超时；单位秒，可填正数；建议 300~600 以容忍长尾排队"
            ),
            "nai_proxy_mode": ConfigField(
                type=str,
                default="auto",
                description="代理模式；可填 auto / inherit / direct：auto=先继承环境代理，失败回退直连；inherit=始终继承；direct=始终直连"
            ),
            "nai_max_tokens": ConfigField(
                type=int,
                default=100000,
                description="单次绘图 token 预算；可填正整数，1 Anlas = 10000 tokens；常用 100000(=10 Anlas)，超出网关返回 400"
            ),
        },
        "model_nai3": {
            "artist_presets": ConfigField(
                type=list,
                default=[
                    {"name": "示例风格1", "prompt": "artist:example1, artist:example2, year 2023"},
                    {"name": "示例风格2", "prompt": "artist:example3, artist:example4, year 2024"}
                ],
                description="画师预设；结构同 model_nai4_5.artist_presets"
            ),
            "default_artist_preset": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.default_artist_preset"
            ),
            "nai_artist_prompt": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.nai_artist_prompt"
            ),
            "nai_size": ConfigField(
                type=str,
                default="832x1216",
                description="作用同 model_nai4_5.nai_size（V3 默认尺寸）"
            ),
            "sampler": ConfigField(
                type=str,
                default="k_euler_ancestral",
                description="作用同 model_nai4_5.sampler"
            ),
            "num_inference_steps": ConfigField(
                type=int,
                default=25,
                description="作用同 model_nai4_5.num_inference_steps"
            ),
            "guidance_scale": ConfigField(
                type=float,
                default=3.5,
                description="作用同 model_nai4_5.guidance_scale"
            ),
            "seed": ConfigField(
                type=int,
                default=-1,
                description="作用同 model_nai4_5.seed"
            ),
            "quality_toggle": ConfigField(
                type=bool,
                default=True,
                description="作用同 model_nai4_5.quality_toggle"
            ),
            "auto_smea": ConfigField(
                type=bool,
                default=False,
                description="作用同 model_nai4_5.auto_smea"
            ),
            "variety_boost": ConfigField(
                type=bool,
                default=False,
                description="作用同 model_nai4_5.variety_boost"
            ),
            "cfg_rescale": ConfigField(
                type=float,
                default=0.0,
                description="作用同 model_nai4_5.cfg_rescale"
            ),
            "noise_schedule": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.noise_schedule"
            ),
            "image_format": ConfigField(
                type=str,
                default="png",
                description="作用同 model_nai4_5.image_format"
            ),
            "default_size": ConfigField(
                type=str,
                default="832x1216",
                description="作用同 model_nai4_5.default_size"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.custom_prompt_add"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.negative_prompt_add"
            ),
            "selfie_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.selfie_prompt_add"
            ),
            "selfie_negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.selfie_negative_prompt_add"
            ),
            "nai_extra_params": ConfigField(
                type=dict,
                default={},
                description="作用同 model_nai4_5.nai_extra_params"
            )
        },
        "model_nai4": {
            "artist_presets": ConfigField(
                type=list,
                default=[
                    {"name": "风格组合1", "prompt": "1.2::artist1::, 1.0::artist2::, 0.9::artist3::"},
                    {"name": "风格组合2", "prompt": "1.5::artist4::, 1.0::artist5::, 0.8::artist6::"}
                ],
                description="画师预设；结构同 model_nai4_5.artist_presets"
            ),
            "default_artist_preset": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.default_artist_preset"
            ),
            "nai_artist_prompt": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.nai_artist_prompt"
            ),
            "nai_size": ConfigField(
                type=str,
                default="竖图",
                description="作用同 model_nai4_5.nai_size"
            ),
            "sampler": ConfigField(
                type=str,
                default="k_euler_ancestral",
                description="作用同 model_nai4_5.sampler"
            ),
            "num_inference_steps": ConfigField(
                type=int,
                default=28,
                description="作用同 model_nai4_5.num_inference_steps"
            ),
            "guidance_scale": ConfigField(
                type=float,
                default=5.0,
                description="作用同 model_nai4_5.guidance_scale"
            ),
            "seed": ConfigField(
                type=int,
                default=-1,
                description="作用同 model_nai4_5.seed"
            ),
            "quality_toggle": ConfigField(
                type=bool,
                default=True,
                description="作用同 model_nai4_5.quality_toggle"
            ),
            "auto_smea": ConfigField(
                type=bool,
                default=False,
                description="作用同 model_nai4_5.auto_smea"
            ),
            "variety_boost": ConfigField(
                type=bool,
                default=False,
                description="作用同 model_nai4_5.variety_boost"
            ),
            "cfg_rescale": ConfigField(
                type=float,
                default=0.0,
                description="作用同 model_nai4_5.cfg_rescale"
            ),
            "noise_schedule": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.noise_schedule"
            ),
            "image_format": ConfigField(
                type=str,
                default="png",
                description="作用同 model_nai4_5.image_format"
            ),
            "default_size": ConfigField(
                type=str,
                default="832x1216",
                description="作用同 model_nai4_5.default_size"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.custom_prompt_add"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.negative_prompt_add"
            ),
            "selfie_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.selfie_prompt_add"
            ),
            "selfie_negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="作用同 model_nai4_5.selfie_negative_prompt_add"
            ),
            "nai_extra_params": ConfigField(
                type=dict,
                default={},
                description="作用同 model_nai4_5.nai_extra_params"
            )
        },
        "model_nai4_5": {
            "artist_presets": ConfigField(
                type=list,
                default=[
                    {"name": "风格示例1", "prompt": "1.2::artist:example1::, 1.0::artist:example2::, 0.8::artist:example3::"},
                    {"name": "风格示例2", "prompt": "1.5::artist:example4::, 1.3::artist:example5::"}
                ],
                description="画师预设列表；每项含 name / prompt，可选 negative_prompt_add；通过 /nai art <名称或序号> 切换"
            ),
            "default_artist_preset": ConfigField(
                type=str,
                default="",
                description="默认画师预设；可填预设名称或序号（从 1 开始），留空时使用第一个预设"
            ),
            "nai_artist_prompt": ConfigField(
                type=str,
                default="",
                description="直接写死的画师串；可填英文 prompt 片段，仅在不用 artist_presets 时设置"
            ),
            "nai_size": ConfigField(
                type=str,
                default="竖图",
                description="图片尺寸；可填 竖图 / 横图 / 方图（或别名 v/h/s、portrait/landscape/square），也可直接写 832x1216 / 1216x832 / 1024x1024；请求时自动转成 [宽,高] 整数数组"
            ),
            "sampler": ConfigField(
                type=str,
                default="k_euler_ancestral",
                description="采样器；可填 k_euler / k_euler_ancestral / k_dpm_2 / k_dpm_2_ancestral / k_dpmpp_2m / k_dpmpp_2s_ancestral / k_dpmpp_sde / ddim；常用 k_euler_ancestral"
            ),
            "num_inference_steps": ConfigField(
                type=int,
                default=28,
                description="去噪步数；可填 1~28 的整数（NewAPI §5 上限）；越高细节越多但也更慢、更耗 anlas"
            ),
            "guidance_scale": ConfigField(
                type=float,
                default=5.0,
                description="提示词跟随强度；可填正浮点数，常用 5.0；越高越听 prompt，也越容易僵硬"
            ),
            "seed": ConfigField(
                type=int,
                default=-1,
                description="随机种子；可填整数固定结果，填 -1 表示由 NewAPI 随机"
            ),
            "quality_toggle": ConfigField(
                type=bool,
                default=True,
                description="质量增强；可填 true / false；开启后追加 NovelAI 的 quality 通路"
            ),
            "auto_smea": ConfigField(
                type=bool,
                default=False,
                description="底层 SMEA 类增强；可填 true / false"
            ),
            "variety_boost": ConfigField(
                type=bool,
                default=False,
                description="多样性增强（NewAPI §5 variety_boost）；可填 true / false；开启后画面构图/姿势更随机"
            ),
            "cfg_rescale": ConfigField(
                type=float,
                default=0.0,
                description="Prompt Guidance Rescale（NewAPI §5 cfg_rescale）；可填 0~1 的数；0 或留空表示不发送让网关用默认；典型值 0.5"
            ),
            "noise_schedule": ConfigField(
                type=str,
                default="",
                description="噪声调度算法（NewAPI §5/§9 noise_schedule）；可填 karras / exponential / polyexponential；留空表示不发送让网关用默认"
            ),
            "image_format": ConfigField(
                type=str,
                default="png",
                description="返回图片格式；可填 png / webp"
            ),
            "default_size": ConfigField(
                type=str,
                default="832x1216",
                description="兜底尺寸；当 nai_size 为空或无法解析时使用；可填 832x1216 / 1216x832 / 1024x1024"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="固定追加到正向提示词；可填英文 prompt 片段；通常放质量词、风格词、通用修饰词"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="固定追加到负面提示词；可填英文 prompt 片段；用于压低坏手、多人乱入、水印等问题"
            ),
            "selfie_prompt_add": ConfigField(
                type=str,
                default="",
                description="自拍模式额外正向外貌词；可填英文 prompt 片段；命中 selfie 时拼到正向"
            ),
            "selfie_negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="自拍模式额外负向外貌词；可填英文 prompt 片段；命中 selfie 时拼在 negative_prompt_add 之前，优先级更高"
            ),
            "nai_extra_params": ConfigField(
                type=dict,
                default={},
                description="额外透传到 NewAPI 内层 draw_params 的字段；可填 {key=value} 表；文档 §5 之外的字段不保证被识别，按服务方说明使用"
            )
        },
        "components": {
            "enable_debug_info": ConfigField(
                type=bool,
                default=False,
                description="是否输出调试日志；可填 true / false"
            ),
        },
        "auto_recall": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用自动撤回；可填 true / false；运行时可用 /nai on|off 切换"
            ),
            "delay_seconds": ConfigField(
                type=int,
                default=5,
                description="自动撤回延迟时间；单位秒，可填正整数"
            ),
            "id_wait_seconds": ConfigField(
                type=int,
                default=15,
                description="等待正式消息 ID 的最长时间；单位秒，可填正整数；超出后改用本地消息 ID 兜底"
            ),
            "manual_max_age_seconds": ConfigField(
                type=int,
                default=3600,
                description="手动撤回允许命中的最老图片年龄；单位秒，可填正整数；超出视为不可撤回，避免反复命中老图"
            ),
            "allowed_groups": ConfigField(
                type=list,
                default=[],
                description="自动撤回会话白名单；填 platform:chat_id 字符串数组，留空数组表示所有会话都允许"
            )
        },
        "admin": {
            "admin_users": ConfigField(
                type=list,
                default=[],
                description="管理员用户 ID 列表；填字符串数组（含纯数字 ID 也用字符串包），管理员可用 /nai st/sp 控制管理员模式"
            ),
            "default_admin_mode": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用管理员模式；可填 true / false；开启后仅 admin_users 中的用户可使用 /nai 生图命令"
            )
        },
        "prompt_show": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用提示词显示；可填 true / false；运行时可用 /nai pt on|off 切换"
            ),
            "hide_selfie_prompt_add": ConfigField(
                type=bool,
                default=False,
                description="提示词显示时是否隐藏 selfie_prompt_add；可填 true / false；仅影响展示，不影响实际生图"
            )
        },
        "nsfw_filter": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否默认启用 NSFW 内容过滤；可填 true / false；运行时可用 /nai nsfw on|off 切换"
            ),
            "filter_tags": ConfigField(
                type=str,
                default="{{{{{nsfw}}}}}",
                description="NSFW 过滤标签；可填英文 prompt 片段（建议高权重大括号）；启用过滤时自动追加到负面提示词最前"
            )
        },
        "prompt_generator": {
            "model_name": ConfigField(
                type=str,
                default="",
                description="提示词生成使用的 LLM 模型代号；可填 model_config 中已定义的代号，留空则自动选择 planner/replyer"
            ),
            "output_format": ConfigField(
                type=str,
                default="json",
                description="提示词生成输出格式；可填 json / text；json 支持多人分段与意图元数据，text 为纯提示词"
            ),
            "selfie_appearance_policy": ConfigField(
                type=str,
                default="auto",
                description="自拍外貌标签策略；可填 auto / never / keep；auto=仅在用户未指定外貌时移除 LLM 随机外貌；never=始终移除（除非用户指定）；keep=不移除"
            ),
            "enforce_tag_order": ConfigField(
                type=bool,
                default=False,
                description="是否对最终提示词做轻量排序；可填 true / false；开启后人数/视角前置、year 后置，降低顺序混乱"
            ),
            "temperature": ConfigField(
                type=float,
                default=0.2,
                description="提示词生成 LLM 温度；可填正浮点数；常用 0.2~1.5，越高越发散"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=500,
                description="提示词生成 LLM 响应的最大 token；可填正整数"
            ),
            "prompt_template": ConfigField(
                type=str,
                default="",
                description="自定义提示词生成模板；可填多行字符串，支持占位符 <<USER_REQUEST>> / <<SELFIE_HINT>> / <<CURRENT_TIME_CONTEXT>> / <<SELFIE_SCENE_CONTEXT>>；留空使用内置模板"
            ),
            "inherit_ttl": ConfigField(
                type=int,
                default=3600,
                description="上一轮提示词继承的有效时间；单位秒，可填正整数；默认 3600（1 小时），0 表示永不过期"
            ),
            "custom_model": ConfigField(
                type=dict,
                default={
                    "model_list": [],
                    "max_tokens": 500,
                    "temperature": 0.2,
                    "slow_threshold": 30.0
                },
                description="自定义模型配置；填 {model_list, max_tokens, temperature, slow_threshold}；model_list 中的模型名必须在系统 model_config 中已定义；留空表示使用上面的 model_name"
            )
        },
        "action_guard": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用 nai_web_draw Action 的触发保护；可填 true / false；含否定意图兜底与频率分级保护"
            ),
            "explicit_request_min_interval_seconds": ConfigField(
                type=int,
                default=5,
                description="用户原话含明确画图/自拍/肖像/追图等强信号时的最小间隔；单位秒，可填正整数；默认 5 秒仅防同秒重复触发"
            ),
            "proactive_min_interval_seconds": ConfigField(
                type=int,
                default=10,
                description="bot 主动判断要发图时的最小间隔；单位秒，可填正整数；默认 10 秒，给 Planner 两轮 reasoning 之间一点缓冲"
            ),
            "weak_negative_ttl_seconds": ConfigField(
                type=int,
                default=60,
                description="弱否定关键词拦截的时效；单位秒，可填正整数；超过此秒数视为 stale，不再拦截"
            ),
            "proactive_self_image_boost": ConfigField(
                type=bool,
                default=True,
                description="主动出图自动注入自拍/肖像标签；可填 true / false；命中 proactive 且描述不含自拍/肖像关键词时启用"
            ),
        },
        "auto_draw_on_reply": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="reply 后置自动跟图开关；可填 true / false；开启后 bot 写出的 reply 命中视觉自指/情感节点时自动跟一张图"
            ),
            "score_threshold": ConfigField(
                type=float,
                default=0.6,
                description="reply 评分阈值；可填 0.0~1.0 的浮点数；评分 ≥ 阈值才触发跟图，越高越保守"
            ),
            "min_interval_seconds": ConfigField(
                type=int,
                default=15,
                description="reply 自动跟图的最小间隔；单位秒，可填正整数；与显式出图独立计时，关键词召回噪音大故略高于 explicit/proactive"
            ),
            "self_image_boost": ConfigField(
                type=bool,
                default=True,
                description="跟图自动注入自拍/肖像标签；可填 true / false；不含自拍/肖像关键词时启用"
            ),
        },
        "random_scene": {
            "temperature": ConfigField(
                type=float,
                default=1.0,
                description="随机场景生成 LLM 温度；可填正浮点数；常用 1.0~1.5，越高越发散"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=200,
                description="随机场景生成 LLM 响应的最大 token；可填正整数"
            ),
            "custom_model": ConfigField(
                type=dict,
                default={
                    "model_list": [],
                    "max_tokens": 200,
                    "temperature": 1.0,
                    "slow_threshold": 30.0
                },
                description="随机场景自定义模型配置；填 {model_list, max_tokens, temperature, slow_threshold}；留空则继承 prompt_generator.custom_model"
            ),
        },
        "custom_prompt": {
            "system_prompt": ConfigField(
                type=str,
                default="",
                description="自定义系统提示词；可填多行字符串；会拼到 LLM 提示词规则的最前面，用于自定义额外指导或规则"
            ),
        },
        "tag_retriever": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用 Danbooru Tag 检索增强；可填 true / false"
            ),
            "mode": ConfigField(
                type=str,
                default="online",
                description="检索模式；可填 online / local；online=远程 DanbooruSearchOnline API，local=本地 embedding（需 data/tag_embeddings.npy）"
            ),
            "api_url": ConfigField(
                type=str,
                default="https://sakizuki-danboorusearch.hf.space/api",
                description="DanbooruSearchOnline API 地址；可填完整 https:// URL"
            ),
            "timeout": ConfigField(
                type=float,
                default=90.0,
                description="在线检索请求超时；单位秒，可填正数"
            ),
            "search_limit": ConfigField(
                type=int,
                default=30,
                description="在线 /search 返回标签上限；可填正整数"
            ),
            "search_top_k": ConfigField(
                type=int,
                default=5,
                description="在线 /search 每个分词段召回数；可填正整数"
            ),
            "related_limit": ConfigField(
                type=int,
                default=20,
                description="在线 /related 返回推荐上限；可填正整数"
            ),
            "related_seed_count": ConfigField(
                type=int,
                default=8,
                description="在线共现推荐使用的种子标签数量；可填正整数"
            ),
            "show_nsfw": ConfigField(
                type=bool,
                default=True,
                description="在线检索是否允许返回 NSFW 标签；可填 true / false"
            ),
            "popularity_weight": ConfigField(
                type=float,
                default=0.15,
                description="在线检索标签热度权重；可填 0~1 的浮点数；越高越偏向热门 tag"
            ),
            "top_k": ConfigField(
                type=int,
                default=50,
                description="本地检索返回的候选 tag 数量；可填正整数（仅 mode=local 生效）"
            ),
            "min_score": ConfigField(
                type=float,
                default=0.6,
                description="本地检索最低相似度阈值；可填 0~1 的浮点数；低于此分数的不返回"
            ),
        },
        "retag": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用 /nai 反推 命令；可填 true / false；PNG 元数据命中走元数据，否则用 WD14 兜底，只输出正向 prompt"
            ),
            "cache_ttl_seconds": ConfigField(
                type=int,
                default=3600,
                description="入站图片缓存保留时间；单位秒，可填正整数；超过后即便回引也无法定位旧图"
            ),
            "image_cache_per_stream": ConfigField(
                type=int,
                default=20,
                description="每个会话保留的最近图片消息数量上限；可填正整数"
            ),
            "wd14_enabled": ConfigField(
                type=bool,
                default=True,
                description="非原图（无元数据）时是否调用 WD14 在线 Space 兜底；可填 true / false；需安装 gradio_client"
            ),
            "wd14_model": ConfigField(
                type=str,
                default="SmilingWolf/wd-eva02-large-tagger-v3",
                description="WD14 模型名；可填 Hugging Face 模型 ID；仅 official 类 Space 生效，其它 Space 走各自固定模型"
            ),
            "wd14_threshold": ConfigField(
                type=float,
                default=0.35,
                description="通用标签置信度阈值；可填 0~1 的浮点数；越高越保守"
            ),
            "wd14_character_threshold": ConfigField(
                type=float,
                default=0.8,
                description="角色标签置信度阈值；可填 0~1 的浮点数；越高越保守"
            ),
            "wd14_request_timeout": ConfigField(
                type=float,
                default=35.0,
                description="单个 Space 请求超时；单位秒，可填正数；1~2MB 大图实测需 16~23s，留点余量"
            ),
            "wd14_max_retries": ConfigField(
                type=int,
                default=1,
                description="单个 Space 失败时的重试次数；可填非负整数"
            ),
            "wd14_retry_delay": ConfigField(
                type=float,
                default=0.5,
                description="单个 Space 重试间隔；单位秒，可填非负数"
            ),
            "wd14_proxy": ConfigField(
                type=str,
                default="",
                description="访问 Hugging Face Space 时使用的代理 URL；可填 http://host:port 或留空；留空则继承 HTTPS_PROXY 环境变量"
            ),
            "wd14_spaces": ConfigField(
                type=list,
                default=[
                    {
                        "name": "animetimm/dbv4-full-witha-playground",
                        "type": "danbooru_v4",
                        "api": "/_fn_submit",
                    },
                    {
                        "name": "pixai-labs/pixai-tagger-demo",
                        "type": "pixai",
                        "api": "/predict_image",
                    },
                    {
                        "name": "DraconicDragon/PixAI-Tagger-v0.9-ONNX",
                        "type": "pixai_onnx",
                        "api": "/run_inference",
                    },
                ],
                description="可并发轮询的 HF Space 列表；填 [{name, type, api}] 数组；name 是 HF Space 全名，type 决定 payload 结构，api 是 Space 入口"
            ),
        },
    }

    def get_default_config(self) -> dict[str, Any]:
        """从 ``config_schema`` 推导默认配置，供 MaiBot Runner 首次启动时自动生成 config.toml。

        MaiBotPlugin SDK 默认通过 ``get_config_model()`` 拼默认配置，但本插件仍走旧版
        ``config_schema`` 字典风格，因此手动遍历一次，避免 Runner 因为 ``default_config``
        为空而跳过 config.toml 初始化。

        ``config_hidden_fields`` 中声明的字段不会写入默认配置，避免 Runner 把它们 dump 到
        首次生成的 config.toml；运行时这些字段仍可被用户手动添加并被代码读取。
        """
        hidden_map = getattr(self, "config_hidden_fields", None) or {}
        default_config: dict[str, Any] = {}
        for section_name, fields in type(self).config_schema.items():
            if not isinstance(fields, dict):
                continue
            hidden = hidden_map.get(section_name) or set()
            section: dict[str, Any] = {}
            for field_name, field in fields.items():
                if field_name in hidden:
                    continue
                if hasattr(field, "default"):
                    section[field_name] = field.default
            if section:
                default_config[section_name] = section
        return default_config

    def __init__(self) -> None:
        """初始化插件实例。"""
        super().__init__()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._active_invocations: WeakSet[NaiInvocation] = WeakSet()
        # reply 自动跟图：同一 session 在同一 reply 链路里只触发一次，避免 retry 重复出图。
        # key=session_id, value=已触发的 reply 文本哈希集合
        self._auto_draw_fired_signatures: dict[str, set[str]] = {}
        # 反推链路：图片缓存与编排服务都在 __init__ 阶段就准备好，避免 HookHandler 在配置加载前触发时 NoneError
        self._image_cache_service: ImageCacheService = ImageCacheService()
        self._reverse_service: ReverseService = ReverseService(wd14_client=None)

    async def on_load(self) -> None:
        """处理插件加载。"""
        self._refresh_runtime_singletons()
        self._refresh_retag_runtime()
        # 主程序 _save_plugin_config 在整文件重写时不会把 ConfigField.description 渲染成注释。
        # 在 on_load 兜底回填一次，保留用户已写入的值，仅在文件里完全没有注释时触发，
        # 避免覆盖用户手写注释。
        try:
            self._regenerate_config_with_comments_if_needed()
        except Exception as exc:  # noqa: BLE001
            from src.common.logger import get_logger
            get_logger("nai_draw_plugin").debug(f"config 注释回填失败（已忽略）：{exc!r}")

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
        self._image_cache_service.clear()
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
            self._refresh_retag_runtime()

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

    def _refresh_retag_runtime(self) -> None:
        """刷新反推链路的运行时单例（图缓存 TTL、WD14 客户端）。"""
        plugin_config = self.get_plugin_config_data()
        retag_config = plugin_config.get("retag") if isinstance(plugin_config, dict) else None
        if not isinstance(retag_config, dict):
            retag_config = {}

        self._image_cache_service.update_config(
            cache_ttl_seconds=float(retag_config.get("cache_ttl_seconds", 3600) or 3600),
            per_stream_capacity=int(retag_config.get("image_cache_per_stream", 20) or 20),
        )

        wd14_enabled = bool(retag_config.get("wd14_enabled", True))
        wd14_threshold = float(retag_config.get("wd14_threshold", 0.35) or 0.35)
        wd14_character_threshold = float(retag_config.get("wd14_character_threshold", 0.8) or 0.8)

        if wd14_enabled:
            spaces_raw = retag_config.get("wd14_spaces")
            spaces_config: list[dict[str, str]] = []
            if isinstance(spaces_raw, list):
                for item in spaces_raw:
                    if isinstance(item, dict) and item.get("name") and item.get("type") and item.get("api"):
                        spaces_config.append(
                            {
                                "name": str(item["name"]),
                                "type": str(item["type"]),
                                "api": str(item["api"]),
                            }
                        )
            wd14_client = WD14Client(
                model=str(retag_config.get("wd14_model", "SmilingWolf/wd-eva02-large-tagger-v3")),
                timeout=float(retag_config.get("wd14_request_timeout", 20.0) or 20.0),
                max_retries=int(retag_config.get("wd14_max_retries", 1) or 1),
                retry_delay=float(retag_config.get("wd14_retry_delay", 0.5) or 0.5),
                spaces_config=spaces_config or None,
                proxy=str(retag_config.get("wd14_proxy", "") or "").strip() or None,
            )
        else:
            wd14_client = None

        self._reverse_service.update_wd14_client(wd14_client)
        self._reverse_service.update_wd14_thresholds(
            threshold=wd14_threshold,
            character_threshold=wd14_character_threshold,
            enabled=wd14_enabled,
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

    @HookHandler(
        "maisaka.replyer.after_response",
        name="nai_draw_plugin_auto_draw_on_reply",
        description="bot reply 命中视觉自指/情感节点时自动跟一张图",
        mode=HookMode.OBSERVE,
    )
    async def handle_replyer_after_response_for_auto_draw(
        self,
        session_id: str = "",
        response: str = "",
        attempt: int = 1,
        **kwargs: Any,
    ) -> None:
        """OBSERVE 模式：reply 文本生成成功时旁路评分，命中阈值就启动后台跟图。"""
        del kwargs
        # 主程序 LLM retry 时本 hook 会被反复触发（attempt>=2 表示当前是 retry 后的版本）；
        # 中间被丢弃的版本不应该启动跟图，否则会污染签名集合并浪费一次评分。
        if attempt > 1:
            return

        normalized_session = (session_id or "").strip()
        reply_text = (response or "").strip()
        if not normalized_session or not reply_text:
            return

        # 读插件配置：未开启就不做
        try:
            plugin_config = await self._load_plugin_config_data()
        except Exception:
            return
        auto_cfg = plugin_config.get("auto_draw_on_reply") if isinstance(plugin_config, dict) else None
        if not isinstance(auto_cfg, dict) or not auto_cfg.get("enabled", True):
            return

        threshold = float(auto_cfg.get("score_threshold", 0.6) or 0.6)
        signal = score_reply_for_auto_draw(reply_text)
        if signal.score < threshold or not signal.should_draw:
            return

        # 同一 session 同一 reply 文本只触发一次（防止 retry 流程重复出图）
        signature = f"{len(reply_text)}:{hash(reply_text) & 0xFFFFFFFF:08x}"
        fired = self._auto_draw_fired_signatures.setdefault(normalized_session, set())
        if signature in fired:
            return
        fired.add(signature)
        # 简单 LRU：每个 session 最多记 16 条最近触发签名，避免无界增长
        if len(fired) > 16:
            self._auto_draw_fired_signatures[normalized_session] = set(list(fired)[-16:])

        description = compose_description_from_reply(reply_text, signal)
        if not description:
            return

        invocation = await self._create_invocation(
            normalized_session,
            action_data={"description": description},
            source="reply_auto_draw",
        )

        async def _runner() -> None:
            try:
                await invocation.handle_auto_draw_from_reply(
                    description,
                    reply_context_text=reply_text,
                )
            except Exception:
                pass

        # 走通用后台启动：同 session 已有生成任务则丢弃这次跟图（避免叠加）
        self._start_image_generation_in_background(normalized_session, _runner)

    @HookHandler(
        "chat.receive.before_process",
        name="nai_draw_plugin_retag_receive_image_cache",
        description="缓存入站图片消息，供 /nai 反推 解析引用回复",
        order=HookOrder.EARLY,
    )
    async def handle_retag_receive_before_process(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """监听所有入站消息，把带图的存到 ImageCacheService。"""
        del kwargs
        if isinstance(message, dict):
            self._image_cache_service.cache_inbound_message(message)
        return {"action": "continue"}

    @HookHandler(
        "chat.command.before_execute",
        name="nai_draw_plugin_retag_command_message_cache",
        description="在需要引用图的命令（反推 / i2i / vibe存 / ref存）执行前缓存当前命令消息（保留 reply 信息）",
        order=HookOrder.EARLY,
    )
    async def handle_retag_command_before_execute(
        self,
        message: dict[str, Any] | None = None,
        command_name: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """仅在需要引用图的命令触发前生效，其它命令直接放行。

        /nai vibe 与 /nai ref 已迁移到命名图库，不再走引用图，所以从这个集合里拿掉了。"""
        del kwargs
        if command_name in {
            "nai_retag_command",
            "nai_i2i_command",
            "nai_vibe_save_command",
            "nai_ref_save_command",
        } and isinstance(message, dict):
            self._image_cache_service.remember_command_message(message)
        return {"action": "continue"}

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

    async def _run_retag(self, *, stream_id: str, user_id: str) -> tuple[bool, str | None, bool]:
        """执行 `/nai 反推`：取目标图 → 反推 → 把结果发回会话。"""
        plugin_config = self.get_plugin_config_data()
        retag_config = plugin_config.get("retag") if isinstance(plugin_config, dict) else None
        if not isinstance(retag_config, dict) or not retag_config.get("enabled", True):
            await self.ctx.send.text("❌ /nai 反推 已在配置中关闭", stream_id, storage_message=False)
            return False, "反推未启用", True

        image_base64 = self._image_cache_service.resolve_image_base64(
            stream_id=stream_id,
            user_id=user_id,
        )
        if not image_base64:
            await self.ctx.send.text(
                "❌ 未找到图片\n请引用回复一张图后发送 /nai 反推，或在同一条消息内发图加命令",
                stream_id,
                storage_message=False,
            )
            return False, "未找到图片", True

        try:
            import base64 as _base64
            payload = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
            image_bytes = _base64.b64decode(payload)
        except Exception as exc:
            await self.ctx.send.text(f"❌ 图片解码失败: {exc}", stream_id, storage_message=False)
            return False, "图片解码失败", True

        await self.ctx.send.text("🔍 正在反推 tag，请稍候...", stream_id, storage_message=False)

        result = await self._reverse_service.reverse(image_bytes)
        if result.source == "failed" or not result.prompt:
            await self.ctx.send.text(
                "❌ 反推失败：" + (result.detail or "未知原因") + "\n（仅 PNG 元数据命中或 WD14 可用时才能拿到 tag）",
                stream_id,
                storage_message=False,
            )
            return False, "反推失败", True

        source_label = {
            "metadata": "📦 PNG 元数据",
            "wd14": "🔍 WD14 在线 Space",
        }.get(result.source, result.source)

        await self.ctx.send.text(
            f"✅ 反推完成（{source_label}，{len(result.tags)} 个 tag）\n\n{result.prompt}\n\n💡 可直接用于 /nai0 <prompt>",
            stream_id,
        )
        return True, "反推成功", True

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

    def _regenerate_config_with_comments_if_needed(self) -> None:
        """把 `ConfigField.description` 渲染成 config.toml 顶部的 `#` 注释。

        触发条件（保守，避免覆盖用户手写注释）：
        - config.toml 存在
        - 文件里目前一条 `#` 注释行都没有

        策略：保留用户已设置的值，按 ``config_schema`` 顺序重写文件，每个字段上方挂
        一行描述。主程序 ``_save_plugin_config`` 增量合并时会保留这些注释；只有完整
        重写（用户删除文件、版本号 bump 触发 rebuild 等）才会再次清空，此时下次
        ``on_load`` 会再回填一次。

        注意：用 tomlkit 直接构造文档时，array-of-tables（如 ``artist_presets``）
        会被强制放到 section 末尾，导致紧跟其后的 scalar 字段注释顺序错乱。改成
        手写 TOML：标量字段先输出（带注释），dict / array-of-tables 在 section 末尾，
        子表/数组本身的注释贴在它们前面。
        """
        plugin_file = inspect.getfile(self.__class__)
        config_path = os.path.join(os.path.dirname(plugin_file), "config.toml")
        if not os.path.isfile(config_path):
            return

        try:
            existing_text = open(config_path, "r", encoding="utf-8").read()
        except OSError:
            return

        if any(line.lstrip().startswith("#") for line in existing_text.splitlines()):
            return  # 已经有注释，留给用户

        try:
            existing_doc = tomlkit.parse(existing_text)
        except Exception:
            return

        new_text = self._compose_commented_config_text(existing_doc)
        if not new_text or new_text == existing_text:
            return

        try:
            with open(config_path, "w", encoding="utf-8") as fp:
                fp.write(new_text)
        except OSError:
            return

    def _compose_commented_config_text(self, existing_doc: Any) -> str:
        """按 schema 顺序手写 TOML，保留用户已有值，给每个字段挂 description 注释。

        渲染骨架：
        1. ``config_file_header``                   → 整个文件顶部说明（多行 # 注释）
        2. ``config_section_group_headers[section]``→ 渲染在某 section 之前的大段分隔符
        3. 每个 section 的字段                       → 上方挂 ``# {description}``，下面是 ``key = value``

        section 顺序：先按 ``config_section_order`` 出现的顺序渲染；剩下的 schema
        section 走字典原顺序兜底；最后是 existing_doc 里 schema 外的自定义 section。
        """
        schema = getattr(self, "config_schema", None) or {}
        section_descs = getattr(self, "config_section_descriptions", None) or {}
        hidden_map = getattr(self, "config_hidden_fields", None) or {}
        if not isinstance(schema, dict) or not schema:
            return ""

        group_headers = getattr(self, "config_section_group_headers", None) or {}
        order = getattr(self, "config_section_order", None) or []

        # 按 config_section_order 先走一遍，再把 schema 里剩下的补在后面，避免漏掉新增字段
        ordered: list[str] = []
        seen_in_order: set[str] = set()
        for name in order:
            if name in schema and isinstance(schema[name], dict) and name not in seen_in_order:
                ordered.append(name)
                seen_in_order.add(name)
        for name in schema:
            if name in seen_in_order or not isinstance(schema[name], dict):
                continue
            ordered.append(name)
            seen_in_order.add(name)

        blocks: list[str] = []

        # 顶部文件说明
        file_header = getattr(self, "config_file_header", "") or ""
        header_text = _format_comment_block(str(file_header)).strip()
        if header_text:
            blocks.append(header_text)

        seen_sections: set[str] = set()
        for section_name in ordered:
            fields = schema.get(section_name)
            if not isinstance(fields, dict):
                continue
            seen_sections.add(section_name)
            hidden = hidden_map.get(section_name) or set()
            visible_fields = {
                fname: fdef for fname, fdef in fields.items() if fname not in hidden
            }
            if not visible_fields:
                continue
            group_header_text = group_headers.get(section_name)
            if isinstance(group_header_text, str) and group_header_text.strip():
                blocks.append(_format_comment_block(group_header_text))
            blocks.append(
                _render_section_with_comments(
                    section_name=section_name,
                    fields=visible_fields,
                    section_desc=section_descs.get(section_name),
                    existing_doc=existing_doc,
                )
            )

        # 未在 schema 中的 section 直接搬过来，避免误删用户自定义节
        if hasattr(existing_doc, "items"):
            for name, value in existing_doc.items():
                if name in seen_sections:
                    continue
                try:
                    tmp = tomlkit.document()
                    tmp.add(name, value)
                    snippet = tomlkit.dumps(tmp).strip()
                    if snippet:
                        blocks.append(snippet)
                except Exception:
                    continue

        return "\n\n".join(s for s in blocks if s).rstrip() + "\n"

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
        "nai_retag_command",
        description="图片反推：/nai 反推（PNG 元数据 → WD14 兜底，只输出正向 prompt）",
        pattern=r"^(?:.*?)(?:/nai\s+反推)(?:\s+.*)?$",
    )
    async def handle_nai_retag_command(
        self,
        stream_id: str = "",
        user_id: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai 反推`。

        反推链路全部走插件内单例，命令本身不接 Invocation。
        """
        del kwargs
        return await self._run_retag(stream_id=stream_id, user_id=user_id)

    @Command(
        "nai_draw",
        description="使用自然语言描述生成图片",
        # negative lookahead 排除所有 /nai 子命令；vibe/ref 后面可接 CJK 后缀（存/图库/删/选），
        # 所以用 ``(?:\b|[一-鿿])`` 覆盖空格后置和中文后缀两种情形，避免 ``vibe存`` 被
        # 通用命令吞掉（vibe\b 在 latin→CJK 边界不成立）
        pattern=r"^(?:.*，说：\s*)?/nai\s+(?!on$|off$|st$|sp$|set\b|art\b|artgen\b|artr$|artfix\b|size\b|ban\b|unban\b|banlist\b|help\b|pt\s|nsfw\b|models$|i2i\b|ref(?:\b|[一-鿿])|vibe(?:\b|[一-鿿])|撤回(?:\s|$)|反推(?:\s|$))(?P<description>[\s\S]+)$",
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

    @Command(
        "nai_models_command",
        description="拉取 NewAPI 网关实时可用模型列表：/nai models",
        pattern=r"^(?:.*，说：\s*)?/nai\s+models$",
    )
    async def handle_nai_models_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai models`。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        return await invocation.handle_models_command()

    @Command(
        "nai_i2i_command",
        description="图生图：/nai i2i <描述>（需引用一张图）",
        # 宽松前缀：/nai i2i 总伴随"回复一张图"链路，各平台的 reply 前缀形态不一，
        # 沿用 /nai 反推 / /nai 撤回 的 (?:.*?) 起手而不是严格的 (?:.*，说：\s*)?，
        # 否则带 reply 前缀的消息匹不上、用户看到"没反应"
        pattern=r"^(?:.*?)/nai\s+i2i\s+(?P<description>[\s\S]+)$",
    )
    async def handle_nai_i2i_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai i2i <描述>`：取引用图执行 NewAPI §20.1 i2i 图生图。"""
        del kwargs
        return await self._run_image_to_image_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            mode="i2i",
        )

    @Command(
        "nai_ref_command",
        description="角色参考：/nai ref [@<名字>] <描述>（用图库里的角色参考图，仅 V4.5 模型）",
        # 宽松前缀，同 nai_i2i_command 注释；可选 @<名字>... 单次覆盖，否则用 /nai ref选 的粘性选定
        # ref 最多 1 张：pattern 允许多个 @<名字> 透传，store 层做硬上限校验给统一错误提示
        pattern=r"^(?:.*?)/nai\s+ref\s+(?P<at_names>(?:@\S+\s+)*)(?P<description>[\s\S]+)$",
    )
    async def handle_nai_ref_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai ref [@<名字>] <描述>`：从角色参考图库取图执行 NewAPI §20.4。"""
        del kwargs
        return await self._run_named_reference_draw_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_vibe_command",
        description="Vibe Transfer：/nai vibe [@<名字1> [@<名字2>...]] <描述>（用图库里的 vibe 图，最多 4 张）",
        # 宽松前缀，同 nai_i2i_command 注释；可选 @<名字>... 单次覆盖，否则用 /nai vibe选 的粘性选定
        # at_names 用 (?:@\S+\s+)* 整体捕获 0~N 个 @ 前缀，命令层 re.findall 拆解；
        # vibe 最多 4 张走 store 层硬限制，超 4 走统一错误提示
        pattern=r"^(?:.*?)/nai\s+vibe\s+(?P<at_names>(?:@\S+\s+)*)(?P<description>[\s\S]+)$",
    )
    async def handle_nai_vibe_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """处理 `/nai vibe [@<名字>] <描述>`：从 vibe 图库取图执行 NewAPI §20.3。"""
        del kwargs
        return await self._run_named_reference_draw_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    # ── 命名图库：存 / 图库 / 删 / 选（vibe + ref 8 条对称命令） ──────────

    @Command(
        "nai_vibe_save_command",
        description="把引用回复的图存入 vibe 图库：/nai vibe存 <名字>",
        pattern=r"^(?:.*?)/nai\s+vibe存\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_vibe_save_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_save_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_vibe_list_command",
        description="列出 vibe 图库的所有命名图：/nai vibe图库",
        pattern=r"^(?:.*?)/nai\s+vibe图库\s*$",
    )
    async def handle_nai_vibe_list_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_list_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_vibe_delete_command",
        description="从 vibe 图库删除一张命名图：/nai vibe删 <名字>",
        pattern=r"^(?:.*?)/nai\s+vibe删\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_vibe_delete_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_delete_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_vibe_select_command",
        description="把本会话的默认 vibe 图设为 1~4 张命名图：/nai vibe选 <名字1> [<名字2>...]",
        # 1 ~ N 个名字，空格分隔；store 层会做 vibe ≤ 4 的硬限制
        pattern=r"^(?:.*?)/nai\s+vibe选\s+(?P<names>\S+(?:\s+\S+)*)\s*$",
    )
    async def handle_nai_vibe_select_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_select_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="vibe",
        )

    @Command(
        "nai_ref_save_command",
        description="把引用回复的图存入角色参考图库：/nai ref存 <名字>",
        pattern=r"^(?:.*?)/nai\s+ref存\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_ref_save_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_save_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_list_command",
        description="列出角色参考图库的所有命名图：/nai ref图库",
        pattern=r"^(?:.*?)/nai\s+ref图库\s*$",
    )
    async def handle_nai_ref_list_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_list_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_delete_command",
        description="从角色参考图库删除一张命名图：/nai ref删 <名字>",
        pattern=r"^(?:.*?)/nai\s+ref删\s+(?P<name>\S+)\s*$",
    )
    async def handle_nai_ref_delete_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_delete_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    @Command(
        "nai_ref_select_command",
        description="把本会话的默认角色参考图设为某张命名图：/nai ref选 <名字>",
        # ref 固定最多 1 张，pattern 与 vibe 选保持一致捕获 names 组；store 层若收到 >1 会拒
        pattern=r"^(?:.*?)/nai\s+ref选\s+(?P<names>\S+(?:\s+\S+)*)\s*$",
    )
    async def handle_nai_ref_select_command(
        self,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        del kwargs
        return await self._run_named_reference_select_command(
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
            scope="ref",
        )

    async def _run_image_to_image_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        mode: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai i2i 的引用图链路（ref 已迁移到命名图库，不再共享此路径）。"""
        description = str((matched_groups or {}).get("description", "") or "").strip()
        image_base64 = self._image_cache_service.resolve_image_base64(
            stream_id=stream_id,
            user_id=user_id,
        )
        if not image_base64:
            await self.ctx.send.text(
                "❌ 未找到参考图\n请引用回复一张图后再发送 /nai i2i，或在同一条消息内附图加命令",
                stream_id,
                storage_message=False,
            )
            return False, "未找到图片", True

        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        if not await invocation.ensure_generation_permission():
            return False, "没有权限", True

        if not await self._start_command_image_generation(
            stream_id,
            lambda: invocation.handle_image_to_image_draw(
                description, image_base64=image_base64, mode=mode
            ),
        ):
            return False, "", True
        return True, "已开始生成图片", True

    # ── 命名图库 helper（vibe / ref 共用骨架，scope 决定走哪个库） ──────

    async def _run_named_reference_draw_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe / /nai ref 共用：从图库取图（@<名字>... 或粘性选定），背后投递。

        命令 pattern 用 ``(?P<at_names>(?:@\\S+\\s+)*)`` 把 0~N 个 ``@<名字>`` 整体捕获，
        这里 ``re.findall`` 拆成 List[str] 透传给 invocation；空列表退化成 None 走粘性选定。
        """
        description = str((matched_groups or {}).get("description", "") or "").strip()
        at_names_str = str((matched_groups or {}).get("at_names", "") or "")
        explicit_names_list = re.findall(r"@(\S+)", at_names_str)
        explicit_names = explicit_names_list or None

        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        if not await invocation.ensure_generation_permission():
            return False, "没有权限", True

        if not await self._start_command_image_generation(
            stream_id,
            lambda: invocation.handle_named_reference_draw(
                scope=scope,
                description=description,
                explicit_names=explicit_names,
            ),
        ):
            return False, "", True
        return True, "已开始生成图片", True

    async def _run_named_reference_save_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe存 / /nai ref存：取引用图存入对应命名图库。"""
        name = str((matched_groups or {}).get("name", "") or "").strip()
        image_base64 = self._image_cache_service.resolve_image_base64(
            stream_id=stream_id,
            user_id=user_id,
        )
        if not image_base64:
            scope_cmd = "vibe存" if scope == "vibe" else "ref存"
            await self.ctx.send.text(
                f"❌ 未找到参考图\n请引用回复一张图后再发送 /nai {scope_cmd} <名字>，"
                "或在同一条消息内附图加命令",
                stream_id,
                storage_message=False,
            )
            return False, "未找到图片", True

        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        return await invocation.handle_named_reference_save(
            scope=scope, name=name, image_base64=image_base64
        )

    async def _run_named_reference_list_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe图库 / /nai ref图库。"""
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        return await invocation.handle_named_reference_list(scope=scope)

    async def _run_named_reference_delete_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe删 / /nai ref删。"""
        name = str((matched_groups or {}).get("name", "") or "").strip()
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        return await invocation.handle_named_reference_delete(scope=scope, name=name)

    async def _run_named_reference_select_command(
        self,
        *,
        stream_id: str,
        group_id: str,
        user_id: str,
        matched_groups: dict[str, str] | None,
        scope: str,
    ) -> tuple[bool, str | None, bool]:
        """/nai vibe选 / /nai ref选：把"空格分隔的多名字"拆成 List[str] 透给 invocation。

        vibe / ref 的 pattern 都用 ``(?P<names>\\S+(?:\\s+\\S+)*)`` 捕获 1~N 个 token，
        store 层会按 scope 的上限（vibe 4 / ref 1）做硬校验，错误统一冒泡。
        """
        names_str = str((matched_groups or {}).get("names", "") or "").strip()
        names = [token for token in names_str.split() if token]
        invocation = await self._create_invocation(
            stream_id,
            group_id=group_id,
            user_id=user_id,
            matched_groups=matched_groups,
        )
        return await invocation.handle_named_reference_select(scope=scope, names=names)

    @Action(
        "nai_web_draw",
        description=(
            "生成图片/照片/自拍/场景图。"
            "可以根据语境发送 bot 本人的自拍、非自拍肖像照，或符合对话场景的图片。"
            "既可以响应用户明确的看图请求，也可以在 bot 自己说出视觉自指/进入情感互动节点时主动跟一张图。"
            "【调用语义 - 重要】本 Action 是 fire-and-forget 异步任务："
            "调用成功只代表'图片任务已提交后台'，图片由插件自行通过会话发送，"
            "不会出现在本次 tool_result 的 content 里。"
            "因此：调用本 Action 后，禁止再调用 send_image / 引用本次 call_id 的 media_index，"
            "也禁止调用 wait 等待图片——图片到时会自行送达，按文字正常推进对话即可。"
        ),
        activation_type=ActivationType.ALWAYS,
        parallel_action=True,
        action_parameters={
            # 五个结构化字段：每个字段只承担一类信息，强制 Planner 分维度思考，
            # 避免一锅炖成关键词堆砌。下游会按字段顺序拼成单行 request；若 Planner
            # 兼容性原因只填了 description，则按整段兜底使用。
            "subject_and_pov": (
                "主体与视角，不写其它。"
                "格式：'一女' / '一男一女' / '两女'，可加视角：'POV' / '自拍' / '第三视角'。"
                "区分：对方看 bot 做事=POV；bot 自己举手机=自拍；旁观叙事=第三视角或留空。"
            ),
            "action": (
                "本轮核心动作，必须用用户原话/reasoning 里的动词，禁止软化。"
                "如'揉胸'写'揉胸'、不要写'轻捧'；'骑'写'骑乘'、不要写'坐在身上'。"
                "纯静态画面可留空或写'站立'。"
                "禁词：轻捧/触碰/贴近/迷离/陶醉/挑逗。"
            ),
            "emotion": (
                "情绪状态，必须贴 reasoning 里 bot 当前心境，不要默认套'迷离咬唇'。"
                "示例：'不情愿 害羞'、'撒娇 期待'、'紧张 微微低头'、'慵懒 半眯眼'。"
                "无明显情绪可留空。"
            ),
            "scene_delta": (
                "本轮相对上一张图新增/变化的场景或服装动作，没变化就留空。"
                "沿用元素（卧室/床上等）由系统自动继承，不要在这里重复。"
                "服装变化（脱/穿/掀）写这里；外貌锚点（长发/瞳色/choker）由配置注入，禁写。"
            ),
            "framing": (
                "构图镜头，1-2 个词："
                "近景/特写/全身/胸部以上/俯视/仰视/侧面/肖像照/生活照/pov_hands。"
                "默认不要每次写'近景'，按本轮重点选。"
            ),
            "description": (
                "兜底字段，正常留空。"
                "只有当本轮内容无法拆进上面 5 个字段时，才在这里写一行完整关键词串。"
                "格式：人数 + 视角 + 动作 + 情绪 + 场景 + 构图；禁写外貌锚点和画质词。"
            ),
            "size": "图片尺寸（默认从配置获取）",
        },
        action_require=[
            "可以触发的典型时机：",
            "1. 用户明确要求看图/画图/发图/自拍/肖像/再来一张",
            "2. 用户明确想看 bot 本人的样子、穿搭、状态、某个身体/服饰视觉重点",
            "3. bot 这一轮要回复的话里包含自身姿态、穿着、动作、所处场景的视觉描写"
            "（例：我刚洗完澡靠在窗边、今天穿了新裙子、在便利店买东西、慵懒地躺在床上）"
            "——这种时机配一张图比纯文字更自然，可以主动跟一张",
            "4. 用户分享情绪、晚安、回家了、到家了、想你了、撒娇等亲密互动节点，"
            "bot 想以一张近照/自拍作为情感回应",
            "不触发：纯知识问答、技术讨论、bot 这一轮明显走理性解释/代码/列点风格的回复，"
            "或者用户明确拒绝出图（'不要画''别画图'）",
            "节奏建议：自然搭图，不刷屏。短间隔内连发要克制；但 bot 自己开口提到视觉细节时不要犹豫——"
            "比起'刚发过图，先不发'，更应该判断'这句话本身配图是否自然'。",
        ],
        associated_types=["text"],
    )
    async def handle_nai_web_draw(
        self,
        stream_id: str = "",
        user_id: str = "",
        group_id: str = "",
        action_data: dict[str, Any] | None = None,
        reasoning: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """处理自动生图 Action。"""
        del kwargs
        invocation = await self._create_invocation(
            stream_id,
            user_id=user_id,
            group_id=group_id,
            action_data=action_data,
            reasoning=reasoning,
            source="action",
        )
        if not await invocation.ensure_user_not_blacklisted():
            return False, "黑名单用户"

        # Action Guard 同步预检：让 Planner 第一时间拿到拦截原因，避免后台默默吞掉
        # 评估结果会缓存到 invocation，后台 handle_action 复用同一次结论，不会重复读消息库
        guard_state = await invocation.preflight_action_guard()
        if guard_state is not None and not guard_state["should_generate"]:
            return False, guard_state["detail"]

        if not self._start_image_generation_in_background(stream_id, invocation.handle_action):
            return False, (
                "同会话已有图片任务在后台进行中，本轮跳过出图、按文字回复推进；"
                "请不要调用 send_image 或 wait，正在生成的那张图会自行送达"
            )
        return True, (
            "图片任务已提交后台，图片由插件异步发送到会话，本次 tool_result 不包含 image 内容；"
            "请不要调用 send_image 引用本次 call_id，也不要 wait，按文字正常推进对话即可"
        )


def create_plugin():
    """创建新版 SDK 插件实例。"""
    return NaiPicPlugin()
