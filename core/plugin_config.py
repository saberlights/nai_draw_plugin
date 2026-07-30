"""NAI 插件配置的 Schema、WebUI 映射与 TOML 迁移。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomlkit

from src.core.config_types import ConfigField


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
_WEBUI_TYPE_NAMES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
_WEBUI_TEXTAREA_FIELDS = {
    "custom_prompt_add",
    "negative_prompt_add",
    "selfie_prompt_add",
    "selfie_negative_prompt_add",
    "nai_artist_prompt",
    "filter_tags",
    "prompt_template",
    "system_prompt",
}
_WEBUI_PASSWORD_FIELDS = {"api_key"}
_WEBUI_SAMPLER_CHOICES = [
    "k_euler",
    "k_euler_ancestral",
    "k_dpm_2",
    "k_dpm_2_ancestral",
    "k_dpmpp_2m",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_sde",
    "ddim",
]
_WEBUI_CUSTOM_MODEL_FIELD_DESCRIPTIONS = {
    "model_list": "候选模型列表；填系统模型配置中已定义的模型名称",
    "max_tokens": "单次响应的最大 token 数；可填正整数",
    "temperature": "生成温度；可填非负浮点数，越高越发散",
    "slow_threshold": "慢请求判定阈值；单位秒，可填正数",
}
_WEBUI_SLIDER_0_1 = {
    "input_type": "slider",
    "ui_type": "slider",
    "min": 0.0,
    "max": 1.0,
    "step": 0.01,
}
_WEBUI_FIELD_OVERRIDES: dict[str, dict[str, Any]] = {
    "plugin.config_version": {"disabled": True},
    "model.nai_proxy_mode": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["direct", "inherit", "auto"],
    },
    "prompt_generator.output_format": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["json", "text"],
    },
    "prompt_generator.selfie_appearance_policy": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["auto", "never", "keep"],
    },
    "tag_retriever.mode": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["online", "local"],
    },
    "character_reference.type": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["character", "style", "character&style"],
    },
    "model_nai4_5.sampler": {
        "input_type": "select",
        "ui_type": "select",
        "choices": _WEBUI_SAMPLER_CHOICES,
    },
    "model_nai4.sampler": {
        "input_type": "select",
        "ui_type": "select",
        "choices": _WEBUI_SAMPLER_CHOICES,
    },
    "model_nai3.sampler": {
        "input_type": "select",
        "ui_type": "select",
        "choices": _WEBUI_SAMPLER_CHOICES,
    },
    "model_nai4_5.image_format": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["png", "webp"],
    },
    "model_nai4.image_format": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["png", "webp"],
    },
    "model_nai3.image_format": {
        "input_type": "select",
        "ui_type": "select",
        "choices": ["png", "webp"],
    },
    "tag_retriever.popularity_weight": _WEBUI_SLIDER_0_1,
    "tag_retriever.min_score": _WEBUI_SLIDER_0_1,
    "retag.wd14_threshold": _WEBUI_SLIDER_0_1,
    "retag.wd14_character_threshold": _WEBUI_SLIDER_0_1,
    "i2i.strength": {
        **_WEBUI_SLIDER_0_1,
        "min": 0.01,
        "max": 0.99,
    },
    "i2i.noise": {
        **_WEBUI_SLIDER_0_1,
        "max": 0.99,
    },
    "vibe.info_extracted": {
        **_WEBUI_SLIDER_0_1,
        "min": 0.01,
    },
    "vibe.reference_strength": {
        **_WEBUI_SLIDER_0_1,
        "min": 0.01,
    },
    "vibe.overall_strength": _WEBUI_SLIDER_0_1,
    "character_reference.fidelity": _WEBUI_SLIDER_0_1,
    "character_reference.strength": _WEBUI_SLIDER_0_1,
    "model_nai4_5.num_inference_steps": {
        "input_type": "slider",
        "ui_type": "slider",
        "min": 1,
        "max": 28,
        "step": 1,
    },
    "model_nai4.num_inference_steps": {
        "input_type": "slider",
        "ui_type": "slider",
        "min": 1,
        "max": 28,
        "step": 1,
    },
    "model_nai3.num_inference_steps": {
        "input_type": "slider",
        "ui_type": "slider",
        "min": 1,
        "max": 28,
        "step": 1,
    },
    "model_nai4_5.cfg_rescale": {
        **_WEBUI_SLIDER_0_1,
        "step": 0.05,
    },
    "model_nai4.cfg_rescale": {
        **_WEBUI_SLIDER_0_1,
        "step": 0.05,
    },
    "model_nai3.cfg_rescale": {
        **_WEBUI_SLIDER_0_1,
        "step": 0.05,
    },
}


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


def _webui_field_type(field_def: ConfigField) -> str:
    """把 ConfigField Python 类型转换成 WebUI 使用的规范类型名。"""
    return _WEBUI_TYPE_NAMES.get(field_def.type, "string")


def _webui_label(field_name: str, field_def: ConfigField) -> str:
    """从描述里提取短标签，避免 WebUI label 变成整段说明。"""
    if field_def.label:
        return field_def.label
    first_line = str(field_def.description or "").strip().splitlines()[0:1]
    if first_line:
        label = first_line[0].split("；", 1)[0].strip()
        if label:
            return label
    return field_name


def _webui_ui_type(field_name: str, field_def: ConfigField) -> str:
    """补齐 WebUI 控件类型；密钥和长 prompt 字段用更合适的控件。"""
    if field_def.input_type:
        return field_def.input_type
    if field_name.lower() in _WEBUI_PASSWORD_FIELDS:
        return "password"
    if field_def.type is str and (
        field_name in _WEBUI_TEXTAREA_FIELDS
        or "\n" in str(field_def.default or "")
        or len(str(field_def.default or "")) > 120
    ):
        return "textarea"
    return field_def.get_ui_type()


def _webui_item_schema_from_value(value: Any) -> dict[str, Any]:
    """为 list[dict] 的元素字段生成 WebUI schema。"""
    value_type = "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else "string"
    if isinstance(value, bool):
        value_type = "boolean"
    return {
        "type": value_type,
        "label": "",
        "default": value,
    }


def _webui_list_item_fields(field_name: str, default: Any) -> dict[str, Any] | None:
    """推断列表对象字段；显式补上常用可选字段，避免 WebUI 只能编辑默认样例。"""
    if field_name == "artist_presets":
        return {
            "name": {"type": "string", "label": "名称", "default": ""},
            "prompt": {"type": "string", "label": "正向提示词", "default": ""},
            "negative_prompt_add": {"type": "string", "label": "负向提示词", "default": ""},
        }
    if field_name == "wd14_spaces":
        return {
            "name": {"type": "string", "label": "Space", "default": ""},
            "type": {"type": "string", "label": "类型", "default": ""},
            "api": {"type": "string", "label": "API", "default": ""},
        }
    if isinstance(default, list) and default and isinstance(default[0], dict):
        return {
            str(key): {**_webui_item_schema_from_value(value), "label": str(key)}
            for key, value in default[0].items()
        }
    return None


def _webui_field_schema(
    section_name: str,
    field_name: str,
    field_def: ConfigField,
    *,
    hidden: bool,
    order: int,
) -> dict[str, Any]:
    """把单个 ConfigField 转换成 WebUI 字段 schema。"""
    field_schema = field_def.to_dict()
    ui_type = _webui_ui_type(field_name, field_def)
    field_schema.update(
        {
            "name": field_name,
            "type": _webui_field_type(field_def),
            "label": _webui_label(field_name, field_def),
            "hint": field_def.hint or field_def.description,
            "hidden": bool(hidden or field_def.hidden),
            "order": field_def.order or order,
            "input_type": field_def.input_type or ("password" if ui_type == "password" else None),
            "ui_type": ui_type,
            "rows": max(field_def.rows, 8) if ui_type == "textarea" else field_def.rows,
        }
    )
    if field_def.type is list:
        default = field_def.default
        item_fields = field_def.item_fields or _webui_list_item_fields(field_name, default)
        if item_fields is not None:
            field_schema["item_type"] = field_def.item_type or "object"
            field_schema["item_fields"] = item_fields
        elif field_def.item_type:
            field_schema["item_type"] = field_def.item_type
        elif isinstance(default, list) and default and isinstance(default[0], (int, float)):
            field_schema["item_type"] = "number"
        else:
            field_schema["item_type"] = "string"
    if field_def.type is int and field_schema.get("step") is None:
        field_schema["step"] = 1
    elif field_def.type is float and field_schema.get("step") is None:
        field_schema["step"] = 0.1
    field_schema.update(_WEBUI_FIELD_OVERRIDES.get(f"{section_name}.{field_name}", {}))
    field_schema.setdefault("depends_on", None)
    field_schema.setdefault("depends_value", None)
    field_schema["section"] = section_name
    return field_schema


def _webui_ordered_sections(schema: dict[str, Any], order: list[str]) -> list[str]:
    """按 config_section_order 输出 section，剩余 section 保持 schema 原顺序。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for section_name in order:
        if section_name in schema and isinstance(schema[section_name], dict):
            ordered.append(section_name)
            seen.add(section_name)
    for section_name in schema:
        if section_name not in seen and isinstance(schema[section_name], dict):
            ordered.append(section_name)
            seen.add(section_name)
    return ordered


def _webui_section_title(section_name: str, group_headers: dict[str, Any]) -> str:
    """从配置文件分组标题提取 WebUI section 标题。"""
    raw = group_headers.get(section_name)
    if isinstance(raw, str):
        for line in raw.splitlines():
            title = line.strip().strip("=- ")
            if title:
                return title
    return section_name


def _webui_section_description(section_name: str, group_headers: dict[str, Any]) -> str | None:
    """保留 TOML 分组标题中的说明行，丢弃只用于源码排版的装饰行。"""
    raw = group_headers.get(section_name)
    if not isinstance(raw, str):
        return None
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith(("=", "-"))
    ]
    return "\n".join(lines) or None


def _dump_scalar_kv(key: str, value: Any) -> str:
    """用 tomlkit 序列化单个 key=value 行，确保字符串转义、数字格式等正确。"""
    try:
        snippet = tomlkit.dumps({key: value}).rstrip("\n")
    except Exception:
        # 兜底：value 不被 tomlkit 接受时，转字符串重试
        snippet = tomlkit.dumps({key: str(value)}).rstrip("\n")
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


class PluginConfigDefinition:
    """以一个 Schema 驱动默认值、WebUI 和带注释 TOML 的深配置 Module。"""

    plugin_name = "nai_draw_plugin"
    plugin_version = "1.11.0"
    plugin_author = "saberlight"
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
        "random_scene",
        "components",
        "prompt_show",
        "nsfw_filter",
        "auto_recall",
        "admin",
        "tag_retriever",
        "retag",
        "i2i",
        "vibe",
        "character_reference",
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
        "random_scene": "========== 随机场景生成（/nai 随机 [角色]） ==========\n未配置的项会回退到 [prompt_generator]",
        "components": "========== 功能开关 ==========",
        "retag": (
            "========== 图片反推（/nai 反推） ==========\n"
            "PNG 元数据可命中 → 直接读 prompt；不可命中 → 用 WD14 在线 Space 兜底（需安装 gradio_client）。\n"
            "只输出正向 prompt，不返回负面。"
        ),
        "i2i": (
            "========== 图生图参数（/nai i2i / vibe / ref） ==========\n"
            "三段对应 NewAPI 文档 §20.1 / §20.3 / §20.4；不需要个性化时全部保留默认即可。\n"
            "\n"
            "----- i2i 图生图（§20.1） -----"
        ),
        "vibe": "----- Vibe Transfer（§20.3） -----",
        "character_reference": "----- 角色参考 / Character Reference（§20.4，仅 V4.5 系列） -----",
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

    # TOML 分组标题面向源码阅读；WebUI 使用独立标题与标签页，避免把装饰性标题
    # 直接当成导航文案，并保证当前默认的 V4.5 配置进入首屏。
    config_webui_section_titles = {
        "prompt_show": "提示词显示",
        "nsfw_filter": "NSFW 过滤",
        "auto_recall": "自动撤回",
        "admin": "管理员权限",
        "tag_retriever": "Danbooru Tag 检索",
        "model_nai4_5": "NAI V4.5 生图参数",
    }
    config_webui_tabs = [
        {
            "id": "generation",
            "title": "生图配置",
            "sections": [
                "model",
                "model_nai4_5",
                "model_nai4_5.artist_presets",
                "i2i",
                "vibe",
                "character_reference",
            ],
        },
        {
            "id": "prompting",
            "title": "提示词",
            "sections": [
                "prompt_generator",
                "prompt_generator.custom_model",
                "random_scene",
                "random_scene.custom_model",
                "custom_prompt",
            ],
        },
        {
            "id": "models",
            "title": "其他模型",
            "sections": [
                "model_nai4",
                "model_nai4.artist_presets",
                "model_nai3",
                "model_nai3.artist_presets",
            ],
        },
        {
            "id": "automation",
            "title": "自动化与权限",
            "sections": [
                "plugin",
                "action_guard",
                "components",
                "prompt_show",
                "nsfw_filter",
                "auto_recall",
                "admin",
            ],
        },
        {
            "id": "retrieval",
            "title": "检索与反推",
            "sections": ["tag_retriever", "retag"],
        },
    ]
    config_webui_nested_sections = {
        "prompt_generator.custom_model": {
            "source_section": "prompt_generator",
            "source_field": "custom_model",
            "title": "提示词生成自定义模型",
            "description": "覆盖提示词生成使用的模型列表与生成参数；模型名必须已在系统模型配置中定义。",
            "field_descriptions": _WEBUI_CUSTOM_MODEL_FIELD_DESCRIPTIONS,
        },
        "random_scene.custom_model": {
            "source_section": "random_scene",
            "source_field": "custom_model",
            "title": "随机场景自定义模型",
            "description": "覆盖随机场景生成使用的模型列表与生成参数；留空模型列表时继承提示词生成配置。",
            "field_descriptions": _WEBUI_CUSTOM_MODEL_FIELD_DESCRIPTIONS,
        },
    }
    config_webui_detached_field_sections = {
        "model_nai4_5.artist_presets": {
            "source_section": "model_nai4_5",
            "source_field": "artist_presets",
            "title": "NAI V4.5 画师预设",
        },
        "model_nai4.artist_presets": {
            "source_section": "model_nai4",
            "source_field": "artist_presets",
            "title": "NAI V4 画师预设",
        },
        "model_nai3.artist_presets": {
            "source_section": "model_nai3",
            "source_field": "artist_presets",
            "title": "NAI V3 / Furry 画师预设",
        },
    }

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
                default="1.7.0",
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
                default="direct",
                description="代理模式；可填 direct / inherit / auto：direct=始终直连；inherit=始终继承环境代理；auto=先直连，网络失败再继承环境代理"
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
                description="Bot 出镜时固定追加的外貌与身材正向词；可填英文 prompt 片段；配置名为历史兼容，不会强制自拍构图"
            ),
            "selfie_negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="Bot 出镜时固定追加的外貌负向词；可填英文 prompt 片段；拼在 negative_prompt_add 之前，优先级更高"
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
                description="提示词生成 LLM 温度；可填正浮点数；常用 0.2~1.5，越高越发散；Bot 情景连续性路径为保证稳定 Tag 确定性固定使用 0.2，不受本项影响"
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
                description="上一轮提示词继承（指定角色等 legacy 路径）与自拍上下文的有效时间；单位秒，可填正整数；默认 3600（1 小时），0 表示永不过期"
            ),
            "visual_state_ttl": ConfigField(
                type=int,
                default=0,
                description="Bot 情景连续性中当前服装/环境的沿用有效期；单位秒；0 表示不过期，直到聊天中明确更换；过期后仅当前装扮失效，服装/环境卡片库始终保留可供 switch 切回"
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
        "random_scene": {
            "temperature": ConfigField(
                type=float,
                default=1.2,
                description="随机场景生成 LLM 温度；可填正浮点数；常用 1.0~1.5，越高越发散"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=240,
                description="随机场景生成 LLM 响应的最大 token；可填正整数"
            ),
            "custom_model": ConfigField(
                type=dict,
                default={
                    "model_list": [],
                    "max_tokens": 240,
                    "temperature": 1.2,
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
            "show_result": ConfigField(
                type=bool,
                default=False,
                description="是否默认显示 Danbooru 检索结果；可填 true / false；运行时可用 /nai tag on|off 切换，仅影响回显"
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
                default=120.0,
                description="单个 Space 请求超时；单位秒，可填正数；冷启动后首次跑常需 30~90s，留余量到 120s"
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
        "i2i": {
            "strength": ConfigField(
                type=float,
                default=0.7,
                description="i2i 变换强度；可填 0.01~0.99 的浮点数；越小越像原图，缺省 0.7（NewAPI §20.1）"
            ),
            "noise": ConfigField(
                type=float,
                default=0.0,
                description="i2i 注入噪声量；可填 0.0~0.99 的浮点数；缺省 0.0（NewAPI §20.1）"
            ),
        },
        "vibe": {
            "info_extracted": ConfigField(
                type=float,
                default=0.7,
                description="每张 vibe 图的信息提取量；可填 0.01~1.0 的浮点数；缺省 0.7（NewAPI §20.3）"
            ),
            "reference_strength": ConfigField(
                type=float,
                default=0.6,
                description="每张 vibe 图的单独参考强度；可填 0.01~1.0 的浮点数；缺省 0.6（NewAPI §20.3）"
            ),
            "overall_strength": ConfigField(
                type=float,
                default=1.0,
                description="ControlNet 整体强度叠加系数；可填 0.0~1.0 的浮点数；缺省 1.0（NewAPI §20.3）"
            ),
        },
        "character_reference": {
            "type": ConfigField(
                type=str,
                default="character&style",
                description="角色参考提取目标；可填 character / style / character&style；缺省 character&style（NewAPI §20.4）"
            ),
            "fidelity": ConfigField(
                type=float,
                default=1.0,
                description="角色参考保真度（次要强度）；可填 0.0~1.0 的浮点数；缺省 1.0（NewAPI §20.4）"
            ),
            "strength": ConfigField(
                type=float,
                default=1.0,
                description="角色参考主参考强度；可填 0.0~1.0 的浮点数；缺省 1.0（NewAPI §20.4）"
            ),
        },
    }

    def default_config(self) -> dict[str, Any]:
        """从单一 Schema 推导 Runner 首次启动使用的默认配置。"""
        default_config: dict[str, Any] = {}
        for section_name, fields in self.config_schema.items():
            if not isinstance(fields, dict):
                continue
            hidden = self.config_hidden_fields.get(section_name) or set()
            section: dict[str, Any] = {}
            for field_name, field in fields.items():
                if field_name in hidden:
                    continue
                if isinstance(field, ConfigField):
                    section[field_name] = field.default
            if section:
                default_config[section_name] = section
        return default_config

    def webui_schema(
        self,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> dict[str, Any]:
        """把同一 Schema 映射为 WebUI 可渲染的配置描述。"""
        sections: dict[str, Any] = {}
        detached_fields = {
            (meta["source_section"], meta["source_field"])
            for meta in self.config_webui_detached_field_sections.values()
        }
        for section_index, section_name in enumerate(
            _webui_ordered_sections(self.config_schema, self.config_section_order)
        ):
            fields = self.config_schema.get(section_name)
            if not isinstance(fields, dict):
                continue
            hidden_fields = self.config_hidden_fields.get(section_name) or set()
            section_fields: dict[str, Any] = {}
            for field_index, (field_name, field_def) in enumerate(fields.items()):
                if not isinstance(field_def, ConfigField):
                    continue
                section_fields[field_name] = _webui_field_schema(
                    section_name,
                    field_name,
                    field_def,
                    # Dashboard 当前没有 object/json 编辑器；固定结构对象在下方展开成
                    # 点路径 section，任意对象保留给源码模式编辑。
                    hidden=(
                        field_name in hidden_fields
                        or field_def.type is dict
                        or (section_name, field_name) in detached_fields
                    ),
                    order=field_index,
                )
            sections[section_name] = {
                "name": section_name,
                "title": self.config_webui_section_titles.get(
                    section_name,
                    _webui_section_title(
                        section_name,
                        self.config_section_group_headers,
                    ),
                ),
                "description": _webui_section_description(
                    section_name,
                    self.config_section_group_headers,
                ),
                "icon": None,
                "collapsed": False,
                "order": section_index,
                "fields": section_fields,
            }

        for nested_index, (nested_name, nested_meta) in enumerate(
            self.config_webui_nested_sections.items(),
            start=len(sections),
        ):
            source_section = self.config_schema.get(nested_meta["source_section"])
            if not isinstance(source_section, dict):
                continue
            source_field = source_section.get(nested_meta["source_field"])
            if not isinstance(source_field, ConfigField) or not isinstance(
                source_field.default,
                dict,
            ):
                continue

            descriptions = nested_meta["field_descriptions"]
            nested_fields: dict[str, Any] = {}
            for field_index, (field_name, default) in enumerate(source_field.default.items()):
                description = descriptions.get(field_name, field_name)
                nested_fields[field_name] = _webui_field_schema(
                    nested_name,
                    field_name,
                    ConfigField(
                        type=type(default),
                        default=default,
                        description=description,
                    ),
                    hidden=False,
                    order=field_index,
                )
            sections[nested_name] = {
                "name": nested_name,
                "title": nested_meta["title"],
                "description": nested_meta["description"],
                "icon": None,
                "collapsed": False,
                "order": nested_index,
                "fields": nested_fields,
            }

        detached_description = (
            "画师预设列表较长，默认折叠以保持页面整洁；需要换行阅读或编辑长画师串时，"
            "请切换右上角“源代码”，编辑器会自动换行。"
        )
        for detached_index, (detached_name, detached_meta) in enumerate(
            self.config_webui_detached_field_sections.items(),
            start=len(sections),
        ):
            source_section_name = detached_meta["source_section"]
            source_field_name = detached_meta["source_field"]
            source_section = self.config_schema.get(source_section_name)
            if not isinstance(source_section, dict):
                continue
            source_field = source_section.get(source_field_name)
            if not isinstance(source_field, ConfigField):
                continue

            sections[detached_name] = {
                # Dashboard 依赖 section.name 决定取值和保存路径；这里必须仍指向原配置节。
                "name": source_section_name,
                "title": detached_meta["title"],
                "description": detached_description,
                "icon": None,
                "collapsed": True,
                "order": detached_index,
                "fields": {
                    source_field_name: _webui_field_schema(
                        source_section_name,
                        source_field_name,
                        source_field,
                        hidden=False,
                        order=0,
                    )
                },
            }

        return {
            "plugin_id": plugin_id or self.plugin_name,
            "plugin_info": {
                "name": plugin_name or self.plugin_name,
                "version": plugin_version or self.plugin_version,
                "description": plugin_description,
                "author": plugin_author or self.plugin_author,
            },
            "sections": sections,
            "layout": {"type": "tabs", "tabs": self.config_webui_tabs},
        }

    @staticmethod
    def load_local(config_path: str | Path) -> dict[str, Any]:
        """读取本地 TOML；缺失或格式损坏时返回空配置。"""
        resolved_path = Path(config_path)
        if not resolved_path.is_file():
            return {}
        try:
            with resolved_path.open("rb") as config_file:
                config_data = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        return config_data if isinstance(config_data, dict) else {}

    @staticmethod
    def merge(
        local_config: dict[str, Any],
        runtime_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """合并本地与宿主运行时配置，运行时值优先。"""
        if not isinstance(runtime_config, dict):
            return local_config
        return _merge_config_dicts(local_config, runtime_config)

    def regenerate_comments_if_needed(self, config_path: str | Path) -> bool:
        """为无注释配置迁移 Schema 注释，同时保留用户值与自定义 section。"""
        resolved_path = Path(config_path)
        if not resolved_path.is_file():
            return False
        try:
            existing_text = resolved_path.read_text(encoding="utf-8")
        except OSError:
            return False
        if any(line.lstrip().startswith("#") for line in existing_text.splitlines()):
            return False
        try:
            existing_doc = tomlkit.parse(existing_text)
        except Exception:
            return False

        new_text = self.compose_commented_text(existing_doc)
        if not new_text or new_text == existing_text:
            return False
        try:
            resolved_path.write_text(new_text, encoding="utf-8")
        except OSError:
            return False
        return True

    def compose_commented_text(self, existing_doc: Any) -> str:
        """按 Schema 顺序渲染带说明的 TOML，并保留 Schema 外的自定义 section。"""
        ordered = _webui_ordered_sections(
            self.config_schema,
            self.config_section_order,
        )
        blocks: list[str] = []
        header_text = _format_comment_block(self.config_file_header).strip()
        if header_text:
            blocks.append(header_text)

        seen_sections: set[str] = set()
        for section_name in ordered:
            fields = self.config_schema.get(section_name)
            if not isinstance(fields, dict):
                continue
            seen_sections.add(section_name)
            hidden = self.config_hidden_fields.get(section_name) or set()
            visible_fields = {
                field_name: field_def
                for field_name, field_def in fields.items()
                if field_name not in hidden
            }
            if not visible_fields:
                continue
            group_header = self.config_section_group_headers.get(section_name)
            if isinstance(group_header, str) and group_header.strip():
                blocks.append(_format_comment_block(group_header))
            blocks.append(
                _render_section_with_comments(
                    section_name=section_name,
                    fields=visible_fields,
                    section_desc=None,
                    existing_doc=existing_doc,
                )
            )

        if hasattr(existing_doc, "items"):
            for name, value in existing_doc.items():
                if name in seen_sections:
                    continue
                try:
                    document = tomlkit.document()
                    document.add(name, value)
                    snippet = tomlkit.dumps(document).strip()
                except Exception:
                    continue
                if snippet:
                    blocks.append(snippet)

        return "\n\n".join(block for block in blocks if block).rstrip() + "\n"


PLUGIN_CONFIG = PluginConfigDefinition()
