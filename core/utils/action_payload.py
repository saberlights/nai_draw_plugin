# -*- coding: utf-8 -*-
"""Planner 给出的 ``nai_web_draw`` action_data 的归一化工具。

Planner 把生图请求拆成 6 个结构化字段（``subject_and_pov`` / ``action`` / ``emotion``
/ ``scene_delta`` / ``dynamic_scene`` / ``framing``）加一个自由文本 ``description``，
以及稳定区变化决定（``outfit_change`` / ``environment_change`` 枚举 +
``*_new_look`` 描述）。提到这里独立成模块是为了能脱离 sdk_runtime 重型 import
单测，避免下游决策只看到结构化字段而漏掉 ``description`` 里的关键锚点
（如具体角色名、服装款式、场景细节）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable

from ..services.visual_continuity import parse_visual_change_directives


# Planner 拆出的结构化字段顺序，与 NAI tag 排序惯例对齐（主体 → 动作 → 情绪 → 稳定区 → 动态场景 → 构图）
STRUCTURED_DESCRIPTION_FIELDS = (
    "subject_and_pov",
    "action",
    "emotion",
    "scene_delta",
    "dynamic_scene",
    "framing",
)
PLANNER_STABLE_CHANGE_FIELDS = (
    "outfit_change",
    "environment_change",
)
PLANNER_STABLE_LOOK_FIELDS = (
    "outfit_new_look",
    "environment_new_look",
)


# Planner 在 ``subject_and_pov`` 里声明"本轮画的是指定角色（非 bot 出镜）"用的约定 token。
# 该 token 用于把"用户要看 bot 自己"和"用户/bot 要画一个指定二次元角色"两类意图在
# 链路上层分开：命中后 sdk_runtime 不会注入 self-image 提示，也不会走 selfie 后处理
# （即不会用 bot 默认外貌覆盖角色发色/瞳色）。
#
# 选用 ``画指定角色`` 作为 token：纯中文、不会被 NAI tag 误解析、与 selfie/portrait
# 三类标签语义正交，Planner 容易理解。
NAMED_CHARACTER_TOKEN = "画指定角色"
BOT_ABSENT_TOKEN = "Bot不出镜"


def is_named_character_intent(action_data: Dict[str, Any]) -> bool:
    """判断 Planner 是否声明"本轮画的是指定角色，而非 bot 出镜"。

    Planner 通过在 ``subject_and_pov`` 字段中包含 ``NAMED_CHARACTER_TOKEN``
    显式声明。链路上层据此跳过 self-image 注入与 selfie 后处理。

    注意：cosplay bot 出镜不算"指定角色"（出镜的还是 bot 本人），由 Planner 自行判断
    不写该 token。
    """
    subject = str(action_data.get("subject_and_pov", "") or "")
    return NAMED_CHARACTER_TOKEN in subject


def is_bot_absent_intent(action_data: Dict[str, Any]) -> bool:
    """判断 Planner 是否明确声明画面只含环境/物件，Bot 不出镜。"""
    subject = str(action_data.get("subject_and_pov", "") or "")
    return BOT_ABSENT_TOKEN in subject


def compose_description_from_action_payload(
    action_data: Dict[str, Any],
    *,
    structured_fields: Iterable[str] = STRUCTURED_DESCRIPTION_FIELDS,
) -> str:
    """把 Planner 的 6 个结构化字段 + ``description`` 拼成单行 request 文本。

    历史 bug：旧策略是"任一结构化字段非空就忽略 ``description``"——意图是避免重复，但
    实际后果是丢失 ``description`` 里**独有**的核心语义（具体角色名、cosplay 名、服装
    款式、场景物件），下游 LLM 只看到 ``"一女 第三视角 站立 微笑 特写"`` 而拿不到
    ``"初音未来, 公式服, 葱色双马尾"``，LLM 翻译时只能猜，配合后续 framing 误判会把
    用户点名的二次元角色洗成 bot 自拍。

    新策略：``description`` 与结构化字段都拼上，``description`` 在前（语义主体），
    结构化字段在后（构图补充）。少量重复 tag 由 LLM 自然消化，相比丢失锚点风险小得多。

    Args:
        action_data: Planner 返回的字段字典，至少应含 ``description`` 与（可选的）6 个结构化字段。
        structured_fields: 结构化字段名顺序，默认 ``STRUCTURED_DESCRIPTION_FIELDS``。

    Returns:
        拼接后的请求文本；两类信息都为空时返回空串。
    """
    structured_parts = []
    for key in structured_fields:
        value = str(action_data.get(key, "") or "").strip()
        if value:
            structured_parts.append(value)

    # 稳定区变化只把语义描述拼入 request 文本（服装/地点内容对 tag 检索有价值）；
    # unchanged / switch:<key> 等控制 token 属于控制信道，不得混入数据信道，
    # 否则会污染 Danbooru 检索 query 与图片 caption。
    directives = parse_visual_change_directives(action_data) or {}
    for kind in ("outfit", "environment"):
        directive = directives.get(kind)
        if directive is not None and directive.description:
            structured_parts.append(directive.description)

    description = str(action_data.get("description", "") or "").strip()
    parts = [p for p in (description, " ".join(structured_parts)) if p]
    return " ".join(parts).strip()


def render_action_payload_for_prompt(
    action_data: Dict[str, Any],
    *,
    effective_description: str = "",
) -> str:
    """保留 Planner 字段边界，供下游 Tag LLM 判断稳定区与动态区。"""

    payload: Dict[str, str] = {}
    effective = str(effective_description or "").strip()
    flattened = compose_description_from_action_payload(action_data)
    if effective and effective != flattened:
        payload["effective_description"] = effective
    description = str(action_data.get("description", "") or "").strip()
    if description:
        payload["description"] = description
    for key in STRUCTURED_DESCRIPTION_FIELDS:
        value = str(action_data.get(key, "") or "").strip()
        if value:
            payload[key] = value
    for key in (*PLANNER_STABLE_CHANGE_FIELDS, *PLANNER_STABLE_LOOK_FIELDS):
        value = str(action_data.get(key, "") or "").strip()
        if value:
            payload[key] = value
    if not payload:
        return ""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"<planner_visual_request>\n{serialized}\n</planner_visual_request>"
