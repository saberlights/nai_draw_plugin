# -*- coding: utf-8 -*-
"""
LLM 输出解析工具

用于从 LLM 返回内容中稳定提取“最终提示词(prompt)”。

支持：
- 结构化 JSON 输出：{"format":"single|multi","prompt":"..."}
- JSON 被 ```json 代码块包裹
- JSON 前后夹杂少量无关文本（尽量提取包含 "prompt" 的 JSON 对象）
- v3 多人 JSON 抽取结构化 characters payload（含 position 网格）

解析失败时返回 None，调用方应回退到原有的纯文本清洗逻辑。
"""

from __future__ import annotations

import json
import re
from typing import Optional, Dict, Any, List


# 5×5 网格坐标 [A-E][1-5]（NewAPI 多角色 position 字面量）
_POSITION_GRID_RE = re.compile(r"^[A-E][1-5]$")


def _strip_code_fence(text: str) -> str:
    """去掉可能的 ```lang ... ``` 包裹（只做轻量处理）。"""
    s = (text or "").strip()
    if not (s.startswith("```") and s.endswith("```")):
        return s

    inner = s[3:-3].strip()
    if "\n" not in inner:
        return inner.strip()

    first_line, rest = inner.split("\n", 1)
    # 常见形式：```json\\n{...}\\n```
    if first_line.strip().isalpha() and len(first_line.strip()) < 15:
        return rest.strip()
    return inner.strip()


def _join_tags(tags) -> str:
    if not tags:
        return ""
    if not isinstance(tags, list):
        return ""
    return ", ".join([t.strip() for t in tags if isinstance(t, str) and t.strip()]).strip()


def parse_structured_prompt_payload(text: str) -> Optional[Dict[str, Any]]:
    """
    从结构化输出中提取原始 payload。

    成功时返回 JSON 对象本身，失败返回 None。
    调用方可进一步读取 intent / continuity / global / people 等字段。
    """
    cleaned = _strip_code_fence(text).strip()
    if not cleaned:
        return None

    candidates = [cleaned]
    if any(token in cleaned for token in ('"prompt"', '"global"', '"people"')):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start:end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        version = obj.get("version")
        has_v2_fields = isinstance(obj.get("global"), list)
        has_v1_prompt = isinstance(obj.get("prompt"), str) and obj.get("prompt", "").strip()
        if version == 2 or version == 3 or (isinstance(version, int) and version >= 2):
            if has_v2_fields or has_v1_prompt:
                return obj
            continue

        if has_v1_prompt:
            return obj

    return None


def _render_from_v2(obj: dict) -> Optional[str]:
    global_tags = obj.get("global")
    if not isinstance(global_tags, list):
        return None

    first_line = _join_tags(global_tags)
    if not first_line:
        return None

    people = obj.get("people", [])
    if people is None:
        people = []

    if not isinstance(people, list):
        people = []

    format_value = str(obj.get("format", "") or "").strip().lower()
    valid_people: list[list[str]] = []
    for person_tags in people:
        if not isinstance(person_tags, list):
            continue
        person_line = [t.strip() for t in person_tags if isinstance(t, str) and t.strip()]
        if person_line:
            valid_people.append(person_line)

    # 单人（或未明确 multi）：不要使用 | 分段，把唯一人物并入同一行
    if format_value != "multi" or len(valid_people) <= 1:
        if valid_people:
            merged = _join_tags(global_tags + valid_people[0])
            return merged if merged else first_line
        return first_line

    # 多人：渲染为多行结构化文本 charX:[tag列表]
    lines = [first_line + ","]
    for i, person_tags in enumerate(valid_people, start=1):
        person_line = _join_tags(person_tags)
        if person_line:
            lines.append(f"char{i}:{person_line},")

    return "\n".join(lines).strip()


def parse_prompt_from_structured_output(text: str) -> Optional[str]:
    """
    从结构化输出中解析 prompt 字段。

    Returns:
        prompt 字符串（可能包含换行，用于多人 | 分段），失败返回 None
    """
    obj = parse_structured_prompt_payload(text)
    if not obj:
        return None

    version = obj.get("version")
    if version == 2 or version == 3 or (isinstance(version, int) and version >= 2):
        rendered = _render_from_v2(obj)
        if rendered:
            return rendered

    prompt = obj.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        normalized = prompt.strip()
        # 一些模型会把换行二次转义成 \\n，导致解析后仍是字面量 \n。
        # 多人 | 分段基本都会出现 "\\n|" 形态，这里做一次温和纠正。
        if "\\n|" in normalized:
            normalized = normalized.replace("\\n", "\n")
        return normalized

    return None


def extract_multi_character_payload(text: str) -> Optional[Dict[str, Any]]:
    """从 v3 multi JSON 抽出结构化角色 payload，供 NewAPI `characters[]` 通道使用。

    Returns:
        成功时返回 ``{"global_text": str,
                       "characters": [{"prompt": str, "negative_prompt": str, "position": str}, ...],
                       "has_coords": bool}``；
        不是 v2/v3 多人输出、人数 < 2 或字段缺失时返回 ``None``，调用方应回退到字符串通道。

    Notes:
        - 仅当 ``format == "multi"`` 且 ``people`` 至少包含 2 个非空角色时才会返回结构化结果
        - ``positions`` 数组长度允许 ≤ ``people``，越界处按 ``""`` 处理
        - ``position`` 字面量不匹配 ``[A-E][1-5]`` 时被规整为 ``""``（不抛错，仅丢弃该坐标）
        - ``has_coords`` 为 ``True`` 当且仅当所有角色都有合法坐标，否则交由后端自动布局
    """
    obj = parse_structured_prompt_payload(text)
    if not obj:
        return None

    version = obj.get("version")
    if not (version == 2 or version == 3 or (isinstance(version, int) and version >= 2)):
        return None

    if str(obj.get("format", "") or "").strip().lower() != "multi":
        return None

    raw_people = obj.get("people", []) or []
    if not isinstance(raw_people, list):
        return None

    valid_people: List[List[str]] = []
    for person_tags in raw_people:
        if not isinstance(person_tags, list):
            continue
        person_line = [t.strip() for t in person_tags if isinstance(t, str) and t.strip()]
        if person_line:
            valid_people.append(person_line)

    if len(valid_people) < 2:
        return None

    global_text = _join_tags(obj.get("global"))
    if not global_text:
        return None

    raw_positions = obj.get("positions", []) or []
    if not isinstance(raw_positions, list):
        raw_positions = []

    characters: List[Dict[str, str]] = []
    normalized_positions: List[str] = []
    for index, tags in enumerate(valid_people):
        position = ""
        if index < len(raw_positions):
            candidate = str(raw_positions[index] or "").strip().upper()
            if _POSITION_GRID_RE.match(candidate):
                position = candidate
        normalized_positions.append(position)
        characters.append(
            {
                "prompt": _join_tags(tags),
                "negative_prompt": "",
                "position": position,
            }
        )

    has_coords = bool(normalized_positions) and all(p for p in normalized_positions)

    return {
        "global_text": global_text,
        "characters": characters,
        "has_coords": has_coords,
    }
