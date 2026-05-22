# -*- coding: utf-8 -*-
"""
LLM 输出解析工具

用于从 LLM 返回内容中稳定提取“最终提示词(prompt)”。

支持：
- 结构化 JSON 输出：{"format":"single|multi","prompt":"..."}
- JSON 被 ```json 代码块包裹
- JSON 前后夹杂少量无关文本（尽量提取包含 "prompt" 的 JSON 对象）

解析失败时返回 None，调用方应回退到原有的纯文本清洗逻辑。
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any


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
