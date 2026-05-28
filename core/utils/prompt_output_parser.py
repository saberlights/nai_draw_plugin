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

# 拍平多人字符串的 charN: 前缀（兼容大小写、中英文冒号、可选空格）
_CHAR_PREFIX_RE = re.compile(r"^char\s*\d+\s*[:：]\s*", re.IGNORECASE)


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


# 完整闭合的代码块：``` 或 ```lang 起手 + 任意内容 + ``` 收尾
_FULL_CODE_BLOCK_RE = re.compile(
    r"```([a-zA-Z][\w-]*)?[ \t]*\n?(.*?)```",
    re.DOTALL,
)


def extract_last_code_block(text: str) -> Optional[str]:
    """从含 thought + 代码块的 LLM 输出里提取最后一个 ``` 代码块的内容。

    应对场景：LLM 输出"思考过程 + ```prompt```" 这种混合格式时，``_strip_code_fence``
    只能识别"整段被 ``` 包裹"，识别不到"前面有 thought / 后面才是代码块"的情形，
    导致 thought 段被当成 prompt 送 API。

    匹配优先级：
    1. 完整闭合的 ```...```（多个时取**最后一个**——LLM 习惯把最终 prompt 放在末尾）
    2. 未闭合（被 max_tokens 截断）：取最后一个 ``` 之后到末尾的内容
    3. 完全没有 ```：返回 None，调用方按"整段即 prompt"处理

    返回内容已 strip 前后空白；首行若是 ``json`` / ``markdown`` 这种语言标识也会被剥掉。
    """
    if not text or "```" not in text:
        return None

    matches = list(_FULL_CODE_BLOCK_RE.finditer(text))
    if matches:
        last = matches[-1]
        content = last.group(2).strip()
        if content:
            return content
        # 闭合代码块但内容为空：当作没找到，继续走截断分支

    # 未闭合：找最后一个 ```，把后面的内容当 prompt
    last_open = text.rfind("```")
    if last_open == -1:
        return None
    tail = text[last_open + 3:]
    # 跳过可选的 lang 标识行（```python\n / ```json\n）
    if "\n" in tail:
        head_line, rest = tail.split("\n", 1)
        head_stripped = head_line.strip()
        if head_stripped.isalpha() and len(head_stripped) < 15:
            tail = rest
    tail = tail.strip()
    return tail or None


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


def _split_multi_person_segments(text: str) -> List[str]:
    """把拍平的多人字符串切成 [global, char1, char2, ...] 段。

    支持两种拍平形态：
    - 多行：``"global,\\nchar1:p1,\\nchar2:p2,"``（_render_from_v2 / LLM 当前主流输出）
    - 单行 ``|`` 分隔：``"global | char1 prompt | char2 prompt"``（旧版本兼容）
    """
    stripped = (text or "").strip()
    if not stripped:
        return []

    if "\n" in stripped:
        return [seg.strip() for seg in stripped.split("\n") if seg.strip()]
    if "|" in stripped:
        return [seg.strip() for seg in stripped.split("|") if seg.strip()]
    return [stripped]


def extract_multi_character_payload_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从拍平后的多人字符串反解出结构化角色 payload。

    用于 LLM 走文本路径（非 JSON）或 v3 JSON 模板被用户覆盖的场景，确保只要 LLM
    输出了 ``char1:/char2:`` 多段格式，就能进入 NewAPI 的 ``characters[]`` 通道。

    Returns:
        与 :func:`extract_multi_character_payload` 同结构；解析为单人或失败时返回 ``None``。
        反解路径不会带 position，因此 ``has_coords`` 永远为 ``False``。
    """
    segments = _split_multi_person_segments(text)
    if len(segments) < 3:
        # 至少 1 段 global + 2 段角色才认为是多人
        return None

    global_text = segments[0].strip().rstrip(",").strip()
    if not global_text:
        return None

    characters: List[Dict[str, str]] = []
    for raw_segment in segments[1:]:
        cleaned = raw_segment.strip().lstrip("|").strip().rstrip(",").strip()
        cleaned = _CHAR_PREFIX_RE.sub("", cleaned).strip()
        if not cleaned:
            continue
        characters.append({"prompt": cleaned, "negative_prompt": "", "position": ""})

    if len(characters) < 2:
        return None

    return {
        "global_text": global_text,
        "characters": characters,
        "has_coords": False,
    }


def resolve_multi_character_payload(
    raw_llm_response: str,
    rendered_text: str,
) -> Optional[Dict[str, Any]]:
    """统一入口：优先用 v3 JSON 抽取，失败时回退到从拍平文本反解。

    Args:
        raw_llm_response: LLM 的原始返回（可能是 JSON、JSON+噪声、纯文本任意一种）
        rendered_text: 经 ``_cleanup_llm_prompt`` 拍平后的最终字符串（含 ``char1:/char2:``）

    Returns:
        结构化 payload；若 LLM 输出为单人或都解析失败，返回 ``None``。
    """
    from_json = extract_multi_character_payload(raw_llm_response)
    if from_json is not None:
        return from_json
    return extract_multi_character_payload_from_text(rendered_text)
