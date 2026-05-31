# -*- coding: utf-8 -*-
"""
提示词后处理工具

目标：
- 自拍模式：在用户未明确指定外貌时，去掉容易产生“随机外貌”的标签（如 black hair/long hair）
- 轻量排序：把人物数量/视角类标签前置，把 year xxxx 放到末尾，降低“顺序混乱”的体感

说明：
- 这里的规则是“温和纠正”，不会尝试完整实现 Danbooru 全分类排序（容易误判、反而更乱）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_COUNT_RE = re.compile(r"^(?:solo|\d+girls|\d+boys|\d+people|1girl|1boy)$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^year\s+\d{4}$", re.IGNORECASE)

# 只覆盖少量“稳定且高频”的构图/视角/自拍标签，用于轻量前置
_CAMERA_TAGS = {
    "pov",
    "female pov",
    "looking at viewer",
    "from above",
    "from below",
    "wide angle",
    "close-up",
    "close up",
    "full body",
    "upper body",
    "lower body",
    "selfie",
    "mirror selfie",
    "group selfie",
    "holding phone",
}

_SFW_BANNED_EXACT_TAGS = {
    "nsfw",
    "nude",
    "naked",
    "sex",
    "sexual",
    "sexy",
    "suggestive",
    "seductive",
    "lewd",
    "erotic",
    "explicit",
    "penis",
    "pussy",
    "vagina",
    "nipples",
    "nipple",
    "anus",
    "anal",
    "penetration",
    "cum",
    "ejaculation",
    "fellatio",
    "cunnilingus",
    "paizuri",
    "footjob",
    "handjob",
    "masturbation",
    "orgasm",
    "topless",
    "bottomless",
    "cameltoe",
    "cleavage",
    "underboob",
    "sideboob",
    "thighs",
    "midriff",
    "lingerie",
    "bikini",
    "swimsuit",
    "panties",
    "underwear",
    "thong",
    "bra",
    "no bra",
    "see-through",
    "see through",
    "transparent clothes",
}
_SFW_BANNED_SUBSTRINGS = (
    "bikini",
    "swimsuit",
    "lingerie",
    "panties",
    "underwear",
    "thong",
    "cameltoe",
    "cleavage",
    "underboob",
    "sideboob",
    "see-through",
    "see through",
    "transparent",
    "covered nipples",
    "no bra",
    "bra lift",
    "pussy juice",
    "grop",
    "fondl",
    "fingering",
    "fingered",
    "grabbing breast",
    "breast grab",
    "biting neck",
)


def _split_prompt_segments(prompt: str) -> List[str]:
    """兼容旧的多行 `|` 分段和新的单行 `base | char1 | char2` 格式。"""
    text = (prompt or "").strip()
    if not text:
        return []

    if "\n" in text:
        return [segment.strip() for segment in text.split("\n") if segment.strip()]

    if "|" in text:
        parts = [part.strip() for part in text.split("|")]
        segments: List[str] = []
        for index, part in enumerate(parts):
            if not part:
                continue
            if index == 0:
                segments.append(part)
            else:
                segments.append(f"| {part}")
        return segments

    return [text]


def _join_prompt_segments(lines: List[str], original_prompt: str) -> str:
    """保持与输入一致的多人分隔风格。"""
    if not lines:
        return ""

    if "\n" in (original_prompt or ""):
        return "\n".join(lines).strip()

    if "|" in (original_prompt or ""):
        normalized: List[str] = []
        for index, line in enumerate(lines):
            raw = line.strip()
            if index == 0:
                normalized.append(raw.lstrip("|").strip())
            else:
                normalized.append(raw.lstrip("|").strip())
        return " | ".join([part for part in normalized if part]).strip()

    return "\n".join(lines).strip()


def _preserve_trailing_comma(rendered_line: str, raw_line: str) -> str:
    """保留结构化多人提示词每行末尾的续接逗号。"""
    line = rendered_line.strip()
    if line and raw_line.rstrip().endswith(",") and not line.endswith(","):
        return f"{line},"
    return line


def user_mentions_appearance(raw_request: str) -> bool:
    """粗略判断用户是否明确提及外貌（发色/发型/眼睛等）。"""
    if not raw_request:
        return False

    s = raw_request.lower()
    # 中文关键词（偏保守，宁可认为“用户提了”）
    cn_keys = [
        "头发", "发色", "发型", "长发", "短发", "双马尾", "马尾", "刘海",
        "黑发", "金发", "白发", "粉发", "蓝发", "红发", "紫发", "银发", "棕发",
        "眼睛", "瞳", "瞳色", "蓝瞳", "红瞳", "金瞳", "绿瞳", "紫瞳",
        # 口语缩写（常见二次元描述）
        "黑长直",
    ]
    if any(k in raw_request for k in cn_keys):
        return True

    # 英文关键词
    en_keys = ["hair", "haired", "eyes", "eyed", "twintails", "ponytail", "bangs"]
    return any(k in s for k in en_keys)


def _strip_wrappers(tag: str) -> str:
    """去掉常见权重/括号包装，便于规则匹配（不改变原 tag 输出，仅用于判断）。"""
    t = tag.strip()
    # 去掉 NAI 花括号/方括号
    t = t.lstrip("{[(").rstrip("}])")
    t = t.strip()
    # 去掉 NAI4/4.5 权重前缀与后缀，如 1.2::blue hair::
    t = re.sub(r"^[+-]?\d+(?:\.\d+)?::", "", t).strip()
    t = re.sub(r"::\s*$", "", t).strip()
    return t


def remove_selfie_appearance_tags(prompt: str) -> str:
    “””
    去掉自拍里常见的”随机外貌标签”（发色/发型/瞳色）。

    只移除明确的外貌 tag，尽量不伤及配饰（如 hair ribbon / hair ornament）。
    支持 NAI4/4.5 高级权重语法（weight::tag::）。
    “””
    if not prompt or not prompt.strip():
        return prompt

    hair_colors = {
        "black", "blonde", "brown", "blue", "pink", "white", "silver",
        "red", "green", "purple", "orange", "gray", "grey", "aqua", "cyan",
    }
    eye_colors = {
        "black", "brown", "blue", "red", "green", "purple", "orange",
        "gray", "grey", "golden", "yellow", "pink", "aqua", "cyan",
    }
    hair_styles_exact = {
        "twintails",
        "twin tails",
        "ponytail",
        "side ponytail",
        "braid",
        "side braid",
        "pigtails",
        "hair bun",
        "bun",
        "bob cut",
        "hime cut",
        "bangs",
        "blunt bangs",
        "straight hair",
        "wavy hair",
        "curly hair",
        "messy hair",
    }

    def should_remove(tag: str) -> bool:
        core = _strip_wrappers(tag).lower()
        core = re.sub(r"\s+", " ", core).strip()

        # 明确配饰：不移除
        if "hair" in core and any(x in core for x in ("ribbon", "ornament", "clip", "pin", "bow", "band", "flower")):
            return False

        # 发色：xxx hair / xxx-haired（如 black hair, blue hair）
        m = re.match(r"^([a-z]+)\s+hair$", core)
        if m and m.group(1) in hair_colors:
            return True
        if re.match(r"^[a-z]+-haired$", core):
            return True

        # 发型/长度：long hair / very long hair / short hair / medium hair
        if re.match(r"^(?:very )?(?:long|short|medium)\s+hair$", core):
            return True

        # 长度+发色组合：long brown hair / very long black hair / short blonde hair
        m_combo = re.match(r"^(?:very )?(long|short|medium)\s+([a-z]+)\s+hair$", core)
        if m_combo and m_combo.group(2) in hair_colors:
            return True

        # 常见发型词
        if core in hair_styles_exact:
            return True

        # 瞳色：xxx eyes
        m2 = re.match(r"^([a-z]+)\s+eyes$", core)
        if m2 and m2.group(1) in eye_colors:
            return True

        return False

    lines = _split_prompt_segments(prompt)
    out_lines: List[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        raw_line = raw

        prefix = ""
        if raw.startswith("|"):
            prefix = "|"
            raw = raw[1:].strip()

        tags = [t.strip() for t in raw.split(",") if t.strip()]
        filtered = [t for t in tags if not should_remove(t)]

        joined = _preserve_trailing_comma(", ".join(filtered), raw_line)
        if prefix:
            out_lines.append(f"{prefix} {joined}".strip())
        else:
            out_lines.append(joined)

    return _join_prompt_segments(out_lines, prompt)


# CJK + 全角清洗：覆盖中文 / 日文假名 / 韩文 / CJK 标点 / 全角形式
# 文档 §8 明确要求 prompt / negative_prompt / characters[i].prompt 必须英文，
# 含 CJK 字符或全角符号一律 400。LLM 偶尔会留中文解释、未翻译角色词、全角逗号；
# 本规则在送 API 前做最后一道清洗。
#
# 范围说明：
# - U+1100–U+11FF / U+A960–U+A97F / U+AC00–U+D7AF：韩文谚文字母 / 扩展 A / 音节
# - U+3000–U+303F：CJK 符号与标点（含全角空格、。、〈〉「」『』）
# - U+3040–U+30FF / U+31F0–U+31FF：平假名 + 片假名 + 片假名拼音扩展
# - U+3400–U+4DBF / U+4E00–U+9FFF：CJK 扩展 A + 统一表意基本区
# - U+F900–U+FAFF：CJK 兼容表意
# - U+FE30–U+FE4F：CJK 兼容形式（中文标点兼容变体）
# - U+FF00–U+FFEF：半/全角形式（全角字母数字、全角标点、半角片假名）
_CJK_AND_FULLWIDTH_RE = re.compile(
    "["
    "ᄀ-ᇿ"
    "　-〿"
    "぀-ヿ"
    "ㇰ-ㇿ"
    "㐀-䶿"
    "一-鿿"
    "ꥠ-꥿"
    "가-힯"
    "豈-﫿"
    "︰-﹏"
    "＀-￯"
    "]+"
)


def strip_cjk_and_fullwidth(prompt: str) -> str:
    """剔除 LLM 翻译残留的 CJK 字符与全角符号。

    NewAPI §8 要求 prompt 必须英文，含 CJK 或全角符号一律 400。LLM 偶尔会漏译，
    本函数在送 API 前做最后一道清洗：
    1. 把连续 CJK / 全角符号整段替换为单个空格（不直接删，避免 "red红色dress" 合成 "reddress"）
    2. 按现有多人段（`|` / `\\n`）拆分，再按 `,` 拆 tag，复用其它后处理函数的分段约定
    3. 单 tag 内合并连续空白、trim；空 tag 直接丢弃；行内全部 tag 被丢弃则整行删
    """
    if not prompt or not prompt.strip():
        return prompt

    lines = _split_prompt_segments(prompt)
    out_lines: List[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        raw_line = raw

        prefix = ""
        if raw.startswith("|"):
            prefix = "|"
            raw = raw[1:].strip()

        # 与 sanitize_sfw_prompt 对齐：保留 char1:/char2: 角色前缀，不在替换里被吃掉
        role_prefix = ""
        role_match = re.match(r"^(char\d+:)\s*(.*)$", raw, re.IGNORECASE)
        if role_match:
            role_prefix = role_match.group(1)
            raw = role_match.group(2)

        # 把"中文版的逗号"（全角逗号 U+FF0C / 顿号 U+3001）先转成英文逗号，
        # 保留 LLM 输出里"逗号当 tag 分隔符"的原本语义；其余 CJK + 全角符号统一替换为空格
        cleaned = raw.replace("，", ",").replace("、", ",")
        cleaned = _CJK_AND_FULLWIDTH_RE.sub(" ", cleaned)

        tags: List[str] = []
        for token in cleaned.split(","):
            t = re.sub(r"\s+", " ", token).strip()
            if t:
                tags.append(t)
        if not tags:
            continue

        joined = ", ".join(tags)
        joined = _preserve_trailing_comma(joined, raw_line)
        rebuilt = f"{role_prefix}{joined}" if role_prefix else joined
        if prefix:
            out_lines.append(f"{prefix} {rebuilt}".strip())
        else:
            out_lines.append(rebuilt)

    return _join_prompt_segments(out_lines, prompt)


def sanitize_sfw_prompt(prompt: str) -> str:
    """移除 SFW 模式下不应出现的擦边/色情标签。"""
    if not prompt or not prompt.strip():
        return prompt

    def is_forbidden(tag: str) -> bool:
        core = _strip_wrappers(tag).lower()
        core = re.sub(r"\s+", " ", core).strip()
        core = re.sub(r"^(?:source|target|mutual)#", "", core).strip()
        if not core:
            return False

        if core in _SFW_BANNED_EXACT_TAGS:
            return True

        return any(token in core for token in _SFW_BANNED_SUBSTRINGS)

    lines = _split_prompt_segments(prompt)
    out_lines: List[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        raw_line = raw

        prefix = ""
        if raw.startswith("|"):
            prefix = "|"
            raw = raw[1:].strip()

        role_prefix = ""
        role_match = re.match(r"^(char\d+:)\s*(.*)$", raw, re.IGNORECASE)
        if role_match:
            role_prefix = role_match.group(1)
            raw = role_match.group(2)

        tags = [t.strip() for t in raw.split(",") if t.strip()]
        filtered = [t for t in tags if not is_forbidden(t)]
        if not filtered:
            continue

        joined = ", ".join(filtered)
        joined = _preserve_trailing_comma(joined, raw_line)
        rebuilt = f"{role_prefix}{joined}" if role_prefix else joined
        if prefix:
            out_lines.append(f"{prefix} {rebuilt}".strip())
        else:
            out_lines.append(rebuilt)

    return _join_prompt_segments(out_lines, prompt)


def normalize_prompt_order(prompt: str) -> str:
    """
    轻量排序（尽量不“过度聪明”）：
    - 把人数/solo 等放到最前
    - 把 POV/自拍/视角等常见镜头词前置
    - 把 year xxxx 放到末尾
    """
    if not prompt or not prompt.strip():
        return prompt

    lines = _split_prompt_segments(prompt)
    out_lines: List[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        raw_line = raw

        prefix = ""
        if raw.startswith("|"):
            prefix = "|"
            raw = raw[1:].strip()

        tags = [t.strip() for t in raw.split(",") if t.strip()]
        if not tags:
            continue

        counts: List[str] = []
        cameras: List[str] = []
        years: List[str] = []
        rest: List[str] = []

        for t in tags:
            core = _strip_wrappers(t)
            core_norm = re.sub(r"\s+", " ", core).strip().lower()

            if _YEAR_RE.match(core_norm):
                years.append(t)
            elif _COUNT_RE.match(core_norm):
                counts.append(t)
            elif core_norm in _CAMERA_TAGS:
                cameras.append(t)
            else:
                rest.append(t)

        # 视角类标签通常比 1girl/1boy 更“前置有效”，所以输出时把 camera 放在 count 之前
        # 但保留原始相对顺序（分别在各自组内稳定）
        new_tags = cameras + counts + rest + years
        joined = _preserve_trailing_comma(", ".join(new_tags).strip(), raw_line)
        if prefix:
            out_lines.append(f"{prefix} {joined}".strip())
        else:
            out_lines.append(joined)

    return _join_prompt_segments(out_lines, prompt)


# ==================== 结构化多角色后处理 ====================
# 这些 wrapper 复用上面的字符串实现，专门服务于 NewAPI `characters[]` 通道。
# 每个角色的 prompt / negative_prompt 都是无 `char1:` 前缀的单行字符串，
# 复用单字符串路径既不需要重写规则，也保证字符串路径与结构化路径行为完全一致。


def _apply_string_filter_to_characters(
    global_text: str,
    characters: List[Dict[str, Any]],
    filter_fn,
) -> Tuple[str, List[Dict[str, Any]]]:
    """把字符串级过滤函数同步作用于 global + 每个 character 的 prompt/negative_prompt。

    过滤后某个 character 的 prompt 被清空时，该 character 整体被丢弃；
    调用方负责判断丢弃后角色数是否还满足结构化通道的最低要求（≥ 2）。
    """
    new_global = filter_fn(global_text) if global_text else global_text

    new_characters: List[Dict[str, Any]] = []
    for raw_item in characters or []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)

        char_prompt = str(item.get("prompt") or "")
        if char_prompt:
            item["prompt"] = filter_fn(char_prompt).strip()
        if not item.get("prompt"):
            continue

        char_negative = str(item.get("negative_prompt") or "")
        if char_negative:
            item["negative_prompt"] = filter_fn(char_negative).strip()

        new_characters.append(item)

    return new_global, new_characters


def sanitize_sfw_characters(
    global_text: str,
    characters: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """SFW 模式下的多角色 payload 清洗：与 sanitize_sfw_prompt 行为一致。"""
    return _apply_string_filter_to_characters(global_text, characters, sanitize_sfw_prompt)


def strip_cjk_and_fullwidth_from_characters(
    global_text: str,
    characters: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """多角色 payload 上的 CJK + 全角清洗：与 strip_cjk_and_fullwidth 行为一致。

    某个 character 的 prompt 被清光时整个 character 会被丢弃，调用方负责按结构化通道
    最低人数（≥ 2）判断是否需要降级回字符串路径。
    """
    return _apply_string_filter_to_characters(
        global_text, characters, strip_cjk_and_fullwidth
    )


def normalize_characters_order(
    global_text: str,
    characters: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """按 normalize_prompt_order 规则对 global 与每个 character 的 tag 顺序做轻量整理。"""
    return _apply_string_filter_to_characters(global_text, characters, normalize_prompt_order)


def remove_selfie_appearance_from_characters(
    global_text: str,
    characters: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """自拍模式下从 global 与每个 character 的 prompt 中剥离随机外貌 tag。"""
    return _apply_string_filter_to_characters(
        global_text, characters, remove_selfie_appearance_tags
    )
