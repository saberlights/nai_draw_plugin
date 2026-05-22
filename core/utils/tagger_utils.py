# -*- coding: utf-8 -*-
"""
打标工具函数

设计目标：
- 仅做纯工具逻辑，方便单测与复用
- 尽量不依赖 PIL 等额外依赖（运行环境可能缺失）
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence


_PICID_RE = re.compile(r"\[picid:([0-9a-fA-F-]{8,})\]")
_DATA_URL_RE = re.compile(r"^data:image/(?P<fmt>[a-zA-Z0-9+.-]+);base64,(?P<b64>[A-Za-z0-9+/=]+)$")

# 图片格式检测模式（参考 custom_pic_plugin/core/image_utils.py）
_IMAGE_FORMAT_PREFIX = {
    "jpeg": ("/9j/",),
    "png": ("iVBORw",),
    "webp": ("UklGR",),
    "gif": ("R0lGOD",),
}

_IMAGE_MAGIC = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG",),
    "webp": (b"RIFF",),  # webp: RIFF....WEBP
    "gif": (b"GIF8",),
}


def find_reply_message_id(message_segment: Any) -> Optional[str]:
    """
    从消息段中递归查找 reply 段，返回被引用消息的 message_id。

    兼容：
    - maim_message.Seg 对象（type/data）
    - dict 结构（{"type": "...", "data": ...}）
    """
    if message_segment is None:
        return None

    seg_type = getattr(message_segment, "type", None)
    seg_data = getattr(message_segment, "data", None)
    if seg_type is None and isinstance(message_segment, dict):
        seg_type = message_segment.get("type")
        seg_data = message_segment.get("data")

    if seg_type == "reply":
        # 常见：data 直接是 message_id
        if isinstance(seg_data, (str, int)):
            mid = str(seg_data).strip()
            return mid or None

        # 兼容：data 是 dict（不同适配器可能用 {"message_id": "..."} 或 {"id": "..."}）
        if isinstance(seg_data, dict):
            for k in ("message_id", "id", "reply_to", "reply_message_id", "quote_message_id"):
                v = seg_data.get(k)
                if isinstance(v, (str, int)) and str(v).strip():
                    return str(v).strip()
        return None

    if seg_type == "seglist":
        items = seg_data if isinstance(seg_data, list) else []
        for item in items:
            mid = find_reply_message_id(item)
            if mid:
                return mid
        return None

    # 其他类型不处理
    return None


def extract_picids(text: str) -> List[str]:
    """从 processed_plain_text 中提取 picid 列表。"""
    if not text:
        return []
    return [m.group(1) for m in _PICID_RE.finditer(text)]


def guess_image_format_from_path(path: str) -> str:
    """根据文件扩展名推断图片格式（给 VLM 用）。"""
    ext = os.path.splitext(path or "")[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg", "png", "webp", "gif"):
        return ext
    # 兜底：NAI/内部处理默认都是 png 保存
    return "png"


def read_image_as_base64(path: str) -> str:
    """读取图片文件并转 base64（不做 PIL 解码）。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def strip_data_url(image_data: str) -> tuple[str, Optional[str]]:
    """
    处理 data:image/...;base64,... 格式，返回 (纯base64, fmt?)。
    非 data url 则原样返回。
    """
    if not isinstance(image_data, str):
        return "", None
    s = image_data.strip()
    m = _DATA_URL_RE.match(s)
    if not m:
        return s, None
    fmt = (m.group("fmt") or "").lower()
    b64 = m.group("b64") or ""
    # 兼容 jpg
    if fmt == "jpg":
        fmt = "jpeg"
    return b64, (fmt or None)


def guess_image_format_from_base64(image_base64: str) -> str:
    """
    根据 base64 内容猜测图片格式（给 VLM 用）。
    优先：
    1) data url 的 fmt
    2) base64 前缀
    3) 解码后的 magic bytes（最多解码前 64 字节）
    """
    b64, fmt = strip_data_url(image_base64)
    if fmt in ("jpeg", "png", "webp", "gif"):
        return fmt

    # 前缀判断
    head = b64[:20]
    for k, prefixes in _IMAGE_FORMAT_PREFIX.items():
        if any(head.startswith(p) for p in prefixes):
            return k

    # magic bytes
    try:
        raw = base64.b64decode(b64[:120], validate=False)
        for k, magics in _IMAGE_MAGIC.items():
            if any(raw.startswith(m) for m in magics):
                # webp 需要进一步确认，但 RIFF 基本够用
                return k
    except Exception:
        pass

    return "png"


def extract_image_base64_list(message_segment: Any) -> List[str]:
    """
    从消息段中递归提取图片/表情包的 base64 列表。

    兼容：
    - maim_message.Seg（type/data）
    - dict（{"type": "...", "data": ...}）
    """
    if message_segment is None:
        return []

    seg_type = getattr(message_segment, "type", None)
    seg_data = getattr(message_segment, "data", None)
    if seg_type is None and isinstance(message_segment, dict):
        seg_type = message_segment.get("type")
        seg_data = message_segment.get("data")

    if seg_type in ("image", "emoji"):
        if isinstance(seg_data, str) and seg_data.strip():
            return [seg_data.strip()]
        return []

    if seg_type == "seglist":
        out: List[str] = []
        items = seg_data if isinstance(seg_data, list) else []
        for item in items:
            out.extend(extract_image_base64_list(item))
        return out

    return []


def extract_image_base64_list_from_payload(payload: Any, max_depth: int = 8) -> List[str]:
    """
    从“可能包含引用消息内容”的任意 payload 中尽力提取 image/emoji 的 base64。

    典型来源：
    - message.raw_message（适配器原始结构，可能含 reply_message / quoted_message）
    - message.message_info.additional_config（某些适配器会把引用消息塞这里）
    - 其它自定义字段

    兼容的常见结构（举例）：
    - {"message_segment": {...SegDict...}}
    - {"message": [{...seg...}, {...seg...}]}
    - {"content": [{...seg...}, {...seg...}]}
    - {"reply_message": {"message_segment": {...}}}
    """

    found: List[str] = []

    def _walk(obj: Any, depth: int) -> None:
        if obj is None or depth > max_depth:
            return

        # 1) 直接当 segment / seglist 解析
        found.extend(extract_image_base64_list(obj))

        # 2) 常见容器结构
        if isinstance(obj, dict):
            ms = obj.get("message_segment")
            if ms is not None:
                found.extend(extract_image_base64_list(ms))

            msg_list = obj.get("message")
            if isinstance(msg_list, list):
                found.extend(extract_image_base64_list({"type": "seglist", "data": msg_list}))

            content_list = obj.get("content")
            if isinstance(content_list, list):
                found.extend(extract_image_base64_list({"type": "seglist", "data": content_list}))

            # 递归遍历
            for v in obj.values():
                _walk(v, depth + 1)
            return

        if isinstance(obj, list):
            for it in obj:
                _walk(it, depth + 1)
            return

    _walk(payload, 0)

    # 去重（保持顺序）
    deduped: List[str] = []
    seen = set()
    for s in found:
        if not isinstance(s, str):
            continue
        ss = s.strip()
        if not ss:
            continue
        key = ss[:64]  # 避免用整段base64做set键造成内存浪费
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ss)

    return deduped


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
        # 允许 ```json\n...\n```
        if "\n" in t:
            first, rest = t.split("\n", 1)
            if first.strip().lower() in ("json", "javascript"):
                t = rest.strip()
    return t


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出中尽力解析 JSON 对象。
    解析策略：
    1) 去掉代码块包裹
    2) 直接 json.loads
    3) 提取第一个 {...} 片段再 json.loads
    """
    raw = _strip_code_fence(text)
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 兜底：提取首个 JSON 对象
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _dedup_tags(tags: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def normalize_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化输出，确保字段存在且类型可用，并补齐 PROMPT/NEGATIVE。

    字段含义：
    - CHARACTER_TAG: 角色标签（danbooru character tag）
    - WORK_TAG: 作品/版权标签（danbooru copyright tag）
    - TAG: 通用标签
    - BAD_TAG: 负面提示词（仅“瑕疵/问题/不想要的元素”，不要写“与图片相反”的否定tag）
    - PROMPT: 可直接复制给 NAI 的 prompt（CHARACTER_TAG + WORK_TAG + TAG）
    - NEGATIVE: 可直接复制给 NAI 的 negative prompt（BAD_TAG）
    """
    character = obj.get("CHARACTER_TAG", [])
    work = obj.get("WORK_TAG", [])
    tags = obj.get("TAG", [])
    bad = obj.get("BAD_TAG", [])

    if not isinstance(character, list):
        character = []
    if not isinstance(work, list):
        work = []
    if not isinstance(tags, list):
        tags = []
    if not isinstance(bad, list):
        bad = []

    character_s = _dedup_tags([str(x) for x in character if isinstance(x, (str, int, float))])
    work_s = _dedup_tags([str(x) for x in work if isinstance(x, (str, int, float))])
    tags_s = _dedup_tags([str(x) for x in tags if isinstance(x, (str, int, float))])
    bad_s = _dedup_tags([str(x) for x in bad if isinstance(x, (str, int, float))])

    prompt = obj.get("PROMPT")
    negative = obj.get("NEGATIVE")

    if not isinstance(prompt, str) or not prompt.strip():
        prompt = ", ".join(_dedup_tags(character_s + work_s + tags_s))
    if not isinstance(negative, str) or not negative.strip():
        negative = ", ".join(_dedup_tags(bad_s))

    return {
        "CHARACTER_TAG": character_s,
        "WORK_TAG": work_s,
        "TAG": tags_s,
        "BAD_TAG": bad_s,
        "PROMPT": prompt.strip(),
        "NEGATIVE": negative.strip(),
    }
