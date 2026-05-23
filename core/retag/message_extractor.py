"""消息回复与图片提取工具。"""

from __future__ import annotations

from typing import Any


_BASE64_IMAGE_PREFIXES = (
    "/9j/",
    "iVBORw",
    "R0lGOD",
    "UklGR",
)


def normalize_base64_data(data: str) -> str:
    """清理图片 base64 字符串中的前缀与换行。"""
    cleaned = str(data or "").strip()
    if cleaned.startswith("data:image/") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    return cleaned.replace("\n", "").replace("\r", "")


def find_reply_message_id(message_segment: Any) -> str | None:
    """从消息段中递归查找被引用消息 ID。"""
    if message_segment is None:
        return None

    if isinstance(message_segment, list):
        for item in message_segment:
            reply_message_id = find_reply_message_id(item)
            if reply_message_id:
                return reply_message_id
        return None

    segment_type = getattr(message_segment, "type", None)
    segment_data = getattr(message_segment, "data", None)
    if segment_type is None and isinstance(message_segment, dict):
        segment_type = message_segment.get("type")
        segment_data = message_segment.get("data")

    if segment_type == "reply":
        if isinstance(segment_data, (str, int)):
            return _clean_message_id(segment_data)

        if isinstance(segment_data, dict):
            for key in (
                "target_message_id",
                "message_id",
                "id",
                "reply_to",
                "reply_to_message_id",
                "reply_message_id",
                "quote_message_id",
                "reply_id",
            ):
                reply_message_id = _clean_message_id(segment_data.get(key))
                if reply_message_id:
                    return reply_message_id
        return None

    if segment_type == "seglist":
        return find_reply_message_id(segment_data if isinstance(segment_data, list) else [])

    return None


def extract_image_base64_list(message_segment: Any) -> list[str]:
    """从消息段中递归提取图片或表情的 base64 数据。"""
    if message_segment is None:
        return []

    if isinstance(message_segment, list):
        images: list[str] = []
        for item in message_segment:
            images.extend(extract_image_base64_list(item))
        return images

    segment_type = getattr(message_segment, "type", None)
    segment_data = getattr(message_segment, "data", None)
    if segment_type is None and isinstance(message_segment, dict):
        segment_type = message_segment.get("type")
        segment_data = message_segment.get("data")

    if segment_type in {"emoji", "image"}:
        image_base64 = _extract_image_data(message_segment) or _extract_image_data(segment_data)
        return [image_base64] if image_base64 else []

    if segment_type == "seglist":
        return extract_image_base64_list(segment_data if isinstance(segment_data, list) else [])

    return []


def extract_image_base64_list_from_payload(payload: Any, max_depth: int = 8) -> list[str]:
    """从任意 payload 中尽力提取嵌套的图片 base64 数据。"""
    found: list[str] = []

    def _walk(obj: Any, depth: int) -> None:
        if obj is None or depth > max_depth:
            return

        found.extend(extract_image_base64_list(obj))

        if isinstance(obj, dict):
            for value in obj.values():
                _walk(value, depth + 1)
            return

        if isinstance(obj, list):
            for item in obj:
                _walk(item, depth + 1)

    _walk(payload, 0)
    return _deduplicate_images(found)


def deep_find_first(payload: Any, keys: set[str], max_depth: int = 8) -> Any:
    """在嵌套 dict/list 结构中查找目标字段的第一个非空值。"""
    if payload is None or max_depth < 0:
        return None

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return value
        for value in payload.values():
            found = deep_find_first(value, keys, max_depth - 1)
            if found not in (None, ""):
                return found
        return None

    if isinstance(payload, list):
        for item in payload:
            found = deep_find_first(item, keys, max_depth - 1)
            if found not in (None, ""):
                return found

    return None


def _clean_message_id(value: Any) -> str | None:
    """将消息 ID 统一规范化为字符串。"""
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _deduplicate_images(images: list[str]) -> list[str]:
    """按顺序去重，避免重复扫描同一张图片。"""
    deduplicated: list[str] = []
    seen: set[str] = set()

    for image in images:
        normalized = normalize_base64_data(image)
        if not normalized:
            continue
        image_key = normalized[:64]
        if image_key in seen:
            continue
        seen.add(image_key)
        deduplicated.append(normalized)

    return deduplicated


def _extract_image_data(segment_data: Any) -> str | None:
    """从图片段的 data 字段中提取 base64。"""
    if isinstance(segment_data, str):
        return normalize_base64_data(segment_data) if _looks_like_image_data(segment_data) else None

    if isinstance(segment_data, dict):
        for key in ("binary_data_base64", "base64", "data", "content", "file"):
            value = segment_data.get(key)
            if isinstance(value, str) and _looks_like_image_data(value):
                return normalize_base64_data(value)

    return None


def _looks_like_image_data(value: str) -> bool:
    """粗略判断字符串是否像图片 base64 或 data URL。"""
    cleaned = str(value or "").strip()
    if not cleaned:
        return False

    if cleaned.startswith("data:image/"):
        return True

    if any(cleaned.startswith(prefix) for prefix in _BASE64_IMAGE_PREFIXES):
        return True

    if len(cleaned) < 80:
        return False

    return all(char.isalnum() or char in "+/=\n\r" for char in cleaned)
