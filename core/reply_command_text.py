from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_NAI_COMMAND_PREFIX_RE = re.compile(r"^\s*/nai0?(?:\s|$)")


def normalize_reply_command_text(message: Mapping[str, Any]) -> str | None:
    """Return command-scan text that excludes quoted reply content when needed."""
    raw_message = message.get("raw_message")
    if not isinstance(raw_message, list):
        return None

    has_reply = any(_segment_type(segment) == "reply" for segment in raw_message)
    if not has_reply:
        return None

    current_text = _extract_current_message_text(raw_message)
    if _looks_like_nai_command(current_text):
        return current_text

    processed_text = str(message.get("processed_plain_text") or "")
    if _reply_target_has_nai_command(raw_message) and _looks_like_nai_command(processed_text):
        return f"[reply]{current_text}" if current_text else "[reply]"

    return None


def _extract_current_message_text(raw_message: list[Any]) -> str:
    parts: list[str] = []
    for segment in raw_message:
        segment_type = _segment_type(segment)
        if segment_type == "reply":
            continue

        if segment_type == "text":
            parts.append(str(_segment_data(segment) or ""))
        elif segment_type == "at":
            at_name = _extract_at_name(_segment_data(segment))
            if at_name:
                parts.append(f"@{at_name}")
        elif segment_type in {"image", "emoji", "voice", "forward"}:
            parts.append(f"[{segment_type}]")

    return "".join(parts).strip()


def _reply_target_has_nai_command(raw_message: list[Any]) -> bool:
    for segment in raw_message:
        if _segment_type(segment) != "reply":
            continue
        data = _segment_data(segment)
        if not isinstance(data, Mapping):
            continue
        if _looks_like_nai_command(str(data.get("target_message_content") or "")):
            return True
    return False


def _looks_like_nai_command(text: str) -> bool:
    return bool(_NAI_COMMAND_PREFIX_RE.match(text))


def _segment_type(segment: Any) -> str:
    if not isinstance(segment, Mapping):
        return ""
    return str(segment.get("type") or "").strip().lower()


def _segment_data(segment: Any) -> Any:
    if not isinstance(segment, Mapping):
        return None
    return segment.get("data")


def _extract_at_name(data: Any) -> str:
    if not isinstance(data, Mapping):
        return ""
    return str(
        data.get("target_user_cardname")
        or data.get("target_user_nickname")
        or data.get("target_user_id")
        or ""
    ).strip()
