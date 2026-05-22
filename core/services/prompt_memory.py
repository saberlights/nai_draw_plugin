# -*- coding: utf-8 -*-
"""
提示词记忆（仅用于 Action 生图）

需求目标：
- 每次 action 生图时，将上一轮 LLM 生成的正向提示词注入到本轮模板中
- 重启后也能从 ActionRecords 中恢复上一轮提示词
- 支持三档继承规则（微调 / 换角色保场景 / 全新主题）
- 支持 TTL 过期机制

注意：
- 这里只记录 LLM 生成的正向提示词（不包含自拍补充、画师串、负面提示词等）
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

try:
    from src.common.logger import get_logger  # type: ignore

    logger = get_logger("nai_draw_plugin")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("nai_draw_plugin")


LAST_PROMPT_RECORD_PREFIX = "NAI_LAST_PROMPT:"
_REQ_LINE_PREFIX = "REQ:"
_REQ_SEPARATOR = "---"


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render_previous_prompt_block(
    last_prompt: Optional[str],
    last_request: Optional[str] = None,
) -> str:
    """Generate the replacement content for the <<PREVIOUS_PROMPT>> placeholder.

    Returns an XML block with three-tier inheritance rules when *last_prompt*
    is present, or a placeholder block when there is no previous prompt.
    """
    previous = (last_prompt or "").strip()
    if not previous:
        return (
            "<previous_prompt_context>\n"
            "（无上一轮提示词，请完全按照本次用户请求生成全新提示词）\n"
            "</previous_prompt_context>"
        )

    # 可选注入上一轮用户请求（帮助 LLM 做 diff 推理）
    request_section = ""
    req = (last_request or "").strip()
    if req:
        request_section = f"\n【上一轮用户请求】\n{req}\n"

    parts = [
        "<previous_prompt_context>\n",
        "【上一轮 LLM 生成的提示词（系统注入，非用户输入的英文tag）】\n",
        previous, "\n",
        request_section, "\n",
        "【三档继承规则（必须遵守）】\n",
        "请对比本次用户请求与上一轮提示词，判断属于以下哪档：\n\n",
        'A. 微调（同一主题的细节调整，如「把背景换成夜晚」、「加个帽子」）\n',
        "   → 以上方提示词为底稿，仅修改用户要求变更的部分，保留其余标签\n\n",
        'B. 换角色保场景（角色变了但场景/构图延续，如「换成另一个角色」、「画成男生版」）\n',
        "   → 保留场景、构图、氛围等环境标签，替换角色相关标签（外貌、服装、身份等）\n\n",
        "C. 全新主题（与上一轮完全无关的新请求）\n",
        "   → 完全忽略上方提示词，按用户请求重新生成\n",
        "</previous_prompt_context>",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# 持久化格式解析
# ---------------------------------------------------------------------------

def extract_last_context_from_record_display(
    action_prompt_display: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract (prompt, request) from ActionRecords.action_prompt_display.

    Supports both new format (with ``REQ:`` line) and old format (plain prompt).

    New format::

        NAI_LAST_PROMPT:
        REQ:用户原始请求
        ---
        prompt tags here

    Old format::

        NAI_LAST_PROMPT:
        prompt tags here
    """
    text = action_prompt_display if isinstance(action_prompt_display, str) else ""
    if not text:
        return None, None
    if not text.startswith(LAST_PROMPT_RECORD_PREFIX):
        return None, None

    body = text[len(LAST_PROMPT_RECORD_PREFIX):].lstrip("\n")

    # Try new format: first non-empty line starts with REQ:
    lines = body.split("\n")
    first_line = ""
    first_idx = 0
    for i, ln in enumerate(lines):
        if ln.strip():
            first_line = ln.strip()
            first_idx = i
            break

    if first_line.startswith(_REQ_LINE_PREFIX):
        request = first_line[len(_REQ_LINE_PREFIX):].strip() or None
        # Find separator ---
        prompt_start = first_idx + 1
        for j in range(first_idx + 1, len(lines)):
            if lines[j].strip() == _REQ_SEPARATOR:
                prompt_start = j + 1
                break
        prompt = "\n".join(lines[prompt_start:]).strip() or None
        return prompt, request

    # Old format: entire body is the prompt
    prompt = body.strip() or None
    return prompt, None


def extract_last_prompt_from_record_display(
    action_prompt_display: str,
) -> Optional[str]:
    """Compat wrapper — returns only the prompt part."""
    prompt, _ = extract_last_context_from_record_display(action_prompt_display)
    return prompt


def load_last_context_from_action_records(
    chat_stream_id: str,
    action_name: str,
    limit: int = 50,
    ttl: float = 0,
) -> Tuple[Optional[str], Optional[str]]:
    """Read (prompt, request) from ActionRecords.

    When *ttl* > 0, records whose ``time`` field is older than *ttl* seconds
    are skipped.
    """
    if not chat_stream_id or not action_name:
        return None, None

    try:
        from src.common.database.database_model import ActionRecords
    except Exception as e:
        logger.debug(f"[prompt_memory] ActionRecords import failed: {e}")
        return None, None

    try:
        records = (
            ActionRecords.select(
                ActionRecords.action_prompt_display,
                ActionRecords.time,
            )
            .where(
                (ActionRecords.chat_id == chat_stream_id)
                & (ActionRecords.action_name == action_name)
            )
            .order_by(ActionRecords.time.desc())
            .limit(max(1, int(limit)))
        )
        cutoff = time.time() - ttl if ttl > 0 else 0
        for r in records:
            if ttl > 0:
                record_time = getattr(r, "time", None)
                if record_time is not None:
                    # record_time may be a datetime or a float timestamp
                    ts = record_time
                    if hasattr(ts, "timestamp"):
                        ts = ts.timestamp()
                    if ts < cutoff:
                        continue
            display = getattr(r, "action_prompt_display", "") or ""
            prompt, request = extract_last_context_from_record_display(display)
            if prompt:
                return prompt, request
        return None, None
    except Exception as e:
        logger.warning(f"[prompt_memory] Failed to read ActionRecords: {e}")
        return None, None


def load_last_prompt_from_action_records(
    chat_stream_id: str, action_name: str, limit: int = 50
) -> Optional[str]:
    """Compat wrapper — returns only the prompt part (no TTL)."""
    prompt, _ = load_last_context_from_action_records(
        chat_stream_id, action_name, limit=limit
    )
    return prompt
