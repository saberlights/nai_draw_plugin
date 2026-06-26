# -*- coding: utf-8 -*-
"""Danbooru tag 检索结果显示辅助。"""


_TAG_RETRIEVER_DISPLAY_HEADER = "🔎 Danbooru Tag 检索结果:"
_TAG_CANDIDATE_WRAPPERS = {"<tag_candidates>", "</tag_candidates>"}
_DEFAULT_MAX_DISPLAY_CHARS = 900


def _normalize_body_lines(tag_candidates_block: str) -> list[str]:
    """提取可直接展示的正文行。"""
    if not isinstance(tag_candidates_block, str):
        return []

    body_lines: list[str] = []
    for raw_line in tag_candidates_block.splitlines():
        normalized = raw_line.strip()
        if not normalized or normalized in _TAG_CANDIDATE_WRAPPERS:
            continue
        body_lines.append(raw_line.rstrip())
    return body_lines


def _split_line_by_width(line: str, max_width: int) -> list[str]:
    """极长单行按字符宽度硬切，避免单条消息仍然超限。"""
    normalized_width = max(1, int(max_width))
    if len(line) <= normalized_width:
        return [line]
    return [line[index : index + normalized_width] for index in range(0, len(line), normalized_width)]


def build_tag_retriever_display_messages(
    tag_candidates_block: str,
    *,
    max_chars: int = _DEFAULT_MAX_DISPLAY_CHARS,
) -> list[str]:
    """把内部 ``<tag_candidates>`` 文本块转换成 1~N 条用户可读消息。"""
    body_lines = _normalize_body_lines(tag_candidates_block)
    if not body_lines:
        return []

    normalized_max_chars = max(len(_TAG_RETRIEVER_DISPLAY_HEADER) + 16, int(max_chars))
    max_header_length = len(_TAG_RETRIEVER_DISPLAY_HEADER) + 10
    max_body_chars = max(1, normalized_max_chars - max_header_length - 1)
    chunk_body_width = max_body_chars

    flattened_lines: list[str] = []
    for line in body_lines:
        flattened_lines.extend(_split_line_by_width(line, chunk_body_width))

    line_groups: list[list[str]] = []
    current_group: list[str] = []
    current_length = 0

    for line in flattened_lines:
        added_length = len(line) if not current_group else len(line) + 1
        if current_group and current_length + added_length > max_body_chars:
            line_groups.append(current_group)
            current_group = [line]
            current_length = len(line)
            continue
        current_group.append(line)
        current_length += added_length

    if current_group:
        line_groups.append(current_group)

    total = len(line_groups)
    messages: list[str] = []
    for index, group in enumerate(line_groups, start=1):
        header = (
            _TAG_RETRIEVER_DISPLAY_HEADER
            if total == 1
            else f"{_TAG_RETRIEVER_DISPLAY_HEADER[:-1]} ({index}/{total}):"
        )
        messages.append(header + "\n" + "\n".join(group))
    return messages


def build_tag_retriever_display_message(tag_candidates_block: str) -> str:
    """把内部 ``<tag_candidates>`` 文本块转换成用户可读消息。"""
    messages = build_tag_retriever_display_messages(tag_candidates_block)
    return messages[0] if messages else ""
