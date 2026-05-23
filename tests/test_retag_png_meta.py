# -*- coding: utf-8 -*-
"""PNG 元数据反推单元测试。"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.retag.png_meta import (
    extract_prompt_from_png,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """组装单个 PNG chunk: length(4) + type(4) + data + crc(4)。"""
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _png_signature() -> bytes:
    return b"\x89PNG\r\n\x1a\n"


def _make_png_with_text(text_chunks: list[tuple[str, str]]) -> bytes:
    """构造一个 1x1 RGB PNG，附带若干 tEXt 块。"""
    sig = _png_signature()
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    parts = [sig, _png_chunk(b"IHDR", ihdr)]
    for keyword, value in text_chunks:
        text_payload = keyword.encode("latin-1") + b"\x00" + value.encode("latin-1")
        parts.append(_png_chunk(b"tEXt", text_payload))
    idat = zlib.compress(b"\x00\x00\x00\x00")
    parts.append(_png_chunk(b"IDAT", idat))
    parts.append(_png_chunk(b"IEND", b""))
    return b"".join(parts)


def test_nai_comment_json_returns_only_positive_prompt() -> None:
    """NAI Comment 是 JSON：只读 prompt，uc/negative 必须丢弃。"""
    comment = json.dumps(
        {
            "prompt": "1girl, long hair, blue eyes, school uniform",
            "uc": "lowres, bad anatomy, worst quality",
            "steps": 28,
        }
    )
    png_bytes = _make_png_with_text([("Comment", comment)])

    result = extract_prompt_from_png(png_bytes)

    assert result is not None
    assert result.tags == ["1girl", "long hair", "blue eyes", "school uniform"]
    assert result.prompt == "1girl, long hair, blue eyes, school uniform"
    # 负面信息绝不允许混入正向输出
    assert "lowres" not in result.prompt
    assert "bad anatomy" not in result.prompt


def test_sd_webui_parameters_returns_only_positive_prompt() -> None:
    """SD WebUI 风格 parameters：Negative prompt / Steps 必须被丢弃。"""
    raw = (
        "masterpiece, best quality, 1girl, blue sky\n"
        "Negative prompt: lowres, bad anatomy, blurry\n"
        "Steps: 28, Sampler: Euler a, CFG scale: 7.0"
    )
    png_bytes = _make_png_with_text([("parameters", raw)])

    result = extract_prompt_from_png(png_bytes)

    assert result is not None
    assert "masterpiece" in result.tags
    assert "1girl" in result.tags
    assert "blue sky" in result.tags
    assert "lowres" not in result.prompt
    assert "bad anatomy" not in result.prompt
    # Steps 行不应该被吞进 tag
    assert all("Steps:" not in tag and "Sampler:" not in tag for tag in result.tags)


def test_generic_prompt_field_fallback() -> None:
    """没有 Comment / parameters，但有通用 prompt 字段。"""
    png_bytes = _make_png_with_text([("prompt", "wide shot, dramatic lighting, masterpiece")])

    result = extract_prompt_from_png(png_bytes)

    assert result is not None
    assert result.tags == ["wide shot", "dramatic lighting", "masterpiece"]


def test_no_text_chunks_returns_none() -> None:
    """没有 tEXt 块时应该返回 None，由上层走 WD14 兜底。"""
    png_bytes = _png_signature() + _png_chunk(
        b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ) + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + _png_chunk(b"IEND", b"")

    result = extract_prompt_from_png(png_bytes)

    assert result is None


def test_invalid_png_signature_returns_none() -> None:
    """非 PNG 数据（JPEG 头）必须安全返回 None。"""
    fake = b"\xff\xd8\xff\xe0" + b"\x00" * 50
    result = extract_prompt_from_png(fake)
    assert result is None


def test_corrupted_chunk_does_not_raise() -> None:
    """有损坏的 chunk 时应该跳过损坏块，不能抛异常。"""
    # 有效签名 + 一个长度声明大于剩余数据的伪 chunk
    bad = _png_signature() + struct.pack(">I", 9_999_999) + b"tEXt" + b"oops"
    result = extract_prompt_from_png(bad)
    # 没解析出 prompt 但不应崩
    assert result is None


def test_nai_comment_non_json_treated_as_plain_prompt() -> None:
    """Comment 字段不是 JSON 时，按纯文本 prompt 处理。"""
    png_bytes = _make_png_with_text([("Comment", "solo, 1boy, blonde hair")])
    result = extract_prompt_from_png(png_bytes)
    assert result is not None
    assert result.tags == ["solo", "1boy", "blonde hair"]


def test_empty_prompt_after_strip_returns_none() -> None:
    """prompt 字段是空白字符串时应当视为无效。"""
    comment = json.dumps({"prompt": "   ,  ,  ", "uc": "ignored"})
    png_bytes = _make_png_with_text([("Comment", comment)])
    result = extract_prompt_from_png(png_bytes)
    assert result is None
