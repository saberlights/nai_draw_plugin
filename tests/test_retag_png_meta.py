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
    describe_png_metadata,
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


# ==================== NAI V4+ 结构化 prompt ====================
# V4 起正向 prompt 搬进 v4_prompt.caption：base_caption 是全局段，
# char_captions 是角色段。只读顶层 prompt 时多角色图会整体反推不出来。


def _nai_structured_comment(
    *,
    top_level_prompt: str,
    structured_key: str,
    base_caption: str,
    char_captions: list[str],
    negative_base: str = "lowres, bad anatomy",
) -> str:
    return json.dumps(
        {
            "prompt": top_level_prompt,
            structured_key: {
                "caption": {
                    "base_caption": base_caption,
                    "char_captions": [
                        {"char_caption": caption, "centers": [{"x": 0.5, "y": 0.5}]}
                        for caption in char_captions
                    ],
                },
                "use_coords": False,
                "use_order": True,
            },
            f"{structured_key.replace('_prompt', '')}_negative_prompt": {
                "caption": {"base_caption": negative_base, "char_captions": []}
            },
            "uc": negative_base,
            "steps": 28,
        },
        ensure_ascii=False,
    )


def test_v4_multi_character_reverses_when_top_level_prompt_is_empty() -> None:
    """多角色 V4+ 图的顶层 prompt 可能为空，正向只在 char_captions 里。"""

    comment = _nai_structured_comment(
        top_level_prompt="",
        structured_key="v4_prompt",
        base_caption="rating:general, 2girls, indoor, soft lighting",
        char_captions=[
            "girl, in foreground, mutual#hug, gentle smile",
            "girl, beside girl, mutual#hug, closed eyes",
        ],
    )
    result = extract_prompt_from_png(_make_png_with_text([("Comment", comment)]))

    assert result is not None, "顶层 prompt 为空时不能整体反推失败"
    # 拍平成 charN: 格式，可直接回灌 /nai0 重画
    lines = result.prompt.split("\n")
    assert lines[0] == "rating:general, 2girls, indoor, soft lighting,"
    assert lines[1] == "char1:girl, in foreground, mutual#hug, gentle smile,"
    assert lines[2] == "char2:girl, beside girl, mutual#hug, closed eyes,"
    # 角色段必须完整保留，不能只剩全局段
    assert "mutual#hug" in result.tags
    assert "in foreground" in result.tags
    assert "beside girl" in result.tags
    # 负面绝不允许串进正向
    assert "lowres" not in result.prompt
    assert "bad anatomy" not in result.prompt


def test_v4_character_segments_survive_even_with_top_level_prompt() -> None:
    """顶层 prompt 存在时也不能丢角色段——它只有全局信息。"""

    comment = _nai_structured_comment(
        top_level_prompt="rating:general, 2girls, indoor",
        structured_key="v4_prompt",
        base_caption="rating:general, 2girls, indoor",
        char_captions=["girl, source#hug", "girl, target#hug"],
    )
    result = extract_prompt_from_png(_make_png_with_text([("Comment", comment)]))

    assert result is not None
    assert "source#hug" in result.tags
    assert "target#hug" in result.tags


def test_structured_prompt_key_matches_version_family() -> None:
    """键名按 v<数字>_prompt 家族匹配，V4.5 / V5 沿用同结构时无需改代码。"""

    for structured_key in ("v4_prompt", "v4_5_prompt", "v5_prompt", "v6_prompt"):
        comment = _nai_structured_comment(
            top_level_prompt="",
            structured_key=structured_key,
            base_caption="rating:general, 1girl, night",
            char_captions=["girl, standing"],
        )
        result = extract_prompt_from_png(_make_png_with_text([("Comment", comment)]))

        assert result is not None, f"{structured_key} 未被识别"
        assert "night" in result.tags
        assert "standing" in result.tags
        assert "lowres" not in result.prompt


def test_structured_negative_prompt_is_never_read_as_positive() -> None:
    """只有负面结构、没有正向时必须失败，绝不能把负面当正向返回。"""

    comment = json.dumps(
        {
            "prompt": "",
            "v4_negative_prompt": {
                "caption": {"base_caption": "lowres, bad anatomy", "char_captions": []}
            },
            "uc": "lowres, bad anatomy",
        }
    )
    result = extract_prompt_from_png(_make_png_with_text([("Comment", comment)]))

    assert result is None


# ==================== 失败原因必须可定位 ====================


def test_describe_reports_non_png_container() -> None:
    """不是 PNG 和「是 PNG 但没元数据」需要区分：前者要改出图格式。"""

    assert "JPEG" in describe_png_metadata(b"\xff\xd8\xff\xe0" + b"\x00" * 40)
    assert "WebP" in describe_png_metadata(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    assert "image_format" in describe_png_metadata(b"\xff\xd8\xff\xe0" + b"\x00" * 40)


def test_describe_reports_stripped_metadata() -> None:
    """平台重新编码会剥掉文本块，这跟 NAI 改结构是两回事。"""

    png_bytes = _png_signature() + _png_chunk(
        b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ) + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + _png_chunk(b"IEND", b"")

    reason = describe_png_metadata(png_bytes)
    assert "没有任何 tEXt" in reason


def test_describe_reports_available_comment_fields_when_prompt_missing() -> None:
    """NAI 改了结构时，要能直接从原因里看到 Comment 里实际有哪些字段。"""

    comment = json.dumps({"prompt": "", "steps": 28, "sampler": "k_euler", "brand_new_field": {}})
    reason = describe_png_metadata(_make_png_with_text([("Comment", comment)]))

    assert "没有可用的正向 prompt" in reason
    assert "brand_new_field" in reason
    assert "sampler" in reason


def test_describe_reports_missing_comment_chunk() -> None:
    reason = describe_png_metadata(_make_png_with_text([("Software", "NovelAI")]))
    assert "没有 NAI 的 Comment 字段" in reason
    assert "Software" in reason
