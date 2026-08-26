# -*- coding: utf-8 -*-
"""反推结果回灌 /nai0 的往返校验。

反推命令的回复文案宣称"可直接用于 /nai0 <prompt>"。多角色图的反推结果是
``global,\\nchar1:...,\\nchar2:...`` 多行格式，而 /nai0 是不过 LLM 的直发路径：
如果它不反解 charN，就会把字面量 `char1:` 当成 tag 送进 NAI 画出来。
这个文件锁住"反推输出 → /nai0 反解"这一环，避免两端各自改动后悄悄脱节。
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.retag.png_meta import extract_prompt_from_png
from plugins.nai_draw_plugin.core.utils.prompt_output_parser import (
    extract_multi_character_payload_from_text,
)


def _png_with_comment(comment: str) -> bytes:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"tEXt", b"Comment\x00" + comment.encode("utf-8"))
        + chunk(b"IDAT", zlib.compress(b"\x00" * 5))
        + chunk(b"IEND", b"")
    )


def _structured_comment(structured_key: str) -> str:
    return json.dumps(
        {
            "prompt": "",
            structured_key: {
                "caption": {
                    "base_caption": "rating:general, 2girls, indoor, soft lighting",
                    "char_captions": [
                        {"char_caption": "girl, in foreground, source#hug, gentle smile"},
                        {"char_caption": "girl, beside girl, target#hug, closed eyes"},
                    ],
                },
                "use_coords": False,
                "use_order": True,
            },
            "uc": "lowres, bad anatomy",
        },
        ensure_ascii=False,
    )


def test_multi_character_reverse_output_round_trips_into_characters() -> None:
    """多角色反推结果必须能被 /nai0 反解回 characters[]，而不是当字面 tag 送出。"""

    for structured_key in ("v4_prompt", "v4_5_prompt", "v5_prompt"):
        reversed_result = extract_prompt_from_png(
            _png_with_comment(_structured_comment(structured_key))
        )
        assert reversed_result is not None, f"{structured_key} 反推失败"

        payload = extract_multi_character_payload_from_text(reversed_result.prompt)
        assert payload is not None, f"{structured_key} 的反推结果无法被 /nai0 反解"
        assert payload["global_text"] == "rating:general, 2girls, indoor, soft lighting"
        assert [item["prompt"] for item in payload["characters"]] == [
            "girl, in foreground, source#hug, gentle smile",
            "girl, beside girl, target#hug, closed eyes",
        ]
        # 反推路径拿不到坐标，必须交由后端自动布局
        assert payload["has_coords"] is False
        # 负面绝不允许经由往返混进正向
        assert "lowres" not in reversed_result.prompt


def test_single_character_reverse_output_stays_on_the_string_path() -> None:
    """单人反推结果不能被误判成多人，否则会白白走 characters[] 通道。"""

    reversed_result = extract_prompt_from_png(
        _png_with_comment(json.dumps({"prompt": "1girl, solo, smile", "uc": "lowres"}))
    )
    assert reversed_result is not None
    assert "\n" not in reversed_result.prompt
    assert extract_multi_character_payload_from_text(reversed_result.prompt) is None


def test_nai0_send_payload_splits_reversed_multi_character_prompt() -> None:
    """_select_send_payload 是 /nai0 与 LLM 路径共用的分流点，行为必须一致。"""

    from plugins.nai_draw_plugin.sdk_runtime import NaiInvocation

    reversed_result = extract_prompt_from_png(
        _png_with_comment(_structured_comment("v5_prompt"))
    )
    structured = extract_multi_character_payload_from_text(reversed_result.prompt)
    prompt, characters = NaiInvocation._select_send_payload(
        reversed_result.prompt, structured
    )

    assert prompt == "rating:general, 2girls, indoor, soft lighting"
    assert characters is not None and len(characters) == 2
    assert "char1:" not in prompt
    assert all("char" not in item["prompt"].split(",")[0] for item in characters)

    # 单人串必须原样走字符串通道
    single_prompt, single_characters = NaiInvocation._select_send_payload(
        "1girl, solo, smile", None
    )
    assert single_prompt == "1girl, solo, smile"
    assert single_characters is None
