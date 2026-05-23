# -*- coding: utf-8 -*-
"""反推编排服务的单元测试。"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.retag.reverser import (
    ReverseService,
    _flatten_wd14_tags,
)
from plugins.nai_draw_plugin.core.retag.wd14_client import WD14ClientError


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _make_nai_png(prompt: str, uc: str = "lowres, bad anatomy") -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    text_payload = b"Comment\x00" + json.dumps({"prompt": prompt, "uc": uc}).encode("latin-1")
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text_payload)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


class _FakeWD14Client:
    """伪造 WD14Client，记录调用并按脚本返回结果。"""

    command_timeout = 5.0

    def __init__(self, *, result: Dict[str, Any] | None = None, raise_exc: Exception | None = None) -> None:
        self.result = result
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, float, float]] = []

    async def tag_image(
        self,
        image_base64: str,
        threshold: float = 0.35,
        character_threshold: float | None = None,
    ) -> Dict[str, Any]:
        self.calls.append((image_base64[:16], threshold, float(character_threshold or 0.0)))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result or {"tags": []}


def test_metadata_hit_does_not_call_wd14() -> None:
    """PNG 元数据命中时绝不应该再请求 WD14。"""
    fake = _FakeWD14Client(result={"tags": [{"label": "should_not_be_used", "score": 0.99}]})
    svc = ReverseService(wd14_client=fake, wd14_enabled=True)

    png = _make_nai_png("1girl, smile, looking at viewer")
    result = asyncio.run(svc.reverse(png))

    assert result.source == "metadata"
    assert result.tags == ["1girl", "smile", "looking at viewer"]
    assert "lowres" not in result.prompt  # 负面不带出
    assert fake.calls == []


def test_metadata_miss_falls_back_to_wd14() -> None:
    """非 PNG / 无元数据时应该走 WD14。"""
    fake = _FakeWD14Client(
        result={
            "tags": [
                {"label": "1girl", "score": 0.95},
                {"label": "long hair", "score": 0.88},
                {"label": "1girl", "score": 0.55},  # 重复，应该被 _flatten_wd14_tags 去重
            ]
        }
    )
    svc = ReverseService(wd14_client=fake, wd14_enabled=True)

    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 1024
    result = asyncio.run(svc.reverse(fake_jpeg))

    assert result.source == "wd14"
    assert result.tags == ["1girl", "long hair"]
    assert result.prompt == "1girl, long hair"
    assert len(fake.calls) == 1


def test_wd14_disabled_returns_failed() -> None:
    """配置关闭 WD14 时，对没有元数据的图直接返回 failed。"""
    fake = _FakeWD14Client(result={"tags": [{"label": "ignored", "score": 0.9}]})
    svc = ReverseService(wd14_client=fake, wd14_enabled=False)

    fake_bytes = b"random bytes" * 32
    result = asyncio.run(svc.reverse(fake_bytes))

    assert result.source == "failed"
    assert result.prompt == ""
    assert fake.calls == []


def test_wd14_error_surfaces_in_detail() -> None:
    """WD14 抛 WD14ClientError 时要把错误信息塞进 detail。"""
    fake = _FakeWD14Client(raise_exc=WD14ClientError("Space 都挂了"))
    svc = ReverseService(wd14_client=fake, wd14_enabled=True)

    result = asyncio.run(svc.reverse(b"not a png" * 32))

    assert result.source == "failed"
    assert "Space 都挂了" in (result.detail or "")


def test_empty_image_returns_failed_without_calling_wd14() -> None:
    """空 bytes 必须短路。"""
    fake = _FakeWD14Client(result={"tags": [{"label": "x", "score": 1.0}]})
    svc = ReverseService(wd14_client=fake, wd14_enabled=True)

    result = asyncio.run(svc.reverse(b""))
    assert result.source == "failed"
    assert fake.calls == []


def test_flatten_wd14_tags_preserves_order() -> None:
    raw = {
        "tags": [
            {"label": "a", "score": 0.9},
            {"label": "b", "score": 0.8},
            {"label": "", "score": 0.7},  # 空 label 跳过
            {"label": "a", "score": 0.6},  # 重复跳过
            {"label": "c", "score": 0.5},
        ]
    }
    assert _flatten_wd14_tags(raw) == ["a", "b", "c"]


def test_update_thresholds_takes_effect() -> None:
    """update_wd14_thresholds 改完后下一次调用应使用新阈值。"""
    fake = _FakeWD14Client(
        result={"tags": [{"label": "1girl", "score": 0.5}]}
    )
    svc = ReverseService(wd14_client=fake, wd14_enabled=True, wd14_threshold=0.35, wd14_character_threshold=0.8)
    svc.update_wd14_thresholds(threshold=0.5, character_threshold=0.9, enabled=True)

    asyncio.run(svc.reverse(b"definitely not a png" * 16))

    assert fake.calls and fake.calls[0][1] == 0.5 and fake.calls[0][2] == 0.9
