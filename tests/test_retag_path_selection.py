# -*- coding: utf-8 -*-
"""反推路径分发的单元测试。

锁住四条路径各自的边界：显式选定的路径不许悄悄走到别的链路上去。这是本功能最容易
退化的地方——历史实现是写死的"元数据 → WD14"瀑布，任何一次"顺手加个兜底"都会让
`/nai 反推 llm` 在视觉模型不可用时静默返回一串 WD14 tag，用户拿到的东西完全不是他要的。
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.retag.reverser import (
    KIND_NATURAL_LANGUAGE,
    KIND_TAGS,
    PATH_AUTO,
    PATH_LLM,
    PATH_META,
    PATH_WD14,
    REVERSE_PATHS,
    ReverseService,
    normalize_path,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _make_nai_png(prompt: str) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    text_payload = b"Comment\x00" + json.dumps(
        {"prompt": prompt, "uc": "lowres, bad anatomy"}
    ).encode("latin-1")
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text_payload)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


_JPEG_WITHOUT_METADATA = b"\xff\xd8\xff\xe0" + b"\x00" * 1024


class _FakeWD14Client:
    command_timeout = 5.0

    def __init__(self, *, result: Dict[str, Any] | None = None) -> None:
        self.result = result
        self.calls: list[str] = []

    async def tag_image(
        self,
        image_base64: str,
        threshold: float = 0.35,
        character_threshold: float | None = None,
    ) -> Dict[str, Any]:
        self.calls.append(image_base64[:16])
        return self.result or {"tags": []}


class _FakeCaptioner:
    """伪造视觉模型，记录被喂进去的 sfw 开关并按脚本返回。"""

    def __init__(self, *, caption: str = "", raise_exc: Exception | None = None) -> None:
        self.caption_text = caption
        self.raise_exc = raise_exc
        self.calls: list[bool] = []

    async def caption(self, *, image_bytes: bytes, sfw: bool) -> str:
        del image_bytes
        self.calls.append(sfw)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.caption_text


def _service(
    *,
    wd14: _FakeWD14Client | None = None,
    captioner: _FakeCaptioner | None = None,
    wd14_enabled: bool = True,
) -> ReverseService:
    return ReverseService(
        wd14_client=wd14,
        wd14_enabled=wd14_enabled,
        captioner=captioner,
    )


# ── normalize_path ────────────────────────────────────────────────────────


def test_normalize_path_accepts_every_registered_path() -> None:
    for path in REVERSE_PATHS:
        assert normalize_path(path) == path


def test_normalize_path_is_case_insensitive_and_trims() -> None:
    assert normalize_path("  LLM  ") == PATH_LLM


def test_normalize_path_falls_back_to_default_only_for_blank() -> None:
    assert normalize_path("", default=PATH_WD14) == PATH_WD14
    assert normalize_path(None) == PATH_AUTO


def test_normalize_path_rejects_unknown_value() -> None:
    """非法值必须抛出而不是静默回落到 auto——否则打错字的用户永远发现不了。"""
    with pytest.raises(ValueError) as excinfo:
        normalize_path("vlm")
    assert "vlm" in str(excinfo.value)
    assert "auto/meta/wd14/llm" in str(excinfo.value)


# ── llm 路径 ──────────────────────────────────────────────────────────────


def test_llm_path_skips_metadata_even_when_png_has_prompt() -> None:
    """显式选 llm 时，即使 PNG 里就躺着原始 prompt 也必须走视觉模型。

    元数据给的是 tag 串，用户要的是自然语言描述；"命中元数据就直接返回"会让
    /nai 反推 llm 对着一张 NAI 原图永远拿不到自然语言。
    """
    captioner = _FakeCaptioner(caption="A young woman sits on a rooftop railing.")
    wd14 = _FakeWD14Client(result={"tags": [{"label": "unused", "score": 0.9}]})
    svc = _service(wd14=wd14, captioner=captioner)

    result = asyncio.run(
        svc.reverse(_make_nai_png("1girl, smile"), path=PATH_LLM, sfw=True)
    )

    assert result.source == "llm"
    assert result.kind == KIND_NATURAL_LANGUAGE
    assert result.prompt == "A young woman sits on a rooftop railing."
    assert result.tags == []
    assert wd14.calls == []
    assert captioner.calls == [True]


def test_llm_path_forwards_nsfw_flag() -> None:
    """会话关掉 NSFW 过滤时，sfw=False 必须原样传到 captioner。"""
    captioner = _FakeCaptioner(caption="She is lying on the bed.")
    svc = _service(captioner=captioner)

    asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path=PATH_LLM, sfw=False))

    assert captioner.calls == [False]


def test_llm_path_failure_does_not_fall_back_to_wd14() -> None:
    """视觉模型报错时必须原样暴露，不能悄悄拿 WD14 的 tag 顶包。"""
    captioner = _FakeCaptioner(raise_exc=RuntimeError("视觉模型拒答"))
    wd14 = _FakeWD14Client(result={"tags": [{"label": "1girl", "score": 0.9}]})
    svc = _service(wd14=wd14, captioner=captioner)

    result = asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path=PATH_LLM))

    assert result.source == "failed"
    assert result.prompt == ""
    assert "视觉模型拒答" in (result.detail or "")
    assert wd14.calls == []


def test_llm_path_without_captioner_reports_disabled() -> None:
    svc = _service(wd14=_FakeWD14Client(result={"tags": [{"label": "x", "score": 1.0}]}))

    result = asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path=PATH_LLM))

    assert result.source == "failed"
    assert "llm_enabled" in (result.detail or "")


# ── meta 路径 ─────────────────────────────────────────────────────────────


def test_meta_path_never_calls_wd14_on_miss() -> None:
    """选 meta 就是"不联网"，未命中要直接失败而不是偷偷联网。"""
    wd14 = _FakeWD14Client(result={"tags": [{"label": "1girl", "score": 0.9}]})
    svc = _service(wd14=wd14)

    result = asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path=PATH_META))

    assert result.source == "failed"
    assert wd14.calls == []
    assert "仅读元数据" in (result.detail or "")


def test_meta_path_returns_metadata_on_hit() -> None:
    svc = _service(wd14=_FakeWD14Client())

    result = asyncio.run(svc.reverse(_make_nai_png("1girl, smile"), path=PATH_META))

    assert result.source == "metadata"
    assert result.kind == KIND_TAGS
    assert result.tags == ["1girl", "smile"]


# ── wd14 路径 ─────────────────────────────────────────────────────────────


def test_wd14_path_skips_metadata_lookup() -> None:
    """显式选 wd14 时，带元数据的原图也要走 Space，不能被元数据抢答。"""
    wd14 = _FakeWD14Client(result={"tags": [{"label": "from wd14", "score": 0.9}]})
    svc = _service(wd14=wd14)

    result = asyncio.run(svc.reverse(_make_nai_png("1girl, smile"), path=PATH_WD14))

    assert result.source == "wd14"
    assert result.tags == ["from wd14"]
    assert len(wd14.calls) == 1


def test_wd14_path_failure_detail_omits_metadata_reason() -> None:
    """没走元数据的路径，失败详情里不该出现"元数据未命中原因"这种误导信息。"""
    svc = _service(wd14=None, wd14_enabled=False)

    result = asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path=PATH_WD14))

    assert result.source == "failed"
    assert "元数据" not in (result.detail or "")


# ── auto 路径（历史行为不许变） ────────────────────────────────────────────


def test_auto_path_keeps_metadata_then_wd14_waterfall() -> None:
    wd14 = _FakeWD14Client(result={"tags": [{"label": "1girl", "score": 0.9}]})
    captioner = _FakeCaptioner(caption="never used")
    svc = _service(wd14=wd14, captioner=captioner)

    hit = asyncio.run(svc.reverse(_make_nai_png("1girl, smile"), path=PATH_AUTO))
    assert hit.source == "metadata"
    assert wd14.calls == []

    miss = asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path=PATH_AUTO))
    assert miss.source == "wd14"
    assert len(wd14.calls) == 1

    # auto 永远不碰视觉模型：llm 是要花钱的独立路径，不能被当成兜底悄悄触发
    assert captioner.calls == []


def test_auto_is_the_default_path() -> None:
    wd14 = _FakeWD14Client(result={"tags": [{"label": "1girl", "score": 0.9}]})
    svc = _service(wd14=wd14)

    result = asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA))

    assert result.source == "wd14"


def test_unknown_path_raises_before_touching_any_backend() -> None:
    wd14 = _FakeWD14Client(result={"tags": [{"label": "x", "score": 1.0}]})
    captioner = _FakeCaptioner(caption="x")
    svc = _service(wd14=wd14, captioner=captioner)

    with pytest.raises(ValueError):
        asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path="danbooru"))

    assert wd14.calls == []
    assert captioner.calls == []


def test_update_captioner_swaps_instance() -> None:
    """配置热更新换掉 captioner 后，下一次反推必须用新实例。"""
    first = _FakeCaptioner(caption="old")
    second = _FakeCaptioner(caption="new")
    svc = _service(captioner=first)

    svc.update_captioner(second)
    result = asyncio.run(svc.reverse(_JPEG_WITHOUT_METADATA, path=PATH_LLM))

    assert result.prompt == "new"
    assert first.calls == []
