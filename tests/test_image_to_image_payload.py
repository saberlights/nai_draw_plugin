# -*- coding: utf-8 -*-
"""image_meta 工具与 §20 图生图 payload 路径的最小校验。"""

from __future__ import annotations

import base64
import struct
import sys
from pathlib import Path

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

# 与 test_nai_web_client 一致：dummy 掉 src.common.logger 以独立测试
import types

dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def __getattr__(self, _):  # noqa: D401
        return lambda *a, **k: None


dummy_logger_module.get_logger = lambda _name=None: _DummyLogger()
sys.modules["src.common.logger"] = dummy_logger_module


from plugins.nai_draw_plugin.core.clients.nai_web_client import NaiWebClient
from plugins.nai_draw_plugin.core.utils.image_meta import (
    normalize_image_base64,
    read_image_dimensions,
)


# ── 工具：制作最小 PNG / JPEG / WebP 头 ────────────────────────────────────────


def _make_png_bytes(width: int, height: int) -> bytes:
    """构造仅含 PNG signature + IHDR 的最小字节串，足够 read_image_dimensions 解析。"""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_length = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # IHDR 后面 CRC 在真实 PNG 里是必需，但解析逻辑不读 CRC，留 0 即可
    return signature + ihdr_length + ihdr_type + ihdr_data + b"\x00" * 4


def _make_jpeg_bytes(width: int, height: int) -> bytes:
    """构造 SOI + APP0 + SOF0 三段的最小 JPEG。"""
    soi = b"\xff\xd8"
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00" + b"\x00\x01" + b"\x00\x01" + b"\x00\x00"
    sof0_header = b"\xff\xc0"
    sof0_length = struct.pack(">H", 17)
    sof0_body = b"\x08" + struct.pack(">H", height) + struct.pack(">H", width) + b"\x03" + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    eoi = b"\xff\xd9"
    return soi + app0 + sof0_header + sof0_length + sof0_body + eoi


# ── normalize_image_base64 ──────────────────────────────────────────────────


def test_normalize_image_base64_strips_data_uri_prefix_and_whitespace() -> None:
    raw = "data:image/png;base64,iVBORw\nXYZ\r\n"
    assert normalize_image_base64(raw) == "iVBORwXYZ"


def test_normalize_image_base64_handles_empty_and_none() -> None:
    assert normalize_image_base64("") == ""
    assert normalize_image_base64(None) == ""  # type: ignore[arg-type]


# ── read_image_dimensions ────────────────────────────────────────────────────


def test_read_image_dimensions_parses_png_header() -> None:
    raw = _make_png_bytes(832, 1216)
    assert read_image_dimensions(raw) == (832, 1216)


def test_read_image_dimensions_parses_png_via_base64() -> None:
    raw = _make_png_bytes(1024, 1024)
    b64 = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    assert read_image_dimensions(b64) == (1024, 1024)


def test_read_image_dimensions_parses_jpeg_header() -> None:
    raw = _make_jpeg_bytes(1216, 832)
    assert read_image_dimensions(raw) == (1216, 832)


def test_read_image_dimensions_returns_none_for_unknown_format() -> None:
    assert read_image_dimensions(b"not_an_image") is None
    assert read_image_dimensions("") is None


# ── §20 payload 构造与校验 ──────────────────────────────────────────────────


def _model_config() -> dict:
    return {
        "default_model": "nai-diffusion-4-5-full",
        "nai_size": "竖图",
    }


def test_build_inner_payload_includes_i2i_block() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        _model_config(),
        None,
        i2i_payload={
            "image": "data:image/png;base64,iVBORw0KGgo",
            "strength": 0.5,
            "noise": 0.1,
        },
    )
    assert inner["i2i"]["image"] == "iVBORw0KGgo"
    assert inner["i2i"]["strength"] == 0.5
    assert inner["i2i"]["noise"] == 0.1


def test_build_inner_payload_includes_character_references() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        _model_config(),
        None,
        character_references_payload=[
            {
                "image": "iVBORw0KGgo",
                "type": "character",
                "fidelity": 0.9,
                "strength": 0.8,
            }
        ],
    )
    assert inner["character_references"] == [
        {
            "image": "iVBORw0KGgo",
            "type": "character",
            "fidelity": 0.9,
            "strength": 0.8,
        }
    ]


def test_build_inner_payload_strips_invalid_character_reference_entries() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        _model_config(),
        None,
        character_references_payload=[
            {"image": ""},  # 空 image 直接丢
            {"image": "iVBORw"},
        ],
    )
    assert inner["character_references"] == [{"image": "iVBORw"}]


def test_build_inner_payload_normalizes_controlnet_cache_mode() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        _model_config(),
        None,
        controlnet_payload={
            "strength": 1.0,
            "images": [
                {"cache_id": "AbCdEfGhIjKlMnOpQrStUv", "strength": 0.7},
                {"image": "data:image/png;base64,xx", "info_extracted": 0.7, "strength": 0.6},
            ],
        },
    )
    assert inner["controlnet"]["strength"] == 1.0
    assert inner["controlnet"]["images"] == [
        {"cache_id": "AbCdEfGhIjKlMnOpQrStUv", "strength": 0.7},
        {"image": "xx", "info_extracted": 0.7, "strength": 0.6},
    ]


def test_filter_character_references_for_model_drops_non_v4_5() -> None:
    payload = [{"image": "xx"}]
    assert NaiWebClient._filter_character_references_for_model("nai-diffusion-4-5-full", payload) is payload
    assert NaiWebClient._filter_character_references_for_model("nai-diffusion-4-full", payload) is None
    assert NaiWebClient._filter_character_references_for_model("nai-diffusion-3", payload) is None


def test_validate_image_payloads_rejects_mutex_violation() -> None:
    """controlnet 与 character_references 互斥（文档 §20.5）。"""
    inner = {
        "prompt": "x",
        "size": [832, 1216],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "controlnet": {"images": [{"image": "x", "info_extracted": 0.7}]},
        "character_references": [{"image": "y"}],
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is not None
    assert "controlnet" in reason and "character_references" in reason


def test_validate_image_payloads_rejects_more_than_four_controlnet_images() -> None:
    inner = {
        "prompt": "x",
        "size": [832, 1216],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "controlnet": {
            "images": [
                {"image": "x1", "info_extracted": 0.7},
                {"image": "x2", "info_extracted": 0.7},
                {"image": "x3", "info_extracted": 0.7},
                {"image": "x4", "info_extracted": 0.7},
                {"image": "x5", "info_extracted": 0.7},
            ]
        },
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is not None
    assert "4" in reason


def test_validate_image_payloads_rejects_more_than_one_character_reference() -> None:
    inner = {
        "prompt": "x",
        "size": [832, 1216],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "character_references": [{"image": "a"}, {"image": "b"}],
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is not None
    assert "character_references" in reason


def test_validate_image_payloads_rejects_controlnet_item_with_both_modes() -> None:
    """单条 controlnet.images[i] 必须严格走完整态或缓存态。"""
    inner = {
        "prompt": "x",
        "size": [832, 1216],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "controlnet": {
            "images": [{"image": "x", "cache_id": "abc"}],
        },
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is not None
    assert "cache_id" in reason


def test_validate_i2i_image_dimension_mismatch_rejects() -> None:
    """i2i.image 宽高必须与外层 size 严格相等（文档 §20.1）。"""
    png_bytes = _make_png_bytes(512, 768)  # 与外层 size 不符
    b64 = base64.b64encode(png_bytes).decode("ascii")
    inner = {
        "prompt": "x",
        "size": [832, 1216],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "i2i": {"image": b64, "strength": 0.5},
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is not None
    assert "i2i.image" in reason


def test_validate_i2i_image_dimension_match_passes() -> None:
    png_bytes = _make_png_bytes(832, 1216)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    inner = {
        "prompt": "x",
        "size": [832, 1216],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "i2i": {"image": b64, "strength": 0.5},
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is None
