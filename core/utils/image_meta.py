# -*- coding: utf-8 -*-
"""图片元数据解析工具。

只依赖 stdlib 解析 PNG / JPEG / WebP 文件头，拿到 ``(width, height)``。
i2i 校验需要这条信息——文档 §20.1 要求 ``i2i.image`` 的宽高与外层 ``size``
严格相等，否则上游直接 400。

故意不引入 Pillow / cv2 等重量级依赖：图片只是过路，不做缩放/编解码。
"""

from __future__ import annotations

import base64
import struct
from typing import Optional, Tuple


__all__ = ["normalize_image_base64", "read_image_dimensions"]


def normalize_image_base64(raw: str) -> str:
    """剥掉 ``data:image/...;base64,`` 前缀与无关换行，返回纯净 base64 字符串。

    传 None / 空串时返回空串；非字符串退化成 ``str()``。
    """
    cleaned = str(raw or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("data:image/") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    return cleaned.replace("\n", "").replace("\r", "").strip()


def read_image_dimensions(image_data: str | bytes) -> Optional[Tuple[int, int]]:
    """解析 PNG / JPEG / WebP 头部，返回 ``(width, height)``；解析失败返回 None。

    入参可以是 raw bytes 或 base64 字符串（含/不含 data URI 前缀都可以）。
    """
    raw_bytes = _coerce_to_bytes(image_data)
    if not raw_bytes:
        return None
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return _read_png_dimensions(raw_bytes)
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return _read_jpeg_dimensions(raw_bytes)
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return _read_webp_dimensions(raw_bytes)
    return None


# ── 内部辅助 ─────────────────────────────────────────────────────────────


def _coerce_to_bytes(image_data: str | bytes) -> bytes:
    if isinstance(image_data, bytes):
        return image_data
    cleaned = normalize_image_base64(image_data)
    if not cleaned:
        return b""
    try:
        return base64.b64decode(cleaned, validate=False)
    except (ValueError, TypeError):
        return b""


def _read_png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """PNG: 8 字节 magic + 4 字节 IHDR 长度 + 4 字节类型 + 4 字节宽 + 4 字节高（大端）。"""
    if len(data) < 24:
        return None
    if data[12:16] != b"IHDR":
        return None
    try:
        width, height = struct.unpack(">II", data[16:24])
    except struct.error:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _read_jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """JPEG: 顺着 marker 链找 SOFn（0xC0~0xCF，去掉 DHT/DAC/RST 等非 SOF 段）。"""
    length = len(data)
    if length < 4:
        return None
    # 跳过开头 0xFF 0xD8 SOI
    pos = 2
    while pos < length - 7:
        # 找到下一个 0xFF marker，跳过填充字节 0xFF
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < length and data[pos] == 0xFF:
            pos += 1
        if pos >= length:
            return None
        marker = data[pos]
        pos += 1
        # 段内有数据的 marker 后面紧跟 2 字节段长度（大端）；段长度含自身 2 字节
        if marker in (0x00, 0xD8, 0xD9):
            continue
        if 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > length:
            return None
        segment_length = struct.unpack(">H", data[pos:pos + 2])[0]
        # SOF0~SOF15，但 DHT(0xC4)/DAC(0xCC)/JPG(0xC8) 不是 SOF
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if pos + 7 > length:
                return None
            # SOF 段：precision(1) height(2) width(2) ...
            height = struct.unpack(">H", data[pos + 3:pos + 5])[0]
            width = struct.unpack(">H", data[pos + 5:pos + 7])[0]
            if width <= 0 or height <= 0:
                return None
            return width, height
        pos += segment_length
    return None


def _read_webp_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """WebP: RIFF header 后 VP8 / VP8L / VP8X 三种 chunk 各自的宽高编码不一样。"""
    if len(data) < 30:
        return None
    chunk_type = data[12:16]
    if chunk_type == b"VP8 ":
        # 简单 VP8: 在第 26 字节起读宽高（14 位 + 14 位，2 字节小端但只用低 14 位）
        try:
            width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        except struct.error:
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height
    if chunk_type == b"VP8L":
        # 无损 VP8L: signature(1=0x2F) + 4 字节 packed (14位宽-1, 14位高-1)
        if len(data) < 25:
            return None
        if data[20] != 0x2F:
            return None
        try:
            packed = struct.unpack("<I", data[21:25])[0]
        except struct.error:
            return None
        width = (packed & 0x3FFF) + 1
        height = ((packed >> 14) & 0x3FFF) + 1
        return width, height
    if chunk_type == b"VP8X":
        # 扩展 VP8X: 在第 24 字节起读 24 位宽-1 / 24 位高-1（小端）
        if len(data) < 30:
            return None
        try:
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
        except (struct.error, ValueError):
            return None
        return width, height
    return None
