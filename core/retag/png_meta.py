# -*- coding: utf-8 -*-
"""PNG 元数据反推 —— 仅输出正向 prompt。

支持三种主流写入方式：
  * NAI 出图：tEXt/iTXt["Comment"] 是 JSON，含 prompt / uc
  * SD WebUI：tEXt["parameters"] 是纯文本，"prompt\\nNegative prompt: ...\\nSteps: ..."
  * 通用：tEXt["prompt"] / ["Description"] / ["UserComment"]

依据需求负面不输出，解析时仍需识别 Negative prompt 行以免被串进正向。
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PngMetaResult:
    """PNG 元数据反推结果。"""

    prompt: str
    tags: List[str] = field(default_factory=list)
    # NAI / SD 输出的额外参数信息，便于排错时回看（不强制使用）
    raw_metadata: Optional[Dict[str, object]] = None


def _parse_png_text_chunks(data: bytes) -> Dict[str, str]:
    """纯标准库解析 PNG tEXt / iTXt / zTXt chunk，返回 {keyword: value}。"""
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return {}

    chunks: Dict[str, str] = {}
    pos = 8
    length_total = len(data)

    while pos + 12 <= length_total:
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # length(4) + type(4) + data + crc(4)

        if chunk_type == b"tEXt":
            try:
                null_idx = chunk_data.index(b"\x00")
                keyword = chunk_data[:null_idx].decode("latin-1")
                value = chunk_data[null_idx + 1 :].decode("latin-1")
                chunks[keyword] = value
            except (ValueError, UnicodeDecodeError):
                pass

        elif chunk_type == b"iTXt":
            # iTXt: keyword\0 comp_flag(1) comp_method(1) language\0 translated_keyword\0 text
            try:
                null_idx = chunk_data.index(b"\x00")
                keyword = chunk_data[:null_idx].decode("latin-1")
                rest = chunk_data[null_idx + 1 :]
                if len(rest) < 2:
                    continue
                comp_flag = rest[0]
                rest = rest[2:]  # 跳过 comp_flag + comp_method
                # 跳过 language 和 translated_keyword（各以 \0 结尾）
                for _ in range(2):
                    ni = rest.index(b"\x00")
                    rest = rest[ni + 1 :]
                if comp_flag == 1:
                    text = zlib.decompress(rest).decode("utf-8")
                else:
                    text = rest.decode("utf-8")
                chunks[keyword] = text
            except (ValueError, UnicodeDecodeError, zlib.error):
                pass

        elif chunk_type == b"zTXt":
            try:
                null_idx = chunk_data.index(b"\x00")
                keyword = chunk_data[:null_idx].decode("latin-1")
                # null + compression_method(1) 共 2 字节，再之后才是压缩文本
                compressed = chunk_data[null_idx + 2 :]
                value = zlib.decompress(compressed).decode("latin-1")
                chunks[keyword] = value
            except (ValueError, UnicodeDecodeError, zlib.error):
                pass

        elif chunk_type == b"IEND":
            break

    return chunks


def _split_tags(prompt: str) -> List[str]:
    """按逗号切分 prompt，过滤空白项。"""
    return [t.strip() for t in prompt.split(",") if t.strip()]


def _normalize_prompt(prompt: str) -> str:
    """规整化：去掉首尾空白，多余空格，重排为 tag1, tag2 形式。"""
    tags = _split_tags(prompt)
    return ", ".join(tags)


def _extract_from_nai_comment(comment_value: str) -> Optional[PngMetaResult]:
    """解析 NAI 写入的 Comment 字段。可能是 JSON 也可能是纯文本。"""
    text = (comment_value or "").strip()
    if not text:
        return None

    # 尝试 JSON
    try:
        meta = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 不是 JSON：当作纯 prompt 处理
        tags = _split_tags(text)
        if tags:
            return PngMetaResult(prompt=", ".join(tags), tags=tags)
        return None

    if not isinstance(meta, dict):
        return None

    raw_prompt = meta.get("prompt") or meta.get("Description") or ""
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        return None
    tags = _split_tags(raw_prompt)
    if not tags:
        return None
    return PngMetaResult(
        prompt=", ".join(tags),
        tags=tags,
        raw_metadata=meta,
    )


def _extract_from_sd_parameters(raw: str) -> Optional[PngMetaResult]:
    """解析 SD WebUI 风格 `parameters` 字段，仅取正向 prompt。

    SD WebUI 的写法：
        <prompt 多行>
        Negative prompt: <neg 多行>
        Steps: 28, Sampler: Euler a, ...
    """
    if not raw or not raw.strip():
        return None

    prompt_lines: List[str] = []
    in_neg = False
    for line in raw.split("\n"):
        if line.startswith("Negative prompt:"):
            in_neg = True
            continue
        # Steps/Size/Sampler 这一行（或之后）标志着参数区开始，整体结束
        if line.startswith("Steps:") or line.startswith("Size:") or line.startswith("Sampler:"):
            break
        if in_neg:
            # 负向区间，跳过
            continue
        prompt_lines.append(line)

    prompt_raw = " ".join(prompt_lines)
    tags = _split_tags(prompt_raw)
    if not tags:
        return None
    return PngMetaResult(
        prompt=", ".join(tags),
        tags=tags,
        raw_metadata={"raw_parameters": raw},
    )


def extract_prompt_from_png(image_bytes: bytes) -> Optional[PngMetaResult]:
    """从 PNG bytes 中尝试解析 prompt。无元数据时返回 None。"""
    try:
        chunks = _parse_png_text_chunks(image_bytes)
    except Exception:
        # 任何解析异常都视为无元数据，由上层走兜底链路
        return None

    if not chunks:
        return None

    # NAI 格式优先
    if "Comment" in chunks:
        result = _extract_from_nai_comment(chunks["Comment"])
        if result is not None:
            return result

    # SD WebUI 格式
    if "parameters" in chunks:
        result = _extract_from_sd_parameters(chunks["parameters"])
        if result is not None:
            return result

    # 通用兜底字段
    for key in ("prompt", "Description", "UserComment"):
        value = chunks.get(key)
        if not isinstance(value, str):
            continue
        tags = _split_tags(value)
        if tags:
            return PngMetaResult(prompt=", ".join(tags), tags=tags)

    return None


__all__ = ["PngMetaResult", "extract_prompt_from_png"]
