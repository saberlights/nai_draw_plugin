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
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# NAI 的结构化正向 prompt 容器键：v4_prompt / v4_5_prompt / v5_prompt …
# 按家族匹配而非硬编码版本号，新版本沿用同一结构时不需要改代码
_STRUCTURED_PROMPT_KEY_RE = re.compile(r"v\d+(?:_\d+)*_prompt")

# 顶层 prompt / Description 也可能直接是 caption 容器而不是字符串
_CAPTION_CONTAINER_KEYS = frozenset({"prompt", "description", "caption"})

# 负面来源，任何情况下都不得当正向读取
_NEGATIVE_KEYS = frozenset({"uc", "negative_prompt", "negativeprompt"})

# 常见图片容器的魔数，用于把"不是 PNG"和"是 PNG 但没元数据"区分开
_IMAGE_MAGICS = (
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
)


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


def _iter_positive_structured_prompts(meta: Dict[str, object]):
    """产出 NAI V4+ 的结构化正向 prompt 容器。

    V4 起 NAI 不再把完整正向写进顶层 ``prompt``，而是放进 ``v4_prompt.caption``：
    ``base_caption`` 是全局段，``char_captions[].char_caption`` 是各角色段。
    多角色图的顶层 ``prompt`` 可能整体为空，只读顶层就会反推不出任何东西。

    键名按 ``v<数字>_prompt`` 家族匹配而不是硬编码 ``v4_prompt``，这样 V5 沿用
    同一套结构时无需改代码；带 ``negative`` 的键一律跳过，避免把负面串进正向。
    """

    for key, value in meta.items():
        lowered = str(key).lower()
        if "negative" in lowered or lowered in _NEGATIVE_KEYS:
            continue
        # 结构化容器既可能挂在 v4_prompt / v5_prompt 这类版本键下，也可能就是
        # 顶层 prompt / Description 本身——NAI 把 prompt 从字符串换成对象时，
        # 只按版本键去找就会整张图反推不出来
        if lowered not in _CAPTION_CONTAINER_KEYS and not _STRUCTURED_PROMPT_KEY_RE.fullmatch(lowered):
            continue
        if isinstance(value, dict):
            yield value


def _caption_of(container: Dict[str, object]) -> Dict[str, object]:
    """取容器里的 caption 子对象；没有嵌套层时容器自身就是 caption。"""

    caption = container.get("caption")
    if isinstance(caption, dict):
        return caption
    return container


def _extract_structured_captions(meta: Dict[str, object]) -> tuple[str, List[str]]:
    """从 V4+ 结构里取 (base_caption, [char_caption...])，缺失处返回空。

    容器允许两种嵌套深度：``{"caption": {"base_caption": ..., "char_captions": [...]}}``
    与直接 ``{"base_caption": ..., "char_captions": [...]}``。
    """

    base_caption = ""
    char_captions: List[str] = []
    for container in _iter_positive_structured_prompts(meta):
        caption = _caption_of(container)
        if not base_caption:
            for candidate_key in ("base_caption", "prompt"):
                raw_base = caption.get(candidate_key)
                if isinstance(raw_base, str) and raw_base.strip():
                    base_caption = raw_base
                    break
        raw_chars = caption.get("char_captions")
        if not isinstance(raw_chars, list):
            continue
        for entry in raw_chars:
            if isinstance(entry, str) and entry.strip():
                char_captions.append(entry)
                continue
            if not isinstance(entry, dict):
                continue
            for candidate_key in ("char_caption", "caption", "prompt"):
                raw_caption = entry.get(candidate_key)
                if isinstance(raw_caption, str) and raw_caption.strip():
                    char_captions.append(raw_caption)
                    break
    return base_caption, char_captions


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

    # V4+ 结构优先：它才带角色段；顶层 prompt 只在结构里没有全局段时兜底
    base_caption, char_captions = _extract_structured_captions(meta)
    if not base_caption:
        legacy = meta.get("prompt") or meta.get("Description") or ""
        if isinstance(legacy, str):
            base_caption = legacy

    base_tags = _split_tags(base_caption)
    char_tag_groups = [_split_tags(caption) for caption in char_captions]
    char_tag_groups = [group for group in char_tag_groups if group]
    if not base_tags and not char_tag_groups:
        return None

    # charN: 拍平格式只在真多人时使用。下游认定多人的阈值是"1 个全局段 + ≥2 个角色段"
    # （extract_multi_character_payload_from_text 要求 ≥3 段，_select_send_payload 要求
    # characters ≥ 2）。V4 单人图也会填一个 char_caption，若在这里就切多行，产出的
    # 单条 char1: 谁都反解不回来，回灌 /nai0 会把字面量 char1: 当 tag 画进图里。
    if len(char_tag_groups) >= 2:
        lines = [", ".join(base_tags) + ","] if base_tags else []
        lines.extend(
            f"char{index}:{', '.join(group)},"
            for index, group in enumerate(char_tag_groups, start=1)
        )
        prompt = "\n".join(lines)
    else:
        # 单人：合并成扁平串，与 V3 行为一致；角色段的 tag 仍然并进来，不丢内容
        merged = list(base_tags)
        for group in char_tag_groups:
            merged.extend(tag for tag in group if tag not in merged)
        prompt = ", ".join(merged)

    tags = list(base_tags)
    for group in char_tag_groups:
        tags.extend(group)
    return PngMetaResult(
        prompt=prompt,
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


def _detect_container(image_bytes: bytes) -> str:
    """识别图片容器类型，未知时返回前 4 字节的十六进制。"""
    for magic, label in _IMAGE_MAGICS:
        if image_bytes.startswith(magic):
            return label
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "WebP"
    return f"未知容器(前4字节 {image_bytes[:4].hex()})"


def describe_png_metadata(image_bytes: bytes) -> str:
    """报告 PNG 元数据的真实状况，供反推未命中时定位原因。

    原先四种完全不同的故障都只表现为 ``extract_prompt_from_png`` 返回 None：
    根本不是 PNG、是 PNG 但没有文本块、有 Comment 但不是 JSON、有 JSON 但正向
    字段为空。它们的处理方式完全不同（换出图格式 / 平台重编码剥元数据 / NAI 改了
    结构），混成一句"无元数据"就没法定位，所以这里逐层报告。

    也可以当独立探针直接调用：元数据其实可解析时会如实说明，不会硬报失败。
    """

    if not image_bytes:
        return "图片数据为空"

    container = _detect_container(image_bytes)
    if container != "PNG":
        return (
            f"不是 PNG 而是 {container}，只有 PNG 带 tEXt 元数据；"
            "检查 image_format 配置，或图片是否被平台转码"
        )

    try:
        chunks = _parse_png_text_chunks(image_bytes)
    except Exception as exc:  # noqa: BLE001 - 诊断路径，任何解析异常都要如实报告
        return f"PNG 文本块解析异常: {exc!r}"

    if not chunks:
        return "PNG 没有任何 tEXt/iTXt/zTXt 文本块（元数据已被剥离或平台重新编码过）"

    # NAI 把模型版本写在 Source chunk（如 "NovelAI Diffusion V4.5 4BDE2A90"），
    # 判断"哪个版本的图反推不出来"必须报出它，否则只能靠猜
    source = str(chunks.get("Source") or chunks.get("Software") or "").strip()
    origin = f"来源={source!r}；" if source else ""
    keys = "、".join(sorted(chunks))

    existing = extract_prompt_from_png(image_bytes)
    if existing is not None and existing.tags:
        return (
            f"元数据其实可解析（{len(existing.tags)} 个 tag）；{origin}"
            f"文本块键：{keys}"
        )

    comment = chunks.get("Comment")
    if comment is None:
        return f"PNG 有文本块但没有 NAI 的 Comment 字段；{origin}现有键：{keys}"

    try:
        meta = json.loads(comment.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return f"Comment 不是合法 JSON（{exc}）；{origin}现有键：{keys}"

    if not isinstance(meta, dict):
        return f"Comment JSON 顶层不是对象而是 {type(meta).__name__}；{origin}现有键：{keys}"

    base_caption, char_captions = _extract_structured_captions(meta)
    structured_keys = [
        key for key in meta
        if _STRUCTURED_PROMPT_KEY_RE.fullmatch(str(key).lower())
    ]
    return (
        "Comment JSON 里没有可用的正向 prompt："
        f"{origin}顶层 prompt={meta.get('prompt')!r}，"
        f"结构化键={structured_keys or '无'}，"
        f"base_caption={base_caption!r}，角色段 {len(char_captions)} 个；"
        f"Comment 顶层字段：{'、'.join(sorted(meta))}"
    )


__all__ = ["PngMetaResult", "describe_png_metadata", "extract_prompt_from_png"]
