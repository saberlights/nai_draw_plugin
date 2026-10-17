# -*- coding: utf-8 -*-
"""反推编排服务：按路径分发到 PNG 元数据 / WD14 / 视觉模型。

四条路径：
  * ``auto``：PNG 元数据 → WD14 兜底（历史行为，也是默认）
  * ``meta``：只读 PNG 元数据，完全不联网
  * ``wd14``：只走 WD14 在线 Space
  * ``llm`` ：只走视觉模型，产出英文自然语言描述

``llm`` 不做元数据兜底：显式选它就是要一段自然语言描述，而元数据给的是 tag 串，
两者语义不同，互相顶包只会让人拿到一个自己没要的东西还以为反推成功了。

tag 类路径只输出正向 prompt。负面来源（NAI Comment.uc / SD parameters Negative prompt）
一律丢弃。
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .png_meta import describe_png_metadata, extract_prompt_from_png
from .wd14_client import WD14Client, WD14ClientError


# 反推路径标识；同时是 `/nai 反推路径` 命令与 `[retag].default_path` 配置的取值域
PATH_AUTO = "auto"
PATH_META = "meta"
PATH_WD14 = "wd14"
PATH_LLM = "llm"

REVERSE_PATHS: tuple[str, ...] = (PATH_AUTO, PATH_META, PATH_WD14, PATH_LLM)

# 展示给用户的路径说明；命令回显与 help 共用，避免两处各写一份说法
REVERSE_PATH_LABELS: Dict[str, str] = {
    PATH_AUTO: "自动（PNG 元数据 → WD14 兜底）",
    PATH_META: "元数据（只读 PNG 元数据，不联网）",
    PATH_WD14: "WD14（在线 Space 反推 Danbooru tag）",
    PATH_LLM: "LLM（视觉模型反推英文自然语言描述，适配 V5）",
}

# 结果形态：tag 串还是自然语言段落。命令回显的计数文案与使用建议按它分支——
# 对一段 300 词的英文描述说"3 个 tag"是错的。
KIND_TAGS = "tags"
KIND_NATURAL_LANGUAGE = "natural_language"


def normalize_path(value: Any, *, default: str = PATH_AUTO) -> str:
    """把用户输入或配置值归一化为合法路径标识。

    Args:
        value: 待归一化的值
        default: 空值时的回落路径

    Returns:
        ``REVERSE_PATHS`` 中的一项

    Raises:
        ValueError: 值非空但不在取值域内
    """
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text not in REVERSE_PATHS:
        raise ValueError(f"未知的反推路径 {text!r}，可选：{'/'.join(REVERSE_PATHS)}")
    return text


class ImageCaptionerProtocol(Protocol):
    """视觉模型反推 Interface；实现在 ``core/services/llm_image_captioner.py``。"""

    async def caption(self, *, image_bytes: bytes, sfw: bool) -> str: ...


@dataclass
class ReverseResult:
    """反推结果。"""

    source: str  # "metadata" | "wd14" | "llm" | "failed"
    prompt: str
    tags: List[str] = field(default_factory=list)
    detail: Optional[str] = None  # 失败原因或额外信息
    kind: str = KIND_TAGS  # KIND_TAGS | KIND_NATURAL_LANGUAGE


class ReverseService:
    """按路径编排 PNG 元数据 / WD14 / 视觉模型三条反推链路。"""

    def __init__(
        self,
        *,
        wd14_client: Optional[WD14Client] = None,
        wd14_threshold: float = 0.35,
        wd14_character_threshold: float = 0.8,
        wd14_enabled: bool = True,
        captioner: Optional[ImageCaptionerProtocol] = None,
    ) -> None:
        self._wd14_client = wd14_client
        self._wd14_threshold = float(wd14_threshold)
        self._wd14_character_threshold = float(wd14_character_threshold)
        self._wd14_enabled = bool(wd14_enabled)
        self._captioner = captioner
        self._logger = logging.getLogger(__name__)

    def update_wd14_client(self, client: Optional[WD14Client]) -> None:
        """配置热更新时替换 WD14 客户端实例。"""
        self._wd14_client = client

    def update_wd14_thresholds(
        self,
        *,
        threshold: float,
        character_threshold: float,
        enabled: bool,
    ) -> None:
        self._wd14_threshold = float(threshold)
        self._wd14_character_threshold = float(character_threshold)
        self._wd14_enabled = bool(enabled)

    def update_captioner(self, captioner: Optional[ImageCaptionerProtocol]) -> None:
        """配置热更新时替换视觉模型反推实例；None 表示 llm 路径未启用。"""
        self._captioner = captioner

    async def reverse(
        self,
        image_bytes: bytes,
        *,
        path: str = PATH_AUTO,
        sfw: bool = True,
    ) -> ReverseResult:
        """按指定路径反推 image_bytes。

        Args:
            image_bytes: 原始图片字节
            path: ``REVERSE_PATHS`` 中的一项
            sfw: 当前会话是否开着 NSFW 过滤，只对 ``llm`` 路径有意义

        Returns:
            ReverseResult；失败时 ``source == "failed"`` 且 ``detail`` 带具体原因
        """
        if not image_bytes:
            return ReverseResult(source="failed", prompt="", detail="image_bytes 为空")

        resolved_path = normalize_path(path)

        if resolved_path == PATH_LLM:
            return await self._reverse_with_llm(image_bytes, sfw=sfw)

        if resolved_path == PATH_WD14:
            return await self._reverse_with_wd14(image_bytes, meta_miss_reason="")

        meta_result = self._reverse_with_metadata(image_bytes)
        if meta_result is not None:
            return meta_result

        # 元数据未命中的具体原因必须留痕：WD14 兜底成功时结果看起来是对的，
        # 但"NAI 改了元数据结构"和"平台把图重编码了"的处理方式完全不同
        meta_miss_reason = describe_png_metadata(image_bytes)
        self._logger.info(f"[nai/反推] PNG 元数据未命中：{meta_miss_reason}")

        if resolved_path == PATH_META:
            return ReverseResult(
                source="failed",
                prompt="",
                detail=f"路径已锁定为仅读元数据；未命中原因：{meta_miss_reason}",
            )

        return await self._reverse_with_wd14(image_bytes, meta_miss_reason=meta_miss_reason)

    # ── Level 1: PNG 元数据 ──

    def _reverse_with_metadata(self, image_bytes: bytes) -> Optional[ReverseResult]:
        """读 PNG 元数据；未命中返回 None，由调用方决定怎么兜。"""
        try:
            meta_result = extract_prompt_from_png(image_bytes)
        except Exception as exc:
            self._logger.warning(f"[nai/反推] PNG 元数据解析异常（跳过）: {exc!r}")
            return None

        if meta_result is None or not meta_result.tags:
            return None

        self._logger.info(f"[nai/反推] 命中 PNG 元数据，{len(meta_result.tags)} 个 tag")
        return ReverseResult(
            source="metadata",
            prompt=meta_result.prompt,
            tags=list(meta_result.tags),
            kind=KIND_TAGS,
        )

    # ── Level 2: WD14 兜底 ──

    async def _reverse_with_wd14(
        self,
        image_bytes: bytes,
        *,
        meta_miss_reason: str,
    ) -> ReverseResult:
        """调 WD14 在线 Space 反推 Danbooru tag。

        Args:
            image_bytes: 原始图片字节
            meta_miss_reason: 元数据未命中的原因；显式选 wd14 路径时为空串，
                失败详情里就不再拼这段（用户压根没走元数据）
        """
        suffix = f"；PNG 元数据未命中原因：{meta_miss_reason}" if meta_miss_reason else ""

        if not self._wd14_enabled:
            return ReverseResult(source="failed", prompt="", detail=f"未启用 WD14 兜底{suffix}")
        if self._wd14_client is None:
            return ReverseResult(source="failed", prompt="", detail=f"WD14 客户端未初始化{suffix}")

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        try:
            wd14_result = await asyncio.wait_for(
                self._wd14_client.tag_image(
                    image_base64=image_b64,
                    threshold=self._wd14_threshold,
                    character_threshold=self._wd14_character_threshold,
                ),
                timeout=self._wd14_client.command_timeout,
            )
        except WD14ClientError as exc:
            self._logger.warning(f"[nai/反推] WD14 调用失败: {exc}")
            return ReverseResult(source="failed", prompt="", detail=f"WD14: {exc}")
        except asyncio.TimeoutError:
            self._logger.warning("[nai/反推] WD14 调用超时")
            return ReverseResult(source="failed", prompt="", detail="WD14 调用超时")
        except Exception as exc:
            self._logger.error(f"[nai/反推] WD14 调用异常: {exc!r}", exc_info=True)
            return ReverseResult(source="failed", prompt="", detail=f"WD14 异常: {exc}")

        tags = _flatten_wd14_tags(wd14_result)
        if not tags:
            return ReverseResult(source="failed", prompt="", detail="WD14 未识别到任何标签")

        prompt = ", ".join(tags)
        self._logger.info(f"[nai/反推] 命中 WD14，{len(tags)} 个 tag")
        return ReverseResult(source="wd14", prompt=prompt, tags=tags, kind=KIND_TAGS)

    # ── 独立路径: 视觉模型自然语言 ──

    async def _reverse_with_llm(self, image_bytes: bytes, *, sfw: bool) -> ReverseResult:
        """调视觉模型拿一段英文自然语言描述。"""
        if self._captioner is None:
            return ReverseResult(
                source="failed",
                prompt="",
                detail="LLM 反推未启用（[retag].llm_enabled = false 或视觉模型未初始化）",
            )

        try:
            caption = await self._captioner.caption(image_bytes=image_bytes, sfw=sfw)
        except Exception as exc:
            # 视觉模型的失败原因（拒答、超时、模型名写错）用户必须原样看到才能自查，
            # 这里只做转述不做兜底
            self._logger.warning(f"[nai/反推] 视觉模型反推失败: {exc}")
            return ReverseResult(source="failed", prompt="", detail=f"LLM: {exc}")

        self._logger.info(f"[nai/反推] 命中视觉模型，描述长度 {len(caption)} 字符")
        return ReverseResult(
            source="llm",
            prompt=caption,
            tags=[],
            kind=KIND_NATURAL_LANGUAGE,
        )


def _flatten_wd14_tags(wd14_result: Dict[str, Any]) -> List[str]:
    """从 WD14 返回结构里提取按分数排好的 label 串，去重。"""
    raw_tags = wd14_result.get("tags") if isinstance(wd14_result, dict) else None
    if not isinstance(raw_tags, list):
        return []

    seen: set[str] = set()
    labels: List[str] = []
    for item in raw_tags:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


__all__ = [
    "ImageCaptionerProtocol",
    "KIND_NATURAL_LANGUAGE",
    "KIND_TAGS",
    "PATH_AUTO",
    "PATH_LLM",
    "PATH_META",
    "PATH_WD14",
    "REVERSE_PATHS",
    "REVERSE_PATH_LABELS",
    "ReverseResult",
    "ReverseService",
    "normalize_path",
]
