# -*- coding: utf-8 -*-
"""反推编排服务：PNG 元数据 → WD14 在线 Space 兜底。

只输出正向 prompt。负面来源（NAI Comment.uc / SD parameters Negative prompt）一律丢弃。
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .png_meta import extract_prompt_from_png
from .wd14_client import WD14Client, WD14ClientError


@dataclass
class ReverseResult:
    """反推结果。"""

    source: str  # "metadata" | "wd14" | "failed"
    prompt: str
    tags: List[str] = field(default_factory=list)
    detail: Optional[str] = None  # 失败原因或额外信息


class ReverseService:
    """编排 PNG 元数据 + WD14 的反推流水线。"""

    def __init__(
        self,
        *,
        wd14_client: Optional[WD14Client] = None,
        wd14_threshold: float = 0.35,
        wd14_character_threshold: float = 0.8,
        wd14_enabled: bool = True,
    ) -> None:
        self._wd14_client = wd14_client
        self._wd14_threshold = float(wd14_threshold)
        self._wd14_character_threshold = float(wd14_character_threshold)
        self._wd14_enabled = bool(wd14_enabled)
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

    async def reverse(self, image_bytes: bytes) -> ReverseResult:
        """对 image_bytes 执行反推，先 PNG 元数据，未命中再走 WD14。"""
        if not image_bytes:
            return ReverseResult(source="failed", prompt="", detail="image_bytes 为空")

        # ── Level 1: PNG 元数据 ──
        try:
            meta_result = extract_prompt_from_png(image_bytes)
        except Exception as exc:
            self._logger.warning(f"[nai/反推] PNG 元数据解析异常（跳过）: {exc!r}")
            meta_result = None

        if meta_result is not None and meta_result.tags:
            self._logger.info(f"[nai/反推] 命中 PNG 元数据，{len(meta_result.tags)} 个 tag")
            return ReverseResult(
                source="metadata",
                prompt=meta_result.prompt,
                tags=list(meta_result.tags),
            )

        # ── Level 2: WD14 兜底 ──
        if not self._wd14_enabled:
            return ReverseResult(source="failed", prompt="", detail="未启用 WD14 兜底")
        if self._wd14_client is None:
            return ReverseResult(source="failed", prompt="", detail="WD14 客户端未初始化")

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
        return ReverseResult(source="wd14", prompt=prompt, tags=tags)


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


__all__ = ["ReverseService", "ReverseResult"]
