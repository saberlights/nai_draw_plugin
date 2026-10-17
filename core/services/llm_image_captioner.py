# -*- coding: utf-8 -*-
"""MaiBot 视觉模型调用 Adapter：把图片转写成英文自然语言描述。

与 ``llm_text_generator`` 对称——那边是纯文本任务，这边多一路图片入参。两者共用
``pinned_task_orchestrator``，所以"自定义模型"的配置形状在两条链路上完全一致。

失败一律抛 :class:`ImageCaptionError`，不返回空串也不静默回落到宿主任务模型：
用户之所以在 ``[retag].llm_custom_model`` 里钉一个视觉模型，就是因为宿主默认的 vlm
任务模型描述不够细；悄悄换回去只会让人对着一段敷衍的描述查半天配置。
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.common.data_models.llm_service_data_models import LLMImageOptions
from src.common.logger import get_logger
from src.common.utils.utils_image import ImageUtils
from src.services import llm_service
from src.services.llm_service import LLMServiceClient

from ..rules.caption_rules import resolve_caption_prompt
from ..utils.prompt_output_parser import extract_last_code_block
from ..utils.prompt_postprocessor import strip_cjk_and_fullwidth
from .pinned_task_orchestrator import (
    PinnedTaskLLMOrchestrator,
    build_pinned_task_config,
    coerce_float,
    coerce_int,
)


logger = get_logger("nai_draw_plugin")


# 图片格式必须按字节头嗅探，不能硬写 "png"：反推的输入来自聊天平台的入站图片缓存，
# QQ 侧绝大多数是 JPEG，格式报错会让部分网关直接拒收整个请求。
_IMAGE_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


class ImageCaptionError(RuntimeError):
    """视觉模型反推失败。"""


def detect_image_format(image_bytes: bytes) -> str:
    """按字节头判断图片格式。

    Args:
        image_bytes: 原始图片字节

    Returns:
        ``png`` / ``jpeg`` / ``gif`` / ``bmp`` / ``webp``

    Raises:
        ImageCaptionError: 字节头不属于任何已知图片格式
    """
    if not image_bytes:
        raise ImageCaptionError("图片字节为空")

    for prefix, image_format in _IMAGE_MAGIC_PREFIXES:
        if image_bytes.startswith(prefix):
            return image_format

    # WEBP 的魔数是分段的：RIFF....WEBP，中间 4 字节是文件长度
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"

    raise ImageCaptionError(f"无法识别的图片格式，字节头为 {image_bytes[:12]!r}")


class ImageCaptioner:
    """反推链路依赖的图像转写 Interface。"""

    async def caption(self, *, image_bytes: bytes, sfw: bool) -> str:  # pragma: no cover - 协议声明
        raise NotImplementedError


class MaiBotImageCaptioner(ImageCaptioner):
    """把图片交给视觉模型，拿回一段纯英文自然语言描述。"""

    def __init__(
        self,
        *,
        log_prefix: str = "nai_draw_plugin",
        model_name: str = "",
        custom_model: dict[str, Any] | None = None,
        max_tokens: int = 3000,
        temperature: float = 0.3,
        timeout: float = 120.0,
        prompt_template: str = "",
    ) -> None:
        self._log_prefix = log_prefix
        self._model_name = str(model_name or "").strip()
        self._custom_model = custom_model if isinstance(custom_model, dict) else {}
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._timeout = float(timeout)
        self._prompt_template = str(prompt_template or "")

    async def caption(self, *, image_bytes: bytes, sfw: bool) -> str:
        """把 image_bytes 转写成英文自然语言描述。

        Args:
            image_bytes: 原始图片字节
            sfw: 当前会话是否开着 NSFW 过滤，决定发哪一版 system_override

        Returns:
            清洗过的纯英文描述

        Raises:
            ImageCaptionError: 模型调用失败、超时，或返回内容清洗后为空
        """
        image_format = detect_image_format(image_bytes)
        image_base64 = ImageUtils.image_bytes_to_base64(image_bytes)
        # 自定义模型走 PinnedTaskLLMOrchestrator，绕过了 LLMServiceClient 里的自动放大，
        # 小图（表情包尺寸）会低于视觉模型的最小识别边长，必须在这里自己补上
        image_base64, image_format, resized = ImageUtils.normalize_image_base64_for_model(
            image_base64,
            image_format,
        )
        if resized:
            logger.info(f"{self._log_prefix} 反推图片低于视觉模型最小识别尺寸，已放大后再请求")

        prompt = resolve_caption_prompt(sfw=sfw, custom_template=self._prompt_template)

        try:
            raw = await asyncio.wait_for(
                self._request_caption(
                    prompt=prompt,
                    image_base64=image_base64,
                    image_format=image_format,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ImageCaptionError(f"视觉模型调用超时（{self._timeout:.0f}s）") from exc

        caption = normalize_caption_text(raw)
        if not caption:
            raise ImageCaptionError("视觉模型返回空描述（可能触发了内容拒答或被 max_tokens 截断）")
        return caption

    async def _request_caption(
        self,
        *,
        prompt: str,
        image_base64: str,
        image_format: str,
    ) -> str:
        """按"自定义模型优先"发起一次视觉请求，返回模型原始文本。"""
        pinned_task = build_pinned_task_config(
            self._custom_model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        if pinned_task is not None:
            logger.info(f"{self._log_prefix} 反推使用自定义视觉模型: {pinned_task.model_list}")
            orchestrator = PinnedTaskLLMOrchestrator(
                pinned_task,
                request_type="nai_draw_plugin.retag_caption",
            )
            result = await orchestrator.generate_response_for_image(
                prompt=prompt,
                image_base64=image_base64,
                image_format=image_format,
                temperature=coerce_float(self._custom_model.get("temperature"), self._temperature),
                max_tokens=coerce_int(self._custom_model.get("max_tokens"), self._max_tokens),
            )
            return result.response or ""

        task_name = self._resolve_task_name()
        logger.info(f"{self._log_prefix} 反推使用宿主任务模型: {task_name}")
        client = LLMServiceClient(
            task_name=task_name,
            request_type="nai_draw_plugin.retag_caption",
        )
        result = await client.generate_response_for_image(
            prompt=prompt,
            image_base64=image_base64,
            image_format=image_format,
            options=LLMImageOptions(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ),
        )
        return result.response or ""

    def _resolve_task_name(self) -> str:
        """解析宿主任务名：配置指定优先，其次 vlm。

        Raises:
            ImageCaptionError: 宿主没有任何可用任务模型
        """
        models = llm_service.get_available_models()
        if not models:
            raise ImageCaptionError("宿主未配置任何可用模型任务")

        for candidate in (self._model_name, "vlm"):
            if candidate and candidate in models:
                return candidate

        if self._model_name:
            raise ImageCaptionError(
                f"配置的 llm_model_name={self._model_name!r} 不在宿主模型任务里，"
                f"可用任务: {sorted(models)}"
            )
        raise ImageCaptionError(f"宿主没有 vlm 任务模型，可用任务: {sorted(models)}")


def normalize_caption_text(raw: str) -> str:
    """把模型原始输出清洗成可直接送 NAI 的英文描述。

    1. 输出被 ``` 包住时取代码块内容（模型偶尔会把描述当成"代码"输出）
    2. 剔除 CJK 与全角符号——NewAPI §8 对含中文的 prompt 一律 400，
       复用提示词链路同一个清洗函数，保证两边对"NAI 收得下什么字符"的判断一致
    """
    text = str(raw or "").strip()
    if not text:
        return ""

    code_block = extract_last_code_block(text)
    if code_block:
        text = code_block

    return strip_cjk_and_fullwidth(text).strip()


__all__ = [
    "ImageCaptionError",
    "ImageCaptioner",
    "MaiBotImageCaptioner",
    "detect_image_format",
    "normalize_caption_text",
]
