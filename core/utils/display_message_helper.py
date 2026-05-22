# -*- coding: utf-8 -*-
"""图片消息展示文案辅助函数。"""

from typing import Optional


NAI_ACTION_IMAGE_DISPLAY_PREFIX = "[NAI图片:"
NAI_ACTION_IMAGE_DISPLAY_FALLBACK = "[NAI图片]"


def build_action_image_display_message(description: Optional[str]) -> str:
    """为 action 生成可读且稳定可识别的图片展示文本。"""
    normalized = " ".join(str(description or "").split())
    if not normalized:
        return NAI_ACTION_IMAGE_DISPLAY_FALLBACK
    return f"{NAI_ACTION_IMAGE_DISPLAY_PREFIX}{normalized}]"


def is_nai_action_image_display_message(text: Optional[str]) -> bool:
    """判断是否为 action 路径生成的 NAI 图片展示文本。"""
    if not isinstance(text, str):
        return False
    normalized = text.strip()
    return (
        normalized == NAI_ACTION_IMAGE_DISPLAY_FALLBACK
        or normalized.startswith(NAI_ACTION_IMAGE_DISPLAY_PREFIX)
    )
