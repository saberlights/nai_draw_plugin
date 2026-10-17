# -*- coding: utf-8 -*-
"""NAI 反推子包：PNG 元数据 / WD14 在线 Space / 视觉模型自然语言，按路径分发。

对外暴露反推编排与图片缓存。视觉模型的具体实现放在 ``core/services/llm_image_captioner``，
本包只持有它的 Protocol，保持这里不依赖宿主 ``src.*``。
"""

from .image_cache import ImageCacheService
from .reverser import (
    KIND_NATURAL_LANGUAGE,
    KIND_TAGS,
    PATH_AUTO,
    PATH_LLM,
    PATH_META,
    PATH_WD14,
    REVERSE_PATH_LABELS,
    REVERSE_PATHS,
    ReverseResult,
    ReverseService,
    normalize_path,
)
from .wd14_client import WD14Client, WD14ClientError

__all__ = [
    "ImageCacheService",
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
    "WD14Client",
    "WD14ClientError",
    "normalize_path",
]
