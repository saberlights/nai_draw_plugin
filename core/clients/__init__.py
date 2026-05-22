# -*- coding: utf-8 -*-
"""NAI 图片生成插件 - 客户端层"""

from .danbooru_online_client import DanbooruOnlineClient
from .nai_web_client import NaiWebClient

__all__ = [
    "DanbooruOnlineClient",
    "NaiWebClient",
]
