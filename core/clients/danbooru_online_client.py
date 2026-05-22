# -*- coding: utf-8 -*-
"""
DanbooruSearchOnline API 客户端

封装 https://sakizuki-danboorusearch.hf.space/api 的三个端点：
- /health  - 探活
- /search  - 语义标签检索
- /related - 共现标签推荐
"""

from typing import Any, Dict, List, Optional

import httpx

from src.common.logger import get_logger

logger = get_logger("nai_draw_plugin")

_DEFAULT_BASE_URL = "https://sakizuki-danboorusearch.hf.space/api"
# HuggingFace Spaces 冷启动较慢，首次请求可能需要较长时间
_DEFAULT_TIMEOUT = 90.0


class DanbooruOnlineClient:
    """DanbooruSearchOnline API 异步客户端"""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._available: Optional[bool] = None

    async def health_check(self) -> bool:
        """
        探活：检查远程服务是否可用。

        Returns:
            True 表示服务在线且模型已加载
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    self._available = data.get("status") == "ok" and data.get("loaded", False)
                else:
                    self._available = False
        except Exception as e:
            logger.warning(f"DanbooruOnline 探活失败: {e}")
            self._available = False
        return self._available

    @property
    def is_available(self) -> Optional[bool]:
        """上次探活结果，None 表示尚未检测"""
        return self._available

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        limit: int = 80,
        popularity_weight: float = 0.15,
        show_nsfw: bool = False,
        use_segmentation: bool = True,
        target_layers: Optional[List[str]] = None,
        target_categories: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        语义标签检索：自然语言 → Danbooru 标签

        Args:
            query: 用户自然语言描述（中文优化）
            top_k: 每个分词段召回数 (1-50)
            limit: 最终返回上限 (1-500)
            popularity_weight: 标签热度对排序的影响 (0.0-1.0)
            show_nsfw: 是否包含 NSFW 标签
            use_segmentation: 是否启用智能分词
            target_layers: 搜索的向量层
            target_categories: 标签类别过滤

        Returns:
            API 响应字典，包含 tags_all, tags_sfw, results, keywords；失败返回 None
        """
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "limit": limit,
            "popularity_weight": popularity_weight,
            "show_nsfw": show_nsfw,
            "use_segmentation": use_segmentation,
        }
        if target_layers is not None:
            payload["target_layers"] = target_layers
        if target_categories is not None:
            payload["target_categories"] = target_categories

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/search", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException:
            logger.warning(f"DanbooruOnline search 超时 (>{self.timeout}s)，query='{query[:30]}'")
            self._available = False
            return None
        except Exception as e:
            logger.warning(f"DanbooruOnline search 失败: {e}")
            return None

    async def related(
        self,
        tags: List[str],
        *,
        limit: int = 50,
        show_nsfw: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        共现标签推荐：已有标签 → 经常搭配出现的标签

        Args:
            tags: 种子标签列表（Danbooru 英文名）
            limit: 推荐上限 (1-200)
            show_nsfw: 是否包含 NSFW 标签

        Returns:
            推荐标签列表，每项含 tag, cn_name, category, cooc_score 等；失败返回 None
        """
        if not tags:
            return []

        payload = {
            "tags": tags,
            "limit": limit,
            "show_nsfw": show_nsfw,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/related", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException:
            logger.warning(f"DanbooruOnline related 超时 (>{self.timeout}s)")
            return None
        except Exception as e:
            logger.warning(f"DanbooruOnline related 失败: {e}")
            return None
