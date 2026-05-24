# -*- coding: utf-8 -*-
"""
在线 Danbooru Tag 检索服务

基于 DanbooruSearchOnline API，提供与本地 TagRetriever 相同的接口，
同时利用 /api/search（语义匹配）和 /api/related（共现推荐）双重候选。
"""

from typing import Any, Callable, Dict, List, Optional

from src.common.logger import get_logger

from ..clients.danbooru_online_client import DanbooruOnlineClient

logger = get_logger("nai_draw_plugin")


class DanbooruOnlineRetriever:
    """基于 DanbooruSearchOnline API 的在线标签检索器"""

    def __init__(
        self,
        base_url: str = "https://sakizuki-danboorusearch.hf.space/api",
        timeout: float = 90.0,
        search_limit: int = 30,
        search_top_k: int = 5,
        related_limit: int = 20,
        related_seed_count: int = 8,
        show_nsfw: bool = False,
        popularity_weight: float = 0.15,
    ):
        """
        Args:
            base_url: DanbooruSearchOnline API 地址
            timeout: 请求超时（秒）
            search_limit: search 端点返回上限
            search_top_k: search 端点每分词段召回数
            related_limit: related 端点返回上限
            related_seed_count: 用多少个 search 结果作为 related 的种子
            show_nsfw: 是否包含 NSFW 标签
            popularity_weight: 标签热度权重
        """
        self.client = DanbooruOnlineClient(base_url=base_url, timeout=timeout)
        self.search_limit = search_limit
        self.search_top_k = search_top_k
        self.related_limit = related_limit
        self.related_seed_count = related_seed_count
        self.show_nsfw = show_nsfw
        self.popularity_weight = popularity_weight

    def update_runtime_config(self, **kwargs) -> None:
        """更新运行时参数"""
        for key in ("search_limit", "search_top_k", "related_limit",
                    "related_seed_count", "show_nsfw", "popularity_weight"):
            if key in kwargs:
                setattr(self, key, kwargs[key])

    async def health_check(self) -> bool:
        """探活远程服务"""
        return await self.client.health_check()

    async def retrieve(
        self,
        query: str,
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        检索与查询最相关的标签，同时返回语义匹配和共现推荐。

        Args:
            query: 用户自然语言描述

        Returns:
            {
                "search": [{"tag": ..., "cn_name": ..., "score": ..., "category": ...}, ...],
                "related": [{"tag": ..., "cn_name": ..., "cooc_score": ..., "category": ...}, ...],
            }
            失败时返回空结构
        """
        empty_result = {"search": [], "related": []}

        if not query or not query.strip():
            return empty_result

        # 第一步：语义检索
        search_resp = await self.client.search(
            query=query,
            top_k=self.search_top_k,
            limit=self.search_limit,
            popularity_weight=self.popularity_weight,
            show_nsfw=self.show_nsfw,
            use_segmentation=True,
        )

        if not search_resp or not search_resp.get("results"):
            logger.warning(f"DanbooruOnline search 无结果，query='{query[:30]}'")
            return empty_result

        search_results = [
            {
                "tag": item["tag"],
                "cn_name": item.get("cn_name", ""),
                "score": item.get("final_score", 0.0),
                "category": item.get("category", "General"),
            }
            for item in search_resp["results"]
        ]

        # 第二步：取 top-N 标签作为种子，获取共现推荐
        seed_tags = [r["tag"] for r in search_results[:self.related_seed_count]]
        related_results = []

        if seed_tags:
            related_resp = await self.client.related(
                tags=seed_tags,
                limit=self.related_limit,
                show_nsfw=self.show_nsfw,
            )
            if related_resp:
                # 去重：排除已在 search 结果中的标签
                search_tag_set = {r["tag"] for r in search_results}
                related_results = [
                    {
                        "tag": item["tag"],
                        "cn_name": item.get("cn_name", ""),
                        "cooc_score": item.get("cooc_score", 0.0),
                        "category": item.get("category", "General"),
                    }
                    for item in related_resp
                    if item["tag"] not in search_tag_set
                ]

        logger.info(
            f"DanbooruOnline 检索完成：query='{query[:30]}' → "
            f"search={len(search_results)} 条, related={len(related_results)} 条"
        )

        return {"search": search_results, "related": related_results}

    def format_candidates(self, results: Dict[str, List[Dict]]) -> str:
        """
        将检索结果格式化为可注入 LLM 模板的文本块。

        Args:
            results: retrieve() 的返回值

        Returns:
            格式化的 <tag_candidates> 文本块
        """
        search_items = results.get("search", [])
        related_items = results.get("related", [])

        if not search_items and not related_items:
            return ""

        # 仅输出候选数据，使用规则统一在 prompt 模板 _TAG_CANDIDATES_USAGE 中描述
        lines = ["<tag_candidates>"]

        # 语义匹配部分
        if search_items:
            lines.append("## 语义匹配（与用户描述直接相关，优先选用）")
            for item in search_items:
                cn = item.get("cn_name", "")
                tag = item["tag"]
                category = item.get("category", "")
                score = item.get("score", 0.0)
                cn_part = f"{cn} → " if cn else ""
                lines.append(f"- {cn_part}{tag} [{category}] (相关度 {score:.2f})")

        # 共现推荐部分
        if related_items:
            lines.append("")
            lines.append("## 共现推荐（与上述标签在真实画作中经常搭配出现）")
            for item in related_items:
                cn = item.get("cn_name", "")
                tag = item["tag"]
                category = item.get("category", "")
                cooc = item.get("cooc_score", 0.0)
                cn_part = f"{cn} → " if cn else ""
                lines.append(f"- {cn_part}{tag} [{category}] (共现度 {cooc:.2f})")

        lines.append("</tag_candidates>")

        return "\n".join(lines)


# 模块级单例
_online_instance: Optional[DanbooruOnlineRetriever] = None


def reset_online_retriever() -> None:
    """重置在线检索器单例"""
    global _online_instance
    _online_instance = None


def get_online_retriever(
    enabled: bool = True,
    base_url: str = "https://sakizuki-danboorusearch.hf.space/api",
    timeout: float = 90.0,
    search_limit: int = 30,
    search_top_k: int = 5,
    related_limit: int = 20,
    related_seed_count: int = 8,
    show_nsfw: bool = False,
    popularity_weight: float = 0.15,
) -> Optional[DanbooruOnlineRetriever]:
    """获取在线检索器单例"""
    global _online_instance
    if not enabled:
        return None

    if _online_instance is None:
        _online_instance = DanbooruOnlineRetriever(
            base_url=base_url,
            timeout=timeout,
            search_limit=search_limit,
            search_top_k=search_top_k,
            related_limit=related_limit,
            related_seed_count=related_seed_count,
            show_nsfw=show_nsfw,
            popularity_weight=popularity_weight,
        )
    else:
        _online_instance.update_runtime_config(
            search_limit=search_limit,
            search_top_k=search_top_k,
            related_limit=related_limit,
            related_seed_count=related_seed_count,
            show_nsfw=show_nsfw,
            popularity_weight=popularity_weight,
        )
    return _online_instance
