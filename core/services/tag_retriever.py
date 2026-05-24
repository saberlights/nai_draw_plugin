# -*- coding: utf-8 -*-
"""
Danbooru Tag 检索服务

通过项目的 embedding API 对 tag 中文描述做 embedding，
提供余弦相似度检索，返回与用户查询最相关的候选 tag。
"""

import asyncio
import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.common.logger import get_logger
from src.config.config import model_config

from ...legacy_llm_request import LegacyLLMRequest

logger = get_logger("nai_draw_plugin")

_plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认路径
_DEFAULT_TAG_JSON = os.path.join(_plugin_root, "data", "danbooru_tags.json")
_DEFAULT_EMBEDDING_CACHE = os.path.join(_plugin_root, "data", "tag_embeddings.npy")


class TagRetriever:
    """基于语义 embedding 的 Danbooru tag 检索器"""

    def __init__(
        self,
        tag_json_path: str = _DEFAULT_TAG_JSON,
        embedding_cache_path: str = _DEFAULT_EMBEDDING_CACHE,
        min_score: float = 0.3,
        top_k: int = 20,
    ):
        self.tag_json_path = tag_json_path
        self.embedding_cache_path = embedding_cache_path
        self.min_score = min_score
        self.top_k = top_k

        self._tags: List[Dict[str, str]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def update_runtime_config(self, *, top_k: int, min_score: float) -> None:
        """更新运行时检索参数，供插件热重载后复用缓存实例。"""
        self.top_k = top_k
        self.min_score = min_score

    async def _ensure_loaded(self):
        """懒加载：首次调用时加载数据和 embeddings"""
        if self._loaded:
            return

        # 加载 tag 数据
        if not os.path.exists(self.tag_json_path):
            raise FileNotFoundError(
                f"Tag JSON 文件不存在: {self.tag_json_path}，"
                f"请先运行 core/utils/tag_data_builder.py 生成"
            )

        with open(self.tag_json_path, "r", encoding="utf-8") as f:
            self._tags = json.load(f)

        if not self._tags:
            raise ValueError("Tag 数据为空")

        # 尝试加载缓存的 embeddings
        if os.path.exists(self.embedding_cache_path) and self._cache_is_valid():
            self._embeddings = np.load(self.embedding_cache_path)
            # 如果有 tag 缓存文件，用它替换（可能有过滤）
            tag_cache_path = self.embedding_cache_path.replace(".npy", "_tags.json")
            if os.path.exists(tag_cache_path):
                with open(tag_cache_path, "r", encoding="utf-8") as f:
                    self._tags = json.load(f)
            logger.info(f"Tag 检索：从缓存加载 {self._embeddings.shape[0]} 条 embedding")
        else:
            logger.info(f"Tag 检索：开始为 {len(self._tags)} 条 tag 构建 embedding（首次可能较慢）...")
            await self._build_embeddings()

        self._loaded = True

    def _cache_is_valid(self) -> bool:
        """检查缓存是否与当前 tag 数据匹配"""
        try:
            cached = np.load(self.embedding_cache_path)
            # 优先检查 tag 缓存文件
            tag_cache_path = self.embedding_cache_path.replace(".npy", "_tags.json")
            if os.path.exists(tag_cache_path):
                with open(tag_cache_path, "r", encoding="utf-8") as f:
                    cached_tags = json.load(f)
                return cached.shape[0] == len(cached_tags)
            # 兼容旧缓存：行数匹配原始 tag 数
            return cached.shape[0] == len(self._tags)
        except Exception:
            return False

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """通过项目 API 获取单条文本的 embedding"""
        embedding_config = model_config.model_task_config.embedding
        llm = LegacyLLMRequest(model_set=embedding_config, request_type="embedding")
        try:
            embedding, _ = await llm.get_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Tag 检索：获取 embedding 失败: {e}")
            return None

    async def _get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 50,
    ) -> List[Optional[List[float]]]:
        """批量获取 embedding，使用并发控制避免 API 过载"""
        results: List[Optional[List[float]]] = [None] * len(texts)
        semaphore = asyncio.Semaphore(batch_size)

        async def _embed_one(idx: int, text: str):
            async with semaphore:
                results[idx] = await self._get_embedding(text)

        tasks = [_embed_one(i, t) for i, t in enumerate(texts)]

        # 分批执行，每批之间打印进度
        total = len(tasks)
        chunk_size = batch_size * 2
        for start in range(0, total, chunk_size):
            chunk = tasks[start : start + chunk_size]
            await asyncio.gather(*chunk)
            done = min(start + chunk_size, total)
            logger.info(f"Tag 检索：embedding 进度 {done}/{total}")

        return results

    async def _build_embeddings(self):
        """为所有 tag 计算 embedding 并缓存"""
        # 用 "中文描述 tag英文" 拼接作为 embedding 文本
        texts = [f"{t['cn']} {t['tag']}" for t in self._tags]

        raw_embeddings = await self._get_embeddings_batch(texts)

        # 过滤失败的，记录成功率
        valid_indices = []
        valid_vectors = []
        for i, emb in enumerate(raw_embeddings):
            if emb is not None:
                valid_indices.append(i)
                valid_vectors.append(emb)

        if not valid_vectors:
            raise RuntimeError("Tag 检索：所有 embedding 请求均失败")

        failed = len(texts) - len(valid_vectors)
        if failed > 0:
            logger.warning(f"Tag 检索：{failed}/{len(texts)} 条 embedding 失败，将跳过这些 tag")
            # 只保留成功的 tag
            self._tags = [self._tags[i] for i in valid_indices]

        # 转 numpy 并 L2 归一化
        embeddings = np.array(valid_vectors, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._embeddings = embeddings / norms

        os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
        np.save(self.embedding_cache_path, self._embeddings)

        # 同时保存对应的 tag 列表（因为可能有失败被过滤的）
        tag_cache_path = self.embedding_cache_path.replace(".npy", "_tags.json")
        with open(tag_cache_path, "w", encoding="utf-8") as f:
            json.dump(self._tags, f, ensure_ascii=False)

        logger.info(f"Tag 检索：embedding 构建完成，共 {len(self._tags)} 条，已缓存")

    def _search_by_vector(
        self,
        query_emb: List[float],
        top_k: int,
        min_score: float,
    ) -> Dict[str, Dict]:
        """用已有的 embedding 向量检索，返回 {tag: {tag, cn, score}}"""
        query_vec = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores = (self._embeddings @ query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = {}
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            tag = self._tags[idx]["tag"]
            hits[tag] = {
                "tag": tag,
                "cn": self._tags[idx]["cn"],
                "score": round(score, 4),
            }
        return hits

    @staticmethod
    def _segment_query(query: str) -> List[str]:
        """将查询拆分为关键词（按空格/逗号/顿号分割，≥2字）"""
        import re
        parts = re.split(r'[\s,，、]+', query)
        keywords = [p.strip() for p in parts if len(p.strip()) >= 2]
        return keywords

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
    ) -> List[Dict]:
        """
        检索与查询最相关的 tag。

        策略：整句 + 分词关键词并发获取 embedding，各自检索后合并去重取最高分。
        """
        if not query or not query.strip():
            return []

        await self._ensure_loaded()

        top_k = top_k or self.top_k
        min_score = min_score if min_score is not None else self.min_score
        per_query_k = max(top_k // 2, 15)

        # 分词
        keywords = self._segment_query(query)
        if keywords:
            logger.info(f"Tag 检索：分词结果: {keywords}")

        # 整句 + 所有关键词并发获取 embedding
        all_queries = [query] + keywords
        embeddings = await asyncio.gather(
            *[self._get_embedding(q) for q in all_queries]
        )

        # 各自检索并合并
        merged: Dict[str, Dict] = {}
        for emb in embeddings:
            if emb is None:
                continue
            hits = self._search_by_vector(emb, per_query_k, min_score)
            for tag, item in hits.items():
                if tag not in merged or item["score"] > merged[tag]["score"]:
                    merged[tag] = item

        # 按分数排序，取 top_k
        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return results

    def format_candidates(self, results: List[Dict]) -> str:
        """将检索结果格式化为可注入模板的文本（使用规则统一在 prompt 模板中描述）"""
        if not results:
            return ""

        lines = [f"- {r['cn']} → {r['tag']}" for r in results]

        return (
            "<tag_candidates>\n"
            "## 语义匹配（与用户描述直接相关，优先选用）\n"
            + "\n".join(lines)
            + "\n</tag_candidates>"
        )


# 模块级单例
_instance: Optional[TagRetriever] = None


def _normalize_top_k(top_k: int) -> int:
    """标准化 top_k，避免热更新时传入非法值。"""
    try:
        normalized = int(top_k)
    except (TypeError, ValueError):
        return 20
    return max(1, normalized)


def _normalize_min_score(min_score: float) -> float:
    """标准化 min_score，避免热更新时传入非法值。"""
    try:
        normalized = float(min_score)
    except (TypeError, ValueError):
        return 0.3
    return max(0.0, normalized)


def reset_tag_retriever() -> None:
    """重置模块级单例，供插件卸载或热重载时清空旧实例。"""
    global _instance
    _instance = None


def get_tag_retriever(
    enabled: bool = True,
    top_k: int = 20,
    min_score: float = 0.3,
) -> Optional[TagRetriever]:
    """获取 TagRetriever 单例"""
    global _instance
    if not enabled:
        return None
    normalized_top_k = _normalize_top_k(top_k)
    normalized_min_score = _normalize_min_score(min_score)
    if _instance is None:
        _instance = TagRetriever(top_k=normalized_top_k, min_score=normalized_min_score)
        return _instance

    if _instance.top_k != normalized_top_k or abs(_instance.min_score - normalized_min_score) > 1e-9:
        _instance.update_runtime_config(top_k=normalized_top_k, min_score=normalized_min_score)
    return _instance
