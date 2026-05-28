# -*- coding: utf-8 -*-
"""
Tag 候选检索调度

按 retriever_config 选择 online/local 的逻辑收敛在这里，
供 sdk_runtime 统一调用。
"""

from typing import Any, Dict

from src.common.logger import get_logger

from .danbooru_online_retriever import get_online_retriever
from .tag_retriever import get_tag_retriever

logger = get_logger("nai_draw_plugin")


async def resolve_tag_candidates(
    retriever_config: Dict[str, Any],
    request_text: str,
    log_prefix: str = "",
) -> str:
    """根据 tag_retriever 配置返回候选标签文本块。

    Args:
        retriever_config: 插件配置中 ``tag_retriever`` 节点的内容
        request_text: 用户当次的中文描述
        log_prefix: 日志前缀，沿用调用方上下文

    Returns:
        可注入 ``<<TAG_CANDIDATES>>`` 的字符串；未启用或失败时返回空串
    """
    try:
        if not isinstance(retriever_config, dict) or not retriever_config.get("enabled", False):
            return ""

        mode = str(retriever_config.get("mode", "local") or "local").strip().lower()
        logger.info(
            f"{log_prefix} Tag 检索已启用，模式={mode}，query='{request_text[:30]}'"
        )

        if mode == "online":
            return await _resolve_online(retriever_config, request_text, log_prefix)
        return await _resolve_local(retriever_config, request_text, log_prefix)
    except Exception as exc:
        logger.warning(f"{log_prefix} Tag 检索失败，已跳过: {exc}")
        return ""


async def _resolve_online(
    retriever_config: Dict[str, Any],
    request_text: str,
    log_prefix: str,
) -> str:
    """在线检索；无结果或异常时回退到本地检索。"""
    try:
        retriever = get_online_retriever(
            enabled=True,
            base_url=retriever_config.get(
                "api_url", "https://sakizuki-danboorusearch.hf.space/api"
            ),
            timeout=retriever_config.get("timeout", 90.0),
            search_limit=retriever_config.get("search_limit", 30),
            search_top_k=retriever_config.get("search_top_k", 5),
            related_limit=retriever_config.get("related_limit", 20),
            related_seed_count=retriever_config.get("related_seed_count", 8),
            show_nsfw=retriever_config.get("show_nsfw", True),
            popularity_weight=retriever_config.get("popularity_weight", 0.15),
        )
    except Exception as exc:
        logger.warning(
            f"{log_prefix} Tag 在线检索初始化失败，回退到本地检索: {exc}"
        )
        return await _resolve_local(retriever_config, request_text, log_prefix)

    if not retriever:
        return ""

    results = await retriever.retrieve(query=request_text)
    search_count = len(results.get("search", []))
    related_count = len(results.get("related", []))
    if search_count == 0 and related_count == 0:
        logger.info(f"{log_prefix} Tag 在线检索无结果，回退到本地检索")
        return await _resolve_local(retriever_config, request_text, log_prefix)

    logger.info(
        f"{log_prefix} Tag 在线检索命中："
        f"query='{request_text[:30]}' search={search_count} related={related_count}"
    )
    return retriever.format_candidates(results)


async def _resolve_local(
    retriever_config: Dict[str, Any],
    request_text: str,
    log_prefix: str,
) -> str:
    """本地 embedding 检索。"""
    top_k = retriever_config.get("top_k", 20)
    min_score = retriever_config.get("min_score", 0.3)

    retriever = get_tag_retriever(
        enabled=True,
        top_k=top_k,
        min_score=min_score,
    )
    if not retriever:
        return ""

    results = await retriever.retrieve(
        query=request_text,
        top_k=top_k,
        min_score=min_score,
    )
    if not results:
        return ""

    tag_list = ", ".join(
        f"{item['cn']}→{item['tag']}({item['score']})" for item in results
    )
    logger.info(f"{log_prefix} Tag 本地检索命中：{tag_list}")
    return retriever.format_candidates(results)
