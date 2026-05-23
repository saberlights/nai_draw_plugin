# -*- coding: utf-8 -*-
"""入站图片缓存 + 引用回复图片解析。

工作流程：
  * 所有入站消息经 `chat.receive.before_process` 钩子写入缓存（按 message_id 与会话）。
  * `/nai 反推` 执行前，`chat.command.before_execute` 把当前命令消息也存一份
    （用来兜底拿命令消息中本身携带的图片或 reply 信息）。
  * 命令执行时按优先级解析目标图：当前命令携图 → 引用回复消息 → 流内最近图。

数据完全保存在内存里，每条入站图有 TTL（默认 1 小时）和每会话上限。
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Set, Tuple

from .message_extractor import (
    deep_find_first,
    extract_image_base64_list_from_payload,
    find_reply_message_id,
)


class ImageCacheService:
    """会话级图片缓存，按 message_id 索引 base64。

    并发场景下读写都在 asyncio 单线程内进行，无需额外加锁。
    """

    def __init__(
        self,
        *,
        cache_ttl_seconds: float = 3600.0,
        per_stream_capacity: int = 20,
    ) -> None:
        self._cache_ttl_seconds = float(cache_ttl_seconds)
        self._per_stream_capacity = max(1, int(per_stream_capacity))

        # 最近命令消息：以 (stream_id, user_id) 为键，缓存最近一次触发反推的消息
        self._recent_command_messages: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # 所有入站消息的图片缓存：message_id -> {timestamp, stream_id, images}
        self._cached_image_messages: Dict[str, Dict[str, Any]] = {}
        # 每个会话保留的 message_id 顺序，用于"最近一张图"回溯
        self._stream_image_message_ids: Dict[str, Deque[str]] = {}

    def update_config(self, *, cache_ttl_seconds: float, per_stream_capacity: int) -> None:
        """配置热更新入口。"""
        self._cache_ttl_seconds = float(cache_ttl_seconds)
        self._per_stream_capacity = max(1, int(per_stream_capacity))

    def cache_inbound_message(self, message_data: Optional[Dict[str, Any]]) -> None:
        """缓存入站消息中的图片 base64。"""
        if not isinstance(message_data, dict):
            return

        message_id = str(message_data.get("message_id", "") or "").strip()
        stream_id = str(message_data.get("session_id", "") or "").strip()
        if not message_id:
            return

        images = self._extract_images(message_data)
        if not images:
            return

        self._cleanup()
        self._cached_image_messages[message_id] = {
            "timestamp": self._extract_timestamp(message_data),
            "stream_id": stream_id,
            "images": images,
        }

        if stream_id:
            stream_cache = self._stream_image_message_ids.setdefault(
                stream_id,
                deque(maxlen=self._per_stream_capacity),
            )
            if message_id in stream_cache:
                stream_cache.remove(message_id)
            stream_cache.append(message_id)

    def remember_command_message(self, message_data: Optional[Dict[str, Any]]) -> None:
        """缓存当前命令消息，方便 Command handler 后续读取 reply 信息。"""
        if not isinstance(message_data, dict):
            return

        stream_id = str(message_data.get("session_id", "") or "").strip()
        user_id = self._extract_user_id(message_data)
        if not stream_id or not user_id:
            return

        self._cleanup()
        self._recent_command_messages[(stream_id, user_id)] = dict(message_data)
        # 命令消息可能自带图，顺手也存图缓存
        self.cache_inbound_message(message_data)

    def consume_command_message(self, stream_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """取出并移除最近一次命令消息缓存。"""
        if not stream_id or not user_id:
            return None
        self._cleanup()
        return self._recent_command_messages.pop((stream_id, user_id), None)

    def resolve_image_base64(
        self,
        *,
        stream_id: str,
        user_id: str,
    ) -> Optional[str]:
        """按优先级解析本次反推目标图片。

        顺序：当前命令消息携带的图 → 引用回复消息缓存图 → 会话最近图片。
        命中即返回；都拿不到时返回 None。
        """
        excluded_ids: Set[str] = set()
        command_message = self.consume_command_message(stream_id, user_id)

        if command_message:
            command_message_id = str(command_message.get("message_id", "") or "")
            if command_message_id:
                excluded_ids.add(command_message_id)

            command_images = self._extract_images(command_message)
            if command_images:
                return command_images[0]

            reply_message_id = self._extract_reply_message_id(command_message)
            if reply_message_id:
                reply_images = self._get_cached_images_by_message_id(reply_message_id)
                if reply_images:
                    return reply_images[0]

        # 会话最近一张图（排除命令消息自己）
        return self._get_recent_stream_image(stream_id, exclude_ids=excluded_ids)

    # ── 内部辅助 ────────────────────────────────────────────────────────────

    def _get_cached_images_by_message_id(self, message_id: str) -> list[str]:
        if not message_id:
            return []
        self._cleanup()
        cache_data = self._cached_image_messages.get(message_id)
        if not isinstance(cache_data, dict):
            return []
        images = cache_data.get("images")
        return list(images) if isinstance(images, list) else []

    def _get_recent_stream_image(self, stream_id: str, *, exclude_ids: Set[str]) -> Optional[str]:
        if not stream_id:
            return None
        self._cleanup()
        message_ids = self._stream_image_message_ids.get(stream_id)
        if not message_ids:
            return None

        for message_id in reversed(message_ids):
            if message_id in exclude_ids:
                continue
            cached_images = self._get_cached_images_by_message_id(message_id)
            if cached_images:
                return cached_images[0]
        return None

    def _cleanup(self) -> None:
        """清理 TTL 过期条目。"""
        now = time.time()
        ttl = self._cache_ttl_seconds

        expired_command_keys = [
            cache_key
            for cache_key, message_data in self._recent_command_messages.items()
            if now - self._extract_timestamp(message_data) > ttl
        ]
        for cache_key in expired_command_keys:
            self._recent_command_messages.pop(cache_key, None)

        expired_message_ids = [
            message_id
            for message_id, cache_data in self._cached_image_messages.items()
            if now - float(cache_data.get("timestamp", 0.0) or 0.0) > ttl
        ]
        for message_id in expired_message_ids:
            self._cached_image_messages.pop(message_id, None)

        # 同步清理 stream 索引
        for stream_id, message_ids in list(self._stream_image_message_ids.items()):
            filtered = [mid for mid in message_ids if mid in self._cached_image_messages]
            if filtered:
                self._stream_image_message_ids[stream_id] = deque(
                    filtered,
                    maxlen=self._per_stream_capacity,
                )
            else:
                self._stream_image_message_ids.pop(stream_id, None)

    def clear(self) -> None:
        """清空全部缓存（插件卸载时调用）。"""
        self._recent_command_messages.clear()
        self._cached_image_messages.clear()
        self._stream_image_message_ids.clear()

    @staticmethod
    def _extract_images(message_data: Optional[Dict[str, Any]]) -> list[str]:
        """从消息字典中提取图片 base64，按优先级遍历可能的字段。"""
        if not isinstance(message_data, dict):
            return []

        payload_candidates = [
            message_data.get("message_segment"),
            message_data.get("raw_message"),
            message_data.get("message_info"),
            message_data,
        ]
        for payload in payload_candidates:
            images = extract_image_base64_list_from_payload(payload)
            if images:
                return images
        return []

    @staticmethod
    def _extract_user_id(message_data: Dict[str, Any]) -> str:
        message_info = message_data.get("message_info", {})
        if not isinstance(message_info, dict):
            return ""
        user_info = message_info.get("user_info", {})
        if not isinstance(user_info, dict):
            return ""
        return str(user_info.get("user_id", "") or "").strip()

    @staticmethod
    def _extract_timestamp(message_data: Dict[str, Any]) -> float:
        raw_timestamp = message_data.get("timestamp")
        try:
            return float(raw_timestamp or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_reply_message_id(message_data: Dict[str, Any]) -> Optional[str]:
        """挖出消息引用了哪条历史消息的 message_id。

        兼容平台差异：QQ 在 raw_message 里塞 reply 段，部分平台在 message_info 里写 reply_to_*。
        """
        reply_to = message_data.get("reply_to")
        if isinstance(reply_to, str) and reply_to.strip():
            return reply_to.strip()

        message_info = message_data.get("message_info")
        if isinstance(message_info, dict):
            for key in (
                "reply_to",
                "reply_to_message_id",
                "reply_message_id",
                "quote_message_id",
                "reply_id",
            ):
                value = message_info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            additional_config = message_info.get("additional_config")
            if isinstance(additional_config, dict):
                for key in (
                    "reply_to",
                    "reply_to_message_id",
                    "reply_message_id",
                    "quote_message_id",
                    "reply_id",
                ):
                    value = additional_config.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        raw_message = message_data.get("raw_message")
        if isinstance(raw_message, list):
            for component in raw_message:
                if not isinstance(component, dict) or component.get("type") != "reply":
                    continue
                reply_value = component.get("data")
                if isinstance(reply_value, str) and reply_value.strip():
                    return reply_value.strip()
                if isinstance(reply_value, dict):
                    for key in (
                        "target_message_id",
                        "message_id",
                        "id",
                        "reply_to",
                        "reply_to_message_id",
                        "reply_message_id",
                        "quote_message_id",
                        "reply_id",
                    ):
                        nested = reply_value.get(key)
                        if isinstance(nested, str) and nested.strip():
                            return nested.strip()

        # 兜底：递归到嵌套结构里搜索常见键
        reply_message_id = find_reply_message_id(message_data.get("message_segment"))
        if reply_message_id:
            return reply_message_id

        found = deep_find_first(
            message_data,
            keys={
                "target_message_id",
                "reply_to",
                "reply_to_message_id",
                "reply_message_id",
                "quote_message_id",
                "reply_id",
            },
        )
        if isinstance(found, int):
            return str(found)
        if isinstance(found, str) and found.strip():
            return found.strip()
        return None


__all__ = ["ImageCacheService"]
