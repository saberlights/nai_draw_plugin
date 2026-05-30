# -*- coding: utf-8 -*-
"""NSFW 过滤会话开关的持久化存储。

跨重启保留 ``/nai nsfw on|off`` 在每个 (platform, chat_id) 上的设定；
缺省（未在该会话执行过 /nai nsfw）时由 ``session_state`` 回退到
``nsfw_filter.enabled`` 配置默认值，不在本 store 里登记。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from src.common.logger import get_logger

logger = get_logger("nai_draw_plugin")


class NsfwStateStore:
    """按 ``platform:chat_id`` 持久化 NSFW 过滤开关，跨重启保留。"""

    _DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "nsfw_state.json"

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._storage_path = storage_path or self._DEFAULT_PATH
        self._entries: Dict[str, bool] = {}
        self._load()

    @staticmethod
    def _make_key(platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    def _load(self) -> None:
        """从磁盘加载已存的会话开关。"""
        with self._lock:
            if not self._storage_path.is_file():
                self._entries = {}
                return

            try:
                raw_data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[nai_draw_plugin] 读取 NSFW 状态文件失败，已忽略: %r", exc)
                self._entries = {}
                return

            if not isinstance(raw_data, dict):
                self._entries = {}
                return

            entries: Dict[str, bool] = {}
            for key, value in raw_data.items():
                normalized_key = str(key or "").strip()
                if not normalized_key:
                    continue
                entries[normalized_key] = bool(value)
            self._entries = entries

    def _save(self) -> None:
        """落盘当前快照，保证原子性靠 ``Path.write_text``。"""
        with self._lock:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(
                dict(sorted(self._entries.items())),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            self._storage_path.write_text(serialized + "\n", encoding="utf-8")

    def get(self, platform: str, chat_id: str) -> Optional[bool]:
        """查询会话开关；未登记返回 ``None``，由调用方回退配置默认。"""
        key = self._make_key(platform, chat_id)
        with self._lock:
            return self._entries.get(key)

    def set(self, platform: str, chat_id: str, enabled: bool) -> None:
        """登记 / 更新会话开关并落盘。"""
        key = self._make_key(platform, chat_id)
        with self._lock:
            if self._entries.get(key) == enabled:
                return
            self._entries[key] = bool(enabled)
            self._save()
        logger.info("[nai_draw_plugin] 会话 %s NSFW 过滤已持久化为: %s", key, enabled)

    def clear(self, platform: str, chat_id: str) -> None:
        """清除指定会话的登记（仅供 clear_session_state 之类显式重置使用）。"""
        key = self._make_key(platform, chat_id)
        with self._lock:
            if key not in self._entries:
                return
            self._entries.pop(key, None)
            self._save()


nsfw_state_store = NsfwStateStore()
