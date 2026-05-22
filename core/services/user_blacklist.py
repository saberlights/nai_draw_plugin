# -*- coding: utf-8 -*-
"""用户黑名单服务。"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.common.logger import get_logger

logger = get_logger("nai_draw_plugin")


class UserBlacklistService:
    """管理插件级用户黑名单，并持久化到本地文件。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._storage_path = Path(__file__).resolve().parents[2] / "data" / "user_blacklist.json"
        self._entries: Dict[str, Dict[str, str]] = {}
        self._load()

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        """标准化用户 ID。"""
        return str(user_id or "").strip()

    def _load(self) -> None:
        """从磁盘加载黑名单。"""
        with self._lock:
            if not self._storage_path.is_file():
                self._entries = {}
                return

            try:
                raw_data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[nai_low] 读取黑名单文件失败，已忽略: %r", exc)
                self._entries = {}
                return

            if not isinstance(raw_data, dict):
                self._entries = {}
                return

            entries: Dict[str, Dict[str, str]] = {}
            for user_id, meta in raw_data.items():
                normalized_user_id = self._normalize_user_id(user_id)
                if not normalized_user_id:
                    continue

                if isinstance(meta, dict):
                    entries[normalized_user_id] = {
                        "created_at": str(meta.get("created_at", "") or ""),
                        "created_by": str(meta.get("created_by", "") or ""),
                    }
                else:
                    entries[normalized_user_id] = {
                        "created_at": "",
                        "created_by": "",
                    }

            self._entries = entries

    def _save(self) -> None:
        """将黑名单写回磁盘。"""
        with self._lock:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(
                dict(sorted(self._entries.items())),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            self._storage_path.write_text(serialized + "\n", encoding="utf-8")

    def is_blacklisted(self, user_id: str) -> bool:
        """检查用户是否在黑名单中。"""
        normalized_user_id = self._normalize_user_id(user_id)
        if not normalized_user_id:
            return False

        with self._lock:
            return normalized_user_id in self._entries

    def add_user(self, user_id: str, operator_id: str = "") -> bool:
        """添加黑名单用户。"""
        normalized_user_id = self._normalize_user_id(user_id)
        if not normalized_user_id:
            return False

        with self._lock:
            if normalized_user_id in self._entries:
                return False

            self._entries[normalized_user_id] = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "created_by": self._normalize_user_id(operator_id),
            }
            self._save()

        logger.info("[nai_low] 用户 %s 已加入黑名单", normalized_user_id)
        return True

    def remove_user(self, user_id: str) -> bool:
        """移除黑名单用户。"""
        normalized_user_id = self._normalize_user_id(user_id)
        if not normalized_user_id:
            return False

        with self._lock:
            if normalized_user_id not in self._entries:
                return False

            self._entries.pop(normalized_user_id, None)
            self._save()

        logger.info("[nai_low] 用户 %s 已移出黑名单", normalized_user_id)
        return True

    def list_entries(self) -> List[Dict[str, str]]:
        """列出全部黑名单用户。"""
        with self._lock:
            entries = []
            for user_id, meta in sorted(self._entries.items()):
                entries.append(
                    {
                        "user_id": user_id,
                        "created_at": str(meta.get("created_at", "") or ""),
                        "created_by": str(meta.get("created_by", "") or ""),
                    }
                )
            return entries


user_blacklist = UserBlacklistService()

