# -*- coding: utf-8 -*-
"""Bot 情景图视觉连续性状态的持久化存储。

跨重启保留每个聊天流的当前服装/环境稳定 Tag 与可切回的卡片库；
过期语义分层：当前服装/环境受 ``ttl`` 约束（视作"下线换过衣服了"），
卡片库不过期，始终可供 Planner/Tag LLM switch 回切。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from src.common.logger import get_logger

from .visual_continuity import StableVisualTags, VisualTagCard

logger = get_logger("nai_draw_plugin")


class VisualContinuityStore:
    """按聊天流持久化视觉连续性状态，跨重启保留。"""

    _DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "visual_continuity.json"

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        *,
        max_entries: int = 2_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.RLock()
        self._storage_path = storage_path or self._DEFAULT_PATH
        self._max_entries = max(1, int(max_entries))
        self._clock = clock
        self._entries: Dict[str, Tuple[StableVisualTags, float]] = {}
        self._load()

    # ==================== 磁盘同步 ====================

    def _load(self) -> None:
        """从磁盘加载已存的连续性状态；文件损坏时告警并按空库启动。"""
        with self._lock:
            if not self._storage_path.is_file():
                self._entries = {}
                return

            try:
                raw_data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[nai_draw_plugin] 读取视觉连续性状态文件失败，已忽略: %r", exc)
                self._entries = {}
                return

            if not isinstance(raw_data, dict):
                self._entries = {}
                return

            entries: Dict[str, Tuple[StableVisualTags, float]] = {}
            for stream_id, record in raw_data.items():
                normalized_id = str(stream_id or "").strip()
                if not normalized_id or not isinstance(record, dict):
                    continue
                entries[normalized_id] = (
                    self._deserialize_stable(record),
                    self._as_float(record.get("updated_at")),
                )
            self._entries = entries

    def _save(self) -> None:
        """落盘当前快照；超出容量时先淘汰最久未更新的聊天流。"""
        with self._lock:
            if len(self._entries) > self._max_entries:
                ordered = sorted(self._entries.items(), key=lambda item: item[1][1])
                for stream_id, _record in ordered[: len(self._entries) - self._max_entries]:
                    self._entries.pop(stream_id, None)
            serialized = json.dumps(
                {
                    stream_id: self._serialize_stable(stable, updated_at)
                    for stream_id, (stable, updated_at) in sorted(self._entries.items())
                },
                ensure_ascii=False,
                indent=2,
            )
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text(serialized + "\n", encoding="utf-8")

    @staticmethod
    def _serialize_stable(stable: StableVisualTags, updated_at: float) -> Dict[str, Any]:
        return {
            "updated_at": updated_at,
            "outfit": list(stable.outfit),
            "environment": list(stable.environment),
            "outfit_key": stable.outfit_key,
            "environment_key": stable.environment_key,
            "outfits": [
                {"key": card.key, "tags": list(card.tags)} for card in stable.outfits
            ],
            "environments": [
                {"key": card.key, "tags": list(card.tags)} for card in stable.environments
            ],
        }

    @classmethod
    def _deserialize_stable(cls, record: Dict[str, Any]) -> StableVisualTags:
        return StableVisualTags(
            outfit=cls._as_tag_tuple(record.get("outfit")),
            environment=cls._as_tag_tuple(record.get("environment")),
            outfit_key=str(record.get("outfit_key") or ""),
            environment_key=str(record.get("environment_key") or ""),
            outfits=cls._as_cards(record.get("outfits")),
            environments=cls._as_cards(record.get("environments")),
        )

    @staticmethod
    def _as_tag_tuple(value: Any) -> Tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(str(tag) for tag in value if isinstance(tag, str) and tag.strip())

    @classmethod
    def _as_cards(cls, value: Any) -> Tuple[VisualTagCard, ...]:
        if not isinstance(value, list):
            return ()
        cards = []
        for item in value:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            tags = cls._as_tag_tuple(item.get("tags"))
            if key and tags:
                cards.append(VisualTagCard(key, tags))
        return tuple(cards)

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # ==================== 读写接口 ====================

    def get(self, stream_id: str, ttl: float = 0) -> Optional[StableVisualTags]:
        """读取聊天流的连续性状态。

        ``ttl`` > 0 且当前服装/环境已超期时，仅保留卡片库返回（当前区清空，
        由 Tag LLM 依据上下文重建）；卡片库为空则返回 ``None``。
        """
        normalized_id = str(stream_id or "").strip()
        if not normalized_id:
            return None
        with self._lock:
            record = self._entries.get(normalized_id)
            if record is None:
                return None
            stable, updated_at = record
            if ttl > 0 and self._clock() - updated_at > ttl:
                if not stable.outfits and not stable.environments:
                    return None
                return StableVisualTags(
                    outfits=stable.outfits,
                    environments=stable.environments,
                )
            return stable

    def set(self, stream_id: str, stable: StableVisualTags) -> None:
        """保存聊天流的连续性状态并落盘。"""
        if not isinstance(stable, StableVisualTags):
            raise TypeError("stable must be StableVisualTags")
        normalized_id = str(stream_id or "").strip()
        if not normalized_id:
            return
        with self._lock:
            self._entries[normalized_id] = (stable, self._clock())
            self._save()

    def clear(self, stream_id: str) -> None:
        """清除指定聊天流的连续性状态（含卡片库）。"""
        normalized_id = str(stream_id or "").strip()
        with self._lock:
            if normalized_id not in self._entries:
                return
            self._entries.pop(normalized_id, None)
            self._save()


visual_continuity_store = VisualContinuityStore()
