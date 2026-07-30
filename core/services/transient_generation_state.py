"""会话生成链路的有界瞬态状态。"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4


@dataclass
class _SessionEntry:
    last_accessed_at: float
    last_action_image_sent_at: float | None = None
    pending_image_generation_started_at: float | None = None
    pending_image_generation_owner: str | None = None
    nai_context: tuple[str, str, float, float | None] | None = None
    selfie_context: tuple[
        str,
        str,
        str,
        dict[str, list[str]],
        float,
        float | None,
    ] | None = None


class TransientGenerationState:
    """集中保存短期生成上下文，并自动淘汰陈旧会话。"""

    def __init__(
        self,
        *,
        max_entries: int = 2_000,
        idle_ttl_seconds: float = 86_400.0,
        pending_ttl_seconds: float = 900.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._idle_ttl_seconds = max(0.0, float(idle_ttl_seconds))
        self._pending_ttl_seconds = max(0.0, float(pending_ttl_seconds))
        self._clock = clock
        self._entries: OrderedDict[str, _SessionEntry] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def clear_session(self, stream_id: str) -> None:
        self._entries.pop(str(stream_id or "").strip(), None)

    def _get_entry(self, stream_id: str) -> _SessionEntry | None:
        normalized_stream_id = str(stream_id or "").strip()
        if not normalized_stream_id:
            return None
        now = self._clock()
        self._prune(now)
        entry = self._entries.get(normalized_stream_id)
        if entry is None:
            return None
        entry.last_accessed_at = now
        self._entries.move_to_end(normalized_stream_id)
        return entry

    def _ensure_entry(
        self,
        stream_id: str,
        *,
        enforce_capacity: bool = True,
    ) -> _SessionEntry | None:
        normalized_stream_id = str(stream_id or "").strip()
        if not normalized_stream_id:
            return None
        now = self._clock()
        self._prune(now)
        entry = self._entries.get(normalized_stream_id)
        if entry is None:
            entry = _SessionEntry(last_accessed_at=now)
            self._entries[normalized_stream_id] = entry
        else:
            entry.last_accessed_at = now
            self._entries.move_to_end(normalized_stream_id)
        if enforce_capacity:
            self._enforce_capacity(now)
        return entry

    def _prune(self, now: float) -> None:
        if self._idle_ttl_seconds <= 0:
            return
        for stream_id, entry in list(self._entries.items()):
            self._expire_contexts(entry, now)
            if now - entry.last_accessed_at <= self._idle_ttl_seconds:
                continue
            if self._pending_is_active(entry, now):
                continue
            if (
                entry.nai_context is not None
                or entry.selfie_context is not None
            ):
                entry.last_action_image_sent_at = None
                entry.pending_image_generation_started_at = None
                entry.pending_image_generation_owner = None
                continue
            self._entries.pop(stream_id, None)

    def _enforce_capacity(self, now: float) -> None:
        while len(self._entries) > self._max_entries:
            evictable_stream_id = next(
                (
                    stream_id
                    for stream_id, entry in self._entries.items()
                    if not self._pending_is_active(entry, now)
                ),
                None,
            )
            if evictable_stream_id is None:
                return
            self._entries.pop(evictable_stream_id, None)

    @staticmethod
    def _expire_contexts(entry: _SessionEntry, now: float) -> None:
        if entry.nai_context is not None:
            expires_at = entry.nai_context[-1]
            if expires_at is not None and now > expires_at:
                entry.nai_context = None
        if entry.selfie_context is not None:
            expires_at = entry.selfie_context[-1]
            if expires_at is not None and now > expires_at:
                entry.selfie_context = None

    def _pending_is_active(self, entry: _SessionEntry, now: float) -> bool:
        started_at = entry.pending_image_generation_started_at
        if started_at is None or entry.pending_image_generation_owner is None:
            return False
        return not (
            self._pending_ttl_seconds > 0
            and now - started_at > self._pending_ttl_seconds
        )

    def _clear_expired_pending(self, entry: _SessionEntry, now: float) -> None:
        if entry.pending_image_generation_started_at is None:
            return
        if self._pending_is_active(entry, now):
            return
        entry.pending_image_generation_started_at = None
        entry.pending_image_generation_owner = None

    def get_pending_image_generation_started_at(self, stream_id: str) -> float | None:
        entry = self._get_entry(stream_id)
        if entry is None or entry.pending_image_generation_started_at is None:
            return None
        self._clear_expired_pending(entry, self._clock())
        return entry.pending_image_generation_started_at

    def acquire_pending_image_generation(self, stream_id: str) -> str | None:
        """原子获取生成 lease；已有活跃任务时返回 ``None``。"""
        entry = self._ensure_entry(stream_id, enforce_capacity=False)
        if entry is None:
            return None
        now = self._clock()
        self._clear_expired_pending(entry, now)
        if entry.pending_image_generation_owner is not None:
            return None
        owner = uuid4().hex
        entry.pending_image_generation_started_at = now
        entry.pending_image_generation_owner = owner
        self._enforce_capacity(now)
        return owner

    def release_pending_image_generation(self, stream_id: str, owner: str) -> bool:
        """仅允许当前 lease owner 释放 pending，防止旧任务清掉新任务状态。"""
        entry = self._get_entry(stream_id)
        if entry is None or entry.pending_image_generation_owner != str(owner or ""):
            return False
        entry.pending_image_generation_started_at = None
        entry.pending_image_generation_owner = None
        return True

    def set_pending_image_generation(
        self,
        stream_id: str,
        started_at: float | None = None,
    ) -> None:
        entry = self._ensure_entry(stream_id, enforce_capacity=False)
        if entry is not None:
            now = self._clock()
            entry.pending_image_generation_started_at = float(
                started_at if started_at is not None else now
            )
            entry.pending_image_generation_owner = uuid4().hex
            self._enforce_capacity(now)

    def clear_pending_image_generation(self, stream_id: str) -> None:
        entry = self._get_entry(stream_id)
        if entry is not None:
            entry.pending_image_generation_started_at = None
            entry.pending_image_generation_owner = None

    def get_last_action_image_sent_at(self, stream_id: str) -> float | None:
        entry = self._get_entry(stream_id)
        return entry.last_action_image_sent_at if entry is not None else None

    def set_last_action_image_sent_at(
        self,
        stream_id: str,
        sent_at: float | None = None,
    ) -> None:
        entry = self._ensure_entry(stream_id)
        if entry is not None:
            entry.last_action_image_sent_at = float(
                sent_at if sent_at is not None else self._clock()
            )

    def get_last_nai_context(
        self,
        stream_id: str,
        ttl: float = 0,
    ) -> tuple[str | None, str | None]:
        entry = self._get_entry(stream_id)
        if entry is None or entry.nai_context is None:
            return None, None
        prompt, request, created_at, _expires_at = entry.nai_context
        if ttl > 0 and self._clock() - created_at > ttl:
            entry.nai_context = None
            return None, None
        return prompt, request or None

    def set_last_nai_context(
        self,
        stream_id: str,
        prompt: str,
        request: str = "",
        ttl: float = 0,
    ) -> None:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return
        entry = self._ensure_entry(stream_id, enforce_capacity=False)
        if entry is not None:
            created_at = self._clock()
            entry.nai_context = (
                prompt_text,
                str(request or "").strip(),
                created_at,
                created_at + float(ttl) if ttl > 0 else None,
            )
            self._enforce_capacity(created_at)

    def get_last_selfie_context(
        self,
        stream_id: str,
        ttl: float = 0,
    ) -> tuple[str | None, str | None, str | None, dict[str, list[str]]]:
        entry = self._get_entry(stream_id)
        if entry is None or entry.selfie_context is None:
            return None, None, None, {}
        prompt, request, scene_summary, anchor_data, created_at, _expires_at = entry.selfie_context
        if ttl > 0 and self._clock() - created_at > ttl:
            entry.selfie_context = None
            return None, None, None, {}
        return prompt or None, request or None, scene_summary or None, dict(anchor_data)

    def set_last_selfie_context(
        self,
        stream_id: str,
        prompt: str,
        request: str = "",
        scene_summary: str = "",
        anchor_data: dict[str, list[str]] | None = None,
        ttl: float = 0,
    ) -> None:
        prompt_text = str(prompt or "").strip()
        scene_text = str(scene_summary or "").strip()
        normalized_anchor_data = dict(anchor_data or {})
        if not prompt_text and not scene_text and not normalized_anchor_data:
            return
        entry = self._ensure_entry(stream_id, enforce_capacity=False)
        if entry is not None:
            created_at = self._clock()
            entry.selfie_context = (
                prompt_text,
                str(request or "").strip(),
                scene_text,
                normalized_anchor_data,
                created_at,
                created_at + float(ttl) if ttl > 0 else None,
            )
            self._enforce_capacity(created_at)
