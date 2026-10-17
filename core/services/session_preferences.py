"""用户可切换的会话偏好状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionPreference:
    admin_mode: bool | None = None
    selected_model: str | None = None
    selected_backend: str | None = None
    # 画师预设索引按模型配置节隔离：{config_section: index}。
    # 原先是一个不带模型维度的裸 int，切模型时索引会串到另一个模型的预设上
    # （V4.5 选了 #5，切 V3 只有 2 个预设 → 越界回落，且该模型自己配的
    # default_artist_preset 被完全忽略）。
    selected_artist_index: dict[str, int] = field(default_factory=dict)
    selected_size: str | None = None
    recall_enabled: bool | None = None
    prompt_show_enabled: bool | None = None
    tag_retriever_show_enabled: bool | None = None
    character_reference_type: str | None = None
    retag_path: str | None = None


class SessionPreferences:
    """集中保存会话偏好，与短期生成上下文保持独立生命周期。"""

    def __init__(self) -> None:
        self._entries: dict[str, SessionPreference] = {}

    def get(self, key: str) -> SessionPreference | None:
        return self._entries.get(str(key or "").strip())

    def update(self, key: str, **changes: Any) -> None:
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return
        entry = self._entries.setdefault(normalized_key, SessionPreference())
        for field_name, value in changes.items():
            if not hasattr(entry, field_name):
                raise AttributeError(f"未知会话偏好字段: {field_name}")
            setattr(entry, field_name, value)

    def summary(self, key: str) -> dict[str, Any]:
        entry = self.get(key)
        if entry is None:
            return {
                "admin_mode": None,
                "model": None,
                "backend": None,
                "artist_index": {},
                "size": None,
                "recall": None,
                "prompt_show": None,
                "tag_retriever_show": None,
                "character_reference_type": None,
                "retag_path": None,
            }
        return {
            "admin_mode": entry.admin_mode,
            "model": entry.selected_model,
            "backend": entry.selected_backend,
            "artist_index": dict(entry.selected_artist_index),
            "size": entry.selected_size,
            "recall": entry.recall_enabled,
            "prompt_show": entry.prompt_show_enabled,
            "tag_retriever_show": entry.tag_retriever_show_enabled,
            "character_reference_type": entry.character_reference_type,
            "retag_path": entry.retag_path,
        }

    def clear(self, key: str) -> None:
        self._entries.pop(str(key or "").strip(), None)
