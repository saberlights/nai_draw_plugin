# -*- coding: utf-8 -*-
"""Vibe Transfer cache_id 本地缓存（文档 §20.3.1 / §20.3.2）。

NewAPI 对 ``controlnet.images[]`` 的图片编码按次收 anlas；网关在响应里以 HTML 注释
形式回传 ``vibe_cache_ids``。本模块把 ``(图片字节 hash, 模型名, info_extracted) → cache_id``
的映射落盘到独立 SQLite 文件，让下一次同图请求直接走 ``cache_id`` 复用态，
节省 1 anlas 流量附加费与编码成本。

故意不复用宿主 ``MaiBot.db``：缓存属于插件私有运行时状态，跨重启复用即可，
没必要污染宿主消息库。
"""

from __future__ import annotations

import base64
import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

from ..utils.image_meta import normalize_image_base64


# info_extracted 是浮点数，命中缓存时小数微差不应让 key 完全错位；
# 用 0.01 粒度量化（与文档允许的最小步长一致），平衡命中率与精度
_INFO_EXTRACTED_QUANTIZATION_STEP: float = 0.01

# cache_id 上限：22 字符 URL-safe base64（文档 §20.3.1）；
# 数据库列没限长，仅做基本合法性过滤
_CACHE_ID_MAX_LENGTH: int = 256


def compute_image_hash(image_data: str | bytes) -> str:
    """对图片字节做 SHA-256，作为 vibe cache 的稳定 key。

    支持 ``data:image/...;base64,xxx`` 形式与裸 base64 字符串；解码失败返回空串。
    """
    if isinstance(image_data, bytes):
        return hashlib.sha256(image_data).hexdigest()
    cleaned = normalize_image_base64(image_data)
    if not cleaned:
        return ""
    try:
        raw_bytes = base64.b64decode(cleaned, validate=False)
    except (ValueError, TypeError):
        return ""
    if not raw_bytes:
        return ""
    return hashlib.sha256(raw_bytes).hexdigest()


def quantize_info_extracted(value: float | int | None) -> float:
    """把 info_extracted 量化到 0.01 粒度；None / 非法值 fallback 到 0.7（文档默认）。"""
    if value is None:
        return 0.70
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.70
    if numeric <= 0.0:
        return 0.01
    if numeric >= 1.0:
        return 1.00
    quantized = round(numeric / _INFO_EXTRACTED_QUANTIZATION_STEP) * _INFO_EXTRACTED_QUANTIZATION_STEP
    return round(quantized, 2)


class VibeCacheService:
    """SQLite 后端的 vibe cache_id 持久化层。"""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS vibe_cache (
        image_hash TEXT NOT NULL,
        model_id TEXT NOT NULL,
        info_extracted REAL NOT NULL,
        cache_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (image_hash, model_id, info_extracted)
    )
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    # ── 公开接口 ────────────────────────────────────────────────────────────

    def lookup(
        self,
        *,
        image_hash: str,
        model_id: str,
        info_extracted: float | int | None,
    ) -> Optional[str]:
        """查询缓存；命中返回 cache_id，否则返回 None。"""
        if not image_hash or not model_id:
            return None
        quantized = quantize_info_extracted(info_extracted)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT cache_id FROM vibe_cache "
                "WHERE image_hash = ? AND model_id = ? AND info_extracted = ?",
                (image_hash, model_id, quantized),
            ).fetchone()
        if not row:
            return None
        cache_id = str(row[0] or "").strip()
        return cache_id or None

    def persist(
        self,
        *,
        image_hash: str,
        model_id: str,
        info_extracted: float | int | None,
        cache_id: str,
    ) -> bool:
        """落库；非法 cache_id / 缺 key 时静默跳过返回 False。"""
        if not image_hash or not model_id:
            return False
        normalized_cache_id = str(cache_id or "").strip()
        if not normalized_cache_id or len(normalized_cache_id) > _CACHE_ID_MAX_LENGTH:
            return False
        quantized = quantize_info_extracted(info_extracted)
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vibe_cache "
                "(image_hash, model_id, info_extracted, cache_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (image_hash, model_id, quantized, normalized_cache_id, now),
            )
            conn.commit()
        return True

    def delete(
        self,
        *,
        image_hash: str,
        model_id: str,
        info_extracted: float | int | None,
    ) -> bool:
        """删除单条缓存，返回是否真的删到行。

        用于服务端 cache_id 已被淘汰时清理本地映射，避免下次请求继续送 stale cache_id
        反复触发 §20.3.1 的 400。
        """
        if not image_hash or not model_id:
            return False
        quantized = quantize_info_extracted(info_extracted)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM vibe_cache "
                "WHERE image_hash = ? AND model_id = ? AND info_extracted = ?",
                (image_hash, model_id, quantized),
            )
            conn.commit()
            return (cursor.rowcount or 0) > 0

    def purge(self) -> int:
        """清空全部缓存，返回删除的行数。仅供测试与运维使用。"""
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM vibe_cache")
            conn.commit()
            return cursor.rowcount or 0

    def count(self) -> int:
        """当前缓存条目数，主要供测试使用。"""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM vibe_cache").fetchone()
        return int(row[0] if row else 0)

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._db_path), timeout=2.0)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(self._SCHEMA)
            conn.commit()


def iter_image_hashes(images: Iterable[dict]) -> List[str]:
    """便捷工具：对一组 controlnet.images[] 顺序计算 hash，仅供测试/日志使用。"""
    hashes: List[str] = []
    for item in images:
        if not isinstance(item, dict):
            hashes.append("")
            continue
        hashes.append(compute_image_hash(item.get("image", "") or ""))
    return hashes


# ── 全局单例 ───────────────────────────────────────────────────────────────

_DEFAULT_DB_FILENAME: str = "vibe_cache.db"
_INSTANCE: Optional[VibeCacheService] = None
_INSTANCE_LOCK = threading.Lock()


def _default_db_path() -> Path:
    """落到本插件目录下的 ``data/vibe_cache.db``，避免污染宿主数据库。"""
    return Path(__file__).resolve().parents[2] / "data" / _DEFAULT_DB_FILENAME


def get_vibe_cache_service(db_path: Optional[Path] = None) -> VibeCacheService:
    """获取（或惰性创建）模块级单例。

    传入 ``db_path`` 时会强制以该路径重建实例，主要供测试用。"""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if db_path is not None:
            _INSTANCE = VibeCacheService(db_path)
            return _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = VibeCacheService(_default_db_path())
        return _INSTANCE


def reset_vibe_cache_service() -> None:
    """重置单例，仅供测试 / 热重载使用。"""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


__all__ = [
    "VibeCacheService",
    "compute_image_hash",
    "quantize_info_extracted",
    "iter_image_hashes",
    "get_vibe_cache_service",
    "reset_vibe_cache_service",
]
