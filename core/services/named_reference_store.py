# -*- coding: utf-8 -*-
"""vibe / ref 命名图库的本地存储。

设计：原始字节落文件系统，metadata 在线推断（不维护单独索引）；选定状态用
单文件 JSON 跨重启保留。这样图片能被文件管理器直接打开看，没有 base64 33%
膨胀，也不引入额外 BLOB 类的数据库依赖。

目录结构：

    <root>/
      users/<user_dir>/vibe/<name>.<ext>
      users/<user_dir>/ref/<name>.<ext>
      selection.json

其中 ``<user_dir>`` 用 user_id 的 sha256 前缀；用户 id 里可能带冒号 / 反斜杠
等不同平台的不可控字符，hash 一遍最稳，跨平台都安全。selection.json 的写入
走"写临时文件 → 原子 rename"避免半写入污染。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


__all__ = [
    "NamedReference",
    "NamedReferenceStore",
    "SCOPE_VIBE",
    "SCOPE_REF",
    "MAX_NAME_LENGTH",
    "max_selection_for_scope",
    "InvalidNameError",
    "InvalidImageError",
    "CapacityExceededError",
]


SCOPE_VIBE = "vibe"
SCOPE_REF = "ref"
_VALID_SCOPES = frozenset({SCOPE_VIBE, SCOPE_REF})

# §20.3 controlnet.images 最多 4 张；§20.4 character_references 最多 1 张
_MAX_SELECTION_PER_SCOPE: Dict[str, int] = {
    SCOPE_VIBE: 4,
    SCOPE_REF: 1,
}


def max_selection_for_scope(scope: str) -> int:
    """对外暴露每个 scope 的选定上限，供命令层做错误提示用。"""
    return _MAX_SELECTION_PER_SCOPE.get(scope, 1)


# 1-32 字符，汉字 + 英文字母 + 数字 + 下划线；空格、路径符、@ 一律拒
MAX_NAME_LENGTH = 32
_VALID_NAME_PATTERN = re.compile(r"^[一-鿿a-zA-Z0-9_]{1,32}$")

# 默认每个 (user_id, scope) 下最多 20 张，跟 image_cache_per_stream 同量级
_DEFAULT_MAX_PER_SCOPE = 20


class InvalidNameError(ValueError):
    """名字不符合规则（长度 / 字符）。"""


class InvalidImageError(ValueError):
    """图片字节不是 PNG / JPEG / WebP。"""


class CapacityExceededError(RuntimeError):
    """超过单个图库容量上限。"""


class NamedReference(NamedTuple):
    """单条命名图的元信息。字节内容按需用 ``get()`` 读。"""

    name: str
    image_format: str  # "png" / "jpeg" / "webp"
    width: int
    height: int
    byte_size: int
    created_at: float
    path: Path


class NamedReferenceStore:
    """vibe / ref 的命名图库。

    所有公共方法都按 ``(scope, user_id, ...)`` 索引；线程安全（asyncio 单线程
    场景实际上不会并发，但 selection.json 的读写要原子）。
    """

    def __init__(
        self,
        root: Path,
        *,
        max_per_scope: int = _DEFAULT_MAX_PER_SCOPE,
    ) -> None:
        self._root = Path(root)
        self._max_per_scope = max(1, int(max_per_scope))
        self._users_root = self._root / "users"
        self._selection_path = self._root / "selection.json"
        self._lock = threading.Lock()
        self._users_root.mkdir(parents=True, exist_ok=True)

    # ── 公共：图本身 ───────────────────────────────────────────────────────

    def save(
        self,
        *,
        scope: str,
        user_id: str,
        name: str,
        image_bytes: bytes,
    ) -> NamedReference:
        """落一张命名图；覆盖同名旧图。

        Raises:
            InvalidNameError: name 不合规则
            InvalidImageError: image_bytes 不是合法 PNG/JPEG/WebP
            CapacityExceededError: 新增（不是覆盖）会超过单库上限
        """
        self._validate_scope(scope)
        self._validate_name(name)
        image_format = self._detect_format(image_bytes)
        if image_format is None:
            raise InvalidImageError(
                "image_bytes 必须是 PNG / JPEG / WebP，文件头未匹配任一格式"
            )
        dims = self._read_dimensions(image_bytes, image_format)
        if dims is None:
            raise InvalidImageError("image_bytes 格式可识别但宽高解析失败，可能已损坏")

        with self._lock:
            scope_dir = self._scope_dir(scope, user_id)
            scope_dir.mkdir(parents=True, exist_ok=True)
            existing = self._list_files(scope_dir)
            target_stem = name
            # 新增（不是覆盖现有同名）才计入容量
            already_present = any(p.stem == target_stem for p in existing)
            if not already_present and len(existing) >= self._max_per_scope:
                raise CapacityExceededError(
                    f"{scope} 图库已存 {len(existing)} 张（上限 {self._max_per_scope}），"
                    "请先 /nai " + scope + "删 一张再存"
                )

            # 同名旧图无论扩展名是否变化都覆盖：先删后写
            for old in existing:
                if old.stem == target_stem:
                    try:
                        old.unlink()
                    except OSError:
                        pass

            final_path = scope_dir / f"{name}.{image_format}"
            self._atomic_write_bytes(final_path, image_bytes)

        width, height = dims
        return NamedReference(
            name=name,
            image_format=image_format,
            width=width,
            height=height,
            byte_size=len(image_bytes),
            created_at=final_path.stat().st_mtime,
            path=final_path,
        )

    def list(self, *, scope: str, user_id: str) -> List[NamedReference]:
        """列出 (scope, user_id) 下所有命名图，按名字字典序。"""
        self._validate_scope(scope)
        scope_dir = self._scope_dir(scope, user_id)
        if not scope_dir.exists():
            return []
        entries: List[NamedReference] = []
        for path in sorted(self._list_files(scope_dir), key=lambda p: p.stem):
            ref = self._read_metadata(path)
            if ref is not None:
                entries.append(ref)
        return entries

    def get(self, *, scope: str, user_id: str, name: str) -> Optional[bytes]:
        """读图字节。不存在返回 None；名字非法直接抛。"""
        self._validate_scope(scope)
        self._validate_name(name)
        scope_dir = self._scope_dir(scope, user_id)
        if not scope_dir.exists():
            return None
        for path in self._list_files(scope_dir):
            if path.stem == name:
                try:
                    return path.read_bytes()
                except OSError:
                    return None
        return None

    def delete(self, *, scope: str, user_id: str, name: str) -> bool:
        """删一张；删了返回 True，没找到返回 False。"""
        self._validate_scope(scope)
        self._validate_name(name)
        scope_dir = self._scope_dir(scope, user_id)
        if not scope_dir.exists():
            return False
        deleted = False
        with self._lock:
            for path in self._list_files(scope_dir):
                if path.stem == name:
                    try:
                        path.unlink()
                        deleted = True
                    except OSError:
                        return False
            # 顺便把这张图相关的"粘性选定"清掉，避免选定指向不存在的图
            if deleted:
                self._mutate_selection(
                    lambda data: self._drop_selection_for_name(
                        data, scope=scope, user_id=user_id, name=name
                    )
                )
        return deleted

    def clear_all(self, *, scope: str, user_id: str) -> int:
        """删 (scope, user_id) 下的所有图 + 清掉该 (scope, user_id) 在所有 stream 上的选定。

        语义是"一键清空当前用户在该图库的状态"，专给 /nai vibe清空 / /nai ref清空 用。
        删除按 best-effort 推进：单张 unlink 失败不会回滚已成功的删除（文件系统级原子性靠
        rename/unlink 自身保证；多张删一半失败时下次调用可继续清剩下的）。

        Returns:
            实际删除的图片张数；目录不存在或图库本来就空时返回 0。
        """
        self._validate_scope(scope)
        scope_dir = self._scope_dir(scope, user_id)
        deleted = 0
        with self._lock:
            if scope_dir.exists():
                for path in self._list_files(scope_dir):
                    try:
                        path.unlink()
                        deleted += 1
                    except OSError:
                        continue
            # 整体清掉该 (scope, user) 在所有 stream 上的选定，跟 delete 的语义对齐
            self._mutate_selection(
                lambda data: self._drop_all_selections_for_user(
                    data, scope=scope, user_id=user_id
                )
            )
        return deleted

    # ── 公共：选定 ────────────────────────────────────────────────────────

    def set_selection(
        self,
        *,
        scope: str,
        user_id: str,
        stream_id: str,
        names: List[str],
    ) -> None:
        """把 (scope, user_id, stream_id) 的粘性选定指向 names 列表。

        ``vibe`` 接受 1~4 张（§20.3 controlnet.images 最多 4 张），
        ``ref`` 仅接受 1 张（§20.4 character_references 最多 1 张）。

        Raises:
            InvalidNameError: 任一 name 不合规则
            KeyError: 任一 name 对应的图不存在
            ValueError: names 数量超出 scope 允许的上限 / 为空 / stream_id 为空
        """
        self._validate_scope(scope)
        if not stream_id:
            raise ValueError("stream_id 不能为空")
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError("names 必须是非空列表；没图请用 clear_selection")
        max_count = _MAX_SELECTION_PER_SCOPE.get(scope, 1)
        if len(names) > max_count:
            raise ValueError(
                f"{scope} 最多选 {max_count} 张，收到 {len(names)} 张"
            )
        # 名字校验 + 存在性校验全部通过后才落库，避免半成功
        normalized: List[str] = []
        for n in names:
            self._validate_name(n)
            if self.get(scope=scope, user_id=user_id, name=n) is None:
                raise KeyError(f"{scope} 图库里没有命名图 {n!r}")
            normalized.append(n)
        user_dir_name = self._user_dir_name(user_id)
        with self._lock:
            self._mutate_selection(
                lambda data: self._set_selection_entry(
                    data,
                    scope=scope,
                    user_dir_name=user_dir_name,
                    stream_id=stream_id,
                    names=normalized,
                )
            )

    def get_selection(
        self,
        *,
        scope: str,
        user_id: str,
        stream_id: str,
    ) -> List[str]:
        """读 (scope, user_id, stream_id) 的粘性选定列表；没选过返回空列表。

        兼容旧版 selection.json 里的单字符串形态：读到 str 时自动升级为 [str]。
        """
        self._validate_scope(scope)
        if not stream_id:
            return []
        data = self._read_selection()
        user_dir_name = self._user_dir_name(user_id)
        scope_data = data.get(scope)
        if not isinstance(scope_data, dict):
            return []
        user_data = scope_data.get(user_dir_name)
        if not isinstance(user_data, dict):
            return []
        value = user_data.get(stream_id)
        # 旧格式：单 string；新格式：list[str]
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str) and v]
        return []

    def clear_selection(
        self,
        *,
        scope: str,
        user_id: str,
        stream_id: str,
    ) -> None:
        """清掉 (scope, user_id, stream_id) 的选定；没选过也安全。"""
        self._validate_scope(scope)
        if not stream_id:
            return
        user_dir_name = self._user_dir_name(user_id)
        with self._lock:
            self._mutate_selection(
                lambda data: self._clear_selection_entry(
                    data,
                    scope=scope,
                    user_dir_name=user_dir_name,
                    stream_id=stream_id,
                )
            )

    # ── 内部辅助：路径 / 校验 ──────────────────────────────────────────────

    def _scope_dir(self, scope: str, user_id: str) -> Path:
        return self._users_root / self._user_dir_name(user_id) / scope

    @staticmethod
    def _user_dir_name(user_id: str) -> str:
        """user_id 跨平台带不可控字符（冒号、@、空格等），hash 后取前 16 位最稳。"""
        digest = hashlib.sha256(str(user_id or "").encode("utf-8")).hexdigest()
        return digest[:16]

    @staticmethod
    def _validate_scope(scope: str) -> None:
        if scope not in _VALID_SCOPES:
            raise ValueError(f"scope 必须是 {sorted(_VALID_SCOPES)} 之一，收到 {scope!r}")

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not _VALID_NAME_PATTERN.match(name):
            raise InvalidNameError(
                f"名字 {name!r} 不合规：1~{MAX_NAME_LENGTH} 字符，"
                "仅允许汉字 / 英文字母 / 数字 / 下划线"
            )

    @staticmethod
    def _list_files(scope_dir: Path) -> List[Path]:
        """列 scope_dir 下所有图片文件（PNG/JPEG/WebP 扩展名）。"""
        if not scope_dir.exists():
            return []
        allowed = (".png", ".jpg", ".jpeg", ".webp")
        return [
            p for p in scope_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed
        ]

    # ── 内部辅助：格式检测 / 尺寸读取（不引入 Pillow） ───────────────────

    @staticmethod
    def _detect_format(image_bytes: bytes) -> Optional[str]:
        if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) < 12:
            return None
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            return "jpeg"
        if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "webp"
        return None

    @staticmethod
    def _read_dimensions(image_bytes: bytes, image_format: str) -> Optional[tuple[int, int]]:
        # 复用插件内的 image_meta 解析；不在这里 import 避免循环依赖，所以延迟 import
        from ..utils.image_meta import read_image_dimensions

        return read_image_dimensions(image_bytes)

    def _read_metadata(self, path: Path) -> Optional[NamedReference]:
        try:
            stat = path.stat()
            image_bytes = path.read_bytes()
        except OSError:
            return None
        image_format = self._detect_format(image_bytes)
        if image_format is None:
            return None
        dims = self._read_dimensions(image_bytes, image_format)
        if dims is None:
            return None
        width, height = dims
        return NamedReference(
            name=path.stem,
            image_format=image_format,
            width=width,
            height=height,
            byte_size=stat.st_size,
            created_at=stat.st_mtime,
            path=path,
        )

    # ── 内部辅助：原子写入 / selection.json ─────────────────────────────

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        """先写到同目录临时文件，flush + fsync 后 rename，避免半写入污染。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(content)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, str(path))
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _read_selection(self) -> Dict:
        if not self._selection_path.exists():
            return {}
        try:
            with self._selection_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_selection(self, data: Dict) -> None:
        self._selection_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        self._atomic_write_bytes(self._selection_path, payload.encode("utf-8"))

    def _mutate_selection(self, mutator) -> None:
        """读 → 改 → 写的原子封装，调用方在已持锁状态下使用。"""
        data = self._read_selection()
        mutator(data)
        self._write_selection(data)

    @staticmethod
    def _set_selection_entry(
        data: Dict,
        *,
        scope: str,
        user_dir_name: str,
        stream_id: str,
        names: List[str],
    ) -> None:
        scope_bucket = data.setdefault(scope, {})
        if not isinstance(scope_bucket, dict):
            scope_bucket = {}
            data[scope] = scope_bucket
        user_bucket = scope_bucket.setdefault(user_dir_name, {})
        if not isinstance(user_bucket, dict):
            user_bucket = {}
            scope_bucket[user_dir_name] = user_bucket
        user_bucket[stream_id] = list(names)

    @staticmethod
    def _clear_selection_entry(
        data: Dict,
        *,
        scope: str,
        user_dir_name: str,
        stream_id: str,
    ) -> None:
        scope_bucket = data.get(scope)
        if not isinstance(scope_bucket, dict):
            return
        user_bucket = scope_bucket.get(user_dir_name)
        if not isinstance(user_bucket, dict):
            return
        user_bucket.pop(stream_id, None)
        if not user_bucket:
            scope_bucket.pop(user_dir_name, None)
        if not scope_bucket:
            data.pop(scope, None)

    def _drop_selection_for_name(
        self,
        data: Dict,
        *,
        scope: str,
        user_id: str,
        name: str,
    ) -> None:
        """图被删时，把所有指向它的粘性选定也一起清掉。

        新版选定是 list，所以这里要从每个 stream 的列表里把 name 剔掉；
        剔完为空再删 stream key。兼容旧 string 格式（直接整条删）。"""
        user_dir_name = self._user_dir_name(user_id)
        scope_bucket = data.get(scope)
        if not isinstance(scope_bucket, dict):
            return
        user_bucket = scope_bucket.get(user_dir_name)
        if not isinstance(user_bucket, dict):
            return
        for stream_id in list(user_bucket.keys()):
            value = user_bucket.get(stream_id)
            if isinstance(value, str):
                # 旧 string 格式：仅在等于待删名时整条干掉
                if value == name:
                    user_bucket.pop(stream_id, None)
            elif isinstance(value, list):
                new_list = [v for v in value if v != name]
                if not new_list:
                    user_bucket.pop(stream_id, None)
                else:
                    user_bucket[stream_id] = new_list
            else:
                # 异常格式直接清掉
                user_bucket.pop(stream_id, None)
        if not user_bucket:
            scope_bucket.pop(user_dir_name, None)
        if not scope_bucket:
            data.pop(scope, None)

    def _drop_all_selections_for_user(
        self,
        data: Dict,
        *,
        scope: str,
        user_id: str,
    ) -> None:
        """清空该 (scope, user_id) 在所有 stream 上的选定，配合 clear_all 用。"""
        user_dir_name = self._user_dir_name(user_id)
        scope_bucket = data.get(scope)
        if not isinstance(scope_bucket, dict):
            return
        scope_bucket.pop(user_dir_name, None)
        if not scope_bucket:
            data.pop(scope, None)


# ── 全局单例（仿照 vibe_cache_service） ──────────────────────────────────

_INSTANCE: Optional[NamedReferenceStore] = None
_INSTANCE_LOCK = threading.Lock()


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "named_refs"


def get_named_reference_store(
    root: Optional[Path] = None,
    *,
    max_per_scope: Optional[int] = None,
) -> NamedReferenceStore:
    """获取（或惰性创建）模块级单例。

    传 root 时强制以新路径重建，主要供测试用。"""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if root is not None:
            _INSTANCE = NamedReferenceStore(
                root,
                max_per_scope=max_per_scope or _DEFAULT_MAX_PER_SCOPE,
            )
            return _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = NamedReferenceStore(
                _default_root(),
                max_per_scope=max_per_scope or _DEFAULT_MAX_PER_SCOPE,
            )
        return _INSTANCE


def reset_named_reference_store() -> None:
    """重置单例，仅供测试 / 热重载使用。"""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None
