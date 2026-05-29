# -*- coding: utf-8 -*-
"""NamedReferenceStore 的最小校验。

测试覆盖：
- 名字 / 图字节 / 容量的拒绝路径
- 保存 → 列出 → 读取 → 删除的往返
- 选定（set / get / clear）的跨重启持久化（用 selection.json）
- 图删除时同步清掉指向它的选定
- 不同 owner（群 / 用户）隔离、不同 scope 隔离
- 旧版 selection.json（扁平 user 结构 + 单 string value）的向后兼容
"""

from __future__ import annotations

import struct
import sys
import types
from pathlib import Path

import pytest


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def __getattr__(self, _):
        return lambda *a, **k: None


dummy_logger_module.get_logger = lambda _=None: _DummyLogger()
sys.modules["src.common.logger"] = dummy_logger_module


from plugins.nai_draw_plugin.core.services.named_reference_store import (
    CapacityExceededError,
    InvalidImageError,
    InvalidNameError,
    NamedReferenceStore,
    OWNER_GROUP,
    OWNER_USER,
    SCOPE_REF,
    SCOPE_VIBE,
)


# ── 工具：构造最小可解析的 PNG / JPEG / WebP 字节 ────────────────────────


def _make_png_bytes(width: int = 832, height: int = 1216) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_length = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + ihdr_length + ihdr_type + ihdr_data + b"\x00" * 4


def _make_jpeg_bytes(width: int = 832, height: int = 1216) -> bytes:
    soi = b"\xff\xd8\xff"
    app0 = b"\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0_marker = b"\xff\xc0"
    sof0_segment = struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", height, width) + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return soi + app0 + sof0_marker + sof0_segment + b"\xff\xd9"


def _make_webp_bytes(width: int = 832, height: int = 1216) -> bytes:
    """构造 VP8L 形式的最小 WebP；read_image_dimensions 至少要 30 字节。"""
    riff = b"RIFF"
    file_size = struct.pack("<I", 30)
    webp = b"WEBP"
    vp8l = b"VP8L"
    chunk_size = struct.pack("<I", 10)
    signature = b"\x2f"
    # packed: 14 位 (width - 1) + 14 位 (height - 1)，剩余位塞 0
    packed = ((height - 1) & 0x3FFF) << 14 | ((width - 1) & 0x3FFF)
    return (
        riff
        + file_size
        + webp
        + vp8l
        + chunk_size
        + signature
        + struct.pack("<I", packed)
        + b"\x00" * 5  # 凑到 30+ 字节，避开 read_image_dimensions 的 len < 30 早退
    )


def _make_store(tmp_path: Path, *, max_per_scope: int = 20) -> NamedReferenceStore:
    return NamedReferenceStore(tmp_path / "store", max_per_scope=max_per_scope)


# 默认 owner 简写，写测试时少打字
def _user(uid: str = "u1") -> dict:
    return {"owner_kind": OWNER_USER, "owner_id": uid}


def _group(gid: str = "g1") -> dict:
    return {"owner_kind": OWNER_GROUP, "owner_id": gid}


# ── 名字校验 ─────────────────────────────────────────────────────────────


def test_invalid_name_rejects_empty_too_long_and_bad_chars(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for bad in ["", " ", "a" * 33, "name with space", "../escape", "with/slash", "with.dot", "with@at"]:
        with pytest.raises(InvalidNameError):
            store.save(scope=SCOPE_VIBE, **_user(), name=bad, image_bytes=png)


def test_valid_names_include_cjk_alnum_underscore(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for ok in ["角色A", "char_b", "GIRL_01", "中文_数字123", "_", "a" * 32]:
        ref = store.save(scope=SCOPE_VIBE, **_user(), name=ok, image_bytes=png)
        assert ref.name == ok


# ── 图字节校验 ───────────────────────────────────────────────────────────


def test_save_rejects_non_image_bytes(tmp_path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(InvalidImageError):
        store.save(scope=SCOPE_VIBE, **_user(), name="x", image_bytes=b"not an image")


def test_save_supports_png_jpeg_webp(tmp_path) -> None:
    store = _make_store(tmp_path)
    cases = [
        ("png", _make_png_bytes(832, 1216), "png"),
        ("jpeg", _make_jpeg_bytes(640, 480), "jpeg"),
        ("webp", _make_webp_bytes(512, 512), "webp"),
    ]
    for name, image_bytes, expected_format in cases:
        ref = store.save(scope=SCOPE_VIBE, **_user(), name=name, image_bytes=image_bytes)
        assert ref.image_format == expected_format
        # 物理文件落到磁盘并能直接读
        assert ref.path.exists()
        assert ref.path.suffix == f".{expected_format}"


def test_save_returns_dimensions_and_byte_size(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes(832, 1216)
    ref = store.save(scope=SCOPE_VIBE, **_user(), name="big", image_bytes=png)
    assert ref.width == 832
    assert ref.height == 1216
    assert ref.byte_size == len(png)


# ── owner 校验 ───────────────────────────────────────────────────────────


def test_invalid_owner_kind_rejected(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    with pytest.raises(ValueError):
        store.save(scope=SCOPE_VIBE, owner_kind="bogus", owner_id="x", name="a", image_bytes=png)


def test_empty_owner_id_rejected(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    with pytest.raises(ValueError):
        store.save(scope=SCOPE_VIBE, owner_kind=OWNER_GROUP, owner_id="", name="a", image_bytes=png)


# ── 保存 / 读取 / 列出 / 删除往返 ────────────────────────────────────────


def test_save_then_get_round_trip(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    store.save(scope=SCOPE_VIBE, **_user(), name="角色A", image_bytes=png)
    assert store.get(scope=SCOPE_VIBE, **_user(), name="角色A") == png


def test_get_returns_none_when_missing(tmp_path) -> None:
    store = _make_store(tmp_path)
    assert store.get(scope=SCOPE_VIBE, **_user(), name="nope") is None


def test_list_returns_sorted_by_name(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for n in ["c", "a", "b"]:
        store.save(scope=SCOPE_VIBE, **_user(), name=n, image_bytes=png)
    listed = store.list(scope=SCOPE_VIBE, **_user())
    assert [r.name for r in listed] == ["a", "b", "c"]


def test_save_overwrites_existing_name_without_counting_capacity(tmp_path) -> None:
    """覆盖同名旧图不算新增，不计入容量。"""
    store = _make_store(tmp_path, max_per_scope=2)
    png_a = _make_png_bytes(832, 1216)
    png_b = _make_png_bytes(512, 512)
    store.save(scope=SCOPE_VIBE, **_user(), name="角色A", image_bytes=png_a)
    store.save(scope=SCOPE_VIBE, **_user(), name="角色B", image_bytes=png_a)
    # 已满（2/2），但覆盖同名仍允许
    ref = store.save(scope=SCOPE_VIBE, **_user(), name="角色A", image_bytes=png_b)
    assert ref.width == 512
    assert store.get(scope=SCOPE_VIBE, **_user(), name="角色A") == png_b
    # 列表里仍是 2 条
    assert len(store.list(scope=SCOPE_VIBE, **_user())) == 2


def test_save_changing_format_replaces_old_extension(tmp_path) -> None:
    """同名图换格式（png → webp）：旧扩展名文件应被清掉，避免同名两份。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    webp = _make_webp_bytes()
    store.save(scope=SCOPE_VIBE, **_user(), name="同图", image_bytes=png)
    store.save(scope=SCOPE_VIBE, **_user(), name="同图", image_bytes=webp)
    listed = store.list(scope=SCOPE_VIBE, **_user())
    assert [r.name for r in listed] == ["同图"]
    assert listed[0].image_format == "webp"


def test_delete_removes_file_and_returns_true(tmp_path) -> None:
    store = _make_store(tmp_path)
    store.save(scope=SCOPE_VIBE, **_user(), name="x", image_bytes=_make_png_bytes())
    assert store.delete(scope=SCOPE_VIBE, **_user(), name="x") is True
    assert store.get(scope=SCOPE_VIBE, **_user(), name="x") is None


def test_delete_returns_false_when_missing(tmp_path) -> None:
    store = _make_store(tmp_path)
    assert store.delete(scope=SCOPE_VIBE, **_user(), name="nope") is False


# ── 容量上限 ────────────────────────────────────────────────────────────


def test_capacity_exceeded_when_full_and_adding_new_name(tmp_path) -> None:
    store = _make_store(tmp_path, max_per_scope=2)
    png = _make_png_bytes()
    store.save(scope=SCOPE_VIBE, **_user(), name="a", image_bytes=png)
    store.save(scope=SCOPE_VIBE, **_user(), name="b", image_bytes=png)
    with pytest.raises(CapacityExceededError):
        store.save(scope=SCOPE_VIBE, **_user(), name="c", image_bytes=png)


# ── 隔离：不同 owner / 不同 scope ────────────────────────────────────────


def test_different_user_owners_are_isolated(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    store.save(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id="alice", name="x", image_bytes=png)
    assert store.get(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id="alice", name="x") == png
    assert store.get(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id="bob", name="x") is None
    assert store.list(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id="bob") == []


def test_group_and_user_owners_with_same_id_are_isolated(tmp_path) -> None:
    """user_id == group_id（极端情况）也要互不打架，分别落到 users/ 与 groups/。"""
    store = _make_store(tmp_path)
    png_a = _make_png_bytes(832, 1216)
    png_b = _make_png_bytes(512, 512)
    same = "12345"
    store.save(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id=same, name="x", image_bytes=png_a)
    store.save(scope=SCOPE_VIBE, owner_kind=OWNER_GROUP, owner_id=same, name="x", image_bytes=png_b)
    assert store.get(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id=same, name="x") == png_a
    assert store.get(scope=SCOPE_VIBE, owner_kind=OWNER_GROUP, owner_id=same, name="x") == png_b


def test_group_owner_shared_across_users_in_same_group(tmp_path) -> None:
    """关键修复：群图库按 group_id 共享，群内任意成员都能读同一份。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    # 成员 alice 在群 g1 里存图
    store.save(scope=SCOPE_VIBE, owner_kind=OWNER_GROUP, owner_id="g1", name="角色A", image_bytes=png)
    # 成员 bob 用同一群 owner 读，能读到
    assert store.get(scope=SCOPE_VIBE, owner_kind=OWNER_GROUP, owner_id="g1", name="角色A") == png
    # 换一个群就读不到
    assert store.get(scope=SCOPE_VIBE, owner_kind=OWNER_GROUP, owner_id="g2", name="角色A") is None


def test_vibe_and_ref_scopes_are_isolated(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    store.save(scope=SCOPE_VIBE, **_user(), name="x", image_bytes=png)
    assert store.get(scope=SCOPE_VIBE, **_user(), name="x") == png
    assert store.get(scope=SCOPE_REF, **_user(), name="x") is None


def test_owner_id_with_special_chars_safe_on_filesystem(tmp_path) -> None:
    """owner_id 含冒号 / @ / 空格也能落盘（内部用 sha256 哈希作目录名）。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    weird = "qq:123456789@platform with space"
    store.save(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id=weird, name="x", image_bytes=png)
    assert store.get(scope=SCOPE_VIBE, owner_kind=OWNER_USER, owner_id=weird, name="x") == png


# ── 选定（粘性，list 形态） ──────────────────────────────────────────────


def test_set_and_get_selection_round_trip(tmp_path) -> None:
    store = _make_store(tmp_path)
    store.save(scope=SCOPE_VIBE, **_user(), name="角色A", image_bytes=_make_png_bytes())
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["角色A"])
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == ["角色A"]


def test_get_selection_returns_empty_list_when_unset(tmp_path) -> None:
    store = _make_store(tmp_path)
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == []


def test_set_selection_rejects_nonexistent_name(tmp_path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(KeyError):
        store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["ghost"])


def test_set_selection_rejects_empty_names(tmp_path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(ValueError):
        store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=[])


def test_selection_isolated_by_stream_owner_and_scope(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for s in [SCOPE_VIBE, SCOPE_REF]:
        store.save(scope=s, **_user(), name="角色A", image_bytes=png)
        store.save(scope=s, **_user(), name="角色B", image_bytes=png)
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["角色A"])
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s2", names=["角色B"])
    store.set_selection(scope=SCOPE_REF, **_user(), stream_id="s1", names=["角色B"])
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == ["角色A"]
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s2") == ["角色B"]
    assert store.get_selection(scope=SCOPE_REF, **_user(), stream_id="s1") == ["角色B"]
    # 不同 owner 完全独立
    assert store.get_selection(scope=SCOPE_VIBE, **_user("u2"), stream_id="s1") == []


def test_group_selection_shared_within_group(tmp_path) -> None:
    """群里设的选定，群里任何人重新查（同 stream_id）都应能读到，不再按人切。"""
    store = _make_store(tmp_path)
    store.save(scope=SCOPE_VIBE, **_group("g1"), name="角色A", image_bytes=_make_png_bytes())
    store.set_selection(scope=SCOPE_VIBE, **_group("g1"), stream_id="s1", names=["角色A"])
    # 同群同 stream 再查
    assert store.get_selection(scope=SCOPE_VIBE, **_group("g1"), stream_id="s1") == ["角色A"]
    # 不同群读不到
    assert store.get_selection(scope=SCOPE_VIBE, **_group("g2"), stream_id="s1") == []


def test_selection_persists_across_store_instances(tmp_path) -> None:
    """重启场景：换一个 store 实例指向同一 root，selection 仍能读出来。"""
    root = tmp_path / "store"
    s1 = NamedReferenceStore(root)
    s1.save(scope=SCOPE_VIBE, **_user(), name="角色A", image_bytes=_make_png_bytes())
    s1.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["角色A"])
    s2 = NamedReferenceStore(root)
    assert s2.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == ["角色A"]


def test_clear_selection_removes_entry(tmp_path) -> None:
    store = _make_store(tmp_path)
    store.save(scope=SCOPE_VIBE, **_user(), name="x", image_bytes=_make_png_bytes())
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["x"])
    store.clear_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1")
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == []


def test_clear_selection_when_unset_is_safe(tmp_path) -> None:
    store = _make_store(tmp_path)
    # 没设过也不应抛
    store.clear_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1")


def test_delete_image_clears_selection_pointing_to_it(tmp_path) -> None:
    """避免选定指向已删除的图，导致后续命令找不到。"""
    store = _make_store(tmp_path)
    store.save(scope=SCOPE_VIBE, **_user(), name="角色A", image_bytes=_make_png_bytes())
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["角色A"])
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s2", names=["角色A"])
    store.delete(scope=SCOPE_VIBE, **_user(), name="角色A")
    # 两个 stream 都被同步清掉（list 剩空，整条 stream key 被移除）
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == []
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s2") == []


# ── 多图选定（vibe 最多 4，ref 最多 1） ─────────────────────────────────


def test_vibe_set_selection_accepts_up_to_four_names(tmp_path) -> None:
    """§20.3 controlnet.images 最多 4 张：set_selection 应允许 1~4。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for n in ["a", "b", "c", "d"]:
        store.save(scope=SCOPE_VIBE, **_user(), name=n, image_bytes=png)
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["a", "b", "c", "d"])
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == ["a", "b", "c", "d"]


def test_vibe_set_selection_rejects_more_than_four(tmp_path) -> None:
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for n in ["a", "b", "c", "d", "e"]:
        store.save(scope=SCOPE_VIBE, **_user(), name=n, image_bytes=png)
    with pytest.raises(ValueError):
        store.set_selection(
            scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["a", "b", "c", "d", "e"]
        )


def test_ref_set_selection_rejects_more_than_one(tmp_path) -> None:
    """§20.4 character_references 最多 1 张：ref 选定多于 1 应报错。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    store.save(scope=SCOPE_REF, **_user(), name="a", image_bytes=png)
    store.save(scope=SCOPE_REF, **_user(), name="b", image_bytes=png)
    with pytest.raises(ValueError):
        store.set_selection(scope=SCOPE_REF, **_user(), stream_id="s1", names=["a", "b"])


def test_set_selection_is_atomic_on_partial_invalid_names(tmp_path) -> None:
    """names 里有一个不存在 → 整批拒绝，旧选定保留不变。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    store.save(scope=SCOPE_VIBE, **_user(), name="a", image_bytes=png)
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["a"])
    with pytest.raises(KeyError):
        store.set_selection(
            scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["a", "nope"]
        )
    # 旧选定仍在
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == ["a"]


def test_delete_one_of_multi_selection_keeps_remainder(tmp_path) -> None:
    """多图选定下删某张图：该张从 list 剔除，其它图仍在选定里。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for n in ["a", "b", "c"]:
        store.save(scope=SCOPE_VIBE, **_user(), name=n, image_bytes=png)
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["a", "b", "c"])
    store.delete(scope=SCOPE_VIBE, **_user(), name="b")
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == ["a", "c"]


def test_legacy_string_selection_format_upgraded_on_read(tmp_path) -> None:
    """selection.json 里旧版单 string 形态应被读取层自动当成 [string] 升级。"""
    import json

    root = tmp_path / "store"
    root.mkdir(parents=True)
    store = NamedReferenceStore(root)
    store.save(scope=SCOPE_VIBE, **_user(), name="角色A", image_bytes=_make_png_bytes())
    # 手动写一份"旧扁平结构 + 旧 string 形态"的 selection.json
    owner_dir = store._owner_dir_name("u1")
    legacy = {"vibe": {owner_dir: {"s1": "角色A"}}}
    (root / "selection.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    # 重新拉一个实例读：因为没有 @user 桶，应回退到旧扁平结构（仅 OWNER_USER 才回退）
    store2 = NamedReferenceStore(root)
    assert store2.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == ["角色A"]


def test_legacy_flat_selection_not_visible_to_group_owner(tmp_path) -> None:
    """旧扁平结构的归属本就是 user，不应被新 group owner 误读。"""
    import json

    root = tmp_path / "store"
    root.mkdir(parents=True)
    store = NamedReferenceStore(root)
    # 在 group 桶里存一张，让 owner_dir hash 落地（避免 list 时为空目录）
    store.save(scope=SCOPE_VIBE, **_group("12345"), name="x", image_bytes=_make_png_bytes())
    # 手动写一份旧扁平结构，hash 与 "12345" group 相同（因为 _owner_dir_name 仅看字符串）
    owner_dir = store._owner_dir_name("12345")
    legacy = {"vibe": {owner_dir: {"s1": ["x"]}}}
    (root / "selection.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    store2 = NamedReferenceStore(root)
    # group 视角看不到旧 user 扁平数据
    assert store2.get_selection(scope=SCOPE_VIBE, **_group("12345"), stream_id="s1") == []
    # user 视角看得到（兼容旧数据）
    assert store2.get_selection(scope=SCOPE_VIBE, **_user("12345"), stream_id="s1") == ["x"]


# ── 一键清空（clear_all） ────────────────────────────────────────────────


def test_clear_all_removes_every_image_and_returns_count(tmp_path) -> None:
    """clear_all 应删该 (scope, owner) 下所有图，返回删除张数。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for name in ["a", "b", "c"]:
        store.save(scope=SCOPE_VIBE, **_user(), name=name, image_bytes=png)
    assert store.clear_all(scope=SCOPE_VIBE, **_user()) == 3
    assert store.list(scope=SCOPE_VIBE, **_user()) == []


def test_clear_all_returns_zero_when_already_empty(tmp_path) -> None:
    """空图库 / 目录不存在时 clear_all 应安全返回 0。"""
    store = _make_store(tmp_path)
    assert store.clear_all(scope=SCOPE_VIBE, **_user("ghost")) == 0


def test_clear_all_resets_selections_across_all_streams(tmp_path) -> None:
    """清空时同一 (scope, owner) 在所有 stream 上的选定都应被同步清掉。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    for name in ["a", "b"]:
        store.save(scope=SCOPE_VIBE, **_user(), name=name, image_bytes=png)
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1", names=["a", "b"])
    store.set_selection(scope=SCOPE_VIBE, **_user(), stream_id="s2", names=["a"])
    store.clear_all(scope=SCOPE_VIBE, **_user())
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s1") == []
    assert store.get_selection(scope=SCOPE_VIBE, **_user(), stream_id="s2") == []


def test_clear_all_does_not_affect_other_owner_or_scope(tmp_path) -> None:
    """clear_all 隔离边界：只动当前 (scope, owner)，不动其它 owner 或其它 scope。"""
    store = _make_store(tmp_path)
    png = _make_png_bytes()
    store.save(scope=SCOPE_VIBE, **_user("alice"), name="x", image_bytes=png)
    store.save(scope=SCOPE_VIBE, **_user("bob"), name="y", image_bytes=png)
    store.save(scope=SCOPE_REF, **_user("alice"), name="z", image_bytes=png)
    store.save(scope=SCOPE_VIBE, **_group("g1"), name="g", image_bytes=png)
    store.set_selection(scope=SCOPE_VIBE, **_user("bob"), stream_id="s1", names=["y"])

    assert store.clear_all(scope=SCOPE_VIBE, **_user("alice")) == 1
    # 仅 alice 的 vibe 图被清；bob 的 vibe / alice 的 ref / 群 g1 的 vibe 不受影响
    assert store.get(scope=SCOPE_VIBE, **_user("alice"), name="x") is None
    assert store.get(scope=SCOPE_VIBE, **_user("bob"), name="y") is not None
    assert store.get(scope=SCOPE_REF, **_user("alice"), name="z") is not None
    assert store.get(scope=SCOPE_VIBE, **_group("g1"), name="g") is not None
    assert store.get_selection(scope=SCOPE_VIBE, **_user("bob"), stream_id="s1") == ["y"]
