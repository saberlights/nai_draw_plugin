"""NSFW 状态持久化的最小行为验证。

  * set 后立即 get 命中；
  * 同路径新实例（模拟重启）能读出之前 set 的值；
  * clear 后 get 回 None，且文件里不再有该 key。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.services.nsfw_state_store import NsfwStateStore


def test_set_get_roundtrip(tmp_path: Path) -> None:
    store = NsfwStateStore(storage_path=tmp_path / "nsfw_state.json")
    assert store.get("stream", "chatA") is None

    store.set("stream", "chatA", True)
    assert store.get("stream", "chatA") is True

    store.set("stream", "chatA", False)
    assert store.get("stream", "chatA") is False


def test_persists_across_instances(tmp_path: Path) -> None:
    """新实例 == 重启插件；同路径下应该读出之前写入的状态。"""
    storage_path = tmp_path / "nsfw_state.json"

    store1 = NsfwStateStore(storage_path=storage_path)
    store1.set("stream", "chatA", True)
    store1.set("stream", "chatB", False)

    store2 = NsfwStateStore(storage_path=storage_path)
    assert store2.get("stream", "chatA") is True
    assert store2.get("stream", "chatB") is False
    assert store2.get("stream", "chatC") is None


def test_clear_removes_entry(tmp_path: Path) -> None:
    storage_path = tmp_path / "nsfw_state.json"
    store = NsfwStateStore(storage_path=storage_path)
    store.set("stream", "chatA", True)
    store.clear("stream", "chatA")

    assert store.get("stream", "chatA") is None
    raw = json.loads(storage_path.read_text(encoding="utf-8"))
    assert "stream:chatA" not in raw


def test_corrupt_file_falls_back_to_empty(tmp_path: Path) -> None:
    """坏数据不能炸初始化；store 退化为空，不影响后续 set。"""
    storage_path = tmp_path / "nsfw_state.json"
    storage_path.write_text("not a json", encoding="utf-8")

    store = NsfwStateStore(storage_path=storage_path)
    assert store.get("stream", "chatA") is None

    store.set("stream", "chatA", True)
    assert store.get("stream", "chatA") is True
