# -*- coding: utf-8 -*-
"""VisualContinuityStore 的持久化与过期语义测试。"""

from pathlib import Path
import sys


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.visual_continuity import (  # noqa: E402
    StableVisualTags,
    VisualTagCard,
)
from plugins.nai_draw_plugin.core.services.visual_continuity_store import (  # noqa: E402
    VisualContinuityStore,
)


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _sample_stable() -> StableVisualTags:
    outfit_card = VisualTagCard(
        "pale_cyan_qipao_set",
        ("pale cyan silk qipao", "high side slit", "black leather shoes"),
    )
    environment_card = VisualTagCard(
        "film_set_corner",
        ("film set", "light stand", "black blackout cloth"),
    )
    return StableVisualTags(
        outfit=outfit_card.tags,
        environment=environment_card.tags,
        outfit_key=outfit_card.key,
        environment_key=environment_card.key,
        outfits=(outfit_card,),
        environments=(environment_card,),
    )


def test_state_survives_process_restart_via_disk_reload(tmp_path) -> None:
    storage = tmp_path / "visual_continuity.json"
    stable = _sample_stable()
    store = VisualContinuityStore(storage_path=storage)
    store.set("stream-1", stable)

    # 模拟重启：从同一文件重新加载
    reloaded = VisualContinuityStore(storage_path=storage)

    assert reloaded.get("stream-1") == stable
    assert reloaded.get("stream-2") is None


def test_ttl_expires_current_look_but_keeps_card_library(tmp_path) -> None:
    clock = _Clock(1_000.0)
    store = VisualContinuityStore(
        storage_path=tmp_path / "visual_continuity.json",
        clock=clock,
    )
    stable = _sample_stable()
    store.set("stream-1", stable)

    clock.now = 1_000.0 + 3_601.0
    expired = store.get("stream-1", ttl=3_600)

    assert expired is not None
    assert expired.outfit == ()
    assert expired.environment == ()
    assert expired.outfit_key == ""
    assert expired.outfits == stable.outfits
    assert expired.environments == stable.environments

    # ttl=0 表示当前装扮不过期
    assert store.get("stream-1", ttl=0) == stable


def test_ttl_expiry_with_empty_library_returns_none(tmp_path) -> None:
    clock = _Clock(1_000.0)
    store = VisualContinuityStore(
        storage_path=tmp_path / "visual_continuity.json",
        clock=clock,
    )
    store.set(
        "stream-1",
        StableVisualTags(outfit=("white dress",), outfit_key="white_dress"),
    )

    clock.now = 1_000.0 + 61.0

    assert store.get("stream-1", ttl=60) is None


def test_clear_removes_stream_state_from_disk(tmp_path) -> None:
    storage = tmp_path / "visual_continuity.json"
    store = VisualContinuityStore(storage_path=storage)
    store.set("stream-1", _sample_stable())

    store.clear("stream-1")

    assert store.get("stream-1") is None
    assert VisualContinuityStore(storage_path=storage).get("stream-1") is None


def test_corrupt_state_file_starts_empty_instead_of_crashing(tmp_path) -> None:
    storage = tmp_path / "visual_continuity.json"
    storage.write_text("{ not valid json", encoding="utf-8")

    store = VisualContinuityStore(storage_path=storage)

    assert store.get("stream-1") is None
    # 仍可正常写入并恢复持久化能力
    store.set("stream-1", _sample_stable())
    assert VisualContinuityStore(storage_path=storage).get("stream-1") is not None


def test_capacity_evicts_least_recently_updated_stream(tmp_path) -> None:
    clock = _Clock(1_000.0)
    store = VisualContinuityStore(
        storage_path=tmp_path / "visual_continuity.json",
        max_entries=2,
        clock=clock,
    )
    stable = _sample_stable()
    store.set("stream-1", stable)
    clock.now = 1_001.0
    store.set("stream-2", stable)
    clock.now = 1_002.0
    store.set("stream-3", stable)

    assert store.get("stream-1") is None
    assert store.get("stream-2") == stable
    assert store.get("stream-3") == stable
