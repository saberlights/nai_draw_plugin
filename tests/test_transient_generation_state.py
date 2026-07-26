from pathlib import Path
import sys


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.transient_generation_state import (
    TransientGenerationState,
)


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_pending_generation_expires_instead_of_blocking_session_forever() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=10,
        idle_ttl_seconds=3_600,
        pending_ttl_seconds=120,
        clock=clock,
    )
    state.set_pending_image_generation("stream-1")

    clock.now = 1_121.0

    assert state.get_pending_image_generation_started_at("stream-1") is None


def test_capacity_evicts_least_recently_used_session() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=2,
        idle_ttl_seconds=3_600,
        pending_ttl_seconds=120,
        clock=clock,
    )
    state.set_last_action_image_sent_at("stream-1", 100.0)
    state.set_last_action_image_sent_at("stream-2", 200.0)
    assert state.get_last_action_image_sent_at("stream-1") == 100.0

    state.set_last_action_image_sent_at("stream-3", 300.0)

    assert state.get_last_action_image_sent_at("stream-1") == 100.0
    assert state.get_last_action_image_sent_at("stream-2") is None
    assert state.get_last_action_image_sent_at("stream-3") == 300.0


def test_context_ttl_and_idle_ttl_remove_stale_generation_history() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=10,
        idle_ttl_seconds=300,
        pending_ttl_seconds=120,
        clock=clock,
    )
    state.set_last_nai_context("stream-1", "1girl, smile", "画一个女孩", ttl=60)
    state.set_last_selfie_context(
        "stream-1",
        "1girl, selfie",
        "来张自拍",
        "indoor selfie",
        {"appearance": ["blue eyes"]},
        ttl=60,
    )

    clock.now = 1_061.0
    assert state.get_last_nai_context("stream-1", ttl=60) == (None, None)
    assert state.get_last_selfie_context("stream-1", ttl=60) == (None, None, None, {})

    state.set_last_action_image_sent_at("stream-2", 1_061.0)
    clock.now = 1_362.0
    assert state.get_last_action_image_sent_at("stream-2") is None


def test_zero_context_ttl_survives_idle_cleanup() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=10,
        idle_ttl_seconds=300,
        pending_ttl_seconds=120,
        clock=clock,
    )
    state.set_last_nai_context("stream-1", "1girl, smile", "画一个女孩")

    clock.now = 2_000.0

    assert state.get_last_nai_context("stream-1", ttl=0) == (
        "1girl, smile",
        "画一个女孩",
    )


def test_capacity_still_bounds_zero_ttl_contexts() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=2,
        idle_ttl_seconds=300,
        pending_ttl_seconds=120,
        clock=clock,
    )
    state.set_last_nai_context("stream-1", "prompt-1", ttl=0)
    state.set_last_nai_context("stream-2", "prompt-2", ttl=0)
    assert state.get_last_nai_context("stream-1", ttl=0) == ("prompt-1", None)

    state.set_last_nai_context("stream-3", "prompt-3", ttl=0)

    assert state.get_last_nai_context("stream-1", ttl=0) == ("prompt-1", None)
    assert state.get_last_nai_context("stream-2", ttl=0) == (None, None)
    assert state.get_last_nai_context("stream-3", ttl=0) == ("prompt-3", None)


def test_expired_pending_owner_cannot_release_new_generation() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=10,
        idle_ttl_seconds=3_600,
        pending_ttl_seconds=120,
        clock=clock,
    )
    old_owner = state.acquire_pending_image_generation("stream-1")

    clock.now = 1_121.0
    new_owner = state.acquire_pending_image_generation("stream-1")

    assert old_owner is not None
    assert new_owner is not None and new_owner != old_owner
    assert state.release_pending_image_generation("stream-1", old_owner) is False
    assert state.get_pending_image_generation_started_at("stream-1") == 1_121.0
    assert state.release_pending_image_generation("stream-1", new_owner) is True
    assert state.get_pending_image_generation_started_at("stream-1") is None


def test_capacity_never_evicts_active_pending_generation() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=2,
        idle_ttl_seconds=3_600,
        pending_ttl_seconds=120,
        clock=clock,
    )
    owner = state.acquire_pending_image_generation("pending-stream")
    state.set_last_action_image_sent_at("old-stream", 100.0)
    state.set_last_action_image_sent_at("new-stream", 200.0)

    assert owner is not None
    assert state.get_pending_image_generation_started_at("pending-stream") == 1_000.0
    assert state.get_last_action_image_sent_at("old-stream") is None
    assert state.get_last_action_image_sent_at("new-stream") == 200.0


def test_capacity_allows_active_pending_sessions_to_temporarily_exceed_limit() -> None:
    clock = _Clock(1_000.0)
    state = TransientGenerationState(
        max_entries=2,
        idle_ttl_seconds=3_600,
        pending_ttl_seconds=120,
        clock=clock,
    )

    owners = [
        state.acquire_pending_image_generation("stream-1"),
        state.acquire_pending_image_generation("stream-2"),
        state.acquire_pending_image_generation("stream-3"),
    ]

    assert all(owner is not None for owner in owners)
    assert state.get_pending_image_generation_started_at("stream-1") == 1_000.0
    assert state.get_pending_image_generation_started_at("stream-2") == 1_000.0
    assert state.get_pending_image_generation_started_at("stream-3") == 1_000.0
