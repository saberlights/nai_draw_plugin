"""生成准入策略的公开行为测试。"""

from pathlib import Path
import sys


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.generation_admission_policy import (
    AdmissionDecision,
    GenerationAdmissionPolicy,
)
from plugins.nai_draw_plugin.core.services.transient_generation_state import (
    TransientGenerationState,
)


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Logger:
    def debug(self, *args, **kwargs) -> None:
        return None


def _config(
    *,
    explicit_interval: int = 45,
    proactive_interval: int = 240,
) -> dict:
    return {
        "action_guard": {
            "enabled": True,
            "explicit_request_min_interval_seconds": explicit_interval,
            "proactive_min_interval_seconds": proactive_interval,
            "weak_negative_ttl_seconds": 60,
        },
    }


def _build_policy(
    *,
    clock: _Clock | None = None,
) -> tuple[GenerationAdmissionPolicy, TransientGenerationState, _Clock]:
    effective_clock = clock or _Clock()
    state = TransientGenerationState(clock=effective_clock)
    policy = GenerationAdmissionPolicy(
        state=state,
        logger=_Logger(),
        clock=effective_clock,
    )
    return policy, state, effective_clock


def test_action_explicit_user_request_returns_typed_decision() -> None:
    policy, _state, _clock = _build_policy()

    decision = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="再来一张自拍",
        user_text_age_seconds=3,
        reasoning="此刻适合配图",
    )

    assert isinstance(decision, AdmissionDecision)
    assert decision.should_generate is True
    assert decision.category == "explicit"
    assert decision.explicit_request is True
    assert decision.signal_source == "user_text"
    assert decision.signal_text == "再来一张自拍"


def test_action_neutral_user_text_stays_proactive_despite_reasoning() -> None:
    policy, _state, _clock = _build_policy()

    decision = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="今天天气真不错",
        user_text_age_seconds=3,
        reasoning="用户要求看图",
    )

    assert decision.should_generate is True
    assert decision.category == "proactive"
    assert decision.explicit_request is False
    assert decision.signal_source == "user_text"


def test_action_empty_user_text_uses_reasoning_fallback() -> None:
    policy, _state, _clock = _build_policy()

    explicit = policy.evaluate_action(
        stream_id="stream-explicit",
        config=_config(),
        reasoning="用户要求看你的穿搭",
    )
    proactive = policy.evaluate_action(
        stream_id="stream-proactive",
        config=_config(),
        reasoning="此刻配图比文字更自然",
    )

    assert explicit.category == "explicit"
    assert explicit.explicit_request is True
    assert explicit.signal_source == "reasoning"
    assert proactive.category == "proactive"
    assert proactive.explicit_request is False


def test_strong_negative_blocks_even_when_stale() -> None:
    policy, _state, _clock = _build_policy()

    decision = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="别画了",
        user_text_age_seconds=3_600,
    )

    assert decision.should_generate is False
    assert decision.category == "blocked"
    assert decision.signal_text == "别画了"


def test_fresh_or_unknown_weak_negative_blocks_conservatively() -> None:
    policy, _state, _clock = _build_policy()

    fresh = policy.evaluate_action(
        stream_id="stream-fresh",
        config=_config(),
        user_text="用文字给我讲",
        user_text_age_seconds=10,
    )
    unknown = policy.evaluate_action(
        stream_id="stream-unknown",
        config=_config(),
        user_text="文字就行",
        user_text_age_seconds=None,
    )

    assert fresh.should_generate is False
    assert "文字" in fresh.detail
    assert unknown.should_generate is False
    assert unknown.category == "blocked"


def test_stale_weak_negative_expires_to_proactive() -> None:
    policy, _state, _clock = _build_policy()

    decision = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="用文字给我讲",
        user_text_age_seconds=120,
    )

    assert decision.should_generate is True
    assert decision.category == "proactive"


def test_action_success_starts_explicit_and_proactive_cooldowns() -> None:
    policy, state, clock = _build_policy()
    policy.record_success(stream_id="stream-1")

    explicit = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="再来一张自拍",
    )
    proactive = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="今天天气真不错",
    )

    assert state.get_last_action_image_sent_at("stream-1") == clock.now
    assert explicit.should_generate is False
    assert "节流" in explicit.detail
    assert proactive.should_generate is False


def test_action_cooldowns_use_their_own_intervals() -> None:
    policy, _state, clock = _build_policy()
    policy.record_success(stream_id="stream-1")
    clock.advance(46)

    explicit = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="再来一张自拍",
    )
    proactive = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="今天天气真不错",
    )

    assert explicit.should_generate is True
    assert proactive.should_generate is False


def test_zero_intervals_allow_immediate_generation() -> None:
    policy, _state, _clock = _build_policy()
    config = _config(explicit_interval=0, proactive_interval=0)
    policy.record_success(stream_id="stream-1")

    assert policy.evaluate_action(
        stream_id="stream-1", config=config, user_text="再来一张自拍"
    ).should_generate is True
    assert policy.evaluate_action(
        stream_id="stream-1", config=config, user_text="今天天气真不错"
    ).should_generate is True
