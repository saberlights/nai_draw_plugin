"""生成准入策略的公开行为测试。"""

from pathlib import Path
import random
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
    auto_draw_interval: int = 180,
) -> dict:
    return {
        "action_guard": {
            "enabled": True,
            "explicit_request_min_interval_seconds": explicit_interval,
            "proactive_min_interval_seconds": proactive_interval,
            "weak_negative_ttl_seconds": 60,
        },
        "auto_draw_on_reply": {
            "enabled": True,
            "min_interval_seconds": auto_draw_interval,
            "score_threshold": 0.6,
        },
    }


def _build_policy(
    *,
    clock: _Clock | None = None,
    max_reply_sessions: int = 2_000,
    reply_claims_per_session: int = 16,
    reply_claim_ttl_seconds: float = 86_400,
) -> tuple[GenerationAdmissionPolicy, TransientGenerationState, _Clock]:
    effective_clock = clock or _Clock()
    state = TransientGenerationState(clock=effective_clock)
    policy = GenerationAdmissionPolicy(
        state=state,
        logger=_Logger(),
        clock=effective_clock,
        max_reply_sessions=max_reply_sessions,
        reply_claims_per_session=reply_claims_per_session,
        reply_claim_ttl_seconds=reply_claim_ttl_seconds,
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
    policy.record_success(stream_id="stream-1", category="explicit")

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
    policy.record_success(stream_id="stream-1", category="proactive")
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


def test_auto_draw_uses_latest_action_or_auto_draw_success() -> None:
    policy, state, clock = _build_policy()
    policy.record_success(stream_id="stream-1", category="explicit", sent_at=900)
    policy.record_success(stream_id="stream-1", category="auto_draw", sent_at=950)
    clock.now = 1_100

    decision = policy.evaluate_auto_draw(
        stream_id="stream-1",
        config=_config(),
    )

    assert state.get_last_action_image_sent_at("stream-1") == 900
    assert state.get_last_auto_draw_sent_at("stream-1") == 950
    assert decision.should_generate is False
    assert decision.category == "auto_draw"


def test_auto_draw_respects_negative_user_intent() -> None:
    policy, _state, _clock = _build_policy()

    strong = policy.evaluate_auto_draw(
        stream_id="stream-strong",
        config=_config(),
        user_text="别画了",
        user_text_age_seconds=3_600,
    )
    weak = policy.evaluate_auto_draw(
        stream_id="stream-weak",
        config=_config(),
        user_text="用文字给我讲",
        user_text_age_seconds=10,
    )

    assert strong.should_generate is False
    assert strong.category == "blocked"
    assert weak.should_generate is False
    assert "文字" in weak.detail


def test_auto_draw_success_does_not_throttle_explicit_action() -> None:
    policy, _state, _clock = _build_policy()
    policy.record_success(stream_id="stream-1", category="auto_draw")

    decision = policy.evaluate_action(
        stream_id="stream-1",
        config=_config(),
        user_text="再来一张自拍",
    )

    assert decision.should_generate is True


def test_zero_intervals_allow_immediate_generation() -> None:
    policy, _state, _clock = _build_policy()
    config = _config(explicit_interval=0, proactive_interval=0, auto_draw_interval=0)
    policy.record_success(stream_id="stream-1", category="explicit")
    policy.record_success(stream_id="stream-1", category="auto_draw")

    assert policy.evaluate_action(
        stream_id="stream-1", config=config, user_text="再来一张自拍"
    ).should_generate is True
    assert policy.evaluate_action(
        stream_id="stream-1", config=config, user_text="今天天气真不错"
    ).should_generate is True
    assert policy.evaluate_auto_draw(
        stream_id="stream-1", config=config
    ).should_generate is True


def test_reply_candidate_returns_description_and_rejects_duplicate() -> None:
    policy, _state, _clock = _build_policy()
    reply = "我刚洗完澡靠在窗边发呆，有点累"

    first = policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=reply, config=_config()
    )
    duplicate = policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text="".join([reply]), config=_config()
    )

    assert first.should_generate is True
    assert first.category == "auto_draw"
    assert first.signal_source == "reply_text"
    assert first.seed_description
    assert "自拍" in first.seed_description
    assert duplicate.should_generate is False
    assert "已提交" in duplicate.detail


def test_reply_claims_are_independent_between_sessions() -> None:
    policy, _state, _clock = _build_policy()
    reply = "我刚洗完澡靠在窗边发呆，有点累"

    first = policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=reply, config=_config()
    )
    second = policy.evaluate_reply_candidate(
        stream_id="stream-2", reply_text=reply, config=_config()
    )

    assert first.should_generate is True
    assert second.should_generate is True


def test_reply_deduplication_is_stable_for_seeded_random_inputs() -> None:
    policy, _state, _clock = _build_policy()
    random_source = random.Random(20260726)

    for index in range(8):
        token = "".join(random_source.choices("abcdef0123456789", k=12))
        stream_id = f"stream-{random_source.randrange(1_000_000)}"
        reply = f"我刚洗完澡靠在窗边发呆，有点累 {token}"

        first = policy.evaluate_reply_candidate(
            stream_id=stream_id, reply_text=reply, config=_config()
        )
        duplicate = policy.evaluate_reply_candidate(
            stream_id=stream_id, reply_text=f"{reply}", config=_config()
        )

        assert first.should_generate is True, index
        assert duplicate.should_generate is False, index


def test_reply_claim_expires_after_ttl() -> None:
    clock = _Clock()
    policy, _state, _clock = _build_policy(
        clock=clock,
        reply_claim_ttl_seconds=60,
    )
    reply = "我刚洗完澡靠在窗边发呆，有点累"
    assert policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=reply, config=_config()
    ).should_generate is True

    clock.advance(61)

    assert policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=reply, config=_config()
    ).should_generate is True


def test_reply_claim_capacity_evicts_oldest_claim_per_session() -> None:
    policy, _state, _clock = _build_policy(reply_claims_per_session=2)
    replies = [
        "我刚洗完澡靠在窗边发呆，有点累",
        "我刚起床坐在床上发呆，有点困",
        "我刚回家坐在沙发上休息，有点累",
    ]
    for reply in replies:
        assert policy.evaluate_reply_candidate(
            stream_id="stream-1", reply_text=reply, config=_config()
        ).should_generate is True

    first_again = policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=replies[0], config=_config()
    )

    assert first_again.should_generate is True


def test_reply_session_capacity_evicts_least_recent_session() -> None:
    policy, _state, _clock = _build_policy(max_reply_sessions=2)
    reply = "我刚洗完澡靠在窗边发呆，有点累"
    for stream_id in ("stream-1", "stream-2", "stream-3"):
        assert policy.evaluate_reply_candidate(
            stream_id=stream_id, reply_text=reply, config=_config()
        ).should_generate is True

    first_session_again = policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=reply, config=_config()
    )

    assert first_session_again.should_generate is True


def test_reply_candidate_respects_enable_switch_and_threshold() -> None:
    policy, _state, _clock = _build_policy()
    disabled_config = _config()
    disabled_config["auto_draw_on_reply"]["enabled"] = False
    high_threshold_config = _config()
    high_threshold_config["auto_draw_on_reply"]["score_threshold"] = 1.1
    reply = "我刚洗完澡靠在窗边发呆，有点累"

    disabled = policy.evaluate_reply_candidate(
        stream_id="stream-disabled", reply_text=reply, config=disabled_config
    )
    below_threshold = policy.evaluate_reply_candidate(
        stream_id="stream-threshold", reply_text=reply, config=high_threshold_config
    )

    assert disabled.should_generate is False
    assert below_threshold.should_generate is False


def test_clear_releases_all_reply_claims() -> None:
    policy, _state, _clock = _build_policy()
    reply = "我刚洗完澡靠在窗边发呆，有点累"
    assert policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=reply, config=_config()
    ).should_generate is True

    policy.clear()

    assert policy.evaluate_reply_candidate(
        stream_id="stream-1", reply_text=reply, config=_config()
    ).should_generate is True
