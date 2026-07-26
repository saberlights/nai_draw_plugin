"""集中图片生成准入、reply 去重、冷却判定与成功记账。"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from ..rules.reply_auto_draw import (
    compose_description_from_reply,
    score_reply_for_auto_draw,
)
from ..rules.selfie_rules import (
    detect_explicit_image_request,
    detect_negative_image_intent_strength,
)


_THROTTLED_DETAIL = (
    "图片节流中，本轮跳过出图、直接用文字回复推进；"
    "插件会自行解除冷却，请不要使用 wait 工具"
)

_REASONING_EXPLICIT_HINTS: tuple[str, ...] = (
    "用户要求", "用户想看", "用户想要", "用户希望", "用户让",
    "对方要求", "对方想看", "对方想要", "对方希望", "对方让",
    "他要求", "他想看", "她要求", "她想看",
    "明确要求", "明确想看", "明确请求",
    "要我画", "让我画", "叫我画", "要我发", "让我发",
    "追图", "继续画", "再画一张", "再来一张",
)


@dataclass(frozen=True)
class AdmissionDecision:
    """一次生成准入判断；调用方无需了解判定规则与状态布局。"""

    should_generate: bool
    category: str
    detail: str
    explicit_request: bool = False
    signal_source: str = ""
    signal_text: str = ""
    seed_description: str = ""


class GenerationAdmissionPolicy:
    """统一 Action 与 reply 自动跟图的准入状态机。"""

    def __init__(
        self,
        *,
        state: Any,
        logger: Any,
        clock: Callable[[], float] = time.time,
        max_reply_sessions: int = 2_000,
        reply_claims_per_session: int = 16,
        reply_claim_ttl_seconds: float = 86_400.0,
    ) -> None:
        self._state = state
        self._logger = logger
        self._clock = clock
        self._max_reply_sessions = max(1, int(max_reply_sessions))
        self._reply_claims_per_session = max(1, int(reply_claims_per_session))
        self._reply_claim_ttl_seconds = max(0.0, float(reply_claim_ttl_seconds))
        self._reply_claims: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()

    def evaluate_action(
        self,
        *,
        stream_id: str,
        config: dict[str, Any],
        user_text: str = "",
        user_text_age_seconds: float | None = None,
        reasoning: str = "",
    ) -> AdmissionDecision:
        """根据用户原话、Planner reasoning 与冷却状态评估 Action。"""
        normalized_user_text = str(user_text or "").strip()
        signal_source = "user_text"
        signal_text = normalized_user_text

        blocked = self._negative_decision(
            config=config,
            user_text=normalized_user_text,
            user_text_age_seconds=user_text_age_seconds,
        )
        if blocked is not None:
            return blocked

        if normalized_user_text:
            explicit_request = detect_explicit_image_request(normalized_user_text)
        else:
            reasoning_text = str(reasoning or "").strip()
            signal_source = "reasoning" if reasoning_text else "none"
            signal_text = reasoning_text
            explicit_request = reasoning_implies_explicit_request(reasoning_text)

        category = "explicit" if explicit_request else "proactive"
        allowed, detail = self._check_interval(
            stream_id=stream_id,
            config=config,
            category=category,
        )
        return AdmissionDecision(
            should_generate=allowed,
            category=category,
            detail=detail,
            explicit_request=explicit_request,
            signal_source=signal_source,
            signal_text=signal_text[:120],
        )

    def evaluate_auto_draw(
        self,
        *,
        stream_id: str,
        config: dict[str, Any],
        user_text: str = "",
        user_text_age_seconds: float | None = None,
    ) -> AdmissionDecision:
        """评估 reply 自动跟图的用户否定意图与独立冷却。"""
        normalized_user_text = str(user_text or "").strip()
        blocked = self._negative_decision(
            config=config,
            user_text=normalized_user_text,
            user_text_age_seconds=user_text_age_seconds,
        )
        if blocked is not None:
            return blocked

        allowed, detail = self._check_interval(
            stream_id=stream_id,
            config=config,
            category="auto_draw",
        )
        return AdmissionDecision(
            should_generate=allowed,
            category="auto_draw",
            detail=detail,
            signal_source="user_text" if normalized_user_text else "none",
            signal_text=normalized_user_text[:120],
        )

    def evaluate_reply_candidate(
        self,
        *,
        stream_id: str,
        reply_text: str,
        config: dict[str, Any],
    ) -> AdmissionDecision:
        """评分并原子认领一条 reply，防止 hook retry 重复提交。"""
        normalized_stream_id = str(stream_id or "").strip()
        normalized_reply = str(reply_text or "").strip()
        auto_config = self._get_config(config, "auto_draw_on_reply", {})
        if not normalized_stream_id or not normalized_reply:
            return AdmissionDecision(False, "blocked", "会话或 reply 为空")
        if not isinstance(auto_config, dict) or not auto_config.get("enabled", True):
            return AdmissionDecision(False, "blocked", "reply 自动跟图已关闭")

        signal = score_reply_for_auto_draw(normalized_reply)
        threshold = float(auto_config.get("score_threshold", 0.6))
        if not signal.should_draw or signal.score < threshold:
            return AdmissionDecision(False, "blocked", "reply 未达到自动跟图阈值")

        description = compose_description_from_reply(normalized_reply, signal)
        if not description:
            return AdmissionDecision(False, "blocked", "reply 未生成有效图片描述")

        if not self._claim_reply(normalized_stream_id, normalized_reply):
            return AdmissionDecision(False, "blocked", "reply 已提交过自动跟图")

        return AdmissionDecision(
            should_generate=True,
            category="auto_draw",
            detail="reply 自动跟图准入",
            signal_source="reply_text",
            signal_text=normalized_reply[:120],
            seed_description=description,
        )

    def record_success(
        self,
        *,
        stream_id: str,
        category: str,
        sent_at: float | None = None,
    ) -> None:
        """图片成功交给发送 Adapter 后，按准入类别更新对应冷却。"""
        timestamp = self._clock() if sent_at is None else float(sent_at)
        if category == "auto_draw":
            self._state.set_last_auto_draw_sent_at(stream_id, timestamp)
            return
        self._state.set_last_action_image_sent_at(stream_id, timestamp)

    def clear(self) -> None:
        """清空本进程持有的 reply 去重状态。"""
        self._reply_claims.clear()

    def _negative_decision(
        self,
        *,
        config: dict[str, Any],
        user_text: str,
        user_text_age_seconds: float | None,
    ) -> AdmissionDecision | None:
        if not user_text:
            return None
        strength = detect_negative_image_intent_strength(user_text)
        if strength == "strong":
            return AdmissionDecision(
                False,
                "blocked",
                "用户明确表示不需要图片",
                signal_source="user_text",
                signal_text=user_text[:120],
            )
        if strength != "weak":
            return None

        weak_ttl = max(
            0,
            int(self._get_config(config, "action_guard.weak_negative_ttl_seconds", 60)),
        )
        if user_text_age_seconds is not None and user_text_age_seconds > weak_ttl:
            return None
        return AdmissionDecision(
            False,
            "blocked",
            "用户刚才偏好文字回复",
            signal_source="user_text",
            signal_text=user_text[:120],
        )

    def _check_interval(
        self,
        *,
        stream_id: str,
        config: dict[str, Any],
        category: str,
    ) -> tuple[bool, str]:
        explicit_interval = max(
            0,
            int(self._get_config(config, "action_guard.explicit_request_min_interval_seconds", 5)),
        )
        proactive_interval = max(
            0,
            int(self._get_config(config, "action_guard.proactive_min_interval_seconds", 10)),
        )
        auto_draw_interval = max(
            0,
            int(self._get_config(config, "auto_draw_on_reply.min_interval_seconds", 15)),
        )
        last_action_at = self._state.get_last_action_image_sent_at(stream_id)
        last_auto_draw_at = self._state.get_last_auto_draw_sent_at(stream_id)

        if category == "auto_draw":
            effective_last = max(
                (value for value in (last_action_at, last_auto_draw_at) if value is not None),
                default=None,
            )
            required_interval = auto_draw_interval
        elif category == "explicit":
            effective_last = last_action_at
            required_interval = explicit_interval
        else:
            effective_last = last_action_at
            required_interval = proactive_interval

        if effective_last is None:
            return True, "首次出图"
        elapsed = max(0.0, self._clock() - effective_last)
        if elapsed >= required_interval:
            return True, "触发条件满足"

        remaining_seconds = int(required_interval - elapsed + 0.999)
        self._logger.debug(
            "生成准入节流命中: category=%s required=%ds remaining=%ds",
            category,
            required_interval,
            remaining_seconds,
        )
        return False, _THROTTLED_DETAIL

    def _claim_reply(self, stream_id: str, reply_text: str) -> bool:
        now = self._clock()
        self._prune_reply_claims(now)
        signature = hashlib.sha256(reply_text.encode("utf-8")).hexdigest()
        claims = self._reply_claims.get(stream_id)
        if claims is None:
            claims = OrderedDict()
            self._reply_claims[stream_id] = claims
        else:
            self._reply_claims.move_to_end(stream_id)
        if signature in claims:
            return False

        claims[signature] = now
        while len(claims) > self._reply_claims_per_session:
            claims.popitem(last=False)
        while len(self._reply_claims) > self._max_reply_sessions:
            self._reply_claims.popitem(last=False)
        return True

    def _prune_reply_claims(self, now: float) -> None:
        if self._reply_claim_ttl_seconds <= 0:
            self._reply_claims.clear()
            return
        for stream_id, claims in list(self._reply_claims.items()):
            while claims:
                _signature, claimed_at = next(iter(claims.items()))
                if now - claimed_at <= self._reply_claim_ttl_seconds:
                    break
                claims.popitem(last=False)
            if not claims:
                self._reply_claims.pop(stream_id, None)

    @staticmethod
    def _get_config(config: dict[str, Any], path: str, default: Any) -> Any:
        current: Any = config
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def reasoning_implies_explicit_request(reasoning: str) -> bool:
    """用户原话不可用时，从 Planner reasoning 保守识别显式请求。"""
    normalized = str(reasoning or "").strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    return any(
        hint in normalized or hint.lower() in lowered
        for hint in _REASONING_EXPLICIT_HINTS
    )
