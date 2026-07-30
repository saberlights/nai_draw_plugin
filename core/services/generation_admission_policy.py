"""集中 Planner Action 的图片生成准入、冷却判定与成功记账。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

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


class GenerationAdmissionPolicy:
    """Planner Action 的准入状态机。"""

    def __init__(
        self,
        *,
        state: Any,
        logger: Any,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._state = state
        self._logger = logger
        self._clock = clock

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

    def record_success(
        self,
        *,
        stream_id: str,
        sent_at: float | None = None,
    ) -> None:
        """图片成功交给发送 Adapter 后更新 Action 冷却。"""
        timestamp = self._clock() if sent_at is None else float(sent_at)
        self._state.set_last_action_image_sent_at(stream_id, timestamp)

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
        last_action_at = self._state.get_last_action_image_sent_at(stream_id)

        if category == "explicit":
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
