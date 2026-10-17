# -*- coding: utf-8 -*-
"""为插件自定义模型固定 TaskConfig 的 LLM 编排器。

提示词生成（``llm_text_generator``）和图像反推（``llm_image_captioner``）都需要
"绕过宿主任务名选择、直接钉一组模型"的能力：宿主的 ``planner`` / ``vlm`` 任务模型
由 MaiBot 全局配置决定，而本插件的两条链路对模型的要求完全不同（反推要视觉强的贵模型，
提示词生成要便宜稳定的）。所以这个编排器不属于任何一方，单独成模块，避免两边各抄一份。
"""

from __future__ import annotations

from typing import Any

from src.config.model_configs import TaskConfig
from src.llm_models.utils_model import LLMOrchestrator


class PinnedTaskLLMOrchestrator(LLMOrchestrator):
    """为插件自定义模型固定 TaskConfig，绕过宿主任务名选择。"""

    def __init__(self, task_config: TaskConfig, request_type: str = "") -> None:
        self._pinned_task_config = task_config
        super().__init__(task_name="planner", request_type=request_type)

    def _get_task_config_or_raise(self) -> TaskConfig:
        return self._pinned_task_config

    def _refresh_task_config(self) -> TaskConfig:
        latest = self._pinned_task_config
        if latest is not self.model_for_task:
            self.model_for_task = latest
        if list(self.model_usage.keys()) != latest.model_list:
            self.model_usage = {
                model: self.model_usage.get(model, (0, 0, 0))
                for model in latest.model_list
            }
        return self.model_for_task


def build_pinned_task_config(
    custom_model: Any,
    *,
    max_tokens: int,
    temperature: float,
) -> TaskConfig | None:
    """从 ``custom_model`` 配置块构造 TaskConfig。

    Args:
        custom_model: 配置里的 ``{model_list, max_tokens, temperature, slow_threshold}``
        max_tokens: ``custom_model`` 未给出时使用的最大 token 数
        temperature: ``custom_model`` 未给出时使用的温度

    Returns:
        可直接交给 :class:`PinnedTaskLLMOrchestrator` 的 TaskConfig；
        ``custom_model`` 不是 dict 或 ``model_list`` 为空时返回 ``None``，
        表示"用户没配自定义模型"，由调用方回落到宿主任务模型
    """
    if not isinstance(custom_model, dict):
        return None

    raw_models = custom_model.get("model_list", [])
    model_items = raw_models if isinstance(raw_models, list) else [raw_models]
    model_list = [str(item).strip() for item in model_items if str(item).strip()]
    if not model_list:
        return None

    # slow_threshold 已从宿主 TaskConfig 移除；旧配置仍可读取，但不再传入。
    return TaskConfig(
        model_list=model_list,
        max_tokens=coerce_int(custom_model.get("max_tokens"), max_tokens),
        temperature=coerce_float(custom_model.get("temperature"), temperature),
        selection_strategy="random",
    )


def coerce_float(value: Any, default: float) -> float:
    """把配置里的任意值归一化为 float，不可解析时取 default。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int) -> int:
    """把配置里的任意值归一化为 int，不可解析时取 default。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "PinnedTaskLLMOrchestrator",
    "build_pinned_task_config",
    "coerce_float",
    "coerce_int",
]
