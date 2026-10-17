"""MaiBot 文本模型调用 Adapter。"""

from __future__ import annotations

from typing import Any, Protocol

from src.common.logger import get_logger
from src.services import llm_service

from .pinned_task_orchestrator import (
    PinnedTaskLLMOrchestrator,
    build_pinned_task_config,
    coerce_float,
    coerce_int,
)


logger = get_logger("nai_draw_plugin")


class LLMTextGenerator(Protocol):
    """提示词工作流依赖的文本生成 Interface。"""

    async def generate(
        self,
        prompt: str,
        *,
        request_type: str,
        generator_config: dict[str, Any],
        default_model_name: str,
        default_temperature: float,
        default_max_tokens: int,
    ) -> str | None: ...


class MaiBotLLMTextGenerator:
    """隐藏自定义模型与宿主任务模型的选择、回退和参数归一化。"""

    def __init__(self, log_prefix: str) -> None:
        self._log_prefix = log_prefix

    @staticmethod
    def _resolve_task_name(preferred_name: str) -> str | None:
        models = llm_service.get_available_models()
        if not models:
            return None

        for candidate in (preferred_name, "planner", "replyer"):
            normalized = str(candidate or "").strip()
            if normalized and normalized in models:
                return normalized
        return next(iter(models.keys()), None)

    async def generate(
        self,
        prompt: str,
        *,
        request_type: str,
        generator_config: dict[str, Any],
        default_model_name: str,
        default_temperature: float,
        default_max_tokens: int,
    ) -> str | None:
        custom_model = generator_config.get("custom_model")
        temperature = self._as_float(
            generator_config.get("temperature", default_temperature),
            default_temperature,
        )
        max_tokens = self._as_int(
            generator_config.get("max_tokens", default_max_tokens),
            default_max_tokens,
        )

        if isinstance(custom_model, dict) and custom_model.get("model_list"):
            response = await self._generate_with_custom_model(
                prompt,
                request_type=request_type,
                custom_model=custom_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response:
                return response

        task_name = self._resolve_task_name(
            str(generator_config.get("model_name", "") or default_model_name)
        )
        if not task_name:
            return None

        result = await llm_service.generate(
            llm_service.LLMServiceRequest(
                task_name=task_name,
                request_type=request_type,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not result.success or not result.completion.response:
            return None
        return result.completion.response.strip()

    async def _generate_with_custom_model(
        self,
        prompt: str,
        *,
        request_type: str,
        custom_model: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        try:
            pinned_task = build_pinned_task_config(
                custom_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if pinned_task is None:
                return None

            orchestrator = PinnedTaskLLMOrchestrator(
                pinned_task,
                request_type=request_type,
            )
            result = await orchestrator.generate_response_async(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = (result.response or "").strip()
            if response:
                return response
            logger.warning(
                "%s 自定义提示词模型返回空响应，回退到宿主任务模型",
                self._log_prefix,
            )
        except Exception as exc:
            logger.warning(
                "%s 自定义提示词模型调用失败，回退到宿主任务模型: %s",
                self._log_prefix,
                exc,
            )
        return None

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        return coerce_float(value, default)

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        return coerce_int(value, default)
