from __future__ import annotations

from typing import Any

from src.common.data_models.llm_service_data_models import LLMGenerationOptions, LLMImageOptions
from src.services.embedding_service import EmbeddingServiceClient
from src.services.llm_service import LLMServiceClient, resolve_task_name, resolve_task_name_from_model_config


class LegacyLLMRequest:
    """兼容旧插件调用习惯的最小 LLM 包装。"""

    def __init__(self, model_set: Any = None, request_type: str = "") -> None:
        self.model_set = model_set
        self.request_type = str(request_type or "")

    def _resolve_task_name(self, preferred_task_name: str = "") -> str:
        if self.model_set is not None:
            try:
                return resolve_task_name_from_model_config(
                    self.model_set,
                    preferred_task_name=preferred_task_name,
                )
            except Exception:
                pass

        if preferred_task_name:
            try:
                return resolve_task_name(preferred_task_name)
            except Exception:
                pass

        return resolve_task_name("")

    async def generate_response_async(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Any]:
        task_name = self._resolve_task_name()
        client = LLMServiceClient(
            task_name=task_name,
            request_type=self.request_type or "legacy_llm_request",
        )
        result = await client.generate_response(
            prompt,
            options=LLMGenerationOptions(
                temperature=temperature,
                max_tokens=max_tokens,
                raise_when_empty=False,
            ),
        )
        return result.response or "", result

    async def generate_response_for_image(
        self,
        *,
        prompt: str,
        image_base64: str,
        image_format: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Any]:
        task_name = self._resolve_task_name("vlm")
        client = LLMServiceClient(
            task_name=task_name,
            request_type=self.request_type or "legacy_llm_request",
        )
        result = await client.generate_response_for_image(
            prompt=prompt,
            image_base64=image_base64,
            image_format=image_format,
            options=LLMImageOptions(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        return result.response or "", result

    async def get_embedding(self, text: str) -> tuple[list[float], Any]:
        task_name = self._resolve_task_name("embedding")
        client = EmbeddingServiceClient(
            task_name=task_name,
            request_type=self.request_type or "embedding",
        )
        result = await client.embed_text(text)
        return list(result.embedding or []), result
