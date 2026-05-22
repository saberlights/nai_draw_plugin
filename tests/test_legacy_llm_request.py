# -*- coding: utf-8 -*-
import os
import sys
import types
import unittest


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIBOT_ROOT = os.path.abspath(os.path.join(PLUGIN_DIR, "../.."))

if MAIBOT_ROOT not in sys.path:
    sys.path.insert(0, MAIBOT_ROOT)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)


class _FakeLLMClient:
    def __init__(self, task_name: str, request_type: str) -> None:
        self.task_name = task_name
        self.request_type = request_type

    async def generate_response(self, prompt, options=None):
        return types.SimpleNamespace(
            response=f"text:{prompt}",
            options=options,
            task_name=self.task_name,
        )

    async def generate_response_for_image(self, *, prompt, image_base64, image_format, options=None):
        return types.SimpleNamespace(
            response=f"image:{prompt}:{image_format}:{len(image_base64)}",
            options=options,
            task_name=self.task_name,
        )


class _FakeEmbeddingClient:
    def __init__(self, task_name: str, request_type: str) -> None:
        self.task_name = task_name
        self.request_type = request_type

    async def embed_text(self, text: str):
        return types.SimpleNamespace(
            embedding=[len(text), len(self.task_name)],
            task_name=self.task_name,
        )


fake_llm_service_module = types.ModuleType("src.services.llm_service")
fake_llm_service_module.LLMServiceClient = _FakeLLMClient
fake_llm_service_module.resolve_task_name = lambda preferred="": preferred or "default_task"
fake_llm_service_module.resolve_task_name_from_model_config = (
    lambda model_config, preferred_task_name="": preferred_task_name or "resolved_from_model"
)
sys.modules["src.services.llm_service"] = fake_llm_service_module

fake_embedding_service_module = types.ModuleType("src.services.embedding_service")
fake_embedding_service_module.EmbeddingServiceClient = _FakeEmbeddingClient
sys.modules["src.services.embedding_service"] = fake_embedding_service_module

fake_data_models_module = types.ModuleType("src.common.data_models.llm_service_data_models")


class _FakeGenerationOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeImageOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


fake_data_models_module.LLMGenerationOptions = _FakeGenerationOptions
fake_data_models_module.LLMImageOptions = _FakeImageOptions
sys.modules["src.common.data_models.llm_service_data_models"] = fake_data_models_module

from legacy_llm_request import LegacyLLMRequest


class LegacyLLMRequestTest(unittest.TestCase):
    def test_generate_response_async_should_use_model_resolver(self):
        import asyncio

        request = LegacyLLMRequest(model_set=object(), request_type="nai_draw_plugin.prompt_generator")
        response, result = asyncio.run(
            request.generate_response_async(
                "hello",
                temperature=0.7,
                max_tokens=128,
            )
        )

        self.assertEqual(response, "text:hello")
        self.assertEqual(result.task_name, "resolved_from_model")
        self.assertEqual(result.options.temperature, 0.7)
        self.assertEqual(result.options.max_tokens, 128)

    def test_generate_response_for_image_should_prefer_vlm_task(self):
        import asyncio

        request = LegacyLLMRequest(model_set=object(), request_type="nai_draw_plugin.vlm")
        response, result = asyncio.run(
            request.generate_response_for_image(
                prompt="tag this",
                image_base64="abcd",
                image_format="png",
                temperature=0.2,
                max_tokens=256,
            )
        )

        self.assertEqual(response, "image:tag this:png:4")
        self.assertEqual(result.task_name, "vlm")
        self.assertEqual(result.options.max_tokens, 256)

    def test_get_embedding_should_prefer_embedding_task(self):
        import asyncio

        request = LegacyLLMRequest(model_set=object(), request_type="embedding")
        embedding, result = asyncio.run(request.get_embedding("danbooru"))

        self.assertEqual(embedding, [8, 9])
        self.assertEqual(result.task_name, "embedding")


if __name__ == "__main__":
    unittest.main()
