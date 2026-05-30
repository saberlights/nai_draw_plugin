import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _get_logger(_name=None):
    return _DummyLogger()


dummy_logger_module.get_logger = _get_logger
sys.modules["src.common.logger"] = dummy_logger_module

src_package = types.ModuleType("src")
src_package.__path__ = [os.path.join(MAIBOT_ROOT, "src")]
sys.modules.setdefault("src", src_package)

src_config_package = types.ModuleType("src.config")
src_config_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "config")]
sys.modules.setdefault("src.config", src_config_package)

config_module = types.ModuleType("src.config.config")
config_module.global_config = types.SimpleNamespace()
config_module.model_config = types.SimpleNamespace(
    model_task_config=types.SimpleNamespace(embedding=None)
)
sys.modules["src.config.config"] = config_module

model_configs_module = types.ModuleType("src.config.model_configs")
model_configs_module.TaskConfig = type("TaskConfig", (), {})
sys.modules["src.config.model_configs"] = model_configs_module

src_llm_models_package = types.ModuleType("src.llm_models")
src_llm_models_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "llm_models")]
sys.modules.setdefault("src.llm_models", src_llm_models_package)

utils_model_module = types.ModuleType("src.llm_models.utils_model")


class _DummyLLMOrchestrator:
    def __init__(self, *args, **kwargs):
        self.model_for_task = None
        self.model_usage = {}


utils_model_module.LLMOrchestrator = _DummyLLMOrchestrator
sys.modules["src.llm_models.utils_model"] = utils_model_module

src_services_module = types.ModuleType("src.services")
src_services_module.llm_service = types.SimpleNamespace()
sys.modules["src.services"] = src_services_module

tag_retriever_module = types.ModuleType("plugins.nai_draw_plugin.core.services.tag_retriever")
tag_retriever_module.get_tag_retriever = lambda **_kwargs: None
sys.modules.setdefault("plugins.nai_draw_plugin.core.services.tag_retriever", tag_retriever_module)

mixins_package = types.ModuleType("plugins.nai_draw_plugin.core.mixins")
mixins_package.__path__ = [os.path.join(MAIBOT_ROOT, "plugins", "nai_draw_plugin", "core", "mixins")]
sys.modules.setdefault("plugins.nai_draw_plugin.core.mixins", mixins_package)

from plugins.nai_draw_plugin.sdk_runtime import NaiInvocation


def _build_invocation() -> NaiInvocation:
    invocation = object.__new__(NaiInvocation)
    invocation.plugin_config = {
        "model": {
            "base_url": "https://std.loliyc.com",
            "nai_size": "竖图",
        }
    }
    invocation.stream_id = "stream-1"
    invocation.group_id = ""
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    return invocation


def test_run_image_pipeline_skips_prompt_echo_for_raw_prompt_vibe() -> None:
    invocation = _build_invocation()
    sent_texts: list[tuple[str, bool]] = []
    generate_calls: list[dict[str, object]] = []
    send_result_calls: list[tuple[str, str]] = []

    async def fake_ensure_generation_permission() -> bool:
        return True

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append((text, storage_message))
        return True

    async def fake_generate_image(**kwargs):
        generate_calls.append(kwargs)
        return True, "image-result"

    async def fake_send_image_result(result: str, description: str) -> tuple[bool, str, bool]:
        send_result_calls.append((result, description))
        return True, "图片生成成功", True

    async def fail_generate_prompt(*args, **kwargs):
        pytest.fail("raw_prompt 路径不应调用 LLM 翻译")

    invocation.ensure_generation_permission = fake_ensure_generation_permission
    invocation.send_text = fake_send_text
    invocation.api_client = types.SimpleNamespace(generate_image=fake_generate_image)
    invocation._generate_prompt_with_llm = fail_generate_prompt
    invocation._get_model_config = lambda is_selfie=False: invocation.plugin_config["model"]
    invocation._is_prompt_show_enabled = lambda: True
    invocation._read_clamped_float_config = lambda key, default, lo, hi: default
    invocation._sanitize_prompt_for_sfw_mode = lambda prompt: prompt
    invocation._sanitize_structured_for_sfw_mode = lambda structured: structured
    invocation._select_send_payload = lambda prompt, structured: (prompt, None)
    invocation._send_image_result = fake_send_image_result

    result = asyncio.run(
        invocation._run_image_pipeline(
            description="1girl, ghost",
            vibe_images_base64=["iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"],
            mode="vibe",
            raw_prompt="1girl, ghost",
        )
    )

    assert result == (True, "图片生成成功", True)
    assert sent_texts == []
    assert generate_calls == [
        {
            "prompt": "1girl, ghost",
            "model_config": invocation.plugin_config["model"],
            "size": "竖图",
            "characters": None,
            "i2i_payload": None,
            "controlnet_payload": {
                "images": [
                    {
                        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
                        "info_extracted": 0.7,
                        "strength": 0.6,
                    }
                ],
                "strength": 1.0,
            },
            "character_references_payload": None,
        }
    ]
    assert send_result_calls == [("image-result", "1girl, ghost")]


def test_run_image_pipeline_shows_prompt_for_llm_vibe_when_enabled() -> None:
    invocation = _build_invocation()
    sent_texts: list[tuple[str, bool]] = []
    generate_calls: list[dict[str, object]] = []

    async def fake_ensure_generation_permission() -> bool:
        return True

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append((text, storage_message))
        return True

    async def fake_generate_prompt(*args, **kwargs):
        return "1girl, ghost", None

    async def fake_generate_image(**kwargs):
        generate_calls.append(kwargs)
        return True, "image-result"

    async def fake_send_image_result(result: str, description: str) -> tuple[bool, str, bool]:
        return True, "图片生成成功", True

    invocation.ensure_generation_permission = fake_ensure_generation_permission
    invocation.send_text = fake_send_text
    invocation.api_client = types.SimpleNamespace(generate_image=fake_generate_image)
    invocation._generate_prompt_with_llm = fake_generate_prompt
    invocation._get_model_config = lambda is_selfie=False: invocation.plugin_config["model"]
    invocation._is_prompt_show_enabled = lambda: True
    invocation._read_clamped_float_config = lambda key, default, lo, hi: default
    invocation._sanitize_prompt_for_sfw_mode = lambda prompt: prompt
    invocation._sanitize_structured_for_sfw_mode = lambda structured: structured
    invocation._select_send_payload = lambda prompt, structured: (prompt, None)
    invocation._send_image_result = fake_send_image_result

    result = asyncio.run(
        invocation._run_image_pipeline(
            description="ghost costume",
            vibe_images_base64=["iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"],
            mode="vibe",
        )
    )

    assert result == (True, "图片生成成功", True)
    assert sent_texts == [("📝 提示词:\n1girl, ghost", False)]
    assert generate_calls[0]["prompt"] == "1girl, ghost"
