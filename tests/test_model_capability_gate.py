# -*- coding: utf-8 -*-
"""``/nai ref`` / ``/nai vibe`` 的模型能力门控。

V5 目前只提供文生图与图生图。旧行为是照常翻译、照常出图，只在日志里 warning 一句
"已降级"，用户只能看到一张参考图没生效的图；现在必须在 LLM 之前就明确回话拦下。
"""
from __future__ import annotations

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


dummy_logger_module.get_logger = lambda _name=None: _DummyLogger()
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


def _build_invocation(model_name: str) -> tuple[NaiInvocation, list[str]]:
    """造一个只够跑能力门控的 invocation，返回 (invocation, 已发送文本列表)。"""
    invocation = object.__new__(NaiInvocation)
    invocation.plugin_config = {
        "model": {
            "base_url": "https://std.loliyc.com",
            "default_model": model_name,
            "nai_size": "竖图",
        }
    }
    invocation.stream_id = "stream-1"
    invocation.group_id = ""
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"

    sent_texts: list[str] = []

    async def fake_ensure_generation_permission() -> bool:
        return True

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    async def fail_generate_prompt(*args, **kwargs):
        pytest.fail("能力门控应在 LLM 翻译之前拦下，不该消耗 LLM 调用")

    async def fail_generate_image(**kwargs):
        pytest.fail("能力门控应在发请求之前拦下")

    invocation.ensure_generation_permission = fake_ensure_generation_permission
    invocation.send_text = fake_send_text
    invocation._generate_prompt_with_llm = fail_generate_prompt
    invocation.api_client = types.SimpleNamespace(generate_image=fail_generate_image)
    # 会话未选定模型 → _resolve_active_model_name 落到 model.default_model
    invocation.action_message = None
    invocation.message = None
    return invocation, sent_texts


def test_v5_rejects_ref_before_calling_llm() -> None:
    """V5 不支持 §20.4 角色参考：必须在 LLM 之前回话拦下。"""
    invocation, sent_texts = _build_invocation("nai-diffusion-5-full")

    ok, reason, handled = asyncio.run(
        invocation._run_image_pipeline(
            description="站在街道，看向镜头",
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
            mode="ref",
        )
    )

    assert ok is False
    assert reason == "当前模型不支持 ref"
    assert handled is True
    assert len(sent_texts) == 1
    assert "不支持角色参考" in sent_texts[0]
    assert "nai-diffusion-5-full" in sent_texts[0]
    assert "V5" in sent_texts[0]
    # 提示里要给出可执行的下一步，而不是只说"不支持"
    assert "/nai set 4.5" in sent_texts[0]


def test_v5_rejects_vibe_before_calling_llm() -> None:
    """V5 不支持 §20.3 氛围转移：同样必须提前拦下。"""
    invocation, sent_texts = _build_invocation("nai-diffusion-5-full")

    ok, reason, handled = asyncio.run(
        invocation._run_image_pipeline(
            description="都市夜景，霓虹氛围",
            vibe_images_base64=["iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"],
            mode="vibe",
        )
    )

    assert ok is False
    assert reason == "当前模型不支持 vibe"
    assert handled is True
    assert len(sent_texts) == 1
    assert "不支持氛围转移" in sent_texts[0]
    assert "/nai set 4.5" in sent_texts[0]


def test_v4_rejects_ref_but_allows_vibe() -> None:
    """V4 没有角色参考但有氛围转移，门控不能一刀切。"""
    invocation, sent_texts = _build_invocation("nai-diffusion-4-full")

    assert asyncio.run(invocation._ensure_mode_supported_by_model("ref")) is False
    assert "不支持角色参考" in sent_texts[0]

    assert asyncio.run(invocation._ensure_mode_supported_by_model("vibe")) is True
    assert len(sent_texts) == 1


def test_v4_5_allows_both_ref_and_vibe() -> None:
    """V4.5 是功能最全的档位，两条命令都不能被拦。"""
    invocation, sent_texts = _build_invocation("nai-diffusion-4-5-full")

    assert asyncio.run(invocation._ensure_mode_supported_by_model("ref")) is True
    assert asyncio.run(invocation._ensure_mode_supported_by_model("vibe")) is True
    assert sent_texts == []


def test_i2i_is_never_gated() -> None:
    """图生图是所有模型都有的能力，V5 下也必须放行。"""
    invocation, sent_texts = _build_invocation("nai-diffusion-5-full")

    assert asyncio.run(invocation._ensure_mode_supported_by_model("i2i")) is True
    assert sent_texts == []
