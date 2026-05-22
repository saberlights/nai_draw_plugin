import os
import sys
import types
from pathlib import Path

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

from plugins.nai_draw_plugin.sdk_runtime import NaiInvocation


def test_runtime_selfie_anchor_helpers_are_disabled() -> None:
    invocation = object.__new__(NaiInvocation)

    assert invocation._extract_selfie_anchor_data(
        "solo, 1girl, selfie, mirror selfie, by window, sweater, black pantyhose"
    ) == {}
    assert invocation._format_selfie_anchor_summary({"scene_type": ["前置自拍"]}) == ""


def test_runtime_selfie_scene_context_ignores_anchor_summary() -> None:
    invocation = object.__new__(NaiInvocation)

    context = invocation._build_selfie_scene_context(
        "再来一张",
        last_selfie_prompt="solo, 1girl, selfie, by window",
        last_selfie_request="发张自拍",
        last_selfie_scene="- 自拍类型：前置自拍",
        last_selfie_anchor={"scene_type": ["前置自拍"]},
    )

    assert "上一轮自拍锚点" not in context
    assert "上一轮用户请求：发张自拍" in context
    assert "上一轮自拍提示词：solo, 1girl, selfie, by window" in context
