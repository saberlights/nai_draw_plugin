from __future__ import annotations

import asyncio
import os
import re
import sys
import types
from pathlib import Path

import pytest


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIBOT_ROOT))


# sdk_runtime imports a few MaiBot services at module import time.  Keep these
# stubs local to the test module, matching the existing isolated runtime tests.
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

tag_retriever_module = types.ModuleType(
    "plugins.nai_draw_plugin.core.services.tag_retriever"
)
tag_retriever_module.get_tag_retriever = lambda **_kwargs: None
sys.modules.setdefault(
    "plugins.nai_draw_plugin.core.services.tag_retriever", tag_retriever_module
)

mixins_package = types.ModuleType("plugins.nai_draw_plugin.core.mixins")
mixins_package.__path__ = [
    os.path.join(MAIBOT_ROOT, "plugins", "nai_draw_plugin", "core", "mixins")
]
sys.modules.setdefault("plugins.nai_draw_plugin.core.mixins", mixins_package)

from plugins.nai_draw_plugin.sdk_runtime import NaiInvocation  # noqa: E402
from plugins.nai_draw_plugin.core.services.random_scene_planner import (  # noqa: E402
    RandomScenePlanner,
)


class _FakeTextGenerator:
    def __init__(self, responses: str | list[str]) -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **_kwargs: object) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]


def _build_random_planner(
    responses: str | list[str],
) -> tuple[RandomScenePlanner, _FakeTextGenerator]:
    text_generator = _FakeTextGenerator(responses)
    planner = RandomScenePlanner(
        config={},
        text_generator=text_generator,
        log_prefix="test",
    )
    return planner, text_generator


def _build_invocation() -> NaiInvocation:
    invocation = object.__new__(NaiInvocation)
    invocation.plugin_config = {
        "random_scene": {},
        "prompt_generator": {},
        "model": {"base_url": "https://example.invalid", "nai_size": "竖图"},
    }
    invocation.stream_id = "random-scene-test"
    invocation.group_id = ""
    invocation.user_id = "user-1"
    invocation.log_prefix = "test"
    return invocation


def _load_nai_draw_pattern() -> re.Pattern[str]:
    source = (PLUGIN_ROOT / "plugin.py").read_text(encoding="utf-8")
    match = re.search(
        r'@Command\(\s*"nai_draw"(?:(?!@Command).)*?pattern=r"(?P<pattern>[^"]+)"',
        source,
        re.DOTALL,
    )
    assert match is not None, "plugin.py 应注册 nai_draw 命令"
    return re.compile(match.group("pattern"))


def test_nai_draw_pattern_passes_random_character_text_to_description() -> None:
    pattern = _load_nai_draw_pattern()

    for sample, expected in [
        ("/nai 随机", "随机"),
        ("/nai random", "random"),
        ("/nai rand", "rand"),
        ("/nai 随机 初音未来", "随机 初音未来"),
        ("/nai随机 初音未来", "随机 初音未来"),
        ("/nai随机自拍", "随机自拍"),
    ]:
        match = pattern.match(sample)
        assert match is not None, f"nai_draw 应匹配 {sample!r}"
        assert match.group("description") == expected


def test_random_scene_prompt_locks_character_and_demands_broad_variation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RandomScenePlanner, "_recent_scenes", [])
    planner, text_generator = _build_random_planner(
        "成年初音未来在酒店浴室对镜自拍，镜头从侧后方拍摄。"
    )
    asyncio.run(planner.generate(character="初音未来"))
    prompt = text_generator.prompts[0]

    assert "初音未来" in prompt
    assert any(term in prompt for term in ("角色锚点", "指定角色", "锁定角色", "固定角色"))
    assert any(term in prompt for term in ("成人向", "成人内容", "NSFW"))

    topic_groups = (
        ("性交", "插入", "性行为", "体位"),
        ("口交", "乳交", "肛交", "后入"),
        ("多人", "群体", "群交", "关系"),
        ("拘束", "触手", "异种", "医疗"),
        ("露出", "公共", "制服", "角色扮演"),
    )
    assert sum(any(term in prompt for term in group) for group in topic_groups) >= 4

    dimensions = (
        ("人数", "人物构成"),
        ("姿势", "动作", "体位"),
        ("视角", "镜头"),
        ("场景", "环境"),
        ("服装", "穿着"),
        ("道具", "物件"),
    )
    assert sum(any(term in prompt for term in group) for group in dimensions) >= 4
    assert any(term in prompt for term in ("随机", "随机化", "每次不同", "主动切换"))
    assert any(term in prompt for term in ("模板", "套路", "固定组合"))

    monkeypatch.setattr(RandomScenePlanner, "_recent_scenes", ["初音未来 制服 教室"])
    planner, text_generator = _build_random_planner(
        ["初音未来 制服 教室", "初音未来 制服 教室"],
    )
    asyncio.run(planner.generate(character="初音未来"))
    history_prompt = text_generator.prompts[1]
    assert "最近" in history_prompt
    assert any(term in history_prompt for term in ("禁止", "避免"))
    assert any(term in history_prompt for term in ("重复", "相似"))
    assert "初音未来 制服 教室" in history_prompt


def test_random_scene_prompt_requires_rich_directed_visual_detail() -> None:
    planner, text_generator = _build_random_planner(
        "成年初音未来在酒店浴室对镜自拍，镜头从侧后方拍摄。"
    )
    asyncio.run(planner.generate(character="初音未来"))
    prompt = text_generator.prompts[0]

    detail_groups = (
        ("情色主轴", "明确成人行为", "成人内容必须是画面主轴"),
        ("角色当下状态", "表情", "身体状态"),
        ("服饰状态", "衣物", "穿着状态"),
        ("配饰", "饰品", "道具"),
        ("前景", "中景", "背景"),
        ("视觉焦点", "构图焦点", "主体位置"),
        ("视角", "景别", "镜头角度"),
        ("材质", "颜色", "服装款式"),
    )
    for group in detail_groups:
        assert any(term in prompt for term in group), f"缺少画面细节要求：{group}"

    assert any(term in prompt for term in ("先在内部构思", "导演", "画面设计"))
    assert any(term in prompt for term in ("自然语言", "完整中文句子", "完整句子"))
    assert "14-22" not in prompt
    assert "空格分隔" not in prompt
    assert any(term in prompt for term in ("发挥想象", "想象力", "新奇组合"))


def test_generate_random_description_reanchors_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RandomScenePlanner, "_recent_scenes", [])
    planner, text_generator = _build_random_planner(
        "一名成年女性在酒店浴室对镜自拍，镜头从侧后方拍摄。"
    )

    result = asyncio.run(planner.generate(character="初音未来"))

    assert result is not None
    assert result == "主角是初音未来。一名成年女性在酒店浴室对镜自拍，镜头从侧后方拍摄。"
    assert text_generator.prompts and "初音未来" in text_generator.prompts[0]


def test_generate_random_description_preserves_natural_language_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RandomScenePlanner, "_recent_scenes", [])
    natural_description = (
        "成年初音未来在霓虹灯照亮的酒店浴室里对镜自拍，"
        "她穿着黑色蕾丝内衣，镜头捕捉镜面反射。"
    )

    planner, _text_generator = _build_random_planner(natural_description)
    result = asyncio.run(planner.generate(character="初音未来"))

    assert result == natural_description


def _install_draw_doubles(
    invocation: NaiInvocation,
    *,
    prompt_result: str,
    selfie_calls: list[str],
    generated_prompts: list[str],
    generated_images: list[dict[str, object]],
) -> None:
    async def fake_ensure_generation_permission() -> bool:
        return True

    async def fake_send_text(text: str, storage_message: bool = True) -> bool:
        return True

    async def fake_generate_prompt(description: str, **kwargs):
        generated_prompts.append(description)
        return prompt_result, None

    def fake_process_selfie_prompt(description: str, *args, **kwargs) -> str:
        selfie_calls.append(description)
        return description

    async def fake_generate_image(**kwargs):
        generated_images.append(kwargs)
        return True, "image-result"

    async def fake_send_image_result(result: str, description: str):
        return True, "图片生成成功", True

    invocation.ensure_generation_permission = fake_ensure_generation_permission
    invocation.send_text = fake_send_text
    invocation._generate_prompt_with_llm = fake_generate_prompt
    invocation._process_selfie_prompt = fake_process_selfie_prompt
    invocation._is_prompt_show_enabled = lambda: False
    invocation._get_model_config = lambda is_selfie=False: {
        "base_url": "https://example.invalid",
        "nai_size": "竖图",
    }
    invocation._sanitize_prompt_for_sfw_mode = lambda prompt: prompt
    invocation._sanitize_structured_for_sfw_mode = lambda structured: structured
    invocation._select_send_payload = lambda prompt, structured: (prompt, None)
    invocation.api_client = types.SimpleNamespace(generate_image=fake_generate_image)
    invocation._send_image_result = fake_send_image_result


def test_named_character_random_scene_with_selfie_word_stays_normal_draw() -> None:
    invocation = _build_invocation()
    random_requests: list[tuple[bool, str]] = []
    generated_prompts: list[str] = []
    selfie_calls: list[str] = []
    generated_images: list[dict[str, object]] = []
    natural_description = (
        "成年初音未来在霓虹灯照亮的酒店浴室里自拍，"
        "她穿着黑色蕾丝内衣，镜头从侧后方拍摄。"
    )

    async def fake_random_description(*, selfie: bool, character: str) -> str:
        random_requests.append((selfie, character))
        return natural_description

    invocation._generate_random_description = fake_random_description
    _install_draw_doubles(
        invocation,
        prompt_result="hatsune miku, selfie, rear entry",
        selfie_calls=selfie_calls,
        generated_prompts=generated_prompts,
        generated_images=generated_images,
    )

    result = asyncio.run(invocation.handle_nai_draw("随机 初音未来"))

    assert result == (True, "图片生成成功", True)
    assert random_requests == [(False, "初音未来")]
    assert generated_prompts == [natural_description]
    assert selfie_calls == []
    assert generated_images and generated_images[0]["prompt"] == "hatsune miku, selfie, rear entry"


def test_random_selfie_has_no_character_anchor_and_keeps_selfie_postprocess() -> None:
    invocation = _build_invocation()
    random_requests: list[tuple[bool, str]] = []
    generated_prompts: list[str] = []
    selfie_calls: list[str] = []
    generated_images: list[dict[str, object]] = []
    natural_description = (
        "一名成年女性在酒店浴室对镜自拍，"
        "她举着手机，镜面映出潮湿的肌肤。"
    )

    async def fake_random_description(*, selfie: bool, character: str) -> str:
        random_requests.append((selfie, character))
        return natural_description

    invocation._generate_random_description = fake_random_description
    _install_draw_doubles(
        invocation,
        prompt_result="girl, mirror selfie, holding phone",
        selfie_calls=selfie_calls,
        generated_prompts=generated_prompts,
        generated_images=generated_images,
    )

    result = asyncio.run(invocation.handle_nai_draw("随机自拍"))

    assert result == (True, "图片生成成功", True)
    assert random_requests == [(True, "")]
    assert generated_prompts == [natural_description]
    assert selfie_calls == ["girl, mirror selfie, holding phone"]
    assert generated_images and generated_images[0]["prompt"] == "girl, mirror selfie, holding phone"
