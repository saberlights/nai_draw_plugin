from __future__ import annotations

import asyncio
import sys
from pathlib import Path


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.prompt_generation_workflow import (  # noqa: E402
    PromptGenerationWorkflow,
)
from plugins.nai_draw_plugin.core.services.session_state import session_state  # noqa: E402
from plugins.nai_draw_plugin.core.services.visual_continuity import (  # noqa: E402
    StableVisualTags,
    VisualChangeDirective,
)


class _FakeTextGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.calls: list[dict[str, object]] = []

    async def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        self.calls.append(kwargs)
        return self.response


class _SequenceTextGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **_kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


async def _ignore_text(_text: str, storage_message: bool = True) -> bool:
    del storage_message
    return True


def test_workflow_reuses_cached_stable_tags_and_only_accepts_new_dynamic_tags() -> None:
    stream_id = "visual-continuity-workflow-test"
    previous = StableVisualTags(
        outfit=("navy ribbed cardigan", "ivory piping", "brass buttons"),
        environment=("walnut bookcase", "cream wall", "sage curtains"),
    )
    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["sitting"],"emotion":["soft smile"],'
        '"scene":["morning light"],"framing":["upper body"]},'
        '"stable":{"outfit":{"mode":"keep","tags":["blue sweater"]},'
        '"environment":{"mode":"keep","tags":["bookshelf"]}}}'
    )
    generator = _FakeTextGenerator(response)
    workflow = PromptGenerationWorkflow(
        config={
            "prompt_generator": {"output_format": "json", "inherit_ttl": 3_600},
            "tag_retriever": {},
        },
        stream_id=stream_id,
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )
    session_state.clear_transient_generation_state(stream_id)
    session_state.set_visual_continuity(stream_id, previous)
    session_state.set_last_nai_context(
        stream_id,
        "rating:general, solo, old pose, old framing",
        "上一轮请求",
        ttl=3_600,
    )
    session_state.set_last_selfie_context(
        stream_id,
        "rating:general, solo, old pose, old framing",
        "上一轮 Bot 情景",
        ttl=3_600,
    )

    try:
        result = asyncio.run(
            workflow.generate(
                "一女，坐在书房里微笑",
                allow_inherit=True,
                use_visual_continuity=True,
            )
        )
    finally:
        session_state.clear_transient_generation_state(stream_id)

    assert result is not None
    assert result.text == (
        "rating:general, solo, 1girl, navy ribbed cardigan, ivory piping, "
        "brass buttons, sitting, soft smile, walnut bookcase, cream wall, "
        "sage curtains, morning light, upper body"
    )
    assert "<visual_continuity_context>" in generator.prompts[0]
    assert "navy ribbed cardigan" in generator.prompts[0]
    assert "old pose" not in generator.prompts[0]
    assert "保留其余标签" not in generator.prompts[0]
    assert "blue sweater" not in result.text


def test_workflow_rejects_invalid_v4_instead_of_sending_raw_json_as_prompt() -> None:
    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":[],"action":["sitting"],"emotion":[],'
        '"scene":[],"framing":[]},"stable":{'
        '"outfit":{"mode":"keep","key":"","tags":[]},'
        '"environment":{"mode":"keep","key":"","tags":[]}}}'
    )
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}},
        stream_id="invalid-visual-continuity-response",
        text_generator=_FakeTextGenerator(response),
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )

    result = asyncio.run(
        workflow.generate(
            "一女坐着",
            allow_inherit=True,
            use_visual_continuity=True,
        )
    )

    session_state.clear_transient_generation_state(
        "invalid-visual-continuity-response"
    )
    assert result is None


def test_workflow_accepts_a_short_first_outfit_without_keyword_rejection() -> None:
    generic = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["standing"],"emotion":[],"scene":[],"framing":["full body"]},'
        '"stable":{"outfit":{"mode":"replace","key":"work_set",'
        '"tags":["skirt"]},"environment":{"mode":"keep","key":"","tags":[]}}}'
    )
    generator = _SequenceTextGenerator([generic])
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id="visual-continuity-repair-test",
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )

    result = asyncio.run(
        workflow.generate(
            "卧室里穿着工作用的短裙",
            allow_inherit=True,
            use_visual_continuity=True,
        )
    )

    session_state.clear_transient_generation_state("visual-continuity-repair-test")
    assert result is not None
    assert "skirt" in result.text
    assert len(generator.prompts) == 1


def test_workflow_accepts_real_qipao_and_film_set_tags_without_false_repair() -> None:
    """复现 21:58 最新日志：合法开放词汇不得被误杀后降级成纯自拍串。"""

    response = (
        '{"version":4,"format":"single","intent":"normal",'
        '"dynamic":{"subject":["rating:questionable","1girl","solo"],'
        '"action":["sitting sideways","hand on lower back","looking at phone"],'
        '"emotion":["panting","resentful eyes","tired"],'
        '"scene":["backlighting","smog","dust motes"],'
        '"framing":["full body"]},"stable":{'
        '"outfit":{"mode":"replace","key":"pale_green_silk_qipao",'
        '"tags":["pale green silk qipao","china dress","silk texture",'
        '"high slit","skin-colored pantyhose","vintage black leather shoes"]},'
        '"environment":{"mode":"replace","key":"messy_film_set_corner",'
        '"tags":["film set","indoor","corner","messy studio","folding chair",'
        '"light stand","black blackout cloth","backdrop"]}}}'
    )
    generator = _FakeTextGenerator(response)
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id="real-qipao-film-set-regression",
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )

    result = asyncio.run(
        workflow.generate(
            "淡青色丝绸旗袍，肉色丝袜，复古黑色皮鞋；凌乱片场、折叠椅、灯架和黑色遮光布",
            allow_inherit=True,
            use_visual_continuity=True,
        )
    )

    session_state.clear_transient_generation_state("real-qipao-film-set-regression")
    assert result is not None
    assert "pale green silk qipao" in result.text
    assert "skin-colored pantyhose" in result.text
    assert "film set" in result.text
    assert "light stand" in result.text
    assert len(generator.prompts) == 1


def test_workflow_rejects_non_v4_text_in_visual_continuity_mode() -> None:
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id="non-v4-visual-continuity-response",
        text_generator=_FakeTextGenerator("一女，淡青色旗袍，坐在片场"),
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )

    result = asyncio.run(
        workflow.generate(
            "一女穿淡青色旗袍坐在片场",
            allow_inherit=True,
            use_visual_continuity=True,
        )
    )

    session_state.clear_transient_generation_state(
        "non-v4-visual-continuity-response"
    )
    assert result is None


def test_workflow_repairs_both_keep_modes_when_scene_delta_declares_change() -> None:
    previous = StableVisualTags(
        outfit=("navy wool blazer", "white silk blouse", "silver buttons"),
        environment=("oak desk", "cream plaster wall", "brass desk lamp"),
        outfit_key="work_set",
        environment_key="home_study",
    )
    invalid_keep = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["sitting"],"emotion":["tired"],"scene":[],'
        '"framing":["medium shot"]},"stable":{'
        '"outfit":{"mode":"keep","key":"work_set","tags":[]},'
        '"environment":{"mode":"keep","key":"home_study","tags":[]}}}'
    )
    repaired = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["sitting"],"emotion":["tired"],"scene":[],'
        '"framing":["medium shot"]},"stable":{'
        '"outfit":{"mode":"replace","key":"pale_cyan_qipao_set",'
        '"tags":["pale cyan silk qipao","high side slit",'
        '"skin-colored pantyhose","vintage black leather shoes"]},'
        '"environment":{"mode":"keep","key":"home_study","tags":[]}}}'
    )
    generator = _SequenceTextGenerator([invalid_keep, repaired])
    stream_id = "scene-delta-repair-test"
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id=stream_id,
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )
    session_state.clear_transient_generation_state(stream_id)
    session_state.set_visual_continuity(stream_id, previous)

    try:
        result = asyncio.run(workflow.generate(
            "<planner_visual_request>...</planner_visual_request>",
            allow_inherit=True,
            use_visual_continuity=True,
            stable_change_text="淡青色丝绸旗袍，肉色丝袜，复古黑色皮鞋",
        ))
    finally:
        session_state.clear_transient_generation_state(stream_id)

    assert result is not None
    assert "pale cyan silk qipao" in result.text
    assert len(generator.prompts) == 2
    assert "visual_continuity_repair" in generator.prompts[1]


def test_workflow_uses_planner_directives_instead_of_llm_stable_modes() -> None:
    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["standing"],"emotion":[],"scene":[],"framing":["full body"]},'
        '"stable":{"outfit":{"mode":"replace","key":"wrong_outfit",'
        '"tags":["red dress"]},"environment":{"mode":"replace",'
        '"key":"wrong_environment","tags":["street"]}}}'
    )
    previous = StableVisualTags(
        outfit=("navy ribbed cardigan", "ivory piping"),
        environment=("walnut bookcase", "cream plaster wall"),
        outfit_key="home_knit_set",
        environment_key="home_study",
    )
    stream_id = "planner-directive-workflow-test"
    generator = _FakeTextGenerator(response)
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id=stream_id,
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )
    session_state.clear_transient_generation_state(stream_id)
    session_state.set_visual_continuity(stream_id, previous)

    try:
        result = asyncio.run(workflow.generate(
            "<planner_visual_request>outfit_change=unchanged; environment_change=unchanged</planner_visual_request>",
            allow_inherit=True,
            use_visual_continuity=True,
            visual_directives={
                "outfit": VisualChangeDirective("keep"),
                "environment": VisualChangeDirective("keep"),
            },
        ))
    finally:
        session_state.clear_transient_generation_state(stream_id)

    assert result is not None
    assert "navy ribbed cardigan" in result.text
    assert "walnut bookcase" in result.text
    assert "red dress" not in result.text
    assert "street" not in result.text


def test_workflow_returns_stable_candidate_without_committing_before_delivery() -> None:
    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["standing"],"emotion":[],"scene":[],'
        '"framing":["full body"]},"stable":{'
        '"outfit":{"mode":"replace","key":"green_coat_set",'
        '"tags":["forest green wool coat","double-breasted coat",'
        '"dark horn buttons"]},"environment":{"mode":"replace",'
        '"key":"old_town_street","tags":["narrow stone street",'
        '"red brick storefront","black iron street lamps"]}}}'
    )
    generator = _FakeTextGenerator(response)
    stream_id = "deferred-visual-continuity-commit"
    workflow = PromptGenerationWorkflow(
        config={
            "prompt_generator": {
                "output_format": "json",
                "inherit_ttl": 3_600,
                "temperature": 1.5,
            },
            "tag_retriever": {},
        },
        stream_id=stream_id,
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )
    session_state.clear_transient_generation_state(stream_id)

    try:
        result = asyncio.run(workflow.generate(
            "新服装与新街道",
            allow_inherit=True,
            use_visual_continuity=True,
            stable_change_text="新服装与新街道",
        ))

        assert result is not None
        assert result.visual_continuity is not None
        assert result.visual_continuity.outfit_key == "green_coat_set"
        assert session_state.get_visual_continuity(stream_id, ttl=3_600) is None
        assert generator.calls[0]["generator_config"]["temperature"] == 0.2
    finally:
        session_state.clear_transient_generation_state(stream_id)


def test_workflow_repairs_malformed_json_with_failure_reason() -> None:
    """非 JSON 响应也要进入修复循环，而不是直接放弃本次出图。"""

    valid = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["standing"],"emotion":[],"scene":[],"framing":["full body"]},'
        '"stable":{"outfit":{"mode":"replace","key":"white_dress_set",'
        '"tags":["white sundress","lace trim"]},'
        '"environment":{"mode":"keep","key":"","tags":[]}}}'
    )
    generator = _SequenceTextGenerator(["好的，我来画一个女孩", valid])
    stream_id = "malformed-json-repair-test"
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id=stream_id,
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )

    try:
        result = asyncio.run(workflow.generate(
            "一女穿白裙站着",
            allow_inherit=True,
            use_visual_continuity=True,
        ))
    finally:
        session_state.clear_transient_generation_state(stream_id)

    assert result is not None
    assert "white sundress" in result.text
    assert len(generator.prompts) == 2
    assert "visual_continuity_repair" in generator.prompts[1]
    assert "JSON" in generator.prompts[1]


def test_workflow_gives_up_after_bounded_repair_attempts() -> None:
    """修复重试有上界（首次 + 2 次修复），持续失败时拒绝发送。"""

    generator = _FakeTextGenerator("始终不是 JSON 的回复")
    stream_id = "bounded-repair-attempts-test"
    workflow = PromptGenerationWorkflow(
        config={"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id=stream_id,
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )

    try:
        result = asyncio.run(workflow.generate(
            "一女站着",
            allow_inherit=True,
            use_visual_continuity=True,
        ))
    finally:
        session_state.clear_transient_generation_state(stream_id)

    assert result is None
    assert len(generator.prompts) == 3


def test_workflow_does_not_write_last_nai_context_before_delivery() -> None:
    """生成成功不等于出图成功：workflow 不再提前登记上一轮提示词。"""

    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["standing"],"emotion":[],"scene":[],"framing":["full body"]},'
        '"stable":{"outfit":{"mode":"replace","key":"white_dress_set",'
        '"tags":["white sundress","lace trim"]},'
        '"environment":{"mode":"keep","key":"","tags":[]}}}'
    )
    stream_id = "no-early-nai-context-commit"
    workflow = PromptGenerationWorkflow(
        config={
            "prompt_generator": {"output_format": "json", "inherit_ttl": 3_600},
            "tag_retriever": {},
        },
        stream_id=stream_id,
        text_generator=_FakeTextGenerator(response),
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
    )
    session_state.clear_transient_generation_state(stream_id)

    try:
        result = asyncio.run(workflow.generate(
            "一女穿白裙站着",
            allow_inherit=True,
            use_visual_continuity=True,
        ))

        assert result is not None
        assert session_state.get_last_nai_context(stream_id, ttl=3_600) == (None, None)
    finally:
        session_state.clear_transient_generation_state(stream_id)
