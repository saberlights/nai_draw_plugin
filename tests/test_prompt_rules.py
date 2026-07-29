import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.rules.prompt_rules import (
    PROMPT_GENERATOR_JSON_TEMPLATE,
    SFW_PROMPT_GENERATOR_JSON_TEMPLATE,
)


def test_prompt_templates_require_a_leading_rating_tag() -> None:
    for template in (SFW_PROMPT_GENERATOR_JSON_TEMPLATE, PROMPT_GENERATOR_JSON_TEMPLATE):
        assert "`global[0]` 必须是且只能是一个 rating tag" in template
        assert "`rating:general`" in template
        assert "`rating:explicit`" in template


def test_prompt_templates_use_nai_weight_and_character_tag_rules() -> None:
    assert "合法范围是 -10 到 10" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "禁止 Stable Diffusion 语法" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "`aris_(blue_archive)`" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "`alternate_costume`" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "视角减法" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "被自拍/肖像规则识别为 bot 本人图片" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "LLM 不得生成、复述或从上下文延续任何发色、发型或瞳色 tag" in PROMPT_GENERATOR_JSON_TEMPLATE
