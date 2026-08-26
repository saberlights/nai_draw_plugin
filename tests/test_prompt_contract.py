"""提示词契约测试：保证规则不丢失、不自相矛盾、不无声膨胀。

这些断言的存在理由：提示词是纯文本，重构时没有编译器和 AST 兜底，
任何一次改写都可能悄悄删掉一条实战踩坑换来的约束，或让同一条规则
在两个规则块里漂移成互相矛盾的两个版本。互动绑定就是这样漂移过一次，
而且没有任何测试发现。
"""

import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.rules.prompt_rules import (
    INTERACTION_BINDING_RULES,
    PROMPT_GENERATOR_JSON_TEMPLATE,
    PROMPT_GENERATOR_TEMPLATE,
    SFW_PROMPT_GENERATOR_JSON_TEMPLATE,
    SFW_PROMPT_GENERATOR_TEMPLATE,
    SFW_VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
    VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
)
from plugins.nai_draw_plugin.core.services.visual_continuity import (
    _interaction_binding_failure,
)


ALL_TEMPLATES = {
    "text_sfw": SFW_PROMPT_GENERATOR_TEMPLATE,
    "text_nsfw": PROMPT_GENERATOR_TEMPLATE,
    "json_sfw": SFW_PROMPT_GENERATOR_JSON_TEMPLATE,
    "json_nsfw": PROMPT_GENERATOR_JSON_TEMPLATE,
    "continuity_sfw": SFW_VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
    "continuity_nsfw": VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
}


# ==================== 示例必须通过程序自己的绑定校验 ====================


def _extract_multi_person_examples(template: str) -> list[list[list[str]]]:
    """把模板示例里连续的 `charN:` 行按示例分组，还原成 people_tags 结构。

    resolve 的校验只看人物段的 tag 列表，所以这里把每个示例的 charN 行
    拆成 [[人物1 tag...], [人物2 tag...]]，可以直接喂给真实校验函数。
    """

    examples: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in template.splitlines():
        line = raw_line.strip()
        match = re.match(r"^char(\d+)\s*:(.*)$", line)
        if not match:
            if current:
                examples.append(current)
                current = []
            continue
        tags = [tag.strip() for tag in match.group(2).split(",") if tag.strip()]
        current.append(tags)
    if current:
        examples.append(current)
    return examples


def test_every_multi_person_example_passes_the_real_binding_validator() -> None:
    """模板示例是 LLM 最强的模仿对象，必须能通过 resolve 的硬校验。

    历史 bug：例 N2/N3 用 `source#groping` 配 `target#groped`，
    程序按逐字同名校验，一律判 interaction_unpaired，触发修复重试。
    """

    checked = 0
    for name, template in ALL_TEMPLATES.items():
        for index, people_tags in enumerate(_extract_multi_person_examples(template)):
            if len(people_tags) < 2:
                continue
            failure = _interaction_binding_failure(people_tags)
            assert not failure, (
                f"{name} 的第 {index + 1} 个多人示例无法通过绑定校验："
                f"{failure} -> {people_tags}"
            )
            checked += 1
    # 防止解析失效导致断言空转
    assert checked >= 4, f"只解析到 {checked} 个多人示例，解析逻辑可能已失效"


def test_no_template_still_demands_mismatched_participles() -> None:
    """旧的「施动方现在分词 / 接收方过去分词」要求与校验器直接冲突，必须彻底消失。"""

    forbidden_phrases = (
        "source 用现在分词",
        "target 用过去分词",
        "动作发出者（现在分词形式",
        "动作接受者（过去分词形式",
        "被动方全部 `target#` + 过去分词",
    )
    for name, template in ALL_TEMPLATES.items():
        for phrase in forbidden_phrases:
            assert phrase not in template, f"{name} 仍在要求分词不一致的绑定写法：{phrase}"


def test_binding_contract_is_stated_exactly_once_per_template() -> None:
    """绑定铁律只允许定义一处，多处复述就是下一次规则漂移的入口。"""

    for name, template in ALL_TEMPLATES.items():
        # 闭合标签只在块定义处出现；开标签形式还会被其它规则块当引用提及
        assert template.count("</interaction_binding>") == 1, f"{name} 缺少或重复绑定规则块"
        assert template.count("两侧必须是同一个 Danbooru tag，只有前缀不同") == 1, (
            f"{name} 的绑定铁律被复述了多次"
        )


def test_interaction_binding_block_keeps_all_load_bearing_clauses() -> None:
    """逐条锁住绑定块里每一项来自实战的约束。"""

    required = (
        "`A → B：动作`",
        "`A ↔ B：动作`",
        "`A：动作`",
        "两侧必须是同一个 Danbooru tag，只有前缀不同",
        "严禁施动方用现在分词、接收方用过去分词",
        "`source#hugging` / `target#hugged`",
        "`source#hug` / `target#hug`",
        "禁止为了凑“被动形态”编造",
        "`target#hand under skirt` 错误",
        "前缀内部禁止逗号、撇号、下划线",
        "的自身动作**禁止**加任何绑定前缀",
        "禁用 `mutual#`",
        "`uncomfortable / crying / struggling`",
        "`forcing / pinning`",
        "接收方除了 `target#` 互动本体，还要照常写出可见反应的裸 tag",
        "Bot 女生作为接收方时这些反应不能丢失",
        "没有第二个人物时不得凭空补出接收方",
        "程序会校验绑定配对",
        "不要把这类描述压缩成“亲密互动”",
    )
    for clause in required:
        assert clause in INTERACTION_BINDING_RULES, f"绑定块丢失约束：{clause}"


def test_multi_person_block_defers_binding_to_the_shared_contract() -> None:
    """多人块只负责落位，不再自己重述语法。"""

    for name, template in ALL_TEMPLATES.items():
        assert "施动方和接收方必须落在**不同**的人物段里" in template, f"{name} 丢失落位约束"
