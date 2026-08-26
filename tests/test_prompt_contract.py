"""提示词契约测试：保证规则不丢失、不自相矛盾、不无声膨胀。

这些断言的存在理由：提示词是纯文本，重构时没有编译器和 AST 兜底，
任何一次改写都可能悄悄删掉一条实战踩坑换来的约束，或让同一条规则
在两个规则块里漂移成互相矛盾的两个版本。互动绑定就是这样漂移过一次，
而且没有任何测试发现。
"""

import json
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
    resolve_visual_continuity,
)
from plugins.nai_draw_plugin.core.utils.prompt_output_parser import (
    extract_multi_character_payload,
    parse_prompt_from_structured_output,
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


# ==================== 形状示例必须能被程序真实解析 ====================


def _extract_json_objects(template: str) -> list[tuple[str, dict]]:
    """抽出模板里可解析的 JSON 形状示例（跳过带 `[...]` 占位符的 schema 行）。"""

    found: list[tuple[str, dict]] = []
    for match in re.finditer(r'\{"version".*\}', template):
        raw = match.group(0)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        found.append((raw, parsed))
    return found


def test_v3_shape_examples_round_trip_through_the_real_parser() -> None:
    """JSON 路径的形状示例必须能被 parse_prompt_from_structured_output 渲染出提示词。"""

    checked = 0
    for name in ("json_sfw", "json_nsfw"):
        for raw, parsed in _extract_json_objects(ALL_TEMPLATES[name]):
            if parsed.get("version") != 3:
                continue
            rendered = parse_prompt_from_structured_output(raw)
            assert rendered, f"{name} 的 v3 示例无法渲染：{raw[:80]}"
            assert rendered.startswith("rating:"), f"{name} 的 v3 示例首 tag 不是 rating：{rendered[:60]}"
            if parsed.get("format") == "multi":
                payload = extract_multi_character_payload(raw)
                assert payload, f"{name} 的多人示例进不了 characters[] 通道"
                assert len(payload["characters"]) == 2
            checked += 1
    assert checked >= 4, f"只校验到 {checked} 个 v3 形状示例，抽取逻辑可能已失效"


def test_v4_shape_examples_pass_the_real_continuity_resolver() -> None:
    """连续性路径的形状示例必须能过 resolve，且 keep 示例要真的复用上一帧缓存。"""

    checked = 0
    for name in ("continuity_sfw", "continuity_nsfw"):
        previous = None
        examples = [
            (raw, parsed)
            for raw, parsed in _extract_json_objects(ALL_TEMPLATES[name])
            if parsed.get("version") == 4
        ]
        assert len(examples) >= 2, f"{name} 需要同时演示 replace 首帧与 keep 后续帧"
        for raw, _parsed in examples:
            result = resolve_visual_continuity(
                raw,
                previous=previous,
                include_outfit=True,
                directives=None,
                stable_change_text="",
            )
            assert not result.failure, f"{name} 的 v4 示例被 resolve 打回：{result.failure}"
            assert result.prompt, f"{name} 的 v4 示例没有产出提示词"
            previous = result.stable
            checked += 1
        # keep 示例必须沿用首帧建立的服装，而不是把稳定区清空
        assert "cream ribbed knit pajama set" in result.prompt, (
            f"{name} 的 keep 示例没有复用首帧服装缓存"
        )
    assert checked >= 4, f"只校验到 {checked} 个 v4 形状示例"


# ==================== 按路径裁剪：该发的发到，不该发的不发 ====================


def test_text_format_block_only_reaches_the_text_channel() -> None:
    """JSON 路径由程序渲染 charN 行，喂文本格式块会诱导 LLM 直接吐 charN 文本。"""

    text_fence = "char1:[人物1 tag],"
    for name in ("text_sfw", "text_nsfw"):
        assert text_fence in ALL_TEMPLATES[name], f"{name} 丢失文本输出格式块"
    for name in ("json_sfw", "json_nsfw", "continuity_sfw", "continuity_nsfw"):
        assert text_fence not in ALL_TEMPLATES[name], f"{name} 仍在教 LLM 直接输出 charN 文本"


def test_positions_rules_only_reach_json_channels() -> None:
    """positions 只有 JSON 通道能承载（文本反解路径 has_coords 恒为 False）。"""

    for name in ("json_sfw", "json_nsfw", "continuity_sfw", "continuity_nsfw"):
        assert "多人坐标（positions，可选）" in ALL_TEMPLATES[name], f"{name} 丢失坐标规则"
        assert "元素只能是 `[A-E][1-5]` 字符串字面量" in ALL_TEMPLATES[name]
    for name in ("text_sfw", "text_nsfw"):
        assert "多人坐标（positions，可选）" not in ALL_TEMPLATES[name], (
            f"{name} 是文本通道，positions 永远不会被解析"
        )


def test_outfit_quota_and_continuity_rule_never_coexist() -> None:
    """一次性配额与跨图完整穿搭互斥，必须二选一，而不是先给再覆盖。"""

    one_shot = "适度：默认 1-2 个关键服装词"
    continuity = "首次建立时必须输出完整、具体、可见的整套穿搭"
    for name in ("text_sfw", "text_nsfw", "json_sfw", "json_nsfw"):
        template = ALL_TEMPLATES[name]
        assert one_shot in template, f"{name} 丢失一次性出图的服装配额"
        assert continuity not in template
    for name in ("continuity_sfw", "continuity_nsfw"):
        template = ALL_TEMPLATES[name]
        assert continuity in template, f"{name} 丢失稳定区完整穿搭要求"
        assert one_shot not in template, f"{name} 仍带互相矛盾的一次性配额"
        # 旧实现靠模板末尾追加 priority 块反向覆盖，规则冲突时 LLM 的取舍不可控
        assert "稳定区规则优先于上方通用的" not in template, (
            f"{name} 仍在用自我覆盖块打补丁"
        )


def test_examples_declare_themselves_as_tag_content_not_wire_format() -> None:
    """示例的逗号串外形曾与 JSON 输出契约直接冲突，必须显式声明只演示 tag 内容。"""

    for name, template in ALL_TEMPLATES.items():
        assert "最终输出格式一律以 `<output_instruction>` 为准" in template, (
            f"{name} 的示例没有声明格式归属"
        )
