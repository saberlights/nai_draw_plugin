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

from plugins.nai_draw_plugin import plugin as plugin_module
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


# ==================== Planner 侧：合并后规则一条都不能少 ====================


def _planner_facing_text() -> str:
    """还原 Planner 实际看到的工具文本。

    SDK 的 _build_action_detailed_description 把 description、action_parameters
    和 action_require 折进同一份工具描述，所以「规则写在哪个入参里」不影响
    Planner 能不能读到；这里按同样的方式拼接三个常量，断言才不会被搬家动作骗过。
    """

    return "\n\n".join(
        (
            plugin_module._NAI_DRAW_DESCRIPTION,
            "\n".join(plugin_module._NAI_DRAW_ACTION_PARAMETERS.values()),
            "\n".join(plugin_module._NAI_DRAW_ACTION_REQUIRE),
        )
    )


# 被删除的 _VISUAL_DIRECTOR_POLICY 里每一条约束，与它现在的落点关键词
DELETED_VISUAL_DIRECTOR_CLAUSES = {
    "静态瞬间选择": "定格的静态瞬间",
    "不压缩整段剧情": "不要把整段剧情压成关键词",
    "must_show 数量": "2~4 个画面证据",
    "exclude 不改剧情": "不能用来偷偷改变剧情",
    "不凭空增加要素": "不要凭空增加人物、动作或道具",
    "导演笔记不外流": "不得复制到 action、scene_delta 或最终提示词",
    "moment 禁前因后果": "醒来→看到消息→拍照→发送",
    "提交前物理自检": "肢体动作是否物理可行",
    "互斥动作自检": "一只手不能同时完成互斥动作",
    "关系行三句型": "`A → B：动作`",
    "禁被字句与代词": "禁止'被'字句，禁止'他/她'代词",
    "自身动作不写箭头": "自身动作绝不能写成箭头句",
    "箭头句的绑定后果": "source#/target# 角色绑定",
    "字段路由-服装": "scene_delta=仅稳定服装和固定环境",
    "字段路由-动态场景": "dynamic_scene=临时物件/倒影/天气/时间/光线",
    "同地点光线不换环境": "同一物理地点的昼夜、天气和光线变化不是换环境",
    "framing 只填一个": "【只填一个景别或构图】",
    "视角只进 subject_and_pov": "framing 不得包含 POV/第三视角",
    "看不见不等于解除": "镜头看不见不等于事实解除",
    "脱衣必须 replace": "脱下一件或多件衣物也必须 outfit_change=replace",
    "脱衣后仍穿集合": "写当前仍穿着的完整集合",
    "全脱才 clear": "全部衣物都已脱下时才 clear",
    "脱下衣物归场景": "写入场景附着物",
    "掀起不算脱下": "只是掀起、拉开或局部移位则 outfit unchanged",
    "emotion 可留空": "emotion 留空",
    "不可见心理不出表情": "不要把'宠溺/幸福/起床气'直接变成 smile、blush",
    "拍摄者判定": "自拍/POV 判定看拍摄者",
    "性行为用原名": "乳交就写乳交、口交就写口交、足交就写足交",
    "不新增性行为": "不得自行升级或增加新的性行为",
    "睡醒效果归 effect": "刚睡醒造成的蓬松乱发",
    "身份匿名化": "先执行会话身份匿名化",
}


def test_deleted_visual_director_policy_lost_no_constraint() -> None:
    """删掉重复注入的协议块，但它的每一条约束都必须仍在 Planner 可读文本里。"""

    planner_text = _planner_facing_text()
    missing = [
        label
        for label, marker in DELETED_VISUAL_DIRECTOR_CLAUSES.items()
        if marker not in planner_text
    ]
    assert not missing, f"合并 _VISUAL_DIRECTOR_POLICY 时丢失了约束：{missing}"


def test_visual_director_policy_is_not_reinjected() -> None:
    """协议块与 action_require 落在同一段文本，重复注入只会互相稀释。"""

    assert not hasattr(plugin_module, "_VISUAL_DIRECTOR_POLICY")
    assert "单帧视觉导演协议" not in _planner_facing_text()


def test_identity_rule_appears_once_in_planner_text() -> None:
    """身份匿名化规则曾在 description 和 action_require 各写一份完整副本。"""

    planner_text = _planner_facing_text()
    rule = plugin_module._CHAT_PARTICIPANT_IDENTITY_RULE
    assert planner_text.count(rule) == 1, "身份匿名化规则被重复注入"
    # 星号条目仍要把它顶到高注意力区，只是改为指针而非整段复制
    assert "填写任何 action_data 字段前，先执行本工具描述开头的会话身份匿名化规则" in planner_text


def test_planner_visual_state_only_carries_runtime_values() -> None:
    """状态块只发本轮才知道的值；判定规则由 action_require 常驻，不再复述。"""

    class _Stable:
        effects = ("wet hair",)
        scene_effects = ("coat on chair",)
        last_framing = ("medium shot",)

    rendered = plugin_module._build_planner_visual_state(_Stable())
    assert "wet hair" in rendered
    assert "coat on chair" in rendered
    assert "medium shot" in rendered
    assert "effect_change" not in rendered, "状态块又开始复述 effect_change 判定规则"
    assert "must_show" not in rendered, "状态块又开始复述镜头柔性变化规则"
    # 规则本体仍必须在 Planner 常驻文本里
    planner_text = _planner_facing_text()
    assert "只延续剧情尚未解除的跨图临时效果" in planner_text
    assert "优先换一个合适景别或朝向" in planner_text
