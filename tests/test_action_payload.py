# -*- coding: utf-8 -*-
"""compose_description_from_action_payload 的回归测试。

历史 bug：旧版"5 字段非空就忽略 description"会丢失 description 里的角色锚点，
导致"画一张初音未来"被下游误判 bot 自拍。本组用例守护"description 永不被丢"。
"""

from __future__ import annotations

from plugins.nai_draw_plugin.core.utils.action_payload import (
    BOT_ABSENT_TOKEN,
    NAMED_CHARACTER_TOKEN,
    STRUCTURED_DESCRIPTION_FIELDS,
    VISUAL_PLAN_FIELDS,
    compose_description_from_action_payload,
    is_bot_absent_intent,
    is_named_character_intent,
    render_action_payload_for_prompt,
)


def test_keeps_description_when_structured_fields_also_present():
    """description 与结构化字段共存：两者都拼，description 在前作为语义主体。

    历史 bug 用例：Planner 给 5 个结构化字段 + description（含角色名）时，旧版会
    完全忽略 description，导致"初音未来"等核心锚点丢失。
    """
    action_data = {
        "subject_and_pov": "一女 第三视角",
        "action": "站立",
        "emotion": "俏皮 微笑",
        "scene_delta": "",
        "dynamic_scene": "舞台灯光",
        "framing": "特写",
        "description": "一女, 初音未来, 公式服, 葱色双马尾, 精致的面容, 灵动的眼神, 背景为舞台, 特写",
    }
    out = compose_description_from_action_payload(action_data)

    # description 必须在前（主体）+ 结构化字段在后（补充）
    assert out.startswith("一女, 初音未来, 公式服")
    # 结构化字段也要拼上来
    assert "一女 第三视角" in out
    assert "站立" in out
    assert "俏皮 微笑" in out
    assert "特写" in out
    assert "舞台灯光" in out
    # 关键：角色锚点不能丢
    assert "初音未来" in out


def test_falls_back_to_structured_only_when_description_missing():
    """description 字段空时仅用结构化字段，跟历史行为兼容。"""
    action_data = {
        "subject_and_pov": "一女 第一视角",
        "action": "微笑",
        "framing": "selfie",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "一女 第一视角 微笑 selfie"


def test_falls_back_to_description_only_when_structured_all_empty():
    """结构化字段全空时仅用 description。"""
    action_data = {
        "description": "初音未来, 公式服",
        "subject_and_pov": "",
        "action": "",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "初音未来, 公式服"


def test_both_empty_returns_empty_string():
    """description 与结构化字段都空 → 空串（调用方负责回落 reasoning）。"""
    assert compose_description_from_action_payload({}) == ""
    assert compose_description_from_action_payload({"description": "", "action": ""}) == ""


def test_structured_field_order_preserved():
    """结构化字段按 STRUCTURED_DESCRIPTION_FIELDS 声明顺序拼接。"""
    action_data = {
        "framing": "F",
        "action": "B",
        "subject_and_pov": "A",
        "emotion": "C",
        "scene_delta": "D",
        "dynamic_scene": "E",
    }
    out = compose_description_from_action_payload(action_data)
    # 顺序：subject_and_pov action emotion scene_delta dynamic_scene framing
    assert out == "A B C D E F"


def test_strips_whitespace_in_each_field():
    """各字段头尾空白被 strip，避免拼出 ``"  a   b"`` 这种脏数据。"""
    action_data = {
        "subject_and_pov": "  pov  ",
        "action": "\nstanding\n",
        "description": "  girl  ",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "girl pov standing"


def test_non_string_values_coerced_to_empty():
    """Planner 偶发把数字/None 塞进字段时不应抛，按空处理。"""
    action_data = {
        "subject_and_pov": None,
        "action": 0,
        "description": "1girl",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "1girl"


def test_structured_description_fields_constant_unchanged():
    """常量值锁死，避免顺序被无意打乱影响下游 NAI tag 排序约定。"""
    assert STRUCTURED_DESCRIPTION_FIELDS == (
        "subject_and_pov",
        "action",
        "emotion",
        "scene_delta",
        "dynamic_scene",
        "framing",
    )


def test_render_action_payload_for_prompt_preserves_field_boundaries():
    """Tag LLM 必须看见 Planner 字段名，不能再收到无边界关键词串。"""

    assert callable(render_action_payload_for_prompt)
    rendered = render_action_payload_for_prompt({
        "subject_and_pov": "一女 第三视角",
        "action": "坐着，揉腰",
        "emotion": "羞恼，脸红",
        "scene_delta": "淡青色丝绸旗袍；片场固定灯架和黑色遮光布",
        "dynamic_scene": "烟雾，聚光灯",
        "framing": "中景",
        "description": "",
    })

    assert rendered.startswith("<planner_visual_request>")
    assert '"action": "坐着，揉腰"' in rendered
    assert '"scene_delta": "淡青色丝绸旗袍；片场固定灯架和黑色遮光布"' in rendered
    assert '"dynamic_scene": "烟雾，聚光灯"' in rendered
    assert rendered.endswith("</planner_visual_request>")


def test_render_action_payload_keeps_visual_plan_separate_from_tag_fields():
    rendered = render_action_payload_for_prompt({
        "visual_plan": (
            "moment=女生刚把热茶递到男生手边；"
            "must_show=茶杯接触、女生回头微笑；exclude=上一秒的开门动作"
        ),
        "subject_and_pov": "一男一女 第三视角",
        "action": "女生递茶，男生接住",
    })

    assert '"visual_plan":' in rendered
    assert "must_show=茶杯接触" in rendered
    assert '"action": "女生递茶，男生接住"' in rendered


def test_visual_plan_is_not_part_of_flat_tag_request():
    action_data = {
        "visual_plan": "moment=递茶；must_show=茶杯；exclude=开门",
        "subject_and_pov": "一男一女 第三视角",
        "action": "女生递茶",
    }

    assert compose_description_from_action_payload(action_data) == "一男一女 第三视角 女生递茶"


def test_visual_plan_field_constant_is_stable():
    assert VISUAL_PLAN_FIELDS == ("visual_plan",)


def test_render_action_payload_omits_duplicate_flat_description():
    """结构化字段已经完整时，不再用同义扁平串稀释字段边界。"""

    action_data = {
        "subject_and_pov": "一女 第三视角",
        "action": "坐着，揉腰",
        "emotion": "疲惫",
        "scene_delta": "淡青色丝绸旗袍；片场固定灯架",
        "dynamic_scene": "烟雾",
        "framing": "中景",
    }
    flat = compose_description_from_action_payload(action_data)

    rendered = render_action_payload_for_prompt(
        action_data,
        effective_description=flat,
    )

    assert '"effective_description"' not in rendered
    assert '"scene_delta": "淡青色丝绸旗袍；片场固定灯架"' in rendered


def test_render_action_payload_includes_planner_stable_change_decisions():
    rendered = render_action_payload_for_prompt({
        "subject_and_pov": "一女 第三视角",
        "outfit_change": "replace: 淡青色丝绸旗袍，肉色丝袜，复古黑色皮鞋",
        "environment_change": "unchanged",
    })

    assert '"outfit_change": "replace: 淡青色丝绸旗袍，肉色丝袜，复古黑色皮鞋"' in rendered
    assert '"environment_change": "unchanged"' in rendered


def test_effect_directive_is_structured_but_only_description_reaches_tag_query():
    action_data = {
        "subject_and_pov": "一女 第三视角",
        "effect_change": "replace",
        "effect_new_state": "人物：头发和衬衫被雨淋湿；场景：折叠伞放在门边",
    }

    flattened = compose_description_from_action_payload(action_data)
    rendered = render_action_payload_for_prompt(action_data)

    assert "replace" not in flattened
    assert "头发和衬衫被雨淋湿" in flattened
    assert '"effect_change": "replace"' in rendered
    assert '"effect_new_state":' in rendered


# ============ is_named_character_intent ============

def test_named_character_intent_matches_when_token_present():
    """Planner 在 subject_and_pov 写了 ``画指定角色`` 前缀 → True。

    这是上游声明"本轮主体是指定角色，非 bot 出镜"的唯一信号。命中后 sdk_runtime
    会跳过 self-image 注入与 selfie 后处理，避免把指定角色洗成 bot 默认外貌。
    """
    assert is_named_character_intent({
        "subject_and_pov": f"{NAMED_CHARACTER_TOKEN} 一女 第三视角",
        "description": "初音未来, 葱色双马尾",
    }) is True


def test_named_character_intent_no_token_means_bot_self():
    """没有 token：Planner 把本轮视为 bot 自己出镜（含 cosplay 场景）→ False。

    Cosplay bot 出镜场景照常返回 False——出镜的还是 bot，只是穿成别的角色。
    """
    assert is_named_character_intent({
        "subject_and_pov": "一女",
        "description": "cosplay 初音未来",
    }) is False
    assert is_named_character_intent({"subject_and_pov": "一女 自拍"}) is False
    assert is_named_character_intent({}) is False


def test_named_character_intent_only_reads_subject_field():
    """token 只在 ``subject_and_pov`` 中识别——其它字段含同名字符串不算数。

    避免 description 里偶然出现"画指定角色"这种自然语言误触发；Planner 必须显式
    在 subject 字段声明意图，约定收口一处。
    """
    assert is_named_character_intent({
        "subject_and_pov": "一女",
        "description": f"{NAMED_CHARACTER_TOKEN} 初音未来",
    }) is False


def test_named_character_intent_token_anywhere_in_subject():
    """token 在 subject 内任意位置（前/中/后）都算命中。

    Planner 偶尔会把 token 放视角后（``一女 画指定角色`` 这种），不强制 token 在最前。
    """
    assert is_named_character_intent({
        "subject_and_pov": f"一女 {NAMED_CHARACTER_TOKEN} 第三视角",
    }) is True
    assert is_named_character_intent({
        "subject_and_pov": f"一女 第三视角 {NAMED_CHARACTER_TOKEN}",
    }) is True


def test_named_character_intent_non_string_subject_safe():
    """Planner 偶发 None/数字时不抛，按未声明处理。"""
    assert is_named_character_intent({"subject_and_pov": None}) is False
    assert is_named_character_intent({"subject_and_pov": 0}) is False


def test_bot_absent_intent_only_matches_explicit_subject_token():
    assert is_bot_absent_intent({
        "subject_and_pov": f"{BOT_ABSENT_TOKEN} 纯环境",
        "description": "空荡的书房",
    }) is True
    assert is_bot_absent_intent({
        "subject_and_pov": "纯环境",
        "description": f"文字里偶然提到{BOT_ABSENT_TOKEN}",
    }) is False
    assert is_bot_absent_intent({"subject_and_pov": None}) is False


def test_self_image_intent_token_does_not_reinject_whole_flat_blob():
    """上游追加 self-image token 时，只补差集，不把拍平串整段塞回去。

    回归历史缺陷：description 被追加 '肖像照' 后 effective != flattened，
    旧实现据此把全部字段拍平的长串写进 payload 且排在结构化字段之前，
    Tag LLM 第一手读到的就是边界已抹平、内容重复两遍的版本。
    """

    action_data = {
        "subject_and_pov": "一女 第三视角",
        "action": "女生：侧卧在沙发上；女生：单手拿着手机",
        "emotion": "温柔",
        "dynamic_scene": "午后柔和光线",
        "framing": "中景",
    }
    flat = compose_description_from_action_payload(action_data)

    rendered = render_action_payload_for_prompt(
        action_data,
        effective_description=f"{flat} 肖像照",
    )

    assert '"extra_request": "肖像照"' in rendered
    # 拍平串本体不得再出现：动作只能通过 action 字段抵达下游
    assert flat not in rendered
    assert '"action": "女生：侧卧在沙发上；女生：单手拿着手机"' in rendered
    assert rendered.count("侧卧在沙发上") == 1
    assert rendered.count("温柔") == 1


def test_self_image_intent_prefix_is_extracted_as_residual():
    action_data = {"action": "女生：靠在窗边", "emotion": "慵懒"}
    flat = compose_description_from_action_payload(action_data)

    rendered = render_action_payload_for_prompt(
        action_data,
        effective_description=f"一女 肖像照 {flat}",
    )

    assert '"extra_request": "一女 肖像照"' in rendered
    assert rendered.count("靠在窗边") == 1


def test_rewritten_effective_description_is_kept_whole():
    """定位不到结构化基线说明上游做了改写，宁可重复也不能丢锚点。"""

    rendered = render_action_payload_for_prompt(
        {"subject_and_pov": "一女 第三视角"},
        effective_description="初音未来 公式服 葱色双马尾",
    )

    assert '"extra_request": "初音未来 公式服 葱色双马尾"' in rendered


def test_visual_plan_leads_payload_and_free_text_trails_it():
    """字段顺序即阅读顺序：导演笔记在前，无边界自由文本垫底。"""

    rendered = render_action_payload_for_prompt(
        {
            "visual_plan": "moment=递茶",
            "subject_and_pov": "一男一女 第三视角",
            "action": "女生 → 男生：递茶",
            "description": "兜底原文",
        },
        effective_description="兜底原文 一男一女 第三视角 女生 → 男生：递茶 额外锚点",
    )

    order = [
        rendered.index('"visual_plan"'),
        rendered.index('"subject_and_pov"'),
        rendered.index('"action"'),
        rendered.index('"description"'),
        rendered.index('"extra_request"'),
    ]
    assert order == sorted(order)
