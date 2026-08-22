import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.rules.prompt_rules import (
    PROMPT_GENERATOR_JSON_TEMPLATE,
    SFW_PROMPT_GENERATOR_JSON_TEMPLATE,
    VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
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


def test_prompt_templates_preserve_bot_passive_state_and_actor_target_roles() -> None:
    for template in (SFW_PROMPT_GENERATOR_JSON_TEMPLATE, PROMPT_GENERATOR_JSON_TEMPLATE):
        assert "`A → B：动作`" in template
        assert "`A ↔ B：动作`" in template
        assert "`A：动作`" in template
        assert "接收方除了 `target#` 互动本体，还要照常写出可见反应的裸 tag" in template
        assert "Bot 女生作为接收方时这些反应不能丢失" in template


def test_prompt_templates_require_identical_tag_on_both_sides_of_interaction() -> None:
    """NAI 只在两侧挂同名 action tag 时才绑定互动；分词不一致会让方向随机化。"""

    for template in (SFW_PROMPT_GENERATOR_JSON_TEMPLATE, PROMPT_GENERATOR_JSON_TEMPLATE):
        assert "两侧必须是同一个 Danbooru tag，只有前缀不同" in template
        assert "`source#hug`，B 的人物段写 `target#hug`" in template
        assert "严禁施动方用现在分词、接收方用过去分词" in template
        assert "`source#hugging` / `target#hugged`" in template
        assert "禁止为了凑“被动形态”编造" in template
        # 自身动作误加前缀会被 NAI 绑到另一个角色，画成“对方在做”
        assert "的自身动作**禁止**加任何绑定前缀" in template
        assert "程序会校验绑定配对" in template


def test_nsfw_template_binds_sex_act_tags_with_identical_names() -> None:
    assert "施动方 `source#paizuri`，接收方 `target#paizuri`" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "禁止编造 `paizured`" in PROMPT_GENERATOR_JSON_TEMPLATE
    assert "只作用于自身的动作不是互动" in PROMPT_GENERATOR_JSON_TEMPLATE


def test_prompt_templates_count_visible_interaction_participants() -> None:
    for template in (SFW_PROMPT_GENERATOR_JSON_TEMPLATE, PROMPT_GENERATOR_JSON_TEMPLATE):
        assert "必须按实际人数输出（例如 `1boy 1girl`）" in template
        assert "不得把实际互动对象压成单人" in template


def test_prompt_templates_apply_visual_plan_as_single_frame_director_note() -> None:
    for template in (SFW_PROMPT_GENERATOR_JSON_TEMPLATE, PROMPT_GENERATOR_JSON_TEMPLATE):
        assert "单帧视觉导演与可见性自检" in template
        assert "must_show" in template
        assert "exclude" in template
        assert "静态截面" in template
        assert "禁止原样输出到英文 tag" in template
        assert "细节预算重新分配" in template
        assert "禁止为了补数量编造新元素" in template
        assert "字段位置可能串位" in template
        assert "按内容的真实语义重新归类" in template
        assert "服装和环境变化控制字段优先级最高" in template
        assert "同一物理地点的昼夜" in template
        assert "气味、清新感" in template
        assert "镜头裁切、遮挡和 `exclude` 只做本帧可见性减法" in template
        assert "emotion` 必须能从当前可见部位验证" in template
        assert "自拍/POV/第三视角按拍摄者判定" in template
        assert "`clear` 只表示事实已解除" in template


def test_nsfw_templates_add_adult_visual_appeal_without_escalating_content() -> None:
    for template in (
        PROMPT_GENERATOR_JSON_TEMPLATE,
        VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
    ):
        assert "成人向视觉吸引力（仅限明确成年角色）" in template
        assert "已有服装或连续性状态已锁定时" in template
        assert "身体与生理反馈" in template
        assert "不得把暧昧/内衣画面擅自升级为露点" in template
        assert "不得为凑 Tag 数编造新元素" in template
        assert "乳交就是乳交，口交就是口交，足交就是足交" in template
        assert "不得降级成“亲密互动”“取悦”“贴近”" in template
        assert "paizuri" in template
        assert "fellatio" in template
        assert "cunnilingus" in template
        assert "footjob" in template
        assert "不得新增上下文没有建立的性行为" in template


def test_sfw_template_does_not_receive_adult_visual_appeal_rules() -> None:
    assert "成人向视觉吸引力" not in SFW_PROMPT_GENERATOR_JSON_TEMPLATE
    assert "肩带滑落" not in SFW_PROMPT_GENERATOR_JSON_TEMPLATE
    assert "乳交就是乳交" not in SFW_PROMPT_GENERATOR_JSON_TEMPLATE
    assert "paizuri" not in SFW_PROMPT_GENERATOR_JSON_TEMPLATE


def test_visual_continuity_template_models_effects_and_soft_camera_variation() -> None:
    template = VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE

    assert '"effects":{"mode":"keep|replace|clear"' in template
    assert "subject 放 Bot 人物" in template
    assert "scene 放已经留在桌椅地面" in template
    assert "上一张实际镜头仅用于柔性变化" in template
    assert "不得为了不同而不同" in template
    assert "脱下一件或多件衣物也属于服装集合变化" in template
    assert "当前仍穿在身上的完整服装集合" in template
    assert "全部衣物都已脱下" in template
    assert "脱下后留在床上、椅上或地面的衣物" in template
