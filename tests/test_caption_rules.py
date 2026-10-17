# -*- coding: utf-8 -*-
"""LLM 反推提示词契约的单元测试。

这个提示词是整个 llm 反推路径的产品本体：覆盖清单少一项，用户就永远拿不到那一类细节，
而且不会报错——只会得到一段"看着挺详细"但漏了配饰和四肢落点的描述。所以逐条锁死。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.rules.caption_rules import resolve_caption_prompt


def _flat(sfw: bool) -> str:
    """把提示词的硬换行压平成单空格，断言只锁短语不锁排版。"""
    return re.sub(r"\s+", " ", resolve_caption_prompt(sfw=sfw))


# 用户明确要求"身体、衣服、配饰、动作、姿态、背景等等的每一个细节，以及它们的状态和位置"，
# 外加"小眼神小动作"。每一项在提示词里都必须有对应的英文指令段。
_REQUIRED_COVERAGE_TOPICS = (
    "Subjects",       # 人数与相对位置
    "Face",           # 脸/眼神/微表情
    "Hair",           # 发型与每缕头发的落点
    "Clothing",       # 逐件服装与穿着状态
    "Legwear",        # 腿部服饰独立成项（袜口构造 / 蝴蝶结 / 飘带）
    "Footwear",       # 鞋履独立成项
    "Accessories",    # 配饰与佩戴位置
    "Pose and action",       # 四肢逐一
    "Contact and occlusion",  # 接触/承重/遮挡
    "Background and environment",
    "Lighting and colour",
    "Camera and composition",
    "Style and medium",
)


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_contains_every_coverage_topic(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)
    for topic in _REQUIRED_COVERAGE_TOPICS:
        assert topic in prompt, f"覆盖清单缺少 {topic}"


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_demands_state_and_position_for_each_item(sfw: bool) -> None:
    """"状态 + 位置"是用户的原始要求，不能退化成"描述一下画面"。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "What state it is in right now" in prompt
    assert "its place on the body" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_demands_limb_by_limb_pose(sfw: bool) -> None:
    """四肢逐一是重绘可还原的关键；泛泛说 pose 会让模型只写一句 standing。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "Left arm, right arm, left leg, right leg" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_demands_micro_expressions_and_gestures(sfw: bool) -> None:
    """"小眼神小动作"：视线方向与细微手势必须被点名，否则模型一定会略过。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "where the gaze points" in prompt
    assert "sidelong glance" in prompt
    assert "small gestures" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_enforces_english_only_output(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "English only" in prompt
    assert "Never emit Chinese" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_forbids_tag_style_output(sfw: bool) -> None:
    """输出形态定的是纯自然语言；放任模型吐 tag 会让结果退化成第二个 WD14。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "Do NOT write Danbooru tags" in prompt
    assert "`1girl`" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_forbids_hedging_and_invention(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)
    for hedge in ("seems", "appears to be", "possibly", "it looks like"):
        assert hedge in prompt, f"未禁止模糊措辞 {hedge}"
    assert "Never invent backstory" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_length_floors_are_tiered_by_scene_complexity(sfw: bool) -> None:
    """单一固定下限会被模型当成目标——实测就是写到 231 词（下限 200）准时收笔。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "at least 250" in prompt
    assert "at least 450" in prompt
    assert "at least 600" in prompt
    assert "never as a target" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_length_has_no_upper_limit_and_is_derived_from_item_count(sfw: bool) -> None:
    """长度必须是 item 数推出来的派生量，且明确无上限，否则又换成一道天花板。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "There is no upper limit" in prompt
    assert "Length is not something you decide up front" in prompt
    assert "Never merge or drop an item to keep the text short" in _flat(sfw)


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_flags_hugging_the_floor_as_a_symptom(sfw: bool) -> None:
    """给模型一个可自查的信号：贴着下限 = 概括了，回去展开。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    assert "sitting near the floor" in prompt
    assert "proof you summarised" in prompt


def test_sfw_prompt_asks_for_a_safe_but_complete_description() -> None:
    prompt = resolve_caption_prompt(sfw=True)
    assert "non-sexual" in prompt
    assert "ALWAYS output a full description" in prompt
    assert "disable_ethics_and_boundaries" not in prompt


def test_nsfw_prompt_unlocks_explicit_description() -> None:
    """NSFW 图不解锁的话，视觉模型会用委婉语把衣物状态糊过去，反推等于白做。"""
    prompt = resolve_caption_prompt(sfw=False)
    assert "disable_ethics_and_boundaries" in prompt
    assert "never replace a body part or a clothing state with a euphemism" in prompt


def test_custom_template_replaces_builtin_entirely() -> None:
    custom = "Just describe the picture."
    assert resolve_caption_prompt(sfw=True, custom_template=custom) == custom
    assert resolve_caption_prompt(sfw=False, custom_template="  " + custom + "  ") == custom


def test_blank_custom_template_falls_back_to_builtin() -> None:
    builtin = resolve_caption_prompt(sfw=True)
    assert resolve_caption_prompt(sfw=True, custom_template="   \n ") == builtin
    assert resolve_caption_prompt(sfw=True, custom_template="") == builtin


@pytest.mark.parametrize("sfw", [True, False])
def test_prompt_itself_is_free_of_cjk(sfw: bool) -> None:
    """提示词正文写成中文会显著提高模型把描述也写成中文的概率，正文必须是纯英文。"""
    prompt = resolve_caption_prompt(sfw=sfw)
    cjk = re.findall(r"[一-鿿぀-ヿ]", prompt)
    assert not cjk, f"提示词正文混入 CJK 字符: {''.join(cjk[:20])}"


# ── item 铁律：本次优化的核心 ──────────────────────────────────────────────
#
# 背景：实测一张复杂立绘只产出 231 词，袜子被压成 `white lace-trimmed knee-high socks`
# 五个词，全图 8 处以上的蝴蝶结只写了尾巴那一处。根因是提示词曾经只有一份"话题清单"，
# 模型对每个话题写一个从句就认为讲完了。下面这组断言锁住"每件东西各自独立成项"这条铁律。


@pytest.mark.parametrize("sfw", [True, False])
def test_item_rule_requires_all_five_facts_per_item(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "inventory of individual things, not as a list of topics" in prompt
    assert "Never merge two" in prompt
    for fact in (
        "What it is",
        "Where it is",
        "What state it is in right now",
        "Its material, colour, pattern, sheen, and transparency",
        "Every decoration attached to it, counted and individually located",
    ):
        assert fact in prompt, f"item 五要素缺少：{fact}"


@pytest.mark.parametrize("sfw", [True, False])
def test_item_rule_recurses_into_decorations(sfw: bool) -> None:
    """递归条款是接住"袜口蝴蝶结"的那一条：装饰本身也是一件独立的东西。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "A decoration attached to an item is itself an item" in prompt
    # 直接用袜子上的蝴蝶结当范例，避免这条被后人改成抽象空话
    assert "A bow on a sock is not" in prompt
    assert "which leg and which side of that leg it sits on" in prompt
    assert "how far down those tails hang" in prompt
    assert "Recurse until nothing is left unnamed" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_item_rule_forces_counting_repeated_decorations(sfw: bool) -> None:
    """重复小装饰只提一次就走，正是袜子/上衣/鞋上的蝴蝶结集体消失的原因。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Repeated decorations are counted, never generalised" in prompt
    assert "Count how many there are and give each" in prompt
    # 反例 / 正例成对给出，模型才知道界线在哪
    assert '"Decorated with bows" is a failure' in prompt
    for decoration in ("bows", "ribbons", "frills", "lace", "charms", "beads", "buckles"):
        assert decoration in prompt, f"重复装饰清单缺少 {decoration}"


@pytest.mark.parametrize("sfw", [True, False])
def test_item_rule_splits_paired_items_left_and_right(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Paired items are described separately" in prompt
    assert "describe left and right individually" in prompt
    assert "If they match, say so explicitly" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_item_rule_bans_bare_category_nouns(sfw: bool) -> None:
    """`shoes` / `accessories` 这种光杆名词是最典型的"看着写了其实没写"。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Bare category nouns are not descriptions" in prompt
    for bare in ('"shoes"', '"accessories"', '"decorations"', '"some bows"'):
        assert bare in prompt, f"裸类名反例缺少 {bare}"


@pytest.mark.parametrize("sfw", [True, False])
def test_item_rule_requires_inventory_before_writing(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Before you write, sweep the image and settle the full inventory" in prompt
    assert "Never switch to summarising partway through" in prompt


# ── 腿部服饰 / 鞋履：从服装项的并列里提出来 ────────────────────────────────


@pytest.mark.parametrize("sfw", [True, False])
def test_legwear_is_its_own_station_with_construction_detail(sfw: bool) -> None:
    """袜子曾被埋在服装项十几个并列名词中间，只配得到一个形容词。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Legwear — its own station, not a trailing word in item 4" in prompt
    # 袜口构造是实测漏掉的第一层
    for cuff in ("lace band", "ruffle or frill cuff", "ribbed welt", "elastic", "scalloped edge"):
        assert cuff in prompt, f"袜口构造清单缺少 {cuff}"
    # 蝴蝶结与飘带是漏掉的第二层
    assert "which leg, which side of that leg" in _flat(sfw)
    assert "how far down they hang" in prompt
    # 光泽 / 堆叠 / 左右差异
    assert "wrinkles or bunching at the ankle" in prompt
    assert "whether the left and right differ at all" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_footwear_is_its_own_station(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Footwear — also its own station" in prompt
    assert "patent, matte, suede" in prompt
    assert "heel shape and height" in prompt
    assert "every decoration on each shoe described per shoe" in _flat(sfw)


# ── 服装状态 / 配饰扫街 / 文字转写 / 前景层 ────────────────────────────────


@pytest.mark.parametrize("sfw", [True, False])
def test_clothing_state_is_mandatory_and_open_is_not_a_state(sfw: bool) -> None:
    """实测把"从两肩滑落挂在上臂"的开衫写成了 `open`——状态必须是强制项。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert 'even if the answer is "worn normally"' in prompt
    assert '"Open" on its own is not a state' in prompt
    assert "slipped off one or both shoulders and caught" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_accessories_sweep_names_the_commonly_missed_four(sfw: bool) -> None:
    """耳坠 / 独立吊坠 / 戒指 / 挂饰是实测整类消失的四种，必须点名。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "head, ears, neck, chest, wrists, fingers, waist, hips, legs, feet" in prompt
    assert "pendants hanging separately from a choker" in prompt
    assert "charms dangling from a held" in prompt
    assert "most commonly missed" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_background_requires_verbatim_text_transcription(sfw: bool) -> None:
    """招牌上的 "KOKORO Bakery Cafe" 实测完全没被转写。"""
    flat = _flat(sfw)

    assert "Transcribe any legible text verbatim in straight ASCII double quotes" in flat
    assert "if text is cropped or too small to read, say that instead of guessing" in flat


# ── 画面文字必须用英文半角引号包起来 ──────────────────────────────────────
#
# 反推结果会被直接回灌 /nai0 变成 NAI prompt，而 NAI 是按半角双引号识别"照这几个字母画"的。
# 模型默认爱用弯引号 “ ”，弯引号在下游就是一串普通字符，画出来的字会糊成随机字形。


@pytest.mark.parametrize("sfw", [True, False])
def test_rendered_lettering_must_use_half_width_double_quotes(sfw: bool) -> None:
    flat = _flat(sfw)

    assert "wrapped in straight ASCII double quotes" in flat
    assert "The quotes must be the half-width" in flat
    # 逐个点名禁用的引号形态，只说"用半角"模型会自行发挥
    for banned in ("full-width", "curly", "single quotes", "corner brackets"):
        assert banned in flat, f"未禁止的引号形态：{banned}"


@pytest.mark.parametrize("sfw", [True, False])
def test_lettering_rule_lists_every_surface_text_can_appear_on(sfw: bool) -> None:
    """只说"招牌"的话，衣服印字和对话气泡会被当成不算数。"""
    flat = _flat(sfw)

    for surface in ("shop signs", "banners", "book pages", "screens", "print on clothing", "speech bubbles"):
        assert surface in flat, f"文字载体清单缺少：{surface}"


@pytest.mark.parametrize("sfw", [True, False])
def test_lettering_rule_quotes_only_the_letters(sfw: bool) -> None:
    """把整句描述一起圈进引号，NAI 会试着把"a banner that reads"也画成字。"""
    flat = _flat(sfw)

    assert "Quote only the lettering itself, never the surrounding description" in flat
    assert 'a banner that reads "SALE"' in flat


@pytest.mark.parametrize("sfw", [True, False])
def test_lettering_rule_forbids_guessing_unreadable_text(sfw: bool) -> None:
    flat = _flat(sfw)

    assert "cropped," in flat and "too small, or not in the Latin alphabet" in flat
    assert "instead of quoting a guess" in flat


@pytest.mark.parametrize("sfw", [True, False])
def test_background_walks_depth_including_blurred_foreground(sfw: bool) -> None:
    """左下角散景的花整层被跳过——虚化不等于可以不写。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Walk the depth explicitly, front to back" in prompt
    assert "a blurred foreground element still has to be described" in _flat(sfw)


@pytest.mark.parametrize("sfw", [True, False])
def test_lighting_requires_shadow_landing_surface(sfw: bool) -> None:
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "where the shadows fall, onto what surface" in prompt


# ── 收尾自检 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sfw", [True, False])
def test_final_check_covers_all_six_observed_failure_modes(sfw: bool) -> None:
    """六条自检逐一对应实测中真实出现过的漏法，少一条就少堵一个洞。"""
    prompt = _flat(sfw)

    assert "<final_check>" in prompt
    checks = (
        "bare category noun",                       # ① 裸类名
        "counted and individually located",         # ② 装饰逐个计数
        "socks, shoes, sleeves, ears, hands, earrings",  # ③ 成对左右分写
        "transcribed verbatim in straight ASCII double quotes",  # ④ 文字转写
        "half-width",                               # ④ 引号形态
        "blurred foreground layer described",       # ⑤ 前景层
        "how it sits on the body right now",        # ⑥ 服装状态
    )
    for check in checks:
        assert check in prompt, f"收尾自检缺少：{check}"


@pytest.mark.parametrize("sfw", [True, False])
def test_final_check_is_not_emitted_to_the_user(sfw: bool) -> None:
    """自检过程写进输出会污染直发 /nai0 的 prompt。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert "Do not output the check itself" in prompt
    assert "output only the final, repaired description" in prompt


@pytest.mark.parametrize("sfw", [True, False])
def test_final_check_is_the_last_block(sfw: bool) -> None:
    """模型对最后读到的指令依从性最高，自检必须压在正文末尾。"""
    prompt = resolve_caption_prompt(sfw=sfw)

    assert prompt.rstrip().endswith("</final_check>")
