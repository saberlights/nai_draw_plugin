import json
import random

from plugins.nai_draw_plugin.core.services.visual_continuity import (
    StableVisualTags,
    VisualChangeDirective,
    VisualTagCard,
    describe_visual_failure,
    parse_visual_change_directive,
    parse_visual_change_directives,
    resolve_visual_continuity,
)


def test_keep_reuses_exact_stable_tags_while_dynamic_tags_change() -> None:
    previous = StableVisualTags(
        outfit=(
            "navy blue ribbed knit cardigan",
            "ivory piping",
            "matte brass buttons",
        ),
        environment=(
            "walnut bookcase",
            "cream plaster wall",
            "sage green curtains",
        ),
    )
    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["standing"],"emotion":["smile"],"scene":[],'
        '"framing":["medium shot"]},"stable":{'
        '"outfit":{"mode":"keep","tags":["blue sweater","gold buttons"]},'
        '"environment":{"mode":"keep","tags":["wooden shelves","green drapes"]}}}'
    )

    result = resolve_visual_continuity(response, previous=previous)

    assert result.prompt == (
        "rating:general, solo, 1girl, navy blue ribbed knit cardigan, "
        "ivory piping, matte brass buttons, standing, smile, walnut bookcase, "
        "cream plaster wall, sage green curtains, medium shot"
    )
    assert result.stable == previous
    assert "blue sweater" not in result.prompt
    assert "wooden shelves" not in result.prompt


def test_replace_creates_new_stable_outfit_and_environment_tags() -> None:
    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["walking"],"emotion":["gentle smile"],'
        '"scene":["evening"],"framing":["full body"]},"stable":{'
        '"outfit":{"mode":"replace","key":"green_coat_set","tags":["forest green wool coat",'
        '"double-breasted coat","dark horn buttons"]},'
        '"environment":{"mode":"replace","key":"old_town_street","tags":["narrow stone street",'
        '"red brick storefront","black iron street lamps"]}}}'
    )

    result = resolve_visual_continuity(response)

    assert result.stable == StableVisualTags(
        outfit=(
            "forest green wool coat",
            "double-breasted coat",
            "dark horn buttons",
        ),
        environment=(
            "narrow stone street",
            "red brick storefront",
            "black iron street lamps",
        ),
        outfit_key="green_coat_set",
        environment_key="old_town_street",
        outfits=(
            VisualTagCard(
                "green_coat_set",
                (
                    "forest green wool coat",
                    "double-breasted coat",
                    "dark horn buttons",
                ),
            ),
        ),
        environments=(
            VisualTagCard(
                "old_town_street",
                (
                    "narrow stone street",
                    "red brick storefront",
                    "black iron street lamps",
                ),
            ),
        ),
    )
    assert result.prompt == (
        "rating:general, solo, 1girl, forest green wool coat, "
        "double-breasted coat, dark horn buttons, walking, gentle smile, "
        "narrow stone street, red brick storefront, black iron street lamps, "
        "evening, full body"
    )


def test_switch_restores_known_outfit_and_environment_without_regenerating_tags() -> None:
    home_outfit = VisualTagCard(
        "home_knit_set",
        ("navy ribbed cardigan", "ivory piping", "brass buttons"),
    )
    street_outfit = VisualTagCard(
        "green_coat_set",
        ("forest green wool coat", "double-breasted coat"),
    )
    study = VisualTagCard(
        "home_study",
        ("walnut bookcase", "cream wall", "sage curtains"),
    )
    street = VisualTagCard(
        "old_town_street",
        ("narrow stone street", "red brick storefront"),
    )
    previous = StableVisualTags(
        outfit=street_outfit.tags,
        environment=street.tags,
        outfit_key=street_outfit.key,
        environment_key=street.key,
        outfits=(home_outfit, street_outfit),
        environments=(study, street),
    )
    response = (
        '{"version":4,"format":"single","intent":"portrait",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["reading"],"emotion":["relaxed"],"scene":["night"],'
        '"framing":["medium shot"]},"stable":{'
        '"outfit":{"mode":"switch","key":"home_knit_set",'
        '"tags":["blue sweater"]},'
        '"environment":{"mode":"switch","key":"home_study",'
        '"tags":["bookshelf"]}}}'
    )

    result = resolve_visual_continuity(response, previous=previous)

    assert result.stable.outfit == home_outfit.tags
    assert result.stable.environment == study.tags
    assert result.stable.outfit_key == home_outfit.key
    assert result.stable.environment_key == study.key
    # switch 命中会把卡片 touch 到队尾（LRU 保护），内容集合不变
    assert set(result.stable.outfits) == set(previous.outfits)
    assert set(result.stable.environments) == set(previous.environments)
    assert result.stable.outfits[-1] == home_outfit
    assert result.stable.environments[-1] == study
    assert "blue sweater" not in result.prompt
    assert "bookshelf" not in result.prompt


def test_random_dynamic_variations_never_mutate_cached_stable_tags() -> None:
    rng = random.Random(20260729)
    previous = StableVisualTags(
        outfit=("charcoal linen jacket", "silver zipper", "white top"),
        environment=("oak desk", "white brick wall", "black desk lamp"),
    )
    actions = ["reading", "standing", "waving", "drinking tea"]
    emotions = ["smile", "serious", "surprised", "sleepy"]
    framings = ["close-up", "medium shot", "full body", "from side"]

    for _ in range(32):
        action = rng.choice(actions)
        emotion = rng.choice(emotions)
        framing = rng.choice(framings)
        response = json.dumps(
            {
                "version": 4,
                "format": "single",
                "intent": "portrait",
                "dynamic": {
                    "subject": ["rating:general", "solo", "1girl"],
                    "action": [action],
                    "emotion": [emotion],
                    "scene": [],
                    "framing": [framing],
                },
                "stable": {
                    "outfit": {"mode": "keep", "tags": ["dark coat"]},
                    "environment": {"mode": "keep", "tags": ["office"]},
                },
            }
        )

        result = resolve_visual_continuity(response, previous=previous)

        assert result.stable == previous
        assert action in result.prompt
        assert emotion in result.prompt
        assert framing in result.prompt
        assert "dark coat" not in result.prompt
        assert "office" not in result.prompt


def test_visual_tag_card_library_keeps_the_twelve_most_recent_designs() -> None:
    state = StableVisualTags()
    for index in range(13):
        response = json.dumps(
            {
                "version": 4,
                "format": "single",
                "intent": "portrait",
                "dynamic": {
                    "subject": ["rating:general", "solo", "1girl"],
                    "action": ["standing"],
                    "emotion": [],
                    "scene": [],
                    "framing": ["full body"],
                },
                "stable": {
                    "outfit": {
                        "mode": "replace",
                        "key": f"outfit_{index}",
                        "tags": [f"fabric design {index}"],
                    },
                    "environment": {"mode": "keep", "key": "", "tags": []},
                },
            }
        )
        state = resolve_visual_continuity(response, previous=state).stable

    assert len(state.outfits) == 12
    assert [card.key for card in state.outfits] == [
        f"outfit_{index}" for index in range(1, 13)
    ]
    assert state.outfit_key == "outfit_12"

    evicted_switch = json.dumps(
        {
            "version": 4,
            "format": "single",
            "intent": "portrait",
            "dynamic": {
                "subject": ["rating:general", "solo", "1girl"],
                "action": [],
                "emotion": [],
                "scene": [],
                "framing": [],
            },
            "stable": {
                "outfit": {"mode": "switch", "key": "outfit_0", "tags": []},
                "environment": {"mode": "keep", "key": "", "tags": []},
            },
        }
    )
    result = resolve_visual_continuity(evicted_switch, previous=state)

    assert result.stable.outfit_key == "outfit_12"
    assert result.stable.outfit == ("fabric design 12",)


def test_environment_only_frame_omits_outfit_without_mutating_current_outfit() -> None:
    previous = StableVisualTags(
        outfit=("navy ribbed cardigan", "ivory piping"),
        environment=("hallway",),
        outfit_key="home_knit_set",
        environment_key="hallway",
        outfits=(
            VisualTagCard(
                "home_knit_set",
                ("navy ribbed cardigan", "ivory piping"),
            ),
        ),
        environments=(VisualTagCard("hallway", ("hallway",)),),
    )
    response = json.dumps(
        {
            "version": 4,
            "format": "single",
            "intent": "normal",
            "dynamic": {
                "subject": ["rating:general", "no humans"],
                "action": [],
                "emotion": [],
                "scene": ["rain outside"],
                "framing": ["wide shot"],
            },
            "stable": {
                "outfit": {"mode": "clear", "key": "", "tags": []},
                "environment": {
                    "mode": "replace",
                    "key": "home_kitchen",
                    "tags": ["oak cabinets", "white tile backsplash"],
                },
            },
        }
    )

    result = resolve_visual_continuity(
        response,
        previous=previous,
        include_outfit=False,
    )

    assert result.stable.outfit == previous.outfit
    assert result.stable.outfit_key == previous.outfit_key
    assert result.stable.outfits == previous.outfits
    assert "navy ribbed cardigan" not in result.prompt
    assert "oak cabinets" in result.prompt


def test_dynamic_duplicate_of_stable_tag_is_removed_before_composition() -> None:
    previous = StableVisualTags(
        outfit=("navy cardigan",),
        environment=("walnut bookcase",),
    )
    response = json.dumps(
        {
            "version": 4,
            "format": "single",
            "intent": "portrait",
            "dynamic": {
                "subject": ["rating:general", "solo", "1girl"],
                "action": ["reading", "navy cardigan"],
                "emotion": ["smile"],
                "scene": ["walnut bookcase", "morning light"],
                "framing": ["medium shot"],
            },
            "stable": {
                "outfit": {"mode": "keep", "tags": []},
                "environment": {"mode": "keep", "tags": []},
            },
        }
    )

    result = resolve_visual_continuity(response, previous=previous)

    assert result.prompt.count("navy cardigan") == 1
    assert result.prompt.count("walnut bookcase") == 1


def test_first_outfit_accepts_a_valid_open_vocabulary_tag() -> None:
    response = json.dumps(
        {
            "version": 4,
            "format": "single",
            "intent": "portrait",
            "dynamic": {
                "subject": ["rating:general", "solo", "1girl"],
                "action": ["standing"],
                "emotion": [],
                "scene": [],
                "framing": ["full body"],
            },
            "stable": {
                "outfit": {
                    "mode": "replace",
                    "key": "work_set",
                    "tags": ["skirt"],
                },
                "environment": {"mode": "keep", "key": "", "tags": []},
            },
        }
    )

    result = resolve_visual_continuity(response, previous=None)

    assert result.failure == ""
    assert "skirt" in result.prompt
    assert result.stable.outfit_key == "work_set"


def test_specific_outfit_has_enough_visible_detail_to_be_reused() -> None:
    response = json.dumps(
        {
            "version": 4,
            "format": "single",
            "intent": "portrait",
            "dynamic": {
                "subject": ["rating:general", "solo", "1girl"],
                "action": ["standing"],
                "emotion": [],
                "scene": [],
                "framing": ["full body"],
            },
            "stable": {
                "outfit": {
                    "mode": "replace",
                    "key": "work_set",
                    "tags": [
                        "charcoal gray wool",
                        "high-waist pencil skirt",
                        "knee-length",
                        "black tights",
                    ],
                },
                "environment": {"mode": "keep", "key": "", "tags": []},
            },
        }
    )

    result = resolve_visual_continuity(response, previous=None)

    assert result.failure == ""
    assert result.stable.outfit == (
        "charcoal gray wool",
        "high-waist pencil skirt",
        "knee-length",
        "black tights",
    )


def test_first_environment_accepts_a_valid_open_vocabulary_tag() -> None:
    response = json.dumps(
        {
            "version": 4,
            "format": "single",
            "intent": "portrait",
            "dynamic": {
                "subject": ["rating:general", "solo", "1girl"],
                "action": ["standing"],
                "emotion": [],
                "scene": [],
                "framing": ["full body"],
            },
            "stable": {
                "outfit": {"mode": "clear", "key": "", "tags": []},
                "environment": {
                    "mode": "replace",
                    "key": "bedroom",
                    "tags": ["bedroom"],
                },
            },
        }
    )

    result = resolve_visual_continuity(response, previous=None)

    assert result.failure == ""
    assert "bedroom" in result.prompt


def test_qipao_and_film_set_details_are_not_rejected_by_closed_vocabulary() -> None:
    """开放词汇（含权重表达）必须能整套进入稳定区，不依赖有限服装/房间词表。"""

    response = json.dumps(
        {
            "version": 4,
            "format": "single",
            "intent": "normal",
            "dynamic": {
                "subject": ["rating:questionable", "1girl", "solo"],
                "action": ["sitting sideways"],
                "emotion": ["tired"],
                "scene": ["backlighting"],
                "framing": ["full body"],
            },
            "stable": {
                "outfit": {
                    "mode": "replace",
                    "key": "pale_green_silk_qipao",
                    "tags": [
                        "pale green silk qipao",
                        "1.1::china dress::",
                        "silk texture",
                        "high slit",
                        "skin-colored pantyhose",
                        "vintage black leather shoes",
                    ],
                },
                "environment": {
                    "mode": "replace",
                    "key": "messy_film_set_corner",
                    "tags": [
                        "film set",
                        "indoor",
                        "corner",
                        "messy studio",
                        "folding chair",
                        "light stand",
                        "black blackout cloth",
                        "backdrop",
                    ],
                },
            },
        }
    )

    result = resolve_visual_continuity(response, previous=None)

    assert result.failure == ""
    assert "pale green silk qipao" in result.prompt
    assert "film set" in result.prompt


def test_unknown_switch_key_is_rejected_instead_of_silently_reusing_current_tags() -> None:
    previous = StableVisualTags(
        outfit=("navy wool blazer", "white blouse", "silver buttons"),
        outfit_key="work_set",
        outfits=(
            VisualTagCard(
                "work_set",
                ("navy wool blazer", "white blouse", "silver buttons"),
            ),
        ),
    )
    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["standing"],
            "emotion": [],
            "scene": [],
            "framing": ["full body"],
        },
        "stable": {
            "outfit": {"mode": "switch", "key": "missing_outfit", "tags": []},
            "environment": {"mode": "keep", "key": "", "tags": []},
        },
    })

    result = resolve_visual_continuity(response, previous=previous)

    assert result.recognized is True
    assert result.prompt == ""
    assert result.failure == "outfit:switch_unknown_key"
    assert result.stable == previous


def test_replace_with_existing_key_reuses_canonical_card_tags() -> None:
    """同一 key 永远对应同一串 Tag：replace 撞上已有 key 时按 switch 处理，丢弃重译。"""

    existing = VisualTagCard(
        "work_set",
        ("navy wool blazer", "white blouse", "silver buttons"),
    )
    other = VisualTagCard(
        "casual_set",
        ("gray hoodie", "denim shorts"),
    )
    previous = StableVisualTags(
        outfit=other.tags,
        outfit_key=other.key,
        outfits=(existing, other),
    )
    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["sitting"],
            "emotion": [],
            "scene": [],
            "framing": ["full body"],
        },
        "stable": {
            "outfit": {
                "mode": "replace",
                "key": "work_set",
                "tags": [
                    "silver sequin miniskirt",
                    "high waist",
                    "black belt",
                    "black pantyhose",
                ],
            },
            "environment": {"mode": "keep", "key": "", "tags": []},
        },
    })

    result = resolve_visual_continuity(response, previous=previous)

    assert result.failure == ""
    assert result.stable.outfit == existing.tags
    assert result.stable.outfit_key == existing.key
    assert "silver sequin miniskirt" not in result.prompt


def test_scene_delta_cannot_be_acknowledged_with_keep_for_both_stable_sections() -> None:
    """Planner 明确给出稳定区变化时，LLM 不能把服装和环境都判为未变化。"""

    previous = StableVisualTags(
        outfit=("navy wool blazer", "white silk blouse", "silver buttons"),
        environment=("oak desk", "cream plaster wall", "brass desk lamp"),
        outfit_key="work_set",
        environment_key="home_study",
    )
    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["sitting"],
            "emotion": ["tired"],
            "scene": [],
            "framing": ["medium shot"],
        },
        "stable": {
            "outfit": {"mode": "keep", "key": "work_set", "tags": []},
            "environment": {"mode": "keep", "key": "home_study", "tags": []},
        },
    })

    result = resolve_visual_continuity(
        response,
        previous=previous,
        stable_change_text="淡青色丝绸旗袍，肉色丝袜，复古黑色皮鞋",
    )

    assert result.prompt == ""
    assert result.failure == "stable_change_ignored"


def test_planner_keep_directive_reuses_previous_tags_even_if_tag_llm_changes_mode() -> None:
    previous = StableVisualTags(
        outfit=("navy ribbed cardigan", "ivory piping"),
        environment=("walnut bookcase", "cream plaster wall"),
        outfit_key="home_knit_set",
        environment_key="home_study",
    )
    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["standing"],
            "emotion": ["smile"],
            "scene": [],
            "framing": ["medium shot"],
        },
        "stable": {
            "outfit": {
                "mode": "replace",
                "key": "wrong_new_key",
                "tags": ["red dress"],
            },
            "environment": {"mode": "keep", "key": "home_study", "tags": []},
        },
    })

    result = resolve_visual_continuity(
        response,
        previous=previous,
        directives={
            "outfit": VisualChangeDirective("keep"),
            "environment": VisualChangeDirective("keep"),
        },
    )

    assert result.prompt.count("navy ribbed cardigan") == 1
    assert result.prompt.count("walnut bookcase") == 1
    assert "red dress" not in result.prompt


def test_planner_replace_directive_only_regenerates_changed_stable_section() -> None:
    previous = StableVisualTags(
        outfit=("navy ribbed cardigan", "ivory piping"),
        environment=("walnut bookcase", "cream plaster wall"),
        outfit_key="home_knit_set",
        environment_key="home_study",
    )
    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["standing"],
            "emotion": [],
            "scene": [],
            "framing": ["full body"],
        },
        "stable": {
            "outfit": {
                "mode": "keep",
                "key": "new_qipao_set",
                "tags": ["pale cyan silk qipao", "black leather shoes"],
            },
            "environment": {"mode": "replace", "key": "wrong_scene", "tags": ["street"]},
        },
    })

    result = resolve_visual_continuity(
        response,
        previous=previous,
        directives={
            "outfit": VisualChangeDirective(
                "replace", description="淡青色丝绸旗袍"
            ),
            "environment": VisualChangeDirective("keep"),
        },
    )

    assert "pale cyan silk qipao" in result.prompt
    assert "black leather shoes" in result.prompt
    assert "walnut bookcase" in result.prompt
    assert "street" not in result.prompt


def test_no_change_phrasings_parse_as_keep_instead_of_replace() -> None:
    """历史 bug：非枚举文本默认 replace，把"保持不变"误判成换装导致服装漂移。"""

    for phrase in (
        "unchanged", "no change", "keep", "同上", "保持不变", "保持原样",
        "没变", "没换", "无变化", "没有变化", "不变。", "维持原样", "无", "none",
    ):
        directive = parse_visual_change_directive(phrase)
        assert directive is not None
        assert directive.mode == "keep", f"{phrase!r} 应解析为 keep，实际 {directive.mode}"


def test_ambiguous_free_text_parses_as_auto_not_replace() -> None:
    """无法归类的自由文本交给 Tag LLM 判定，解析层不猜测为换装。"""

    directive = parse_visual_change_directive("她还穿着刚才那身衣服躺在床上")
    assert directive is not None
    assert directive.mode == "auto"
    assert directive.description == "她还穿着刚才那身衣服躺在床上"


def test_switch_with_non_ascii_target_keeps_description_for_llm() -> None:
    directive = parse_visual_change_directive("switch:白色连衣裙")
    assert directive is not None
    assert directive.mode == "switch"
    assert directive.key == ""
    assert directive.description == "白色连衣裙"

    known = parse_visual_change_directive("switch:home_knit_set")
    assert known is not None
    assert known.mode == "switch"
    assert known.key == "home_knit_set"


def test_directives_read_enum_and_new_look_fields() -> None:
    directives = parse_visual_change_directives({
        "outfit_change": "replace",
        "outfit_new_look": "淡青色丝绸旗袍，肉色丝袜",
        "environment_change": "unchanged",
    })

    assert directives is not None
    assert directives["outfit"].mode == "replace"
    assert directives["outfit"].description == "淡青色丝绸旗袍，肉色丝袜"
    assert directives["environment"].mode == "keep"

    # 只填描述没填枚举：降级为 auto，由 Tag LLM 判定
    only_look = parse_visual_change_directives({"outfit_new_look": "白色泳装"})
    assert only_look is not None
    assert only_look["outfit"].mode == "auto"
    assert only_look["outfit"].description == "白色泳装"

    assert parse_visual_change_directives({}) is None


def test_planner_switch_by_description_lets_llm_pick_library_key() -> None:
    """Planner 只给口语目标时，由 Tag LLM 从库中选 key，程序按卡片原 Tag 复用。"""

    home_outfit = VisualTagCard(
        "home_knit_set",
        ("navy ribbed cardigan", "ivory piping"),
    )
    coat_outfit = VisualTagCard(
        "green_coat_set",
        ("forest green wool coat", "double-breasted coat"),
    )
    previous = StableVisualTags(
        outfit=coat_outfit.tags,
        outfit_key=coat_outfit.key,
        outfits=(home_outfit, coat_outfit),
    )
    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["sitting"],
            "emotion": [],
            "scene": [],
            "framing": ["medium shot"],
        },
        "stable": {
            "outfit": {"mode": "switch", "key": "home_knit_set", "tags": []},
            "environment": {"mode": "keep", "key": "", "tags": []},
        },
    })

    result = resolve_visual_continuity(
        response,
        previous=previous,
        directives={
            "outfit": VisualChangeDirective("switch", description="之前那套针织开衫"),
            "environment": VisualChangeDirective("keep"),
        },
    )

    assert result.failure == ""
    assert result.stable.outfit == home_outfit.tags
    assert result.stable.outfit_key == home_outfit.key


def test_planner_switch_without_match_falls_back_to_llm_rebuild() -> None:
    """库中无匹配时（如重启丢库），switch 指令允许 LLM 用 replace 重建而不是整单失败。"""

    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["standing"],
            "emotion": [],
            "scene": [],
            "framing": ["full body"],
        },
        "stable": {
            "outfit": {
                "mode": "replace",
                "key": "white_dress_set",
                "tags": ["white sundress", "lace trim"],
            },
            "environment": {"mode": "keep", "key": "", "tags": []},
        },
    })

    result = resolve_visual_continuity(
        response,
        previous=None,
        directives={
            "outfit": VisualChangeDirective("switch", description="之前那条白裙子"),
            "environment": VisualChangeDirective("keep"),
        },
    )

    assert result.failure == ""
    assert result.stable.outfit == ("white sundress", "lace trim")
    assert result.stable.outfit_key == "white_dress_set"


def test_planner_keep_with_lost_cache_reestablishes_via_replace() -> None:
    """unchanged 撞上缓存丢失（TTL 过期/重启）时，接受 LLM 重建而不是死锁。"""

    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["sitting"],
            "emotion": [],
            "scene": [],
            "framing": ["medium shot"],
        },
        "stable": {
            "outfit": {
                "mode": "replace",
                "key": "school_uniform_set",
                "tags": ["dark blue sailor uniform", "red neckerchief"],
            },
            "environment": {"mode": "keep", "key": "", "tags": []},
        },
    })

    result = resolve_visual_continuity(
        response,
        previous=None,
        directives={
            "outfit": VisualChangeDirective("keep"),
            "environment": VisualChangeDirective("keep"),
        },
    )

    assert result.failure == ""
    assert result.stable.outfit == ("dark blue sailor uniform", "red neckerchief")


def test_bot_visible_frame_requires_outfit_to_be_established() -> None:
    """Bot 出镜时服装稳定区必须建立，否则下一轮"逐字复用"无从谈起。"""

    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl", "white dress"],
            "action": ["standing"],
            "emotion": [],
            "scene": [],
            "framing": ["full body"],
        },
        "stable": {
            "outfit": {"mode": "keep", "key": "", "tags": []},
            "environment": {"mode": "keep", "key": "", "tags": []},
        },
    })

    result = resolve_visual_continuity(response, previous=None)

    assert result.prompt == ""
    assert result.failure == "outfit:not_established"

    # 明确 clear（如特殊 NSFW 情景）仍然允许服装区为空
    cleared = json.loads(response)
    cleared["stable"]["outfit"] = {"mode": "clear", "key": "", "tags": []}
    cleared_result = resolve_visual_continuity(json.dumps(cleared), previous=None)
    assert cleared_result.failure == ""


def test_dynamic_synonym_shells_of_stable_tags_are_deduplicated() -> None:
    """下划线/权重壳变体也要与稳定 Tag 去重，不能靠原文精确匹配漏网。"""

    previous = StableVisualTags(
        outfit=("navy cardigan",),
        environment=("walnut bookcase",),
    )
    response = json.dumps({
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["reading", "Navy_Cardigan"],
            "emotion": [],
            "scene": ["1.1::walnut bookcase::", "morning light"],
            "framing": ["medium shot"],
        },
        "stable": {
            "outfit": {"mode": "keep", "tags": []},
            "environment": {"mode": "keep", "tags": []},
        },
    })

    result = resolve_visual_continuity(response, previous=previous)

    assert result.prompt.lower().count("cardigan") == 1
    assert result.prompt.count("walnut bookcase") == 1


def test_switch_touch_protects_current_card_from_eviction() -> None:
    """switch 命中的卡片移到队尾后，连续 replace 不会把"当前在穿"的卡挤出库。"""

    state = StableVisualTags()
    first_replace = {
        "version": 4,
        "format": "single",
        "intent": "portrait",
        "dynamic": {
            "subject": ["rating:general", "solo", "1girl"],
            "action": ["standing"],
            "emotion": [],
            "scene": [],
            "framing": ["full body"],
        },
        "stable": {
            "outfit": {"mode": "replace", "key": "outfit_seed", "tags": ["seed dress"]},
            "environment": {"mode": "keep", "key": "", "tags": []},
        },
    }
    state = resolve_visual_continuity(json.dumps(first_replace), previous=state).stable
    for index in range(10):
        payload = json.loads(json.dumps(first_replace))
        payload["stable"]["outfit"] = {
            "mode": "replace",
            "key": f"outfit_{index}",
            "tags": [f"fabric design {index}"],
        }
        state = resolve_visual_continuity(json.dumps(payload), previous=state).stable

    switch_back = json.loads(json.dumps(first_replace))
    switch_back["stable"]["outfit"] = {"mode": "switch", "key": "outfit_seed", "tags": []}
    state = resolve_visual_continuity(json.dumps(switch_back), previous=state).stable
    assert state.outfit_key == "outfit_seed"
    assert state.outfits[-1].key == "outfit_seed"

    # 再连续换 11 套新装：seed 卡因 touch 过仍应在 12 张容量内存活
    for index in range(10, 21):
        payload = json.loads(json.dumps(first_replace))
        payload["stable"]["outfit"] = {
            "mode": "replace",
            "key": f"outfit_{index}",
            "tags": [f"fabric design {index}"],
        }
        state = resolve_visual_continuity(json.dumps(payload), previous=state).stable
    assert any(card.key == "outfit_seed" for card in state.outfits)


def test_describe_visual_failure_gives_actionable_chinese_hint() -> None:
    assert "JSON" in describe_visual_failure("not_json")
    assert "rating" in describe_visual_failure("missing_rating_subject")
    assert "服装" in describe_visual_failure("outfit:not_established")
    assert "环境" in describe_visual_failure("environment:switch_unknown_key")
    assert describe_visual_failure("unknown_code")  # 未知码也要有兜底指引文本
