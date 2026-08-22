"""会话级视觉连续性 Tag 的确定性解析与合成。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class VisualTagCard:
    """一套可再次切回的服装或环境 Tag。"""

    key: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class StableVisualTags:
    """已经交付给 NovelAI、后续需要逐字复用的稳定 Tag。"""

    outfit: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    scene_effects: tuple[str, ...] = ()
    last_framing: tuple[str, ...] = field(default=(), compare=False)
    outfit_key: str = ""
    environment_key: str = ""
    outfits: tuple[VisualTagCard, ...] = ()
    environments: tuple[VisualTagCard, ...] = ()


@dataclass(frozen=True)
class VisualContinuityResult:
    """解析结果；``failure`` 非空时给出可驱动修复重试的原因码。"""

    prompt: str
    stable: StableVisualTags
    recognized: bool = False
    failure: str = ""
    structured: dict[str, Any] | None = None


@dataclass(frozen=True)
class VisualChangeDirective:
    """Planner 已经判定的一个稳定区变化决定。

    ``auto`` 表示 Planner 给了无法归类的自由文本：变化与否交给 Tag LLM
    结合上下文判定，绝不在解析层猜测为换装（历史上默认 replace 会把
    "保持不变"之类的表述误判成换装，造成服装漂移）。
    """

    mode: str
    description: str = ""
    key: str = ""


@dataclass(frozen=True)
class _SectionOutcome:
    """单个稳定区的解析结果；``failure`` 为空即有效。"""

    tags: tuple[str, ...]
    key: str
    cards: tuple[VisualTagCard, ...]
    failure: str = ""
    cleared: bool = False


_VALID_MODES = frozenset({"keep", "replace", "switch", "clear"})
_MAX_CARDS_PER_KIND = 12
_RATING_TAGS = frozenset(
    {"rating:general", "rating:sensitive", "rating:questionable", "rating:explicit"}
)
_POSITION_GRID_RE = re.compile(r"^[A-E][1-5]$")
_FEMALE_IDENTITY_TAGS = frozenset({"girl", "woman", "female"})
_GLOBAL_PERSON_TAG_RE = re.compile(r"^(?:solo|\d+girls?|\d+boys?|\d+others?|\d+boy \d+girl)$")
# NAI v4 的角色互动绑定前缀；两侧必须挂同名 action tag 才能绑成一次互动。
_INTERACTION_PREFIXES = frozenset({"source", "target", "mutual"})

# Planner 枚举与常见口语的整句匹配表：只在全句命中时生效，
# 更复杂的自由文本一律降级为 auto，交给 Tag LLM 判定。
_KEEP_FULLMATCH = re.compile(
    r"unchanged|unchange|no[ _-]?change|keep|keep the same|same|as before|none"
    r"|无|不变|沿用|照旧|同上|原样|一样|无变化|保持不变|保持原样|维持不变|维持原样"
    r"|没变|没换|没变化|没有变|没有换|没有变化|(?:和|跟)之前一样",
    re.IGNORECASE,
)
_CLEAR_FULLMATCH = re.compile(r"clear|清除|清空|不可见", re.IGNORECASE)
_SWITCH_BARE = frozenset({"switch", "切换", "切回"})
_REPLACE_BARE = frozenset({"replace", "更换", "换装"})


def parse_visual_change_directive(
    value: Any,
    extra_description: Any = "",
) -> VisualChangeDirective | None:
    """解析 Planner 的稳定区决定，不对服装或地点文本做语义猜测。

    ``extra_description`` 对应新协议的 ``*_new_look`` 字段：模式为裸枚举时
    从这里取 replace 的新描述或 switch 的口语目标。
    """

    text = str(value or "").strip()
    extra = str(extra_description or "").strip()
    if not text:
        # 只填了 *_new_look 没填模式：意图不明，交给 Tag LLM 判定
        return VisualChangeDirective("auto", description=extra) if extra else None
    normalized = text.lower().replace("：", ":").strip()
    stripped = normalized.strip("。.!！~～ ")
    if _KEEP_FULLMATCH.fullmatch(stripped):
        return VisualChangeDirective("keep")
    if _CLEAR_FULLMATCH.fullmatch(stripped):
        return VisualChangeDirective("clear")
    if stripped in _SWITCH_BARE:
        return VisualChangeDirective("switch", description=extra)
    if stripped in _REPLACE_BARE:
        return VisualChangeDirective("replace", description=extra)
    for prefix in ("switch:", "switch ", "切换:", "切换 ", "切回:", "切回 "):
        if normalized.startswith(prefix):
            remainder = text[len(prefix):].strip()
            key = _as_key(remainder)
            if key:
                return VisualChangeDirective("switch", key=key)
            # key 不合法（如中文口语目标）：保留为描述，由 Tag LLM 从库中选 key
            return VisualChangeDirective("switch", description=remainder or extra)
    for prefix in (
        "replace:", "replace ", "changed:", "changed ",
        "change:", "change ", "changed_to:", "changed_to ",
        "更换:", "更换 ", "变化:", "变化 ", "换装:", "换装 ",
    ):
        if normalized.startswith(prefix):
            return VisualChangeDirective(
                "replace",
                description=text[len(prefix):].strip() or extra,
            )
    # 其余自由文本：变化与否交给 Tag LLM 判定，避免把"没换装"误判成换装
    return VisualChangeDirective("auto", description=text)


def parse_visual_change_directives(
    action_data: dict[str, Any],
) -> dict[str, VisualChangeDirective] | None:
    """读取 Planner 独立的服装/环境变化字段；字段全缺时返回 None 以兼容旧调用。"""

    directives: dict[str, VisualChangeDirective] = {}
    for kind, mode_field, look_field in (
        ("outfit", "outfit_change", "outfit_new_look"),
        ("environment", "environment_change", "environment_new_look"),
        ("effects", "effect_change", "effect_new_state"),
    ):
        directive = parse_visual_change_directive(
            action_data.get(mode_field),
            extra_description=action_data.get(look_field),
        )
        if directive is not None:
            directives[kind] = directive
    return directives or None


def _parse_response_json(response: str) -> dict[str, Any] | None:
    """接受模型偶发包裹的 Markdown code fence，仍按 JSON 协议校验。"""

    candidate = str(response or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        tag.strip()
        for tag in value
        if isinstance(tag, str) and tag.strip()
    )


def _as_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    if not key or len(key) > 64:
        return ""
    if not all(
        character.isascii() and (character.isalnum() or character in "_-")
        for character in key
    ):
        return ""
    return key


def _normalize_stable_tag(tag: str) -> str:
    """移除权重外壳并统一分隔符，只用于结构校验与去重，不改写最终 Tag。"""

    normalized = str(tag or "").strip().lower()
    weighted = re.fullmatch(r"[+-]?\d+(?:\.\d+)?::(.*?)::", normalized)
    if weighted:
        normalized = weighted.group(1).strip()
    normalized = normalized.strip("{}[]() ").replace("_", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _find_card(cards: tuple[VisualTagCard, ...], key: str) -> VisualTagCard | None:
    return next((card for card in cards if card.key == key), None)


def _collect_interaction_bindings(
    people_tags: Iterable[Iterable[str]],
) -> dict[tuple[str, str], set[int]]:
    """按人物索引归集 source#/target#/mutual# 绑定，key 为 (前缀, 归一化动作)。"""

    bindings: dict[tuple[str, str], set[int]] = {}
    for index, tags in enumerate(people_tags):
        for tag in tags:
            prefix, separator, action = _normalize_stable_tag(tag).partition("#")
            action = action.strip()
            if not separator or not action or prefix not in _INTERACTION_PREFIXES:
                continue
            bindings.setdefault((prefix, action), set()).add(index)
    return bindings


def _interaction_binding_failure(people_tags: Iterable[Iterable[str]]) -> str:
    """校验 NAI 互动前缀是否成对，未成对即判失败。

    NAI 只有在两个人物段出现**同名** action tag 时才能把它们绑成同一次互动
    （官方示例是 ``source#hug`` 配 ``target#hug``）。历史规则曾要求施动方写现在
    分词、接收方写过去分词，产出 ``source#groping`` / ``target#groped`` 这类永远
    配不上的组合；NAI 于是把两侧当成互不相关的独立描述，谁对谁做变成随机，
    实测还会整体互换或让接收方自己对自己做。同理，自身动作被误加 ``source#``
    （如 ``source#groping own breasts``）也会因为找不到对家而把动作绑到别人身上。
    这里做确定性校验，配不上就打回重写，不做静默兜底。
    """

    bindings = _collect_interaction_bindings(people_tags)
    for (prefix, action), holders in bindings.items():
        if prefix == "mutual":
            if len(holders) < 2:
                return "interaction_mutual_unpaired"
            continue
        opposite = "target" if prefix == "source" else "source"
        # 对家必须是另一个人物段：同一人物同时写 source# 和 target# 不构成互动
        if not (bindings.get((opposite, action), set()) - holders):
            return "interaction_unpaired"
    return ""


def _remember_card(
    cards: tuple[VisualTagCard, ...],
    key: str,
    tags: tuple[str, ...],
) -> tuple[VisualTagCard, ...]:
    if not key or not tags or _find_card(cards, key) is not None:
        return cards
    return (cards + (VisualTagCard(key, tags),))[-_MAX_CARDS_PER_KIND:]


def _touch_card(
    cards: tuple[VisualTagCard, ...],
    key: str,
) -> tuple[VisualTagCard, ...]:
    """把命中的卡片移到队尾，防止"当前在用"的卡被容量淘汰挤出库。"""

    card = _find_card(cards, key)
    if card is None:
        return cards
    return tuple(c for c in cards if c.key != key) + (card,)


def _resolve_stable_section(
    raw_section: Any,
    previous_tags: tuple[str, ...],
    previous_key: str,
    cards: tuple[VisualTagCard, ...],
) -> _SectionOutcome:
    """按 Tag LLM 输出解析单个稳定区；程序保证"同一 key 永远对应同一串 Tag"。"""

    if not isinstance(raw_section, dict):
        return _SectionOutcome(previous_tags, previous_key, cards, failure="section_missing")

    mode = str(raw_section.get("mode", "keep") or "keep").strip().lower()
    if mode not in _VALID_MODES:
        return _SectionOutcome(previous_tags, previous_key, cards, failure="invalid_mode")
    key = _as_key(raw_section.get("key"))
    if mode == "keep":
        if previous_tags:
            return _SectionOutcome(previous_tags, previous_key, cards)
        if _as_tags(raw_section.get("tags")):
            # 首次建立必须走 replace 携带 key，否则进不了卡片库、后续无法 switch
            return _SectionOutcome(
                previous_tags, previous_key, cards, failure="keep_cannot_initialize"
            )
        return _SectionOutcome((), "", cards)
    if mode == "clear":
        return _SectionOutcome((), "", cards, cleared=True)
    if mode == "switch":
        known = _find_card(cards, key)
        if known is not None:
            return _SectionOutcome(known.tags, known.key, _touch_card(cards, key))
        if key and key == previous_key and previous_tags:
            # 兼容旧数据：当前区有 key 但卡片库尚未建立
            return _SectionOutcome(previous_tags, previous_key, cards)
        return _SectionOutcome(previous_tags, previous_key, cards, failure="switch_unknown_key")

    replacement = _as_tags(raw_section.get("tags"))
    known = _find_card(cards, key)
    if known is not None:
        # key 已存在：同一 key 永远对应同一串 Tag，忽略重译结果直接视作 switch
        return _SectionOutcome(known.tags, known.key, _touch_card(cards, key))
    if key and key == previous_key and previous_tags:
        return _SectionOutcome(previous_tags, previous_key, cards)
    if not key or not replacement:
        return _SectionOutcome(previous_tags, previous_key, cards, failure="replace_incomplete")
    return _SectionOutcome(replacement, key, _remember_card(cards, key, replacement))


def _resolve_llm_change(
    raw_section: Any,
    previous_tags: tuple[str, ...],
    previous_key: str,
    cards: tuple[VisualTagCard, ...],
    *,
    allow_current: bool,
    failure_code: str,
) -> _SectionOutcome:
    """Planner 已声明变化：按内容解读 LLM 翻译（回库 key 或新 key+tags）。

    只看 key/tags 内容、不迷信 mode 标签——LLM 偶发把 replace 翻译标成 keep，
    内容可用时直接采纳，省一轮修复。``allow_current`` 为 False 时（replace 指令）
    不接受解析回当前 key，否则"换装"会静默变成"没换"。
    """

    if isinstance(raw_section, dict):
        key = _as_key(raw_section.get("key"))
        tags = _as_tags(raw_section.get("tags"))
        known = _find_card(cards, key)
        if known is not None and (allow_current or known.key != previous_key):
            return _SectionOutcome(known.tags, known.key, _touch_card(cards, key))
        if key and key != previous_key and tags:
            return _SectionOutcome(tags, key, _remember_card(cards, key, tags))
        if allow_current and key and key == previous_key and previous_tags:
            return _SectionOutcome(previous_tags, previous_key, cards)
    return _SectionOutcome(previous_tags, previous_key, cards, failure=failure_code)


def _resolve_directive_section(
    directive: VisualChangeDirective | None,
    previous_tags: tuple[str, ...],
    previous_key: str,
    cards: tuple[VisualTagCard, ...],
    raw_section: Any,
) -> _SectionOutcome:
    """按 Planner 决策解析稳定区；unchanged/clear 完全确定性，其余由 LLM 提供翻译。"""

    if directive is None:
        return _SectionOutcome(previous_tags, previous_key, cards)
    mode = directive.mode.strip().lower()
    if mode == "keep":
        if previous_tags:
            return _SectionOutcome(previous_tags, previous_key, cards)
        # 缓存已失效（TTL 过期或重启丢失）：退回 LLM 判定，允许 replace 重建
        return _resolve_stable_section(raw_section, previous_tags, previous_key, cards)
    if mode == "clear":
        return _SectionOutcome((), "", cards, cleared=True)
    if mode == "auto":
        return _resolve_stable_section(raw_section, previous_tags, previous_key, cards)
    if mode == "switch":
        known = _find_card(cards, directive.key)
        if known is not None:
            return _SectionOutcome(known.tags, known.key, _touch_card(cards, directive.key))
        if directive.key and directive.key == previous_key and previous_tags:
            return _SectionOutcome(previous_tags, previous_key, cards)
        # Planner 未给出可用 key：由 LLM 从库中选 key，库中无匹配时重建
        return _resolve_llm_change(
            raw_section, previous_tags, previous_key, cards,
            allow_current=True,
            failure_code="switch_target_unresolved",
        )
    if mode == "replace":
        return _resolve_llm_change(
            raw_section, previous_tags, previous_key, cards,
            allow_current=False,
            failure_code="replace_not_translated",
        )
    return _SectionOutcome(previous_tags, previous_key, cards, failure="invalid_mode")


def _resolve_effects(
    raw_section: Any,
    previous_subject: tuple[str, ...],
    previous_scene: tuple[str, ...],
    directive: VisualChangeDirective | None,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """解析会跨图延续但不进入服装/环境卡片库的临时效果。"""

    if directive is not None and directive.mode == "keep":
        return previous_subject, previous_scene, ""
    if directive is not None and directive.mode == "clear":
        return (), (), ""
    if directive is not None and directive.mode == "switch":
        return previous_subject, previous_scene, "invalid_mode"

    if not isinstance(raw_section, dict):
        if directive is not None and directive.mode in {"replace", "auto"}:
            return previous_subject, previous_scene, "replace_not_translated"
        return previous_subject, previous_scene, ""

    mode = str(raw_section.get("mode", "keep") or "keep").strip().lower()
    subject = _as_tags(raw_section.get("subject")) or _as_tags(raw_section.get("tags"))
    scene = _as_tags(raw_section.get("scene"))
    if directive is not None and directive.mode == "replace":
        return (
            (subject, scene, "")
            if subject or scene
            else (previous_subject, previous_scene, "replace_not_translated")
        )
    if mode == "keep" and directive is None:
        return previous_subject, previous_scene, ""
    if mode == "clear":
        return (), (), ""
    if mode != "replace":
        return previous_subject, previous_scene, "invalid_mode"
    if not subject and not scene:
        return previous_subject, previous_scene, "replace_incomplete"
    return subject, scene, ""


def _join_tags(groups: Iterable[tuple[str, ...]]) -> str:
    return ", ".join(tag for group in groups for tag in group)


def _raw_section_mode(raw_section: Any) -> str:
    if not isinstance(raw_section, dict):
        return "missing"
    return str(raw_section.get("mode", "keep") or "keep").strip().lower()


def _resolve_multi_prompt(
    dynamic: dict[str, Any],
    current_state: StableVisualTags,
    *,
    include_outfit: bool,
    stable_normalized: frozenset[str],
) -> tuple[str, dict[str, Any] | None, str]:
    """把 v4 multi 的全局与人物 Tag 分开，固定外观只归属显式 Bot 角色。"""

    if _as_tags(dynamic.get("action")) or _as_tags(dynamic.get("emotion")):
        return "", None, "multi:unscoped_character_tags"
    raw_people = dynamic.get("people")
    if not isinstance(raw_people, list):
        return "", None, "multi:people_missing"
    if len(raw_people) < 2:
        return "", None, "multi:people_too_few"

    raw_positions = dynamic.get("positions", [])
    if raw_positions is None:
        raw_positions = []
    if not isinstance(raw_positions, list):
        return "", None, "multi:positions_invalid"
    if raw_positions and len(raw_positions) != len(raw_people):
        return "", None, "multi:positions_length_mismatch"
    positions = [str(value or "").strip().upper() for value in raw_positions]
    if any(not _POSITION_GRID_RE.fullmatch(value) for value in positions):
        return "", None, "multi:positions_invalid"

    characters: list[dict[str, str]] = []
    bot_indices: list[int] = []
    people_dynamic_tags: list[tuple[str, ...]] = []
    for index, raw_person in enumerate(raw_people):
        if not isinstance(raw_person, dict):
            return "", None, "multi:person_invalid"
        role = str(raw_person.get("role", "") or "").strip().lower()
        if role not in {"bot", "other"}:
            return "", None, "multi:invalid_role"
        if role == "bot":
            bot_indices.append(index)

        if not isinstance(raw_person.get("subject"), list):
            return "", None, "multi:person_subject_missing"
        if not isinstance(raw_person.get("action", []), list) or not isinstance(
            raw_person.get("emotion", []), list
        ):
            return "", None, "multi:person_invalid"
        person_subject = _as_tags(raw_person.get("subject"))
        if not person_subject:
            return "", None, "multi:person_subject_missing"
        person_dynamic = (
            *person_subject,
            *_as_tags(raw_person.get("action")),
            *_as_tags(raw_person.get("emotion")),
        )
        if any(
            tag.lower() in _RATING_TAGS
            or _GLOBAL_PERSON_TAG_RE.fullmatch(_normalize_stable_tag(tag))
            for tag in person_dynamic
        ):
            return "", None, "multi:global_tag_in_person"
        people_dynamic_tags.append(person_dynamic)

        def _person_tags(tags: Iterable[str]) -> tuple[str, ...]:
            return tuple(
                tag
                for tag in tags
                if _normalize_stable_tag(tag) not in stable_normalized
            )

        character_prompt = _join_tags((
            _person_tags(person_subject),
            current_state.outfit if include_outfit and role == "bot" else (),
            current_state.effects if include_outfit and role == "bot" else (),
            _person_tags(_as_tags(raw_person.get("action"))),
            _person_tags(_as_tags(raw_person.get("emotion"))),
        ))
        characters.append({
            "prompt": character_prompt,
            "negative_prompt": "",
            "position": positions[index] if positions else "",
        })

    if include_outfit and not bot_indices:
        return "", None, "multi:bot_missing"
    if len(bot_indices) > 1:
        return "", None, "multi:bot_ambiguous"
    if bot_indices:
        bot_subject = _as_tags(raw_people[bot_indices[0]].get("subject"))
        if _normalize_stable_tag(bot_subject[0]) not in _FEMALE_IDENTITY_TAGS:
            return "", None, "multi:bot_not_female"

    binding_failure = _interaction_binding_failure(people_dynamic_tags)
    if binding_failure:
        return "", None, f"multi:{binding_failure}"

    global_text = _join_tags((
        _as_tags(dynamic.get("subject")),
        current_state.environment,
        current_state.scene_effects,
        _as_tags(dynamic.get("scene")),
        _as_tags(dynamic.get("framing")),
    ))
    lines = [f"{global_text},"]
    lines.extend(
        f"char{index}:{character['prompt']},"
        for index, character in enumerate(characters, start=1)
    )
    structured: dict[str, Any] = {
        "global_text": global_text,
        "characters": characters,
        "has_coords": bool(positions),
    }
    if bot_indices:
        structured["bot_character_index"] = bot_indices[0]
    return "\n".join(lines), structured, ""


def render_visual_continuity_context(
    stable: StableVisualTags | None,
    *,
    include_outfit: bool = True,
) -> str:
    """把缓存 Tag 渲染成 LLM 只读上下文；最终是否复用仍由程序裁决。"""

    current = stable or StableVisualTags()
    outfit = ", ".join(current.outfit) if include_outfit else "（本轮 Bot 不出镜）"
    outfit = outfit or "（尚无缓存）"
    environment = ", ".join(current.environment) or "（尚无缓存）"
    effects = ", ".join(current.effects) if include_outfit else "（本轮 Bot 不出镜）"
    effects = effects or "（无）"
    scene_effects = ", ".join(current.scene_effects) or "（无）"
    last_framing = ", ".join(current.last_framing) or "（无）"
    outfit_library = "; ".join(
        f"{card.key}=[{', '.join(card.tags)}]" for card in current.outfits
    ) if include_outfit else "（本轮 Bot 不出镜）"
    outfit_library = outfit_library or "（无）"
    environment_library = "; ".join(
        f"{card.key}=[{', '.join(card.tags)}]" for card in current.environments
    ) or "（无）"
    return (
        "<visual_continuity_context>\n"
        "这是当前聊天流已经使用过的稳定 NovelAI Tag；它们不是普通参考文本。\n"
        f"当前服装：key={current.outfit_key or 'none'}；Tag={outfit}\n"
        f"当前环境：key={current.environment_key or 'none'}；Tag={environment}\n"
        f"当前跨图临时效果：{effects}\n"
        f"当前场景附着效果：{scene_effects}\n"
        f"上一张实际镜头：{last_framing}\n"
        f"已知服装库：{outfit_library}\n"
        f"已知环境库：{environment_library}\n"
        "如果 planner_visual_request 提供 outfit_change / environment_change / effect_change，变化判断已经由 Planner 完成；"
        "严格遵守 Planner 的决定。程序会逐字复用 unchanged 区域的缓存 Tag，并忽略本轮对它的任何同义改写。\n"
        "Planner 声明 switch 但没有给出库中 key 时，由你根据描述从上方库中选定唯一匹配的 key 输出 mode=switch；"
        "库中确实没有对应项时改用 mode=replace 依据描述重建完整 Tag。\n"
        "Planner 声明 unchanged 而对应缓存为空（过期或重启丢失）时，"
        "用 mode=replace 依据聊天上下文重建当前应有的稳定区。\n"
        "没有 Planner 决定时才根据上下文判断是否变化，没有明确变化必须用 keep；"
        "但聊天进入睡觉/洗澡/回家/出门/运动等会自然换装的生活节点时视为明确变化——"
        "结合 current_time_context 与情景用 replace 换成当下应穿的衣服（深夜床上=睡衣，不是白天的外出装），"
        "场所随之变化时环境同样要更新。\n"
        "回到已知服装或地点时必须用 switch 和库中的原 key；"
        "只有聊天情景明确建立了全新服装或地点时才用 replace，并生成完整、具体的新 Tag。"
        "不要按固定场景分类套模板。\n"
        "replace 设计新服装时：用户明确指定的细节必须忠实保留；用户没有指定的部分由你主动设计——"
        "优先给出有审美记忆点的完整穿搭（配色呼应、材质与剪裁细节、少量配饰点缀），贴合角色气质与场合；"
        "居家、睡前场景可以朴素但仍要有质感细节，避免每次都是同质化基础款。\n"
        "服装稳定区描述颜色、材质、剪裁、结构、纹样、扣件、鞋袜和配饰等可见设计；"
        "环境稳定区描述空间布局、固定物件、材质和配色。"
        "动作、表情、姿态、构图、镜头、当下天气、时间和光线只写入 dynamic。\n"
        "人物临时效果只保存剧情尚未解除且跨图可见的湿透、撕裂、污渍、泪痕、凌乱状态或持续持物；"
        "放在桌椅地面等处、下一张仍存在的物品写入场景附着效果。"
        "普通动作和瞬时表情不得写入。上一张镜头只用于柔性避免连续同构图；"
        "用户或当前情景明确要求同一镜头时必须服从，不得强制变化。\n"
        "</visual_continuity_context>"
    )


def resolve_visual_continuity(
    response: str,
    *,
    previous: StableVisualTags | None = None,
    include_outfit: bool = True,
    directives: dict[str, VisualChangeDirective] | None = None,
    stable_change_text: str = "",
) -> VisualContinuityResult:
    """解析 v4 连续性 envelope，并由程序按固定次序合成最终 Prompt。

    校验失败时返回空 prompt，同时在 ``failure`` 中给出原因码；
    所有原因码都是 Tag LLM 重试可修复的（确定性指令冲突已在解析层消解）。
    """

    previous_state = previous or StableVisualTags()
    if directives is not None:
        # Planner 未提供的稳定区按协议视为 unchanged
        directives = {
            "outfit": directives.get("outfit", VisualChangeDirective("keep")),
            "environment": directives.get("environment", VisualChangeDirective("keep")),
            "effects": directives.get("effects", VisualChangeDirective("keep")),
        }
    payload = _parse_response_json(response)
    if payload is None:
        return VisualContinuityResult("", previous_state, failure="not_json")
    if payload.get("version") != 4:
        return VisualContinuityResult("", previous_state, failure="wrong_version")
    output_format = str(payload.get("format", "single") or "single").strip().lower()
    if output_format not in {"single", "multi"}:
        return VisualContinuityResult(
            "", previous_state, recognized=True, failure="invalid_format"
        )

    dynamic = payload.get("dynamic")
    stable = payload.get("stable")
    if not isinstance(dynamic, dict) or not isinstance(stable, dict):
        return VisualContinuityResult(
            "", previous_state, recognized=True, failure="missing_sections"
        )

    subject = _as_tags(dynamic.get("subject"))
    if not subject or subject[0].lower() not in _RATING_TAGS:
        return VisualContinuityResult(
            "", previous_state, recognized=True, failure="missing_rating_subject"
        )

    raw_outfit = stable.get("outfit")
    raw_environment = stable.get("environment")
    raw_effects = stable.get("effects")
    if not include_outfit:
        # Bot 不出镜：服装状态原样穿透，不参与本轮画面
        outfit_outcome = _SectionOutcome(
            previous_state.outfit, previous_state.outfit_key, previous_state.outfits
        )
    elif directives is None:
        outfit_outcome = _resolve_stable_section(
            raw_outfit, previous_state.outfit,
            previous_state.outfit_key, previous_state.outfits,
        )
    else:
        outfit_outcome = _resolve_directive_section(
            directives["outfit"], previous_state.outfit,
            previous_state.outfit_key, previous_state.outfits, raw_outfit,
        )
    if directives is None:
        environment_outcome = _resolve_stable_section(
            raw_environment, previous_state.environment,
            previous_state.environment_key, previous_state.environments,
        )
    else:
        environment_outcome = _resolve_directive_section(
            directives["environment"], previous_state.environment,
            previous_state.environment_key, previous_state.environments, raw_environment,
        )
    if outfit_outcome.failure:
        return VisualContinuityResult(
            "", previous_state, recognized=True, failure=f"outfit:{outfit_outcome.failure}"
        )
    if environment_outcome.failure:
        return VisualContinuityResult(
            "", previous_state, recognized=True,
            failure=f"environment:{environment_outcome.failure}",
        )
    if include_outfit:
        effects_outcome, scene_effects_outcome, effects_failure = _resolve_effects(
            raw_effects,
            previous_state.effects,
            previous_state.scene_effects,
            directives.get("effects") if directives is not None else None,
        )
    else:
        effects_outcome = previous_state.effects
        scene_effects_outcome = previous_state.scene_effects
        effects_failure = ""
    if effects_failure:
        return VisualContinuityResult(
            "", previous_state, recognized=True, failure=f"effects:{effects_failure}"
        )
    if include_outfit and not outfit_outcome.tags and not outfit_outcome.cleared:
        # Bot 出镜时服装稳定区必须建立，否则下一轮无从"逐字复用"
        return VisualContinuityResult(
            "", previous_state, recognized=True, failure="outfit:not_established"
        )
    if directives is None and str(stable_change_text or "").strip():
        # scene_delta 兼容路径：Planner 声明了稳定区变化，两区都 keep 视为忽略指令
        modes = {_raw_section_mode(raw_environment)}
        if include_outfit:
            modes.add(_raw_section_mode(raw_outfit))
        if modes == {"keep"}:
            return VisualContinuityResult(
                "", previous_state, recognized=True, failure="stable_change_ignored"
            )

    current_state = StableVisualTags(
        outfit=outfit_outcome.tags,
        environment=environment_outcome.tags,
        effects=effects_outcome,
        scene_effects=scene_effects_outcome,
        last_framing=_as_tags(dynamic.get("framing")),
        outfit_key=outfit_outcome.key,
        environment_key=environment_outcome.key,
        outfits=outfit_outcome.cards,
        environments=environment_outcome.cards,
    )
    # 归一化后去重：避免 dynamic 里的同义写法（下划线/权重壳）复述稳定 Tag
    stable_normalized = frozenset(
        normalized
        for normalized in (
            _normalize_stable_tag(tag)
            for tag in (
                *current_state.outfit,
                *current_state.environment,
                *current_state.effects,
                *current_state.scene_effects,
            )
        )
        if normalized
    )

    def _dynamic_tags(value: Any) -> tuple[str, ...]:
        return tuple(
            tag
            for tag in _as_tags(value)
            if _normalize_stable_tag(tag) not in stable_normalized
        )

    if output_format == "multi":
        prompt, structured, failure = _resolve_multi_prompt(
            dynamic,
            current_state,
            include_outfit=include_outfit,
            stable_normalized=stable_normalized,
        )
        if failure:
            return VisualContinuityResult(
                "", previous_state, recognized=True, failure=failure
            )
        return VisualContinuityResult(
            prompt,
            current_state,
            recognized=True,
            structured=structured,
        )

    prompt = _join_tags(
        (
            _dynamic_tags(dynamic.get("subject")),
            current_state.outfit if include_outfit else (),
            current_state.effects if include_outfit else (),
            _dynamic_tags(dynamic.get("action")),
            _dynamic_tags(dynamic.get("emotion")),
            current_state.environment,
            current_state.scene_effects,
            _dynamic_tags(dynamic.get("scene")),
            _dynamic_tags(dynamic.get("framing")),
        )
    )
    return VisualContinuityResult(prompt, current_state, recognized=True)


# ==================== 失败原因 → 修复指引 ====================

_TOP_FAILURE_HINTS = {
    "not_json": "响应不是可解析的 JSON，必须只输出一行 version=4 JSON，不要任何解释或前后缀。",
    "wrong_version": "JSON 的 version 字段必须是数字 4。",
    "invalid_format": "format 只能是 single 或 multi；画面有两个以上人物时必须使用 multi。",
    "missing_sections": "JSON 缺少 dynamic 或 stable 对象，两者都必须是对象。",
    "missing_rating_subject": "dynamic.subject 的第一个元素必须是唯一的 rating:* tag。",
    "stable_change_ignored": (
        "Planner 已声明稳定区发生变化，stable.outfit / stable.environment 不能都用 keep；"
        "请为发生变化的稳定区输出 replace（或库中已有同款时 switch）。"
    ),
}

_MULTI_FAILURE_HINTS = {
    "people_missing": "multi 必须提供 dynamic.people 人物对象数组。",
    "people_too_few": "multi 的 dynamic.people 至少需要两个独立人物对象。",
    "person_invalid": "dynamic.people 每项必须是含 role/subject/action/emotion 的对象和 Tag 数组。",
    "invalid_role": "每个人物 role 只能是 bot 或 other，且 Bot 女生必须明确标为 bot。",
    "bot_missing": "Bot 出镜的 multi 必须恰有一个 role=bot 的女生人物段。",
    "bot_ambiguous": "multi 只能有一个 role=bot，不能把多个人物都标为 Bot。",
    "bot_not_female": "role=bot 的人物 subject 第一个身份 Tag 必须是 girl、woman 或 female。",
    "person_subject_missing": "每个人物必须提供非空 subject，首项写 girl/boy/woman/man 等身份 Tag。",
    "global_tag_in_person": "rating 和总人数 Tag 只能放 dynamic.subject，不能放进人物段。",
    "unscoped_character_tags": "multi 的顶层 action/emotion 必须为空，动作和情绪分别写入对应人物段。",
    "positions_invalid": "positions 只能为空数组或与人物对应的 A1-E5 坐标数组。",
    "positions_length_mismatch": "非空 positions 必须与 dynamic.people 等长。",
    "interaction_unpaired": (
        "互动绑定没有配对：每个 source#<tag> 必须在另一个人物段有逐字同名的 target#<tag>，反之亦然。"
        "禁止施动方用现在分词、接收方用过去分词（source#groping / target#groped 是错的），"
        "两侧必须是同一个 Danbooru tag，只有前缀不同（source#groping / target#groping）。"
        "自身动作（自己揉胸、抱住自己等）不属于互动，必须去掉前缀写成裸 tag。"
    ),
    "interaction_mutual_unpaired": (
        "mutual#<tag> 表示双方同时互相做同一件事，必须在至少两个人物段同时出现同名 mutual# tag；"
        "只有一方主动时改写成 source#/target# 同名配对。"
    ),
}

_SECTION_FAILURE_HINTS = {
    "section_missing": "stable.{label_key} 必须是包含 mode/key/tags 的对象。",
    "invalid_mode": "stable.{label_key}.mode 只能是 keep / switch / replace / clear。",
    "keep_cannot_initialize": (
        "当前没有可沿用的{label}缓存，keep 不能携带 tags；"
        "请改用 replace 建立完整的{label} Tag，并给出全新的 snake_case key。"
    ),
    "switch_unknown_key": (
        "switch 使用的 key 不在已知{label}库中；只能使用库中列出的 key，"
        "库中没有匹配项时改用 replace 重建。"
    ),
    "switch_target_unresolved": (
        "Planner 要求切回旧{label}：请从已知{label}库中选择唯一匹配的 key 输出 mode=switch；"
        "库中确实没有对应项时改用 mode=replace 依据描述重建完整 Tag。"
    ),
    "replace_incomplete": (
        "replace 必须同时给出全新的 snake_case key 和完整、具体的{label} Tag 列表。"
    ),
    "replace_not_translated": (
        "Planner 已声明更换{label}：请输出 mode=replace 的完整新 Tag（新 key），"
        "或库中已有同款时输出 mode=switch 与库中原 key。"
    ),
    "not_established": (
        "Bot 本轮出镜但{label}稳定区为空；请用 mode=replace 依据聊天上下文"
        "建立完整、具体的{label} Tag。"
    ),
}

_SECTION_LABELS = {
    "outfit": "服装",
    "environment": "环境",
    "effects": "跨图临时效果",
}

_EFFECT_FAILURE_HINTS = {
    "invalid_mode": "stable.effects.mode 只能是 keep / replace / clear，临时效果不支持 switch。",
    "replace_incomplete": "replace 临时效果时必须在 stable.effects.subject / scene 至少输出一组当前完整效果 Tag。",
    "replace_not_translated": "Planner 已声明临时效果变化；请用 replace 输出当前完整效果 Tag，或状态已解除时用 clear。",
}


def describe_visual_failure(failure: str) -> str:
    """把 resolve 的原因码翻译成 Tag LLM 可执行的修复指引。"""

    hint = _TOP_FAILURE_HINTS.get(failure)
    if hint:
        return hint
    kind, _, code = failure.partition(":")
    if kind == "multi" and code in _MULTI_FAILURE_HINTS:
        return _MULTI_FAILURE_HINTS[code]
    if kind == "effects" and code in _EFFECT_FAILURE_HINTS:
        return _EFFECT_FAILURE_HINTS[code]
    label = _SECTION_LABELS.get(kind)
    template = _SECTION_FAILURE_HINTS.get(code)
    if label and template:
        return template.format(label=label, label_key=kind)
    return "稳定区继承协议不合法，请对照规则重新输出完整 JSON。"
