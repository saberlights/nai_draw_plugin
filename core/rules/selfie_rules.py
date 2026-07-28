# -*- coding: utf-8 -*-
"""
自拍/肖像/画图意图判定模块

提供两类 bot 本人图片场景的判定与提示：
- 自拍（selfie）：拍摄方式本身是重点
- 肖像/生活照（portrait）：看 bot 本人但不强调自拍方式

以及统一的"用户是否明确请求画图/发图"判定，供 Action Guard 节流分级使用。
所有 bot 主动出图判定的关键词，统一收口在本模块。
"""

import re
from typing import List, Tuple

from .constants import (
    BOT_SELF_IMAGE_INTENT_KEYWORDS,
    BOT_SELF_REFERENCE_KEYWORDS,
    EXPLICIT_IMAGE_REQUEST_KEYWORDS,
    EYE_COLOR_KEYWORDS,
    EYE_SPECIAL_TAGS,
    HAIR_RELATED_KEYWORDS,
    NEGATIVE_IMAGE_INTENT_KEYWORDS,
    NEGATIVE_IMAGE_INTENT_KEYWORDS_STRONG,
    NEGATIVE_IMAGE_INTENT_KEYWORDS_WEAK,
    PORTRAIT_OUTPUT_TAGS,
    PORTRAIT_TRIGGER_KEYWORDS,
    SELFIE_OUTPUT_TAGS,
    SELFIE_TRIGGER_KEYWORDS,
)


# ==================== LLM 提示模板 ====================

SELFIE_HINT_FOR_LLM = """
<selfie_portrait_decision>
## 三类意图判定（按优先级从上到下）

| 意图 | 触发线索 | 输出特征 |
|------|----------|----------|
| **肖像/生活照（portrait）** | 用户输入含：肖像 / portrait / 肖像照 | 只表示 bot 本人出镜的普通展示照，不需要任何肖像意图前缀；按用户重点自由选择普通 framing tag |
| **自拍（selfie）** | 用户输入含：自拍 / selfie / 自拍照 | 必含 `selfie` / `mirror selfie` / `group selfie` 之一；除用户或场景明确要求外，不固定视角、景别或手机动作 |
| **普通画图（normal）** | 用户明确要"画一个 X"，与 bot 本人无关 | 按场景生成普通 tag，不加 selfie/portrait 类标签；可补 `solo, 1girl/1boy` |

**【最高优先级】** 用户输入只要含肖像类关键词（肖像/portrait/肖像照），**强制走肖像路径**，禁止输出任何 `selfie` 系标签，即使下方"自拍意图"也匹配。

**【画指定角色优先级 > 肖像/自拍】** 用户输入含具体二次元角色名（如"初音未来"/"蕾姆"/"芙兰朵露"）或前缀 `画指定角色`，**强制走普通画图路径**，本轮主体是角色而非 bot：
- 禁止输出任何 `selfie` / `mirror selfie` / `group selfie` / `portrait photo` / `candid photo` / `upper body portrait` / `full body portrait` 等"bot 出镜"语义 framing
- 即使 description 同时含"肖像照"等 bot 出镜提示词也不要翻译成上述 tag——这些是上游为 bot 出镜准备的兜底，与本轮角色主体冲突
- 需要构图时改用纯 framing tag：`close-up` / `upper body` / `cowboy shot` / `full body`

## 肖像路径输出规则

肖像意图时：
- 不要输出固定肖像意图标签：禁止把"肖像/portrait/肖像照"机械翻译成 `portrait photo` / `candid photo` / `headshot`。
- 不要固定图片架构：不因"肖像"默认加 `upper body`、`close-up`、`pov`、`female pov`、`from above`、`from below` 等额外视角或景别。
- 只在用户明确要求时选择 framing：看脸/头像 → `close-up` 或 `upper body`；看穿搭/全身/腿/鞋 → `full body`；用户没指定时保持普通展示照，由动作、服装、场景和情绪承载画面。
- 若用户要求全身，必须只选全身档 `full body`，禁止同段再出现 `portrait` / `close-up` / `upper body` / `cowboy shot`。
- 可加：`looking at viewer`（直视镜头）、自然光线、合理背景、姿态/动作。
- **禁止**：`selfie` / `mirror selfie` / `group selfie` / `holding phone` / `pov` / `female pov` / `selfie stick`。

## 自拍路径输出规则

- 自拍意图必须含一个自拍语义 tag：默认 `selfie`；用户明确镜子/穿衣镜/对镜 → `mirror selfie`；用户明确合照/多人 → `group selfie`。
- 不要因为自拍默认加 `pov` / `female pov` / `from above` / `from below` / `close-up` / `upper body` / `holding phone`。这些只在用户明确要求、或镜子自拍确实需要手机入镜时使用。
- 用户说全身/全身照/全身穿搭/腿/鞋/袜子时：必须在人数之后立即写 `full body`，且同段禁止 `close-up` / `portrait` / `upper body` / `cowboy shot`；优先补 `standing`、能看清腿脚的服装和背景，而不是切成半身自拍。
- 镜子自拍只有在用户说镜子/对镜/穿衣镜，或上下文明确是镜前场景时才用；此时可写 `mirror selfie, holding phone`，但全身请求仍必须保留 `full body`。
- 高低角度只响应明确角度词；展示下半身/腿/鞋时禁止 `from above`。

## 自拍 + 肖像通用要求
- 默认是 bot 本人出镜（非二创角色），所以默认**不要补充角色名/作品名/版权 tag**（`character (series)`、cosplay 名等）
- 仅当用户明确要求 cosplay/cos/扮演某角色时，才输出 cosplay 标签：
  - **格式**：`被 cosplay 的角色名_(cosplay)`，如 `shiranui_mai_(cosplay)`、`hatsune_miku_(cosplay)`、`rem_(cosplay)`
  - **禁止**使用 `角色名 (作品名)` 格式（如 `hatsune miku (vocaloid)`），那是画角色本人而非 cosplay
  - **禁止**单独输出 `cosplay` 标签，必须与角色名组合成 `角色名_(cosplay)` 形式
  - cosplay 时出镜的是 bot 本人，只是穿着该角色的服装/发型，不要补充该角色的默认外貌特征
- **不要输出 `selfie stick` / `holding selfie stick`**
- **不要输出 `arm up`**（自拍是手臂前伸而非向上举）
- 前置自拍手机默认在画面外，不加 `holding phone` / `smartphone`；只有镜子自拍才加 `holding phone`
- 不重复表达同一概念（`mirror selfie` 已含镜子，不再加 `mirror`/`reflection`）

## 服装与连续性（与主模板 _HARD_RULES.6 保持一致）
- 没有上下文 → 默认不凭空补服装；只有用户描述或场景逻辑明确需要时，才补最少且具体的款式、颜色或材质，禁止 `casual wear` 这类宽泛词
- 有连续性上下文且用户没说要换装 → 延续上一轮的服装款式、主色、材质
- 看腿/袜子/鞋/全身穿搭 → 必须用 `full body`，不要切到近景、半身或固定角度

## 类型连续性（避免跳变）
当用户说"再来一张/换个姿势/继续/还是这个/这身/这套"等连续请求时，**默认延续上一轮的图片类型**，仅修改用户明确指定的部分：

- 上一轮是**自拍**（输出含 `selfie` / `mirror selfie` / `group selfie`）→ 本轮默认仍是自拍，并沿用用户明确指定过的自拍语义；不要自动改角度或景别
- 上一轮是**肖像**（用户原话或上下文指向肖像/生活照）→ 本轮默认仍是肖像/生活照，并沿用用户明确指定过的构图重点；不要补 `portrait photo` / `candid photo`
- 上一轮是**普通画图**（无自拍/肖像标签）→ 本轮默认仍是普通画图，不强加 selfie/portrait 类标签

**只有以下情况才允许切换类型**：
- 用户本轮明确要求自拍：含"自拍/selfie"等关键词 → 切到自拍路径
- 用户本轮明确要求肖像：含"肖像/portrait"等关键词 → 切到肖像路径
- 用户本轮明确要求"换成普通画图/画一个X"→ 切到普通画图

切换类型时，仍延续场景、服装、光线、时间氛围等可继承元素，仅改拍摄方式/构图。

</selfie_portrait_decision>
""".strip()


# ==================== 检测函数 ====================

def detect_selfie_mode(description: str) -> bool:
    """
    检测是否触发"bot 本人图片"模式（自拍 OR 肖像）。

    触发后：
    - 注入 selfie_prompt_add 配置中的 bot 角色特征（黑发/挑染/瞳色等）
    - 走 selfie 后处理路径

    Args:
        description: 用户输入的描述

    Returns:
        bool: 是否需要走 bot 本人图片路径
    """
    description_lower = description.lower()

    for keyword in SELFIE_TRIGGER_KEYWORDS:
        if keyword.lower() in description_lower:
            return True
    for keyword in PORTRAIT_TRIGGER_KEYWORDS:
        if keyword.lower() in description_lower:
            return True
    return False


def detect_portrait_intent(description: str) -> bool:
    """
    检测是否为肖像意图（不要自拍标签）。

    用于在 LLM 调用前给出明确指令"本次禁止输出 selfie 系标签"。

    Args:
        description: 用户输入的描述

    Returns:
        bool: 是否为肖像意图
    """
    description_lower = description.lower()
    for keyword in PORTRAIT_TRIGGER_KEYWORDS:
        if keyword.lower() in description_lower:
            return True
    return False


def detect_bot_self_image_intent(text: str) -> bool:
    """判断输入文本是否明确指向 bot 本人图片（自拍/肖像/想看 bot 自己）。

    覆盖 selfie/portrait 触发词 + 隐式"想看 bot 本人"表达（"看看你"/"你的腿"等）。

    用于 ``/nai`` 命令路径决定是否走 ``_process_selfie_prompt`` 后处理（注入 bot
    默认外貌、删冲突发色/瞳色 tag）。与 ``detect_selfie_from_output`` 的区别：后者
    从 LLM 输出标签反推，会把作为 framing 的 ``portrait photo``/``full body
    portrait`` 误判成"bot 本人图片"，结果把用户点名的二次元角色（如 ``中野二乃``）
    洗成 bot 自己。

    Args:
        text: 用户原话或等价的中文意图描述。

    Returns:
        bool: 是否需要走 bot 自拍/肖像后处理。
    """
    if not text:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in BOT_SELF_IMAGE_INTENT_KEYWORDS)


# LLM 输出中表示自拍的标签
_SELFIE_OUTPUT_TAGS = [
    "selfie", "mirror selfie", "group selfie",
    "self-shot", "self shot",
]

# LLM 输出中表示肖像的标签
_PORTRAIT_OUTPUT_TAGS = [
    "portrait photo", "candid photo",
    "upper body portrait", "full body portrait",
    "headshot",
]


def detect_selfie_from_output(prompt: str) -> bool:
    """从 LLM 生成的提示词中检测是否为 bot 本人图片（自拍或肖像）。

    返回 True 时下游会执行"合并 bot 角色特征 + 移除冲突外貌标签"等后处理；
    所以语义上覆盖自拍（selfie / mirror selfie / group selfie）和肖像
    （portrait photo / candid photo / upper body portrait / full body portrait）两类——
    它们都是"bot 本人出镜的图片"，需要叠加配置中的角色特征。

    历史命名保留以避免大范围重命名调用点，新代码可优先用
    detect_bot_self_image_from_output（同语义别名）。
    """
    prompt_lower = prompt.lower()
    if any(tag in prompt_lower for tag in _SELFIE_OUTPUT_TAGS):
        return True
    if any(tag in prompt_lower for tag in _PORTRAIT_OUTPUT_TAGS):
        return True
    return False


def detect_bot_self_image_from_output(prompt: str) -> bool:
    """detect_selfie_from_output 的语义清晰别名，新代码推荐使用。"""
    return detect_selfie_from_output(prompt)


def detect_portrait_from_output(prompt: str) -> bool:
    """从 LLM 生成的提示词中检测是否含肖像标签。"""
    prompt_lower = prompt.lower()
    return any(tag in prompt_lower for tag in PORTRAIT_OUTPUT_TAGS)


def get_selfie_hint() -> str:
    """获取自拍/肖像决策提示文本，注入到 LLM 模板。"""
    return SELFIE_HINT_FOR_LLM


def get_portrait_enforcement_hint() -> str:
    """
    肖像意图触发时追加的硬指令，保证 LLM 不把展示照固化成自拍或肖像前缀。

    应在 prompt 渲染时拼接到 user_request 段附近。
    """
    return (
        "<portrait_enforcement>\n"
        "【本轮强制约束】用户请求肖像/portrait 类图片，输出必须满足：\n"
        "- 不添加 `portrait photo` / `candid photo` / `headshot` 等固定肖像前缀\n"
        "- 不默认添加 `close-up` / `upper body` / `pov` / `female pov` 或高低角度；仅响应用户明确指定的构图和视角\n"
        "- 用户要求全身/穿搭/腿/鞋时只用 `full body`，不得叠加其它 framing\n"
        "- 绝对禁止输出：`selfie` / `mirror selfie` / `group selfie` / `holding phone` / `selfie stick`\n"
        "</portrait_enforcement>"
    )


# ==================== 后处理：合并角色特征 ====================

def merge_selfie_prompt(generated_prompt: str, selfie_prompt_add: str) -> str:
    """
    智能合并自拍/肖像提示词，配置中的角色特征优先。

    Args:
        generated_prompt: LLM 生成的提示词
        selfie_prompt_add: 配置文件中的 bot 角色特征

    Returns:
        合并后的提示词
    """
    if not selfie_prompt_add:
        return generated_prompt

    # 解析要添加的角色特征标签
    add_tags = [
        tag.strip()
        for tag in selfie_prompt_add.split(",")
        if tag.strip()
    ]

    if not add_tags:
        return generated_prompt

    def normalize_tag(tag: str) -> str:
        """移除常见权重包装，便于判断外貌冲突。"""
        tag = tag.strip()
        tag = re.sub(r"^[+-]?\d+(?:\.\d+)?::", "", tag).strip()
        tag = re.sub(r"::\s*$", "", tag).strip()
        tag = tag.strip("{}[]() ")
        return re.sub(r"\s+", " ", tag.lower()).strip()

    def is_hair_related(tag: str) -> bool:
        core = normalize_tag(tag)
        return any(keyword in core for keyword in HAIR_RELATED_KEYWORDS)

    def is_eye_related(tag: str) -> bool:
        core = normalize_tag(tag)
        if core in EYE_SPECIAL_TAGS:
            return True
        match = re.search(r"\b([a-z]+)\s+eyes\b", core)
        if match and match.group(1) in EYE_COLOR_KEYWORDS:
            return True
        return bool(re.search(r"\b[a-z]+-eyed\b", core))

    has_hair_anchor = any(is_hair_related(tag) for tag in add_tags)
    has_eye_anchor = any(is_eye_related(tag) for tag in add_tags)

    # 解析 LLM 生成的标签，移除与配置冲突的标签
    generated_tags = [
        tag.strip()
        for tag in generated_prompt.replace("\n", ",").split(",")
        if tag.strip()
    ]

    filtered_tags = []
    for tag in generated_tags:
        if has_hair_anchor and is_hair_related(tag):
            continue
        if has_eye_anchor and is_eye_related(tag):
            continue
        filtered_tags.append(tag)

    # 保持 LLM 主 prompt 在前，自拍固定人设在后补强。
    merged_parts = []
    filtered_prompt = ", ".join(filtered_tags).strip(", ")
    if filtered_prompt:
        merged_parts.append(filtered_prompt)
    merged_parts.append(", ".join(add_tags))
    merged = ", ".join(part for part in merged_parts if part)

    return merged.strip(", ")


# ==================== Action Guard 判定 ====================

def detect_explicit_image_request(text: str) -> bool:
    """判断用户原话是否含明确的看图/画图/发图/自拍/肖像/追图请求。

    命中即视为"用户显式请求"：Action Guard 节流走 explicit 短间隔档。
    未命中但 Planner 仍调用了 Action，视为"bot 主动发图"，走 proactive 长间隔档。
    """
    if not text:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in EXPLICIT_IMAGE_REQUEST_KEYWORDS)


def detect_negative_image_intent(text: str) -> bool:
    """判断用户原话是否明确拒绝出图（strong + weak 合并视图）。

    保留以兼容旧调用方；新代码请用 ``detect_negative_image_intent_strength``
    拿到具体档位，按 stale 与否做差异化拦截。
    """
    if not text:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in NEGATIVE_IMAGE_INTENT_KEYWORDS)


def detect_negative_image_intent_strength(text: str) -> str:
    """返回否定强度：``"strong"`` / ``"weak"`` / ``""``。

    - strong：明确拒绝出图（"不要画" / "别画"），命中应永久拦截
    - weak：偏好文字回复（"用文字" / "文字就行"），拦截前再做 stale 判定
    - 空串：无否定信号
    """
    if not text:
        return ""
    lowered = text.lower()
    if any(keyword.lower() in lowered for keyword in NEGATIVE_IMAGE_INTENT_KEYWORDS_STRONG):
        return "strong"
    if any(keyword.lower() in lowered for keyword in NEGATIVE_IMAGE_INTENT_KEYWORDS_WEAK):
        return "weak"
    return ""
