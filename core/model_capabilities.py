# -*- coding: utf-8 -*-
"""
模型注册表：模型能力、硬约束、配置节归属与官方推荐默认值的单一事实源。

原先"模型名 → 配置节"的 if-elif 链在 model_config_mixin / session_state / sdk_runtime
三处各写了一份，"模型能力"又散在 nai_web_client 的两个 ``_filter_*_for_model`` 里，
分辨率与步数上限则是 nai_web_client 的模块常量。新增 NAI V5 时只改了配置节那几处，
漏掉了客户端的多角色白名单，导致 V5 下 ``characters[]`` 被整体丢弃。这里把四类知识
（配置节、能力、约束、官方默认）收敛到一张表，新增模型只需要加一条 ``ModelSpec``。

配置节按**模型 id** 划分而不是按大版本：官方文档 ``/image/qualitytags/`` 给出的质量词
配方在同版本的 Full 与 Curated 之间就是不同的（例如 V4.5 Curated 独有
``-0.8::feet::, rating:general``，Furry V3 只用花括号语法），Curated 的数据集也更小，
适用的画师串不一样。共用一个配置节就没法分别配画师预设。

字段依据分四类，改这张表时请保持标注：
- 【已核实】能在 docs.novelai.net 或官方公告里找到原文依据
- 【沿用现状】不改变本插件既有行为，仅从常量搬到这里
- 【网关相关】取决于 NewAPI 兼容网关的实现，公开文档未定义，仅作 WebUI 候选项用，
  不做请求前拦截（避免用一张没核实的白名单把合法请求拦下）
- 【实测经验】官方文档没有逐条给出配方，来自本项目实际出图对比得到的结论
"""
from typing import Dict, NamedTuple, Optional, Tuple

# 配置缺失 / 未选定时的兜底模型。客户端组装请求与上层能力门控必须用同一个值，
# 否则会出现"门控按未知模型拦下、客户端却按 V4.5 发出去"的错位。
DEFAULT_MODEL_NAME = "nai-diffusion-4-5-full"

# 【沿用现状】原 nai_web_client._MAX_SIZE_* 三个模块常量。
# 【已核实】这一档对应官方的"normal resolution"：Opus 免 Anlas 的条件是单图、不超过
# 约 1024x1024 像素、步数不超过 28（novelai.net 定价页脚注与 FAQ 的 Opus 用量说明）。
NORMAL_SIZES: Dict[str, Tuple[int, int]] = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}

# 【已核实】"images generated with 28 or less Steps won't cost any Anlas"
# （docs.novelai.net/en/image/stepsguidance/），V5 的 Opus 用量说明同样以 28 步为界。
MAX_STEPS = 28

# 【网关相关】官方文档 /image/sampling/ 列出的采样器是：DPM++ 2M、Euler Ancestral、
# Euler、DPM2、DPM++ 2S Ancestral、DPM++ SDE、DPM Fast、DDIM。这里是对应的 API 代号，
# 只用于 WebUI 下拉候选，不做请求前校验。文档里没有 "DPM++ 2M SDE"，故不列。
SAMPLERS: Tuple[str, ...] = (
    "k_euler",
    "k_euler_ancestral",
    "k_dpm_2",
    "k_dpm_2_ancestral",
    "k_dpm_fast",
    "k_dpmpp_2m",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_sde",
    "ddim",
)

# 【网关相关】公开文档没有枚举 noise_schedule 的取值，这三个是本插件既有的白名单，
# 原样保留；留空表示不发送该字段、由网关取默认。
NOISE_SCHEDULES: Tuple[str, ...] = ("karras", "exponential", "polyexponential")


class ModelSpec(NamedTuple):
    """单个模型 id 的配置节归属、能力开关、硬约束与官方推荐默认值。"""

    model_id: str
    # 未知模型没有版本配置节，用 None 表示"跳过版本层配置"
    config_section: Optional[str]
    label: str
    # /nai set 的简写代号，也是 WebUI 与帮助文案的唯一来源
    alias: str

    # ---- 能力门控 ----
    # 【已核实】"A unique feature only found on our V4 models or higher"
    #（/image/multiplecharacters/）
    multi_character: bool
    # 【已核实】V4/V4.5 官方上限 6 个角色；V5 公告实测 "up to 22 distinct characters"
    max_characters: int
    # 【已核实】"Precise Reference is only available on our V4.5 models"
    #（/image/precisereference/）；V5 公告 "Still In Progress" 段明确未随发布上线
    character_reference: bool
    # 【已核实】V3/V4/V4.5 均支持；V5 公告把 Vibe Transfer 列入未上线能力
    vibe_transfer: bool
    # 【已核实】SMEA 是为 V1 时代补偿高分辨率一致性设计的采样器变体，
    # "V2 also marks our increase in training resolution, meaning that it generates
    # higher resolution images without the need of SMEA sampling"（/image/sampling/）。
    # V4 起不再提供，继续发送 autoSmea 是无效字段。
    smea: bool
    # 【已核实】"Numerical Emphasis is only available on V4 or higher models"
    numeric_emphasis: bool
    # 【已核实】"Negative Numerical Emphasis is only available on V4.5 or higher models"
    negative_numeric_emphasis: bool
    # 【已核实】"When using our NovelAI Anime V3 model or lower, the earlier a tag
    # appears in the prompt, the stronger its effect"（/image/tags/），所以 V3 及更早
    # 的质量词要放在 prompt 末尾
    quality_tags_at_end: bool
    # 【实测经验】V5 对英文自然语言从句的理解明显强于 V4.5：主体归属、接触点、
    # 前后遮挡这类 tag 之间说不清的关系，写成一句自然语言比拆成几个孤立 tag 更准。
    # V4.5 及更早仍是纯 tag 模型，喂长句只会稀释每个 tag 的权重，所以这个开关只对
    # V5 打开——它控制的是提示词生成模板要不要放行自然语言，不是请求字段。
    natural_language_prompt: bool

    # ---- 官方推荐默认值 ----
    # 【已核实】/image/qualitytags/ 逐模型给出的质量词配方。V5 官方尚未公布配方，留空。
    default_quality_tags: str
    # 【沿用现状】保持各版本既有 schema 默认，不借这次改动churn用户可见默认值
    default_guidance: float
    default_steps: int
    default_size: str


# 按模型 id 登记。顺序即 WebUI 与帮助文案里的展示顺序（新到旧）。
MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="nai-diffusion-5-full",
        config_section="model_v5_full",
        label="V5 Full",
        alias="5",
        multi_character=True,
        max_characters=22,
        character_reference=False,
        vibe_transfer=False,
        smea=False,
        numeric_emphasis=True,
        negative_numeric_emphasis=True,
        quality_tags_at_end=False,
        natural_language_prompt=True,
        default_quality_tags="",
        default_guidance=5.0,
        default_steps=28,
        default_size="竖图",
    ),
    ModelSpec(
        model_id="nai-diffusion-5-curated",
        config_section="model_v5_curated",
        label="V5 Curated",
        alias="5c",
        multi_character=True,
        max_characters=22,
        character_reference=False,
        vibe_transfer=False,
        smea=False,
        numeric_emphasis=True,
        negative_numeric_emphasis=True,
        quality_tags_at_end=False,
        natural_language_prompt=True,
        default_quality_tags="",
        default_guidance=5.0,
        default_steps=28,
        default_size="竖图",
    ),
    ModelSpec(
        model_id="nai-diffusion-4-5-full",
        config_section="model_v45_full",
        label="V4.5 Full",
        alias="4.5",
        multi_character=True,
        max_characters=6,
        character_reference=True,
        vibe_transfer=True,
        smea=False,
        numeric_emphasis=True,
        negative_numeric_emphasis=True,
        quality_tags_at_end=False,
        natural_language_prompt=False,
        default_quality_tags="location, very aesthetic, masterpiece, no text",
        default_guidance=5.0,
        default_steps=28,
        default_size="竖图",
    ),
    ModelSpec(
        model_id="nai-diffusion-4-5-curated",
        config_section="model_v45_curated",
        label="V4.5 Curated",
        alias="4.5c",
        multi_character=True,
        max_characters=6,
        character_reference=True,
        vibe_transfer=True,
        smea=False,
        numeric_emphasis=True,
        negative_numeric_emphasis=True,
        quality_tags_at_end=False,
        natural_language_prompt=False,
        default_quality_tags="location, masterpiece, no text, -0.8::feet::, rating:general",
        default_guidance=5.0,
        default_steps=28,
        default_size="竖图",
    ),
    ModelSpec(
        model_id="nai-diffusion-4-full",
        config_section="model_v4_full",
        label="V4 Full",
        alias="4",
        multi_character=True,
        max_characters=6,
        character_reference=False,
        vibe_transfer=True,
        smea=False,
        numeric_emphasis=True,
        negative_numeric_emphasis=False,
        quality_tags_at_end=False,
        natural_language_prompt=False,
        default_quality_tags="no text, best quality, very aesthetic, absurdres",
        default_guidance=5.0,
        default_steps=28,
        default_size="竖图",
    ),
    ModelSpec(
        model_id="nai-diffusion-4-curated",
        config_section="model_v4_curated",
        label="V4 Curated",
        alias="4c",
        multi_character=True,
        max_characters=6,
        character_reference=False,
        vibe_transfer=True,
        smea=False,
        numeric_emphasis=True,
        negative_numeric_emphasis=False,
        quality_tags_at_end=False,
        natural_language_prompt=False,
        default_quality_tags="rating:general, amazing quality, very aesthetic, absurdres",
        default_guidance=5.0,
        default_steps=28,
        default_size="竖图",
    ),
    ModelSpec(
        model_id="nai-diffusion-3",
        config_section="model_v3_anime",
        label="V3 Anime",
        alias="3",
        multi_character=False,
        max_characters=1,
        character_reference=False,
        vibe_transfer=True,
        smea=True,
        numeric_emphasis=False,
        negative_numeric_emphasis=False,
        quality_tags_at_end=True,
        natural_language_prompt=False,
        default_quality_tags="best quality, amazing quality, very aesthetic, absurdres",
        default_guidance=3.5,
        default_steps=25,
        default_size="832x1216",
    ),
    ModelSpec(
        model_id="nai-diffusion-3-furry",
        config_section="model_v3_furry",
        label="V3 Furry",
        alias="f3",
        multi_character=False,
        max_characters=1,
        character_reference=False,
        vibe_transfer=True,
        smea=True,
        numeric_emphasis=False,
        negative_numeric_emphasis=False,
        quality_tags_at_end=True,
        natural_language_prompt=False,
        default_quality_tags="{best quality}, {amazing quality}",
        default_guidance=3.5,
        default_steps=25,
        default_size="832x1216",
    ),
)

# 未知模型的兜底：只关掉需要显式支持的能力，不误伤历史上一直放行的 vibe。
# config_section 为 None 表示"没有版本配置节"，上层据此跳过版本层配置。
_UNKNOWN_MODEL = ModelSpec(
    model_id="",
    config_section=None,
    label="未知模型",
    alias="",
    multi_character=False,
    max_characters=1,
    character_reference=False,
    vibe_transfer=True,
    smea=False,
    numeric_emphasis=False,
    negative_numeric_emphasis=False,
    quality_tags_at_end=False,
    natural_language_prompt=False,
    default_quality_tags="",
    default_guidance=5.0,
    default_steps=28,
    default_size="竖图",
)


def _build_match_table() -> Dict[str, ModelSpec]:
    """构造"模型名子串 → ModelSpec"的匹配表。

    除了每个 spec 自己的完整 model_id，额外登记"裸版本名"（如 ``nai-diffusion-4``、
    ``nai-diffusion-4-5``）指向该版本的 Full spec —— 网关或用户手填不带 ``-full`` /
    ``-curated`` 后缀时仍能解析到合理的版本，保持本插件既有行为。
    """
    table: Dict[str, ModelSpec] = {spec.model_id: spec for spec in MODEL_SPECS}
    bare_version_fallbacks = {
        "nai-diffusion-5": "nai-diffusion-5-full",
        "nai-diffusion-4-5": "nai-diffusion-4-5-full",
        "nai-diffusion-4": "nai-diffusion-4-full",
    }
    by_id = {spec.model_id: spec for spec in MODEL_SPECS}
    for bare, target in bare_version_fallbacks.items():
        table.setdefault(bare, by_id[target])
    return table


_MATCH_TABLE = _build_match_table()

# 长键优先的匹配顺序，模块加载时算一次。避免 "nai-diffusion-4" 抢先命中
# "nai-diffusion-4-5-full"，也避免 "nai-diffusion-3" 抢先命中 "nai-diffusion-3-furry"。
_MATCH_ORDER = sorted(_MATCH_TABLE, key=len, reverse=True)

# 配置节名 → spec，供按配置节反查（迁移与 WebUI 排序用）
_BY_SECTION: Dict[str, ModelSpec] = {spec.config_section: spec for spec in MODEL_SPECS}

# 简写代号 → 模型 id，供 /nai set 使用
ALIAS_TO_MODEL_ID: Dict[str, str] = {spec.alias: spec.model_id for spec in MODEL_SPECS}

# 所有登记模型的 id，供 model.available_models 默认值与 default_model 下拉使用
ALL_MODEL_IDS: Tuple[str, ...] = tuple(spec.model_id for spec in MODEL_SPECS)

# 所有版本配置节名，供配置渲染顺序与迁移使用
ALL_MODEL_SECTIONS: Tuple[str, ...] = tuple(spec.config_section for spec in MODEL_SPECS)


def resolve_spec(model_name: str) -> ModelSpec:
    """按模型名解析注册记录；不认识的模型返回保守兜底。"""
    lowered = str(model_name or "").lower()
    if not lowered:
        return _UNKNOWN_MODEL
    for keyword in _MATCH_ORDER:
        if keyword in lowered:
            return _MATCH_TABLE[keyword]
    return _UNKNOWN_MODEL


def spec_for_section(config_section: str) -> Optional[ModelSpec]:
    """配置节名 → ModelSpec；不是版本配置节则返回 None。"""
    return _BY_SECTION.get(config_section)


def resolve_config_section(model_name: str) -> Optional[str]:
    """模型名 → 版本配置节；未知模型返回 None（上层据此跳过版本层配置）。"""
    return resolve_spec(model_name).config_section


def model_label(model_name: str) -> str:
    """面向用户的模型档位标签，用于提示"当前模型不支持 X"。"""
    return resolve_spec(model_name).label


def supports_multi_character(model_name: str) -> bool:
    """是否支持 NewAPI §7 的 ``characters[]`` 多角色通道。"""
    return resolve_spec(model_name).multi_character


def max_characters(model_name: str) -> int:
    """单张图允许的角色段数量上限。"""
    return resolve_spec(model_name).max_characters


def supports_character_reference(model_name: str) -> bool:
    """是否支持 NewAPI §20.4 角色参考（``/nai ref``）。"""
    return resolve_spec(model_name).character_reference


def supports_vibe_transfer(model_name: str) -> bool:
    """是否支持 NewAPI §20.3 Vibe Transfer（``/nai vibe``）。"""
    return resolve_spec(model_name).vibe_transfer


def supports_smea(model_name: str) -> bool:
    """是否还提供 SMEA 类增强（仅 V3 及更早；V4 起该字段无效）。"""
    return resolve_spec(model_name).smea


def supports_natural_language_prompt(model_name: str) -> bool:
    """提示词是否可以在标准 Danbooru tag 之上混写英文自然语言（仅 V5）。"""
    return resolve_spec(model_name).natural_language_prompt
