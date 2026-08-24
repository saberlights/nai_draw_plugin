# -*- coding: utf-8 -*-
"""
模型能力与配置节的单一事实源。

原先"模型名 → 配置节"的 if-elif 链在 model_config_mixin / session_state / sdk_runtime
三处各写了一份，"模型能力"又散在 nai_web_client 的两个 `_filter_*_for_model` 里。
新增 NAI V5 时只改了配置节那几处，漏掉了客户端的多角色白名单，导致 V5 下 `characters[]`
被整体丢弃、角色段静默消失（出图角色完全不对）。这里把两类知识收敛到一张表，
新增模型只需要加一条 `_MODEL_CAPABILITIES` 记录。

能力取值的依据：
- ``multi_character``：NewAPI §7 ``characters[]`` / ``use_coords``，V4 系列与 V5 均支持。
- ``character_reference``：NewAPI §20.4，只有 V4.5 系列支持。
- ``vibe_transfer``：NewAPI §20.3 controlnet。V5 暂未提供该能力（V5 目前只有文生图与
  图生图）；其余模型沿用历史行为，不额外收窄。
"""
from typing import Dict, NamedTuple, Optional

# 配置缺失 / 未选定时的兜底模型。客户端组装请求与上层能力门控必须用同一个值，
# 否则会出现"门控按未知模型拦下、客户端却按 V4.5 发出去"的错位。
DEFAULT_MODEL_NAME = "nai-diffusion-4-5-full"


class ModelCapability(NamedTuple):
    """单个模型系列的配置节归属与能力开关。"""

    config_section: Optional[str]
    multi_character: bool
    character_reference: bool
    vibe_transfer: bool
    label: str


# 按模型名匹配的能力表。键是模型名子串，匹配时**长键优先**，
# 避免 "nai-diffusion-4" 抢先命中 "nai-diffusion-4-5"。
_MODEL_CAPABILITIES: Dict[str, ModelCapability] = {
    "nai-diffusion-3": ModelCapability(
        config_section="model_nai3",
        multi_character=False,
        character_reference=False,
        vibe_transfer=True,
        label="V3",
    ),
    "nai-diffusion-4-5": ModelCapability(
        config_section="model_nai4_5",
        multi_character=True,
        character_reference=True,
        vibe_transfer=True,
        label="V4.5",
    ),
    "nai-diffusion-4": ModelCapability(
        config_section="model_nai4",
        multi_character=True,
        character_reference=False,
        vibe_transfer=True,
        label="V4",
    ),
    # V5 复用 V4.5 的画师串与模型参数配置节；角色参考与氛围转移暂未开放
    "nai-diffusion-5": ModelCapability(
        config_section="model_nai4_5",
        multi_character=True,
        character_reference=False,
        vibe_transfer=False,
        label="V5",
    ),
}

# 长键优先的匹配顺序，模块加载时算一次
_MATCH_ORDER = sorted(_MODEL_CAPABILITIES, key=len, reverse=True)

# 未知模型的兜底：只关掉需要显式支持的新能力，不误伤历史上一直放行的 vibe
_UNKNOWN_MODEL = ModelCapability(
    config_section=None,
    multi_character=False,
    character_reference=False,
    vibe_transfer=True,
    label="未知模型",
)


def resolve_capability(model_name: str) -> ModelCapability:
    """按模型名解析能力记录；不认识的模型返回保守兜底。"""
    lowered = str(model_name or "").lower()
    if not lowered:
        return _UNKNOWN_MODEL
    for keyword in _MATCH_ORDER:
        if keyword in lowered:
            return _MODEL_CAPABILITIES[keyword]
    return _UNKNOWN_MODEL


def resolve_config_section(model_name: str) -> Optional[str]:
    """模型名 → 版本配置节（``model_nai3`` / ``model_nai4`` / ``model_nai4_5``）。"""
    return resolve_capability(model_name).config_section


def model_label(model_name: str) -> str:
    """面向用户的模型档位标签，用于提示"当前模型不支持 X"。"""
    return resolve_capability(model_name).label


def supports_multi_character(model_name: str) -> bool:
    """是否支持 NewAPI §7 的 ``characters[]`` 多角色通道。"""
    return resolve_capability(model_name).multi_character


def supports_character_reference(model_name: str) -> bool:
    """是否支持 NewAPI §20.4 角色参考（``/nai ref``）。"""
    return resolve_capability(model_name).character_reference


def supports_vibe_transfer(model_name: str) -> bool:
    """是否支持 NewAPI §20.3 Vibe Transfer（``/nai vibe``）。"""
    return resolve_capability(model_name).vibe_transfer
