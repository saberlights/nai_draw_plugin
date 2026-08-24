# -*- coding: utf-8 -*-
"""模型能力表的断言。

这张表是"模型名 → 配置节 / 能力开关"的单一事实源；新增模型时漏改能力开关正是
V5 多角色丢失那个 bug 的根因，所以逐个模型钉死预期值。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


dummy_logger_module.get_logger = lambda _name=None: _DummyLogger()
sys.modules["src.common.logger"] = dummy_logger_module

from plugins.nai_draw_plugin.core.model_capabilities import (
    model_label,
    resolve_config_section,
    supports_character_reference,
    supports_multi_character,
    supports_vibe_transfer,
)


def test_config_section_mapping_covers_every_shipped_model() -> None:
    """三处重复的 if-elif 链收敛到这里，映射结果必须与原实现逐一对齐。"""
    assert resolve_config_section("nai-diffusion-3") == "model_nai3"
    assert resolve_config_section("nai-diffusion-3-furry") == "model_nai3"
    assert resolve_config_section("nai-diffusion-4-curated") == "model_nai4"
    assert resolve_config_section("nai-diffusion-4-full") == "model_nai4"
    assert resolve_config_section("nai-diffusion-4-5-curated") == "model_nai4_5"
    assert resolve_config_section("nai-diffusion-4-5-full") == "model_nai4_5"
    # V5 复用 V4.5 的画师串与模型参数配置节
    assert resolve_config_section("nai-diffusion-5-full") == "model_nai4_5"
    assert resolve_config_section("some-unknown-model") is None
    assert resolve_config_section("") is None


def test_v4_5_is_matched_before_v4() -> None:
    """长键优先：``nai-diffusion-4`` 不能抢先命中 ``nai-diffusion-4-5-full``。"""
    assert resolve_config_section("nai-diffusion-4-5-full") == "model_nai4_5"
    assert supports_character_reference("nai-diffusion-4-5-full") is True
    assert supports_character_reference("nai-diffusion-4-full") is False


def test_multi_character_support_matrix() -> None:
    """§7 characters[]：V4 系列与 V5 支持，V3 不支持。"""
    assert supports_multi_character("nai-diffusion-4-full") is True
    assert supports_multi_character("nai-diffusion-4-5-full") is True
    assert supports_multi_character("nai-diffusion-5-full") is True
    assert supports_multi_character("nai-diffusion-3") is False
    assert supports_multi_character("nai-diffusion-3-furry") is False


def test_v5_has_no_character_reference_or_vibe_transfer() -> None:
    """V5 目前只有文生图与图生图，角色参考 / 氛围转移都还没有。"""
    assert supports_character_reference("nai-diffusion-5-full") is False
    assert supports_vibe_transfer("nai-diffusion-5-full") is False
    # 对照组：V4.5 两者都有
    assert supports_character_reference("nai-diffusion-4-5-full") is True
    assert supports_vibe_transfer("nai-diffusion-4-5-full") is True


def test_unknown_model_denies_new_capabilities_but_keeps_vibe() -> None:
    """未知模型兜底：只关掉需要显式支持的能力，不误伤历史上一直放行的 vibe。"""
    assert supports_multi_character("brand-new-model") is False
    assert supports_character_reference("brand-new-model") is False
    assert supports_vibe_transfer("brand-new-model") is True


def test_model_label_is_user_facing() -> None:
    """标签会拼进"当前模型不支持 X"的提示语，必须是人看得懂的档位名。"""
    assert model_label("nai-diffusion-5-full") == "V5"
    assert model_label("nai-diffusion-4-5-full") == "V4.5"
    assert model_label("nai-diffusion-4-full") == "V4"
    assert model_label("nai-diffusion-3") == "V3"
    assert model_label("whatever") == "未知模型"


def test_capability_lookup_is_case_insensitive() -> None:
    """/nai set 与配置文件里大小写不一致时不能漏判能力。"""
    assert supports_multi_character("NAI-Diffusion-5-Full") is True
    assert resolve_config_section("NAI-DIFFUSION-3") == "model_nai3"
