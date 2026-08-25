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
    ALIAS_TO_MODEL_ID,
    ALL_MODEL_IDS,
    ALL_MODEL_SECTIONS,
    MODEL_SPECS,
    max_characters,
    model_label,
    resolve_config_section,
    resolve_spec,
    spec_for_section,
    supports_character_reference,
    supports_multi_character,
    supports_smea,
    supports_vibe_transfer,
)


def test_每个模型_id_都有独立配置节() -> None:
    """配置节按模型 id 划分，不再让 Full 与 Curated 共用一节。

    官方 /image/qualitytags/ 给出的质量词配方在同版本的 Full 与 Curated 之间就是
    不同的（V4.5 Curated 独有 ``-0.8::feet::, rating:general``、Furry V3 只用花括号），
    共用一节就没法分别配质量词和画师预设。
    """
    assert resolve_config_section("nai-diffusion-3") == "model_v3_anime"
    assert resolve_config_section("nai-diffusion-3-furry") == "model_v3_furry"
    assert resolve_config_section("nai-diffusion-4-curated") == "model_v4_curated"
    assert resolve_config_section("nai-diffusion-4-full") == "model_v4_full"
    assert resolve_config_section("nai-diffusion-4-5-curated") == "model_v45_curated"
    assert resolve_config_section("nai-diffusion-4-5-full") == "model_v45_full"
    assert resolve_config_section("nai-diffusion-5-full") == "model_v5_full"
    assert resolve_config_section("nai-diffusion-5-curated") == "model_v5_curated"
    # 8 个模型 → 8 个互不相同的配置节
    assert len(set(ALL_MODEL_SECTIONS)) == 8
    assert resolve_config_section("some-unknown-model") is None
    assert resolve_config_section("") is None


def test_v5_与_v45_不再共用画师预设配置节() -> None:
    """这是本次改动的核心诉求：每个模型适合的画师串不一样，配置必须分开。"""
    assert resolve_config_section("nai-diffusion-5-full") != resolve_config_section(
        "nai-diffusion-4-5-full"
    )


def test_长键优先避免版本前缀互相抢占() -> None:
    """``nai-diffusion-4`` 不能抢先命中 ``nai-diffusion-4-5-full``；
    ``nai-diffusion-3`` 不能抢先命中 ``nai-diffusion-3-furry``。"""
    assert resolve_config_section("nai-diffusion-4-5-full") == "model_v45_full"
    assert resolve_config_section("nai-diffusion-3-furry") == "model_v3_furry"
    assert supports_character_reference("nai-diffusion-4-5-full") is True
    assert supports_character_reference("nai-diffusion-4-full") is False


def test_裸版本名回落到该版本的_full() -> None:
    """网关或用户手填不带 -full / -curated 后缀时仍要解析到合理的版本。"""
    assert resolve_config_section("nai-diffusion-4-5") == "model_v45_full"
    assert resolve_config_section("nai-diffusion-4") == "model_v4_full"
    assert resolve_config_section("nai-diffusion-5") == "model_v5_full"


def test_multi_character_support_matrix() -> None:
    """§7 characters[]：V4 及以上支持，V3 不支持。

    依据 /image/multiplecharacters/ 原文 "only found on our V4 models or higher"。
    """
    assert supports_multi_character("nai-diffusion-4-full") is True
    assert supports_multi_character("nai-diffusion-4-5-full") is True
    assert supports_multi_character("nai-diffusion-5-full") is True
    assert supports_multi_character("nai-diffusion-3") is False
    assert supports_multi_character("nai-diffusion-3-furry") is False


def test_角色数量上限按模型区分() -> None:
    """V4/V4.5 官方上限 6 个；V5 公告实测可到 22 个。"""
    assert max_characters("nai-diffusion-4-full") == 6
    assert max_characters("nai-diffusion-4-5-full") == 6
    assert max_characters("nai-diffusion-5-full") == 22
    assert max_characters("nai-diffusion-5-curated") == 22


def test_v5_has_no_character_reference_or_vibe_transfer() -> None:
    """V5 公告 "Still In Progress" 段明确 Precise Reference 与 Vibe Transfer 未上线。"""
    assert supports_character_reference("nai-diffusion-5-full") is False
    assert supports_vibe_transfer("nai-diffusion-5-full") is False
    assert supports_character_reference("nai-diffusion-5-curated") is False
    assert supports_vibe_transfer("nai-diffusion-5-curated") is False
    # 对照组：V4.5 两者都有
    assert supports_character_reference("nai-diffusion-4-5-full") is True
    assert supports_vibe_transfer("nai-diffusion-4-5-full") is True


def test_角色参考仅_v45_系列() -> None:
    """/image/precisereference/ 原文 "only available on our V4.5 models"。"""
    capable = {spec.model_id for spec in MODEL_SPECS if spec.character_reference}
    assert capable == {"nai-diffusion-4-5-full", "nai-diffusion-4-5-curated"}


def test_smea_仅_v3_及更早() -> None:
    """SMEA 是补偿 V1 时代低训练分辨率的手段，V4 起官方不再提供。

    删掉这条门控，V4+ 就会重新收到无效的 autoSmea 字段。
    """
    assert supports_smea("nai-diffusion-3") is True
    assert supports_smea("nai-diffusion-3-furry") is True
    assert supports_smea("nai-diffusion-4-full") is False
    assert supports_smea("nai-diffusion-4-5-full") is False
    assert supports_smea("nai-diffusion-5-full") is False


def test_数值权重语法门槛按版本递进() -> None:
    """数值权重 V4 起可用，负数值权重 V4.5 起可用（/image/strengthening-weakening/）。"""
    assert resolve_spec("nai-diffusion-3").numeric_emphasis is False
    assert resolve_spec("nai-diffusion-4-full").numeric_emphasis is True
    assert resolve_spec("nai-diffusion-4-full").negative_numeric_emphasis is False
    assert resolve_spec("nai-diffusion-4-5-full").negative_numeric_emphasis is True
    assert resolve_spec("nai-diffusion-5-full").negative_numeric_emphasis is True


def test_官方质量词配方按模型_id_区分() -> None:
    """同版本的 Full 与 Curated 配方不同，这是配置节必须按模型 id 拆的直接依据。"""
    by_id = {spec.model_id: spec.default_quality_tags for spec in MODEL_SPECS}
    assert by_id["nai-diffusion-4-5-full"] == "location, very aesthetic, masterpiece, no text"
    assert by_id["nai-diffusion-4-5-curated"] == (
        "location, masterpiece, no text, -0.8::feet::, rating:general"
    )
    assert by_id["nai-diffusion-4-full"] == "no text, best quality, very aesthetic, absurdres"
    assert by_id["nai-diffusion-4-curated"] == (
        "rating:general, amazing quality, very aesthetic, absurdres"
    )
    assert by_id["nai-diffusion-3"] == "best quality, amazing quality, very aesthetic, absurdres"
    # Furry V3 只认花括号权重，配方也只能用花括号
    assert by_id["nai-diffusion-3-furry"] == "{best quality}, {amazing quality}"
    # NovelAI 尚未公布 V5 的推荐配方，不许凭空编一份
    assert by_id["nai-diffusion-5-full"] == ""
    assert by_id["nai-diffusion-5-curated"] == ""


def test_v3_质量词放末尾() -> None:
    """V3 及更早 prompt 里位置越前权重越高，所以质量词要后置。"""
    assert resolve_spec("nai-diffusion-3").quality_tags_at_end is True
    assert resolve_spec("nai-diffusion-3-furry").quality_tags_at_end is True
    assert resolve_spec("nai-diffusion-4-full").quality_tags_at_end is False


def test_unknown_model_denies_new_capabilities_but_keeps_vibe() -> None:
    """未知模型兜底：只关掉需要显式支持的能力，不误伤历史上一直放行的 vibe。"""
    assert supports_multi_character("brand-new-model") is False
    assert supports_character_reference("brand-new-model") is False
    assert supports_vibe_transfer("brand-new-model") is True
    assert supports_smea("brand-new-model") is False


def test_model_label_is_user_facing() -> None:
    """标签会拼进"当前模型不支持 X"的提示语，必须能区分 Full 与 Curated。"""
    assert model_label("nai-diffusion-5-full") == "V5 Full"
    assert model_label("nai-diffusion-5-curated") == "V5 Curated"
    assert model_label("nai-diffusion-4-5-full") == "V4.5 Full"
    assert model_label("nai-diffusion-4-5-curated") == "V4.5 Curated"
    assert model_label("nai-diffusion-4-full") == "V4 Full"
    assert model_label("nai-diffusion-3") == "V3 Anime"
    assert model_label("nai-diffusion-3-furry") == "V3 Furry"
    assert model_label("whatever") == "未知模型"


def test_capability_lookup_is_case_insensitive() -> None:
    """/nai set 与配置文件里大小写不一致时不能漏判能力。"""
    assert supports_multi_character("NAI-Diffusion-5-Full") is True
    assert resolve_config_section("NAI-DIFFUSION-3") == "model_v3_anime"


def test_命令别名与模型清单同源() -> None:
    """/nai set 的代号、可用模型清单、帮助文案都从注册表派生，避免加模型漏改文案。"""
    assert ALIAS_TO_MODEL_ID["5"] == "nai-diffusion-5-full"
    assert ALIAS_TO_MODEL_ID["5c"] == "nai-diffusion-5-curated"
    assert ALIAS_TO_MODEL_ID["4.5"] == "nai-diffusion-4-5-full"
    assert ALIAS_TO_MODEL_ID["f3"] == "nai-diffusion-3-furry"
    # 别名与模型 id 都不能重复
    assert len(ALIAS_TO_MODEL_ID) == len(MODEL_SPECS)
    assert len(set(ALL_MODEL_IDS)) == len(MODEL_SPECS)


def test_配置节可反查回_spec() -> None:
    """迁移与 WebUI 排序都要按配置节反查模型。"""
    for spec in MODEL_SPECS:
        assert spec_for_section(spec.config_section) is spec
    assert spec_for_section("model_defaults") is None
    assert spec_for_section("nope") is None
