# -*- coding: utf-8 -*-
"""1.x → 2.0.0 配置迁移的断言。

迁移会整份重写 config.toml（配置节要改名），用户手调的画师预设与长负面词全在里面，
所以这里逐字段钉死"搬过去的值必须原样保留"，任何静默丢数据都要红。
"""
from __future__ import annotations

import sys
import tomllib
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
sys.modules.setdefault("src.common.logger", dummy_logger_module)

from plugins.nai_draw_plugin.core.config_migration import (  # noqa: E402
    CONFIG_VERSION_2,
    migrate_to_v2,
    needs_migration,
)

_LEGACY_CONFIG = """
[plugin]
name = "nai_draw_plugin"
config_version = "1.8.0"
enabled = true

[model]
base_url = "https://example.com"
api_key = "sk-test"
default_model = "nai-diffusion-4-5-full"

[model_nai4_5]
nai_size = "竖图"
sampler = "k_euler_ancestral"
num_inference_steps = 28
guidance_scale = 6.0
seed = -1
quality_toggle = true
auto_smea = true
variety_boost = true
cfg_rescale = 0.0
noise_schedule = "karras"
image_format = "png"
default_size = "832x1216"
custom_prompt_add = "5::best quality::"
negative_prompt_add = "v45-negative"
selfie_prompt_add = "warm_honey-brown_hair"
selfie_negative_prompt_add = "1.3::aged_down::"
default_artist_preset = ""
nai_artist_prompt = ""

[[model_nai4_5.artist_presets]]
name = "channel风"
prompt = "1.4::kazutake hazano::"

[[model_nai4_5.artist_presets]]
name = "厚涂风"
prompt = "1.2::efe::"
negative_prompt_add = "preset-negative"

[model_nai4]
nai_size = "竖图"
sampler = "k_euler_ancestral"
num_inference_steps = 28
guidance_scale = 5.0
seed = -1
quality_toggle = true
auto_smea = true
variety_boost = false
cfg_rescale = 0.0
noise_schedule = ""
image_format = "png"
default_size = "832x1216"
custom_prompt_add = "{{masterpiece}}"
negative_prompt_add = "v4-negative"
selfie_prompt_add = "{{black hair}}"
selfie_negative_prompt_add = "{{loli}}"
default_artist_preset = ""
nai_artist_prompt = ""

[[model_nai4.artist_presets]]
name = "风格组合1"
prompt = "1.2::artist1::"

[model_nai3]
nai_size = "832x1216"
sampler = "k_euler_ancestral"
num_inference_steps = 25
guidance_scale = 3.5
seed = -1
quality_toggle = true
auto_smea = true
variety_boost = false
cfg_rescale = 0.0
noise_schedule = ""
image_format = "png"
default_size = "832x1216"
custom_prompt_add = "{masterpiece}"
negative_prompt_add = "v3-negative"
selfie_prompt_add = "{{slim body}}"
selfie_negative_prompt_add = "{child}"
default_artist_preset = ""
nai_artist_prompt = ""

[model_nai4_5.nai_extra_params]
legacy_key = "legacy_value"

[tag_retriever]
enabled = true
mode = "online"
"""


def _write_legacy(tmp_path: Path, text: str = _LEGACY_CONFIG) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(text.lstrip(), encoding="utf-8")
    return config_path


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_只在存在旧模型节时才迁移(tmp_path) -> None:
    """迁移会整份重写文件，触发条件必须收紧到真正需要搬结构的场景。"""
    legacy = _write_legacy(tmp_path)
    assert needs_migration(legacy) is True

    # 结构已经正确的配置（哪怕版本号偏低）不该被重写，否则用户手写注释会被冲掉
    modern = tmp_path / "modern.toml"
    modern.write_text(
        '# 用户自己的注释\n[plugin]\nconfig_version = "1.0.0"\n'
        '[model_defaults]\nseed = -1\n',
        encoding="utf-8",
    )
    assert needs_migration(modern) is False
    original = modern.read_text(encoding="utf-8")
    assert bool(migrate_to_v2(modern)) is False
    assert modern.read_text(encoding="utf-8") == original


def test_缺失或损坏的配置不会被迁移(tmp_path) -> None:
    assert needs_migration(tmp_path / "不存在.toml") is False

    broken = tmp_path / "broken.toml"
    broken.write_text("这不是 = = 合法 TOML [[[", encoding="utf-8")
    # 解析不了就别动它，让后续加载流程把真实错误抛出来
    assert needs_migration(broken) is False


def test_迁移后旧节消失新节就位(tmp_path) -> None:
    config_path = _write_legacy(tmp_path)
    assert bool(migrate_to_v2(config_path)) is True

    data = _load(config_path)
    for legacy in ("model_nai3", "model_nai4", "model_nai4_5"):
        assert legacy not in data
    for new_section in (
        "model_defaults",
        "model_v5_full",
        "model_v5_curated",
        "model_v45_full",
        "model_v45_curated",
        "model_v4_full",
        "model_v4_curated",
        "model_v3_anime",
        "model_v3_furry",
    ):
        assert new_section in data, f"缺少新配置节 {new_section}"
    assert data["plugin"]["config_version"] == CONFIG_VERSION_2


def test_画师预设原样搬到对应模型节(tmp_path) -> None:
    """用户手调的预设是最不能丢的数据。"""
    config_path = _write_legacy(tmp_path)
    migrate_to_v2(config_path)
    data = _load(config_path)

    v45_presets = data["model_v45_full"]["artist_presets"]
    assert len(v45_presets) == 2
    assert v45_presets[0] == {"name": "channel风", "prompt": "1.4::kazutake hazano::"}
    assert v45_presets[1]["negative_prompt_add"] == "preset-negative"

    assert data["model_v4_full"]["artist_presets"] == [
        {"name": "风格组合1", "prompt": "1.2::artist1::"}
    ]
    # 没有 1.x 对应关系的模型拿到空预设库，等用户自己填
    assert data["model_v5_full"]["artist_presets"] == []
    assert data["model_v45_curated"]["artist_presets"] == []


def test_每模型的提示词与参数逐字段保留(tmp_path) -> None:
    config_path = _write_legacy(tmp_path)
    original = _load(config_path)
    migrate_to_v2(config_path)
    data = _load(config_path)

    mapping = {
        "model_nai4_5": "model_v45_full",
        "model_nai4": "model_v4_full",
        "model_nai3": "model_v3_anime",
    }
    fields = (
        "custom_prompt_add",
        "negative_prompt_add",
        "selfie_prompt_add",
        "selfie_negative_prompt_add",
        "nai_size",
        "num_inference_steps",
        "guidance_scale",
        "variety_boost",
        "noise_schedule",
        "default_artist_preset",
        "nai_artist_prompt",
    )
    for old, new in mapping.items():
        for field in fields:
            assert data[new][field] == original[old][field], f"{old}.{field} 迁移后变了"


def test_三节一致的共享标量收进_model_defaults(tmp_path) -> None:
    config_path = _write_legacy(tmp_path)
    migrate_to_v2(config_path)
    data = _load(config_path)

    defaults = data["model_defaults"]
    assert defaults["sampler"] == "k_euler_ancestral"
    assert defaults["seed"] == -1
    assert defaults["quality_toggle"] is True
    assert defaults["image_format"] == "png"
    assert defaults["default_size"] == "832x1216"

    # 收敛掉的字段不该在模型节里重复出现
    for section in ("model_v45_full", "model_v4_full", "model_v3_anime"):
        for field in ("sampler", "seed", "quality_toggle", "image_format", "default_size"):
            assert field not in data[section], f"{section}.{field} 应已收敛到 model_defaults"

    # 各模型的负面/自拍词留在自己节里，共享层留空 —— 保证合并结果与迁移前一致
    assert defaults["negative_prompt_add"] == ""
    assert defaults["selfie_prompt_add"] == ""


def test_共享标量取值冲突时按当前模型取值并报告(tmp_path) -> None:
    """三节取值不一致时不能静默挑一个，必须把被舍弃的值报出来。"""
    conflicting = _LEGACY_CONFIG.replace(
        '[model_nai3]\nnai_size = "832x1216"\nsampler = "k_euler_ancestral"',
        '[model_nai3]\nnai_size = "832x1216"\nsampler = "k_dpmpp_2m"',
    )
    config_path = _write_legacy(tmp_path, conflicting)
    report = migrate_to_v2(config_path)

    data = _load(config_path)
    # default_model 是 V4.5，所以共享层取 V4.5 那节的值
    assert data["model_defaults"]["sampler"] == "k_euler_ancestral"

    # 被舍弃的值必须出现在报告里，否则用户无从知道 V3 的采样器被改掉了
    conflict_notes = [note for note in report.conflicts if "sampler" in note]
    assert conflict_notes, f"取值冲突必须被报告，实际 conflicts={report.conflicts}"
    assert "k_dpmpp_2m" in conflict_notes[0]
    assert "model_nai3" in conflict_notes[0]


def test_取值一致时不产生冲突报告(tmp_path) -> None:
    config_path = _write_legacy(tmp_path)
    report = migrate_to_v2(config_path)
    assert report.conflicts == []


def test_auto_smea_只保留给_v3(tmp_path) -> None:
    """SMEA 是 V3 及更早的能力；V4+ 此前发的 autoSmea 是无效字段，迁移时明确丢弃。"""
    config_path = _write_legacy(tmp_path)
    report = migrate_to_v2(config_path)
    data = _load(config_path)

    assert data["model_v3_anime"]["auto_smea"] is True
    assert "auto_smea" not in data["model_v45_full"]
    assert "auto_smea" not in data["model_v4_full"]

    # 丢弃动作必须留痕，不能静默去掉一个用户开着的开关
    dropped = [note for note in report.notes if "auto_smea" in note]
    assert len(dropped) == 2, f"V4.5 与 V4 各该有一条丢弃说明，实际 {report.notes}"


def test_迁移报告记录每个节的搬运结果(tmp_path) -> None:
    config_path = _write_legacy(tmp_path)
    report = migrate_to_v2(config_path)

    assert report.migrated is True
    assert report.backup_path is not None and report.backup_path.is_file()
    moves = [note for note in report.notes if "→" in note]
    assert any("model_nai4_5 → model_v45_full（画师预设 2 条）" == note for note in moves)
    assert any("model_nai4 → model_v4_full（画师预设 1 条）" == note for note in moves)
    assert any("model_nai3 → model_v3_anime（画师预设 0 条）" == note for note in moves)


def test_额外透传参数从子表转成键值数组(tmp_path) -> None:
    """Dashboard 没有 dict 编辑器，子表形态在 WebUI 里根本编不了。"""
    config_path = _write_legacy(tmp_path)
    migrate_to_v2(config_path)
    data = _load(config_path)

    assert data["model_v45_full"]["nai_extra_params"] == [
        {"key": "legacy_key", "value": "legacy_value"}
    ]
    assert data["model_v4_full"]["nai_extra_params"] == []


def test_非模型配置节完整保留(tmp_path) -> None:
    """迁移只负责搬模型节，其它节的用户值一个都不能变。

    注意迁移会按 schema 重渲染整份文件，所以缺失字段会被补上默认值 —— 断言的是
    "用户写过的值不变"，而不是"字段集合不变"。
    """
    config_path = _write_legacy(tmp_path)
    original = _load(config_path)
    migrate_to_v2(config_path)
    data = _load(config_path)

    for field, value in original["tag_retriever"].items():
        assert data["tag_retriever"][field] == value, f"tag_retriever.{field} 被改了"
    assert data["model"]["api_key"] == "sk-test"
    assert data["model"]["base_url"] == "https://example.com"
    assert data["model"]["default_model"] == "nai-diffusion-4-5-full"


def test_迁移会补齐缺失字段的默认值(tmp_path) -> None:
    """按 schema 重渲染的副作用：配置文件变完整，且新字段自带说明注释。"""
    config_path = _write_legacy(tmp_path)
    before = _load(config_path)
    assert "search_limit" not in before["tag_retriever"]

    migrate_to_v2(config_path)
    after = _load(config_path)

    assert after["tag_retriever"]["search_limit"] == 30
    assert "# 在线 /search 返回标签上限" in config_path.read_text(encoding="utf-8")


def test_迁移前留下原文备份(tmp_path) -> None:
    config_path = _write_legacy(tmp_path)
    original_text = config_path.read_text(encoding="utf-8")
    migrate_to_v2(config_path)

    backup = config_path.with_suffix(f"{config_path.suffix}.pre-{CONFIG_VERSION_2}")
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == original_text


def test_重复迁移不会覆盖更早的备份(tmp_path) -> None:
    """第二次迁移必须另起备份名，否则会把真正的原始版本冲掉。"""
    config_path = _write_legacy(tmp_path)
    first_text = config_path.read_text(encoding="utf-8")
    migrate_to_v2(config_path)

    # 人为退回 1.x 结构再迁一次
    config_path.write_text(first_text, encoding="utf-8")
    migrate_to_v2(config_path)

    base_backup = config_path.with_suffix(f"{config_path.suffix}.pre-{CONFIG_VERSION_2}")
    second_backup = base_backup.with_name(f"{base_backup.name}.1")
    assert base_backup.is_file()
    assert second_backup.is_file()


def test_迁移结果可被正常解析且带字段注释(tmp_path) -> None:
    config_path = _write_legacy(tmp_path)
    migrate_to_v2(config_path)
    text = config_path.read_text(encoding="utf-8")

    tomllib.loads(text)  # 解析不了会直接抛
    assert "# V4.5 Full 图片尺寸" in text
    assert "# ----- V4.5 Full -----" in text
    # 迁移后的文件不该再出现"作用同 model_nai4_5.xxx"这种零信息量注释
    assert "作用同 model_nai4_5" not in text
