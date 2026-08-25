# -*- coding: utf-8 -*-
"""config.toml 从 1.x 到 2.0.0 的一次性迁移。

1.x 把 8 个 NAI 模型塞进 3 个配置节（``model_nai3`` / ``model_nai4`` / ``model_nai4_5``），
同版本的 Full 与 Curated 共用一节、V5 直接借用 V4.5 的节 —— 于是画师预设无法按模型分开。
2.0.0 改成"一个模型 id 一节 + 一个 [model_defaults] 共享层"。

迁移原则是**不静默丢数据**：
- 三节各自的 artist_presets / 质量词 / 负面词 / 自拍词 / 尺寸 / 步数 / CFG 原样搬到
  对应的新节，不做合并
- 只有确认三节取值一致的共享标量才收进 [model_defaults]；一旦发现取值不一致，
  按当前 default_model 所属那节取值，并把被舍弃的值打进日志让用户自己决定
- 老节里 2.0.0 已经不再提供的字段（例如 V4+ 的 auto_smea）会明确记录在日志里
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tomllib
import tomlkit

from src.common.logger import get_logger

from .model_capabilities import DEFAULT_MODEL_NAME, resolve_config_section, supports_smea

logger = get_logger("nai_draw_plugin")

CONFIG_VERSION_2 = "2.0.0"

# 老配置节 → 新配置节。老节里的 Curated 变体没有独立配置，迁移后由用户按需填。
_OLD_TO_NEW_SECTION: Dict[str, str] = {
    "model_nai4_5": "model_v45_full",
    "model_nai4": "model_v4_full",
    "model_nai3": "model_v3_anime",
}

# 迁移后由 [model_defaults] 统一承载的共享标量。
# 这些字段在 1.x 的三节里值完全相同（纯冗余复制），所以才收敛。
_SHARED_SCALAR_FIELDS: Tuple[str, ...] = (
    "sampler",
    "seed",
    "quality_toggle",
    "cfg_rescale",
    "image_format",
    "default_size",
)

# 直接按模型搬运的字段（2.0.0 的每模型节里仍然存在）
_PER_MODEL_FIELDS: Tuple[str, ...] = (
    "artist_presets",
    "default_artist_preset",
    "nai_artist_prompt",
    "custom_prompt_add",
    "negative_prompt_add",
    "nai_size",
    "num_inference_steps",
    "guidance_scale",
    "variety_boost",
    "noise_schedule",
    "selfie_prompt_add",
    "selfie_negative_prompt_add",
)


def _extra_params_to_list(raw: Any) -> List[Dict[str, Any]]:
    """1.x 的 nai_extra_params 是 TOML 子表；2.0.0 改成 [{key, value}] 数组。

    改数组形态是因为 Dashboard 没有 dict/json 编辑器，子表形态在 WebUI 里根本渲染
    不出来（只能靠 hidden 藏起来）；数组能走现成的列表编辑器。
    """
    if isinstance(raw, dict):
        return [{"key": str(key), "value": value} for key, value in raw.items()]
    if isinstance(raw, list):
        migrated: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict) and "key" in item:
                migrated.append({"key": str(item.get("key")), "value": item.get("value")})
        return migrated
    return []


def _resolve_shared_defaults(
    old_sections: Dict[str, Dict[str, Any]],
    preferred_section: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """挑出进 [model_defaults] 的共享标量，并报告取值不一致的字段。

    Returns:
        (shared_values, conflict_notes)
    """
    shared: Dict[str, Any] = {}
    conflicts: List[str] = []
    for field_name in _SHARED_SCALAR_FIELDS:
        present = {
            name: section[field_name]
            for name, section in old_sections.items()
            if field_name in section
        }
        if not present:
            continue
        distinct = {repr(value) for value in present.values()}
        if preferred_section in present:
            chosen_from, chosen = preferred_section, present[preferred_section]
        else:
            chosen_from, chosen = next(iter(present.items()))
        shared[field_name] = chosen
        if len(distinct) > 1:
            detail = "，".join(f"{name}={value!r}" for name, value in present.items())
            conflicts.append(
                f"{field_name}: 三节取值不一致（{detail}），"
                f"[model_defaults] 采用 {chosen_from} 的 {chosen!r}，"
                f"其余值请自行确认是否需要在对应模型节补回"
            )
    return shared, conflicts


@dataclass
class MigrationReport:
    """一次迁移的结果，供调用方与测试检查具体做了什么。

    做成返回值而不是只打日志，是因为"哪些值被舍弃了"属于必须能被断言的行为；
    只写进日志的话，回归测试只能去抓日志文本，很脆。
    """

    migrated: bool = False
    backup_path: Optional[Path] = None
    # 每个老节搬到哪个新节、带多少条预设
    notes: List[str] = field(default_factory=list)
    # 需要用户人工确认的取值冲突与被丢弃的字段
    conflicts: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.migrated


def needs_migration(config_path: str | Path) -> bool:
    """判断这份 config.toml 是否还带着 1.x 的模型节结构。

    判据只有"存在至少一个 1.x 模型节"这一条，**不看 config_version**。迁移会整份重写
    文件（配置节要改名，没法只改局部），用户手写的注释只能靠备份保留；所以必须把触发
    条件收紧到真正需要搬动结构的场景，不能因为版本号偏低就把一份结构已经正确的配置
    重写一遍。
    """
    resolved = Path(config_path)
    if not resolved.is_file():
        return False
    try:
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        # 解析不了就别动它，让后续加载流程把真实错误抛出来
        return False
    if not isinstance(data, dict):
        return False
    return any(isinstance(data.get(name), dict) for name in _OLD_TO_NEW_SECTION)


def migrate_to_v2(config_path: str | Path) -> MigrationReport:
    """把 1.x 的 config.toml 原地迁移到 2.0.0 结构，迁移前留备份。

    Returns:
        MigrationReport；``bool(report)`` 为 False 表示不需要迁移。
    """
    resolved = Path(config_path)
    if not needs_migration(resolved):
        return MigrationReport()

    original_text = resolved.read_text(encoding="utf-8")
    document = tomlkit.parse(original_text)
    data: Dict[str, Any] = {
        str(key): (value.unwrap() if hasattr(value, "unwrap") else value)
        for key, value in document.items()
    }

    old_sections = {
        name: section
        for name, section in ((n, data.get(n)) for n in _OLD_TO_NEW_SECTION)
        if isinstance(section, dict)
    }
    if not old_sections:
        # needs_migration 已经保证走到这里必有老节；防御性返回，避免误重写用户文件
        return MigrationReport()

    active_model = str(data.get("model", {}).get("default_model") or DEFAULT_MODEL_NAME)
    # 当前默认模型在 1.x 里落在哪个老节 —— 共享标量以它为准
    preferred_old = next(
        (old for old, new in _OLD_TO_NEW_SECTION.items() if new == resolve_config_section(active_model)),
        "model_nai4_5",
    )

    shared, conflicts = _resolve_shared_defaults(old_sections, preferred_old)
    notes: List[str] = []

    # [model_defaults]：共享标量 + 空的负面/自拍位（各模型的原值留在自己节里，行为不变）
    defaults_section: Dict[str, Any] = dict(shared)
    defaults_section.setdefault("negative_prompt_add", "")
    defaults_section.setdefault("selfie_prompt_add", "")
    defaults_section.setdefault("selfie_negative_prompt_add", "")
    defaults_section["nai_extra_params"] = []

    for old_name, old_section in old_sections.items():
        new_name = _OLD_TO_NEW_SECTION[old_name]
        migrated: Dict[str, Any] = {}
        for field_name in _PER_MODEL_FIELDS:
            if field_name in old_section:
                migrated[field_name] = old_section[field_name]
        migrated["nai_extra_params"] = _extra_params_to_list(old_section.get("nai_extra_params"))

        # auto_smea 只有 V3 及更早还认；V4+ 发这个字段本来就是无效参数
        if "auto_smea" in old_section:
            example_model = {
                "model_v3_anime": "nai-diffusion-3",
                "model_v4_full": "nai-diffusion-4-full",
                "model_v45_full": "nai-diffusion-4-5-full",
            }[new_name]
            if supports_smea(example_model):
                migrated["auto_smea"] = old_section["auto_smea"]
            elif old_section["auto_smea"]:
                notes.append(
                    f"{old_name}.auto_smea={old_section['auto_smea']!r} 已丢弃："
                    "SMEA 是 V3 及更早模型的能力，V4 起官方不再提供，"
                    "此前发送的 autoSmea 是无效字段"
                )

        preset_count = len(migrated.get("artist_presets") or [])
        notes.append(f"{old_name} → {new_name}（画师预设 {preset_count} 条）")
        data[new_name] = migrated
        data.pop(old_name, None)

    data["model_defaults"] = defaults_section
    data.setdefault("plugin", {})["config_version"] = CONFIG_VERSION_2

    backup_path = _write_migrated(resolved, original_text, data)

    logger.info("[nai_draw_plugin] 配置已迁移到 %s，原文件备份于 %s", CONFIG_VERSION_2, backup_path)
    for note in notes:
        logger.info("[nai_draw_plugin] 迁移: %s", note)
    for conflict in conflicts:
        logger.warning("[nai_draw_plugin] 迁移需人工确认: %s", conflict)
    logger.info(
        "[nai_draw_plugin] V5 Full / V5 Curated / V4.5 Curated / V4 Curated / V3 Furry "
        "五个新配置节使用官方推荐默认值，画师预设为空，按需自行填写"
    )
    return MigrationReport(
        migrated=True,
        backup_path=backup_path,
        notes=notes,
        conflicts=conflicts,
    )


def _write_migrated(
    config_path: Path,
    original_text: str,
    data: Dict[str, Any],
) -> Path:
    """备份原文件后写出迁移结果（带 2.0.0 的字段注释）。"""
    backup_path = config_path.with_suffix(f"{config_path.suffix}.pre-{CONFIG_VERSION_2}")
    if backup_path.exists():
        # 已经有同名备份说明之前迁移过一次，别覆盖掉更早的原始版本
        index = 1
        while backup_path.with_name(f"{backup_path.name}.{index}").exists():
            index += 1
        backup_path = backup_path.with_name(f"{backup_path.name}.{index}")
    backup_path.write_text(original_text, encoding="utf-8")

    # 用 2.0.0 的 schema 重新渲染，让新字段带上说明注释
    from .plugin_config import PLUGIN_CONFIG

    document = tomlkit.document()
    for name, value in data.items():
        document[name] = value
    config_path.write_text(PLUGIN_CONFIG.compose_commented_text(document), encoding="utf-8")
    return backup_path
