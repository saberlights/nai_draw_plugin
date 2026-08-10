# -*- coding: utf-8 -*-
"""插件统一持久化目录解析与旧式数据目录迁移。

宿主运行时为每个插件分配 ``<项目根>/data/plugins/<插件ID>`` 作为持久化目录
（见宿主 ``src/plugin_runtime/runner/plugin_paths.py`` 的 ``build_plugin_paths``），
并在检测到旧式源码目录 ``<插件根>/data`` 时打启动警告。

本模块按同一规则解析统一目录，并在导入期把旧目录中的存量数据一次性搬迁
过去。各 store 的默认路径都以本模块为唯一来源，因此任何 store 在求值默认
路径前都必然先完成迁移，不会读到搬迁前的旧位置。

刻意不依赖 ``src.*``：``python -m core.utils.tag_data_builder`` 这类脱离宿主
进程的独立运行方式也要能导入本模块。宿主已把日志 handler 挂在 root logger
上，标准库 logging 的消息同样会进入宿主日志。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger("nai_draw_plugin")

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_DATA_DIR = _PLUGIN_ROOT / "data"


def _read_plugin_id(manifest_path: Path) -> str:
    """从 ``_manifest.json`` 读取插件 ID，与宿主分配目录所用 ID 保持同源。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plugin_id = str(manifest.get("id", "")).strip()
    if not plugin_id:
        raise ValueError(f"_manifest.json 缺少有效的插件 id: {manifest_path}")
    return plugin_id


PLUGIN_ID = _read_plugin_id(_PLUGIN_ROOT / "_manifest.json")

# 与宿主 build_plugin_paths 的 ``<项目根>/data/plugins/<插件ID>`` 保持一致；
# 插件固定部署在 ``<项目根>/plugins/<目录名>`` 下，项目根即插件根的上两级。
PLUGIN_DATA_DIR = _PLUGIN_ROOT.parents[1] / "data" / "plugins" / PLUGIN_ID


def plugin_data_dir() -> Path:
    """返回统一持久化目录，并保证目录存在。"""
    PLUGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PLUGIN_DATA_DIR


def plugin_data_path(*parts: str) -> Path:
    """返回统一持久化目录下的文件/子目录路径（不创建条目本身）。"""
    return plugin_data_dir().joinpath(*parts)


def migrate_legacy_data_dir(
    legacy_dir: Path = _LEGACY_DATA_DIR,
    target_dir: Path = PLUGIN_DATA_DIR,
) -> List[str]:
    """把旧式 ``<插件根>/data`` 的存量数据整体搬迁到统一目录。

    逐条目移动；统一目录已存在同名条目时不覆盖，旧条目保留原位并告警，
    留待人工处置（此时旧目录不会被删除，宿主警告会继续提示）。
    旧目录搬空后移除，宿主的旧式目录警告随之消失。

    Returns:
        已迁移的条目名列表（无旧目录时为空列表）。
    """
    if not legacy_dir.is_dir():
        return []

    target_dir.mkdir(parents=True, exist_ok=True)
    moved: List[str] = []
    for entry in sorted(legacy_dir.iterdir()):
        destination = target_dir / entry.name
        if destination.exists():
            logger.warning(
                "[nai_draw_plugin] 旧数据条目 %s 与统一目录 %s 下的同名条目冲突，"
                "保留原位待人工处置",
                entry,
                target_dir,
            )
            continue
        shutil.move(str(entry), str(destination))
        moved.append(entry.name)

    if moved:
        logger.info(
            "[nai_draw_plugin] 已把旧数据目录 %s 中的 %d 个条目迁移到 %s：%s",
            legacy_dir,
            len(moved),
            target_dir,
            "、".join(moved),
        )
    if not any(legacy_dir.iterdir()):
        legacy_dir.rmdir()
        logger.info("[nai_draw_plugin] 旧数据目录 %s 已搬空并移除", legacy_dir)
    return moved


# 导入期执行一次真实迁移：本模块是所有 store 默认路径的唯一来源，
# 迁移必须发生在任何 store 求值默认路径之前。
migrate_legacy_data_dir()
