# -*- coding: utf-8 -*-
"""统一持久化目录解析与旧式数据目录迁移的测试。"""

from pathlib import Path

from plugins.nai_draw_plugin.core import data_paths
from plugins.nai_draw_plugin.core.services.nsfw_state_store import NsfwStateStore
from plugins.nai_draw_plugin.core.services.user_blacklist import UserBlacklistService
from plugins.nai_draw_plugin.core.services.visual_continuity_store import VisualContinuityStore
from src.plugin_runtime.runner.plugin_paths import build_plugin_paths


class TestPluginDataDir:
    """统一目录解析必须与宿主分配规则同源同结果。"""

    def test_plugin_id_read_from_manifest(self):
        assert data_paths.PLUGIN_ID == "saberlights.nai-draw-plugin"

    def test_data_dir_matches_host_build_plugin_paths(self):
        # 与宿主 build_plugin_paths 用同一套输入交叉验证，宿主挪目录布局时此测试会挂
        plugin_root = Path(data_paths.__file__).resolve().parents[1]
        project_root = plugin_root.parents[1]
        host_paths = build_plugin_paths(data_paths.PLUGIN_ID, project_root)
        assert data_paths.PLUGIN_DATA_DIR == host_paths.data_dir

    def test_plugin_data_path_joins_under_unified_dir(self):
        assert (
            data_paths.plugin_data_path("named_refs", "selection.json")
            == data_paths.PLUGIN_DATA_DIR / "named_refs" / "selection.json"
        )


class TestStoreDefaultPaths:
    """各 store 的默认路径必须钉死在统一持久化目录下，防止回退到旧式源码目录。"""

    def test_store_default_paths_point_to_unified_dir(self):
        import importlib

        vibe_cache = importlib.import_module("plugins.nai_draw_plugin.core.services.vibe_cache")
        named_reference_store = importlib.import_module(
            "plugins.nai_draw_plugin.core.services.named_reference_store"
        )

        unified = data_paths.PLUGIN_DATA_DIR
        assert NsfwStateStore._DEFAULT_PATH == unified / "nsfw_state.json"
        assert VisualContinuityStore._DEFAULT_PATH == unified / "visual_continuity.json"
        assert UserBlacklistService()._storage_path == unified / "user_blacklist.json"
        assert vibe_cache._default_db_path() == unified / "vibe_cache.db"
        assert named_reference_store._default_root() == unified / "named_refs"

    def test_no_source_file_builds_legacy_data_dir(self):
        """禁止任何源码文件再拼 `<插件根>/data` 旧路径。

        tag_retriever 会被兄弟测试用 sys.modules 桩替换、无法按真身导入断言，
        这里用源码级守卫统一兜住所有模块（含未来新增）。
        runtime_recall.py 的 `parents[2] / "data" / "MaiBot.db"` 指向宿主消息库
        （文件在插件根，parents[2] 是 MaiBot 根），属合法引用，显式放行。
        """
        import re

        plugin_root = Path(data_paths.__file__).resolve().parents[1]
        legacy_idioms = re.compile(
            r'parents\[2\]\s*/\s*"data"(?!\s*/\s*"MaiBot\.db")'
            r'|os\.path\.join\(\s*_?plugin_root\s*,\s*"data"'
        )
        offenders = []
        for source_file in plugin_root.rglob("*.py"):
            if "tests" in source_file.parts or "__pycache__" in source_file.parts:
                continue
            if legacy_idioms.search(source_file.read_text(encoding="utf-8")):
                offenders.append(str(source_file.relative_to(plugin_root)))
        assert offenders == []


class TestMigrateLegacyDataDir:
    """旧目录搬迁：整体移动、冲突保留原位、搬空即删。"""

    def test_moves_entries_and_removes_legacy_dir(self, tmp_path):
        legacy = tmp_path / "legacy"
        target = tmp_path / "unified"
        (legacy / "named_refs" / "users").mkdir(parents=True)
        (legacy / "named_refs" / "users" / "u1.png").write_bytes(b"\x89PNG-fake-bytes")
        (legacy / "nsfw_state.json").write_text('{"qq:1": true}', encoding="utf-8")

        moved = data_paths.migrate_legacy_data_dir(legacy, target)

        assert moved == ["named_refs", "nsfw_state.json"]
        assert not legacy.exists()
        assert (target / "named_refs" / "users" / "u1.png").read_bytes() == b"\x89PNG-fake-bytes"
        assert (target / "nsfw_state.json").read_text(encoding="utf-8") == '{"qq:1": true}'

    def test_conflict_entry_kept_in_place_and_dir_preserved(self, tmp_path):
        legacy = tmp_path / "legacy"
        target = tmp_path / "unified"
        legacy.mkdir()
        target.mkdir()
        (legacy / "user_blacklist.json").write_text("legacy-side", encoding="utf-8")
        (target / "user_blacklist.json").write_text("unified-side", encoding="utf-8")
        (legacy / "vibe_cache.db").write_bytes(b"sqlite-fake")

        moved = data_paths.migrate_legacy_data_dir(legacy, target)

        assert moved == ["vibe_cache.db"]
        # 冲突条目两侧都不动：不覆盖统一目录、旧目录保留待人工处置
        assert (legacy / "user_blacklist.json").read_text(encoding="utf-8") == "legacy-side"
        assert (target / "user_blacklist.json").read_text(encoding="utf-8") == "unified-side"
        assert legacy.is_dir()
        assert (target / "vibe_cache.db").read_bytes() == b"sqlite-fake"

    def test_empty_legacy_dir_removed(self, tmp_path):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        target = tmp_path / "unified"

        assert data_paths.migrate_legacy_data_dir(legacy, target) == []
        assert not legacy.exists()

    def test_noop_without_legacy_dir(self, tmp_path):
        target = tmp_path / "unified"

        assert data_paths.migrate_legacy_data_dir(tmp_path / "absent", target) == []
        # 无迁移时不应产生建目录副作用
        assert not target.exists()
