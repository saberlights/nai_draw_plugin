# -*- coding: utf-8 -*-
"""插件测试公共夹具。"""

import importlib

import pytest

from plugins.nai_draw_plugin.core.services.visual_continuity_store import (
    VisualContinuityStore,
)

# services 包的 __init__ 用 session_state 单例遮蔽了同名子模块的包属性，
# 普通 import 语句会拿到实例而非模块，必须经 importlib 直查 sys.modules
_session_state_module = importlib.import_module(
    "plugins.nai_draw_plugin.core.services.session_state"
)
_visual_store_module = importlib.import_module(
    "plugins.nai_draw_plugin.core.services.visual_continuity_store"
)


@pytest.fixture(autouse=True)
def _isolate_visual_continuity_store(tmp_path, monkeypatch):
    """把视觉连续性持久化重定向到临时目录，避免测试污染仓库 data/。"""
    store = VisualContinuityStore(storage_path=tmp_path / "visual_continuity.json")
    monkeypatch.setattr(_visual_store_module, "visual_continuity_store", store)
    monkeypatch.setattr(_session_state_module, "visual_continuity_store", store)
    yield store
