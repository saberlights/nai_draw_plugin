# -*- coding: utf-8 -*-
"""画师预设索引按模型隔离的断言。

1.x 的 ``selected_artist_index`` 是一个不带模型维度的裸 int：在 V4.5 下 ``/nai art 2``
之后切到 V3，索引 2 仍在 V3 的预设范围内，于是**静默套用 V3 的第 2 个预设**，而该模型
自己配的 ``default_artist_preset`` 被完全忽略。这一组测试把"每个模型各记一份索引"钉死。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

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

from plugins.nai_draw_plugin.core.services.session_state import (  # noqa: E402
    SessionStateManager,
)

V45 = "nai-diffusion-4-5-full"
V5 = "nai-diffusion-5-full"
V3 = "nai-diffusion-3"

_CONFIG = {
    "model_v45_full": {
        "artist_presets": [
            {"name": "v45-a", "prompt": "artist:v45a"},
            {"name": "v45-b", "prompt": "artist:v45b"},
            {"name": "v45-c", "prompt": "artist:v45c"},
        ],
        "default_artist_preset": "",
    },
    "model_v5_full": {
        "artist_presets": [
            {"name": "v5-a", "prompt": "artist:v5a"},
            {"name": "v5-b", "prompt": "artist:v5b"},
        ],
        "default_artist_preset": "",
    },
    "model_v3_anime": {
        "artist_presets": [
            {"name": "v3-a", "prompt": "artist:v3a"},
            {"name": "v3-b", "prompt": "artist:v3b"},
        ],
        # V3 明确指定默认用第 2 个预设
        "default_artist_preset": 2,
    },
}


def _get_config(key, default=None):
    if "." in key:
        section, field = key.split(".", 1)
        return _CONFIG.get(section, {}).get(field, default)
    return _CONFIG.get(key, default)


def _state() -> SessionStateManager:
    """SessionStateManager 是 __new__ 单例，这里拿到的始终是同一个实例。

    所以状态由下面的 autouse fixture 在每个测试前重置，不能靠"新建实例"隔离。
    """
    return SessionStateManager()


@pytest.fixture(autouse=True)
def _reset_session_state():
    """单例会在测试之间泄漏会话偏好，每个测试前重置一次。"""
    SessionStateManager()._init_state()
    yield
    SessionStateManager()._init_state()


def test_不同模型各记一份画师索引() -> None:
    state = _state()
    state.set_selected_artist_index("qq", "1", 3, V45)
    state.set_selected_artist_index("qq", "1", 2, V5)

    assert state.get_selected_artist_index("qq", "1", V45) == 3
    assert state.get_selected_artist_index("qq", "1", V5) == 2
    # 没设过的模型返回 None，交给调用方回落 default_artist_preset
    assert state.get_selected_artist_index("qq", "1", V3) is None


def test_切换模型不会串用另一个模型的预设() -> None:
    """这是 1.x 最容易被忽略的 bug：索引恰好在范围内时会静默换风格。"""
    state = _state()
    state.set_selected_artist_index("qq", "1", 2, V45)

    v45_preset = state.get_selected_artist_preset_config("qq", "1", V45, _get_config)
    assert v45_preset["prompt"] == "artist:v45b"

    # 切到 V3：索引 2 在 V3 的 2 个预设内，1.x 会套用 v3-b；
    # 现在 V3 没有自己的会话选择，必须回落到它自己的 default_artist_preset=2
    v3_preset = state.get_selected_artist_preset_config("qq", "1", V3, _get_config)
    assert v3_preset["prompt"] == "artist:v3b"
    # 且 V3 走的是配置默认，不是继承 V4.5 的会话选择
    assert state.get_selected_artist_index("qq", "1", V3) is None


def test_模型自己的默认预设不再被越界索引吞掉() -> None:
    """V4.5 选了第 3 个，切到只有 2 个预设的 V3 时，V3 的默认值必须仍然生效。"""
    state = _state()
    state.set_selected_artist_index("qq", "1", 3, V45)

    index = state.get_effective_artist_index("qq", "1", V3, _get_config)
    assert index == 2, "V3 的 default_artist_preset=2 应生效，而不是被越界回落成 1"

    preset = state.get_selected_artist_preset_config("qq", "1", V3, _get_config)
    assert preset["prompt"] == "artist:v3b"


def test_v5_与_v45_的画师选择互不影响() -> None:
    """用户的原始诉求：每个模型适合的画师串不一样，选择也必须分开记。"""
    state = _state()
    state.set_selected_artist_index("qq", "1", 1, V45)
    state.set_selected_artist_index("qq", "1", 2, V5)

    assert state.get_selected_artist_preset_config("qq", "1", V45, _get_config)["prompt"] == (
        "artist:v45a"
    )
    assert state.get_selected_artist_preset_config("qq", "1", V5, _get_config)["prompt"] == (
        "artist:v5b"
    )


def test_生效索引在越界时回落但不污染其它模型() -> None:
    state = _state()
    # 人为写入一个对 V5 越界的索引（V5 只有 2 个预设）
    state.set_selected_artist_index("qq", "1", 99, V5)

    assert state.get_effective_artist_index("qq", "1", V5, _get_config) == 1
    # V4.5 不受影响，仍走自己的默认
    assert state.get_effective_artist_index("qq", "1", V45, _get_config) == 1


def test_未登记的模型不写入画师索引() -> None:
    """未知模型没有配置节，写索引没有归属，直接忽略而不是塞进某个节。"""
    state = _state()
    state.set_selected_artist_index("qq", "1", 2, "brand-new-model")

    assert state.get_selected_artist_index("qq", "1", "brand-new-model") is None
    assert state.get_selected_artist_index("qq", "1", V45) is None


def test_不同会话之间画师索引互不影响() -> None:
    state = _state()
    state.set_selected_artist_index("qq", "1", 3, V45)
    state.set_selected_artist_index("qq", "2", 1, V45)

    assert state.get_selected_artist_index("qq", "1", V45) == 3
    assert state.get_selected_artist_index("qq", "2", V45) == 1
