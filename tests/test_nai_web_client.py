# -*- coding: utf-8 -*-
from __future__ import annotations

import json
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

from plugins.nai_draw_plugin.core.clients.nai_web_client import NaiWebClient


def test_build_inner_draw_params_uses_integer_array_size_and_omits_random_seed() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {
            "default_model": "nai-diffusion-4-5-full",
            "nai_size": "竖图",
            "seed": -1,
            "quality_toggle": True,
            "auto_smea": True,
            "sm": True,
            "sm_dyn": False,
            "variety_boost": True,
            "image_format": "webp",
        },
        None,
    )

    assert inner["size"] == [832, 1216]
    assert "seed" not in inner
    assert inner["qualityToggle"] is True
    assert inner["autoSmea"] is True
    assert inner["sm"] is True
    assert inner["sm_dyn"] is False
    assert inner["variety_boost"] is True
    assert inner["image_format"] == "webp"
    assert "model" not in inner


def test_build_request_body_keeps_model_only_in_outer_body() -> None:
    inner = {
        "prompt": "1girl",
        "negative_prompt": "lowres",
        "size": [1216, 832],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "qualityToggle": True,
        "autoSmea": True,
    }

    body = NaiWebClient._build_request_body("nai-diffusion-4-5-full", inner, 100000)
    payload = json.loads(body["messages"][0]["content"])

    assert body["model"] == "nai-diffusion-4-5-full"
    assert body["stream"] is False
    assert body["max_tokens"] == 100000
    assert payload["size"] == [1216, 832]
    assert "seed" not in payload
    assert payload["qualityToggle"] is True
    assert payload["autoSmea"] is True
    assert "model" not in payload


def test_build_inner_draw_params_keeps_explicit_seed() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {
            "default_model": "nai-diffusion-4-5-full",
            "nai_size": "方图",
            "seed": 123456789,
        },
        None,
    )

    assert inner["size"] == [1024, 1024]
    assert inner["seed"] == 123456789


def test_build_inner_draw_params_injects_characters_with_position() -> None:
    """全部 character 都给了合法 position 时启用 use_coords=true。"""
    characters = [
        {"prompt": "girl, blue hair, blue dress", "negative_prompt": "white hair", "position": "B3"},
        {"prompt": "girl, white hair, white kimono", "negative_prompt": "blue hair", "position": "D3"},
    ]
    inner = NaiWebClient._build_inner_draw_params(
        "2girls, indoor",
        {"default_model": "nai-diffusion-4-5-full", "nai_size": "横图"},
        None,
        characters=characters,
    )

    assert inner["characters"] == [
        {"prompt": "girl, blue hair, blue dress", "negative_prompt": "white hair", "position": "B3"},
        {"prompt": "girl, white hair, white kimono", "negative_prompt": "blue hair", "position": "D3"},
    ]
    assert inner["use_coords"] is True
    assert inner["use_order"] is True


def test_build_inner_draw_params_clears_position_when_partial() -> None:
    """只要一个角色缺 position，整组 position 全部丢弃且 use_coords=false。"""
    characters = [
        {"prompt": "girl, smile", "position": "B3"},
        {"prompt": "girl, laugh", "position": ""},
    ]
    inner = NaiWebClient._build_inner_draw_params(
        "2girls",
        {"default_model": "nai-diffusion-4-5-full"},
        None,
        characters=characters,
    )

    for item in inner["characters"]:
        assert "position" not in item
    assert inner["use_coords"] is False
    assert inner["use_order"] is True


def test_build_inner_draw_params_omits_characters_when_only_one() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {"default_model": "nai-diffusion-4-5-full"},
        None,
        characters=[{"prompt": "girl, smile"}],
    )

    assert "characters" not in inner
    assert "use_coords" not in inner


def test_filter_characters_for_model_keeps_v4_series() -> None:
    characters = [{"prompt": "a"}, {"prompt": "b"}]
    assert (
        NaiWebClient._filter_characters_for_model("nai-diffusion-4-5-full", characters)
        is characters
    )
    assert (
        NaiWebClient._filter_characters_for_model("nai-diffusion-4-full", characters)
        is characters
    )


def test_filter_characters_for_model_drops_legacy_models() -> None:
    characters = [{"prompt": "a"}, {"prompt": "b"}]
    assert NaiWebClient._filter_characters_for_model("nai-diffusion-3", characters) is None
    assert NaiWebClient._filter_characters_for_model("nai-diffusion-3-furry", characters) is None


def test_validate_inner_payload_rejects_bad_position() -> None:
    inner = {
        "prompt": "global",
        "negative_prompt": "",
        "size": [832, 1216],
        "steps": 20,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "characters": [
            {"prompt": "a", "position": "Z9"},
            {"prompt": "b", "position": "D3"},
        ],
        "use_coords": True,
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is not None
    assert "position" in reason


def test_validate_inner_payload_rejects_empty_character_prompt() -> None:
    inner = {
        "prompt": "global",
        "negative_prompt": "",
        "size": [832, 1216],
        "steps": 20,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
        "characters": [{"prompt": ""}, {"prompt": "b"}],
    }
    reason = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", inner)
    assert reason is not None
    assert "prompt" in reason
