# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import secrets
import sys
import types
from pathlib import Path

import pytest
import requests

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


def test_prompt_merge_keeps_rating_before_configured_prefixes() -> None:
    prompt = NaiWebClient._merge_artist_into_prompt(
        "rating:general, solo, 1girl, smile",
        "artist:example",
        "best quality",
    )

    assert prompt == "rating:general, best quality, artist:example, solo, 1girl, smile"


def test_chami_request_maps_internal_model_name() -> None:
    body = NaiWebClient._build_chami_request_body(
        "1girl",
        "nai-diffusion-4-5-full",
        {"size": [832, 1216], "scale": 5.0, "sampler": "k_euler", "steps": 28},
    )

    assert body["model"] == "nai-diffusion-4-5-full"


def test_chami_request_keeps_newapi_sampler_name() -> None:
    body = NaiWebClient._build_chami_request_body(
        "1girl",
        "novelai-v3",
        {
            "size": [832, 1216],
            "scale": 5.0,
            "sampler": "k_euler_ancestral",
            "steps": 28,
        },
    )

    assert body["parameters"]["sampler"] == "k_euler_ancestral"


def test_build_chami_request_body_reuses_existing_inner_parameters() -> None:
    body = NaiWebClient._build_chami_request_body(
        "1girl, best quality",
        "novelai-v3",
        {
            "size": [832, 1216],
            "scale": 5.5,
            "sampler": "k_dpmpp_2m",
            "steps": 28,
            "n_samples": 1,
            "negative_prompt": "lowres",
            "seed": 123,
            "cfg_rescale": 0.18,
            "qualityToggle": True,
            "variety_boost": True,
        },
    )

    assert body["input"] == "1girl, best quality"
    assert body["model"] == "novelai-v3"
    assert body["action"] == "generate"
    assert body["parameters"]["width"] == 832
    assert body["parameters"]["height"] == 1216
    assert body["parameters"]["steps"] == 28
    assert body["parameters"]["seed"] == 123
    assert body["parameters"]["skip_cfg_above_sigma"] == 58


def test_parse_chami_error_response_preserves_server_message() -> None:
    response = types.SimpleNamespace(
        status_code=403,
        text='{"error":{"message":"invalid combined key"}}',
        json=lambda: {"error": {"message": "invalid combined key"}},
    )
    client = NaiWebClient(log_prefix="test", run_blocking=lambda *_args, **_kwargs: None)

    success, result = client._parse_chami_response(response)

    assert success is False
    assert result == "HTTP 403: invalid combined key"


def test_parse_chami_zip_response_extracts_first_image() -> None:
    import io
    import zipfile

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("image_0.png", b"png-bytes")

    response = types.SimpleNamespace(status_code=200, content=output.getvalue())
    client = NaiWebClient(log_prefix="test", run_blocking=lambda *_args, **_kwargs: None)
    success, result = client._parse_chami_response(response)

    assert success is True
    assert result == "cG5nLWJ5dGVz"


def test_http_sessions_are_created_lazily_and_closed_when_initialized(monkeypatch) -> None:
    created: list[tuple[bool, _RecordingSession]] = []

    def create_session(trust_env: bool) -> _RecordingSession:
        session = _RecordingSession()
        session.close_calls = 0
        session.close = lambda: setattr(session, "close_calls", session.close_calls + 1)
        created.append((trust_env, session))
        return session

    monkeypatch.setattr(NaiWebClient, "_create_session", staticmethod(create_session))

    async def run_inline(function, *args, **_kwargs):
        return function(*args)

    client = NaiWebClient(
        log_prefix="test",
        run_blocking=run_inline,
    )

    assert created == []
    assert client._get_session(False) is created[0][1]
    assert client._get_session(False) is created[0][1]
    assert client._get_session(True) is created[1][1]

    client.close()

    assert [(trust_env, session.close_calls) for trust_env, session in created] == [
        (False, 1),
        (True, 1),
    ]


def test_client_uses_injected_blocking_runner_instead_of_default_executor(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    response = types.SimpleNamespace(
        status_code=200,
        text="",
        close=lambda: None,
        json=lambda: {"data": [{"id": "nai-diffusion-4-5-full"}]},
    )

    async def run_inline(function, *args, **_kwargs):
        calls.append((function, *args))
        return function(*args)

    async def fail_to_thread(*_args, **_kwargs):
        pytest.fail("NaiWebClient 不应使用 asyncio 默认执行器")

    client = NaiWebClient(
        log_prefix="test",
        run_blocking=run_inline,
    )
    monkeypatch.setattr(client, "_send_get_request", lambda *_args: response)
    monkeypatch.setattr(asyncio, "to_thread", fail_to_thread)

    result = asyncio.run(client.list_models({"base_url": "https://gateway.example.com"}))

    assert result == (True, ["nai-diffusion-4-5-full"])
    assert len(calls) == 1


def test_build_inner_draw_params_uses_integer_array_size_and_omits_random_seed() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {
            "default_model": "nai-diffusion-4-5-full",
            "nai_size": "竖图",
            "seed": -1,
            "quality_toggle": True,
            "auto_smea": True,
            "variety_boost": True,
            "image_format": "webp",
        },
        None,
    )

    assert inner["size"] == [832, 1216]
    assert "seed" not in inner
    assert inner["qualityToggle"] is True
    # SMEA 是补偿 V1 时代低训练分辨率的手段，V4 起官方不再提供；
    # 即使配置里写了 auto_smea=true，也不该给 V4.5 发这个无效字段
    assert "autoSmea" not in inner
    # 文档 §5 表外的 sm/sm_dyn 字段已不再透传，避免触发网关 400
    assert "sm" not in inner
    assert "sm_dyn" not in inner
    assert inner["variety_boost"] is True
    assert inner["image_format"] == "webp"
    assert "model" not in inner


def test_auto_smea_只发给_v3_及更早模型() -> None:
    """V3 仍认 SMEA，配置开了就要发；V4+ 一律不发。"""
    v3_inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {"default_model": "nai-diffusion-3", "auto_smea": True},
        None,
    )
    assert v3_inner["autoSmea"] is True

    for model in ("nai-diffusion-4-full", "nai-diffusion-4-5-full", "nai-diffusion-5-full"):
        inner = NaiWebClient._build_inner_draw_params(
            "1girl",
            {"default_model": model, "auto_smea": True},
            None,
        )
        assert "autoSmea" not in inner, f"{model} 不该收到 autoSmea"


def test_quality_toggle_可以显式关闭() -> None:
    """原先只在 True 时写入字段，用户没法显式关掉质量增强。"""
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {"default_model": "nai-diffusion-4-5-full", "quality_toggle": False},
        None,
    )
    assert inner["qualityToggle"] is False


def test_显式传入的_size_优先于_nai_size() -> None:
    """i2i 会把参考图真实宽高作为 size 传进来，而 nai_size 几乎永远非空。

    原先 nai_size 排在 size 之前，会把参考图尺寸吞掉，导致 §20.1 的
    "i2i.image 宽高必须与外层 size 一致" 校验必然失败。
    """
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {"default_model": "nai-diffusion-4-5-full", "nai_size": "竖图"},
        "1216x832",
    )
    assert inner["size"] == [1216, 832]

    # 调用方没传 size 时仍走模型节的 nai_size
    fallback = NaiWebClient._build_inner_draw_params(
        "1girl",
        {"default_model": "nai-diffusion-4-5-full", "nai_size": "竖图"},
        None,
    )
    assert fallback["size"] == [832, 1216]


def test_steps_兜底取该模型的推荐值() -> None:
    """原先硬编码兜底 23，与 schema 默认 28 不一致。"""
    v45 = NaiWebClient._build_inner_draw_params(
        "1girl", {"default_model": "nai-diffusion-4-5-full"}, None
    )
    assert v45["steps"] == 28

    v3 = NaiWebClient._build_inner_draw_params(
        "1girl", {"default_model": "nai-diffusion-3"}, None
    )
    assert v3["steps"] == 25


def test_额外透传参数支持键值数组与历史子表两种写法() -> None:
    """配置改成 [{key, value}] 数组（Dashboard 没有 dict 编辑器），但要兼容旧子表。"""
    from_list = NaiWebClient._build_inner_draw_params(
        "1girl",
        {
            "default_model": "nai-diffusion-4-5-full",
            "nai_extra_params": [
                {"key": "custom_field", "value": "abc"},
                {"key": "", "value": "无 key 应被跳过"},
                {"key": "empty_value", "value": ""},
            ],
        },
        None,
    )
    assert from_list["custom_field"] == "abc"
    assert "empty_value" not in from_list

    from_dict = NaiWebClient._build_inner_draw_params(
        "1girl",
        {
            "default_model": "nai-diffusion-4-5-full",
            "nai_extra_params": {"legacy_field": 42},
        },
        None,
    )
    assert from_dict["legacy_field"] == 42


def test_角色数量上限按模型校验() -> None:
    """V4/V4.5 上限 6 个，V5 上限 22 个。"""
    def make_characters(count: int) -> list[dict]:
        return [{"prompt": f"girl {index}"} for index in range(count)]

    over_v45 = NaiWebClient._build_inner_draw_params(
        "2girls", {"default_model": "nai-diffusion-4-5-full"}, None, make_characters(7)
    )
    error = NaiWebClient._validate_inner_payload("nai-diffusion-4-5-full", over_v45)
    assert error is not None and "最多 6 个角色段" in error

    # 同样 7 个角色，V5 允许
    over_v5 = NaiWebClient._build_inner_draw_params(
        "2girls", {"default_model": "nai-diffusion-5-full"}, None, make_characters(7)
    )
    assert NaiWebClient._validate_inner_payload("nai-diffusion-5-full", over_v5) is None


def test_build_inner_draw_params_normalizes_image_format_whitelist() -> None:
    """非 png/webp 的 image_format 会被回退到 png，避免触发 NewAPI 400。"""
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {"default_model": "nai-diffusion-4-5-full", "image_format": "jpeg"},
        None,
    )
    assert inner["image_format"] == "png"


def test_build_inner_draw_params_emits_cfg_rescale_and_noise_schedule() -> None:
    """cfg_rescale / noise_schedule 在合法范围内时写入 inner JSON。"""
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {
            "default_model": "nai-diffusion-4-5-full",
            "cfg_rescale": 0.5,
            "noise_schedule": "karras",
        },
        None,
    )
    assert inner["cfg_rescale"] == 0.5
    assert inner["noise_schedule"] == "karras"


def test_build_inner_draw_params_skips_blank_cfg_rescale_and_invalid_noise_schedule() -> None:
    """cfg_rescale=0 与非法 noise_schedule 应被丢弃，不污染 inner JSON。"""
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {
            "default_model": "nai-diffusion-4-5-full",
            "cfg_rescale": 0.0,
            "noise_schedule": "invalid_value",
        },
        None,
    )
    assert "cfg_rescale" not in inner
    assert "noise_schedule" not in inner


def test_build_inner_draw_params_clamps_cfg_rescale_above_one() -> None:
    inner = NaiWebClient._build_inner_draw_params(
        "1girl",
        {"default_model": "nai-diffusion-4-5-full", "cfg_rescale": 2.5},
        None,
    )
    assert inner["cfg_rescale"] == 1.0


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


def test_filter_characters_for_model_keeps_v5_series() -> None:
    """V5 支持 §7 多角色，不能被嗅探降级掉。

    回归 bug：白名单只有 ``nai-diffusion-4``，导致 V5 下 ``characters`` 被整体丢弃，
    inner JSON 只剩 ``global_text``，两个角色段（角色名 + source#/target# 绑定）静默消失。
    """
    characters = [{"prompt": "a"}, {"prompt": "b"}]
    assert (
        NaiWebClient._filter_characters_for_model("nai-diffusion-5-full", characters)
        is characters
    )


def test_build_inner_draw_params_keeps_characters_on_v5() -> None:
    """端到端确认 V5 的 inner JSON 真的带上了 characters/use_order，而不是只剩 global。"""
    inner = NaiWebClient._build_inner_draw_params(
        "rating:explicit, 1boy 1girl, city at night",
        {"default_model": "nai-diffusion-5-full", "nai_size": "竖图"},
        None,
        NaiWebClient._filter_characters_for_model(
            "nai-diffusion-5-full",
            [
                {"prompt": "boy, shota, 1.4::source#holding_money::", "position": ""},
                {
                    "prompt": "girl, 1.05::aemeath_(wuthering_waves)::, 1.4::target#holding_money::",
                    "position": "",
                },
            ],
        ),
    )

    assert [item["prompt"] for item in inner["characters"]] == [
        "boy, shota, 1.4::source#holding_money::",
        "girl, 1.05::aemeath_(wuthering_waves)::, 1.4::target#holding_money::",
    ]
    assert inner["use_coords"] is False
    assert inner["use_order"] is True


def test_filter_characters_for_model_drops_legacy_models() -> None:
    characters = [{"prompt": "a"}, {"prompt": "b"}]
    assert NaiWebClient._filter_characters_for_model("nai-diffusion-3", characters) is None
    assert NaiWebClient._filter_characters_for_model("nai-diffusion-3-furry", characters) is None


def test_resolve_characters_flattens_instead_of_dropping_on_unsupported_model() -> None:
    """模型不支持 characters[] 时角色段必须拍平回 prompt，绝不能凭空消失。

    回归 bug：旧实现只把 characters 丢成 None，prompt 仍是 global_text，于是
    ``rating:general, 2girls, / char1:hatsune_miku... / char2:luo_tianyi...``
    实际只发出了 ``rating:general, 2girls``——角色名与 mutual#hug 全丢，
    请求照常成功、前置校验也放行，用户只看到一张完全不对的图。
    """
    prompt, characters = NaiWebClient._resolve_characters_for_model(
        "nai-diffusion-3",
        "rating:general, 2girls",
        [
            {"prompt": "girl, 1.05::hatsune_miku::, mutual#hug"},
            {"prompt": "girl, 1.05::luo_tianyi::, mutual#hug"},
        ],
    )

    assert characters is None
    assert prompt == (
        "rating:general, 2girls,\n"
        "char1:girl, 1.05::hatsune_miku::, mutual#hug,\n"
        "char2:girl, 1.05::luo_tianyi::, mutual#hug,"
    )


def test_resolve_characters_keeps_structured_channel_on_supported_model() -> None:
    """支持多角色的模型必须原样走结构化通道，不能被拍平。"""
    characters = [{"prompt": "a"}, {"prompt": "b"}]
    prompt, effective = NaiWebClient._resolve_characters_for_model(
        "nai-diffusion-5-full", "global tags", characters
    )

    assert prompt == "global tags"
    assert effective is characters


def test_unsupported_model_inner_payload_retains_every_character_tag() -> None:
    """端到端：V3 下 inner JSON 没有 characters 键，但角色 tag 一个都不能少。"""
    prompt, characters = NaiWebClient._resolve_characters_for_model(
        "nai-diffusion-3",
        "rating:general, 2girls",
        [
            {"prompt": "girl, 1.05::hatsune_miku::, mutual#hug"},
            {"prompt": "girl, 1.05::luo_tianyi::, mutual#hug"},
        ],
    )
    inner = NaiWebClient._build_inner_draw_params(
        prompt,
        {"default_model": "nai-diffusion-3", "nai_size": "竖图"},
        None,
        characters,
    )

    assert "characters" not in inner
    for tag in ("hatsune_miku", "luo_tianyi", "mutual#hug", "char1:", "char2:"):
        assert tag in inner["prompt"]
    assert NaiWebClient._validate_inner_payload("nai-diffusion-3", inner) is None


def test_filter_character_references_only_on_v4_5() -> None:
    """§20.4 角色参考只有 V4.5 支持；V5 目前只有文生图与图生图，必须被挡下。"""
    payload = [{"image": "x"}]
    assert (
        NaiWebClient._filter_character_references_for_model("nai-diffusion-4-5-full", payload)
        is payload
    )
    assert (
        NaiWebClient._filter_character_references_for_model("nai-diffusion-5-full", payload)
        is None
    )
    assert (
        NaiWebClient._filter_character_references_for_model("nai-diffusion-4-full", payload)
        is None
    )


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


def _make_valid_inner(**overrides: object) -> dict:
    inner = {
        "prompt": "1girl",
        "negative_prompt": "",
        "size": [832, 1216],
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "n_samples": 1,
        "image_format": "png",
    }
    inner.update(overrides)
    return inner


def test_validate_inner_payload_rejects_steps_over_max() -> None:
    """文档 §5：steps 最大 28，超过直接 400。"""
    reason = NaiWebClient._validate_inner_payload(
        "nai-diffusion-4-5-full", _make_valid_inner(steps=29)
    )
    assert reason is not None
    assert "steps" in reason


def test_validate_inner_payload_rejects_zero_or_negative_steps() -> None:
    reason = NaiWebClient._validate_inner_payload(
        "nai-diffusion-4-5-full", _make_valid_inner(steps=0)
    )
    assert reason is not None
    assert "steps" in reason


def test_validate_inner_payload_rejects_unknown_image_format() -> None:
    """文档 §11：image_format 仅允许 png/webp。"""
    reason = NaiWebClient._validate_inner_payload(
        "nai-diffusion-4-5-full", _make_valid_inner(image_format="jpeg")
    )
    assert reason is not None
    assert "image_format" in reason


def test_validate_inner_payload_rejects_invalid_noise_schedule() -> None:
    reason = NaiWebClient._validate_inner_payload(
        "nai-diffusion-4-5-full",
        _make_valid_inner(noise_schedule="not_a_schedule"),
    )
    assert reason is not None
    assert "noise_schedule" in reason


def test_validate_inner_payload_rejects_cfg_rescale_out_of_range() -> None:
    reason = NaiWebClient._validate_inner_payload(
        "nai-diffusion-4-5-full", _make_valid_inner(cfg_rescale=1.5)
    )
    assert reason is not None
    assert "cfg_rescale" in reason


def test_validate_inner_payload_accepts_steps_at_max_and_valid_noise_schedule() -> None:
    """steps=28（边界）+ 合法 noise_schedule 应当通过。"""
    reason = NaiWebClient._validate_inner_payload(
        "nai-diffusion-4-5-full",
        _make_valid_inner(steps=28, noise_schedule="karras", cfg_rescale=0.7),
    )
    assert reason is None


def test_extract_vibe_cache_ids_parses_comment_block() -> None:
    """§20.3.1：响应正文末尾以 HTML 注释附加 vibe_cache_ids。"""
    content = (
        "![image_0](data:image/png;base64,iVBORw0KGgo)\n"
        "<!-- seeds:[123456789] -->\n"
        '<!-- vibe_cache_ids:[{"index":0,"cache_id":"AbCdEfGhIjKlMnOpQrStUv"},'
        '{"index":1,"cache_id":"ZyXwVuTsRqPoNmLkJiHgFe"}] -->'
    )
    parsed = NaiWebClient._extract_vibe_cache_ids(content)
    assert parsed == [
        {"index": 0, "cache_id": "AbCdEfGhIjKlMnOpQrStUv"},
        {"index": 1, "cache_id": "ZyXwVuTsRqPoNmLkJiHgFe"},
    ]


def test_extract_vibe_cache_ids_returns_empty_without_comment() -> None:
    content = "![image_0](data:image/png;base64,xxxx)\n<!-- seeds:[1] -->"
    assert NaiWebClient._extract_vibe_cache_ids(content) == []


def test_extract_vibe_cache_ids_skips_invalid_entries() -> None:
    """缺 cache_id 或非 dict 的条目应被丢弃，避免污染缓存键。"""
    content = (
        '<!-- vibe_cache_ids:[{"index":0,"cache_id":""},'
        '"bad",{"index":1,"cache_id":"ok"}] -->'
    )
    assert NaiWebClient._extract_vibe_cache_ids(content) == [
        {"index": 1, "cache_id": "ok"}
    ]


def test_format_usage_renders_compact_fields() -> None:
    """usage 渲染应保持字段顺序并附带 anlas 换算，方便对账。"""
    formatted = NaiWebClient._format_usage(
        {"prompt_tokens": 1, "completion_tokens": 30000, "total_tokens": 30001}
    )
    assert formatted == "usage[prompt=1, completion=30000(3.00 anlas), total=30001]"


def test_format_usage_handles_missing_or_invalid() -> None:
    assert NaiWebClient._format_usage(None) == ""
    assert NaiWebClient._format_usage("not-a-dict") == ""
    assert NaiWebClient._format_usage({"prompt_tokens": "x"}) == ""


def _make_response(status_code: int, json_body: object | None = None, text: str = "") -> object:
    """构造一个最小可用的 requests.Response 替身。"""

    class _FakeResponse:
        def __init__(self) -> None:
            self.status_code = status_code
            self._json = json_body
            self.text = text or (json.dumps(json_body) if json_body is not None else "")

        def json(self):
            if self._json is None:
                raise ValueError("no json")
            return self._json

    return _FakeResponse()


def _make_client_stub() -> NaiWebClient:
    """绕过 __init__ 的 Session 创建，只测纯解析逻辑。"""
    client = NaiWebClient.__new__(NaiWebClient)

    async def run_inline(function, *args, **_kwargs):
        return function(*args)

    client._run_blocking = run_inline
    return client


class _RecordingSession:
    def __init__(self, *, response: object | None = None, exc: Exception | None = None) -> None:
        self.response = response or _SessionResponse()
        self.exc = exc
        self.post_calls: list[dict[str, object]] = []

    def post(self, **kwargs):
        self.post_calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return self.response


class _SessionResponse:
    status_code = 200
    headers: dict[str, str] = {}
    url = "https://example.com/v1/chat/completions"


def _make_proxy_client_stub() -> NaiWebClient:
    client = _make_client_stub()
    client.log_prefix = "test"
    return client


def test_resolve_proxy_mode_defaults_to_direct() -> None:
    assert NaiWebClient._resolve_proxy_mode({}) == "direct"


def test_send_request_auto_uses_direct_session_before_env_proxy() -> None:
    client = _make_proxy_client_stub()
    direct_response = _SessionResponse()
    client.direct_session = _RecordingSession(response=direct_response)
    client.session = _RecordingSession()

    response = client._send_request(
        "https://example.com/v1/chat/completions",
        {"model": "nai-diffusion-4-5-full"},
        {"Authorization": "Bearer token"},
        "auto",
        12.0,
    )

    assert response is direct_response
    assert len(client.direct_session.post_calls) == 1
    assert client.session.post_calls == []


def test_send_request_auto_falls_back_to_env_proxy_when_direct_transport_fails() -> None:
    client = _make_proxy_client_stub()
    env_response = _SessionResponse()
    client.direct_session = _RecordingSession(
        exc=requests.exceptions.ConnectionError("direct network unavailable")
    )
    client.session = _RecordingSession(response=env_response)

    response = client._send_request(
        "https://example.com/v1/chat/completions",
        {"model": "nai-diffusion-4-5-full"},
        {"Authorization": "Bearer token"},
        "auto",
        12.0,
    )

    assert response is env_response
    assert len(client.direct_session.post_calls) == 1
    assert len(client.session.post_calls) == 1


def test_send_request_inherit_still_uses_environment_proxy_session() -> None:
    client = _make_proxy_client_stub()
    env_response = _SessionResponse()
    client.direct_session = _RecordingSession()
    client.session = _RecordingSession(response=env_response)

    response = client._send_request(
        "https://example.com/v1/chat/completions",
        {"model": "nai-diffusion-4-5-full"},
        {"Authorization": "Bearer token"},
        "inherit",
        12.0,
    )

    assert response is env_response
    assert client.direct_session.post_calls == []
    assert len(client.session.post_calls) == 1


def test_parse_models_response_extracts_ids_in_order() -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    response = _make_response(
        200,
        {
            "object": "list",
            "data": [
                {"id": "nai-diffusion-4-5-full", "object": "model"},
                {"id": "nai-diffusion-4-full"},
                {"id": "nai-diffusion-4-5-full"},  # 重复条目应被去重
                {"id": ""},  # 空 id 应被忽略
                "not-a-dict",  # 非 dict 应被忽略
            ],
        },
    )
    success, payload = client._parse_models_response(response)
    assert success is True
    assert payload == ["nai-diffusion-4-5-full", "nai-diffusion-4-full"]


def test_parse_models_response_reports_http_error_with_body() -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    response = _make_response(
        401,
        {"error": {"message": "invalid api key", "code": "UNAUTHORIZED"}},
    )
    success, payload = client._parse_models_response(response)
    assert success is False
    assert "401" in payload  # type: ignore[operator]
    assert "invalid api key" in payload  # type: ignore[operator]


def test_parse_models_response_rejects_missing_data_array() -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    response = _make_response(200, {"object": "list"})
    success, payload = client._parse_models_response(response)
    assert success is False
    assert "data" in payload  # type: ignore[operator]


def test_download_image_bytes_uses_get_and_closes_response(monkeypatch) -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    response = types.SimpleNamespace(
        status_code=200,
        headers={"content-type": "image/png"},
        content=b"png-bytes",
        text="",
        close_calls=0,
    )
    response.close = lambda: setattr(response, "close_calls", response.close_calls + 1)
    calls: list[tuple[object, ...]] = []

    def send_get(*args):
        calls.append(args)
        return response

    async def run_inline(function, *args, **_kwargs):
        return function(*args)

    monkeypatch.setattr(client, "_send_get_request", send_get)
    client._run_blocking = run_inline

    content = asyncio.run(
        client.download_image_bytes(
            "https://cdn.example.com/image.png",
            {
                "base_url": "https://cdn.example.com/v1",
                "api_key": "secret",
                "nai_proxy_mode": "auto",
                "nai_request_timeout": 12.5,
            },
        )
    )

    assert content == b"png-bytes"
    assert calls == [
        (
            "https://cdn.example.com/image.png",
            {"Accept": "image/*", "Authorization": "Bearer secret"},
            "auto",
            12.5,
        )
    ]
    assert response.close_calls == 1


def test_download_image_bytes_does_not_forward_api_key_cross_origin(monkeypatch) -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    response = types.SimpleNamespace(
        status_code=200,
        headers={"content-type": "image/png"},
        content=b"png-bytes",
        close_calls=0,
    )
    response.close = lambda: setattr(response, "close_calls", response.close_calls + 1)
    calls: list[tuple[object, ...]] = []

    def send_get(*args):
        calls.append(args)
        return response

    async def run_inline(function, *args, **_kwargs):
        return function(*args)

    monkeypatch.setattr(client, "_send_get_request", send_get)
    client._run_blocking = run_inline

    api_key = secrets.token_hex(16)
    content = asyncio.run(
        client.download_image_bytes(
            "https://cdn.example.com/image.png",
            {
                "base_url": "https://gateway.example.com/v1",
                "api_key": api_key,
            },
        )
    )

    assert content == b"png-bytes"
    assert calls[0][1] == {"Accept": "image/*"}
    assert api_key not in str(calls[0][1])
    assert response.close_calls == 1


def test_same_origin_normalizes_default_ports_and_rejects_scheme_changes() -> None:
    assert NaiWebClient._same_origin(
        "https://gateway.example.com/image.png",
        "https://gateway.example.com:443/v1",
    )
    assert not NaiWebClient._same_origin(
        "http://gateway.example.com/image.png",
        "https://gateway.example.com/v1",
    )
    assert not NaiWebClient._same_origin(
        "https://gateway.example.com:444/image.png",
        "https://gateway.example.com/v1",
    )


def test_download_image_bytes_rejects_http_error_non_image_and_empty_content(monkeypatch) -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    responses = [
        types.SimpleNamespace(
            status_code=404,
            headers={"content-type": "text/plain"},
            content=b"not found",
            text="not found",
            close_calls=0,
        ),
        types.SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"error":"not an image"}',
            text='{"error":"not an image"}',
            close_calls=0,
        ),
        types.SimpleNamespace(
            status_code=200,
            headers={"content-type": "image/png"},
            content=b"",
            text="",
            close_calls=0,
        ),
    ]
    for response in responses:
        response.close = lambda response=response: setattr(
            response,
            "close_calls",
            response.close_calls + 1,
        )

    def send_get(*_args):
        return responses.pop(0)

    async def run_inline(function, *args, **_kwargs):
        return function(*args)

    monkeypatch.setattr(client, "_send_get_request", send_get)
    client._run_blocking = run_inline
    model_config = {"nai_proxy_mode": "direct", "nai_request_timeout": 8.0}
    error_response, json_response, empty_response = list(responses)

    assert asyncio.run(client.download_image_bytes("https://cdn.example.com/missing", model_config)) is None
    assert asyncio.run(client.download_image_bytes("https://cdn.example.com/error", model_config)) is None
    assert asyncio.run(client.download_image_bytes("https://cdn.example.com/empty", model_config)) is None
    assert error_response.close_calls == 1
    assert json_response.close_calls == 1
    assert empty_response.close_calls == 1


def test_list_models_closes_response_after_parsing(monkeypatch) -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    response = types.SimpleNamespace(
        status_code=200,
        text="",
        close_calls=0,
        json=lambda: {"data": [{"id": "nai-diffusion-4-5-full"}]},
    )
    response.close = lambda: setattr(response, "close_calls", response.close_calls + 1)

    async def run_inline(function, *args, **_kwargs):
        return response

    client._run_blocking = run_inline

    result = asyncio.run(client.list_models({"base_url": "https://gateway.example.com"}))

    assert result == (True, ["nai-diffusion-4-5-full"])
    assert response.close_calls == 1


def test_generate_image_closes_response_after_parsing(monkeypatch) -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    client._last_response_vibe_cache_ids = []
    response = types.SimpleNamespace(close_calls=0)
    response.close = lambda: setattr(response, "close_calls", response.close_calls + 1)

    async def send_request(**_kwargs):
        return response

    monkeypatch.setattr(client, "_send_request_with_retry", send_request)
    monkeypatch.setattr(client, "_parse_response", lambda _response: (True, "image-base64"))

    result = asyncio.run(
        client.generate_image(
            "1girl",
            {
                "base_url": "https://gateway.example.com",
                "default_model": "nai-diffusion-4-5-full",
            },
        )
    )

    assert result == (True, "image-base64")
    assert response.close_calls == 1


def test_request_retry_closes_intermediate_response(monkeypatch) -> None:
    client = _make_client_stub()
    client.log_prefix = "test"
    first = types.SimpleNamespace(status_code=503, close_calls=0)
    final = types.SimpleNamespace(status_code=200, close_calls=0)
    for response in (first, final):
        response.close = lambda response=response: setattr(
            response,
            "close_calls",
            response.close_calls + 1,
        )
    responses = [first, final]

    def send_request(*_args):
        return responses.pop(0)

    async def run_inline(function, *args, **_kwargs):
        return function(*args)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(client, "_send_request", send_request)
    client._run_blocking = run_inline
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    response = asyncio.run(
        client._send_request_with_retry(
            "https://gateway.example.com/v1/chat/completions",
            {},
            {},
            "direct",
            10.0,
        )
    )

    assert response is final
    assert first.close_calls == 1
    assert final.close_calls == 0


def test_chami_sign_in_logs_in_and_posts_daily_checkin(monkeypatch) -> None:
    calls = []

    class _Response:
        def __init__(self, status_code, url, payload=None):
            self.status_code = status_code
            self.url = url
            self._payload = payload or {}

        def json(self):
            return self._payload

    class _Session:
        trust_env = True

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/login"):
                return _Response(200, "http://chami.yyqzx.com/dashboard")
            return _Response(200, url, {"status": "success", "message": "已签到"})

        def close(self):
            calls.append(("closed", {}))

    monkeypatch.setattr(
        "plugins.nai_draw_plugin.core.clients.nai_web_client.requests.Session",
        _Session,
    )
    success, message = NaiWebClient._perform_chami_sign_in(
        "demo-user", "demo-password", "direct", 12.0
    )

    assert (success, message) == (True, "已签到")
    assert calls[0][0] == "http://chami.yyqzx.com/login"
    assert calls[0][1]["data"] == {
        "username_or_email": "demo-user",
        "password": "demo-password",
        "real_ip": "",
    }
    assert calls[1][0] == "http://chami.yyqzx.com/sign_in"
    assert calls[1][1]["data"] == {}
    assert calls[-1][0] == "closed"


def test_chami_v3_payload_includes_legacy_negative_prompt_fields() -> None:
    body = NaiWebClient._build_chami_request_body(
        "1girl",
        "nai-diffusion-3-furry",
        {"size": [832, 1216], "scale": 5.0, "sampler": "k_euler_ancestral", "steps": 28, "negative_prompt": "lowres"},
    )

    assert body["parameters"]["uc"] == "lowres"
    assert body["parameters"]["sm"] is False
    assert body["parameters"]["sm_dyn"] is False


def test_chami_v5_uses_same_native_prompt_structure_as_v45() -> None:
    body = NaiWebClient._build_chami_request_body(
        "1girl, solo",
        "nai-diffusion-5-full",
        {
            "size": [832, 1216],
            "scale": 6.0,
            "sampler": "k_euler_ancestral",
            "steps": 28,
            "negative_prompt": "lowres",
            "noise_schedule": "karras",
        },
    )

    assert body["model"] == "nai-diffusion-5-full"
    assert body["parameters"]["legacy_v3_extend"] is False
    assert body["parameters"]["legacy_uc"] is False
    assert body["parameters"]["v4_prompt"]["caption"]["base_caption"] == "1girl, solo"
    assert body["parameters"]["v4_negative_prompt"]["caption"]["base_caption"] == "lowres"
