# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

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
    assert inner["autoSmea"] is True
    # 文档 §5 表外的 sm/sm_dyn 字段已不再透传，避免触发网关 400
    assert "sm" not in inner
    assert "sm_dyn" not in inner
    assert inner["variety_boost"] is True
    assert inner["image_format"] == "webp"
    assert "model" not in inner


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
    return NaiWebClient.__new__(NaiWebClient)


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
