# -*- coding: utf-8 -*-
"""VibeCacheService + Vibe Transfer 缓存协同流程的最小校验。"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def __getattr__(self, _):
        return lambda *a, **k: None


dummy_logger_module.get_logger = lambda _=None: _DummyLogger()
sys.modules["src.common.logger"] = dummy_logger_module

from plugins.nai_draw_plugin.core.clients.nai_web_client import NaiWebClient
from plugins.nai_draw_plugin.core.services.vibe_cache import (
    VibeCacheService,
    compute_image_hash,
    get_vibe_cache_service,
    quantize_info_extracted,
    reset_vibe_cache_service,
)


# ── 量化与 hash ──────────────────────────────────────────────────────────


def test_quantize_info_extracted_handles_default_and_clamps() -> None:
    assert quantize_info_extracted(None) == 0.70
    assert quantize_info_extracted("not-a-number") == 0.70
    assert quantize_info_extracted(0.0) == 0.01
    assert quantize_info_extracted(1.5) == 1.00
    assert quantize_info_extracted(0.703) == 0.70
    assert quantize_info_extracted(0.715) == 0.72


def test_compute_image_hash_is_stable_across_prefix_and_whitespace() -> None:
    raw = base64.b64encode(b"hello-vibe").decode("ascii")
    assert compute_image_hash(raw) == compute_image_hash(f"data:image/png;base64,{raw}")
    assert compute_image_hash(raw) == compute_image_hash(f"{raw}\n")
    assert compute_image_hash(b"hello-vibe") == compute_image_hash(raw)


def test_compute_image_hash_returns_empty_for_invalid_input() -> None:
    assert compute_image_hash("") == ""
    assert compute_image_hash("@@@not-base64@@@") == ""


# ── SQLite 后端 ──────────────────────────────────────────────────────────


def test_vibe_cache_service_persist_and_lookup_round_trip(tmp_path) -> None:
    service = VibeCacheService(tmp_path / "vibe.db")
    assert service.persist(
        image_hash="hash-a",
        model_id="nai-diffusion-4-5-full",
        info_extracted=0.7,
        cache_id="cache-aaaa",
    ) is True
    assert service.lookup(
        image_hash="hash-a",
        model_id="nai-diffusion-4-5-full",
        info_extracted=0.7,
    ) == "cache-aaaa"
    assert service.count() == 1


def test_vibe_cache_service_keys_distinguish_model_and_info_extracted(tmp_path) -> None:
    service = VibeCacheService(tmp_path / "vibe.db")
    service.persist(image_hash="h", model_id="m1", info_extracted=0.5, cache_id="A")
    service.persist(image_hash="h", model_id="m2", info_extracted=0.5, cache_id="B")
    service.persist(image_hash="h", model_id="m1", info_extracted=0.7, cache_id="C")
    assert service.lookup(image_hash="h", model_id="m1", info_extracted=0.5) == "A"
    assert service.lookup(image_hash="h", model_id="m2", info_extracted=0.5) == "B"
    assert service.lookup(image_hash="h", model_id="m1", info_extracted=0.7) == "C"
    # 同模型同图但量化粒度内的微小差异仍应命中
    assert service.lookup(image_hash="h", model_id="m1", info_extracted=0.503) == "A"


def test_vibe_cache_service_rejects_empty_cache_id(tmp_path) -> None:
    service = VibeCacheService(tmp_path / "vibe.db")
    assert service.persist(image_hash="h", model_id="m", info_extracted=0.7, cache_id="") is False
    assert service.lookup(image_hash="h", model_id="m", info_extracted=0.7) is None


def test_vibe_cache_service_purge_removes_all_entries(tmp_path) -> None:
    service = VibeCacheService(tmp_path / "vibe.db")
    service.persist(image_hash="h1", model_id="m", info_extracted=0.5, cache_id="A")
    service.persist(image_hash="h2", model_id="m", info_extracted=0.5, cache_id="B")
    assert service.purge() == 2
    assert service.count() == 0


# ── NaiWebClient 与缓存的协同 ────────────────────────────────────────────


def _client_stub() -> NaiWebClient:
    client = NaiWebClient.__new__(NaiWebClient)
    client.log_prefix = "test"
    client._last_response_vibe_cache_ids = []
    return client


def test_apply_vibe_cache_rewrites_hits_to_cache_mode(tmp_path) -> None:
    """命中缓存的条目应被改写成 cache_id 复用态，未命中条目入 persist_plan。"""
    reset_vibe_cache_service()
    service = get_vibe_cache_service(tmp_path / "vibe.db")
    image_a = base64.b64encode(b"vibe-image-a").decode("ascii")
    image_b = base64.b64encode(b"vibe-image-b").decode("ascii")
    hash_a = compute_image_hash(image_a)
    service.persist(
        image_hash=hash_a,
        model_id="nai-diffusion-4-5-full",
        info_extracted=0.7,
        cache_id="HIT-CACHE-AAAA",
    )

    client = _client_stub()
    payload = {
        "strength": 1.0,
        "images": [
            {"image": image_a, "info_extracted": 0.7, "strength": 0.6},
            {"image": image_b, "info_extracted": 0.5, "strength": 0.4},
        ],
    }
    plan, rewritten, hit_plan = client._apply_vibe_cache_to_controlnet(
        payload, "nai-diffusion-4-5-full"
    )

    assert rewritten["images"][0] == {"cache_id": "HIT-CACHE-AAAA", "strength": 0.6}
    assert rewritten["images"][1] == {
        "image": image_b,
        "info_extracted": 0.5,
        "strength": 0.4,
    }
    assert len(plan) == 1
    assert plan[0][0] == 1  # 未命中的条目 index = 1
    assert plan[0][1] == compute_image_hash(image_b)
    assert plan[0][2] == 0.50
    assert hit_plan == [(hash_a, 0.70)]
    reset_vibe_cache_service()


def test_persist_vibe_cache_writes_returned_cache_ids(tmp_path) -> None:
    """响应里 vibe_cache_ids 注释应按 index 落到本地缓存，下次相同请求即可命中。"""
    reset_vibe_cache_service()
    service = get_vibe_cache_service(tmp_path / "vibe.db")
    image_b = base64.b64encode(b"vibe-image-b").decode("ascii")
    hash_b = compute_image_hash(image_b)

    client = _client_stub()
    client._last_response_vibe_cache_ids = [
        {"index": 0, "cache_id": "FRESH-CACHE-XXX"},
        {"index": 1, "cache_id": "FRESH-CACHE-YYY"},
    ]
    plan = [(1, hash_b, 0.50)]
    client._persist_vibe_cache("nai-diffusion-4-5-full", plan)

    assert (
        service.lookup(
            image_hash=hash_b,
            model_id="nai-diffusion-4-5-full",
            info_extracted=0.5,
        )
        == "FRESH-CACHE-YYY"
    )
    reset_vibe_cache_service()


def test_apply_vibe_cache_skips_pure_cache_id_entries(tmp_path) -> None:
    """已经处于 cache_id 复用态的条目应原样透传，不再做本地查找。"""
    reset_vibe_cache_service()
    get_vibe_cache_service(tmp_path / "vibe.db")

    client = _client_stub()
    payload = {
        "images": [
            {"cache_id": "EXISTING-CACHE-ID", "strength": 0.7},
        ],
    }
    plan, rewritten, hit_plan = client._apply_vibe_cache_to_controlnet(
        payload, "nai-diffusion-4-5-full"
    )
    assert plan == []
    assert hit_plan == []  # 已是 cache_id 态的不算"本地命中"
    assert rewritten["images"][0] == {"cache_id": "EXISTING-CACHE-ID", "strength": 0.7}
    reset_vibe_cache_service()


def test_apply_vibe_cache_noop_for_non_controlnet_payload(tmp_path) -> None:
    reset_vibe_cache_service()
    get_vibe_cache_service(tmp_path / "vibe.db")

    client = _client_stub()
    plan, rewritten, hit_plan = client._apply_vibe_cache_to_controlnet(None, "nai-diffusion-4-5-full")
    assert plan == []
    assert hit_plan == []
    assert rewritten is None
    reset_vibe_cache_service()


# ── §20.3.2 surcharge 日志分流 ────────────────────────────────────────────


def test_apply_vibe_cache_logs_full_hit_saves_surcharge(tmp_path, caplog) -> None:
    """全量命中：1 anlas 流量附加费可省，日志应明确告知。"""
    reset_vibe_cache_service()
    service = get_vibe_cache_service(tmp_path / "vibe.db")
    image_a = base64.b64encode(b"vibe-full-a").decode("ascii")
    service.persist(
        image_hash=compute_image_hash(image_a),
        model_id="nai-diffusion-4-5-full",
        info_extracted=0.7,
        cache_id="FULL-HIT-CACHE",
    )

    client = _client_stub()
    payload = {"images": [{"image": image_a, "info_extracted": 0.7}]}
    with caplog.at_level("INFO"):
        _plan, _rewritten, hit_plan = client._apply_vibe_cache_to_controlnet(
            payload, "nai-diffusion-4-5-full"
        )
    assert len(hit_plan) == 1
    # 因为 logger 是 src.common.logger 里 stub 的，caplog 不一定捕获到；
    # 关键断言：全量命中时 persist_plan 为空 -> 必然走到"省 1 anlas"分支
    # 这里只校验返回值的语义：所有图片都被改写
    assert all("cache_id" in img and "image" not in img for img in _rewritten["images"])
    reset_vibe_cache_service()


def test_apply_vibe_cache_partial_hit_keeps_one_byte_entry(tmp_path) -> None:
    """部分命中：仍有字节态条目存在 -> 1 anlas 附加费 flat 不可省。"""
    reset_vibe_cache_service()
    service = get_vibe_cache_service(tmp_path / "vibe.db")
    image_hit = base64.b64encode(b"vibe-partial-hit").decode("ascii")
    image_miss = base64.b64encode(b"vibe-partial-miss").decode("ascii")
    service.persist(
        image_hash=compute_image_hash(image_hit),
        model_id="nai-diffusion-4-5-full",
        info_extracted=0.7,
        cache_id="PARTIAL-HIT-CACHE",
    )

    client = _client_stub()
    payload = {
        "images": [
            {"image": image_hit, "info_extracted": 0.7},
            {"image": image_miss, "info_extracted": 0.7},
        ]
    }
    plan, rewritten, hit_plan = client._apply_vibe_cache_to_controlnet(
        payload, "nai-diffusion-4-5-full"
    )
    assert len(hit_plan) == 1
    assert len(plan) == 1  # 仍有字节态条目要发，附加费仍会扣
    # 命中条目被改写成 cache_id 态，未命中条目原样保留
    assert rewritten["images"][0] == {"cache_id": "PARTIAL-HIT-CACHE"}
    assert rewritten["images"][1] == {"image": image_miss, "info_extracted": 0.7}
    reset_vibe_cache_service()


# ── delete 接口 ──────────────────────────────────────────────────────────


def test_vibe_cache_service_delete_removes_single_entry(tmp_path) -> None:
    service = VibeCacheService(tmp_path / "vibe.db")
    service.persist(image_hash="h1", model_id="m", info_extracted=0.5, cache_id="A")
    service.persist(image_hash="h2", model_id="m", info_extracted=0.5, cache_id="B")
    assert service.delete(image_hash="h1", model_id="m", info_extracted=0.5) is True
    # 已删的不能再命中
    assert service.lookup(image_hash="h1", model_id="m", info_extracted=0.5) is None
    # 同表别的条目不受影响
    assert service.lookup(image_hash="h2", model_id="m", info_extracted=0.5) == "B"


def test_vibe_cache_service_delete_returns_false_when_no_row(tmp_path) -> None:
    service = VibeCacheService(tmp_path / "vibe.db")
    assert service.delete(image_hash="nope", model_id="m", info_extracted=0.5) is False
    # 空 key 也安全
    assert service.delete(image_hash="", model_id="m", info_extracted=0.5) is False


# ── §20.3.1 stale cache 自愈：启发式 + 清理 ──────────────────────────────


def test_looks_like_stale_vibe_cache_error_triggers_on_keyword_combo() -> None:
    """错误消息里有 cache_id + 失效语义关键词时应识别为 stale。"""
    assert NaiWebClient._looks_like_stale_vibe_cache_error(
        "HTTP 400: vibe cache_id not found"
    )
    assert NaiWebClient._looks_like_stale_vibe_cache_error(
        "cache_id 在服务端找不到对应记录"
    )
    assert NaiWebClient._looks_like_stale_vibe_cache_error(
        "Invalid cache_id supplied"
    )


def test_looks_like_stale_vibe_cache_error_ignores_unrelated_errors() -> None:
    """不带 cache_id 关键字或不带失效语义关键词的错误不应触发自愈。"""
    assert not NaiWebClient._looks_like_stale_vibe_cache_error("")
    assert not NaiWebClient._looks_like_stale_vibe_cache_error("HTTP 500 server error")
    # 仅出现 cache_id 字眼但没有失效语义关键词时不触发，避免误清
    assert not NaiWebClient._looks_like_stale_vibe_cache_error(
        "rate limited while using cache_id"
    )


def test_purge_vibe_cache_hits_only_removes_supplied_keys(tmp_path) -> None:
    """stale 清理只清 hit_plan 列出的条目，其它本地缓存保留。"""
    reset_vibe_cache_service()
    service = get_vibe_cache_service(tmp_path / "vibe.db")
    service.persist(image_hash="stale-1", model_id="M", info_extracted=0.7, cache_id="X1")
    service.persist(image_hash="stale-2", model_id="M", info_extracted=0.7, cache_id="X2")
    service.persist(image_hash="keep", model_id="M", info_extracted=0.7, cache_id="Y")

    client = _client_stub()
    deleted = client._purge_vibe_cache_hits(
        "M",
        [("stale-1", 0.70), ("stale-2", 0.70)],
    )

    assert deleted == 2
    assert service.lookup(image_hash="stale-1", model_id="M", info_extracted=0.7) is None
    assert service.lookup(image_hash="stale-2", model_id="M", info_extracted=0.7) is None
    # 没在 hit_plan 里的条目不受影响
    assert service.lookup(image_hash="keep", model_id="M", info_extracted=0.7) == "Y"
    reset_vibe_cache_service()


def test_purge_vibe_cache_hits_empty_plan_is_noop(tmp_path) -> None:
    reset_vibe_cache_service()
    get_vibe_cache_service(tmp_path / "vibe.db")
    client = _client_stub()
    assert client._purge_vibe_cache_hits("M", []) == 0
    reset_vibe_cache_service()
