# -*- coding: utf-8 -*-
"""图片缓存与引用回复解析的单元测试。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.retag.image_cache import ImageCacheService


def _msg_with_image(message_id: str, stream_id: str, image_b64: str, *, ts: float | None = None) -> dict:
    return {
        "message_id": message_id,
        "session_id": stream_id,
        "timestamp": ts if ts is not None else time.time(),
        "raw_message": [
            {"type": "image", "data": {"binary_data_base64": image_b64}}
        ],
    }


def _png_like_b64(marker: str) -> str:
    # 以 PNG base64 magic 开头，加上足够长度让 _looks_like_image_data 通过
    return "iVBORw0KGgo" + marker + "A" * 100


def test_cache_inbound_then_resolve_via_stream() -> None:
    svc = ImageCacheService(cache_ttl_seconds=600, per_stream_capacity=5)
    svc.cache_inbound_message(_msg_with_image("m1", "s1", _png_like_b64("X")))
    svc.cache_inbound_message(_msg_with_image("m2", "s1", _png_like_b64("Y")))

    # 没有命令消息时 resolve 应该返回最近一张图（m2）
    img = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    assert img is not None and img.startswith("iVBORw0KGgoY")


def test_resolve_prefers_command_image_over_stream() -> None:
    svc = ImageCacheService(cache_ttl_seconds=600, per_stream_capacity=5)
    svc.cache_inbound_message(_msg_with_image("m1", "s1", _png_like_b64("OLD")))

    # 当前命令消息自己就带图
    cmd_msg = {
        "message_id": "mc",
        "session_id": "s1",
        "timestamp": time.time(),
        "message_info": {"user_info": {"user_id": "u1"}},
        "raw_message": [
            {"type": "image", "data": {"binary_data_base64": _png_like_b64("CMD")}},
            {"type": "text", "data": "/nai 反推"},
        ],
    }
    svc.remember_command_message(cmd_msg)

    img = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    assert img is not None and img.startswith("iVBORw0KGgoCMD")


def test_resolve_via_reply_target() -> None:
    svc = ImageCacheService(cache_ttl_seconds=600, per_stream_capacity=5)
    svc.cache_inbound_message(_msg_with_image("m1", "s1", _png_like_b64("TARGET")))
    svc.cache_inbound_message(_msg_with_image("m2", "s1", _png_like_b64("LATER")))

    cmd_msg = {
        "message_id": "mc",
        "session_id": "s1",
        "timestamp": time.time(),
        "message_info": {"user_info": {"user_id": "u1"}},
        "raw_message": [
            {"type": "reply", "data": {"target_message_id": "m1"}},
            {"type": "text", "data": "/nai 反推"},
        ],
    }
    svc.remember_command_message(cmd_msg)

    img = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    # 命令消息引用了 m1，应该拿到 TARGET 而不是 LATER
    assert img is not None and img.startswith("iVBORw0KGgoTARGET")


def test_command_message_consumed_only_once() -> None:
    svc = ImageCacheService(cache_ttl_seconds=600, per_stream_capacity=5)
    svc.cache_inbound_message(_msg_with_image("m1", "s1", _png_like_b64("A")))
    cmd_msg = {
        "message_id": "mc",
        "session_id": "s1",
        "timestamp": time.time(),
        "message_info": {"user_info": {"user_id": "u1"}},
        "raw_message": [
            {"type": "reply", "data": {"target_message_id": "m1"}},
            {"type": "text", "data": "/nai 反推"},
        ],
    }
    svc.remember_command_message(cmd_msg)

    # 第一次解析消费掉命令消息
    img1 = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    assert img1 is not None and img1.startswith("iVBORw0KGgoA")

    # 第二次再 resolve 不应该再走 reply，会退到流内最近图
    img2 = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    assert img2 is not None and img2.startswith("iVBORw0KGgoA")  # 流内还是 m1


def test_ttl_expires_old_images() -> None:
    """超过 TTL 的缓存条目应被清理。"""
    svc = ImageCacheService(cache_ttl_seconds=1, per_stream_capacity=5)
    old_ts = time.time() - 100
    svc.cache_inbound_message(_msg_with_image("m_old", "s1", _png_like_b64("OLD"), ts=old_ts))

    img = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    assert img is None


def test_per_stream_capacity_evicts_oldest() -> None:
    svc = ImageCacheService(cache_ttl_seconds=600, per_stream_capacity=2)
    svc.cache_inbound_message(_msg_with_image("m1", "s1", _png_like_b64("A")))
    svc.cache_inbound_message(_msg_with_image("m2", "s1", _png_like_b64("B")))
    svc.cache_inbound_message(_msg_with_image("m3", "s1", _png_like_b64("C")))

    # 容量 2 后只保留 m2/m3，最近的是 m3=C
    img = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    assert img is not None and img.startswith("iVBORw0KGgoC")


def test_clear_drops_everything() -> None:
    svc = ImageCacheService(cache_ttl_seconds=600, per_stream_capacity=5)
    svc.cache_inbound_message(_msg_with_image("m1", "s1", _png_like_b64("A")))
    svc.clear()
    img = svc.resolve_image_base64(stream_id="s1", user_id="u1")
    assert img is None
