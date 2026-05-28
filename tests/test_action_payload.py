# -*- coding: utf-8 -*-
"""compose_description_from_action_payload 的回归测试。

历史 bug：旧版"5 字段非空就忽略 description"会丢失 description 里的角色锚点，
导致"画一张初音未来"被下游误判 bot 自拍。本组用例守护"description 永不被丢"。
"""

from __future__ import annotations

import os
import sys
import importlib.util


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOD_PATH = os.path.join(PLUGIN_DIR, "core", "utils", "action_payload.py")

_spec = importlib.util.spec_from_file_location("action_payload", MOD_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载模块: {MOD_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

compose_description_from_action_payload = _mod.compose_description_from_action_payload
STRUCTURED_DESCRIPTION_FIELDS = _mod.STRUCTURED_DESCRIPTION_FIELDS


def test_keeps_description_when_structured_fields_also_present():
    """description 与结构化字段共存：两者都拼，description 在前作为语义主体。

    历史 bug 用例：Planner 给 5 个结构化字段 + description（含角色名）时，旧版会
    完全忽略 description，导致"初音未来"等核心锚点丢失。
    """
    action_data = {
        "subject_and_pov": "一女 第三视角",
        "action": "站立",
        "emotion": "俏皮 微笑",
        "scene_delta": "",
        "framing": "特写",
        "description": "一女, 初音未来, 公式服, 葱色双马尾, 精致的面容, 灵动的眼神, 背景为舞台, 特写",
    }
    out = compose_description_from_action_payload(action_data)

    # description 必须在前（主体）+ 结构化字段在后（补充）
    assert out.startswith("一女, 初音未来, 公式服")
    # 结构化字段也要拼上来
    assert "一女 第三视角" in out
    assert "站立" in out
    assert "俏皮 微笑" in out
    assert "特写" in out
    # 关键：角色锚点不能丢
    assert "初音未来" in out


def test_falls_back_to_structured_only_when_description_missing():
    """description 字段空时仅用结构化字段，跟历史行为兼容。"""
    action_data = {
        "subject_and_pov": "一女 第一视角",
        "action": "微笑",
        "framing": "selfie",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "一女 第一视角 微笑 selfie"


def test_falls_back_to_description_only_when_structured_all_empty():
    """结构化字段全空时仅用 description。"""
    action_data = {
        "description": "初音未来, 公式服",
        "subject_and_pov": "",
        "action": "",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "初音未来, 公式服"


def test_both_empty_returns_empty_string():
    """description 与结构化字段都空 → 空串（调用方负责回落 reasoning）。"""
    assert compose_description_from_action_payload({}) == ""
    assert compose_description_from_action_payload({"description": "", "action": ""}) == ""


def test_structured_field_order_preserved():
    """结构化字段按 STRUCTURED_DESCRIPTION_FIELDS 声明顺序拼接。"""
    action_data = {
        "framing": "F",
        "action": "B",
        "subject_and_pov": "A",
        "emotion": "C",
        "scene_delta": "D",
    }
    out = compose_description_from_action_payload(action_data)
    # 顺序：subject_and_pov action emotion scene_delta framing
    assert out == "A B C D F"


def test_strips_whitespace_in_each_field():
    """各字段头尾空白被 strip，避免拼出 ``"  a   b"`` 这种脏数据。"""
    action_data = {
        "subject_and_pov": "  pov  ",
        "action": "\nstanding\n",
        "description": "  girl  ",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "girl pov standing"


def test_non_string_values_coerced_to_empty():
    """Planner 偶发把数字/None 塞进字段时不应抛，按空处理。"""
    action_data = {
        "subject_and_pov": None,
        "action": 0,
        "description": "1girl",
    }
    out = compose_description_from_action_payload(action_data)
    assert out == "1girl"


def test_structured_description_fields_constant_unchanged():
    """常量值锁死，避免顺序被无意打乱影响下游 NAI tag 排序约定。"""
    assert STRUCTURED_DESCRIPTION_FIELDS == (
        "subject_and_pov",
        "action",
        "emotion",
        "scene_delta",
        "framing",
    )
