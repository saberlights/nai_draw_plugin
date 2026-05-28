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
is_named_character_intent = _mod.is_named_character_intent
NAMED_CHARACTER_TOKEN = _mod.NAMED_CHARACTER_TOKEN


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


# ============ is_named_character_intent ============

def test_named_character_intent_matches_when_token_present():
    """Planner 在 subject_and_pov 写了 ``画指定角色`` 前缀 → True。

    这是上游声明"本轮主体是指定角色，非 bot 出镜"的唯一信号。命中后 sdk_runtime
    会跳过 self-image 注入与 selfie 后处理，避免把指定角色洗成 bot 默认外貌。
    """
    assert is_named_character_intent({
        "subject_and_pov": f"{NAMED_CHARACTER_TOKEN} 一女 第三视角",
        "description": "初音未来, 葱色双马尾",
    }) is True


def test_named_character_intent_no_token_means_bot_self():
    """没有 token：Planner 把本轮视为 bot 自己出镜（含 cosplay 场景）→ False。

    Cosplay bot 出镜场景照常返回 False——出镜的还是 bot，只是穿成别的角色。
    """
    assert is_named_character_intent({
        "subject_and_pov": "一女",
        "description": "cosplay 初音未来",
    }) is False
    assert is_named_character_intent({"subject_and_pov": "一女 自拍"}) is False
    assert is_named_character_intent({}) is False


def test_named_character_intent_only_reads_subject_field():
    """token 只在 ``subject_and_pov`` 中识别——其它字段含同名字符串不算数。

    避免 description 里偶然出现"画指定角色"这种自然语言误触发；Planner 必须显式
    在 subject 字段声明意图，约定收口一处。
    """
    assert is_named_character_intent({
        "subject_and_pov": "一女",
        "description": f"{NAMED_CHARACTER_TOKEN} 初音未来",
    }) is False


def test_named_character_intent_token_anywhere_in_subject():
    """token 在 subject 内任意位置（前/中/后）都算命中。

    Planner 偶尔会把 token 放视角后（``一女 画指定角色`` 这种），不强制 token 在最前。
    """
    assert is_named_character_intent({
        "subject_and_pov": f"一女 {NAMED_CHARACTER_TOKEN} 第三视角",
    }) is True
    assert is_named_character_intent({
        "subject_and_pov": f"一女 第三视角 {NAMED_CHARACTER_TOKEN}",
    }) is True


def test_named_character_intent_non_string_subject_safe():
    """Planner 偶发 None/数字时不抛，按未声明处理。"""
    assert is_named_character_intent({"subject_and_pov": None}) is False
    assert is_named_character_intent({"subject_and_pov": 0}) is False
