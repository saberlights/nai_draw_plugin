# -*- coding: utf-8 -*-
"""验证 `/nai help` 的结构化文档覆盖关键公开命令。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "utils" / "help_renderer.py"
_SPEC = importlib.util.spec_from_file_location("nai_help_renderer", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"无法加载帮助渲染模块：{_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

HELP_DOC = _MODULE.HELP_DOC
HELP_FALLBACK_TEXT = _MODULE.HELP_FALLBACK_TEXT


def _documented_commands() -> tuple[str, ...]:
    return tuple(command for section in HELP_DOC.sections for command, _ in section.items)


def _description_for(command: str) -> str:
    for section in HELP_DOC.sections:
        for documented_command, description in section.items:
            if documented_command == command:
                return description
    raise AssertionError(f"HELP_DOC 缺少命令：{command}")


def test_help_lists_random_role_and_key_operational_commands() -> None:
    """帮助页必须覆盖用户容易遗漏但实际可用的关键命令。"""
    commands = _documented_commands()
    required_commands = (
        "/nai 随机 [角色]",
        "/nai 随机自拍 [角色]",
        "/nai tag on/off",
        "/nai models",
        "/nai planner [on|off]",
        "/nai ref类型 [character|style|both]",
        "/nai vibe清空",
        "/nai ref清空",
    )

    for command in required_commands:
        assert command in commands, f"HELP_DOC 缺少关键命令：{command}"
        assert command in HELP_FALLBACK_TEXT, f"纯文本回退缺少关键命令：{command}"


def test_fallback_text_contains_every_structured_help_command() -> None:
    """HTML 渲染失败时，纯文本回退不能丢掉结构化帮助里的命令。"""
    for command in _documented_commands():
        assert command in HELP_FALLBACK_TEXT, f"纯文本回退遗漏命令：{command}"


def test_planner_help_states_permission_scope_and_default() -> None:
    description = _description_for("/nai planner [on|off]")

    assert "仅管理员" in description
    assert "会话级" in description
    assert "默认开启" in description
