# -*- coding: utf-8 -*-
"""验证 `/nai help` 的结构化文档覆盖关键公开命令。"""

from __future__ import annotations

# help_renderer 现在从模型注册表取模型代号清单（避免帮助文案与注册表漂移），
# 因此必须按包导入，不能再用 spec_from_file_location 单文件加载。
from plugins.nai_draw_plugin.core.utils.help_renderer import (
    HELP_DOC,
    HELP_FALLBACK_TEXT,
)


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
