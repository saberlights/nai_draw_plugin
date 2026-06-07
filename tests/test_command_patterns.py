# -*- coding: utf-8 -*-
"""守护图生图命令（/nai i2i / ref / vibe）的 pattern 兼容 reply 前缀。

这几条命令必然伴随"引用回复一张图"链路，各平台的 reply 前缀形态不一
（CQ:reply / [回复 xxx] / 转述 xxx，说：），曾经因为 pattern 用了严格的
``(?:.*，说：\\s*)?`` 起手导致带 reply 前缀的消息匹不上、用户看到"没反应"。

本测试直接从 ``plugin.py`` 抠出命令注册时声明的 ``pattern=...`` 字面量，
避免拉起整个 plugin module，只验证字符串本身的语义。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


_PLUGIN_FILE = Path(__file__).resolve().parents[1] / "plugin.py"
# 跨越多行 / 跨越 description 注释，但不能跨越下一个 @Command 装饰器，
# 否则会把上一个命令的 name 跟下一个命令的 pattern 误配。
# name 用 \w（含数字）匹配，否则 ``nai_i2i_command`` 里的 ``2`` 会卡住
_COMMAND_BLOCK_RE = re.compile(
    r'@Command\(\s*"(?P<name>nai_\w+)"'
    r'(?:(?!@Command).)*?'
    r'pattern=r"(?P<pattern>[^"]+)"',
    re.DOTALL,
)


def _load_command_patterns() -> Dict[str, str]:
    """从 plugin.py 抠出 ``@Command(name=..., pattern=...)`` 的字面量映射。"""
    source = _PLUGIN_FILE.read_text(encoding="utf-8")
    return {
        m.group("name"): m.group("pattern")
        for m in _COMMAND_BLOCK_RE.finditer(source)
    }


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _matching_command_names(patterns: Dict[str, str], message: str) -> list[str]:
    """按 plugin.py 注册顺序返回所有能匹配该消息的命令名。"""
    return [
        name
        for name, pattern in patterns.items()
        if _compile(pattern).search(message)
    ]


# ── 图生图族命令 pattern ─────────────────────────────────────────────────


def test_i2i_ref_vibe_patterns_match_plain_invocation() -> None:
    """无 reply 前缀时直接 /nai i2i / ref / vibe <描述> 必须能匹配。"""
    patterns = _load_command_patterns()
    for name, sample in [
        ("nai_i2i_command", "/nai i2i 变成躺姿"),
        ("nai_ref_command", "/nai ref 站在街道，看向镜头"),
        ("nai_vibe_command", "/nai vibe 都市夜景，霓虹氛围"),
    ]:
        regex = _compile(patterns[name])
        match = regex.match(sample)
        assert match is not None, f"{name} 应匹配 {sample!r}"
        assert match.group("description"), f"{name} 应能解析出 description"


def test_i2i_ref_vibe_patterns_tolerate_quote_reply_prefix() -> None:
    """各平台引用回复前缀（非 ``xxx，说：`` 形态）也必须能命中。

    这是历史回归点：原 pattern 用 ``(?:.*，说：\\s*)?`` 严格起手时，下述任一前缀都会
    让命令匹不上、用户看到"没反应"。
    """
    patterns = _load_command_patterns()
    quote_prefixes = [
        "[回复 alice 的消息: 来一张] ",
        "[回复了alice的消息: 来一张] ",
        "[CQ:reply,id=12345] ",
        "alice，说：",  # 老的 MaiBot 转述前缀仍需兼容
    ]
    for name, body in [
        ("nai_i2i_command", "/nai i2i 改成森林背景"),
        ("nai_ref_command", "/nai ref 红发少女"),
        ("nai_vibe_command", "/nai vibe 赛博朋克氛围"),
    ]:
        regex = _compile(patterns[name])
        for prefix in quote_prefixes:
            message = f"{prefix}{body}"
            match = regex.match(message)
            assert match is not None, f"{name} 应匹配带前缀的 {message!r}"


def test_generic_nai_pattern_still_skips_i2i_ref_vibe() -> None:
    """generic /nai pattern 的 negative lookahead 仍应排除 i2i / ref / vibe（含 CJK 子命令）。

    保证 ``/nai i2i ...`` / ``/nai vibe存 ...`` 不会被通用 ``/nai <描述>`` 命令吞掉。
    """
    patterns = _load_command_patterns()
    generic = _compile(patterns["nai_draw"])
    skip_samples = [
        "/nai i2i 任意描述",
        "/nai ref 任意描述",
        "/nai vibe 任意描述",
        # CJK 子命令也必须被排除（latin→CJK 没有 \b，曾经的 vibe\b 在这里失效）
        "/nai vibe存 角色A",
        "/nai vibe图库",
        "/nai vibe删 角色A",
        "/nai vibe选 角色A",
        "/nai vibe选 角色A 角色B",  # 多名字选定
        "/nai vibe @角色A @角色B 描述",  # 多 @ 单次覆盖
        "/nai vibe清空",  # 一键清空（CJK 子命令）
        "/nai ref存 角色A",
        "/nai ref图库",
        "/nai ref删 角色A",
        "/nai ref选 角色A",
        "/nai ref清空",
    ]
    for sample in skip_samples:
        assert generic.match(sample) is None, f"通用 /nai 不应吞掉 {sample!r}"
    # 但正常 /nai <描述> 仍需匹配
    assert generic.match("/nai 画一张初音未来") is not None
    assert generic.match("/nai artgen 画师风格生成") is not None
    assert generic.match("/nai artr 画师风格重随机") is not None
    assert generic.match("/nai artfix 画师风格修复") is not None


# ── 命名图库子命令 pattern ───────────────────────────────────────────────


def test_named_reference_subcommands_match_with_chinese_name() -> None:
    """7 条命名图库子命令（save / delete / select）应能匹中文 / 英文 / 下划线名字。

    save / delete pattern 用 ``(?P<name>...)`` 单 token 捕获；
    select pattern 用 ``(?P<names>...)`` 捕获 1~N 个空格分隔的 token（vibe 多图 / ref 单图）。
    """
    patterns = _load_command_patterns()
    # (cmd_name, sample, expected, group_name)
    cases = [
        ("nai_vibe_save_command", "/nai vibe存 角色A", "角色A", "name"),
        ("nai_vibe_save_command", "/nai vibe存 char_b", "char_b", "name"),
        ("nai_vibe_delete_command", "/nai vibe删 角色A", "角色A", "name"),
        ("nai_vibe_select_command", "/nai vibe选 角色A", "角色A", "names"),
        ("nai_ref_save_command", "/nai ref存 角色A", "角色A", "name"),
        ("nai_ref_delete_command", "/nai ref删 角色A", "角色A", "name"),
        ("nai_ref_select_command", "/nai ref选 角色A", "角色A", "names"),
    ]
    for cmd_name, sample, expected, group_name in cases:
        regex = _compile(patterns[cmd_name])
        match = regex.match(sample)
        assert match is not None, f"{cmd_name} 应匹配 {sample!r}"
        assert match.group(group_name) == expected


def test_named_reference_list_subcommand_matches_without_args() -> None:
    """/nai vibe图库 / /nai ref图库 都是纯关键字命令，没有参数。"""
    patterns = _load_command_patterns()
    for cmd_name, sample in [
        ("nai_vibe_list_command", "/nai vibe图库"),
        ("nai_ref_list_command", "/nai ref图库"),
    ]:
        regex = _compile(patterns[cmd_name])
        assert regex.match(sample) is not None, f"{cmd_name} 应匹配 {sample!r}"


def test_vibe_ref_draw_patterns_capture_at_names_block() -> None:
    """/nai vibe @<n1> [@<n2>...] <描述> / /nai ref @<n> <描述> 的 at_names 整段应被独立捕获。

    pattern 用 ``(?P<at_names>(?:@\\S+\\s+)*)`` 把 0~N 个 ``@<名字>`` 整体捕获成一段字符串，
    命令层用 ``re.findall(r"@(\\S+)", ...)`` 拆解成 List[str]。不带 @ 时 at_names 为空串。
    """
    patterns = _load_command_patterns()
    cases = [
        ("nai_vibe_command", "/nai vibe @角色A 都市夜景", "@角色A ", "都市夜景"),
        ("nai_vibe_command", "/nai vibe @角色A @角色B 都市夜景", "@角色A @角色B ", "都市夜景"),
        ("nai_vibe_command", "/nai vibe @a @b @c @d 描述", "@a @b @c @d ", "描述"),
        ("nai_vibe_command", "/nai vibe 都市夜景", "", "都市夜景"),
        ("nai_ref_command", "/nai ref @角色A 站街道", "@角色A ", "站街道"),
        ("nai_ref_command", "/nai ref 站街道", "", "站街道"),
    ]
    for cmd_name, sample, expected_at_names, expected_desc in cases:
        regex = _compile(patterns[cmd_name])
        match = regex.match(sample)
        assert match is not None, f"{cmd_name} 应匹配 {sample!r}"
        assert match.group("at_names") == expected_at_names, (
            f"{cmd_name} 的 at_names 段不匹配 {sample!r}"
        )
        assert match.group("description") == expected_desc


def test_vibe_ref_select_patterns_capture_multiple_names() -> None:
    """/nai vibe选 <n1> [<n2>...] 的 names 段应捕获 1~N 个空格分隔的 token；ref选 同结构。

    store 层会按 scope 上限（vibe 4 / ref 1）做硬校验，pattern 不在这里限张数，
    避免 pattern 拒绝后用户看不到上限错误。
    """
    patterns = _load_command_patterns()
    cases = [
        ("nai_vibe_select_command", "/nai vibe选 角色A", "角色A"),
        ("nai_vibe_select_command", "/nai vibe选 角色A 角色B", "角色A 角色B"),
        ("nai_vibe_select_command", "/nai vibe选 a b c d", "a b c d"),
        ("nai_ref_select_command", "/nai ref选 角色A", "角色A"),
        # ref 单图 scope 也允许多 token 透传到 store 层报错（避免 pattern 静默吞）
        ("nai_ref_select_command", "/nai ref选 a b", "a b"),
    ]
    for cmd_name, sample, expected_names in cases:
        regex = _compile(patterns[cmd_name])
        match = regex.match(sample)
        assert match is not None, f"{cmd_name} 应匹配 {sample!r}"
        assert match.group("names") == expected_names


def test_subcommand_patterns_tolerate_quote_reply_prefix() -> None:
    """vibe存 / ref存 也常带 reply 前缀（用户回复一张图后存图），同样要兼容。"""
    patterns = _load_command_patterns()
    prefixes = ["[回复 alice 的消息: 这张] ", "[回复了alice的消息: 这张] ", "[CQ:reply,id=999] ", "alice，说："]
    for cmd_name, body in [
        ("nai_vibe_save_command", "/nai vibe存 角色A"),
        ("nai_ref_save_command", "/nai ref存 角色A"),
    ]:
        regex = _compile(patterns[cmd_name])
        for prefix in prefixes:
            message = f"{prefix}{body}"
            assert regex.match(message) is not None, f"{cmd_name} 应匹配 {message!r}"


def test_quote_reply_content_does_not_trigger_command() -> None:
    """引用内容里的历史命令不能被当成本次消息命令执行。"""
    patterns = _load_command_patterns()
    quoted_command_messages = [
        ("nai_retag_command", "[回复 alice 的消息: /nai 反推] 收到"),
        ("nai_retag_command", "[回复了alice的消息: /nai 反推] 收到"),
        ("nai_i2i_command", "[回复 alice 的消息: /nai i2i 改成森林背景] 收到"),
        ("nai_i2i_command", "[回复了alice的消息: /nai i2i 改成森林背景] 收到"),
        ("nai_vibe_save_command", "[回复 alice 的消息: /nai vibe存 角色A] 收到"),
        ("nai_ref_type_command", "[回复 alice 的消息: /nai ref类型 both] 收到"),
    ]
    for cmd_name, message in quoted_command_messages:
        regex = _compile(patterns[cmd_name])
        assert regex.match(message) is None, f"{cmd_name} 不应被引用内容触发: {message!r}"
        assert _matching_command_names(patterns, message) == []


def test_current_command_after_quoted_command_still_matches() -> None:
    """引用内容可以包含历史命令；只要当前正文另有命令，仍应按正文命令匹配。"""
    patterns = _load_command_patterns()
    message = "[回复 alice 的消息: /nai 反推] /nai i2i 改成森林背景"
    regex = _compile(patterns["nai_i2i_command"])
    match = regex.match(message)
    assert match is not None
    assert match.group("description") == "改成森林背景"
    assert _matching_command_names(patterns, message) == ["nai_i2i_command"]


def test_vibe_ref_clear_patterns_match_keyword_only() -> None:
    """/nai vibe清空 / /nai ref清空 是纯关键字命令，不接参数。"""
    patterns = _load_command_patterns()
    for cmd_name, sample in [
        ("nai_vibe_clear_command", "/nai vibe清空"),
        ("nai_ref_clear_command", "/nai ref清空"),
    ]:
        regex = _compile(patterns[cmd_name])
        assert regex.match(sample) is not None, f"{cmd_name} 应匹配 {sample!r}"
    # 接了多余参数应不匹配（防止误吞）
    assert _compile(patterns["nai_vibe_clear_command"]).match("/nai vibe清空 角色A") is None


def test_nai0_vibe_ref_patterns_capture_tags_and_at_names() -> None:
    """/nai0 vibe / /nai0 ref 走"不过 LLM 英文 tag"路径，支持 @<名字>... 单次覆盖。

    pattern 与 /nai vibe / /nai ref 同结构，仅 tags 段直接当 prompt 而非 LLM 翻译。
    """
    patterns = _load_command_patterns()
    cases = [
        ("nai_0_vibe_command", "/nai0 vibe 1girl, blue sky", "", "1girl, blue sky"),
        (
            "nai_0_vibe_command",
            "/nai0 vibe @角色A @角色B 1girl, looking at viewer",
            "@角色A @角色B ",
            "1girl, looking at viewer",
        ),
        ("nai_0_ref_command", "/nai0 ref 1girl, smile", "", "1girl, smile"),
        (
            "nai_0_ref_command",
            "/nai0 ref @角色A 1girl, standing",
            "@角色A ",
            "1girl, standing",
        ),
    ]
    for cmd_name, sample, expected_at, expected_tags in cases:
        regex = _compile(patterns[cmd_name])
        match = regex.match(sample)
        assert match is not None, f"{cmd_name} 应匹配 {sample!r}"
        assert match.group("at_names") == expected_at
        assert match.group("tags") == expected_tags


def test_generic_nai0_pattern_skips_vibe_and_ref_subcommands() -> None:
    """通用 /nai0 命令的负向预查应跳过 /nai0 vibe / /nai0 ref，否则会把 ``vibe`` 当成 tag。

    与 /nai 同结构：vibe / ref 后用 CJK 边界覆盖，避免 latin→CJK 边界让 ``\\b`` 不成立。
    """
    patterns = _load_command_patterns()
    nai0 = _compile(patterns["nai_0_draw"])
    skip_samples = [
        "/nai0 vibe 1girl, blue sky",
        "/nai0 vibe @角色A 1girl",
        "/nai0 ref 1girl, smile",
        "/nai0 ref @角色A 1girl",
    ]
    for sample in skip_samples:
        assert nai0.match(sample) is None, f"通用 /nai0 不应吞掉 {sample!r}"
    # 但正常 /nai0 <英文 tags> 仍需匹配
    assert nai0.match("/nai0 1girl, hatsune miku, smile") is not None
