# -*- coding: utf-8 -*-
"""锁住 `docs/v5_prompts.md` 与提示词源码不漂移。

这份文档是给人整段复制走的（贴进 config 的 `llm_prompt_template` / `prompt_template`，
或贴进别的工具）。一旦有人改了 `caption_rules` / `prompt_rules` 却没重新生成文档，
复制走的就是一份过期提示词，而且不会有任何报错——只会在出图效果上莫名其妙地对不上。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.rules.dump_v5_prompts import (
    DOCUMENT_PATH,
    render_document,
)


def test_committed_document_matches_current_prompt_sources() -> None:
    assert DOCUMENT_PATH.is_file(), f"文档不存在：{DOCUMENT_PATH}"

    committed = DOCUMENT_PATH.read_text(encoding="utf-8")
    assert committed == render_document(), (
        "docs/v5_prompts.md 与提示词源码已漂移。重新生成：\n"
        "  python3 -m plugins.nai_draw_plugin.core.rules.dump_v5_prompts "
        "> plugins/nai_draw_plugin/docs/v5_prompts.md"
    )


def test_document_contains_both_parts_in_order() -> None:
    text = render_document()

    idx_caption = text.find("## 一、反推提示词")
    idx_translate = text.find("## 二、翻译提示词")
    assert idx_caption != -1 and idx_translate != -1
    assert idx_caption < idx_translate


def test_document_carries_the_shared_half_width_quote_rule() -> None:
    """两部分共有的那条规则必须在文档开头讲清楚，否则读者只会看到两段各自的英文条款。"""
    import re

    text = render_document()
    # 提示词正文按 100 列硬换行，断言只锁短语不锁排版
    flat = re.sub(r"\s+", " ", text)

    assert "画面里的文字一律用英文半角双引号" in text
    # 两侧各自的规则原文也都要真的在文档里
    assert "wrapped in straight ASCII double quotes" in flat
    assert "那串字必须用英文半角双引号" in flat


def test_document_embeds_real_prompt_bodies_not_summaries() -> None:
    """文档里放的必须是提示词原文；退化成"摘要 + 链接"就失去了整段取用的意义。"""
    text = render_document()

    # 反推侧的 item 铁律与递归条款
    assert "A decoration attached to an item is itself an item" in text
    assert "Repeated decorations are counted, never generalised" in text
    # 翻译侧的 V5 自然语言块与互动绑定铁律
    assert "## 自然语言描述（当前模型是 NovelAI V5）" in text
    assert "两侧必须是同一个 Danbooru tag，只有前缀不同" in text


def test_document_explains_every_placeholder_it_ships() -> None:
    """翻译段带 `<<...>>` 占位符，手工使用时不解释清楚就是一堆没法替换的乱码。"""
    import re

    text = render_document()
    placeholders = set(re.findall(r"<<[A-Z_]+>>", text))
    assert placeholders, "翻译段应当带占位符"

    # 占位符说明表用 `<<NAME>>` 形式列出；逐个确认表里都有
    for placeholder in placeholders:
        assert text.count(placeholder) >= 2, f"占位符 {placeholder} 没有在说明表里出现"


def test_document_states_it_is_generated() -> None:
    """没有这句话，下一个人会直接改 markdown，然后被测试打回来还不知道为什么。"""
    text = render_document()

    assert "本文件由代码生成" in text
    assert "python3 -m plugins.nai_draw_plugin.core.rules.dump_v5_prompts" in text
