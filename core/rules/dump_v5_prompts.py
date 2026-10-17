# -*- coding: utf-8 -*-
"""把 V5 档的两份提示词渲染成一份可整段取用的文档。

存在的理由：``docs/v5_prompts.md`` 要给人整段复制（贴进 config 的
``llm_prompt_template`` / ``prompt_template``，或贴进别的工具），而提示词本体分散在
``caption_rules`` 与 ``prompt_rules`` 两个模块里。手抄一份进 markdown 必然会和源码漂移，
所以文档由本模块渲染，配套测试锁住"仓库里那份 == 本模块现在渲染出来的"。

用法::

    python3 -m plugins.nai_draw_plugin.core.rules.dump_v5_prompts > plugins/nai_draw_plugin/docs/v5_prompts.md
"""

from __future__ import annotations

from pathlib import Path

from .caption_rules import resolve_caption_prompt
from .prompt_rules import resolve_prompt_generator_template


# 文档固定取 NSFW 解锁档：SFW 档只差开头的 system_override 一块，正文逐字相同，
# 两份都贴进来只会让这份本就很长的文档翻倍。
_DOCUMENT_SFW = False

# 翻译侧固定取 text 通道：它输出的就是一串可直接用的提示词；json 通道输出的是待程序
# 渲染的中间结构，脱离插件没法用。
_DOCUMENT_CHANNEL = "text"

DOCUMENT_PATH = Path(__file__).resolve().parents[2] / "docs" / "v5_prompts.md"


_HEADER = '''# NovelAI V5 提示词全文（反推 + 翻译）

给 **V5 自然语言档** 用的两份完整提示词，整理在一起方便整段取用。两份都是插件运行时真正发出去
的原文，不是节选或改写。

| 部分 | 干什么 | 源 | 对应配置项 |
|------|--------|----|-----------|
| 一、反推 | 图 → 穷尽的英文自然语言描述 | `core/rules/caption_rules.py` | `[retag].llm_prompt_template` |
| 二、翻译 | 中文需求 → V5 提示词（Danbooru tag + 自然语言混写） | `core/rules/prompt_rules.py` | `[prompt_generator].prompt_template` |

两份共有的一条规则：**画面里的文字一律用英文半角双引号 `"` 包起来**。反推侧从图里读出招牌上的
字并用半角引号包住，翻译侧把「招牌上写 OPEN」翻成 `a sign that reads "OPEN"`——两端用同一种引号，
「反推 → 回灌 `/nai0`」的往返才不会在引号形态上掉链子。弯引号 `“ ”` 和全角 `＂` 在下游只是普通
字符，画出来会糊成随机字形。

本文件由代码生成，改提示词请改上表里的源文件，不要直接改这里。重新生成：

```bash
python3 -m plugins.nai_draw_plugin.core.rules.dump_v5_prompts > plugins/nai_draw_plugin/docs/v5_prompts.md
```

> 下面两段都取 **NSFW 解锁档**（`sfw=False`）。SFW 档只有开头的 `<system_override>` 一块不同，
> 其余正文逐字相同；反推侧 SFW 版把"解锁直白描述"换成"改写成安全版但必须照常输出完整描述"，
> 翻译侧 SFW 版把 `<nsfw_capability>` 换成 `<sfw_restriction>`。

---

## 一、反推提示词（图 → 英文自然语言描述）

单张图连同这段提示词一起发给视觉模型，模型返回一段纯英文自然语言描述。输出没有长度上限，
长度由画面里的 item 数推出来，复杂立绘常达 600~900 词。

```text
'''


_MIDDLE = '''```

---

## 二、翻译提示词（中文需求 → V5 提示词）

下面这段里的 `<<...>>` 是插件在运行时替换的占位符，手工使用时请自行替换或整段删掉：

| 占位符 | 替换成什么 |
|--------|-----------|
| `<<CUSTOM_SYSTEM_PROMPT>>` | `[custom_prompt].system_prompt` 的内容，没有就留空 |
| `<<TAG_CANDIDATES>>` | Danbooru 检索出的候选 tag 清单，没有就留空 |
| `<<PREVIOUS_PROMPT>>` | 上一轮提示词（继承路径用），没有就留空 |
| `<<REASONING_CONTEXT>>` | Planner 的 reasoning，命令链路留空 |
| `<<USER_REQUEST>>` | **用户的中文需求，必填** |
| `<<CURRENT_TIME_CONTEXT>>` | 当前时间描述，没有就留空 |
| `<<SELFIE_HINT>>` / `<<SELFIE_SCENE_CONTEXT>>` | Bot 出镜时的外貌锚点与场景上下文，没有就留空 |

```text
'''


_FOOTER = "```\n"


def render_document() -> str:
    """渲染整份 V5 提示词文档。"""
    caption = resolve_caption_prompt(sfw=_DOCUMENT_SFW).rstrip()
    translate = resolve_prompt_generator_template(
        channel=_DOCUMENT_CHANNEL,
        sfw=_DOCUMENT_SFW,
        natural_language=True,
    ).rstrip()
    return f"{_HEADER}{caption}\n{_MIDDLE}{translate}\n{_FOOTER}"


if __name__ == "__main__":
    print(render_document(), end="")
