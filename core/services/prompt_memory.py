# -*- coding: utf-8 -*-
"""
提示词记忆 - 渲染层

将 session_state 中保存的"上一轮 LLM 正向提示词"渲染成 LLM 模板可消费的
<previous_prompt_context> 块；附带三档继承规则（微调 / 换角色保场景 / 全新主题）。

注意：
- 这里只负责渲染。运行时存储完全由 session_state.last_nai_context 承担。
- 不存储任何状态、不读取任何持久化数据源。
"""

from __future__ import annotations

from typing import Optional


def render_previous_prompt_block(
    last_prompt: Optional[str],
    last_request: Optional[str] = None,
) -> str:
    """Generate the replacement content for the <<PREVIOUS_PROMPT>> placeholder.

    Returns an XML block with three-tier inheritance rules when *last_prompt*
    is present, or a placeholder block when there is no previous prompt.
    """
    previous = (last_prompt or "").strip()
    if not previous:
        return (
            "<previous_prompt_context>\n"
            "（无上一轮提示词，请完全按照本次用户请求生成全新提示词）\n"
            "</previous_prompt_context>"
        )

    # 可选注入上一轮用户请求（帮助 LLM 做 diff 推理）
    request_section = ""
    req = (last_request or "").strip()
    if req:
        request_section = f"\n【上一轮用户请求】\n{req}\n"

    parts = [
        "<previous_prompt_context>\n",
        "【上一轮 LLM 生成的提示词（系统注入，非用户输入的英文tag）】\n",
        previous, "\n",
        request_section, "\n",
        "【三档继承规则（必须遵守）】\n",
        "请对比本次用户请求与上一轮提示词，判断属于以下哪档：\n\n",
        'A. 微调（同一主题的细节调整，如「把背景换成夜晚」、「加个帽子」）\n',
        "   → 以上方提示词为底稿，仅修改用户要求变更的部分，保留其余标签\n\n",
        'B. 换角色保场景（角色变了但场景/构图延续，如「换成另一个角色」、「画成男生版」）\n',
        "   → 保留场景、构图、氛围等环境标签，替换角色相关标签（外貌、服装、身份等）\n\n",
        "C. 全新主题（与上一轮完全无关的新请求）\n",
        "   → 完全忽略上方提示词，按用户请求重新生成\n",
        "</previous_prompt_context>",
    ]
    return "".join(parts)
