# -*- coding: utf-8 -*-
"""自然搭图触发模块：从 bot reply 文本判断本轮是否值得跟一张图。

核心思路：bot 的 reply 已经写好（来自 `maisaka.replyer.after_response` hook），
插件读一眼。如果 reply 里 bot 在描述自身视觉状态、所处场景，或处在情感互动节点，
就背地里跟一张图，让用户感到"她说的这一刻顺便发了张照片"。

判分仅基于关键词：可控、可测、可调；不调 LLM，避免 hook 阻塞。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# ==================== 触发词库 ====================

# 强信号：bot 自指视觉细节（穿/坐/躺/动作/位置/即时状态）。命中即偏 selfie。
SELF_VISUAL_KEYWORDS: tuple[str, ...] = (
    # 服装 / 着装动作
    "我穿", "我换上", "穿着", "穿了", "脱下", "脱了", "换上", "披着", "戴着", "戴了",
    # 姿态 / 动作
    "我坐", "我躺", "我靠", "我趴", "我蹲", "我站", "我抱", "我抬", "我伸", "我抓",
    "我笑", "我哭", "我累", "我饿", "我困", "我冷", "我热",
    # 即时状态
    "我刚", "我现在", "我在", "我正", "我刚刚",
    "刚洗完", "刚洗澡", "刚出浴", "刚起床", "刚醒", "刚到家", "刚回来", "刚吃完",
    "刚做完", "刚下班", "刚下课", "刚化完妆", "刚换好",
    # 身体局部自指（克制：避免误命中知识科普）
    "我的头发", "我的发型", "我的腿", "我的脚", "我的手", "我的脸", "我的眼睛",
)

# 中信号：情感节点 / 亲密互动。命中偏 portrait 或场景。
EMOTIONAL_BEAT_KEYWORDS: tuple[str, ...] = (
    "晚安", "早安", "早上好", "想你了", "好想你", "想见你", "抱抱", "亲亲", "么么哒",
    "回来了", "到家了", "回家了", "我回来啦", "我到啦",
    "吃饱了", "吃完啦", "睡了", "睡觉啦", "困了", "累了一天",
    "在想你", "陪我", "陪你", "等你", "等等我",
)

# 场景 / 地点 / 活动信号。命中偏 portrait 生活照 / 场景图。
SCENE_KEYWORDS: tuple[str, ...] = (
    "窗边", "阳台", "床上", "沙发", "厨房", "浴室", "书桌", "桌前",
    "咖啡店", "便利店", "超市", "餐厅", "公园", "地铁", "公交", "学校", "教室", "图书馆",
    "路上", "回家路上", "下班路上", "便道", "海边", "山上",
    "外面下雨", "下雪了", "天黑了", "天亮了", "夕阳",
)

# 强负向：bot 在做理性 / 工具性回答。命中直接放弃。
DISQUALIFYING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"```"),                                                       # 代码块
    re.compile(r"^\s*\d+[.)、]\s", flags=re.MULTILINE),                        # 行首 "1. " / "1) " / "1、"
    re.compile(r"^\s*[-•·]\s", flags=re.MULTILINE),                            # 列表项
    re.compile(r"(?:首先|其次|另外|最后|总结一下|综上|换句话说)"),
    re.compile(r"(?:函数|参数|变量|API|接口|算法|数据库|sql|json|http)", re.IGNORECASE),
)

# 弱负向：bot 在询问 / 不确定 / 解释。扣分但不直接淘汰。
SOFT_NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "你觉得呢", "你怎么看", "我不太懂", "我不太清楚", "我想想",
    "这个我", "这个怎么", "为什么", "怎么办",
)


@dataclass(frozen=True)
class ReplyDrawSignal:
    """评分结果。score 范围 [0, 1]，mode 决定后续 description 走向。"""

    score: float
    mode: str  # "selfie" | "portrait" | "scene" | ""
    hits: tuple[str, ...]

    @property
    def should_draw(self) -> bool:
        return self.score > 0.0 and self.mode != ""


def _count_hits(text: str, keywords: Iterable[str]) -> list[str]:
    """统计命中的关键词。返回原词列表，便于日志与单测断言。"""
    hits: list[str] = []
    for keyword in keywords:
        if keyword and keyword in text:
            hits.append(keyword)
    return hits


def score_reply_for_auto_draw(reply_text: str) -> ReplyDrawSignal:
    """给 bot 的一条 reply 打分，决定是否、按什么模式跟图。

    评分规则（设计目标：保守，宁可不出也别错出）：

    - 强自指（"我穿/我刚/刚洗完"等）单条命中即 +0.5，多条加成更明显，mode=selfie。
    - 情感节点（"晚安/想你了"）单条 +0.3，mode=portrait（贴近"近照"）。
    - 场景词（"窗边/便利店"）单条 +0.3，单纯场景命中走 mode=scene；与自指叠加时仍走 selfie。
    - 强负向（代码块、列点、技术词）直接清零，无论其他信号多强。
    - 弱负向（"你觉得呢/我不太懂"）一处扣 0.15。
    - 句长 < 6 字符或纯标点：直接 0 分（短促回应不适合配图）。
    """
    text = (reply_text or "").strip()
    if not text:
        return ReplyDrawSignal(score=0.0, mode="", hits=())

    # 长度 / 内容快速淘汰
    if len(text) < 6:
        return ReplyDrawSignal(score=0.0, mode="", hits=())
    if not re.search(r"[一-鿿A-Za-z]", text):
        return ReplyDrawSignal(score=0.0, mode="", hits=())

    # 强负向直接出局
    for pattern in DISQUALIFYING_PATTERNS:
        if pattern.search(text):
            return ReplyDrawSignal(score=0.0, mode="", hits=())

    self_hits = _count_hits(text, SELF_VISUAL_KEYWORDS)
    emo_hits = _count_hits(text, EMOTIONAL_BEAT_KEYWORDS)
    scene_hits = _count_hits(text, SCENE_KEYWORDS)
    soft_neg_hits = _count_hits(text, SOFT_NEGATIVE_KEYWORDS)

    score = 0.0
    if self_hits:
        # 首条 +0.5，后续每条 +0.15，封顶 +0.8
        score += min(0.5 + 0.15 * (len(self_hits) - 1), 0.8)
    if emo_hits:
        score += min(0.3 + 0.1 * (len(emo_hits) - 1), 0.5)
    if scene_hits:
        score += min(0.3 + 0.1 * (len(scene_hits) - 1), 0.5)

    # 弱负向：每条 -0.15
    score -= 0.15 * len(soft_neg_hits)

    # mode 选择：自指优先；否则情感 → portrait；否则场景 → scene
    if self_hits:
        mode = "selfie"
    elif emo_hits:
        mode = "portrait"
    elif scene_hits:
        mode = "scene"
    else:
        mode = ""

    # 收敛到 [0, 1]
    score = max(0.0, min(score, 1.0))
    if mode == "":
        score = 0.0

    hits = tuple(self_hits + emo_hits + scene_hits)
    return ReplyDrawSignal(score=score, mode=mode, hits=hits)


def compose_description_from_reply(
    reply_text: str,
    signal: ReplyDrawSignal,
) -> str:
    """根据评分结果与 reply 文本，拼一段直接喂给生图流程的 description。

    保持简短：只保证 mode 标签 + 关键场景词；后续 `_generate_prompt_with_llm`
    还会基于 description 做完整 prompt 生成，这里不必把视觉细节写满。
    """
    if not signal.should_draw:
        return ""

    pieces: list[str] = ["一女"]
    if signal.mode == "selfie":
        pieces.append("自拍 近景")
    elif signal.mode == "portrait":
        pieces.append("肖像照 近景")
    elif signal.mode == "scene":
        pieces.append("生活照")

    # 把 hit 里出现的"明确视觉名词/场景词"挑出来塞进去，避免噪音长串
    visible_hints = [
        h for h in signal.hits
        # 跳过纯动作短语，保留场景/状态短语
        if any(h.startswith(prefix) for prefix in ("窗边", "阳台", "床上", "沙发", "厨房", "浴室",
                                                    "咖啡店", "便利店", "超市", "餐厅", "公园",
                                                    "海边", "山上", "夕阳"))
        or h in ("晚安", "回家了", "到家了", "下雪了", "天黑了", "刚洗完", "刚起床")
    ]
    if visible_hints:
        pieces.append(" ".join(dict.fromkeys(visible_hints)))  # 去重保序

    return " ".join(pieces).strip()
