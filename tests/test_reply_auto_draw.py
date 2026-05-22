"""reply 后置自动跟图评分模块单测。"""

import sys
from pathlib import Path

import pytest

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.rules.reply_auto_draw import (
    ReplyDrawSignal,
    compose_description_from_reply,
    score_reply_for_auto_draw,
)


# ==================== 强信号：自指视觉 ====================


def test_score_self_visual_triggers_selfie_mode() -> None:
    signal = score_reply_for_auto_draw("我刚洗完澡靠在窗边发呆，有点累")
    assert signal.mode == "selfie"
    assert signal.score >= 0.6
    assert signal.should_draw


def test_score_multiple_self_hits_boost() -> None:
    one = score_reply_for_auto_draw("我穿了新裙子")
    two = score_reply_for_auto_draw("我刚换上新裙子，我现在坐在沙发上")
    assert two.score > one.score
    assert two.mode == "selfie"


# ==================== 中信号：情感节点 ====================


def test_score_emotional_beat_triggers_portrait() -> None:
    signal = score_reply_for_auto_draw("晚安，今天累了一天，我先去睡了")
    assert signal.mode in {"portrait", "selfie"}  # "睡了"+"晚安"或与自指叠加
    assert signal.score >= 0.3


def test_score_pure_greeting_does_not_pass_threshold() -> None:
    # 太短，淘汰
    signal = score_reply_for_auto_draw("嗯")
    assert signal.score == 0.0
    assert not signal.should_draw


# ==================== 场景词 ====================


def test_score_scene_only_triggers_scene_mode() -> None:
    signal = score_reply_for_auto_draw("现在在咖啡店写写东西，外面下雨了")
    assert signal.mode == "scene"
    assert signal.score > 0.0


# ==================== 负向：技术/列点直接 0 分 ====================


def test_score_code_block_disqualifies() -> None:
    text = "你看这段代码：```\nprint('hello')\n```，我刚试过没问题"
    signal = score_reply_for_auto_draw(text)
    assert signal.score == 0.0
    assert signal.mode == ""


def test_score_numbered_list_disqualifies() -> None:
    text = "我刚洗完澡，但解释一下：\n1. 第一点\n2. 第二点"
    signal = score_reply_for_auto_draw(text)
    assert signal.score == 0.0


def test_score_technical_terms_disqualifies() -> None:
    signal = score_reply_for_auto_draw("我穿好衣服了，但这个 API 接口需要传 json 参数")
    assert signal.score == 0.0  # 出现 API 直接淘汰，无视前面的"我穿"


# ==================== 弱负向只扣分不归零 ====================


def test_score_soft_negative_subtracts() -> None:
    base = score_reply_for_auto_draw("我刚洗完澡，靠在窗边")
    weak = score_reply_for_auto_draw("我刚洗完澡，靠在窗边，你觉得呢")
    assert weak.score < base.score
    assert weak.mode == base.mode


# ==================== compose_description_from_reply ====================


def test_compose_description_selfie() -> None:
    signal = ReplyDrawSignal(score=0.7, mode="selfie", hits=("刚洗完",))
    desc = compose_description_from_reply("我刚洗完澡靠窗", signal)
    assert "一女" in desc
    assert "自拍 近景" in desc
    assert "刚洗完" in desc


def test_compose_description_portrait_drops_action_hits() -> None:
    signal = ReplyDrawSignal(score=0.4, mode="portrait", hits=("晚安",))
    desc = compose_description_from_reply("晚安啦", signal)
    assert "一女" in desc
    assert "肖像照" in desc
    assert "晚安" in desc  # 情感词允许保留


def test_compose_description_scene_includes_location() -> None:
    signal = ReplyDrawSignal(score=0.3, mode="scene", hits=("咖啡店",))
    desc = compose_description_from_reply("在咖啡店写东西", signal)
    assert "一女" in desc
    assert "生活照" in desc
    assert "咖啡店" in desc


def test_compose_description_returns_empty_when_should_not_draw() -> None:
    signal = ReplyDrawSignal(score=0.0, mode="", hits=())
    assert compose_description_from_reply("任何文本", signal) == ""


# ==================== 边界 ====================


@pytest.mark.parametrize(
    "text",
    ["", "   ", ".....", "👍👍", "啊"],
)
def test_score_empty_or_trivial(text: str) -> None:
    assert score_reply_for_auto_draw(text).score == 0.0
