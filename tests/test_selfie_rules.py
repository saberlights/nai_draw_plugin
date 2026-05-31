import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.rules.selfie_rules import (
    detect_bot_self_image_intent,
    detect_explicit_image_request,
    detect_negative_image_intent,
    detect_selfie_from_output,
    merge_selfie_prompt,
)


def test_merge_selfie_prompt_appends_selfie_tags_after_main_prompt() -> None:
    merged = merge_selfie_prompt(
        "solo, 1girl, selfie, smile, bedroom",
        "young woman, pink eyes",
    )

    assert merged == "solo, 1girl, selfie, smile, bedroom, young woman, pink eyes"


def test_merge_selfie_prompt_removes_conflicting_hair_and_eye_tags_before_appending() -> None:
    merged = merge_selfie_prompt(
        "solo, 1girl, selfie, black hair, blue eyes, smile",
        "light blue hair, pink eyes",
    )

    assert merged == "solo, 1girl, selfie, smile, light blue hair, pink eyes"


# ============ detect_explicit_image_request ============

def test_explicit_request_direct_draw_words_match() -> None:
    """用户原话含画图/出图类强信号 → True"""
    assert detect_explicit_image_request("来一张萝莉") is True
    assert detect_explicit_image_request("给我整一只小狗") is True
    assert detect_explicit_image_request("帮我画一张赛博朋克城市") is True
    assert detect_explicit_image_request("发图") is True
    assert detect_explicit_image_request("再来一张") is True


def test_explicit_request_selfie_and_portrait_words_match() -> None:
    """自拍/肖像类关键词 → True"""
    assert detect_explicit_image_request("发张自拍") is True
    assert detect_explicit_image_request("镜子里拍一张") is True
    assert detect_explicit_image_request("想要肖像") is True
    assert detect_explicit_image_request("看你的生活照") is True


def test_explicit_request_see_bot_self_words_match() -> None:
    """想看 bot 本人样子的请求 → True"""
    assert detect_explicit_image_request("想看你妹妹") is True
    assert detect_explicit_image_request("看看你今天穿了什么") is True
    assert detect_explicit_image_request("你长什么样") is True
    assert detect_explicit_image_request("看看黑丝") is True


def test_explicit_request_continuation_words_match() -> None:
    """连续追图类关键词 → True"""
    assert detect_explicit_image_request("换个角度") is True
    assert detect_explicit_image_request("这身衣服全身看看") is True
    assert detect_explicit_image_request("同一套再来一张") is True


def test_explicit_request_casual_chat_does_not_match() -> None:
    """日常闲聊、知识问答、评价类不命中 → False（此时由 Planner 决定是否主动发图，进 proactive 档）"""
    assert detect_explicit_image_request("今天天气真不错") is False
    assert detect_explicit_image_request("Python 字典怎么用") is False
    assert detect_explicit_image_request("你觉得这个电影怎么样") is False
    assert detect_explicit_image_request("刚才那张图不错") is False
    assert detect_explicit_image_request("") is False


# ============ detect_negative_image_intent ============

def test_negative_intent_blocks_explicit_refusals() -> None:
    """用户明确拒绝出图 → True（即使 Planner 调了 Action 也应拦截）"""
    assert detect_negative_image_intent("不要画") is True
    assert detect_negative_image_intent("别给我画图") is True
    assert detect_negative_image_intent("文字回复就行") is True
    assert detect_negative_image_intent("不用配图") is True


def test_negative_intent_does_not_block_normal_requests() -> None:
    """正常请求/闲聊不应命中否定意图"""
    assert detect_negative_image_intent("想看你") is False
    assert detect_negative_image_intent("画一张萝莉") is False
    assert detect_negative_image_intent("今天天气真好") is False
    assert detect_negative_image_intent("") is False


# ============ detect_bot_self_image_intent ============

def test_bot_self_image_intent_explicit_selfie_or_portrait_match() -> None:
    """用户原话含自拍/肖像等显式关键词 → True"""
    assert detect_bot_self_image_intent("自拍") is True
    assert detect_bot_self_image_intent("发张自拍") is True
    assert detect_bot_self_image_intent("镜子里来一张") is True
    assert detect_bot_self_image_intent("肖像照") is True
    assert detect_bot_self_image_intent("来张生活照") is True
    assert detect_bot_self_image_intent("生活照") is True


def test_bot_self_image_intent_implicit_see_bot_match() -> None:
    """隐式"想看 bot 本人"表达 → True"""
    assert detect_bot_self_image_intent("看看你今天穿了什么") is True
    assert detect_bot_self_image_intent("你长什么样") is True
    assert detect_bot_self_image_intent("看看黑丝") is True
    assert detect_bot_self_image_intent("你的腿") is True
    assert detect_bot_self_image_intent("拍给我看") is True


def test_bot_self_image_intent_named_character_does_not_match() -> None:
    """回归测试：用户指定二次元角色的请求不应触发 bot 自拍后处理。

    /nai 中野二乃，展示身材 等点名二创角色的请求，必须返回 False，
    否则 _process_selfie_prompt 会把 bot 默认外貌叠加进去，把角色洗成 bot 自己。
    """
    assert detect_bot_self_image_intent("中野二乃，展示身材") is False
    assert detect_bot_self_image_intent("画一张初音未来") is False
    assert detect_bot_self_image_intent("蕾姆，女仆装") is False
    assert detect_bot_self_image_intent("芙兰朵露，红裙") is False


def test_bot_self_image_intent_neutral_descriptions_do_not_match() -> None:
    """场景/物品/原创人物描述不应命中。"""
    assert detect_bot_self_image_intent("一只猫躺在窗台") is False
    assert detect_bot_self_image_intent("赛博朋克城市夜景") is False
    assert detect_bot_self_image_intent("一个女孩在雨中") is False
    assert detect_bot_self_image_intent("") is False


def test_bot_self_image_intent_planner_composed_text_for_named_character() -> None:
    """Action 链路真实样本：Planner 把"画一张初音未来"拆成 5 字段 + description，
    compose_description_from_action_payload 拼出的整段不应命中 bot 自拍意图。

    历史 bug：handle_action 之前用 detect_selfie_from_output(LLM 翻译后 prompt)
    判 selfie，会被 LLM 用作 framing 的 portrait/full body portrait 误命中，把
    用户点名的二次元角色洗成 bot 自拍。修复后改用 detect_bot_self_image_intent
    判定 raw_description，本用例守护"初音未来 + 第三视角 + 特写"不会被误判。
    """
    raw_description = (
        "一女, 初音未来, 公式服, 葱色双马尾, 精致的面容, 灵动的眼神, "
        "背景为舞台, 特写 一女 第三视角 站立 俏皮 微笑 特写"
    )
    assert detect_bot_self_image_intent(raw_description) is False


def test_selfie_from_output_misfires_on_framing_words_documented_behavior() -> None:
    """文档化 detect_selfie_from_output 的已知误判：framing 词 ``portrait photo`` /
    ``full body portrait`` 即使非 bot 自拍意图也会触发 True——这正是当前 Action 链路
    不再依赖它判 selfie 的原因。

    本用例仅锁定该行为，避免未来"治好它"再回归——真正的判定要走
    detect_bot_self_image_intent(用户原话/raw_description) 这条治根路径。
    """
    # LLM 翻译"初音未来，全身像"时常输出含 full body portrait 的 framing
    assert detect_selfie_from_output("1girl, hatsune miku, full body portrait, smile") is True
    # 哪怕主体是命名二次元角色（非 bot），detect_selfie_from_output 仍会误命中
    assert detect_selfie_from_output("1girl, nakano nino, portrait photo") is True
