"""V5 自然语言档的规则发放与落地。

V5 对英文自然语言从句的理解强于 V4.5，但这条放宽只能作用在 V5：给 V4.5 及更早
灌长句只会稀释每个 tag 的权重。所以这里同时钉死两件事——
1. 规则按模型分档发放，V5 拿到放行文本、V4.5 拿到禁止文本，两档不许互相污染；
2. 放宽只影响"能不能写句子"，rating 首 tag、互动绑定、权重语法这些程序会硬校验的
   协议在 V5 档一条都不能少，而且自然语言真的能活着走到最终提示词里。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.rules.prompt_rules import (  # noqa: E402
    PROMPT_CHANNELS,
    resolve_prompt_generator_template,
    resolve_visual_continuity_output_instruction,
)
from plugins.nai_draw_plugin.core.services.prompt_generation_workflow import (  # noqa: E402
    PromptGenerationWorkflow,
)
from plugins.nai_draw_plugin.core.services.session_state import session_state  # noqa: E402
from plugins.nai_draw_plugin.core.utils.prompt_output_parser import (  # noqa: E402
    parse_prompt_from_structured_output,
)
from plugins.nai_draw_plugin.core.utils.prompt_postprocessor import (  # noqa: E402
    normalize_prompt_order,
    strip_cjk_and_fullwidth,
)


V5_MARKERS = (
    "你是 NovelAI V5 提示词专家",
    "## 自然语言描述（当前模型是 NovelAI V5）",
    "在标准 Danbooru tag 的基础上，尽可能用自然语言把 tag 说不清的部分说清楚",
    "骨架仍然是 Danbooru tag",
    "### 例 7：V5 自然语言补充",
)

TAG_ONLY_MARKERS = (
    "你是 NovelAI 4/4.5 提示词专家",
    "## 自然语言短句（NAI 4/4.5 兜底）",
    "JSON 模式严格禁止自然语言",
)


def _template(channel: str, *, sfw: bool, natural_language: bool) -> str:
    return resolve_prompt_generator_template(
        channel=channel, sfw=sfw, natural_language=natural_language
    )


def test_每个通道的_v5_档都发放自然语言规则() -> None:
    for channel in PROMPT_CHANNELS:
        for sfw in (True, False):
            template = _template(channel, sfw=sfw, natural_language=True)
            for marker in V5_MARKERS:
                assert marker in template, f"{channel}/sfw={sfw} 的 V5 档丢失：{marker}"


def test_v45_及更早仍然拿到纯_tag_档() -> None:
    """放宽不许泄漏到老模型：V4.5 灌长句会稀释 tag 权重，出图直接变糊。"""

    for channel in PROMPT_CHANNELS:
        for sfw in (True, False):
            template = _template(channel, sfw=sfw, natural_language=False)
            for marker in TAG_ONLY_MARKERS:
                assert marker in template, f"{channel}/sfw={sfw} 的纯 tag 档丢失：{marker}"
            for marker in V5_MARKERS:
                assert marker not in template, (
                    f"{channel}/sfw={sfw} 的纯 tag 档混入了 V5 规则：{marker}"
                )


def test_两档规则互斥不共存() -> None:
    """两套自然语言尺度同时出现，LLM 的取舍就不可控了。"""

    for channel in PROMPT_CHANNELS:
        for sfw in (True, False):
            for natural_language in (True, False):
                template = _template(
                    channel, sfw=sfw, natural_language=natural_language
                )
                assert template.count("</natural_language>") == 1, (
                    f"{channel}/sfw={sfw}/nl={natural_language} 的自然语言块缺失或重复"
                )


def test_json_通道的元素契约随档位切换() -> None:
    """JSON 数组元素规则是自然语言真正落地的地方，两档必须给出相反的指示。"""

    for sfw in (True, False):
        tag_only = _template("json", sfw=sfw, natural_language=False)
        assert "禁止输出自然语言句子，所有内容必须可拆为 tag 或权重表达" in tag_only

        natural = _template("json", sfw=sfw, natural_language=True)
        assert "禁止输出自然语言句子" not in natural, "V5 档仍在禁止自然语言，规则自相矛盾"
        assert "或一句自然语言，禁止内部含逗号" in natural
        # 元素内含逗号会被下游按 tag 边界切开，句子必须靠 and / with / 从句断句
        assert "句子需要停顿时改用 `and` / `with` / 从句" in natural


def test_连续性输出契约随档位切换且稳定区有额外约束() -> None:
    """稳定区 Tag 会被逐字缓存复用，句子写错档会连续污染后续每一张图。"""

    tag_only = resolve_visual_continuity_output_instruction(False)
    assert "所有数组元素只能是最终英文 Danbooru tag 或单个权重表达，禁止自然语言句子。" in tag_only

    natural = resolve_visual_continuity_output_instruction(True)
    assert "禁止自然语言句子" not in natural
    assert "或一句英文自然语言短句" in natural
    assert "稳定区的自然语言会被逐字缓存并在后续图片复用" in natural


def test_v5_档不得放宽程序会硬校验的协议() -> None:
    """自然语言只加在 tag 之上；写成句子就配不上对家的绑定必须仍被禁止。"""

    for channel in PROMPT_CHANNELS:
        template = _template(channel, sfw=False, natural_language=True)
        # 绑定：程序按逐字同名校验，句子形态永远配不上
        assert "两侧必须是同一个 Danbooru tag，只有前缀不同" in template
        assert "写成句子配不上" in template
        # rating / 人数 / framing 仍然只认 tag
        assert "首个 rating tag、人数 tag、framing 与视角 tag：NAI 按 tag 识别，写进句子不生效" in template
        # 权重块只能包单 tag
        assert "禁止把整句包进权重块" in template
        # 中文会被程序剥掉，句子必须是英文
        assert "必须是英文；中文和全角符号会被程序剥掉" in template


class _FakeTextGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **_kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.response


async def _ignore_text(_text: str, storage_message: bool = True) -> bool:
    del storage_message
    return True


def _run_workflow(
    model_name: str,
    *,
    response: str,
    config: dict | None = None,
    use_visual_continuity: bool = False,
) -> tuple[_FakeTextGenerator, object]:
    generator = _FakeTextGenerator(response)
    workflow = PromptGenerationWorkflow(
        config=config or {"prompt_generator": {"output_format": "json"}, "tag_retriever": {}},
        stream_id=f"nl-{model_name or 'none'}",
        text_generator=generator,
        send_text=_ignore_text,
        show_tag_candidates=False,
        log_prefix="test",
        model_name=model_name,
    )
    result = asyncio.run(
        workflow.generate(
            "女孩坐在窗台上，猫趴在她膝盖上",
            allow_inherit=False,
            use_visual_continuity=use_visual_continuity,
        )
    )
    return generator, result


def test_工作流按会话选定模型决定发哪一档(monkeypatch) -> None:
    monkeypatch.setattr(
        session_state, "is_nsfw_filter_enabled", lambda *_a, **_kw: False
    )
    response = (
        '{"version":3,"format":"single","intent":"normal","continuity":"new",'
        '"global":["rating:general","solo","1girl"],"people":[],"positions":[]}'
    )

    v5_generator, v5_result = _run_workflow("nai-diffusion-5-full", response=response)
    assert v5_result is not None
    assert "## 自然语言描述（当前模型是 NovelAI V5）" in v5_generator.prompts[0]

    v45_generator, v45_result = _run_workflow("nai-diffusion-4-5-full", response=response)
    assert v45_result is not None
    assert "## 自然语言短句（NAI 4/4.5 兜底）" in v45_generator.prompts[0]
    assert "## 自然语言描述（当前模型是 NovelAI V5）" not in v45_generator.prompts[0]

    # 未传模型名（历史调用方）按最保守的纯 tag 档走，不会误放行
    none_generator, none_result = _run_workflow("", response=response)
    assert none_result is not None
    assert "## 自然语言短句（NAI 4/4.5 兜底）" in none_generator.prompts[0]


def test_自定义模板的连续性路径也按档位补输出契约(monkeypatch) -> None:
    """自定义模板整体替换了内置规则，输出契约由程序补回，不能永远补纯 tag 那一份。"""

    monkeypatch.setattr(
        session_state, "is_nsfw_filter_enabled", lambda *_a, **_kw: False
    )
    config = {
        "prompt_generator": {
            "output_format": "json",
            "prompt_template": "我的自定义模板：<<USER_REQUEST>>",
        },
        "tag_retriever": {},
    }
    response = (
        '{"version":4,"format":"single","intent":"normal",'
        '"dynamic":{"subject":["rating:general","solo","1girl"],'
        '"action":["sitting on windowsill"],"emotion":["calm"],'
        '"scene":["afternoon sunlight"],"framing":["upper body"],'
        '"people":[],"positions":[]},'
        '"stable":{"outfit":{"mode":"replace","key":"white_knit",'
        '"tags":["white knit sweater"]},'
        '"environment":{"mode":"replace","key":"her_room","tags":["small bedroom"]}}}'
    )

    v5_generator, v5_result = _run_workflow(
        "nai-diffusion-5-curated",
        response=response,
        config=config,
        use_visual_continuity=True,
    )
    assert v5_result is not None
    assert "或一句英文自然语言短句" in v5_generator.prompts[0]

    v45_generator, v45_result = _run_workflow(
        "nai-diffusion-4-5-curated",
        response=response,
        config=config,
        use_visual_continuity=True,
    )
    assert v45_result is not None
    assert "禁止自然语言句子" in v45_generator.prompts[0]


def test_自然语言元素能活着走完整条渲染与后处理链() -> None:
    """规则放行了却在下游被切碎/丢弃，这条放宽就等于没做。"""

    sentence = "the grey tabby cat is curled up asleep on her lap"
    payload = (
        '{"version":3,"format":"single","intent":"normal","continuity":"new",'
        '"global":["rating:general","solo","1girl","cat","upper body",'
        f'"{sentence}"],"people":[],"positions":[]}}'
    )

    rendered = parse_prompt_from_structured_output(payload)
    assert rendered is not None
    assert sentence in rendered, "JSON 渲染阶段就把自然语言元素弄丢了"

    # 送 API 前必过的两道后处理：剥 CJK 与可选的轻量排序
    assert sentence in strip_cjk_and_fullwidth(rendered)
    ordered = normalize_prompt_order(rendered)
    assert sentence in ordered
    assert ordered.startswith("rating:general"), "排序后 rating 必须仍在首位"


def test_中文自然语言会被后处理剥掉所以规则必须要求英文() -> None:
    """规则里"必须是英文"不是风格偏好：NewAPI 收到 CJK 直接 400。"""

    mixed = "rating:general, solo, 1girl, 猫趴在她的膝盖上, cat"
    cleaned = strip_cjk_and_fullwidth(mixed)
    assert "猫趴在她的膝盖上" not in cleaned
    assert "cat" in cleaned


# ── 画面文字必须用英文半角引号包起来 ──────────────────────────────────────
#
# 与反推侧（core/rules/caption_rules.py）成对：反推从图里读出 "OPEN" 并用半角引号包住，
# 翻译侧把用户的"招牌上写 OPEN"翻成 `a sign that reads "OPEN"`。两端用同一种引号，
# 反推 → 回灌 /nai0 的往返才不会在引号形态上掉链子。


def test_v5_档要求画面文字用半角双引号包起来() -> None:
    for channel in PROMPT_CHANNELS:
        for sfw in (True, False):
            template = _template(channel, sfw=sfw, natural_language=True)
            assert "### 画面里要出现文字时" in template, f"{channel}/sfw={sfw} 缺少文字规则块"
            assert "那串字必须用英文半角双引号" in template
            # 逐个点名禁用形态；只说"用半角"模型会自行发挥成弯引号
            assert "不能用全角" in template
            assert "中文" in template and "单引号" in template


def test_v5_档的文字规则给出正例() -> None:
    """光说规则不给例子，模型会把整句描述一起圈进引号。"""

    template = _template("json", sfw=False, natural_language=True)

    assert 'a wooden shop sign that reads "OPEN"' in template
    assert '引号只包要画出来的那串字本身' in template
    assert 'a banner that reads "SALE"' in template


def test_v5_档的文字规则说明引号内也必须是英文() -> None:
    """引号内写中文会被 strip_cjk_and_fullwidth 剥成一对空引号，等于白写。"""

    template = _template("text", sfw=False, natural_language=True)

    assert "引号内的内容也必须是" in template
    assert "中文会被程序剥掉" in template


def test_v5_档的文字规则不与_no_text_负面词冲突() -> None:
    """质量词配方里带 `no text`，两条规则必须共存而不是互相取消。"""

    template = _template("json", sfw=False, natural_language=True)

    assert "no text" in template
    assert "两者不冲突" in template


def test_纯tag档不发放文字规则() -> None:
    """本次只按需求给 V5 档加规则；V4.5 档不该被顺手改掉。"""

    for channel in PROMPT_CHANNELS:
        template = _template(channel, sfw=False, natural_language=False)
        assert "### 画面里要出现文字时" not in template
