"""自然语言到 NovelAI Prompt 的完整工作流。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from src.common.logger import get_logger

from ..rules.prompt_rules import (
    PROMPT_GENERATOR_JSON_TEMPLATE,
    PROMPT_GENERATOR_TEMPLATE,
    SFW_PROMPT_GENERATOR_JSON_TEMPLATE,
    SFW_PROMPT_GENERATOR_TEMPLATE,
    SFW_VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
    VISUAL_CONTINUITY_JSON_OUTPUT_INSTRUCTION,
    VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE,
)
from ..rules.selfie_rules import (
    detect_explicit_image_request,
    detect_selfie_from_output,
    get_selfie_hint,
)
from ..tag_retriever_display import build_tag_retriever_display_messages
from ..utils.prompt_output_parser import (
    extract_last_code_block,
    parse_prompt_from_structured_output,
    resolve_multi_character_payload,
)
from .llm_text_generator import LLMTextGenerator
from .prompt_memory import render_previous_prompt_block
from .session_state import session_state
from .tag_candidate_resolver import resolve_tag_candidates
from .visual_continuity import (
    StableVisualTags,
    VisualChangeDirective,
    describe_visual_failure,
    render_visual_continuity_context,
    resolve_visual_continuity,
)


logger = get_logger("nai_draw_plugin")


class TextSender(Protocol):
    async def __call__(self, text: str, storage_message: bool = True) -> bool: ...


@dataclass(frozen=True)
class PromptGenerationResult:
    text: str
    structured: dict[str, Any] | None
    visual_continuity: StableVisualTags | None = None


class PromptGenerationWorkflow:
    """隐藏模板、检索、记忆、LLM 调用和输出解析的深 Module。"""

    # 首次响应之外允许的修复重试次数；所有失败原因码都是 LLM 可修复的
    _MAX_VISUAL_REPAIR_ATTEMPTS = 2

    def __init__(
        self,
        *,
        config: dict[str, Any],
        stream_id: str,
        text_generator: LLMTextGenerator,
        send_text: TextSender,
        show_tag_candidates: bool,
        log_prefix: str,
    ) -> None:
        self._config = config
        self._stream_id = stream_id
        self._text_generator = text_generator
        self._send_text = send_text
        self._show_tag_candidates = show_tag_candidates
        self._log_prefix = log_prefix

    async def generate(
        self,
        request_text: str,
        *,
        allow_inherit: bool,
        tag_query_text: str = "",
        include_custom_system_prompt: bool = True,
        reasoning_context_text: str = "",
        use_visual_continuity: bool = False,
        include_outfit: bool = True,
        stable_change_text: str = "",
        visual_directives: dict[str, VisualChangeDirective] | None = None,
    ) -> PromptGenerationResult | None:
        request_text = str(request_text or "").strip()
        if not request_text:
            return None
        tag_query_text = str(tag_query_text or request_text).strip()

        generator_config = self._get_dict_config("prompt_generator")
        output_format = str(
            generator_config.get("output_format", "json") or "json"
        ).strip().lower()
        nsfw_filter_enabled = session_state.is_nsfw_filter_enabled(
            "stream",
            self._stream_id,
            self._get_config,
        )
        if use_visual_continuity:
            default_template = (
                SFW_VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE
                if nsfw_filter_enabled
                else VISUAL_CONTINUITY_PROMPT_GENERATOR_TEMPLATE
            )
        elif output_format == "json":
            default_template = (
                SFW_PROMPT_GENERATOR_JSON_TEMPLATE
                if nsfw_filter_enabled
                else PROMPT_GENERATOR_JSON_TEMPLATE
            )
        else:
            default_template = (
                SFW_PROMPT_GENERATOR_TEMPLATE
                if nsfw_filter_enabled
                else PROMPT_GENERATOR_TEMPLATE
            )

        previous_prompt = ""
        previous_request = ""
        last_selfie_prompt = ""
        last_selfie_request = ""
        stable_visual_tags: StableVisualTags | None = None
        inherit_ttl = self._as_float(
            self._get_config("prompt_generator.inherit_ttl", 0),
            0.0,
        )
        if allow_inherit and self._stream_id:
            previous_prompt, previous_request = session_state.get_last_nai_context(
                self._stream_id,
                ttl=inherit_ttl,
            )
            (
                last_selfie_prompt,
                last_selfie_request,
                _last_selfie_scene,
                _last_selfie_anchor,
            ) = session_state.get_last_selfie_context(
                self._stream_id,
                ttl=inherit_ttl,
            )
            if use_visual_continuity:
                stable_visual_tags = session_state.get_visual_continuity(
                    self._stream_id,
                    ttl=self._as_float(
                        self._get_config("prompt_generator.visual_state_ttl", 0),
                        0.0,
                    ),
                )

        custom_template = str(generator_config.get("prompt_template") or "").strip()
        if custom_template and use_visual_continuity:
            prompt_template = (
                f"{custom_template}\n\n<<VISUAL_CONTINUITY_CONTEXT>>\n\n"
                f"{VISUAL_CONTINUITY_JSON_OUTPUT_INSTRUCTION}"
            )
        else:
            prompt_template = custom_template or default_template
        prompt = self._render_prompt(
            prompt_template,
            request_text,
            include_custom_system_prompt=(
                include_custom_system_prompt and not nsfw_filter_enabled
            ),
            previous_prompt=(previous_prompt or "")
            if allow_inherit and not use_visual_continuity
            else "",
            previous_request=(previous_request or "")
            if allow_inherit and not use_visual_continuity
            else "",
            last_selfie_prompt=(last_selfie_prompt or "")
            if allow_inherit and not use_visual_continuity
            else "",
            last_selfie_request=(last_selfie_request or "")
            if allow_inherit and not use_visual_continuity
            else "",
            reasoning_context_text=reasoning_context_text,
            stable_visual_tags=stable_visual_tags,
            include_outfit=include_outfit,
        )

        tag_candidates = await self._retrieve_tag_candidates(tag_query_text)
        if self._show_tag_candidates:
            await self._send_tag_retriever_display(tag_candidates)
        prompt = prompt.replace("<<TAG_CANDIDATES>>", tag_candidates).strip()

        effective_generator_config = generator_config
        if use_visual_continuity:
            # 连续性路径要求确定性输出，固定低温覆盖用户配置（见 temperature 配置说明）
            effective_generator_config = {**generator_config, "temperature": 0.2}
        response = await self._text_generator.generate(
            prompt,
            request_type="nai_draw_plugin.prompt_generator",
            generator_config=effective_generator_config,
            default_model_name="planner",
            default_temperature=0.2,
            default_max_tokens=200,
        )
        if not response:
            return None

        continuity_result = None
        if use_visual_continuity:
            # resolve 是唯一校验事实源：失败原因码直接驱动修复重试，
            # 覆盖 JSON 解析失败 / 缺 rating / 稳定区协议冲突等全部失败模式
            for repair_attempt in range(self._MAX_VISUAL_REPAIR_ATTEMPTS + 1):
                if repair_attempt:
                    response = await self._text_generator.generate(
                        self._build_visual_repair_prompt(
                            prompt,
                            response,
                            continuity_result.failure,
                        ),
                        request_type="nai_draw_plugin.prompt_generator",
                        generator_config=effective_generator_config,
                        default_model_name="planner",
                        default_temperature=0.2,
                        default_max_tokens=200,
                    )
                    if not response:
                        return None
                extracted = extract_last_code_block(response)
                continuity_result = resolve_visual_continuity(
                    extracted if extracted is not None else response,
                    previous=stable_visual_tags,
                    include_outfit=include_outfit,
                    directives=visual_directives,
                    stable_change_text=stable_change_text,
                )
                if continuity_result.prompt:
                    break
                if repair_attempt < self._MAX_VISUAL_REPAIR_ATTEMPTS:
                    logger.info(
                        "%s v4 视觉连续性响应不可用（原因: %s），发起第 %d 次修复",
                        self._log_prefix,
                        continuity_result.failure,
                        repair_attempt + 1,
                    )
            if not continuity_result.prompt:
                logger.warning(
                    "%s 未获得可用的 v4 视觉连续性 Tag（原因: %s），已拒绝发送",
                    self._log_prefix,
                    continuity_result.failure,
                )
                return None

        cleaned_prompt = (
            continuity_result.prompt
            if continuity_result is not None
            else self._cleanup_llm_prompt(response)
        )
        if not cleaned_prompt:
            return None

        structured_payload = (
            None
            if continuity_result is not None
            else resolve_multi_character_payload(response, cleaned_prompt)
        )
        return PromptGenerationResult(
            cleaned_prompt,
            structured_payload,
            continuity_result.stable if continuity_result is not None else None,
        )

    def _get_config(self, key: str, default: Any = None) -> Any:
        current: Any = self._config
        for part in str(key or "").split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def _get_dict_config(self, key: str) -> dict[str, Any]:
        value = self._get_config(key, {})
        return value if isinstance(value, dict) else {}

    def _render_prompt(
        self,
        template: str,
        request_text: str,
        *,
        include_custom_system_prompt: bool,
        previous_prompt: str,
        previous_request: str,
        last_selfie_prompt: str,
        last_selfie_request: str,
        reasoning_context_text: str,
        stable_visual_tags: StableVisualTags | None,
        include_outfit: bool,
    ) -> str:
        custom_system_prompt = ""
        if include_custom_system_prompt:
            custom_system_prompt = str(
                self._get_config("custom_prompt.system_prompt", "") or ""
            ).strip()
        if custom_system_prompt:
            custom_system_prompt += "\n\n"

        prompt = template.replace("<<CUSTOM_SYSTEM_PROMPT>>", custom_system_prompt).strip()
        prompt = prompt.replace(
            "<<PREVIOUS_PROMPT>>",
            render_previous_prompt_block(previous_prompt, previous_request),
        ).strip()
        prompt = prompt.replace(
            "<<VISUAL_CONTINUITY_CONTEXT>>",
            render_visual_continuity_context(
                stable_visual_tags,
                include_outfit=include_outfit,
            ),
        ).strip()
        prompt = prompt.replace(
            "<<REASONING_CONTEXT>>",
            self._render_reasoning_context(reasoning_context_text),
        ).strip()
        prompt = prompt.replace(
            "<<CURRENT_TIME_CONTEXT>>",
            self._build_current_time_context(),
        ).strip()
        prompt = prompt.replace("<<SELFIE_HINT>>", get_selfie_hint()).strip()
        prompt = prompt.replace(
            "<<SELFIE_SCENE_CONTEXT>>",
            self._build_selfie_scene_context(
                request_text,
                last_selfie_prompt=last_selfie_prompt,
                last_selfie_request=last_selfie_request,
            ),
        ).strip()
        return prompt.replace("<<USER_REQUEST>>", request_text.strip() or "N/A")

    @staticmethod
    def _build_visual_repair_prompt(
        base_prompt: str,
        rejected_response: str,
        failure: str,
    ) -> str:
        """把 resolve 的失败原因翻译成修复指引，附上被拒响应供 LLM 对照。"""

        return (
            f"{base_prompt}\n\n<visual_continuity_repair>\n"
            "上一次输出未通过程序校验，不能使用。"
            f"本次失败原因：{describe_visual_failure(failure)}\n"
            "请对照 planner_visual_request 与上方稳定 Tag 库逐项修正并重新输出完整 JSON；"
            "**Planner 的 outfit_change / environment_change 决定优先于本 JSON，"
            "不要自行猜测稳定区是否变化。**"
            "stable.outfit 应尽量是整套当前可见穿搭，并优先保留用户明确给出的颜色、材质、"
            "版型/剪裁、长度/层次、结构、鞋袜或配饰；不要为了凑数量凭空编造。"
            "stable.environment 应尽量保留用户明确给出的空间布局、固定家具/物件、材质和配色；"
            "dynamic 只放动作、表情、镜头、光线和临时物件。不得遗漏用户明确描述。"
            f"\n<rejected_visual_continuity_json>\n{rejected_response}\n"
            "</rejected_visual_continuity_json>\n"
            "只输出修正后的 JSON。\n</visual_continuity_repair>"
        )

    async def _retrieve_tag_candidates(self, request_text: str) -> str:
        return await resolve_tag_candidates(
            self._get_dict_config("tag_retriever"),
            request_text,
            log_prefix=self._log_prefix,
        )

    async def _send_tag_retriever_display(self, tag_candidates: str) -> None:
        retriever_config = self._get_dict_config("tag_retriever")
        retriever_mode = str(
            retriever_config.get("mode", "online") or "online"
        ).strip().lower() or "online"
        if not isinstance(tag_candidates, str) or not tag_candidates.strip():
            tag_candidates = (
                "<tag_candidates>\n"
                f"⚠️ 未检索到候选标签（mode={retriever_mode}）\n"
                "</tag_candidates>"
            )

        for max_chars in (180, 120, 90, 72):
            display_messages = build_tag_retriever_display_messages(
                tag_candidates,
                max_chars=max_chars,
            )
            if not display_messages:
                return

            for display_message in display_messages:
                if await self._send_text(display_message, storage_message=False):
                    continue
                logger.warning(
                    "%s Danbooru 检索结果回显发送失败，尝试缩小分段重试: max_chars=%s",
                    self._log_prefix,
                    max_chars,
                )
                break
            else:
                return

        logger.warning(
            "%s Danbooru 检索结果回显发送失败，已放弃回显",
            self._log_prefix,
        )

    @staticmethod
    def _render_reasoning_context(text: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return ""
        return (
            "<planner_reasoning>\n"
            "（Planner 本轮 reasoning。与 user_request 冲突时以本块为准："
            "动词保持原意，情绪贴 reasoning，不要默认套'迷离/陶醉'。）\n"
            f"{normalized}\n"
            "</planner_reasoning>"
        )

    @staticmethod
    def _build_current_time_context() -> str:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 8:
            period = "清晨"
        elif 8 <= hour < 11:
            period = "上午"
        elif 11 <= hour < 14:
            period = "中午"
        elif 14 <= hour < 17:
            period = "下午"
        elif 17 <= hour < 19:
            period = "傍晚"
        elif 19 <= hour < 23:
            period = "夜晚"
        else:
            period = "深夜"
        return (
            "<current_time_context>\n"
            f"当前本地时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（{period}）。\n"
            "仅在用户未明确指定时，用于补全时间、光线和背景氛围。\n"
            "</current_time_context>"
        )

    @classmethod
    def _build_selfie_scene_context(
        cls,
        request_text: str,
        *,
        last_selfie_prompt: str = "",
        last_selfie_request: str = "",
    ) -> str:
        current_request = str(request_text or "").strip()
        previous_prompt = str(last_selfie_prompt or "").strip()
        if not cls._is_likely_selfie_request(current_request, previous_prompt):
            return ""

        lines = [
            "<selfie_scene_context>",
            "这轮请求可能属于 Bot 本人的连续情景图；Bot 出镜不代表本轮必须自拍。",
            "若聊天没有明确建立换装或地点变化，服装设计和环境结构必须沿用已有稳定 Tag。",
            "动作、表情、姿态、构图和镜头按本轮情景动态生成，不要求继承上一张图；"
            "当下时间、天气和光线也只按当前语境决定。",
            "用户明确指定视觉重点时，选择能看清该重点的构图，但不要借此改写未变化的服装或环境。",
        ]
        if last_selfie_request:
            lines.append(f"上一轮用户请求：{last_selfie_request.strip()}")
        if previous_prompt:
            lines.append(f"上一轮 Bot 情景图提示词：{previous_prompt}")
        lines.append("</selfie_scene_context>")
        return "\n".join(lines)

    @staticmethod
    def _is_likely_selfie_request(
        request_text: str,
        last_selfie_prompt: str = "",
    ) -> bool:
        text = str(request_text or "").strip()
        if not text:
            return False
        if detect_explicit_image_request(text):
            return True
        if last_selfie_prompt and detect_selfie_from_output(last_selfie_prompt):
            continuation_patterns = (
                r"继续",
                r"还是.*",
                r"来点不一样",
                r"换成.+",
                r"改成.+",
                r"换地方",
                r"同一个场景",
                r"同样背景",
            )
            return any(re.search(pattern, text) for pattern in continuation_patterns)
        return False

    @staticmethod
    def _cleanup_llm_prompt(prompt: str) -> str:
        if not prompt:
            return ""
        extracted = extract_last_code_block(prompt)
        candidate = extracted if extracted is not None else prompt
        parsed_prompt = parse_prompt_from_structured_output(candidate)
        if parsed_prompt:
            return parsed_prompt

        cleaned = candidate.strip()
        cleaned = re.sub(r"^\s*prompt\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("，", ", ")
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip("` \n")

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return default
