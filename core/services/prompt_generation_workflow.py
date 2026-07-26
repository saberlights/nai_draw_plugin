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


logger = get_logger("nai_draw_plugin")


class TextSender(Protocol):
    async def __call__(self, text: str, storage_message: bool = True) -> bool: ...


@dataclass(frozen=True)
class PromptGenerationResult:
    text: str
    structured: dict[str, Any] | None


class PromptGenerationWorkflow:
    """隐藏模板、检索、记忆、LLM 调用和输出解析的深 Module。"""

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
        include_custom_system_prompt: bool = True,
        reply_context_text: str = "",
        reasoning_context_text: str = "",
    ) -> PromptGenerationResult | None:
        request_text = str(request_text or "").strip()
        if not request_text:
            return None

        generator_config = self._get_dict_config("prompt_generator")
        output_format = str(
            generator_config.get("output_format", "json") or "json"
        ).strip().lower()
        nsfw_filter_enabled = session_state.is_nsfw_filter_enabled(
            "stream",
            self._stream_id,
            self._get_config,
        )
        if output_format == "json":
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

        prompt_template = str(generator_config.get("prompt_template") or default_template)
        prompt = self._render_prompt(
            prompt_template,
            request_text,
            include_custom_system_prompt=(
                include_custom_system_prompt and not nsfw_filter_enabled
            ),
            previous_prompt=(previous_prompt or "") if allow_inherit else "",
            previous_request=(previous_request or "") if allow_inherit else "",
            last_selfie_prompt=(last_selfie_prompt or "") if allow_inherit else "",
            last_selfie_request=(last_selfie_request or "") if allow_inherit else "",
            reply_context_text=reply_context_text,
            reasoning_context_text=reasoning_context_text,
        )

        tag_candidates = await self._retrieve_tag_candidates(request_text)
        if self._show_tag_candidates:
            await self._send_tag_retriever_display(tag_candidates)
        prompt = prompt.replace("<<TAG_CANDIDATES>>", tag_candidates).strip()

        response = await self._text_generator.generate(
            prompt,
            request_type="nai_draw_plugin.prompt_generator",
            generator_config=generator_config,
            default_model_name="planner",
            default_temperature=0.2,
            default_max_tokens=200,
        )
        if not response:
            return None

        cleaned_prompt = self._cleanup_llm_prompt(response)
        if not cleaned_prompt:
            return None

        structured_payload = resolve_multi_character_payload(response, cleaned_prompt)
        if allow_inherit and self._stream_id:
            session_state.set_last_nai_context(
                self._stream_id,
                cleaned_prompt,
                request_text,
                inherit_ttl,
            )
        return PromptGenerationResult(cleaned_prompt, structured_payload)

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
        reply_context_text: str,
        reasoning_context_text: str,
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
            "<<REPLY_CONTEXT>>",
            self._render_reply_context(reply_context_text),
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
    def _render_reply_context(text: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return ""
        return (
            "<bot_reply_context>\n"
            "（这是 bot 本人这一轮即将说出去的回复原文。请基于这段语境扩展画面细节"
            "——衣着、姿态、光照、室内陈设等——让生成的图与文匹配，"
            "而不是仅看 user_request 的关键词。）\n"
            f"{normalized}\n"
            "</bot_reply_context>"
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
            "这轮请求很可能属于 bot 本人自拍/展示照 的连续发图。",
            "若用户没有明确要求切换场景、换穿搭或改光线，默认延续上一轮的背景、穿搭、时间氛围与构图重点。",
            "服装连续性要尽量真实：若用户没有明确要求换衣服、换颜色、换材质、换风格，默认延续上一轮服装款式、主色、材质、袜子和鞋子的视觉设定，不要突然从白衣变黑衣，或从针织变皮衣。",
            "如果用户明确指定了本轮想看的重点（如黑丝、鞋子、腿部、全身穿搭、背景），优先保留该重点，并选择能看清它的构图。",
        ]
        if last_selfie_request:
            lines.append(f"上一轮用户请求：{last_selfie_request.strip()}")
        if previous_prompt:
            lines.append(f"上一轮自拍提示词：{previous_prompt}")
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
