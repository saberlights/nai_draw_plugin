# -*- coding: utf-8 -*-
"""
/nai 命令：使用自然语言描述生成图片
"""
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger
from src.plugin_system import llm_api

from ..clients.nai_web_client import NaiWebClient
from ..mixins.auto_recall_mixin import AutoRecallMixin
from ..mixins.model_config_mixin import ModelConfigMixin
from ..rules.prompt_rules import PROMPT_GENERATOR_TEMPLATE, SFW_PROMPT_GENERATOR_TEMPLATE
from ..rules.selfie_rules import (
    detect_selfie_from_output,
    get_selfie_hint,
    merge_selfie_prompt,
)
from ..services.session_state import session_state
from ..services.tag_candidate_resolver import resolve_tag_candidates
from ..utils.prompt_output_parser import (
    extract_multi_character_payload,
    parse_prompt_from_structured_output,
)
from ..utils.prompt_postprocessor import (
    normalize_characters_order,
    normalize_prompt_order,
    remove_selfie_appearance_tags,
    sanitize_sfw_characters,
    sanitize_sfw_prompt,
    user_mentions_appearance,
)
from ..utils.random_scene_description import (
    get_random_scene_similarity_score,
    is_random_scene_too_similar,
    normalize_random_scene_description,
)
from ..constants import NAI_PIC_IMAGE_DISPLAY_MARKER

logger = get_logger("nai_draw_plugin")


class NaiDrawCommand(ModelConfigMixin, AutoRecallMixin, BaseCommand):
    """NovelAI 快速生图命令：/nai [描述]"""

    command_name = "nai_draw"
    command_description = "使用自然语言描述生成图片，例如：/nai 画一张初音未来"
    command_pattern = r"(?:.*，说：\s*)?/nai\s+(?!on$|off$|st$|sp$|set\b|art\b|artgen\b|artr$|artfix\b|size\b|help$|pt\s|nsfw\b|撤回(?:\s|$))(?P<description>[\s\S]+)$"

    # 类变量：记录最近的随机场景，避免重复
    _recent_random_scenes: list = []
    _MAX_RECENT_SCENES = 5
    _MAX_RANDOM_SCENE_ATTEMPTS = 4
    _RANDOM_SCENE_REPEAT_THRESHOLD = 0.6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_client = NaiWebClient(self)

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行 /nai 命令"""
        logger.info(f"{self.log_prefix} [LLM生图] 收到请求")

        # 检查用户权限
        has_permission = self._check_user_permission()
        if not has_permission:
            return False, "没有权限", True

        # 获取用户输入的描述
        description = self.matched_groups.get("description", "").strip()

        if not description:
            await self.send_text("请输入你想画的内容，例如：/nai 画一张初音未来")
            return False, "未提供描述", True

        # 随机模式：LLM 先生成随机场景关键词
        is_random_selfie = description in ("随机自拍", "random selfie")
        if description in ("随机", "random", "rand") or is_random_selfie:
            description = await self._generate_random_description(selfie=is_random_selfie)
            if not description:
                await self.send_text("随机场景生成失败，请稍后再试~")
                return False, "随机生成失败", True
            logger.info(f"{self.log_prefix} [LLM生图] 随机场景: {description}")

        # 使用 LLM 生成提示词（自拍意图由 LLM 自行判断）
        llm_result = await self._generate_prompt_with_llm(description)

        if not llm_result:
            logger.warning(f"{self.log_prefix} [LLM生图] 提示词生成失败")
            await self.send_text("提示词生成失败，请稍后再试~")
            return False, "提示词生成失败", True

        generated_prompt, structured_payload = llm_result
        logger.debug(f"{self.log_prefix} [LLM生图] 原始提示词: {generated_prompt}")

        # 从 LLM 输出检测是否为自拍
        is_selfie = detect_selfie_from_output(generated_prompt)

        # 处理自拍模式（添加角色特征）
        selfie_base_prompt = generated_prompt
        if is_selfie:
            generated_prompt = self._process_selfie_prompt(
                selfie_base_prompt,
                description,
                include_selfie_prompt_add=True,
                log_changes=True,
            )
            # 自拍场景按单字符串路径处理（_process_selfie_prompt 只作用于字符串）
            structured_payload = None

        # 轻量排序（可配置关闭）
        if self.get_config("prompt_generator.enforce_tag_order", False):
            generated_prompt = normalize_prompt_order(generated_prompt)
            structured_payload = self._normalize_structured_order(structured_payload)

        try:
            platform, chat_id, _ = self._get_chat_identity()
            if platform and chat_id and session_state.is_nsfw_filter_enabled(platform, chat_id, self.get_config):
                generated_prompt = sanitize_sfw_prompt(generated_prompt)
                structured_payload = self._sanitize_structured_for_sfw(structured_payload)
        except Exception:
            pass

        logger.info(f"{self.log_prefix} [LLM生图] 最终提示词: {generated_prompt}")

        # 检查是否需要显示提示词
        if self._is_prompt_show_enabled():
            show_prompt = generated_prompt
            header = "📝 提示词:"
            if is_selfie and self.get_config("prompt_show.hide_selfie_prompt_add", False):
                show_prompt = self._process_selfie_prompt(
                    selfie_base_prompt,
                    description,
                    include_selfie_prompt_add=False,
                    log_changes=False,
                )
                header = "📝 提示词(已隐藏自拍补充):"
            try:
                platform, chat_id, _ = self._get_chat_identity()
                if platform and chat_id and session_state.is_nsfw_filter_enabled(platform, chat_id, self.get_config):
                    show_prompt = sanitize_sfw_prompt(show_prompt)
            except Exception:
                pass
            await self.send_text(f"{header}\n{show_prompt}", storage_message=False)

        # 获取模型配置
        model_config = self._get_model_config(is_selfie=is_selfie)
        if not model_config or not model_config.get("base_url"):
            await self.send_text("NovelAI 配置错误，请检查配置文件")
            return False, "配置错误", True

        # 获取图片尺寸
        image_size = model_config.get("nai_size") or model_config.get("default_size", "1024x1280")

        # 显示处理信息
        enable_debug = self.get_config("components.enable_debug_info", False)
        if enable_debug:
            await self.send_text(f"正在生成图片，请稍候...")

        try:
            # 调用 API 生成图片（异步，不阻塞事件循环）
            request_prompt, request_characters = self._select_send_payload(
                generated_prompt, structured_payload
            )
            success, result = await self.api_client.generate_image(
                prompt=request_prompt,
                model_config=model_config,
                size=image_size,
                characters=request_characters,
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} [LLM生图] 图片生成失败: {e!r}", exc_info=True)
            await self.send_text(f"生成图片时出错: {str(e)[:100]}")
            return False, f"生成失败: {e}", True

        if success:
            final_image_data = self._process_api_response(result)

            if final_image_data:
                send_time = time.time()

                # 判断是 URL 还是 base64
                if final_image_data.startswith(("http://", "https://")):
                    # 直接发送图片 URL（参考 lolicon 插件）
                    try:
                        send_success = await self.send_custom(
                            "imageurl",
                            final_image_data,
                            display_message=NAI_PIC_IMAGE_DISPLAY_MARKER,
                        )
                        if send_success:
                            self._last_send_timestamp = send_time
                            if enable_debug:
                                await self.send_text("图片生成完成！")
                            await self._schedule_auto_recall()
                            return True, "图片生成成功", True
                        else:
                            await self.send_text("图片发送失败")
                            return False, "发送失败", True
                    except Exception as e:
                        logger.error(f"{self.log_prefix} [LLM生图] 图片URL发送失败: {e!r}")
                        await self.send_text(f"图片发送失败: {str(e)[:100]}")
                        return False, "发送失败", True
                elif final_image_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                    # Base64 格式 -> 以 image 段直发，napcat 等适配器原生支持
                    send_success = await self.send_custom(
                        "image",
                        final_image_data,
                        display_message=NAI_PIC_IMAGE_DISPLAY_MARKER,
                    )

                    if send_success:
                        self._last_send_timestamp = send_time
                        if enable_debug:
                            await self.send_text("图片生成完成！")
                        await self._schedule_auto_recall()
                        return True, "图片生成成功", True
                    else:
                        await self.send_text("图片发送失败")
                        return False, "发送失败", True
                else:
                    await self.send_text("API 返回了无法识别的图片格式")
                    return False, "数据格式错误", True
            else:
                await self.send_text("API 返回了无效的数据")
                return False, "数据格式错误", True
        else:
            await self.send_text(f"生成图片失败：{result}")
            return False, f"生成失败: {result}", True

    async def _generate_prompt_with_llm(
        self,
        request_text: str,
    ) -> Optional[Tuple[str, Optional[Dict[str, Any]]]]:
        """使用 LLM 生成英文提示词（自拍意图由 LLM 自行判断）。

        Returns:
            ``(text, structured)`` 二元组；``text`` 为拍平后的字符串，``structured`` 在 v3 multi
            JSON 且 ≥2 人时为 ``{"global_text", "characters", "has_coords"}``，否则为 ``None``；
            整体失败返回 ``None``。
        """
        generator_config = self._get_prompt_generator_config()

        # 检查是否启用 NSFW 过滤，选择对应模板
        try:
            platform, chat_id, _ = self._get_chat_identity()
            nsfw_filter_enabled = False
            if platform and chat_id:
                nsfw_filter_enabled = session_state.is_nsfw_filter_enabled(platform, chat_id, self.get_config)
        except Exception:
            nsfw_filter_enabled = False

        # 根据过滤状态与输出格式选择模板
        output_format = (generator_config.get("output_format") or "json").strip().lower()
        if nsfw_filter_enabled:
            if output_format == "json":
                from ..rules.prompt_rules import SFW_PROMPT_GENERATOR_JSON_TEMPLATE
                default_template = SFW_PROMPT_GENERATOR_JSON_TEMPLATE
            else:
                default_template = SFW_PROMPT_GENERATOR_TEMPLATE
        else:
            if output_format == "json":
                from ..rules.prompt_rules import PROMPT_GENERATOR_JSON_TEMPLATE
                default_template = PROMPT_GENERATOR_JSON_TEMPLATE
            else:
                default_template = PROMPT_GENERATOR_TEMPLATE

        prompt_template = generator_config.get("prompt_template") or default_template
        prompt = self._render_generator_prompt(prompt_template, request_text)

        # Tag 检索增强
        tag_candidates_text = await self._retrieve_tag_candidates(request_text)
        prompt = prompt.replace("<<TAG_CANDIDATES>>", tag_candidates_text).strip()

        # 获取 LLM 模型配置
        model_config = self._resolve_llm_model_config(generator_config.get("model_name", ""))
        if not model_config:
            logger.error(f"{self.log_prefix} 未找到可用的 LLM 模型")
            return None

        temperature = generator_config.get("temperature", 0.2)
        max_tokens = generator_config.get("max_tokens", 200)

        try:
            success, response, reasoning, model_name = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config,
                request_type="nai_draw_plugin.prompt_generator",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} LLM 调用失败: {e}", exc_info=True)
            return None

        if not success or not response:
            logger.error(f"{self.log_prefix} LLM 生成失败")
            return None

        cleaned = self._cleanup_llm_prompt(response)
        if not cleaned:
            return None

        # 从原始 LLM 响应中抽出 v3 multi 结构化 payload，供 NewAPI characters[] 通道使用
        structured_payload = extract_multi_character_payload(response)
        return cleaned, structured_payload

    def _normalize_structured_order(
        self,
        structured: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """按 normalize_prompt_order 规则整理多角色 payload。"""
        if not structured:
            return None
        new_global, new_chars = normalize_characters_order(
            structured.get("global_text", ""),
            structured.get("characters") or [],
        )
        return {**structured, "global_text": new_global, "characters": new_chars}

    def _sanitize_structured_for_sfw(
        self,
        structured: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """SFW 过滤多角色 payload，过滤后人数不足 2 时返回 None 触发字符串降级。"""
        if not structured:
            return None
        new_global, new_chars = sanitize_sfw_characters(
            structured.get("global_text", ""),
            structured.get("characters") or [],
        )
        if len(new_chars) < 2:
            logger.info(
                f"{self.log_prefix} SFW 过滤后多角色 payload 剩余 {len(new_chars)} 项，"
                "降级回单字符串路径"
            )
            return None
        return {**structured, "global_text": new_global, "characters": new_chars}

    @staticmethod
    def _select_send_payload(
        prompt: str,
        structured: Optional[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """根据是否存在合法结构化 payload，决定送往 generate_image 的 (prompt, characters)。"""
        if structured and len(structured.get("characters") or []) >= 2:
            return structured.get("global_text", "") or prompt, list(structured["characters"])
        return prompt, None

    def _render_generator_prompt(
        self,
        template: str,
        original_request: str,
    ) -> str:
        """渲染提示词生成模板（/nai 命令不使用提示词继承）"""
        # 永远注入自拍提示，由 LLM 自行判断是否为自拍意图
        selfie_hint = get_selfie_hint()

        # /nai 命令不需要提示词继承和自定义系统提示词，清除这些占位符
        prompt = template.replace("<<CUSTOM_SYSTEM_PROMPT>>", "").strip()
        prompt = prompt.replace("<<PREVIOUS_PROMPT>>", "").strip()
        prompt = prompt.replace("<<CURRENT_TIME_CONTEXT>>", self._build_current_time_context()).strip()
        prompt = prompt.replace("<<SELFIE_HINT>>", selfie_hint).strip()
        prompt = prompt.replace("<<SELFIE_SCENE_CONTEXT>>", "").strip()
        prompt = prompt.replace("<<USER_REQUEST>>", original_request.strip() or "N/A")
        return prompt

    async def _retrieve_tag_candidates(self, request_text: str) -> str:
        """检索候选 danbooru tag（支持本地/在线两种模式，与 Action 路径共用调度逻辑）"""
        retriever_config = self.get_config("tag_retriever", None) or {}
        return await resolve_tag_candidates(
            retriever_config,
            request_text,
            log_prefix=self.log_prefix,
        )

    async def _generate_random_description(self, selfie: bool = False) -> Optional[str]:
        """LLM 生成随机场景关键词"""
        random_config = self._get_random_scene_config()
        model_config = self._resolve_llm_model_config(
            "",
            override_config=random_config,
        )
        if not model_config:
            return None

        best_candidate: Optional[str] = None
        best_score: Optional[float] = None
        rejected_candidates: list[str] = []

        for attempt in range(self._MAX_RANDOM_SCENE_ATTEMPTS):
            random_prompt = self._build_random_scene_prompt(selfie=selfie, rejected_candidates=rejected_candidates)
            result = await self._request_random_scene_candidate(random_prompt, model_config, random_config)
            if not result:
                continue

            score = get_random_scene_similarity_score(result, NaiDrawCommand._recent_random_scenes)
            if not is_random_scene_too_similar(
                result,
                NaiDrawCommand._recent_random_scenes,
                threshold=self._RANDOM_SCENE_REPEAT_THRESHOLD,
            ):
                self._remember_random_scene(result)
                return result

            rejected_candidates.append(result)
            logger.info(
                f"{self.log_prefix} [随机场景] 第{attempt + 1}次候选与历史过近，重复分数={score:.2f}，候选={result}"
            )
            if best_score is None or score < best_score:
                best_candidate = result
                best_score = score

        if best_candidate:
            logger.warning(
                f"{self.log_prefix} [随机场景] 多次重试后仍偏相似，使用最低重复分数候选 {best_score:.2f}: {best_candidate}"
            )
            self._remember_random_scene(best_candidate)
            return best_candidate

        return None

    def _build_random_scene_prompt(self, selfie: bool = False, rejected_candidates: Optional[list[str]] = None) -> str:
        """构建随机场景提示词。"""
        selfie_extra = ""
        if selfie:
            selfie_extra = (
                "\n\n额外要求（自拍模式）：\n"
                "- 必须明确是自拍，输出中包含'自拍'或具体自拍方式（镜子自拍、手机自拍、高角度自拍等）\n"
                "- 自拍内容同样要明确偏成人向：裸体自拍、自慰自拍、事中自拍、事后自拍、露出自拍等，不要只是穿着暴露的普通自拍\n"
                "- 自拍场景和方式也要多样化，不要总是同一种自拍套路"
            )

        random_prompt = f"""随机生成一个二次元 NSFW 场景，并用空格分隔的中文短标签描述它。

要求：
- 题材不限，强度不限，可以是任何成人向内容
- 结果必须具体、可视化、适合转成 Danbooru 风格标签
- 只输出 1 行，包含 6-10 个中文短标签
- 标签尽量简短，使用明确视觉概念，不要写成句子
- 标签尽量覆盖人数（几男几女），人物构成、状态、核心互动、视角、场景
- 不要和最近结果过于相似，尽量主动切换题材和画面类型{selfie_extra}"""

        # 注入最近生成历史，防止重复
        if NaiDrawCommand._recent_random_scenes:
            history = "\n".join(NaiDrawCommand._recent_random_scenes)
            random_prompt += (
                "\n\n以下是最近已生成过的内容，禁止与它们重复或相似：\n"
                f"{history}\n"
                "新的结果必须在核心行为、互动结构、视角重点里至少有一项明显不同，"
                "不能只换地点、服装、发型、表情这种表层元素。"
            )

        if rejected_candidates:
            rejected_text = "\n".join(rejected_candidates)
            random_prompt += (
                "\n\n以下候选刚刚被判定为与历史题材簇过于接近，禁止继续沿着这些方向小修小补：\n"
                f"{rejected_text}\n"
                "下一次必须彻底切换题材簇；不要只替换地点、衣服、体位细节。"
            )
        return random_prompt

    async def _request_random_scene_candidate(self, random_prompt: str, model_config, random_config: Dict[str, Any]) -> Optional[str]:
        """向 LLM 请求一个随机场景候选。"""

        try:
            success, response, _, _ = await llm_api.generate_with_model(
                prompt=random_prompt,
                model_config=model_config,
                request_type="nai_draw_plugin.random_scene",
                temperature=random_config.get("temperature", 1.0),
                max_tokens=random_config.get("max_tokens", 200),
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} 随机场景生成失败: {e}")
            return None

        if not success or not response:
            return None

        # 清理：LLM 自行选优后，代码只取第一条有效结果
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return normalize_random_scene_description(lines[0]) if lines else None

    def _remember_random_scene(self, result: str) -> None:
        """记录随机场景历史。"""
        if not result:
            return
        NaiDrawCommand._recent_random_scenes.append(result)
        if len(NaiDrawCommand._recent_random_scenes) > NaiDrawCommand._MAX_RECENT_SCENES:
            NaiDrawCommand._recent_random_scenes.pop(0)

    def _build_current_time_context(self) -> str:
        """为命令式生图提供轻量时间上下文。"""
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

    def _resolve_llm_model_config(self, preferred_name: str, override_config: Dict[str, Any] = None):
        """获取可用的 LLM 模型配置

        Args:
            preferred_name: 优先使用的模型名
            override_config: 覆盖配置，提供时优先使用其中的 custom_model
        """
        from src.config.api_ada_configs import TaskConfig

        # 如果有覆盖配置，优先使用其 custom_model
        if override_config:
            custom_model = override_config.get("custom_model")
            if custom_model and isinstance(custom_model, dict):
                model_list = custom_model.get("model_list", [])
                if model_list:
                    try:
                        custom_task_config = TaskConfig(
                            model_list=model_list if isinstance(model_list, list) else [model_list],
                            max_tokens=custom_model.get("max_tokens", 1024),
                            temperature=custom_model.get("temperature", 0.3),
                            slow_threshold=custom_model.get("slow_threshold", 30.0),
                            selection_strategy="random"
                        )
                        logger.info(f"{self.log_prefix} 使用自定义模型配置: {model_list}")
                        return custom_task_config
                    except Exception as e:
                        logger.warning(f"{self.log_prefix} 自定义模型配置创建失败: {e}，回退到默认逻辑")

        # 检查 prompt_generator 的自定义模型配置
        generator_config = self._get_prompt_generator_config()
        custom_model = generator_config.get("custom_model")

        if custom_model and isinstance(custom_model, dict):
            model_list = custom_model.get("model_list", [])
            if model_list:
                # 使用自定义模型配置创建 TaskConfig
                from src.config.api_ada_configs import TaskConfig
                try:
                    custom_task_config = TaskConfig(
                        model_list=model_list if isinstance(model_list, list) else [model_list],
                        max_tokens=custom_model.get("max_tokens", 1024),
                        temperature=custom_model.get("temperature", 0.3),
                        slow_threshold=custom_model.get("slow_threshold", 30.0),
                        selection_strategy="random"  # 固定使用随机选择
                    )
                    logger.info(f"{self.log_prefix} 使用自定义模型配置: {model_list}")
                    return custom_task_config
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 自定义模型配置创建失败: {e}，回退到系统模型")

        # 回退到系统模型
        models = llm_api.get_available_models()
        if not models:
            return None

        candidate_names = []
        if preferred_name:
            candidate_names.append(preferred_name)
        candidate_names.extend(["planner", "replyer"])

        for name in candidate_names:
            config = models.get(name)
            if config:
                logger.info(f"{self.log_prefix} 使用模型: {name}")
                return config

        fallback_name, fallback_config = next(iter(models.items()))
        logger.info(f"{self.log_prefix} 使用默认模型: {fallback_name}")
        return fallback_config

    def _cleanup_llm_prompt(self, prompt: str) -> str:
        """清理 LLM 返回的提示词"""
        if not prompt:
            return ""

        parsed = parse_prompt_from_structured_output(prompt)
        if parsed:
            logger.debug(f"{self.log_prefix} [LLM生图] 结构化提示词解析命中（JSON->prompt），将跳过文本清洗")
            return parsed

        cleaned = prompt.strip()

        # 处理代码块包裹
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
            # 移除可能的语言标识如 ```text
            if cleaned and not cleaned[0].isalnum() and cleaned[0] not in "{[(":
                pass  # 保持原样
            elif "\n" in cleaned:
                first_line, rest = cleaned.split("\n", 1)
                # 如果第一行看起来像语言标识（纯字母且较短）
                if first_line.strip().isalpha() and len(first_line.strip()) < 15:
                    cleaned = rest.strip()

        # 处理单行代码包裹
        if cleaned.startswith("`") and cleaned.endswith("`") and cleaned.count("`") == 2:
            cleaned = cleaned[1:-1].strip()

        # 处理引号包裹
        if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()

        # 处理常见前缀（不区分大小写）
        prefix_patterns = [
            r"^(?:output|result|prompt|here(?:'s| is)(?: the)?(?: prompt)?)\s*[:：]\s*",
            r"^(?:the )?(?:generated )?prompt\s*(?:is|:)\s*",
        ]
        for pattern in prefix_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # 处理多行内容
        if "\n" in cleaned:
            lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

            # 检测是否为多人场景分段格式（包含 | 分隔符）
            has_multi_person_format = any(line.startswith("|") for line in lines)

            if has_multi_person_format:
                # 多人场景：保留所有有效行，用换行符连接
                valid_lines = []
                for line in lines:
                    # 跳过以解释性词语开头的行
                    if re.match(r"^(note|explanation|this|i |the above|here)", line, re.IGNORECASE):
                        continue
                    valid_lines.append(line)
                if valid_lines:
                    cleaned = "\n".join(valid_lines)
            else:
                # 单人场景：只取第一行有效内容
                valid_lines = []
                for line in lines:
                    # 跳过以解释性词语开头的行
                    if re.match(r"^(note|explanation|this|i |the above|here)", line, re.IGNORECASE):
                        continue
                    valid_lines.append(line)
                if valid_lines:
                    cleaned = valid_lines[0]

        return cleaned

    def _get_prompt_generator_config(self) -> Dict[str, Any]:
        """获取提示词生成器配置"""
        return self.get_config("prompt_generator", None) or {}

    def _get_random_scene_config(self) -> Dict[str, Any]:
        """获取随机场景生成配置，未配置时回退到 prompt_generator"""
        random_config = self.get_config("random_scene", None) or {}
        if not random_config:
            # 回退：使用 prompt_generator 配置但覆盖默认值
            fallback = dict(self._get_prompt_generator_config())
            fallback.setdefault("temperature", 1.0)
            fallback.setdefault("max_tokens", 200)
            return fallback
        return random_config

    def _process_api_response(self, result: str) -> Optional[str]:
        """处理 API 响应"""
        if not result:
            return None

        if result.startswith(("http://", "https://")):
            return result

        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            return result

        if "," in result and result.startswith("data:image"):
            return result.split(",", 1)[1]

        return result

    def _process_selfie_prompt(
        self,
        description: str,
        raw_request: str = "",
        include_selfie_prompt_add: bool = True,
        log_changes: bool = True,
    ) -> str:
        """处理自拍模式的提示词：可选移除随机外貌 + （可选）合并配置中的自拍特征"""
        model_config = self._get_model_config(is_selfie=True)
        selfie_prompt_add = model_config.get("selfie_prompt_add", "") if model_config else ""

        policy = (self.get_config("prompt_generator.selfie_appearance_policy", "auto") or "auto").strip().lower()
        user_specified = user_mentions_appearance(raw_request)

        original = description

        # auto: 合并前先移除 LLM 随机外貌（保留配置中的自拍特征）
        if policy in {"auto", "never"} and not user_specified and policy == "auto":
            description = remove_selfie_appearance_tags(description)

        if include_selfie_prompt_add and selfie_prompt_add:
            description = merge_selfie_prompt(description, selfie_prompt_add)

        # never: 合并后再移除一次（连配置外貌也移除），但用户明确指定时不移除
        if policy in {"auto", "never"} and not user_specified and policy == "never":
            description = remove_selfie_appearance_tags(description)

        if log_changes and description != original:
            logger.debug(f"{self.log_prefix} [LLM生图] 自拍提示词后处理已生效：policy={policy}, user_specified={user_specified}")

        return description

    def _is_auto_recall_enabled(self, platform: str, chat_id: str) -> bool:
        """检查是否启用自动撤回"""
        return session_state.is_recall_enabled(platform, chat_id, self.get_config)

    def _is_prompt_show_enabled(self) -> bool:
        """检查是否启用提示词显示"""
        try:
            platform, chat_id, _ = self._get_chat_identity()
            if not platform or not chat_id:
                return False
            return session_state.is_prompt_show_enabled(platform, chat_id, self.get_config)
        except Exception as e:
            logger.error(f"{self.log_prefix} 检查提示词显示状态时出错: {e}")
            return False

    def _check_user_permission(self) -> bool:
        """检查当前用户是否有权限使用生图命令"""
        try:
            platform, chat_id, user_id = self._get_chat_identity()
            if not platform or not chat_id or not user_id:
                logger.warning(f"{self.log_prefix} 无法获取会话信息，默认允许")
                return True
            return session_state.check_user_permission(platform, chat_id, user_id, self.get_config)
        except Exception as e:
            logger.error(f"{self.log_prefix} 检查用户权限时出错: {e}", exc_info=True)
            return True
