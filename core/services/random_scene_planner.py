"""开放题材随机场景规划。"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from src.common.logger import get_logger

from ..utils.random_scene_description import (
    ensure_random_scene_character,
    get_random_scene_similarity_score,
    is_random_scene_too_similar,
    normalize_random_scene_narrative,
)
from .llm_text_generator import LLMTextGenerator


logger = get_logger("nai_draw_plugin")


class RandomScenePlanner:
    """生成自然语言随机场景，并在多次调用间抑制近期重复。"""

    _recent_scenes: list[str] = []
    _max_recent_scenes = 5
    _max_attempts = 4
    _repeat_threshold = 0.6

    def __init__(
        self,
        *,
        config: dict[str, Any],
        text_generator: LLMTextGenerator,
        log_prefix: str,
    ) -> None:
        self._config = config
        self._text_generator = text_generator
        self._log_prefix = log_prefix

    async def generate(
        self,
        *,
        selfie: bool = False,
        character: str = "",
    ) -> str | None:
        character = str(character or "").strip()
        best_candidate: str | None = None
        best_score: float | None = None
        rejected_candidates: list[str] = []

        for _attempt in range(self._max_attempts):
            prompt = self._build_prompt(
                selfie=selfie,
                character=character,
                rejected_candidates=rejected_candidates,
            )
            response = await self._text_generator.generate(
                prompt,
                request_type="nai_draw_plugin.random_scene",
                generator_config=self._config,
                default_model_name="planner",
                default_temperature=1.2,
                default_max_tokens=240,
            )
            if not response:
                continue

            lines = [line.strip() for line in response.splitlines() if line.strip()]
            if not lines:
                continue
            normalized = normalize_random_scene_narrative(lines[0])
            if not normalized:
                continue

            candidate = ensure_random_scene_character(normalized, character)
            score = get_random_scene_similarity_score(candidate, self._recent_scenes)
            if not is_random_scene_too_similar(
                candidate,
                self._recent_scenes,
                threshold=self._repeat_threshold,
            ):
                self._remember(candidate)
                return candidate

            rejected_candidates.append(candidate)
            if best_score is None or score < best_score:
                best_candidate = candidate
                best_score = score
            logger.info(
                "%s 随机场景过于相似，重试中: %.2f %s",
                self._log_prefix,
                score,
                candidate,
            )

        if best_candidate:
            self._remember(best_candidate)
        return best_candidate

    def _build_prompt(
        self,
        *,
        selfie: bool,
        character: str,
        rejected_candidates: list[str],
    ) -> str:
        character_instruction = ""
        if character:
            character_literal = json.dumps(character, ensure_ascii=False)
            character_instruction = (
                "\n\n指定角色锚点（必须原样保留）：\n"
                f"- 自然语言描述必须包含字面角色名 {character_literal}，不得替换成泛称或其它角色\n"
                "- 以该角色为画面主体；可以按随机结果增加其它成年配角\n"
            )

        selfie_extra = ""
        if selfie:
            selfie_extra = (
                "\n\n额外要求（自拍模式）：\n"
                "- 必须明确是手持相机、手机自拍、镜子自拍或其它可视化自拍方式\n"
                "- 自拍内容仍然是明确成人向色图，不要退化成普通写真\n"
                "- 自拍只是镜头形式，不限制其它题材和成人行为"
            )

        prompt = (
            "你是二次元成人向生图的开放式随机场景设计器。\n"
            "目标是生成一张明确的 NSFW 色图场景，而不是普通插画或安全写真。\n"
            f"本轮随机熵（只用于内部抽签，不要输出）：{uuid4().hex}\n\n"
            "随机规则：\n"
            "- 不要从固定清单或少数安全模板中轮换；题材空间开放，可以选择常见、冷门、跨题材、"
            "超现实或新奇的成人设定，并主动发明没有见过的组合。\n"
            "- 独立决定人物构成与关系、成人行为/癖好、服装与身体状态、姿势互动、镜头构图、"
            "地点与时代、道具、光线和情绪；每次至少改变其中 4 个维度。\n"
            "- 不要因为‘随机’就总是单人、卧室、站立、自拍或同一种体位；最近结果中出现过的"
            "题材、地点、镜头和动作都要主动避开。\n"
            "- 所有参与者必须明确是成年人（18+）且自愿，不生成未成年人色情。\n\n"
            "开放灵感（仅为非穷举示例，绝不是白名单）：可以探索性交/插入、口交/乳交/后入等"
            "不同体位，多人或复杂关系，拘束/支配、触手/异种、医疗/实验、露出/公共场所、制服/"
            "角色扮演，以及任何你能构思的其它成人题材；不要把随机结果限制在这些例子里。\n\n"
            "创作流程（不要输出思考过程）：\n"
            "- 先在内部完成一份导演式画面设计，再整理成简洁连贯的自然语言；情色主轴必须清晰，至少出现明确成人行为、"
            "身体接触、裸露状态或性兴奋状态，不能只写暧昧、泳装、漂亮或普通写真。\n"
            "- 角色当下状态要具体：表情、视线、呼吸/汗/脸红等身体反应、姿态重心、手脚位置、头发和"
            "肌肤状态，以及角色之间正在发生的互动。\n"
            "- 服饰状态要具体：服装款式、材质、颜色、层次和穿着变化（例如扣子解开、肩带滑落、半穿、"
            "内衣、袜子、鞋）；服饰类型要主动变化，不要固定成校服或单一裸身模板。示例不是白名单。\n"
            "- 加入 1-3 个与动作和地点有关系的配饰/道具，例如首饰、项圈、眼镜、发饰、手套、丝袜、"
            "家具、镜子、手机或成人用品；要说明它们在画面中的作用，而不是孤立罗列名词。\n"
            "- 认真设计构图：明确视觉焦点和主体位置，安排前景、中景、背景，交代景别、留白、裁切、"
            "透视和动作线，让画面有层次、平衡、可读性，不要把人物和道具堆在画面中央。\n"
            "- 明确视角与镜头：第一人称/旁观/自拍/镜面/俯视/低角度/侧后方等视角，配合近景/中景/"
            "全身、镜头角度、透视和焦段感；镜头形式也要随机变化。\n"
            "- 描述完整环境：地点、时间、天气、光源、色调、空气和氛围，让角色状态、服饰和场景彼此"
            "呼应；发挥想象力，创造新奇但合理的成人画面组合。\n\n"
            "输出格式：\n"
            "- 只输出 1 行自然语言，由 1-2 个完整中文句子组成；不要解释、编号、Markdown 或思考过程。\n"
            "- 不要输出标签清单或用空格堆砌词条；使用正常中文语序和标点，把人物之间的关系与动作写清楚。\n"
            "- 描述必须包含具体可视化细节，覆盖角色/人数、情色行为与状态、服饰与配饰、姿势互动、"
            "场景、构图视角和光线氛围，供后续在线检索和 Danbooru tag 生成使用。\n"
            "- 例子只是帮助理解维度，不是白名单；不要把输出限制在例子范围内。"
            f"{character_instruction}{selfie_extra}"
        )
        if self._recent_scenes:
            prompt += "\n\n以下是最近已生成过的内容，禁止与它们重复或相似：\n"
            prompt += "\n".join(self._recent_scenes)
        if rejected_candidates:
            prompt += "\n\n以下候选刚刚被判定为过于相似，禁止继续沿着这些方向小修小补：\n"
            prompt += "\n".join(rejected_candidates)
        return prompt

    @classmethod
    def _remember(cls, result: str) -> None:
        if not result:
            return
        cls._recent_scenes.append(result)
        if len(cls._recent_scenes) > cls._max_recent_scenes:
            cls._recent_scenes.pop(0)
