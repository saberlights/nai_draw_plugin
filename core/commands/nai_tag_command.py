# -*- coding: utf-8 -*-
"""
/打标 命令：引用回复图片进行打标，输出 NAI 可直接复制的 JSON。

约束：
- 仅支持“引用回复”触发（不从历史图片/最近图片兜底，避免误打标）
- BAD_TAG 仅用于负面提示词（瑕疵/不希望出现的元素），不输出“与图片相反”的否定tag
- 输出必须包含角色与作品字段（可为空数组，但字段必须存在）
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand

from ..utils.tagger_utils import (
    extract_picids,
    extract_image_base64_list,
    extract_image_base64_list_from_payload,
    find_reply_message_id,
    guess_image_format_from_base64,
    guess_image_format_from_path,
    normalize_output,
    parse_json_object,
    read_image_as_base64,
    strip_data_url,
)

logger = get_logger("nai_draw_plugin")


class NaiTaggerCommand(BaseCommand):
    command_name = "nai_tagger_command"
    command_description = "引用回复图片打标：/打标"
    command_pattern = r"(?:.*，说：\s*)?/打标$"

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        # 基础校验：必须是引用回复
        reply_message_id = self._extract_reply_message_id()
        if not reply_message_id:
            await self.send_text("❌ 请引用回复一张图片，再发送 `/打标`", storage_message=False)
            return False, "缺少引用回复", True

        # 优先：如果上游在 message.reply 里携带了被引用消息内容，直接从 message_segment 提取 image/emoji base64
        reply_msg = getattr(self.message, "reply", None)
        reply_seg = getattr(reply_msg, "message_segment", None) if reply_msg else None
        direct_images = extract_image_base64_list(reply_seg)
        if direct_images:
            raw_b64, forced_fmt = strip_data_url(direct_images[0])
            image_base64 = raw_b64
            image_format = forced_fmt or guess_image_format_from_base64(image_base64)
            return await self._tag_and_send_prompt(
                image_base64=image_base64,
                image_format=image_format,
                source_hint=f"reply:{reply_message_id}",
            )

        # 次优：从 raw_message / additional_config 等 payload 中尝试提取“引用消息内容”的 image/emoji base64
        payload_candidates = []
        # 有的适配器会把“被引用消息内容”塞进当前消息的 reply 段 data 里，因此也把 message_segment 纳入扫描范围
        payload_candidates.append(getattr(self.message, "message_segment", None))
        payload_candidates.append(getattr(self.message, "raw_message", None))
        mi = getattr(self.message, "message_info", None)
        if mi is not None:
            payload_candidates.append(getattr(mi, "additional_config", None))
            if hasattr(mi, "to_dict"):
                try:
                    payload_candidates.append(mi.to_dict())  # type: ignore[call-arg]
                except Exception:
                    pass

        for payload in payload_candidates:
            imgs = extract_image_base64_list_from_payload(payload)
            if imgs:
                raw_b64, forced_fmt = strip_data_url(imgs[0])
                image_base64 = raw_b64
                image_format = forced_fmt or guess_image_format_from_base64(image_base64)
                return await self._tag_and_send_prompt(
                    image_base64=image_base64,
                    image_format=image_format,
                    source_hint=f"payload:{reply_message_id}",
                )

        # 回退：从数据库取被引用消息（适用于普通 image，会有 picid；emoji 通常无法通过 DB 反查到原始 base64）
        chat_stream = getattr(self.message, "chat_stream", None)
        stream_id = getattr(chat_stream, "stream_id", None) if chat_stream else None
        if not stream_id:
            await self.send_text("❌ 无法获取会话 stream_id", storage_message=False)
            return False, "无法获取 stream_id", True

        db_msg = self._find_db_message(stream_id, reply_message_id)
        if not db_msg:
            await self.send_text(
                f"❌ 找不到被引用的原消息（引用ID: {reply_message_id}）\n"
                "可能原因：适配器未传递引用信息、消息未入库、或该消息不在本会话内。",
                storage_message=False,
            )
            return False, "找不到被引用消息", True

        picids = extract_picids(getattr(db_msg, "processed_plain_text", "") or "")
        if not picids:
            await self.send_text(
                "❌ 被引用的消息里没有检测到图片（picid）\n"
                "请确认你引用的是“图片消息本体”，不是转发/合并消息/链接卡片。\n"
                "另外：表情包/emoji 若适配器未在引用字段里带入原消息内容，插件也无法从数据库反查到原图（这是存储层限制）。",
                storage_message=False,
            )
            return False, "引用消息无图片", True

        # 默认只处理第一张图，避免刷屏/误触
        picid = picids[0]

        image_path = self._find_image_path_by_picid(picid)
        if not image_path:
            await self.send_text("❌ 找不到图片文件（可能图片缓存已被清理）", storage_message=False)
            return False, "找不到图片文件", True

        try:
            image_base64 = read_image_as_base64(image_path)
        except Exception as e:
            logger.error(f"{self.log_prefix} [打标] 读取图片失败: {e!r}", exc_info=True)
            await self.send_text("❌ 读取图片失败", storage_message=False)
            return False, "读取图片失败", True

        image_format = guess_image_format_from_path(image_path)

        return await self._tag_and_send_prompt(
            image_base64=image_base64,
            image_format=image_format,
            source_hint=f"picid:{picid}",
        )

    async def _tag_and_send_prompt(self, image_base64: str, image_format: str, source_hint: str) -> Tuple[bool, Optional[str], bool]:
        # 调用 VLM 打标（模型任务名由 config 配置）
        prompt = self._build_tagger_prompt()

        tagger_task = (self.get_config("tagger.model_task", "vlm") or "vlm").strip()
        temperature = self.get_config("tagger.temperature", 0.2)
        max_tokens = self.get_config("tagger.max_tokens", 800)
        custom_model = self.get_config("tagger.custom_model", None)

        # 如果启用了独立模型 custom_model，但用户未显式配置 tagger.max_tokens，则优先使用 custom_model.max_tokens
        try:
            tagger_cfg_raw = self.plugin_config.get("tagger", {}) if isinstance(self.plugin_config, dict) else {}
            max_tokens_explicit = isinstance(tagger_cfg_raw, dict) and "max_tokens" in tagger_cfg_raw
            temp_explicit = isinstance(tagger_cfg_raw, dict) and "temperature" in tagger_cfg_raw

            if isinstance(custom_model, dict):
                model_list = custom_model.get("model_list", [])
                model_list_ok = bool(model_list) if isinstance(model_list, list) else bool(model_list)
                if model_list_ok:
                    if not max_tokens_explicit and "max_tokens" in custom_model:
                        max_tokens = int(custom_model.get("max_tokens") or max_tokens)
                    if not temp_explicit and "temperature" in custom_model:
                        temperature = float(custom_model.get("temperature") or temperature)
        except Exception:
            pass

        try:
            content = await self._tag_image(
                task_name=tagger_task,
                custom_model=custom_model,
                prompt=prompt,
                image_base64=image_base64,
                image_format=image_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} [打标] 模型调用失败: {e!r}", exc_info=True)
            await self.send_text("❌ 打标失败（模型调用异常）", storage_message=False)
            return False, "模型调用异常", True

        logger.info(f"{self.log_prefix} [打标] 完成: source={source_hint}, model_task={tagger_task}, temperature={temperature}")

        obj = parse_json_object(content)
        if not obj:
            await self.send_text(
                "❌ 打标失败：模型输出未能解析为 JSON（可能被 max_tokens 截断或未按要求输出）。\n"
                "请调大 `[tagger] max_tokens` 或 `custom_model.max_tokens`，并确保模型支持图像输入。",
                storage_message=False,
            )
            return False, "模型输出不可解析", True

        normalized = normalize_output(obj)

        prompt_text = self._format_nai_prompt(
            character_tags=normalized.get("CHARACTER_TAG", []),
            work_tags=normalized.get("WORK_TAG", []),
            tags=normalized.get("TAG", []),
        )
        if not prompt_text.strip():
            await self.send_text(
                "⚠️ 打标结果为空（角色/作品/通用标签均为空）。\n"
                "可能原因：模型未识别到有效内容，或输出未按要求生成标签。",
                storage_message=False,
            )
            return True, "空结果", True

        await self.send_text(prompt_text, storage_message=False)
        return True, "打标成功", True

    def _extract_reply_message_id(self) -> Optional[str]:
        """
        尽量从多种来源提取“被引用消息”的 message_id。

        不同适配器/平台的“引用回复”字段差异较大，因此做多路兜底：
        1) message.reply（若上游填充）
        2) message.message_info.*（reply_to 等字段 / additional_config / to_dict）
        3) message.raw_message（适配器原始 payload）
        4) message.message_segment（reply 段）
        """

        def _clean(v) -> Optional[str]:
            if isinstance(v, int):
                v = str(v)
            if isinstance(v, str):
                s = v.strip()
                return s or None
            return None

        # 1) message.reply
        rep = getattr(self.message, "reply", None)
        rep_info = getattr(rep, "message_info", None) if rep else None
        mid = _clean(getattr(rep_info, "message_id", None)) if rep_info else None
        if mid:
            return mid

        # 2) message.message_info
        mi = getattr(self.message, "message_info", None)
        if mi:
            for attr in (
                "reply_to",
                "reply_to_message_id",
                "reply_message_id",
                "quote_message_id",
                "reply_id",
            ):
                mid = _clean(getattr(mi, attr, None))
                if mid:
                    return mid

            add_cfg = getattr(mi, "additional_config", None)
            if isinstance(add_cfg, dict):
                for k in (
                    "reply_to",
                    "reply_to_message_id",
                    "reply_message_id",
                    "quote_message_id",
                    "reply_id",
                    "message_id",
                ):
                    mid = _clean(add_cfg.get(k))
                    if mid:
                        return mid

            # 一些实现会把 reply 信息藏在 to_dict 结构里
            if hasattr(mi, "to_dict"):
                try:
                    d = mi.to_dict()  # type: ignore[call-arg]
                    mid = _clean(self._deep_find_first(d, keys={
                        "reply_to",
                        "reply_to_message_id",
                        "reply_message_id",
                        "quote_message_id",
                        "reply_id",
                        "message_id",
                    }))
                    if mid:
                        return mid
                except Exception:
                    pass

        # 3) raw_message
        raw = getattr(self.message, "raw_message", None)
        if isinstance(raw, dict):
            mid = _clean(self._deep_find_first(raw, keys={
                "reply_to",
                "reply_to_message_id",
                "reply_message_id",
                "quote_message_id",
                "reply_id",
                "message_id",
            }))
            if mid:
                return mid

        # 4) message_segment
        return find_reply_message_id(getattr(self.message, "message_segment", None))

    def _deep_find_first(self, obj, keys: set[str]):
        """在 dict/list 结构中递归查找给定 keys 的第一个值。"""
        try:
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj:
                        v = obj.get(k)
                        if v not in (None, ""):
                            return v
                for v in obj.values():
                    hit = self._deep_find_first(v, keys)
                    if hit not in (None, ""):
                        return hit
                return None
            if isinstance(obj, list):
                for it in obj:
                    hit = self._deep_find_first(it, keys)
                    if hit not in (None, ""):
                        return hit
                return None
            return None
        except Exception:
            return None

    def _build_tagger_prompt(self) -> str:
        # 你要求“提示词格式弄好看一点”，这里做结构化排版
        return (
            "你是图片内容打标器（Danbooru/NAI tag 体系）。请客观、详细地给出标签。\n"
            "\n"
            "要求：只输出 JSON；标签用英文小写下划线；必须包含角色与作品字段（可为空数组）。\n"
            "BAD_TAG 仅放 negative prompt（瑕疵/不希望出现的元素），不要写“与图片相反”的否定tag。\n"
            "PROMPT/NEGATIVE 必须是可直接复制给 NAI 的逗号分隔字符串。\n"
            "\n"
            "输出数量限制（为避免输出被截断）：\n"
            "- CHARACTER_TAG 最多 5 个\n"
            "- WORK_TAG 最多 5 个\n"
            "- TAG 最多 80 个（按重要性排序）\n"
            "- BAD_TAG 最多 40 个（按重要性排序）\n"
            "\n"
            "JSON 结构：\n"
            "{\n"
            '  "CHARACTER_TAG": ["..."],\n'
            '  "WORK_TAG": ["..."],\n'
            '  "TAG": ["..."],\n'
            '  "BAD_TAG": ["..."],\n'
            '  "PROMPT": "...",\n'
            '  "NEGATIVE": "..."\n'
            "}\n"
        )

    def _format_nai_prompt(self, character_tags, work_tags, tags) -> str:
        """
        输出一行可直接复制给 NAI 的 prompt：
        角色名称（作品名称）, tag1, tag2, ...
        """
        if not isinstance(character_tags, list):
            character_tags = []
        if not isinstance(work_tags, list):
            work_tags = []
        if not isinstance(tags, list):
            tags = []

        work = str(work_tags[0]).strip() if work_tags else ""

        head_parts = []
        for c in character_tags:
            cs = str(c).strip()
            if not cs:
                continue
            head_parts.append(f"{cs} ({work})" if work else cs)

        tail_parts = []
        for t in tags:
            ts = str(t).strip()
            if not ts:
                continue
            tail_parts.append(ts)

        def _dedup(items):
            seen = set()
            out = []
            for it in items:
                k = it.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        parts = _dedup(head_parts) + _dedup(tail_parts)
        return ", ".join(parts).strip()

    async def _tag_image(
        self,
        task_name: str,
        custom_model,
        prompt: str,
        image_base64: str,
        image_format: str,
        temperature,
        max_tokens,
    ) -> str:
        """
        使用指定任务配置调用 LLMRequest 的图像接口。

        注意：这里用的是 TaskConfig（model_config.model_task_config.<name>），由 config.toml 控制 name。
        """
        from ...legacy_llm_request import LegacyLLMRequest

        from src.config.config import model_config
        from src.config.model_configs import TaskConfig

        model_task_config = getattr(model_config, "model_task_config", None)
        if not model_task_config:
            raise RuntimeError("model_config.model_task_config 不存在")

        # 1) 优先使用 tagger.custom_model（完全独立于 model_task）
        task_cfg = None
        if isinstance(custom_model, dict):
            model_list = custom_model.get("model_list", [])
            if isinstance(model_list, str):
                model_list = [model_list]
            if isinstance(model_list, list) and model_list:
                task_cfg = TaskConfig(
                    model_list=model_list,
                    max_tokens=int(custom_model.get("max_tokens", 1024)),
                    temperature=float(custom_model.get("temperature", 0.2)),
                    slow_threshold=float(custom_model.get("slow_threshold", 30.0)),
                    selection_strategy="random",
                )
                logger.info(f"{self.log_prefix} [打标] 使用自定义模型: {model_list}")

        # 2) 回退到 model_task_config.<task_name>
        if task_cfg is None:
            task_cfg = getattr(model_task_config, task_name, None)
            if task_cfg is None:
                logger.warning(f"{self.log_prefix} [打标] 未找到任务配置 '{task_name}'，回退到 'vlm'")
                task_cfg = getattr(model_task_config, "vlm", None)
            if task_cfg is None:
                raise RuntimeError("未找到可用的 VLM 任务配置（vlm）")

        effective_max_tokens = self._cap_max_tokens(task_cfg, int(max_tokens or 0) or 1200)

        req = LegacyLLMRequest(model_set=task_cfg, request_type="nai_draw_plugin.tagger")
        content, _ = await req.generate_response_for_image(
            prompt=prompt,
            image_base64=image_base64,
            image_format=image_format,
            temperature=temperature,
            max_tokens=effective_max_tokens,
        )
        return content or ""

    def _cap_max_tokens(self, task_cfg, requested: int) -> int:
        """
        统一处理 max_tokens：
        - 用户在插件 config.toml 里可能会填一个非常大的数（例如 30000）
        - 但模型/提供商往往有自己的输出上限，超过会被裁剪并产生“超过最大 max_token 限制”的提示
        - 这里做两级裁剪，尽量避免误导与无意义的超大配置
        """
        # 1) 安全上限：打标 JSON 本身不需要巨量输出，避免无意义超大值
        safe_upper = 4096
        effective = min(max(requested, 1), safe_upper)

        # 2) 若 model_config 对具体模型标了 max_tokens，则按“模型最小上限”进一步裁剪
        try:
            from src.config.config import model_config

            model_list = getattr(task_cfg, "model_list", None) or []
            caps = []
            for name in model_list:
                try:
                    info = model_config.get_model_info(name)
                    if getattr(info, "max_tokens", None):
                        caps.append(int(info.max_tokens))  # type: ignore[arg-type]
                except Exception:
                    continue
            if caps:
                effective = min(effective, min(caps))
        except Exception:
            pass

        return effective

    def _find_db_message(self, stream_id: str, message_id: str):
        from src.common.database.database import db
        from src.common.message_repository import find_messages

        try:
            db.connect(reuse_if_open=True)
        except Exception:
            pass

        # 优先精确按 chat_id + message_id 查
        msgs = find_messages(
            message_filter={"chat_id": stream_id, "message_id": message_id},
            limit=1,
            limit_mode="latest",
        )
        return msgs[0] if msgs else None

    def _find_image_path_by_picid(self, picid: str) -> Optional[str]:
        try:
            from src.common.database.database import db
            from src.common.database.database_model import Images
        except Exception:
            return None

        try:
            db.connect(reuse_if_open=True)
        except Exception:
            pass

        img = Images.get_or_none(Images.image_id == picid)
        if not img:
            return None

        path = getattr(img, "path", None)
        if not path:
            return None
        return str(path)
