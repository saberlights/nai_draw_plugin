"""WD14 Tagger 客户端 - 多 Space 并发轮询负载均衡。

从 prompt_generator_plugin 移植，关键差异：gradio_client 改为软依赖，
缺失时构造不抛、调用时返回 WD14ClientError，让 PNG 元数据反推可独立运行。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    handle_file = None  # type: ignore


@dataclass
class WD14ClientError(Exception):
    """WD14 客户端错误"""
    message: str

    def __str__(self) -> str:
        return self.message


class WD14Client:
    """WD14 Tagger 客户端 - 多 Space 轮询负载均衡"""

    # 实测 1~2MB 大图在 HF Space 上推理耗时 16~23s（含队列等待 + 推理 + 上传），
    # 但 PixAI-Tagger-v0.9-ONNX 等冷启动后首次跑常常远超 35s，把单 Space 上限抬到
    # 120s 给冷启留余量；整体走完 3 个 Space 最坏 360s。
    SAFE_COMMAND_TIMEOUT = 360.0
    SAFE_SPACE_TIMEOUT_CAP = 120.0

    # 默认的 Space 列表（如果配置文件未提供）
    DEFAULT_SPACES = [
        {
            "name": "animetimm/dbv4-full-witha-playground",
            "type": "danbooru_v4",
            "api": "/_fn_submit",
        },
        {
            "name": "pixai-labs/pixai-tagger-demo",
            "type": "pixai",
            "api": "/predict_image",
        },
        {
            "name": "DraconicDragon/PixAI-Tagger-v0.9-ONNX",
            "type": "pixai_onnx",
            "api": "/run_inference",
        },
    ]

    _space_lock = None
    _space_warmup_done: Dict[str, bool] = {}

    AVAILABLE_MODELS = [
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
        "SmilingWolf/wd-swinv2-tagger-v3",
        "SmilingWolf/wd-convnext-tagger-v3",
        "SmilingWolf/wd-vit-tagger-v3",
        "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        "SmilingWolf/wd-v1-4-vit-tagger-v2",
    ]

    def __init__(
        self,
        model: str = "SmilingWolf/wd-eva02-large-tagger-v3",
        timeout: float = 60.0,
        algorithms: Optional[List[str]] = None,
        *,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        spaces_config: Optional[List[Dict[str, str]]] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self.model = model if model in self.AVAILABLE_MODELS else self.AVAILABLE_MODELS[0]
        normalized_timeout = float(timeout or self.SAFE_SPACE_TIMEOUT_CAP)
        self.timeout = max(3.0, min(normalized_timeout, self.SAFE_SPACE_TIMEOUT_CAP))
        self.command_timeout = max(self.timeout + 1.0, self.SAFE_COMMAND_TIMEOUT)
        self.algorithms = algorithms or ["Use WD Tagger"]
        self.logger = logging.getLogger(__name__)
        self.max_retries = max(1, int(max_retries or 1))
        self.retry_delay = max(0.5, float(retry_delay or 0.5))

        # 使用配置文件提供的 Space 列表，如果未提供则使用默认列表
        self.available_spaces = spaces_config if spaces_config else self.DEFAULT_SPACES
        self.logger.info(f"已配置 {len(self.available_spaces)} 个 Space 进行轮询")

        # 可选代理：空字符串当作 None；让 httpx 沿用环境变量
        proxy_value = (proxy or "").strip()
        self.proxy: Optional[str] = proxy_value or None
        if self.proxy:
            self.logger.info(f"WD14 Space 连接将通过代理: {self.proxy}")

        # gradio_client 为软依赖：缺失时仍允许实例化，调用 tag_image 时再抛错
        self._gradio_available = GRADIO_AVAILABLE
        if not GRADIO_AVAILABLE:
            self.logger.warning(
                "未检测到 gradio_client，WD14 兜底将不可用；"
                "如需启用请执行: uv add gradio_client 或 pip install gradio_client"
            )

        self.clients: Dict[str, Optional[Client]] = {}
        self.current_space_name: Optional[str] = None

        if WD14Client._space_lock is None:
            WD14Client._space_lock = threading.Lock()

    def _build_httpx_kwargs(self) -> Dict[str, Any]:
        """组装传给 gradio_client.Client 的 httpx 参数，按需带上代理。

        httpx 0.28+ 使用 ``proxy=...``；早期 0.x 用 ``proxies=...``。
        这里两种都试一下兼容，构造失败由调用方捕获。
        """
        kwargs: Dict[str, Any] = {"timeout": self.timeout}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        return kwargs

    def _get_or_create_client(self, space_name: str) -> Optional[Client]:
        if not GRADIO_AVAILABLE:
            return None

        if space_name in self.clients and self.clients[space_name] is not None:
            return self.clients[space_name]

        try:
            self.logger.info(f"正在连接到 Hugging Face Space: {space_name}")
            try:
                client = Client(
                    space_name,
                    verbose=False,
                    httpx_kwargs=self._build_httpx_kwargs(),
                )
            except TypeError as proxy_err:
                # 旧版 httpx 不识别 proxy= 时回退用 proxies={}
                if not self.proxy or "proxy" not in str(proxy_err):
                    raise
                self.logger.info("httpx 不支持 proxy=，回退使用 proxies= 写法")
                client = Client(
                    space_name,
                    verbose=False,
                    httpx_kwargs={"timeout": self.timeout, "proxies": self.proxy},
                )
            self.clients[space_name] = client
            self.logger.info(f"✓ Space 客户端已初始化: {space_name}")
            return client
        except Exception as e:
            self.logger.warning(f"✗ 连接 Space 失败 ({space_name}): {e}")
            self.clients[space_name] = None
            return None

    async def tag_image(
        self,
        image_base64: str,
        threshold: float = 0.35,
        character_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self._gradio_available:
            raise WD14ClientError("未安装 gradio_client，无法调用 WD14 在线 Space")
        if not self.available_spaces:
            raise WD14ClientError("未配置可用的 WD14 Space")

        errors: list[str] = []
        total = len(self.available_spaces)
        deadline = asyncio.get_event_loop().time() + self.command_timeout

        for idx, space_info in enumerate(self.available_spaces, start=1):
            space_name = str(space_info.get("name", "") or "")
            self.logger.info(f"依次尝试 Space [{idx}/{total}]: {space_name}")

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                errors.append(f"整体识别超时（>{self.command_timeout:.1f}s）")
                break

            try:
                result = await asyncio.wait_for(
                    self._tag_with_space(
                        image_base64=image_base64,
                        threshold=threshold,
                        character_threshold=character_threshold,
                        space_info=space_info,
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                error_text = f"Space 调用超时（>{remaining:.1f}s）: {space_name}"
                errors.append(error_text)
                self.logger.warning(error_text)
                # 整体已经超时，没必要继续轮询
                break
            except Exception as exc:
                error_text = f"{space_name} 失败: {exc}"
                errors.append(error_text)
                self.logger.warning(error_text)
                continue

            self.current_space_name = space_name
            self.logger.info(f"✓ 识别成功，使用 Space: {space_name}")
            return result

        last_error = errors[-1] if errors else "未知错误"
        self.logger.error(f"所有 Spaces 都无法使用，最后错误: {last_error}")
        raise WD14ClientError(f"所有 Spaces 都无法使用: {last_error}")

    async def _tag_with_space(
        self,
        image_base64: str,
        threshold: float,
        character_threshold: Optional[float],
        space_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        space_name = space_info["name"]
        space_type = space_info["type"]

        # gradio_client.Client(...) 内部用 httpx 同步抓 Space manifest；直接 await 之外的
        # 同步调用会冻结整个 event loop，把其它插件的 OBSERVE hook 全部怼到 timeout。
        # 这里始终把构造扔到默认线程池，避免 12s 量级的阻塞。
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(None, self._get_or_create_client, space_name)
        if not client:
            raise WD14ClientError(f"无法连接到 Space: {space_name}")

        image_bytes = self._decode_image(image_base64)
        self.logger.info(f"准备调用 Space API，图片大小: {len(image_bytes)} bytes")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            if character_threshold is None:
                character_threshold = 0.8

            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._predict_space_with_retry,
                    client,
                    tmp_path,
                    threshold,
                    character_threshold,
                    space_info,
                ),
                timeout=self.timeout,
            )

            processed = self._process_space_result(result, threshold, space_type)
            self.logger.info(f"Space 标注成功: {len(processed['tags'])} 个标签 (阈值: {threshold})")
            return processed
        except TimeoutError as exc:
            raise WD14ClientError(f"Space 调用超时（{self.timeout:.1f}s）: {space_name}") from exc
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def _predict_space_with_retry(
        self,
        client: Client,
        image_path: str,
        general_threshold: float,
        character_threshold: float,
        space_info: Dict[str, Any],
    ) -> Any:
        space_type = space_info["type"]
        api_name = space_info["api"]
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if space_type == "official":
                    return client.predict(
                        image=handle_file(image_path),
                        model_repo=self.model,
                        general_thresh=general_threshold,
                        general_mcut_enabled=False,
                        character_thresh=character_threshold,
                        character_mcut_enabled=False,
                        api_name=api_name,
                    )
                elif space_type == "pixai_onnx":
                    if WD14Client._space_lock is not None:
                        with WD14Client._space_lock:
                            if not WD14Client._space_warmup_done.get(space_info["name"], False):
                                try:
                                    client.predict(api_name="/init_app")
                                except Exception as e:
                                    self.logger.debug(f"Space 预热失败（忽略）: {e!r}")
                                WD14Client._space_warmup_done[space_info["name"]] = True
                    elif not WD14Client._space_warmup_done.get(space_info["name"], False):
                        try:
                            client.predict(api_name="/init_app")
                        except Exception as e:
                            self.logger.debug(f"Space 预热失败（忽略）: {e!r}")
                        WD14Client._space_warmup_done[space_info["name"]] = True

                    return client.predict(
                        image=handle_file(image_path),
                        gen_thresh=general_threshold,
                        char_thresh=character_threshold,
                        resolve_mapping=True,
                        api_name=api_name,
                    )
                elif space_type == "pixai":
                    return client.predict(
                        image=handle_file(image_path),
                        url="",
                        general_threshold=general_threshold,
                        character_threshold=character_threshold,
                        mode_val="threshold",
                        topk_general_val=25,
                        topk_character_val=10,
                        include_scores_val=True,
                        underscore_mode_val=False,
                        api_name=api_name,
                    )
                elif space_type == "danbooru_v4":
                    return client.predict(
                        image=handle_file(image_path),
                        _use_tag_thresholds=True,
                        param_2=general_threshold,
                        param_3=general_threshold,
                        param_4=character_threshold,
                        param_5=general_threshold,
                        api_name=api_name,
                    )
                else:
                    raise WD14ClientError(f"未知的 Space 类型: {space_type}")

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * attempt
                    self.logger.warning(f"Space API 调用失败（第 {attempt} 次）: {e}，等待 {wait_time:.1f}s 后重试")
                    time.sleep(wait_time)
                else:
                    raise WD14ClientError(f"API 调用失败: {e}")

        raise WD14ClientError(f"API 调用失败: {last_error}")

    def _process_space_result(
        self,
        result: Any,
        threshold: float,
        space_type: str,
    ) -> Dict[str, Any]:
        """处理和分类标签（支持不同 Space 的返回格式）

        Args:
            result: Gradio API 返回的结果
            threshold: 置信度阈值
            space_type: Space 类型

        Returns:
            分类后的标签字典
        """
        try:
            if space_type == "official":
                return self._process_official_result(result, threshold)
            elif space_type == "pixai_onnx":
                return self._process_pixai_onnx_result(result, threshold)
            elif space_type == "pixai":
                return self._process_pixai_result(result, threshold)
            elif space_type == "danbooru_v4":
                return self._process_danbooru_v4_result(result, threshold)
            else:
                raise WD14ClientError(f"未知的 Space 类型: {space_type}")

        except Exception as e:
            self.logger.error(f"处理标签结果失败: {e!r}", exc_info=True)
            return {
                "tags": [],
                "character_tags": [],
                "general_tags": [],
                "copyright_tags": [],
                "rating_tags": [],
            }

    def _process_official_result(
        self,
        result: Any,
        threshold: float
    ) -> Dict[str, Any]:
        """处理官方 WD14 Space 的返回结果"""
        # 新 API 返回 4 个值：output_string, rating, output_characters, output_tags
        output_string = result[0] if len(result) > 0 else ""
        rating = result[1] if len(result) > 1 else {}
        output_characters = result[2] if len(result) > 2 else {}
        output_tags = result[3] if len(result) > 3 else {}

        self.logger.debug(f"Official API 返回: output_tags keys={output_tags.keys() if isinstance(output_tags, dict) else 'not dict'}")

        all_tags = []
        character_tags = []
        general_tags = []
        rating_tags = []

        # 处理角色标签
        if isinstance(output_characters, dict) and 'confidences' in output_characters:
            for item in output_characters['confidences']:
                label = item.get('label', '')
                score = item.get('confidence', 0.0)
                if label and score > 0:
                    character_tags.append({
                        "label": label,
                        "score": score
                    })
                    all_tags.append({
                        "label": label,
                        "score": score
                    })

        # 处理通用标签
        if isinstance(output_tags, dict) and 'confidences' in output_tags:
            for item in output_tags['confidences']:
                label = item.get('label', '')
                score = item.get('confidence', 0.0)
                if label and score >= threshold:
                    general_tags.append({
                        "label": label,
                        "score": score
                    })
                    all_tags.append({
                        "label": label,
                        "score": score
                    })

        # 处理评级标签
        if isinstance(rating, dict) and 'confidences' in rating:
            for item in rating['confidences']:
                label = item.get('label', '')
                score = item.get('confidence', 0.0)
                if label and score > 0:
                    rating_tags.append({
                        "label": f"rating:{label}",
                        "score": score
                    })

        # 按分数排序
        all_tags.sort(key=lambda x: x["score"], reverse=True)

        self.logger.info(f"解析成功: {len(all_tags)} 个标签 (角色: {len(character_tags)}, 通用: {len(general_tags)})")

        return {
            "tags": all_tags,
            "character_tags": character_tags,
            "general_tags": general_tags,
            "copyright_tags": [],  # 新 API 不单独返回版权标签
            "rating_tags": rating_tags,
        }

    def _process_pixai_result(
        self,
        result: Any,
        threshold: float
    ) -> Dict[str, Any]:
        """处理 PixAI Tagger Space 的返回结果"""
        # PixAI API 返回 6 个值：general_tags, character_tags, ip_tags, combined_tags, timings, raw_json
        general_tags_str = result[0] if len(result) > 0 else ""
        character_tags_str = result[1] if len(result) > 1 else ""
        ip_tags_str = result[2] if len(result) > 2 else ""
        combined_tags_str = result[3] if len(result) > 3 else ""
        raw_json = result[5] if len(result) > 5 else {}

        self.logger.debug(f"PixAI API 返回长度: {len(result)}")

        all_tags = []
        character_tags = []
        general_tags = []
        copyright_tags = []
        rating_tags = []

        # 从 raw_json 中提取带分数的标签
        if isinstance(raw_json, dict):
            # 处理通用标签 (feature)
            feature_scores = raw_json.get('feature_scores', {})
            if feature_scores:
                for label, score in feature_scores.items():
                    if score >= threshold:
                        general_tags.append({
                            "label": label,
                            "score": float(score)
                        })
                        all_tags.append({
                            "label": label,
                            "score": float(score)
                        })

            # 处理角色标签 (character)
            character_scores = raw_json.get('character_scores', {})
            if character_scores:
                for label, score in character_scores.items():
                    if score > 0:
                        character_tags.append({
                            "label": label,
                            "score": float(score)
                        })
                        all_tags.append({
                            "label": label,
                            "score": float(score)
                        })

            # 处理 IP/版权标签
            ip_scores = raw_json.get('ip_scores', {})
            if ip_scores:
                for label, score in ip_scores.items():
                    if score > 0:
                        copyright_tags.append({
                            "label": label,
                            "score": float(score)
                        })

        # 按分数排序
        all_tags.sort(key=lambda x: x["score"], reverse=True)

        self.logger.info(f"解析成功: {len(all_tags)} 个标签 (角色: {len(character_tags)}, 通用: {len(general_tags)}, IP: {len(copyright_tags)})")

        return {
            "tags": all_tags,
            "character_tags": character_tags,
            "general_tags": general_tags,
            "copyright_tags": copyright_tags,
            "rating_tags": rating_tags,
        }

    def _process_pixai_onnx_result(
        self,
        result: Any,
        threshold: float,
    ) -> Dict[str, Any]:
        """处理 PixAI-Tagger-ONNX Space 的返回结果

        该 Space 的 `/run_inference` 返回 6 个元素，其中包含角色/通用概率字典：
        - result[3]: 角色概率 dict(confidences=[{label, confidence}, ...])
        - result[4]: 通用概率 dict(confidences=[{label, confidence}, ...])
        """
        character_prob = result[3] if isinstance(result, (list, tuple)) and len(result) > 3 else {}
        general_prob = result[4] if isinstance(result, (list, tuple)) and len(result) > 4 else {}

        all_tags: List[Dict[str, Any]] = []
        character_tags: List[Dict[str, Any]] = []
        general_tags: List[Dict[str, Any]] = []

        def _extract_confidences(prob_dict: Any) -> List[Dict[str, Any]]:
            if not isinstance(prob_dict, dict):
                return []
            confidences = prob_dict.get("confidences")
            if not isinstance(confidences, list):
                return []
            return [item for item in confidences if isinstance(item, dict)]

        for item in _extract_confidences(character_prob):
            label = item.get("label")
            score = item.get("confidence")
            if not isinstance(label, str) or not label:
                continue
            if score is None:
                continue
            try:
                score_f = float(score)
            except Exception:
                continue
            if score_f > 0:
                tag = {"label": label, "score": score_f}
                character_tags.append(tag)
                all_tags.append(tag)

        for item in _extract_confidences(general_prob):
            label = item.get("label")
            score = item.get("confidence")
            if not isinstance(label, str) or not label:
                continue
            if score is None:
                continue
            try:
                score_f = float(score)
            except Exception:
                continue
            if score_f >= threshold:
                tag = {"label": label, "score": score_f}
                general_tags.append(tag)
                all_tags.append(tag)

        all_tags.sort(key=lambda x: x["score"], reverse=True)
        self.logger.info(f"解析成功: {len(all_tags)} 个标签 (角色: {len(character_tags)}, 通用: {len(general_tags)})")

        return {
            "tags": all_tags,
            "character_tags": character_tags,
            "general_tags": general_tags,
            "copyright_tags": [],
            "rating_tags": [],
        }

    def _process_danbooru_v4_result(
        self,
        result: Any,
        threshold: float
    ) -> Dict[str, Any]:
        """处理 Danbooru V4 Space 的返回结果"""
        # Danbooru V4 API 返回 5 个值：output_string, general_dict, artist_dict, character_dict, rating_dict
        output_string = result[0] if len(result) > 0 else ""
        general_dict = result[1] if len(result) > 1 else {}
        artist_dict = result[2] if len(result) > 2 else {}
        character_dict = result[3] if len(result) > 3 else {}
        rating_dict = result[4] if len(result) > 4 else {}

        self.logger.debug(f"Danbooru V4 API 返回: general_dict type={type(general_dict)}")

        all_tags = []
        character_tags = []
        general_tags = []
        artist_tags = []  # Artist 标签（作为版权标签的一部分）
        rating_tags = []

        # 处理通用标签
        if isinstance(general_dict, dict) and 'confidences' in general_dict:
            for item in general_dict['confidences']:
                label = item.get('label', '')
                score = item.get('confidence', 0.0)
                if label and score >= threshold:
                    general_tags.append({
                        "label": label,
                        "score": score
                    })
                    all_tags.append({
                        "label": label,
                        "score": score
                    })

        # 处理角色标签
        if isinstance(character_dict, dict) and 'confidences' in character_dict:
            for item in character_dict['confidences']:
                label = item.get('label', '')
                score = item.get('confidence', 0.0)
                if label and score > 0:
                    character_tags.append({
                        "label": label,
                        "score": score
                    })
                    all_tags.append({
                        "label": label,
                        "score": score
                    })

        # 处理艺术家标签
        if isinstance(artist_dict, dict) and 'confidences' in artist_dict:
            for item in artist_dict['confidences']:
                label = item.get('label', '')
                score = item.get('confidence', 0.0)
                if label and score > 0:
                    artist_tags.append({
                        "label": label,
                        "score": score
                    })

        # 处理评级标签
        if isinstance(rating_dict, dict) and 'confidences' in rating_dict:
            for item in rating_dict['confidences']:
                label = item.get('label', '')
                score = item.get('confidence', 0.0)
                if label and score > 0:
                    rating_tags.append({
                        "label": f"rating:{label}",
                        "score": score
                    })

        # 按分数排序
        all_tags.sort(key=lambda x: x["score"], reverse=True)

        self.logger.info(f"解析成功: {len(all_tags)} 个标签 (角色: {len(character_tags)}, 通用: {len(general_tags)}, 艺术家: {len(artist_tags)})")

        return {
            "tags": all_tags,
            "character_tags": character_tags,
            "general_tags": general_tags,
            "copyright_tags": artist_tags,  # 艺术家标签作为版权信息
            "rating_tags": rating_tags,
        }

    def _decode_image(self, image_base64: str) -> bytes:
        """解码 base64 图片"""
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            return base64.b64decode(image_base64)
        except Exception as e:
            raise WD14ClientError(f"Base64 解码失败: {e}")

    def format_tags_as_string(
        self,
        tag_data: Dict[str, Any],
        format: str = "danbooru"
    ) -> str:
        """将标签格式化为字符串

        Args:
            tag_data: 标签数据字典
            format: 格式类型 ("danbooru" 或 "natural")

        Returns:
            格式化后的标签字符串
        """
        tags = tag_data.get("tags", [])

        if format == "danbooru":
            # Danbooru 格式：tag1, tag2, tag3
            return ", ".join([tag["label"] for tag in tags])
        else:
            # 自然语言格式，包含置信度
            lines = []
            for tag in tags[:20]:  # 最多显示 20 个
                label = tag["label"]
                score = tag["score"]
                lines.append(f"{label} ({score:.2%})")
            return "\n".join(lines)
