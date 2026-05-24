import asyncio
import base64
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from requests.exceptions import ProxyError

from src.common.logger import get_logger

logger = get_logger("nai_draw_plugin")


# 文档要求图片尺寸的硬上限（参考 §6）
_MAX_SIZE_PORTRAIT: Tuple[int, int] = (832, 1216)
_MAX_SIZE_LANDSCAPE: Tuple[int, int] = (1216, 832)
_MAX_SIZE_SQUARE: Tuple[int, int] = (1024, 1024)

# 网关临时故障的可重试状态码（§15）
_RETRYABLE_STATUS_CODES: frozenset = frozenset({429, 500, 502, 503, 504})

# 多角色（NewAPI §7）能力检测：仅 nai-diffusion-4 系列稳定支持 `characters[]` / `use_coords`
_MULTI_CHARACTER_MODEL_KEYWORDS: Tuple[str, ...] = ("nai-diffusion-4",)

# 多角色 position 字面量 [A-E][1-5]，与 prompt_output_parser 保持一致
_POSITION_GRID_RE = re.compile(r"^[A-E][1-5]$")

# 响应正文里 markdown 形式的图片 data URI
_CHAT_IMAGE_DATA_URI_PATTERN = re.compile(
    r"!\[[^\]]*]\((data:image/(?P<format>[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=]+))\)"
)

# 末尾 seed 注释，例如 <!-- seeds:[123456789,null] -->
_SEEDS_COMMENT_PATTERN = re.compile(r"<!--\s*seeds:\s*(\[[^\]]*])\s*-->")


class NaiWebClient:
    """NewAPI（OpenAI 兼容）网关客户端。

    端点固定走 ``POST {base_url}/v1/chat/completions``；真正的绘图参数被序列化成
    JSON 字符串塞到 ``messages[0].content`` 中，响应里图片以 markdown
    ``![image_0](data:image/png;base64,...)`` 形式回传。
    """

    _DEFAULT_REQUEST_TIMEOUT = 600.0
    _DEFAULT_MAX_TOKENS = 100000
    _MAX_RESPONSE_RETRY_ATTEMPTS = 2
    _MAX_TRANSPORT_RETRY_ATTEMPTS = 3
    _RETRY_DELAY_SECONDS = 1.5
    _PROTECTION_RETRY_DELAY_SECONDS = 6.0

    def __init__(self, action_instance):
        self.action = action_instance
        self.log_prefix = action_instance.log_prefix
        self.session: requests.Session = self._create_session(trust_env=True)
        self.direct_session: requests.Session = self._create_session(trust_env=False)
        self._auto_proxy_direct_only = False

    def close(self) -> None:
        """关闭底层 HTTP Session，供插件热重载时清理资源。"""
        for session in (self.session, self.direct_session):
            try:
                session.close()
            except Exception:
                continue

    # ========== Session 与配置解析 ==========

    @staticmethod
    def _create_session(trust_env: bool) -> requests.Session:
        session = requests.Session()
        session.trust_env = trust_env
        return session

    def _get_session(self, trust_env: bool) -> requests.Session:
        return self.session if trust_env else self.direct_session

    @staticmethod
    def _resolve_proxy_mode(model_config: Dict[str, Any]) -> str:
        value = model_config.get("nai_proxy_mode") or model_config.get("proxy_mode") or "auto"
        return str(value).strip().lower() or "auto"

    @classmethod
    def _resolve_request_timeout(cls, model_config: Dict[str, Any]) -> float:
        raw_timeout = model_config.get("nai_request_timeout")
        if raw_timeout in (None, ""):
            return cls._DEFAULT_REQUEST_TIMEOUT
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            logger.warning(
                f"(NewAPI) 非法超时配置 {raw_timeout!r}，回退到默认 {cls._DEFAULT_REQUEST_TIMEOUT:.1f}s"
            )
            return cls._DEFAULT_REQUEST_TIMEOUT
        if timeout <= 0:
            return cls._DEFAULT_REQUEST_TIMEOUT
        return timeout

    @classmethod
    def _resolve_max_tokens(cls, model_config: Dict[str, Any]) -> int:
        raw_value = model_config.get("nai_max_tokens")
        if raw_value in (None, ""):
            return cls._DEFAULT_MAX_TOKENS
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                f"(NewAPI) 非法 max_tokens 配置 {raw_value!r}，回退到默认 {cls._DEFAULT_MAX_TOKENS}"
            )
            return cls._DEFAULT_MAX_TOKENS
        if value <= 0:
            return cls._DEFAULT_MAX_TOKENS
        return value

    # ========== Prompt / Size 组装 ==========

    @staticmethod
    def _merge_artist_into_prompt(prompt: str, artist_prompt: str, custom_prompt_add: str = "") -> str:
        """按 质量词 → 画师串 → 用户词 顺序拼接，与 NovelAI 推荐顺序保持一致。

        - 质量词（custom_prompt_add）置首，建立整体画质基调
        - 画师串紧随其后，定义风格
        - 用户词最后，承载本次具体描述
        - 去重：用户词若已以画师串开头，跳过画师以免重复
        """
        normalized_user = str(prompt or "").strip()
        normalized_artist = str(artist_prompt or "").strip().strip(",")
        normalized_quality = str(custom_prompt_add or "").strip().strip(",")

        if normalized_artist and normalized_user:
            lowered_user = normalized_user.lower()
            lowered_artist = normalized_artist.lower()
            if lowered_user == lowered_artist or lowered_user.startswith(f"{lowered_artist},"):
                normalized_artist = ""

        parts = [p for p in (normalized_quality, normalized_artist, normalized_user) if p]
        return ", ".join(parts)

    @staticmethod
    def _resolve_size(size_value: Any) -> Tuple[int, int]:
        """把人类可读的尺寸描述（竖图/横图/方图/portrait/...）归一化成 (w, h)。"""
        if isinstance(size_value, (list, tuple)) and len(size_value) == 2:
            try:
                return int(size_value[0]), int(size_value[1])
            except (TypeError, ValueError):
                return _MAX_SIZE_PORTRAIT

        text = str(size_value or "").strip()
        if not text:
            return _MAX_SIZE_PORTRAIT
        size_alias = {
            "竖图": _MAX_SIZE_PORTRAIT,
            "竖": _MAX_SIZE_PORTRAIT,
            "v": _MAX_SIZE_PORTRAIT,
            "portrait": _MAX_SIZE_PORTRAIT,
            "横图": _MAX_SIZE_LANDSCAPE,
            "横": _MAX_SIZE_LANDSCAPE,
            "h": _MAX_SIZE_LANDSCAPE,
            "landscape": _MAX_SIZE_LANDSCAPE,
            "方图": _MAX_SIZE_SQUARE,
            "方": _MAX_SIZE_SQUARE,
            "s": _MAX_SIZE_SQUARE,
            "square": _MAX_SIZE_SQUARE,
        }
        mapped = size_alias.get(text.lower())
        if mapped is not None:
            return mapped
        # 显式 widthxheight 形式
        if "x" in text.lower():
            try:
                width_text, height_text = text.lower().split("x", 1)
                return int(width_text.strip()), int(height_text.strip())
            except (ValueError, AttributeError):
                pass
        return _MAX_SIZE_PORTRAIT

    @staticmethod
    def _resolve_model_name(model_config: Dict[str, Any]) -> str:
        model_name = str(model_config.get("default_model") or "").strip()
        return model_name or "nai-diffusion-4-5-full"

    # ========== 请求构造 ==========

    @classmethod
    def _build_inner_draw_params(
        cls,
        prompt: str,
        model_config: Dict[str, Any],
        size: Optional[str],
        characters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """组装 messages[0].content 内层 JSON（NewAPI 文档 §5 / §7）。

        Args:
            characters: 已规范化的多角色列表。每项含 ``prompt`` / ``negative_prompt`` / ``position``。
                位置非空时启用 ``use_coords=true``，否则交由后端自动布局。仅当模型支持时注入；
                由 :meth:`_filter_characters_for_model` 在入口处嗅探，本函数信任入参已通过过滤。
        """
        custom_prompt_add = str(model_config.get("custom_prompt_add") or "").strip()
        artist_prompt = str(
            model_config.get("nai_artist_prompt") or model_config.get("artist_prompt") or ""
        ).strip()
        negative_prompt = str(model_config.get("negative_prompt_add") or "").strip()

        full_prompt = cls._merge_artist_into_prompt(prompt, artist_prompt, custom_prompt_add)

        size_value = model_config.get("nai_size") or size or model_config.get("default_size") or "竖图"
        width, height = cls._resolve_size(size_value)
        size_payload: List[int] = [width, height]

        seed_value = model_config.get("seed")
        try:
            seed_int = int(seed_value) if seed_value not in (None, "") else -1
        except (TypeError, ValueError):
            seed_int = -1
        inner: Dict[str, Any] = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "size": size_payload,
            "steps": int(model_config.get("num_inference_steps", 23)),
            "scale": float(model_config.get("guidance_scale", 5.0)),
            "sampler": str(model_config.get("sampler") or "k_euler_ancestral"),
            "n_samples": 1,
            "image_format": str(model_config.get("image_format", "png")),
        }
        if seed_int >= 0:
            inner["seed"] = seed_int

        # 多角色字段（文档 §7）。characters 已经过模型嗅探与规范化，这里直接写入。
        normalized_characters = cls._normalize_characters_for_inner(characters)
        if normalized_characters:
            inner["characters"] = normalized_characters
            inner["use_coords"] = all(bool(item.get("position")) for item in normalized_characters)
            inner["use_order"] = True

        # 质量增强参数（文档 §5 表外，透传由 NewAPI 决定是否吃下）
        if bool(model_config.get("quality_toggle", True)):
            inner["qualityToggle"] = True

        # SMEA 系列参数（同上，透传给上游 NovelAI 决定）
        if bool(model_config.get("auto_smea", False)):
            inner["autoSmea"] = True
        if "sm" in model_config and model_config["sm"] not in (None, ""):
            inner["sm"] = bool(model_config["sm"])
        if "sm_dyn" in model_config and model_config["sm_dyn"] not in (None, ""):
            inner["sm_dyn"] = bool(model_config["sm_dyn"])

        # 多样性增强（透传到上游 NovelAI，文档 §5 表外字段）
        if bool(model_config.get("variety_boost", False)):
            inner["variety_boost"] = True

        # nai_extra_params 透传（用户在 config.toml 显式声明的扩展字段）
        extra_params = model_config.get("nai_extra_params") or {}
        if isinstance(extra_params, dict):
            for key, value in extra_params.items():
                if value not in (None, ""):
                    inner[key] = value

        return inner

    @staticmethod
    def _normalize_characters_for_inner(
        characters: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, str]]:
        """把上层传入的角色列表归一为 inner JSON 直接可用的形态。

        - 丢弃 prompt 为空的项
        - position 不匹配 ``[A-E][1-5]`` 时落为 ``""``（由调用方决定 ``use_coords``）
        - 当任一项缺 position 时统一不发送 position 字段，避免文档约束冲突
        """
        if not characters:
            return []

        cleaned: List[Dict[str, str]] = []
        for item in characters:
            if not isinstance(item, dict):
                continue
            char_prompt = str(item.get("prompt") or "").strip()
            if not char_prompt:
                continue

            char_negative = str(item.get("negative_prompt") or "").strip()
            raw_position = str(item.get("position") or "").strip().upper()
            position = raw_position if _POSITION_GRID_RE.match(raw_position) else ""

            cleaned.append(
                {
                    "prompt": char_prompt,
                    "negative_prompt": char_negative,
                    "position": position,
                }
            )

        if len(cleaned) < 2:
            return []

        # 部分角色缺 position 时，把所有 position 一并清空，让后端自动布局
        if any(not item["position"] for item in cleaned):
            for item in cleaned:
                item["position"] = ""

        # 输出体里 position 为空的项需要剔除 position 键，文档对该字段只接受字符串
        result: List[Dict[str, str]] = []
        for item in cleaned:
            entry: Dict[str, str] = {"prompt": item["prompt"]}
            if item["negative_prompt"]:
                entry["negative_prompt"] = item["negative_prompt"]
            if item["position"]:
                entry["position"] = item["position"]
            result.append(entry)
        return result

    @classmethod
    def _filter_characters_for_model(
        cls,
        model_name: str,
        characters: Optional[List[Dict[str, Any]]],
        log_prefix: str = "",
    ) -> Optional[List[Dict[str, Any]]]:
        """按模型嗅探决定是否启用多角色字段。

        非 ``nai-diffusion-4*`` 系列降级为单字符串路径，并打一条 warning 让用户知道
        发生了静默降级。

        Returns:
            支持时返回原列表（可能为 None / 空）；不支持时返回 None。
        """
        if not characters:
            return None
        lowered = str(model_name or "").lower()
        if any(keyword in lowered for keyword in _MULTI_CHARACTER_MODEL_KEYWORDS):
            return characters
        logger.warning(
            f"{log_prefix} (NewAPI) 模型 {model_name!r} 不在多角色支持列表内，"
            f"已自动降级为单 prompt 路径，characters({len(characters)} 项) 已忽略"
        )
        return None

    @staticmethod
    def _validate_inner_payload(model_name: str, inner: Dict[str, Any]) -> Optional[str]:
        """对内层参数做最关键的硬约束（参考 NewAPI 文档 §5 / §15）。

        返回非空字符串表示拒绝原因；返回 None 表示通过。
        故意不做语种校验：让 NewAPI 自己回 400 反而给出更精确的错误。
        """
        prompt = str(inner.get("prompt") or "").strip()
        if not prompt:
            return "prompt 为空，无法发起绘图请求"

        normalized_model_name = str(model_name or "").strip()
        if "inpainting" in normalized_model_name.lower():
            return f"NewAPI 当前不支持 inpainting 模型：{normalized_model_name}"

        size_value = inner.get("size")
        if not (
            isinstance(size_value, list)
            and len(size_value) == 2
            and all(isinstance(value, int) for value in size_value)
        ):
            return f"size 必须是 [width, height] 整数数组，当前收到 {size_value!r}"
        width, height = size_value
        if width <= 0 or height <= 0:
            return f"size 宽高必须大于 0，当前收到 {size_value!r}"
        if width % 64 != 0 or height % 64 != 0:
            return f"size 宽高必须是 64 的倍数，当前收到 {size_value!r}"
        if width == height:
            if width > _MAX_SIZE_SQUARE[0]:
                return f"方图最大尺寸为 {_MAX_SIZE_SQUARE}，当前收到 {size_value!r}"
        elif height > width:
            if width > _MAX_SIZE_PORTRAIT[0] or height > _MAX_SIZE_PORTRAIT[1]:
                return f"竖图最大尺寸为 {_MAX_SIZE_PORTRAIT}，当前收到 {size_value!r}"
        else:
            if width > _MAX_SIZE_LANDSCAPE[0] or height > _MAX_SIZE_LANDSCAPE[1]:
                return f"横图最大尺寸为 {_MAX_SIZE_LANDSCAPE}，当前收到 {size_value!r}"

        if int(inner.get("n_samples", 1)) != 1:
            return f"NewAPI 当前只允许 n_samples=1，配置中收到 {inner.get('n_samples')}"

        characters_value = inner.get("characters")
        if characters_value is not None:
            if not isinstance(characters_value, list) or not characters_value:
                return f"characters 必须是非空数组，当前收到 {characters_value!r}"
            for index, item in enumerate(characters_value):
                if not isinstance(item, dict):
                    return f"characters[{index}] 必须是对象，当前收到 {item!r}"
                char_prompt = str(item.get("prompt") or "").strip()
                if not char_prompt:
                    return f"characters[{index}].prompt 为空"
                position = item.get("position")
                if position is not None:
                    if not isinstance(position, str) or not _POSITION_GRID_RE.match(position):
                        return (
                            f"characters[{index}].position 必须是 [A-E][1-5] 字符串，"
                            f"当前收到 {position!r}"
                        )

            use_coords_value = inner.get("use_coords")
            if use_coords_value is not None and not isinstance(use_coords_value, bool):
                return f"use_coords 必须是布尔值，当前收到 {use_coords_value!r}"

        return None

    @staticmethod
    def _build_request_body(
        model_name: str,
        inner: Dict[str, Any],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """构造 OpenAI 兼容外层 body。"""
        return {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(inner, ensure_ascii=False),
                }
            ],
            "stream": False,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _build_request_headers(api_key: str) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        token = (api_key or "").strip()
        if token:
            if token.lower().startswith("bearer "):
                token = token.split(" ", 1)[1].strip()
            headers["Authorization"] = f"Bearer {token}"
        return headers

    # ========== 主入口 ==========

    async def generate_image(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: Optional[str] = None,
        input_image_base64: Optional[str] = None,
        characters: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, str]:
        """调用 NewAPI 兼容网关生成图片。

        成功时返回 (True, base64_string)；失败时返回 (False, 错误描述)。

        Args:
            characters: 可选的多角色 payload，每项含 ``prompt`` / ``negative_prompt`` / ``position``。
                仅当模型支持（nai-diffusion-4 系列）时生效；其它模型会自动降级为单字符串路径。
        """
        if input_image_base64:
            logger.warning(f"{self.log_prefix} (NewAPI) 当前网关不支持图生图")
            return False, "NewAPI 当前不支持图生图"

        try:
            base_url = str(model_config.get("base_url") or "").rstrip("/")
            if not base_url:
                return False, "base_url 未配置"

            endpoint = str(model_config.get("nai_endpoint") or "/v1/chat/completions").strip()
            if not endpoint.startswith("/"):
                endpoint = f"/{endpoint}"
            # NewAPI 协议强制走 chat/completions
            if not endpoint.lower().endswith("/chat/completions"):
                logger.warning(
                    f"{self.log_prefix} (NewAPI) nai_endpoint={endpoint} 不是 chat/completions，"
                    "已强制覆盖为 /v1/chat/completions"
                )
                endpoint = "/v1/chat/completions"
            url = f"{base_url}{endpoint}"

            model_name = self._resolve_model_name(model_config)
            effective_characters = self._filter_characters_for_model(
                model_name, characters, log_prefix=self.log_prefix
            )
            inner_params = self._build_inner_draw_params(
                prompt, model_config, size, characters=effective_characters
            )
            reject_reason = self._validate_inner_payload(model_name, inner_params)
            if reject_reason:
                logger.error(f"{self.log_prefix} (NewAPI) 请求被前置校验拒绝: {reject_reason}")
                return False, reject_reason

            max_tokens = self._resolve_max_tokens(model_config)
            body = self._build_request_body(model_name, inner_params, max_tokens)
            headers = self._build_request_headers(model_config.get("api_key") or "")
            proxy_mode = self._resolve_proxy_mode(model_config)
            request_timeout = self._resolve_request_timeout(model_config)

            size_value = inner_params.get("size")
            character_count = len(inner_params.get("characters") or [])
            logger.info(
                f"{self.log_prefix} (NewAPI) 请求URL: {url}, model={model_name}, "
                f"size={size_value}, steps={inner_params['steps']}, "
                f"scale={inner_params['scale']}, sampler={inner_params['sampler']}, "
                f"seed={inner_params.get('seed')}, characters={character_count}, "
                f"use_coords={inner_params.get('use_coords')}, max_tokens={max_tokens}, "
                f"proxy={proxy_mode}, timeout={request_timeout:.1f}s"
            )
            logger.debug(
                f"{self.log_prefix} (NewAPI) inner_params keys: "
                f"{sorted(inner_params.keys())}"
            )

            response = await self._send_request_with_retry(
                url=url,
                body=body,
                headers=headers,
                proxy_mode=proxy_mode,
                request_timeout=request_timeout,
            )
            return self._parse_response(response)

        except requests.RequestException as exc:
            logger.error(f"{self.log_prefix} (NewAPI) 网络异常: {exc}")
            return False, self._format_request_exception(exc)
        except Exception as exc:
            logger.error(f"{self.log_prefix} (NewAPI) 请求异常: {exc!r}", exc_info=True)
            return False, f"NewAPI 请求失败: {str(exc)[:160]}"

    # ========== HTTP 发送与重试 ==========

    async def _send_request_with_retry(
        self,
        url: str,
        body: Dict[str, Any],
        headers: Dict[str, str],
        proxy_mode: str,
        request_timeout: float,
    ) -> requests.Response:
        """对可重试 HTTP 故障做受控重试。"""
        response: Optional[requests.Response] = None
        for attempt in range(1, self._MAX_TRANSPORT_RETRY_ATTEMPTS + 1):
            try:
                response = await asyncio.to_thread(
                    self._send_request,
                    url,
                    body,
                    headers,
                    proxy_mode,
                    request_timeout,
                )
            except requests.RequestException as exc:
                if (
                    attempt >= self._MAX_TRANSPORT_RETRY_ATTEMPTS
                    or self._is_proxy_related_exception(exc)
                    or not self._is_retryable_request_exception(exc)
                ):
                    raise
                retry_delay = self._get_retry_delay_seconds(attempt)
                logger.warning(
                    f"{self.log_prefix} (NewAPI) 第{attempt}次请求遇到可重试网络异常: {exc}; "
                    f"{retry_delay:.1f}s 后重试"
                )
                await asyncio.sleep(retry_delay)
                continue

            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt >= self._MAX_RESPONSE_RETRY_ATTEMPTS
            ):
                return response

            retry_delay = self._get_retry_delay_seconds(attempt)
            if response.status_code == 429:
                retry_delay = max(retry_delay, self._PROTECTION_RETRY_DELAY_SECONDS)
            logger.warning(
                f"{self.log_prefix} (NewAPI) 第{attempt}次请求返回可重试 HTTP {response.status_code}，"
                f"{retry_delay:.1f}s 后重试"
            )
            await asyncio.sleep(retry_delay)

        assert response is not None
        return response

    def _send_request(
        self,
        url: str,
        body: Dict[str, Any],
        headers: Dict[str, str],
        proxy_mode: str,
        request_timeout: float,
    ) -> requests.Response:
        """同步发送 POST 请求，必要时自动切换代理。"""
        if proxy_mode == "direct":
            return self._request_with_session(False, url, body, headers, request_timeout)
        if proxy_mode == "inherit":
            return self._request_with_session(True, url, body, headers, request_timeout)
        if self._auto_proxy_direct_only:
            return self._request_with_session(False, url, body, headers, request_timeout)

        try:
            return self._request_with_session(True, url, body, headers, request_timeout)
        except requests.RequestException as exc:
            if not self._is_proxy_related_exception(exc):
                raise
            self._auto_proxy_direct_only = True
            logger.warning(
                f"{self.log_prefix} (NewAPI) 代理连接失败，自动回退直连: {exc}"
            )
            return self._request_with_session(False, url, body, headers, request_timeout)

    def _request_with_session(
        self,
        trust_env: bool,
        url: str,
        body: Dict[str, Any],
        headers: Dict[str, str],
        request_timeout: float,
    ) -> requests.Response:
        session = self._get_session(trust_env=trust_env)
        response = session.post(
            url=url,
            json=body,
            headers=headers,
            timeout=request_timeout,
            allow_redirects=False,
        )
        return self._follow_post_redirects(session, response, body, headers, request_timeout)

    def _follow_post_redirects(
        self,
        session: requests.Session,
        response: requests.Response,
        body: Dict[str, Any],
        headers: Dict[str, str],
        request_timeout: float,
        max_redirects: int = 3,
    ) -> requests.Response:
        """手动跟随重定向，确保 POST 不会被 requests 自动改成 GET。"""
        current = response
        redirect_count = 0
        while (
            redirect_count < max_redirects
            and current.status_code in {301, 302, 303, 307, 308}
            and current.headers.get("Location")
        ):
            location = str(current.headers.get("Location") or "").strip()
            redirect_url = urljoin(current.url, location)
            logger.warning(
                f"{self.log_prefix} (NewAPI) 检测到重定向，保持 POST 继续请求: "
                f"{current.url} -> {redirect_url}"
            )
            current.close()
            current = session.post(
                url=redirect_url,
                json=body,
                headers=headers,
                timeout=request_timeout,
                allow_redirects=False,
            )
            redirect_count += 1
        return current

    # ========== 响应解析 ==========

    def _parse_response(self, response: requests.Response) -> Tuple[bool, str]:
        if response.status_code != 200:
            return False, self._extract_error_message(response)

        try:
            data = response.json()
        except Exception:
            logger.error(
                f"{self.log_prefix} (NewAPI) 响应不是 JSON: {response.text[:300]}"
            )
            return False, f"NewAPI 返回了非 JSON 响应: {response.text[:160]}"

        if not isinstance(data, dict):
            return False, "NewAPI 响应数据格式错误"

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return False, self._extract_json_error_message(data) or "NewAPI 未返回 choices"

        content_text = self._extract_message_content(choices[0])
        if not content_text:
            return False, self._extract_json_error_message(data) or "NewAPI 未返回 message.content"

        seeds = self._extract_seeds(content_text)
        if seeds:
            logger.info(
                f"{self.log_prefix} (NewAPI) 返回 seeds={seeds}, "
                f"usage={data.get('usage')}"
            )
        else:
            logger.info(f"{self.log_prefix} (NewAPI) 返回 usage={data.get('usage')}")

        matches = list(_CHAT_IMAGE_DATA_URI_PATTERN.finditer(content_text))
        if matches:
            if len(matches) > 1:
                logger.warning(
                    f"{self.log_prefix} (NewAPI) 响应包含 {len(matches)} 张图片，"
                    "只取第一张"
                )
            image_base64 = matches[0].group("data")
            try:
                image_bytes = base64.b64decode(image_base64, validate=False)
                logger.info(
                    f"{self.log_prefix} (NewAPI) 图片生成成功，大小 {len(image_bytes)} bytes"
                )
            except Exception:
                logger.error(
                    f"{self.log_prefix} (NewAPI) 解码返回的 base64 失败"
                )
                return False, "NewAPI 返回的 base64 数据无法解码"
            return True, image_base64

        return False, self._extract_text_error_message(content_text) or "NewAPI 响应中没有图片"

    @staticmethod
    def _extract_message_content(choice: Any) -> str:
        """从 chat/completions 的 choice 里提取文本内容。"""
        if not isinstance(choice, dict):
            return ""
        message = choice.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        # OpenAI 兼容协议中 content 偶尔为 list，逐段拼接
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _extract_seeds(content: str) -> List[Any]:
        match = _SEEDS_COMMENT_PATTERN.search(content or "")
        if not match:
            return []
        try:
            data = json.loads(match.group(1))
        except (ValueError, TypeError):
            return []
        return data if isinstance(data, list) else []

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            data = response.json()
        except Exception:
            logger.error(
                f"{self.log_prefix} (NewAPI) HTTP错误 {response.status_code}: "
                f"{response.text[:200]}"
            )
            text = (response.text or "").strip()
            return f"HTTP {response.status_code}: {text[:160]}" if text else f"HTTP {response.status_code}"
        message = self._extract_json_error_message(data)
        return f"HTTP {response.status_code}: {message}" if message else f"HTTP {response.status_code}"

    @staticmethod
    def _extract_json_error_message(data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        error = data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            code = str(error.get("code") or "").strip()
            if message and code:
                return f"{message} (code={code})"
            return message or code
        if isinstance(error, str) and error.strip():
            return error.strip()
        for key in ("message", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @classmethod
    def _extract_text_error_message(cls, text: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return ""
        try:
            data = json.loads(normalized)
        except (ValueError, TypeError):
            return ""
        return cls._extract_json_error_message(data)

    # ========== 异常识别 ==========

    @staticmethod
    def _is_proxy_related_exception(exc: requests.RequestException) -> bool:
        if isinstance(exc, ProxyError):
            return True
        current: Optional[BaseException] = exc
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            message = str(current).lower()
            if "proxy" in message or "407" in message:
                return True
            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        return False

    @staticmethod
    def _is_retryable_request_exception(exc: requests.RequestException) -> bool:
        if isinstance(
            exc,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ),
        ):
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "connection broken",
                "remote end closed",
                "connection reset",
                "read timed out",
                "chunked",
            )
        )

    @classmethod
    def _get_retry_delay_seconds(cls, attempt: int) -> float:
        return min(cls._RETRY_DELAY_SECONDS * (2 ** max(attempt - 1, 0)), 6.0)

    @classmethod
    def _format_request_exception(cls, exc: requests.RequestException) -> str:
        if isinstance(exc, requests.exceptions.Timeout):
            return "NewAPI 请求超时，请稍后重试"
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "NewAPI 连接失败，请检查网关地址或网络"
        if cls._is_retryable_request_exception(exc):
            return "NewAPI 连接不稳定，请稍后重试"
        return f"网络请求失败: {str(exc)[:160]}"
