# -*- coding: utf-8 -*-
"""
图片生成服务

统一处理图片生成、响应处理、发送逻辑

主要功能：
- 调用 NAI API 生成图片
- 处理 API 响应（URL/Base64）
- 发送图片
- 自拍模式处理
"""
import time
from typing import Tuple, Optional, Any, Callable, Awaitable

from src.common.logger import get_logger

from ..rules.selfie_rules import merge_selfie_prompt
from ..constants import NAI_PIC_IMAGE_DISPLAY_MARKER

logger = get_logger("nai_draw_plugin")


class ImageGenerationService:
    """
    图片生成服务

    使用方式：
        service = ImageGenerationService(self, self.log_prefix)
        success, message = await service.generate_and_send(
            prompt="1girl, hatsune miku",
            model_config=model_config,
            size="832x1216",
            on_success=lambda: self._schedule_auto_recall()
        )
    """

    def __init__(self, component: Any, log_prefix: str = ""):
        """
        Args:
            component: 调用方组件（Command 或 Action），用于发送消息和获取 API 客户端
            log_prefix: 日志前缀
        """
        self.component = component
        self.log_prefix = log_prefix
        self._api_client = None

    @property
    def api_client(self):
        """延迟初始化 API 客户端"""
        if self._api_client is None:
            # 尝试从新路径导入
            try:
                from ..clients.nai_web_client import NaiWebClient
            except ImportError:
                # 兼容旧路径
                from ..nai_web_client import NaiWebClient
            self._api_client = NaiWebClient(self.component)
        return self._api_client

    async def generate_and_send(
        self,
        prompt: str,
        model_config: dict,
        size: str,
        on_success: Optional[Callable[[], Awaitable[None]]] = None
    ) -> Tuple[bool, str]:
        """
        生成图片并发送

        Args:
            prompt: 提示词
            model_config: 模型配置
            size: 图片尺寸
            on_success: 成功后的异步回调（用于自动撤回等）

        Returns:
            (success, message) 元组
        """
        # 验证配置
        if not model_config:
            return False, "模型配置无效"

        if not model_config.get("base_url"):
            return False, "base_url 未配置"

        # 调用 API 生成图片
        try:
            success, result = await self.api_client.generate_image(
                prompt=prompt,
                model_config=model_config,
                size=size
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} 图片生成失败: {e!r}", exc_info=True)
            return False, f"生成失败: {str(e)[:100]}"

        if not success:
            return False, f"生成失败: {result}"

        # 处理响应
        image_data = self._process_response(result)
        if not image_data:
            return False, "API返回数据格式错误"

        # 发送图片
        send_time = time.time()
        send_success = await self._send_image(image_data)

        if send_success:
            # 记录发送时间戳（供自动撤回使用）
            if hasattr(self.component, '_last_send_timestamp'):
                self.component._last_send_timestamp = send_time

            # 执行成功回调
            if on_success:
                try:
                    await on_success()
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 成功回调执行失败: {e}")

            return True, "图片已成功生成并发送"
        else:
            return False, "图片发送失败"

    def _process_response(self, result: str) -> Optional[str]:
        """
        处理 API 响应，返回可用的图片数据

        支持格式：
        - URL（http:// 或 https://）
        - Base64（PNG/JPEG/WEBP/GIF）
        - Data URI（data:image/...）
        """
        if not result:
            return None

        # URL 格式
        if result.startswith(("http://", "https://")):
            return result

        # Base64 格式（检测常见图片格式的 Base64 前缀）
        # PNG: iVBORw, JPEG: /9j/, WEBP: UklGR, GIF: R0lGOD
        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            return result

        # Data URI 格式
        if "," in result and result.startswith("data:image"):
            return result.split(",", 1)[1]

        # 尝试作为 Base64 处理
        return result

    async def _send_image(self, image_data: str) -> bool:
        """
        发送图片

        Args:
            image_data: 图片数据（URL 或 Base64）

        Returns:
            是否发送成功
        """
        try:
            # URL 格式
            if image_data.startswith(("http://", "https://")):
                try:
                    return await self.component.send_custom(
                        "imageurl",
                        image_data,
                        display_message=NAI_PIC_IMAGE_DISPLAY_MARKER,
                    )
                except TypeError:
                    return await self.component.send_custom("imageurl", image_data)

            # Base64 格式
            if image_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                # 尝试保存为文件再发送
                image_path = self._save_base64_to_file(image_data)
                if image_path:
                    try:
                        return await self.component.send_custom(
                            "imageurl",
                            f"file://{image_path}",
                            display_message=NAI_PIC_IMAGE_DISPLAY_MARKER,
                        )
                    except TypeError:
                        return await self.component.send_custom("imageurl", f"file://{image_path}")
                else:
                    # 回退为直接发送 Base64
                    logger.warning(f"{self.log_prefix} 图片保存失败，回退为Base64发送")
                    try:
                        return await self.component.send_custom(
                            "image",
                            image_data,
                            display_message=NAI_PIC_IMAGE_DISPLAY_MARKER,
                        )
                    except TypeError:
                        return await self.component.send_image(image_data)

            # 其他格式尝试直接发送
            try:
                return await self.component.send_custom(
                    "image",
                    image_data,
                    display_message=NAI_PIC_IMAGE_DISPLAY_MARKER,
                )
            except TypeError:
                return await self.component.send_image(image_data)

        except Exception as e:
            logger.error(f"{self.log_prefix} 图片发送失败: {e!r}")
            return False

    def _save_base64_to_file(self, base64_data: str) -> Optional[str]:
        """保存 Base64 图片到文件"""
        try:
            # 尝试从新路径导入
            try:
                from ..utils.image_url_helper import save_base64_image_to_file
            except ImportError:
                # 兼容旧路径
                from ..image_url_helper import save_base64_image_to_file
            return save_base64_image_to_file(base64_data)
        except Exception as e:
            logger.warning(f"{self.log_prefix} 保存图片失败: {e}")
            return None

    def process_selfie_prompt(self, prompt: str, model_config: dict) -> str:
        """
        处理自拍模式的提示词，使用智能合并避免冲突

        Args:
            prompt: 原始提示词
            model_config: 模型配置

        Returns:
            添加了自拍提示的提示词
        """
        selfie_prompt_add = model_config.get("selfie_prompt_add", "")
        if selfie_prompt_add:
            # 使用智能合并函数，避免标签冲突
            return merge_selfie_prompt(prompt, selfie_prompt_add)
        return prompt
