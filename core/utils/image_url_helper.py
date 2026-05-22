from typing import List, Optional, Tuple

import base64
import os
import time
import uuid

from src.common.logger import get_logger

logger = get_logger("nai_draw_plugin")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMAGE_OUTPUT_DIR = os.path.join(_BASE_DIR, "generated_images")
os.makedirs(_IMAGE_OUTPUT_DIR, exist_ok=True)
_MAX_FILE_AGE_SECONDS = 30 * 60  # 30分钟保留时间
_MAX_FILE_COUNT = 80  # 限制缓存文件数量
_CLEANUP_INTERVAL_SECONDS = 5 * 60  # 每5分钟尝试清理一次
_last_cleanup_ts = 0.0


def _detect_image_type_from_header(image_base64: str) -> Optional[str]:
    if not image_base64.startswith("data:image/"):
        return None

    header = image_base64.split(",", 1)[0]
    image_type = header[len("data:image/") :].split(";", 1)[0].lower()

    if image_type == "jpg":
        return "jpeg"

    if image_type in {"bmp", "gif", "jpeg", "png", "svg+xml", "tiff", "webp"}:
        return image_type

    return None


def _detect_image_type_from_bytes(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"

    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "gif"

    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"

    if image_bytes.startswith(b"BM"):
        return "bmp"

    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"

    stripped_bytes = image_bytes.lstrip()
    if stripped_bytes.startswith(b"<?xml") or stripped_bytes.startswith(b"<svg"):
        return "svg+xml"

    return "png"


def _maybe_cleanup_generated_files():
    global _last_cleanup_ts
    now = time.time()
    if now - _last_cleanup_ts < _CLEANUP_INTERVAL_SECONDS:
        return
    _last_cleanup_ts = now
    _cleanup_generated_files(now)


def _cleanup_generated_files(now: float):
    try:
        entries: List[Tuple[str, float]] = []
        for entry in os.scandir(_IMAGE_OUTPUT_DIR):
            if entry.is_file():
                try:
                    stat = entry.stat()
                    entries.append((entry.path, stat.st_mtime))
                except FileNotFoundError:
                    continue
    except FileNotFoundError:
        return

    removed = 0
    remaining: List[Tuple[str, float]] = []

    for path, mtime in entries:
        if now - mtime > _MAX_FILE_AGE_SECONDS:
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"[nai_pic] 删除过期图片失败: {e}")
        else:
            remaining.append((path, mtime))

    if len(remaining) > _MAX_FILE_COUNT:
        overflow = len(remaining) - _MAX_FILE_COUNT
        remaining.sort(key=lambda item: item[1])
        for path, _ in remaining[:overflow]:
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"[nai_pic] 删除多余图片失败: {e}")

    if removed:
        logger.debug(f"[nai_pic] 已清理 {removed} 个临时图片文件")


def save_base64_image_to_file(image_base64: str) -> Optional[str]:
    """将Base64图片保存为本地文件并返回文件路径"""
    _maybe_cleanup_generated_files()
    try:
        data = image_base64.split(",", 1)[1] if image_base64.startswith("data:image") else image_base64
        image_bytes = base64.b64decode(data)
    except Exception as e:
        logger.error(f"[nai_pic] 解码Base64图片失败: {e}")
        return None

    image_type = _detect_image_type_from_header(image_base64) or _detect_image_type_from_bytes(image_bytes)
    if image_type == "svg+xml":
        image_type = "svg"
    extension = "jpg" if image_type == "jpeg" else image_type
    file_name = f"nai_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.{extension}"
    file_path = os.path.join(_IMAGE_OUTPUT_DIR, file_name)

    try:
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        logger.debug(f"[nai_pic] 图片已保存: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"[nai_pic] 保存图片失败: {e}")
        return None
