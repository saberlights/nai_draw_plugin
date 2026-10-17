# -*- coding: utf-8 -*-
"""视觉模型反推 Adapter 的单元测试。

覆盖三件容易悄悄坏掉的事：图片格式嗅探（写死 png 会让 JPEG 请求被网关拒收）、
模型输出清洗（残留中文会让 NewAPI §8 直接 400）、失败暴露（静默返回空串会让
用户对着"反推失败：未知原因"查半天）。
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.services.llm_image_captioner import (
    ImageCaptionError,
    MaiBotImageCaptioner,
    detect_image_format,
    normalize_caption_text,
)


# ── 图片格式嗅探 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 32, "jpeg"),
        (b"GIF89a" + b"\x00" * 32, "gif"),
        (b"GIF87a" + b"\x00" * 32, "gif"),
        (b"BM" + b"\x00" * 32, "bmp"),
        (b"RIFF" + b"\x24\x00\x00\x00" + b"WEBP" + b"\x00" * 32, "webp"),
    ],
)
def test_detect_image_format_reads_magic_bytes(payload: bytes, expected: str) -> None:
    assert detect_image_format(payload) == expected


def test_detect_image_format_rejects_riff_that_is_not_webp() -> None:
    """RIFF 头也可能是 WAV：只有第 8~12 字节是 WEBP 才算图片。"""
    wav = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"\x00" * 32
    with pytest.raises(ImageCaptionError):
        detect_image_format(wav)


def test_detect_image_format_rejects_unknown_bytes() -> None:
    with pytest.raises(ImageCaptionError) as excinfo:
        detect_image_format(b"not an image at all" * 4)
    assert "无法识别的图片格式" in str(excinfo.value)


def test_detect_image_format_rejects_empty() -> None:
    with pytest.raises(ImageCaptionError):
        detect_image_format(b"")


# ── 输出清洗 ──────────────────────────────────────────────────────────────


def test_normalize_caption_strips_chinese_residue() -> None:
    """模型偶尔会夹一句中文；NewAPI §8 对含 CJK 的 prompt 一律 400，必须清掉。"""
    raw = "A girl stands by the window 她站在窗边 wearing a white dress"
    cleaned = normalize_caption_text(raw)

    assert "她站在窗边" not in cleaned
    assert "A girl stands by the window" in cleaned
    assert "wearing a white dress" in cleaned


def test_normalize_caption_strips_fullwidth_punctuation() -> None:
    cleaned = normalize_caption_text("She smiles，her eyes half closed。")

    assert "，" not in cleaned
    assert "。" not in cleaned
    assert "her eyes half closed" in cleaned


def test_normalize_caption_extracts_code_block() -> None:
    """模型把描述当"代码"输出时要取代码块内容，不能把 ``` 一起送进 NAI。"""
    raw = (
        "Here is the description:\n"
        "```\n"
        "A young woman leans against a brick wall, her right hand in her pocket.\n"
        "```"
    )
    cleaned = normalize_caption_text(raw)

    assert cleaned.startswith("A young woman leans against a brick wall")
    assert "```" not in cleaned
    assert "Here is the description" not in cleaned


def test_normalize_caption_keeps_plain_prose_intact() -> None:
    raw = "A girl sits on the railing, her left hand resting on the cat's back."
    assert normalize_caption_text(raw) == raw


def test_normalize_caption_returns_empty_for_blank() -> None:
    assert normalize_caption_text("   \n  ") == ""
    assert normalize_caption_text("") == ""


# ── 调用链与失败暴露 ──────────────────────────────────────────────────────


def _make_png(width: int = 512, height: int = 512) -> bytes:
    """造一张真 PNG：captioner 会把图交给 PIL 做最小边长归一化，假字节过不了。"""
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), (120, 80, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


_PNG_LARGE = _make_png()


class _RecordingCaptioner(MaiBotImageCaptioner):
    """把真实的模型请求换成脚本返回，保留格式嗅探与清洗这两段真实逻辑。"""

    def __init__(self, *, response: str = "", raise_exc: Exception | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._response = response
        self._raise_exc = raise_exc
        self.requests: list[dict[str, str]] = []

    async def _request_caption(self, *, prompt: str, image_base64: str, image_format: str) -> str:
        self.requests.append(
            {"prompt": prompt, "image_base64": image_base64, "image_format": image_format}
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def test_caption_passes_sniffed_format_to_the_model() -> None:
    captioner = _RecordingCaptioner(response="A cat sleeps on a windowsill.")

    result = asyncio.run(captioner.caption(image_bytes=_PNG_LARGE, sfw=True))

    assert result == "A cat sleeps on a windowsill."
    assert captioner.requests[0]["image_format"] == "png"


def test_caption_prompt_switches_on_sfw_flag() -> None:
    """SFW 与 NSFW 必须发不同的 system_override，否则 NSFW 图会被模型拒答。"""
    sfw_captioner = _RecordingCaptioner(response="A girl waves.")
    nsfw_captioner = _RecordingCaptioner(response="A girl waves.")

    asyncio.run(sfw_captioner.caption(image_bytes=_PNG_LARGE, sfw=True))
    asyncio.run(nsfw_captioner.caption(image_bytes=_PNG_LARGE, sfw=False))

    sfw_prompt = sfw_captioner.requests[0]["prompt"]
    nsfw_prompt = nsfw_captioner.requests[0]["prompt"]

    assert sfw_prompt != nsfw_prompt
    assert "non-sexual" in sfw_prompt
    assert "disable_ethics_and_boundaries" in nsfw_prompt


def test_caption_uses_custom_template_verbatim() -> None:
    captioner = _RecordingCaptioner(
        response="ok",
        prompt_template="Describe this image in English, exhaustively.",
    )

    asyncio.run(captioner.caption(image_bytes=_PNG_LARGE, sfw=True))

    assert captioner.requests[0]["prompt"] == "Describe this image in English, exhaustively."


def test_caption_raises_when_model_returns_blank() -> None:
    """模型拒答通常表现为空串；返回 "" 会让上层报"未知原因"，必须在这里就炸。"""
    captioner = _RecordingCaptioner(response="   ")

    with pytest.raises(ImageCaptionError) as excinfo:
        asyncio.run(captioner.caption(image_bytes=_PNG_LARGE, sfw=True))
    assert "空描述" in str(excinfo.value)


def test_caption_raises_when_model_returns_only_chinese() -> None:
    """整段中文清洗后就没内容了，同样必须当失败处理而不是送一个空 prompt 给 NAI。"""
    captioner = _RecordingCaptioner(response="一个女孩坐在窗台上，猫趴在她的膝盖上。")

    with pytest.raises(ImageCaptionError):
        asyncio.run(captioner.caption(image_bytes=_PNG_LARGE, sfw=True))


def test_caption_wraps_timeout_with_configured_seconds() -> None:
    async def _never_returns(**_kwargs):
        await asyncio.sleep(10)

    captioner = _RecordingCaptioner(response="ignored", timeout=0.05)
    captioner._request_caption = _never_returns  # type: ignore[method-assign]

    with pytest.raises(ImageCaptionError) as excinfo:
        asyncio.run(captioner.caption(image_bytes=_PNG_LARGE, sfw=True))
    assert "超时" in str(excinfo.value)


def test_caption_propagates_model_error() -> None:
    captioner = _RecordingCaptioner(raise_exc=RuntimeError("上游 429"))

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(captioner.caption(image_bytes=_PNG_LARGE, sfw=True))
    assert "上游 429" in str(excinfo.value)


def test_caption_rejects_non_image_bytes_before_calling_model() -> None:
    captioner = _RecordingCaptioner(response="should not be reached")

    with pytest.raises(ImageCaptionError):
        asyncio.run(captioner.caption(image_bytes=b"plain text payload" * 8, sfw=True))
    assert captioner.requests == []
