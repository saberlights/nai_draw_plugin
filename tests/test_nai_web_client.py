# -*- coding: utf-8 -*-
import asyncio
import base64
from http.client import IncompleteRead
import os
import sys
import types
import unittest
import importlib
from unittest.mock import patch


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIBOT_ROOT = os.path.abspath(os.path.join(PLUGIN_DIR, "../.."))

if MAIBOT_ROOT not in sys.path:
    sys.path.insert(0, MAIBOT_ROOT)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _get_logger(_name=None):
    return _DummyLogger()


dummy_logger_module.get_logger = _get_logger
sys.modules["src.common.logger"] = dummy_logger_module

core_package = types.ModuleType("core")
core_package.__path__ = [os.path.join(PLUGIN_DIR, "core")]
sys.modules.setdefault("core", core_package)

core_clients_package = types.ModuleType("core.clients")
core_clients_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "clients")]
sys.modules.setdefault("core.clients", core_clients_package)

nai_web_client_module = importlib.import_module("core.clients.nai_web_client")
NaiWebClient = nai_web_client_module.NaiWebClient


class _DummyAction:
    log_prefix = "test_nai_pic"


class _DummyQQAction(_DummyAction):
    @staticmethod
    def _get_target_platform():
        return "qq"


_ONE_PIXEL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGioAAAAASUVORK5CYII="
)


class _DummyResponse:
    def __init__(
        self,
        *,
        status_code=200,
        headers=None,
        text="{}",
        json_data=None,
        content=b"",
    ):
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.text = text
        self._json_data = json_data if json_data is not None else {"url": "https://example.com/result.png"}
        self.content = content

    def json(self):
        return self._json_data


class NaiWebClientTest(unittest.TestCase):
    def test_generate_image_should_not_return_generation_url_directly_for_qq_by_default(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyQQAction()
        client.log_prefix = _DummyQQAction.log_prefix
        client._auto_proxy_direct_only = False
        captured = {}

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            captured["params"] = dict(params)
            return _DummyResponse()

        client._send_request = fake_send_request  # type: ignore[method-assign]

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "api_key": "token-123",
                    "nai_artist_prompt": "artist:test_a",
                    "nai_size": "竖图",
                },
            ))

        self.assertTrue(success)
        self.assertEqual(result, "https://example.com/result.png")
        self.assertEqual(captured["params"]["artist"], "artist:test_a")
        self.assertEqual(captured["params"]["tag"], "artist:test_a, 1girl, smile")

    def test_generate_image_should_not_send_artist_when_only_custom_prompt_add_exists(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        captured = {}

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            captured["url"] = url
            captured["params"] = dict(params)
            captured["proxy_mode"] = proxy_mode
            captured["request_timeout"] = request_timeout
            captured["request_headers"] = dict(request_headers or {})
            return _DummyResponse()

        client._send_request = fake_send_request  # type: ignore[method-assign]

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "custom_prompt_add": "masterpiece, best quality",
                    "nai_artist_prompt": "",
                },
            ))

        self.assertTrue(success)
        self.assertEqual(result, "https://example.com/result.png")
        self.assertEqual(
            captured["params"]["tag"],
            "1girl, smile, masterpiece, best quality",
        )
        self.assertNotIn("artist", captured["params"])
        self.assertEqual(captured["proxy_mode"], "auto")
        self.assertEqual(captured["request_timeout"], NaiWebClient._DEFAULT_REQUEST_TIMEOUT)
        self.assertEqual(captured["request_headers"]["Referer"], "https://example.com/")

    def test_generate_image_should_use_configured_request_timeout(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        captured = {}

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            captured["request_timeout"] = request_timeout
            return _DummyResponse()

        client._send_request = fake_send_request  # type: ignore[method-assign]

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_request_timeout": 321,
                },
            ))

        self.assertTrue(success)
        self.assertEqual(result, "https://example.com/result.png")
        self.assertEqual(captured["request_timeout"], 321.0)

    def test_generate_image_should_merge_artist_prompt_into_tag_for_web_compatibility(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        captured = {}

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            captured["params"] = dict(params)
            return _DummyResponse()

        client._send_request = fake_send_request  # type: ignore[method-assign]

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_artist_prompt": "artist:test_a, artist:test_b",
                },
            ))

        self.assertTrue(success)
        self.assertEqual(result, "https://example.com/result.png")
        self.assertEqual(captured["params"]["artist"], "artist:test_a, artist:test_b")
        self.assertEqual(
            captured["params"]["tag"],
            "artist:test_a, artist:test_b, 1girl, smile",
        )

    def test_generate_image_should_not_retry_generation_request_when_gateway_timeout(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False
        calls = []
        responses = [
            _DummyResponse(
                status_code=504,
                headers={"content-type": "text/html"},
                text='<html class="no-js" lang="en-US"><head><title>网站请求超时</title></head></html>',
            ),
        ]

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            calls.append((url, dict(params), proxy_mode, request_timeout))
            return responses.pop(0)

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_direct_url_fallback": False,
                    "nai_nocache": 1,
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成服务网关超时（HTTP 504），请稍后重试")
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(call[3] == NaiWebClient._DEFAULT_REQUEST_TIMEOUT for call in calls))

    def test_generate_image_should_not_retry_generation_request_multiple_times_when_incomplete_read(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False
        calls = []
        remaining = [
            nai_web_client_module.requests.exceptions.ChunkedEncodingError(
                "Connection broken",
                IncompleteRead(b"x" * 1048576, 486510),
            ),
            nai_web_client_module.requests.exceptions.ChunkedEncodingError(
                "Connection broken",
                IncompleteRead(b"x" * 1048576, 486510),
            ),
        ]

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            calls.append((url, dict(params), proxy_mode, request_timeout))
            current = remaining.pop(0)
            if isinstance(current, Exception):
                raise current
            return current

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_direct_url_fallback": False,
                    "nai_nocache": 1,
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成结果传输中断，请稍后重试")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "https://example.com/generate")
        self.assertIn("/generate?", calls[1][0])

    def test_generate_image_should_recover_complete_png_from_incomplete_read_partial(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False
        calls = []

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            calls.append((url, dict(params), proxy_mode, request_timeout))
            raise nai_web_client_module.requests.exceptions.ChunkedEncodingError(
                "Connection broken",
                IncompleteRead(_ONE_PIXEL_PNG_BYTES, 123),
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                },
            ))

        self.assertTrue(success)
        self.assertEqual(result, base64.b64encode(_ONE_PIXEL_PNG_BYTES).decode("utf-8"))
        self.assertEqual(len(calls), 1)

    def test_generate_image_should_followup_fetch_when_transport_breaks_even_if_nocache_enabled(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False
        calls = []
        followup_calls = []

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            calls.append((url, dict(params), proxy_mode, request_timeout))
            raise nai_web_client_module.requests.exceptions.ConnectionError(
                "Connection broken",
                IncompleteRead(b"x" * 1048576, 486510),
            )

        async def fake_download(generation_url, model_config, request_headers):
            followup_calls.append((generation_url, dict(model_config), dict(request_headers)))
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]
        client._download_generated_image_as_base64 = fake_download  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_nocache": 1,
                },
            ))

        self.assertTrue(success)
        self.assertEqual(result, "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(followup_calls), 1)
        self.assertIn("nocache=1", followup_calls[0][0])

    def test_generate_image_should_return_transport_error_when_incomplete_read_persists(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            raise nai_web_client_module.requests.exceptions.ConnectionError(
                "Connection broken",
                IncompleteRead(b"x" * 1048576, 486510),
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_request_timeout": 321,
                    "nai_nocache": 1,
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成结果传输中断，请稍后重试")

    def test_generate_image_should_hide_incomplete_read_detail_from_user(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            raise nai_web_client_module.requests.exceptions.ConnectionError(
                "Connection broken",
                IncompleteRead(b"x" * 1048576, 486510),
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_direct_url_fallback": False,
                    "nai_nocache": 1,
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成结果传输中断，请稍后重试")

    def test_generate_image_should_allow_disabling_direct_url_fallback(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            raise nai_web_client_module.requests.exceptions.ConnectionError(
                "Connection broken",
                IncompleteRead(b"x" * 1048576, 486510),
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_direct_url_fallback": False,
                    "nai_nocache": 1,
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成结果传输中断，请稍后重试")

    def test_generate_image_should_hide_gateway_timeout_html_from_user(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            return _DummyResponse(
                status_code=504,
                headers={"content-type": "text/html"},
                text='<html class="no-js" lang="en-US"><head><title>网站请求超时</title><meta charset="UTF-8"></head></html>',
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_direct_url_fallback": False,
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成服务网关超时（HTTP 504），请稍后重试")

    def test_generate_image_should_return_gateway_timeout_when_retryable_http_persists(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            return _DummyResponse(
                status_code=504,
                headers={"content-type": "text/html"},
                text='<html class="no-js" lang="en-US"><head><title>网站请求超时</title></head></html>',
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成服务网关超时（HTTP 504），请稍后重试")

    def test_generate_image_should_retry_once_when_upstream_protection_page_returns_418(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False
        calls = []
        responses = [
            _DummyResponse(
                status_code=418,
                headers={"content-type": "text/html"},
                text=(
                    "<html><head><title>安全防护</title><style>"
                    ":root { --gcp-bg: #f8f9fa; --gcp-card: #ffffff; }"
                    "</style></head><body>安全防护</body></html>"
                ),
            ),
            _DummyResponse(
                status_code=200,
                headers={"content-type": "image/png"},
                text="",
                content=_ONE_PIXEL_PNG_BYTES,
            ),
        ]

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            calls.append((url, dict(params), proxy_mode, request_timeout))
            return responses.pop(0)

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                },
            ))

        self.assertTrue(success)
        self.assertEqual(result, base64.b64encode(_ONE_PIXEL_PNG_BYTES).decode("utf-8"))
        self.assertEqual(len(calls), 2)

    def test_generate_image_should_hide_protection_html_and_return_short_message(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            return _DummyResponse(
                status_code=418,
                headers={"content-type": "text/html"},
                text=(
                    "<html><head><title>安全防护</title><style>"
                    ":root { --gcp-bg: #f8f9fa; --gcp-card: #ffffff; }"
                    "</style></head><body>安全防护</body></html>"
                ),
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(*_args, **_kwargs):
            return None

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread), patch.object(
            nai_web_client_module.asyncio, "sleep", side_effect=fake_sleep
        ):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成服务触发上游安全防护（HTTP 418），请稍后重试")

    def test_generate_image_should_fail_when_json_returns_generate_url(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            return _DummyResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                json_data={
                    "url": "https://example.com/generate?tag=1girl&model=nai-diffusion-4-5-full&token=abc",
                },
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "上游返回了生成请求链接，已停止自动补拉以避免重复扣费，请稍后重试")

    def test_generate_image_should_reject_unexpected_html_with_200_status(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            return _DummyResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                text="<html><head><title>Temporary page</title></head><body>waiting</body></html>",
                content=b"<html><body>waiting</body></html>",
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                    "nai_direct_url_fallback": False,
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成服务返回了异常页面，请稍后重试")

    def test_generate_image_should_return_html_error_when_unexpected_html_persists(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        client._auto_proxy_direct_only = False

        def fake_send_request(
            url,
            params,
            proxy_mode="auto",
            request_timeout=NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            request_headers=None,
        ):
            return _DummyResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                text="<html><head><title>Temporary page</title></head><body>waiting</body></html>",
                content=b"<html><body>waiting</body></html>",
            )

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        client._send_request = fake_send_request  # type: ignore[method-assign]

        with patch.object(nai_web_client_module.asyncio, "to_thread", side_effect=fake_to_thread):
            success, result = asyncio.run(client.generate_image(
                prompt="1girl, smile",
                model_config={
                    "base_url": "https://example.com",
                    "nai_endpoint": "/generate",
                    "default_model": "nai-diffusion-4-5-full",
                },
            ))

        self.assertFalse(success)
        self.assertEqual(result, "图片生成服务返回了异常页面，请稍后重试")

    def test_send_request_should_fallback_to_direct_when_proxy_fails_in_auto_mode(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        calls = []

        def fake_request_with_session(trust_env, url, params, request_timeout, request_headers):
            calls.append((trust_env, url, dict(params), request_timeout))
            if trust_env:
                raise nai_web_client_module.ProxyError("proxy down")
            return _DummyResponse()

        client._request_with_session = fake_request_with_session  # type: ignore[method-assign]

        response = client._send_request(
            "https://example.com/generate",
            {"tag": "1girl", "model": "nai-diffusion-4-5-full"},
            "auto",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][0])
        self.assertFalse(calls[1][0])
        self.assertTrue(all(call[3] == NaiWebClient._DEFAULT_REQUEST_TIMEOUT for call in calls))

    def test_send_request_should_not_fallback_when_direct_mode_selected(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        calls = []

        def fake_request_with_session(trust_env, url, params, request_timeout, request_headers):
            calls.append((trust_env, request_timeout))
            return _DummyResponse()

        client._request_with_session = fake_request_with_session  # type: ignore[method-assign]

        response = client._send_request(
            "https://example.com/generate",
            {"tag": "1girl", "model": "nai-diffusion-4-5-full"},
            "direct",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [(False, NaiWebClient._DEFAULT_REQUEST_TIMEOUT)])

    def test_send_request_should_fallback_when_connection_error_contains_proxy_keyword(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        calls = []

        def fake_request_with_session(trust_env, url, params, request_timeout, request_headers):
            calls.append((trust_env, request_timeout))
            if trust_env:
                raise nai_web_client_module.requests.exceptions.ConnectionError(
                    "Unable to connect to proxy"
                )
            return _DummyResponse()

        client._request_with_session = fake_request_with_session  # type: ignore[method-assign]

        response = client._send_request(
            "https://example.com/generate",
            {"tag": "1girl", "model": "nai-diffusion-4-5-full"},
            "auto",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            calls,
            [
                (True, NaiWebClient._DEFAULT_REQUEST_TIMEOUT),
                (False, NaiWebClient._DEFAULT_REQUEST_TIMEOUT),
            ],
        )
        self.assertTrue(client._auto_proxy_direct_only)

    def test_send_request_should_skip_inherited_proxy_after_first_proxy_failure(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        calls = []

        def fake_request_with_session(trust_env, url, params, request_timeout, request_headers):
            calls.append((trust_env, request_timeout))
            if trust_env:
                raise nai_web_client_module.ProxyError("proxy down")
            return _DummyResponse()

        client._request_with_session = fake_request_with_session  # type: ignore[method-assign]

        first_response = client._send_request(
            "https://example.com/generate",
            {"tag": "1girl", "model": "nai-diffusion-4-5-full"},
            "auto",
        )
        second_response = client._send_request(
            "https://example.com/generate",
            {"tag": "1girl", "model": "nai-diffusion-4-5-full"},
            "auto",
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(
            calls,
            [
                (True, NaiWebClient._DEFAULT_REQUEST_TIMEOUT),
                (False, NaiWebClient._DEFAULT_REQUEST_TIMEOUT),
                (False, NaiWebClient._DEFAULT_REQUEST_TIMEOUT),
            ],
        )

    def test_request_with_session_should_send_connection_close_header(self):
        client = object.__new__(NaiWebClient)
        client.action = _DummyAction()
        client.log_prefix = _DummyAction.log_prefix
        captured = {}

        class _FakeSession:
            def get(self, **kwargs):
                captured.update(kwargs)
                return _DummyResponse()

        client._get_session = lambda trust_env: _FakeSession()  # type: ignore[method-assign]

        response = client._request_with_session(
            False,
            "https://example.com/generate",
            {"tag": "1girl"},
            NaiWebClient._DEFAULT_REQUEST_TIMEOUT,
            NaiWebClient._build_request_headers("https://example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["headers"]["Connection"], "close")
        self.assertIn("image/", captured["headers"]["Accept"])
        self.assertEqual(captured["headers"]["Referer"], "https://example.com/")


if __name__ == "__main__":
    unittest.main()
