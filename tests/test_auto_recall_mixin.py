# -*- coding: utf-8 -*-
import os
import sys
import types
import unittest
import importlib


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

src_package = types.ModuleType("src")
src_package.__path__ = [os.path.join(MAIBOT_ROOT, "src")]
sys.modules.setdefault("src", src_package)

src_chat_package = types.ModuleType("src.chat")
src_chat_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "chat")]
sys.modules.setdefault("src.chat", src_chat_package)

src_chat_utils_package = types.ModuleType("src.chat.utils")
src_chat_utils_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "chat", "utils")]
sys.modules.setdefault("src.chat.utils", src_chat_utils_package)

chat_utils_module = types.ModuleType("src.chat.utils.utils")
chat_utils_module.parse_platform_accounts = lambda _platforms: {}
sys.modules["src.chat.utils.utils"] = chat_utils_module

src_config_package = types.ModuleType("src.config")
src_config_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "config")]
sys.modules.setdefault("src.config", src_config_package)

config_module = types.ModuleType("src.config.config")
config_module.global_config = types.SimpleNamespace(
    bot=types.SimpleNamespace(
        platforms=[],
        qq_account="bot",
        telegram_account="",
        nickname="bot",
    )
)
sys.modules["src.config.config"] = config_module

core_package = types.ModuleType("core")
core_package.__path__ = [os.path.join(PLUGIN_DIR, "core")]
sys.modules.setdefault("core", core_package)

core_mixins_package = types.ModuleType("core.mixins")
core_mixins_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "mixins")]
sys.modules.setdefault("core.mixins", core_mixins_package)

core_constants_module = importlib.import_module("core.constants")
display_message_helper_module = importlib.import_module("core.utils.display_message_helper")
auto_recall_mixin_module = importlib.import_module("core.mixins.auto_recall_mixin")

NAI_PIC_IMAGE_DISPLAY_MARKER = core_constants_module.NAI_PIC_IMAGE_DISPLAY_MARKER
build_action_image_display_message = display_message_helper_module.build_action_image_display_message

AutoRecallMixin = auto_recall_mixin_module.AutoRecallMixin
_extract_sender_user_id = auto_recall_mixin_module._extract_sender_user_id
_is_image_message = auto_recall_mixin_module._is_image_message
_is_nai_draw_plugin_image_message = auto_recall_mixin_module._is_nai_draw_plugin_image_message
_text_looks_like_image = auto_recall_mixin_module._text_looks_like_image


class _DummyUserInfo:
    def __init__(self, user_id: str):
        self.user_id = user_id


class _DummyMsg:
    def __init__(self, user_id: str, text: str):
        self.user_info = _DummyUserInfo(user_id)
        self.processed_plain_text = text
        self.display_message = None
        self.raw_message = None
        self.is_picid = False


class _DummyRecallHost(AutoRecallMixin):
    log_prefix = "test_nai_pic"

    def _is_auto_recall_enabled(self, platform: str, chat_id: str) -> bool:
        return True


class AutoRecallMixinUtilsTest(unittest.TestCase):
    def test_text_looks_like_image_prefix(self):
        self.assertTrue(_text_looks_like_image("[imageurl:file:///a.png]"))
        self.assertTrue(_text_looks_like_image("   [图片：xxx]"))
        self.assertTrue(_text_looks_like_image("[NAI图片:自拍 黑丝]"))
        self.assertFalse(_text_looks_like_image("[回复<xx> 的消息：[imageurl:file:///a.png]]"))

    def test_is_image_message_avoid_reply_false_positive(self):
        msg = _DummyMsg("bot", "[回复<xx> 的消息：[imageurl:file:///a.png]] 你好")
        self.assertFalse(_is_image_message(msg))

        msg2 = _DummyMsg("bot", "[imageurl:file:///a.png]")
        self.assertTrue(_is_image_message(msg2))

    def test_extract_sender_user_id(self):
        msg = _DummyMsg("123", "[imageurl:file:///a.png]")
        self.assertEqual(_extract_sender_user_id(msg), "123")

        msg_dict = {"message_info": {"user_info": {"user_id": "456"}}}
        self.assertEqual(_extract_sender_user_id(msg_dict), "456")

    def test_is_nai_draw_plugin_image_message_marker(self):
        msg = _DummyMsg("bot", "[imageurl:file:///a.png]")
        msg.display_message = NAI_PIC_IMAGE_DISPLAY_MARKER
        self.assertTrue(_is_nai_draw_plugin_image_message(msg))

        msg2 = _DummyMsg("bot", "[imageurl:file:///a.png]")
        self.assertFalse(_is_nai_draw_plugin_image_message(msg2))

        msg_dict = {
            "processed_plain_text": "[imageurl:file:///a.png]",
            "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
        }
        self.assertTrue(_is_nai_draw_plugin_image_message(msg_dict))

    def test_is_nai_draw_plugin_image_message_action_display_message(self):
        msg = _DummyMsg("bot", "[imageurl:file:///a.png]")
        msg.display_message = build_action_image_display_message("自拍 黑丝 卧室")
        self.assertTrue(_is_nai_draw_plugin_image_message(msg))

    def test_select_best_message_id_should_pick_closest_message_after_send_timestamp(self):
        host = _DummyRecallHost()
        msgs = [
            {
                "message_id": "msg_late",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///late.png]",
                "time": 101.20,
            },
            {
                "message_id": "msg_target",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///target.png]",
                "time": 100.35,
            },
            {
                "message_id": "msg_old",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///old.png]",
                "time": 99.80,
            },
        ]

        resolved_id, placeholder_id = host._select_best_message_id(
            msgs=msgs,
            require_marker=True,
            bot_account="",
            send_timestamp=100.30,
            timestamp_tolerance=0.2,
        )

        self.assertEqual(resolved_id, "msg_target")
        self.assertIsNone(placeholder_id)

    def test_select_best_message_id_should_fallback_to_formal_bot_image_without_marker(self):
        host = _DummyRecallHost()
        msgs = [
            {
                "message_id": "send_api_123",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///placeholder.png]",
                "time": 100.30,
            },
            {
                "message_id": "msg_formal",
                "processed_plain_text": "[imageurl:file:///formal.png]",
                "time": 100.45,
                "user_info": {"user_id": "bot"},
            },
        ]

        resolved_id, placeholder_id = host._select_best_message_id(
            msgs=msgs,
            require_marker=True,
            bot_account="bot",
            send_timestamp=100.30,
            timestamp_tolerance=0.2,
        )

        self.assertEqual(resolved_id, "msg_formal")
        self.assertIsNone(placeholder_id)

    def test_select_best_message_id_without_send_timestamp_should_pick_latest_by_time(self):
        host = _DummyRecallHost()
        msgs = [
            {
                "message_id": "msg_old",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///old.png]",
                "time": 100.10,
            },
            {
                "message_id": "msg_latest",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///latest.png]",
                "time": 100.90,
            },
        ]

        resolved_id, placeholder_id = host._select_best_message_id(
            msgs=msgs,
            require_marker=True,
            bot_account="",
            send_timestamp=None,
            timestamp_tolerance=0.2,
        )

        self.assertEqual(resolved_id, "msg_latest")
        self.assertIsNone(placeholder_id)

    def test_select_best_message_id_without_send_timestamp_should_fallback_to_latest_formal_bot_image(self):
        host = _DummyRecallHost()
        msgs = [
            {
                "message_id": "send_api_old",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///placeholder.png]",
                "time": 100.10,
            },
            {
                "message_id": "msg_latest_formal",
                "processed_plain_text": "[imageurl:file:///latest.png]",
                "time": 100.90,
                "user_info": {"user_id": "bot"},
            },
        ]

        resolved_id, placeholder_id = host._select_best_message_id(
            msgs=msgs,
            require_marker=True,
            bot_account="bot",
            send_timestamp=None,
            timestamp_tolerance=0.2,
        )

        self.assertEqual(resolved_id, "msg_latest_formal")
        self.assertIsNone(placeholder_id)

    def test_select_best_message_id_without_send_timestamp_should_prefer_newer_formal_without_marker(self):
        host = _DummyRecallHost()
        msgs = [
            {
                "message_id": "msg_old_with_marker",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///old.png]",
                "time": 100.10,
                "user_info": {"user_id": "bot"},
            },
            {
                "message_id": "msg_new_formal",
                "processed_plain_text": "[imageurl:file:///new.png]",
                "time": 100.90,
                "user_info": {"user_id": "bot"},
            },
        ]

        resolved_id, placeholder_id = host._select_best_message_id(
            msgs=msgs,
            require_marker=True,
            bot_account="bot",
            send_timestamp=None,
            timestamp_tolerance=0.2,
        )

        self.assertEqual(resolved_id, "msg_new_formal")
        self.assertIsNone(placeholder_id)

    def test_select_best_message_id_without_send_timestamp_should_prefer_newest_placeholder_over_older_formal(self):
        host = _DummyRecallHost()
        msgs = [
            {
                "message_id": "msg_old_formal",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///old.png]",
                "time": 100.10,
                "user_info": {"user_id": "bot"},
            },
            {
                "message_id": "send_api_new",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///new.png]",
                "time": 100.90,
                "user_info": {"user_id": "bot"},
            },
        ]

        resolved_id, placeholder_id, candidate_time = host._select_best_message_candidate(
            msgs=msgs,
            require_marker=True,
            bot_account="bot",
            send_timestamp=None,
            timestamp_tolerance=0.2,
        )

        self.assertIsNone(resolved_id)
        self.assertEqual(placeholder_id, "send_api_new")
        self.assertEqual(candidate_time, 100.90)

    def test_select_best_message_id_should_prefer_closest_newer_formal_without_marker(self):
        host = _DummyRecallHost()
        msgs = [
            {
                "message_id": "msg_old_with_marker",
                "display_message": NAI_PIC_IMAGE_DISPLAY_MARKER,
                "processed_plain_text": "[imageurl:file:///old.png]",
                "time": 100.10,
                "user_info": {"user_id": "bot"},
            },
            {
                "message_id": "msg_target_formal",
                "processed_plain_text": "[imageurl:file:///target.png]",
                "time": 100.42,
                "user_info": {"user_id": "bot"},
            },
        ]

        resolved_id, placeholder_id = host._select_best_message_id(
            msgs=msgs,
            require_marker=True,
            bot_account="bot",
            send_timestamp=100.30,
            timestamp_tolerance=0.2,
        )

        self.assertEqual(resolved_id, "msg_target_formal")
        self.assertIsNone(placeholder_id)


if __name__ == "__main__":
    unittest.main()
