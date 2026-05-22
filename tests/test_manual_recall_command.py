# -*- coding: utf-8 -*-
import os
import re
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

plugin_system_module = types.ModuleType("src.plugin_system")
sys.modules["src.plugin_system"] = plugin_system_module

plugin_system_base_package = types.ModuleType("src.plugin_system.base")
sys.modules["src.plugin_system.base"] = plugin_system_base_package

base_command_module = types.ModuleType("src.plugin_system.base.base_command")


class _DummyBaseCommand:
    pass


base_command_module.BaseCommand = _DummyBaseCommand
sys.modules["src.plugin_system.base.base_command"] = base_command_module

core_package = types.ModuleType("core")
core_package.__path__ = [os.path.join(PLUGIN_DIR, "core")]
sys.modules.setdefault("core", core_package)

core_commands_package = types.ModuleType("core.commands")
core_commands_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "commands")]
sys.modules.setdefault("core.commands", core_commands_package)

core_mixins_package = types.ModuleType("core.mixins")
core_mixins_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "mixins")]
sys.modules.setdefault("core.mixins", core_mixins_package)

core_utils_package = types.ModuleType("core.utils")
core_utils_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "utils")]
sys.modules.setdefault("core.utils", core_utils_package)

core_constants_module = importlib.import_module("core.constants")
manual_recall_module = importlib.import_module("core.commands.nai_manual_recall_command")

NaiManualRecallCommand = manual_recall_module.NaiManualRecallCommand


class _DummyMessageInfo:
    def __init__(self, message_id: str, additional_config=None):
        self.message_id = message_id
        self.additional_config = additional_config


class _DummyMessage:
    def __init__(self, message_id: str, additional_config=None, message_segment=None, reply=None):
        self.reply = reply
        self.message_info = _DummyMessageInfo(message_id=message_id, additional_config=additional_config)
        self.raw_message = None
        self.message_segment = message_segment
        self.chat_stream = type("ChatStream", (), {"stream_id": "stream_123"})()
        self.display_message = None
        self.processed_plain_text = "[imageurl:file:///a.png]"
        self.is_picid = False


class ManualRecallCommandTest(unittest.TestCase):
    def test_command_pattern_should_match_reply_prefixed_manual_recall(self):
        text = "[回复<xx> 的消息：[imageurl:file:///a.png]] /nai 撤回"
        self.assertIsNotNone(re.match(NaiManualRecallCommand.command_pattern, text))

    def test_command_pattern_should_match_manual_recall_with_trailing_image_artifact(self):
        text = "/nai 撤回 [图片]"
        self.assertIsNotNone(re.match(NaiManualRecallCommand.command_pattern, text))

    def test_execute_should_recall_latest_image_only(self):
        cmd = object.__new__(NaiManualRecallCommand)
        cmd.log_prefix = "[test]"
        cmd.message = _DummyMessage(
            message_id="cmd_123",
            additional_config={},
            message_segment=None,
        )
        sent_texts = []

        async def _get_last_message_candidate(**kwargs):
            return "latest_001", None, 100.0

        async def _do_recall(message_id, source):
            return True, f"{source}:{message_id}", True

        async def _send_text(text, **kwargs):
            sent_texts.append(text)

        cmd._get_last_message_candidate = _get_last_message_candidate
        cmd._do_recall = _do_recall
        cmd.send_text = _send_text

        import asyncio
        ok, reason, intercept = asyncio.run(cmd.execute())

        self.assertTrue(ok)
        self.assertEqual(reason, "最近图片:latest_001")
        self.assertTrue(intercept)
        self.assertEqual(sent_texts, [])

    def test_execute_should_skip_recently_recalled_message_on_next_call(self):
        import asyncio

        NaiManualRecallCommand._recent_manual_recall_ids = {}

        first = object.__new__(NaiManualRecallCommand)
        first.log_prefix = "[test]"
        first.message = _DummyMessage(
            message_id="cmd_1",
            additional_config={},
            message_segment=None,
        )

        second = object.__new__(NaiManualRecallCommand)
        second.log_prefix = "[test]"
        second.message = _DummyMessage(
            message_id="cmd_2",
            additional_config={},
            message_segment=None,
        )

        captured_excludes = []
        first_targets = iter(["img_2"])
        second_targets = iter(["img_1"])

        async def _get_last_message_candidate_first(**kwargs):
            captured_excludes.append(set(kwargs.get("exclude_message_ids") or set()))
            return next(first_targets), None, 101.0

        async def _get_last_message_candidate_second(**kwargs):
            captured_excludes.append(set(kwargs.get("exclude_message_ids") or set()))
            return next(second_targets), None, 100.0

        async def _do_recall(self, message_id, source):
            return await NaiManualRecallCommand._do_recall(self, message_id, source)

        async def _try_recall_message(_message_id):
            return True

        async def _send_text(_text, **kwargs):
            return None

        first._get_last_message_candidate = _get_last_message_candidate_first
        first._try_recall_message = _try_recall_message
        first.send_text = _send_text

        second._get_last_message_candidate = _get_last_message_candidate_second
        second._try_recall_message = _try_recall_message
        second.send_text = _send_text

        ok1, _, _ = asyncio.run(first.execute())
        ok2, _, _ = asyncio.run(second.execute())

        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(captured_excludes[0], set())
        self.assertEqual(captured_excludes[1], {"img_2"})

    def test_execute_should_resolve_newest_placeholder_instead_of_recalling_older_formal(self):
        import asyncio

        cmd = object.__new__(NaiManualRecallCommand)
        cmd.log_prefix = "[test]"
        cmd.message = _DummyMessage(
            message_id="cmd_3",
            additional_config={},
            message_segment=None,
        )

        captured_target_timestamps = []

        async def _get_last_message_candidate(**kwargs):
            return None, "send_api_latest", 200.5

        async def _resolve_latest_message_id(message_id, target_send_timestamp=None):
            captured_target_timestamps.append((message_id, target_send_timestamp))
            return "img_latest_formal"

        async def _do_recall(message_id, source):
            return True, f"{source}:{message_id}", True

        async def _send_text(_text, **kwargs):
            return None

        cmd._get_last_message_candidate = _get_last_message_candidate
        cmd._resolve_latest_message_id = _resolve_latest_message_id
        cmd._do_recall = _do_recall
        cmd.send_text = _send_text

        ok, reason, intercept = asyncio.run(cmd.execute())

        self.assertTrue(ok)
        self.assertEqual(reason, "最近图片:img_latest_formal")
        self.assertTrue(intercept)
        self.assertEqual(captured_target_timestamps, [("send_api_latest", 200.5)])


if __name__ == "__main__":
    unittest.main()
