# -*- coding: utf-8 -*-
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from runtime_recall import (
    MANUAL_RECALL_TTL_SECONDS,
    PLUGIN_IMAGE_MARKER_CONFIG_KEY,
    attach_plugin_image_marker_to_message,
    build_recall_command_payloads,
    does_napcat_message_exist,
    is_napcat_action_accepted,
    load_recent_plugin_image_rows,
    load_recent_tracked_plugin_image_rows,
    normalize_db_timestamp,
    prune_recent_ids,
    remember_pending_plugin_image_send,
    remember_recent_id,
    remember_sent_plugin_image_message,
    resolve_db_path,
    select_recent_plugin_image_row,
    wait_for_formal_message_id,
)

MARKER = "[nai_draw_plugin:image]"


class RuntimeRecallTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "MaiBot.db"
        self._init_db()

    def tearDown(self):
        self.tempdir.cleanup()

    def _init_db(self):
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE mai_messages (
                    message_id TEXT,
                    timestamp TEXT,
                    session_id TEXT,
                    is_picture INTEGER,
                    display_message TEXT,
                    processed_plain_text TEXT
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _insert_message(
        self,
        *,
        message_id: str,
        timestamp: str,
        session_id: str,
        is_picture: int,
        display_message: str,
        processed_plain_text: str,
    ):
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """
                INSERT INTO mai_messages (
                    message_id,
                    timestamp,
                    session_id,
                    is_picture,
                    display_message,
                    processed_plain_text
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    timestamp,
                    session_id,
                    is_picture,
                    display_message,
                    processed_plain_text,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def test_resolve_db_path(self):
        runtime_file = "/root/maimai/MaiBot/plugins/nai_draw_plugin/sdk_runtime.py"
        self.assertEqual(
            resolve_db_path(runtime_file),
            Path("/root/maimai/MaiBot/data/MaiBot.db"),
        )

    def test_normalize_db_timestamp_should_parse_iso_datetime(self):
        value = normalize_db_timestamp("2026-04-05 12:34:56.123456")
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value, 1775363696.123456, places=3)

    def test_load_recent_plugin_image_rows_should_filter_marker_and_session(self):
        self._insert_message(
            message_id="img_ok",
            timestamp="2026-04-05 12:00:03",
            session_id="stream_1",
            is_picture=1,
            display_message="[nai_draw_plugin:image] [NAI图片:自拍]",
            processed_plain_text="[图片]",
        )
        self._insert_message(
            message_id="img_other_session",
            timestamp="2026-04-05 12:00:02",
            session_id="stream_2",
            is_picture=1,
            display_message="[nai_draw_plugin:image] [NAI图片:自拍]",
            processed_plain_text="[图片]",
        )
        self._insert_message(
            message_id="img_no_marker",
            timestamp="2026-04-05 12:00:01",
            session_id="stream_1",
            is_picture=1,
            display_message="[NAI图片:自拍]",
            processed_plain_text="[图片]",
        )
        self._insert_message(
            message_id="text_with_marker",
            timestamp="2026-04-05 12:00:00",
            session_id="stream_1",
            is_picture=0,
            display_message="[文本消息]",
            processed_plain_text="不是图片",
        )

        rows = load_recent_plugin_image_rows(
            self.db_path,
            "stream_1",
            "[nai_draw_plugin:image]",
            limit=10,
        )

        self.assertEqual([row["message_id"] for row in rows], ["img_ok"])

    def test_load_recent_plugin_image_rows_should_include_marker_rows_even_when_not_flagged_as_picture(self):
        self._insert_message(
            message_id="img_url_row",
            timestamp="2026-04-05 12:00:04",
            session_id="stream_1",
            is_picture=0,
            display_message="[nai_draw_plugin:image] [imageurl:file:///tmp/test.png]",
            processed_plain_text="[图片]",
        )

        rows = load_recent_plugin_image_rows(
            self.db_path,
            "stream_1",
            MARKER,
            limit=10,
        )

        self.assertEqual([row["message_id"] for row in rows], ["img_url_row"])

    def test_select_recent_plugin_image_row_without_target_should_pick_latest(self):
        rows = [
            {
                "message_id": "img_old",
                "timestamp": "2026-04-05 12:00:01",
            },
            {
                "message_id": "img_latest",
                "timestamp": "2026-04-05 12:00:03",
            },
        ]

        row = select_recent_plugin_image_row(rows)

        self.assertIsNotNone(row)
        self.assertEqual(row["message_id"], "img_latest")

    def test_select_recent_plugin_image_row_should_pick_closest_after_send_time(self):
        rows = [
            {
                "message_id": "img_late",
                "timestamp": "2026-04-05 12:00:04.000000",
            },
            {
                "message_id": "img_target",
                "timestamp": "2026-04-05 12:00:02.400000",
            },
            {
                "message_id": "img_too_old",
                "timestamp": "2026-04-05 11:59:58.000000",
            },
        ]

        row = select_recent_plugin_image_row(
            rows,
            target_send_timestamp=normalize_db_timestamp("2026-04-05 12:00:02.200000"),
        )

        self.assertIsNotNone(row)
        self.assertEqual(row["message_id"], "img_target")

    def test_select_recent_plugin_image_row_should_skip_excluded_ids(self):
        rows = [
            {
                "message_id": "img_latest",
                "timestamp": "2026-04-05 12:00:03",
            },
            {
                "message_id": "img_prev",
                "timestamp": "2026-04-05 12:00:02",
            },
        ]

        row = select_recent_plugin_image_row(rows, exclude_message_ids={"img_latest"})

        self.assertIsNotNone(row)
        self.assertEqual(row["message_id"], "img_prev")

    def test_recent_recall_tracking(self):
        state = {}

        remember_recent_id(state, "stream_1", "img_1", now=100.0)
        remember_recent_id(state, "stream_1", "img_2", now=150.0)

        self.assertEqual(
            prune_recent_ids(state, "stream_1", ttl_seconds=MANUAL_RECALL_TTL_SECONDS, now=200.0),
            {"img_1", "img_2"},
        )
        self.assertEqual(
            prune_recent_ids(state, "stream_1", ttl_seconds=30.0, now=200.0),
            set(),
        )

    def test_build_recall_command_payloads(self):
        payloads = build_recall_command_payloads("123")

        self.assertEqual(payloads[0], {"name": "DELETE_MSG", "args": {"message_id": "123"}})
        self.assertEqual(payloads[-1], {"name": "recall_msg", "args": {"message_id": "123"}})

    def test_is_napcat_action_accepted_should_handle_sdk_wrapper_and_raw_response(self):
        self.assertTrue(is_napcat_action_accepted({"success": True, "result": {"status": "ok"}}))
        self.assertTrue(is_napcat_action_accepted({"status": "ok", "retcode": 0, "data": None}))
        self.assertFalse(is_napcat_action_accepted({"success": False, "error": "boom"}))
        self.assertFalse(is_napcat_action_accepted({"status": "failed"}))

    def test_does_napcat_message_exist_should_handle_sdk_wrapper_and_empty_result(self):
        self.assertTrue(
            does_napcat_message_exist(
                {"success": True, "result": {"message_id": "123", "message": [{"type": "text", "data": {}}]}}
            )
        )
        self.assertTrue(
            does_napcat_message_exist(
                {"message_id": "456", "message": [{"type": "image", "data": {"file": "x"}}]}
            )
        )
        self.assertFalse(does_napcat_message_exist({"success": True, "result": None}))
        self.assertFalse(does_napcat_message_exist({"success": False, "error": "not found"}))

    def test_wait_for_formal_message_id_should_upgrade_placeholder(self):
        import asyncio

        rows = iter(
            [
                {"message_id": "img_formal", "timestamp": "2026-04-05 12:00:02"},
            ]
        )
        sleep_calls = []
        clock = iter([0.0, 0.05, 0.10])

        async def _row_loader():
            return next(rows, None)

        async def _sleep(delay):
            sleep_calls.append(delay)

        message_id = asyncio.run(
            wait_for_formal_message_id(
                _row_loader,
                initial_row={"message_id": "send_api_123", "timestamp": "2026-04-05 12:00:01"},
                id_wait_seconds=1.0,
                poll_interval=0.1,
                monotonic=lambda: next(clock),
                sleep=_sleep,
            )
        )

        self.assertEqual(message_id, "img_formal")
        self.assertEqual(sleep_calls, [0.1])

    def test_wait_for_formal_message_id_should_fallback_to_placeholder_on_timeout(self):
        import asyncio

        sleep_calls = []
        clock = iter([0.0, 0.10, 0.30])

        async def _row_loader():
            return {"message_id": "send_api_456", "timestamp": "2026-04-05 12:00:02"}

        async def _sleep(delay):
            sleep_calls.append(delay)

        message_id = asyncio.run(
            wait_for_formal_message_id(
                _row_loader,
                initial_row={"message_id": "send_api_123", "timestamp": "2026-04-05 12:00:01"},
                id_wait_seconds=0.2,
                poll_interval=0.1,
                monotonic=lambda: next(clock),
                sleep=_sleep,
            )
        )

        self.assertEqual(message_id, "send_api_456")
        self.assertEqual(sleep_calls, [0.1])

    def test_attach_plugin_image_marker_to_message_should_mark_pending_imageurl_message(self):
        remember_pending_plugin_image_send("stream_1", 1000.0, now=1000.0)
        message = {
            "message_id": "send_api_124",
            "timestamp": "1000.05",
            "session_id": "stream_1",
            "is_picture": False,
            "raw_message": [
                {
                    "type": "dict",
                    "data": {
                        "type": "imageurl",
                        "data": "file:///tmp/test.png",
                    },
                }
            ],
            "message_info": {
                "additional_config": {},
            },
        }

        attached = attach_plugin_image_marker_to_message(message, MARKER, now=1000.1)

        self.assertTrue(attached)
        self.assertTrue(message["is_picture"])
        self.assertEqual(
            message["message_info"]["additional_config"][PLUGIN_IMAGE_MARKER_CONFIG_KEY],
            MARKER,
        )

    def test_remember_sent_plugin_image_message_should_track_imageurl_row(self):
        remember_pending_plugin_image_send("stream_1", 1000.0, now=1000.0)
        message = {
            "message_id": "123457",
            "timestamp": "1000.08",
            "session_id": "stream_1",
            "is_picture": False,
            "raw_message": [
                {
                    "type": "dict",
                    "data": {
                        "type": "imageurl",
                        "data": "file:///tmp/test.png",
                    },
                }
            ],
            "message_info": {
                "additional_config": {
                    PLUGIN_IMAGE_MARKER_CONFIG_KEY: MARKER,
                },
            },
        }

        remembered = remember_sent_plugin_image_message(message, MARKER, now=1000.2)
        rows = load_recent_tracked_plugin_image_rows("stream_1", limit=10, now=1000.3)

        self.assertTrue(remembered)
        self.assertTrue(message["is_picture"])
        self.assertEqual([row["message_id"] for row in rows], ["123457"])


if __name__ == "__main__":
    unittest.main()
