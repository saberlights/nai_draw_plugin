import asyncio
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

if "src.common.logger" not in sys.modules:
    logger_module = types.ModuleType("src.common.logger")

    class _Logger:
        def debug(self, *_args, **_kwargs):
            return None

        def info(self, *_args, **_kwargs):
            return None

        def warning(self, *_args, **_kwargs):
            return None

        def error(self, *_args, **_kwargs):
            return None

    logger_module.get_logger = lambda _name=None: _Logger()
    sys.modules["src.common.logger"] = logger_module

    src_package = types.ModuleType("src")
    src_package.__path__ = [os.path.join(MAIBOT_ROOT, "src")]
    sys.modules.setdefault("src", src_package)

from plugins.nai_draw_plugin.core.services.recall_workflow import RecallWorkflow
from plugins.nai_draw_plugin.runtime_recall import reset_runtime_recall_tracking_state


MARKER = "[nai_draw_plugin:image]"


@pytest.fixture(autouse=True)
def _reset_recall_tracking() -> None:
    reset_runtime_recall_tracking_state()


def _create_message_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE mai_messages (
                message_id TEXT,
                timestamp REAL,
                session_id TEXT,
                is_picture INTEGER,
                display_message TEXT,
                processed_plain_text TEXT,
                additional_config TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def _insert_plugin_image(path: Path, *, message_id: str, timestamp: float) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            INSERT INTO mai_messages (
                message_id,
                timestamp,
                session_id,
                is_picture,
                display_message,
                processed_plain_text,
                additional_config
            ) VALUES (?, ?, ?, 1, ?, '[图片]', '')
            """,
            (message_id, timestamp, "stream-1", MARKER),
        )
        connection.commit()
    finally:
        connection.close()


def test_manual_recall_rejects_stale_image_without_platform_delete(tmp_path: Path) -> None:
    db_path = tmp_path / "MaiBot.db"
    _create_message_db(db_path)
    _insert_plugin_image(db_path, message_id="101", timestamp=1_000.0)

    api_calls: list[tuple[str, dict[str, object]]] = []
    sent_texts: list[str] = []

    async def call_api(name: str, **kwargs: object) -> dict[str, object]:
        api_calls.append((name, kwargs))
        return {"success": True}

    async def send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    workflow = RecallWorkflow(
        config={"auto_recall": {"manual_max_age_seconds": 600}},
        stream_id="stream-1",
        context=types.SimpleNamespace(api=types.SimpleNamespace(call=call_api)),
        send_text=send_text,
        track_task=lambda _task: None,
        log_prefix="test",
        db_path=db_path,
        napcat_config_path=tmp_path / "missing-napcat.toml",
        recent_recall_state={},
        wall_clock=lambda: 2_000.0,
    )

    result = asyncio.run(workflow.execute_manual_recall())

    assert result == (False, "找不到可撤回的消息", True)
    assert api_calls == []
    assert sent_texts == ["❌ 找不到近期可撤回的图片（图片可能已超过平台撤回时限）"]


def test_manual_recall_deletes_latest_image_and_reports_success(tmp_path: Path) -> None:
    db_path = tmp_path / "MaiBot.db"
    _create_message_db(db_path)
    _insert_plugin_image(db_path, message_id="202", timestamp=1_900.0)

    api_calls: list[tuple[str, dict[str, object]]] = []
    sent_texts: list[str] = []

    async def call_api(name: str, **kwargs: object) -> dict[str, object]:
        api_calls.append((name, kwargs))
        return {"success": True, "result": {"status": "ok"}}

    async def send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    workflow = RecallWorkflow(
        config={"auto_recall": {"manual_max_age_seconds": 600, "id_wait_seconds": 0}},
        stream_id="stream-1",
        context=types.SimpleNamespace(api=types.SimpleNamespace(call=call_api)),
        send_text=send_text,
        track_task=lambda _task: None,
        log_prefix="test",
        db_path=db_path,
        napcat_config_path=tmp_path / "missing-napcat.toml",
        recent_recall_state={},
        wall_clock=lambda: 2_000.0,
    )

    result = asyncio.run(workflow.execute_manual_recall())

    assert result == (True, "手动撤回成功", True)
    assert api_calls == [
        ("adapter.napcat.message.delete_msg", {"message_id": 202}),
    ]
    assert sent_texts == ["✅ 已撤回"]


def test_manual_recall_falls_back_to_previous_image_after_delete_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "MaiBot.db"
    _create_message_db(db_path)
    _insert_plugin_image(db_path, message_id="301", timestamp=1_900.0)
    _insert_plugin_image(db_path, message_id="302", timestamp=1_950.0)

    recalled_ids: list[int] = []
    sent_texts: list[str] = []

    async def call_api(_name: str, **kwargs: object) -> dict[str, object]:
        message_id = int(kwargs["message_id"])
        recalled_ids.append(message_id)
        return {"success": message_id == 301}

    async def send_text(text: str, storage_message: bool = True) -> bool:
        sent_texts.append(text)
        return True

    workflow = RecallWorkflow(
        config={"auto_recall": {"manual_max_age_seconds": 600, "id_wait_seconds": 0}},
        stream_id="stream-1",
        context=types.SimpleNamespace(api=types.SimpleNamespace(call=call_api)),
        send_text=send_text,
        track_task=lambda _task: None,
        log_prefix="test",
        db_path=db_path,
        napcat_config_path=tmp_path / "missing-napcat.toml",
        recent_recall_state={},
        wall_clock=lambda: 2_000.0,
    )

    result = asyncio.run(workflow.execute_manual_recall())

    assert result == (True, "手动撤回成功", True)
    assert recalled_ids == [302, 301]
    assert sent_texts == ["✅ 已撤回"]


def test_auto_recall_tracks_task_and_deletes_image_matching_send_time(tmp_path: Path) -> None:
    db_path = tmp_path / "MaiBot.db"
    _create_message_db(db_path)
    _insert_plugin_image(db_path, message_id="401", timestamp=1_975.0)

    recalled_ids: list[int] = []
    tracked_tasks: list[asyncio.Task[object]] = []

    async def call_api(_name: str, **kwargs: object) -> dict[str, object]:
        recalled_ids.append(int(kwargs["message_id"]))
        return {"success": True}

    async def send_text(_text: str, storage_message: bool = True) -> bool:
        pytest.fail("自动撤回不应发送手动撤回终态文本")

    workflow = RecallWorkflow(
        config={"auto_recall": {"delay_seconds": 0, "id_wait_seconds": 0}},
        stream_id="stream-1",
        context=types.SimpleNamespace(api=types.SimpleNamespace(call=call_api)),
        send_text=send_text,
        track_task=tracked_tasks.append,
        log_prefix="test",
        db_path=db_path,
        napcat_config_path=tmp_path / "missing-napcat.toml",
        recent_recall_state={},
    )

    async def run_scenario() -> None:
        await workflow.schedule_auto_recall(enabled=True, send_timestamp=1_975.0)
        assert len(tracked_tasks) == 1
        await tracked_tasks[0]

    asyncio.run(run_scenario())

    assert recalled_ids == [401]
