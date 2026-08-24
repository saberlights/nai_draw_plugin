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

from plugins.nai_draw_plugin.core.services.recall_workflow import (
    RecallWorkflow,
    _discover_napcat_config_path,
    _has_napcat_server_section,
)
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
        start_task=lambda job_factory, **_kwargs: asyncio.create_task(job_factory()),
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
        start_task=lambda job_factory, **_kwargs: asyncio.create_task(job_factory()),
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
        start_task=lambda job_factory, **_kwargs: asyncio.create_task(job_factory()),
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
        start_task=lambda job_factory, **_kwargs: (
            tracked_tasks.append(asyncio.create_task(job_factory())) or tracked_tasks[-1]
        ),
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


def _build_probe_workflow(config: dict, tmp_path: Path) -> RecallWorkflow:
    """造一个只用来看 _napcat_config_path 解析结果的 workflow。"""
    async def send_text(text: str, storage_message: bool = True) -> bool:
        return True

    return RecallWorkflow(
        config=config,
        stream_id="stream-1",
        context=types.SimpleNamespace(),
        send_text=send_text,
        start_task=lambda job_factory, **_kwargs: None,
        log_prefix="test",
        db_path=tmp_path / "MaiBot.db",
        recent_recall_state={},
    )


def test_has_napcat_server_section_rejects_unrelated_toml(tmp_path: Path) -> None:
    """只认带 [napcat_server] 的配置，别的适配器配置不能被误读。"""
    napcat = tmp_path / "napcat.toml"
    napcat.write_text('[napcat_server]\nhost = "127.0.0.1"\nport = 3001\n', encoding="utf-8")
    other = tmp_path / "other.toml"
    other.write_text('[telegram]\ntoken = "x"\n', encoding="utf-8")
    broken = tmp_path / "broken.toml"
    broken.write_text("这不是 toml = = =\n[[[", encoding="utf-8")

    assert _has_napcat_server_section(napcat) is True
    assert _has_napcat_server_section(other) is False
    assert _has_napcat_server_section(broken) is False
    assert _has_napcat_server_section(tmp_path / "nonexistent.toml") is False


def test_explicit_config_path_setting_wins(tmp_path: Path) -> None:
    """配置项 auto_recall.napcat_config_path 指定的路径优先于自动查找。"""
    custom = tmp_path / "my-adapter" / "config.toml"
    custom.parent.mkdir()
    custom.write_text('[napcat_server]\nhost = "10.0.0.1"\nport = 4001\n', encoding="utf-8")

    workflow = _build_probe_workflow(
        {"auto_recall": {"napcat_config_path": str(custom)}}, tmp_path
    )

    assert workflow._napcat_config_path == custom
    # 解析出来的连接信息要真的来自这份自定义配置
    assert workflow._load_napcat_server_config() == {
        "ws_url": "ws://10.0.0.1:4001",
        "token": "",
        "action_timeout_sec": 15.0,
    }


def test_discovery_finds_differently_named_adapter_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """适配器目录名不是 MaiBot-Napcat-Adapter 时也要能找到。

    回归 bug：目录名被硬编码成 ``MaiBot-Napcat-Adapter``，用户装成
    ``maibot-team_napcat-adapter`` 就整条直连通道失效，日志只有一句
    "未找到可用的 Napcat 连接配置"，很难联想到是目录名不匹配。
    """
    import plugins.nai_draw_plugin.core.services.recall_workflow as rw

    plugins_dir = tmp_path / "plugins"
    # 干扰项：名字不含 napcat，且自己也有 config.toml
    (plugins_dir / "some-other-adapter").mkdir(parents=True)
    (plugins_dir / "some-other-adapter" / "config.toml").write_text(
        '[telegram]\ntoken = "x"\n', encoding="utf-8"
    )
    target = plugins_dir / "maibot-team_napcat-adapter"
    target.mkdir()
    (target / "config.toml").write_text(
        '[napcat_server]\nhost = "127.0.0.1"\nport = 3001\n', encoding="utf-8"
    )

    monkeypatch.setattr(rw, "_PLUGINS_DIR", plugins_dir)
    monkeypatch.setattr(
        rw, "_DEFAULT_NAPCAT_CONFIG_PATH", plugins_dir / "MaiBot-Napcat-Adapter" / "config.toml"
    )

    assert _discover_napcat_config_path() == target / "config.toml"


def test_discovery_prefers_official_directory_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """官方目录名存在时直接用它，不去扫描。"""
    import plugins.nai_draw_plugin.core.services.recall_workflow as rw

    plugins_dir = tmp_path / "plugins"
    official = plugins_dir / "MaiBot-Napcat-Adapter"
    official.mkdir(parents=True)
    (official / "config.toml").write_text(
        '[napcat_server]\nhost = "127.0.0.1"\nport = 3001\n', encoding="utf-8"
    )
    other = plugins_dir / "zzz_napcat-adapter"
    other.mkdir()
    (other / "config.toml").write_text(
        '[napcat_server]\nhost = "9.9.9.9"\nport = 9999\n', encoding="utf-8"
    )

    monkeypatch.setattr(rw, "_PLUGINS_DIR", plugins_dir)
    monkeypatch.setattr(rw, "_DEFAULT_NAPCAT_CONFIG_PATH", official / "config.toml")

    assert _discover_napcat_config_path() == official / "config.toml"


def test_discovery_falls_back_to_default_when_nothing_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一个都找不到时返回默认路径，由调用方按"文件不存在"走 api.call 兜底。"""
    import plugins.nai_draw_plugin.core.services.recall_workflow as rw

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    default_path = plugins_dir / "MaiBot-Napcat-Adapter" / "config.toml"

    monkeypatch.setattr(rw, "_PLUGINS_DIR", plugins_dir)
    monkeypatch.setattr(rw, "_DEFAULT_NAPCAT_CONFIG_PATH", default_path)

    assert _discover_napcat_config_path() == default_path
    workflow = _build_probe_workflow({}, tmp_path)
    # 找不到配置时不能抛，只返回 None 让上层走 api.call
    assert workflow._load_napcat_server_config() is None
