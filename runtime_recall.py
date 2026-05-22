from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable


MANUAL_RECALL_TTL_SECONDS = 600.0
AUTO_RECALL_TIMESTAMP_TOLERANCE = 0.2
PLACEHOLDER_MESSAGE_ID_PREFIX = "send_api_"
PENDING_PLUGIN_IMAGE_SEND_TTL_SECONDS = 120.0
TRACKED_PLUGIN_IMAGE_ROW_TTL_SECONDS = 86400.0
PLUGIN_IMAGE_PENDING_MATCH_TOLERANCE_SECONDS = 10.0
PLUGIN_IMAGE_MARKER_CONFIG_KEY = "nai_draw_plugin_recall_marker"

_PENDING_PLUGIN_IMAGE_SENDS: dict[str, list[float]] = {}
_TRACKED_PLUGIN_IMAGE_ROWS: dict[str, list[dict[str, Any]]] = {}


def resolve_db_path(runtime_file: str | Path) -> Path:
    """根据插件运行时文件定位宿主消息数据库。"""
    return Path(runtime_file).resolve().parents[2] / "data" / "MaiBot.db"


def normalize_db_timestamp(value: Any) -> float | None:
    """将 SQLite 中的时间字段归一化为 Unix 时间戳。"""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, datetime):
        return value.timestamp()

    normalized = str(value).strip()
    if not normalized:
        return None

    try:
        return float(normalized)
    except ValueError:
        pass

    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load_recent_plugin_image_rows(
    db_path: str | Path,
    stream_id: str,
    marker: str,
    *,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """从宿主消息库中精确读取当前会话最近的本插件图片消息。"""
    normalized_stream_id = str(stream_id or "").strip()
    normalized_marker = str(marker or "").strip()
    normalized_limit = max(1, int(limit))
    resolved_db_path = Path(db_path).resolve()

    if not normalized_stream_id or not normalized_marker or not resolved_db_path.is_file():
        return []

    db_uri = f"file:{resolved_db_path.as_posix()}?mode=ro"
    connection = sqlite3.connect(db_uri, uri=True, timeout=1.0)
    connection.row_factory = sqlite3.Row

    try:
        columns = {
            str(row[1]).strip()
            for row in connection.execute("PRAGMA table_info(mai_messages)").fetchall()
            if len(row) >= 2 and str(row[1]).strip()
        }
        select_columns = [
            column_name
            for column_name in (
                "message_id",
                "timestamp",
                "session_id",
                "is_picture",
                "display_message",
                "processed_plain_text",
                "additional_config",
            )
            if column_name in columns
        ]
        if not select_columns:
            return []

        marker_columns = [
            column_name
            for column_name in ("additional_config", "display_message", "processed_plain_text")
            if column_name in columns
        ]
        if not marker_columns:
            return []

        marker_predicate = " OR ".join(f"{column_name} LIKE ?" for column_name in marker_columns)
        cursor = connection.execute(
            f"""
            SELECT
                {", ".join(select_columns)}
            FROM mai_messages
            WHERE session_id = ?
              AND ({marker_predicate})
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (
                normalized_stream_id,
                *(f"%{normalized_marker}%" for _ in marker_columns),
                normalized_limit,
            ),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def load_recent_session_image_rows(
    db_path: str | Path,
    stream_id: str,
    *,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """从宿主消息库中读取当前会话最近的图片消息，供旧数据兼容回退。"""
    normalized_stream_id = str(stream_id or "").strip()
    normalized_limit = max(1, int(limit))
    resolved_db_path = Path(db_path).resolve()

    if not normalized_stream_id or not resolved_db_path.is_file():
        return []

    db_uri = f"file:{resolved_db_path.as_posix()}?mode=ro"
    connection = sqlite3.connect(db_uri, uri=True, timeout=1.0)
    connection.row_factory = sqlite3.Row

    try:
        columns = {
            str(row[1]).strip()
            for row in connection.execute("PRAGMA table_info(mai_messages)").fetchall()
            if len(row) >= 2 and str(row[1]).strip()
        }
        select_columns = [
            column_name
            for column_name in (
                "message_id",
                "timestamp",
                "session_id",
                "is_picture",
                "display_message",
                "processed_plain_text",
                "additional_config",
            )
            if column_name in columns
        ]
        if not select_columns:
            return []

        cursor = connection.execute(
            f"""
            SELECT
                {", ".join(select_columns)}
            FROM mai_messages
            WHERE session_id = ?
              AND is_picture = 1
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (normalized_stream_id, normalized_limit),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def select_recent_plugin_image_row(
    rows: list[dict[str, Any]],
    *,
    target_send_timestamp: float | None = None,
    exclude_message_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """从候选消息中挑选最适合当前撤回场景的一条。"""
    excluded_ids = {str(item) for item in (exclude_message_ids or set()) if item}
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None

    for index, row in enumerate(rows):
        message_id = extract_plugin_row_message_id(row)
        if not message_id or message_id in excluded_ids:
            continue

        timestamp_value = normalize_db_timestamp(row.get("timestamp"))
        is_placeholder = is_placeholder_message_id(message_id)

        if target_send_timestamp is None:
            sort_key = (
                timestamp_value if timestamp_value is not None else float("-inf"),
                0.0 if is_placeholder else 1.0,
                -float(index),
            )
        else:
            if (
                timestamp_value is not None
                and timestamp_value + AUTO_RECALL_TIMESTAMP_TOLERANCE < target_send_timestamp
            ):
                continue

            sort_key = (
                0.0 if is_placeholder else 1.0,
                -abs(timestamp_value - target_send_timestamp)
                if timestamp_value is not None
                else float("-inf"),
                timestamp_value if timestamp_value is not None else float("-inf"),
                -float(index),
            )

        if best_key is None or sort_key > best_key:
            best_key = sort_key
            best_row = row

    return best_row


def extract_plugin_row_message_id(row: dict[str, Any] | None) -> str:
    """读取候选消息行中的 message_id。"""
    if not isinstance(row, dict):
        return ""
    return str(row.get("message_id", "") or "").strip()


def is_placeholder_message_id(message_id: str) -> bool:
    """判断消息 ID 是否仍是宿主发送阶段的占位值。"""
    return str(message_id or "").strip().startswith(PLACEHOLDER_MESSAGE_ID_PREFIX)


def reset_runtime_recall_tracking_state() -> None:
    """清空运行时发送跟踪状态。"""
    _PENDING_PLUGIN_IMAGE_SENDS.clear()
    _TRACKED_PLUGIN_IMAGE_ROWS.clear()


def remember_pending_plugin_image_send(
    stream_id: str,
    send_timestamp: float,
    *,
    now: float | None = None,
) -> None:
    """记录一条待匹配的本插件图片发送事件。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return

    normalized_timestamp = float(send_timestamp)
    current_time = time.time() if now is None else float(now)
    pending_list = [
        float(item)
        for item in _PENDING_PLUGIN_IMAGE_SENDS.get(normalized_stream_id, [])
        if current_time - float(item) <= PENDING_PLUGIN_IMAGE_SEND_TTL_SECONDS
    ]
    pending_list.append(normalized_timestamp)
    pending_list.sort(reverse=True)
    _PENDING_PLUGIN_IMAGE_SENDS[normalized_stream_id] = pending_list


def discard_pending_plugin_image_send(
    stream_id: str,
    send_timestamp: float,
    *,
    now: float | None = None,
) -> None:
    """丢弃一条未成功发送的待匹配图片事件。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return

    current_time = time.time() if now is None else float(now)
    normalized_timestamp = float(send_timestamp)
    filtered = []
    removed = False

    for item in _PENDING_PLUGIN_IMAGE_SENDS.get(normalized_stream_id, []):
        item_timestamp = float(item)
        if current_time - item_timestamp > PENDING_PLUGIN_IMAGE_SEND_TTL_SECONDS:
            continue
        if not removed and abs(item_timestamp - normalized_timestamp) <= PLUGIN_IMAGE_PENDING_MATCH_TOLERANCE_SECONDS:
            removed = True
            continue
        filtered.append(item_timestamp)

    if filtered:
        _PENDING_PLUGIN_IMAGE_SENDS[normalized_stream_id] = filtered
    else:
        _PENDING_PLUGIN_IMAGE_SENDS.pop(normalized_stream_id, None)


def _message_looks_like_image(message: dict[str, Any]) -> bool:
    """判断消息是否应被按图片消息追踪。"""
    if bool(message.get("is_picture", False)):
        return True

    raw_message = message.get("raw_message")
    if not isinstance(raw_message, list):
        return False

    for segment in raw_message:
        if not isinstance(segment, dict):
            continue

        segment_type = str(segment.get("type") or "").strip().lower()
        if segment_type in {"image", "emoji"}:
            return True

        if segment_type != "dict":
            continue

        segment_data = segment.get("data")
        if not isinstance(segment_data, dict):
            continue

        custom_type = str(segment_data.get("type") or "").strip().lower()
        if custom_type in {"image", "imageurl"}:
            return True

    return False


def attach_plugin_image_marker_to_message(
    message: dict[str, Any],
    marker: str,
    *,
    now: float | None = None,
) -> bool:
    """为本插件刚构建出的图片消息补充撤回标记。"""
    normalized_stream_id = str(message.get("session_id", "") or "").strip()
    if not normalized_stream_id or not _message_looks_like_image(message):
        return False

    message_timestamp = normalize_db_timestamp(message.get("timestamp"))
    if message_timestamp is None:
        return False

    if not _has_pending_plugin_image_send(
        normalized_stream_id,
        message_timestamp,
        now=now,
    ):
        return False

    message_info = message.get("message_info")
    if not isinstance(message_info, dict):
        return False

    additional_config = message_info.get("additional_config")
    if not isinstance(additional_config, dict):
        additional_config = {}
        message_info["additional_config"] = additional_config

    message["is_picture"] = True
    additional_config[PLUGIN_IMAGE_MARKER_CONFIG_KEY] = str(marker or "").strip()
    return True


def remember_sent_plugin_image_message(
    message: dict[str, Any],
    marker: str,
    *,
    now: float | None = None,
) -> bool:
    """记录一条已成功发送且属于本插件的图片消息。"""
    normalized_stream_id = str(message.get("session_id", "") or "").strip()
    normalized_message_id = str(message.get("message_id", "") or "").strip()
    if not normalized_stream_id or not normalized_message_id or not _message_looks_like_image(message):
        return False

    message_timestamp = normalize_db_timestamp(message.get("timestamp"))
    if message_timestamp is None:
        return False

    current_time = time.time() if now is None else float(now)
    marker_attached = _message_has_plugin_image_marker(message, marker)
    pending_matched = _consume_pending_plugin_image_send(
        normalized_stream_id,
        message_timestamp,
        now=current_time,
    )
    if not marker_attached and not pending_matched:
        return False

    message["is_picture"] = True
    retained_rows = []
    for row in _TRACKED_PLUGIN_IMAGE_ROWS.get(normalized_stream_id, []):
        row_timestamp = normalize_db_timestamp(row.get("timestamp"))
        if row_timestamp is None:
            continue
        if current_time - row_timestamp > TRACKED_PLUGIN_IMAGE_ROW_TTL_SECONDS:
            continue
        if extract_plugin_row_message_id(row) == normalized_message_id:
            continue
        retained_rows.append(row)

    retained_rows.insert(
        0,
        {
            "message_id": normalized_message_id,
            "timestamp": message_timestamp,
            "session_id": normalized_stream_id,
            "is_picture": 1,
        },
    )
    _TRACKED_PLUGIN_IMAGE_ROWS[normalized_stream_id] = retained_rows[:200]
    return True


def load_recent_tracked_plugin_image_rows(
    stream_id: str,
    *,
    limit: int = 120,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """读取运行期已确认属于本插件的图片消息记录。"""
    normalized_stream_id = str(stream_id or "").strip()
    normalized_limit = max(1, int(limit))
    if not normalized_stream_id:
        return []

    current_time = time.time() if now is None else float(now)
    retained_rows = []
    for row in _TRACKED_PLUGIN_IMAGE_ROWS.get(normalized_stream_id, []):
        row_timestamp = normalize_db_timestamp(row.get("timestamp"))
        if row_timestamp is None:
            continue
        if current_time - row_timestamp > TRACKED_PLUGIN_IMAGE_ROW_TTL_SECONDS:
            continue
        retained_rows.append(row)

    if retained_rows:
        _TRACKED_PLUGIN_IMAGE_ROWS[normalized_stream_id] = retained_rows
    else:
        _TRACKED_PLUGIN_IMAGE_ROWS.pop(normalized_stream_id, None)

    return [dict(row) for row in retained_rows[:normalized_limit]]


def _consume_pending_plugin_image_send(
    stream_id: str,
    message_timestamp: float,
    *,
    now: float | None = None,
) -> bool:
    """匹配并消费一条待发送图片记录。"""
    return _match_pending_plugin_image_send(
        stream_id,
        message_timestamp,
        now=now,
        consume=True,
    )


def _has_pending_plugin_image_send(
    stream_id: str,
    message_timestamp: float,
    *,
    now: float | None = None,
) -> bool:
    """判断当前消息时间是否命中一条待发送图片记录。"""
    return _match_pending_plugin_image_send(
        stream_id,
        message_timestamp,
        now=now,
        consume=False,
    )


def _match_pending_plugin_image_send(
    stream_id: str,
    message_timestamp: float,
    *,
    now: float | None = None,
    consume: bool,
) -> bool:
    """按时间戳匹配一条待发送图片记录。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return False

    current_time = time.time() if now is None else float(now)
    pending_list = [
        float(item)
        for item in _PENDING_PLUGIN_IMAGE_SENDS.get(normalized_stream_id, [])
        if current_time - float(item) <= PENDING_PLUGIN_IMAGE_SEND_TTL_SECONDS
    ]
    if not pending_list:
        _PENDING_PLUGIN_IMAGE_SENDS.pop(normalized_stream_id, None)
        return False

    matched_index = None
    matched_distance = None

    for index, item_timestamp in enumerate(pending_list):
        distance = abs(float(item_timestamp) - float(message_timestamp))
        if distance > PLUGIN_IMAGE_PENDING_MATCH_TOLERANCE_SECONDS:
            continue
        if matched_distance is None or distance < matched_distance:
            matched_index = index
            matched_distance = distance

    if matched_index is None:
        _PENDING_PLUGIN_IMAGE_SENDS[normalized_stream_id] = pending_list
        return False

    if consume:
        pending_list.pop(matched_index)

    if pending_list:
        _PENDING_PLUGIN_IMAGE_SENDS[normalized_stream_id] = pending_list
    else:
        _PENDING_PLUGIN_IMAGE_SENDS.pop(normalized_stream_id, None)
    return True


def _message_has_plugin_image_marker(message: dict[str, Any], marker: str) -> bool:
    """判断消息元数据里是否已有本插件图片标记。"""
    message_info = message.get("message_info")
    if not isinstance(message_info, dict):
        return False

    additional_config = message_info.get("additional_config")
    if not isinstance(additional_config, dict):
        return False

    return str(additional_config.get(PLUGIN_IMAGE_MARKER_CONFIG_KEY, "") or "").strip() == str(marker or "").strip()


async def wait_for_formal_message_id(
    row_loader: Callable[[], Awaitable[dict[str, Any] | None]],
    *,
    initial_row: dict[str, Any] | None = None,
    id_wait_seconds: float = 15.0,
    poll_interval: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> str | None:
    """若当前候选还是占位 ID，则优先等待正式 ID 落库。"""
    wait_seconds = max(0.0, float(id_wait_seconds))
    interval = (
        min(1.0, max(0.2, wait_seconds / 10))
        if poll_interval is None and wait_seconds
        else 0.5 if poll_interval is None
        else max(0.0, float(poll_interval))
    )
    deadline = monotonic() + wait_seconds
    fallback_message_id = extract_plugin_row_message_id(initial_row)
    row = initial_row

    while True:
        if row is None:
            row = await row_loader()

        message_id = extract_plugin_row_message_id(row)
        if message_id:
            fallback_message_id = message_id
            if not is_placeholder_message_id(message_id):
                return message_id

        if wait_seconds <= 0:
            return fallback_message_id or None

        if monotonic() >= deadline:
            return fallback_message_id or None

        await sleep(interval)
        row = None


def prune_recent_ids(
    recent_state: dict[str, dict[str, float]],
    stream_id: str,
    *,
    ttl_seconds: float = MANUAL_RECALL_TTL_SECONDS,
    now: float | None = None,
) -> set[str]:
    """清理并返回当前会话最近已尝试撤回过的消息 ID。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return set()

    current_time = time.monotonic() if now is None else float(now)
    recent_map = recent_state.get(normalized_stream_id, {})
    filtered = {
        str(message_id): float(timestamp)
        for message_id, timestamp in recent_map.items()
        if current_time - float(timestamp) <= float(ttl_seconds)
    }

    if filtered:
        recent_state[normalized_stream_id] = filtered
    else:
        recent_state.pop(normalized_stream_id, None)

    return set(filtered.keys())


def remember_recent_id(
    recent_state: dict[str, dict[str, float]],
    stream_id: str,
    message_id: str,
    *,
    now: float | None = None,
) -> None:
    """记录当前会话刚尝试撤回过的消息 ID。"""
    normalized_stream_id = str(stream_id or "").strip()
    normalized_message_id = str(message_id or "").strip()
    if not normalized_stream_id or not normalized_message_id:
        return

    current_time = time.monotonic() if now is None else float(now)
    recent_state.setdefault(normalized_stream_id, {})[normalized_message_id] = current_time


def build_recall_command_payloads(message_id: str) -> list[dict[str, Any]]:
    """构造兼容旧链撤回命令的候选载荷。"""
    normalized_message_id = str(message_id or "").strip()
    if not normalized_message_id:
        return []

    return [
        {"name": command_name, "args": {"message_id": normalized_message_id}}
        for command_name in ("DELETE_MSG", "delete_msg", "RECALL_MSG", "recall_msg")
    ]


def unwrap_sdk_api_result(result: Any) -> tuple[bool | None, Any]:
    """拆开 SDK `api.call` 包装后的结果。"""
    if not isinstance(result, dict) or "success" not in result:
        return None, result

    wrapper_success = bool(result.get("success"))
    payload = result.get("result")
    if payload is None and not wrapper_success and "error" in result:
        payload = result.get("error")
    return wrapper_success, payload


def is_napcat_action_accepted(result: Any) -> bool:
    """判断 Napcat 动作调用是否被平台接受。"""
    wrapper_success, payload = unwrap_sdk_api_result(result)
    if wrapper_success is False:
        return False
    if wrapper_success is True and payload is None:
        return True

    target = payload if wrapper_success is not None else result

    if isinstance(target, bool):
        return target
    if isinstance(target, str):
        return str(target).strip().lower() in {"ok", "success", "true", "1"}
    if not isinstance(target, dict):
        return bool(target)

    if "status" in target:
        status = str(target.get("status") or "").strip().lower()
        if status in {"ok", "success"}:
            return True
        if status in {"failed", "error"}:
            return False

    if "retcode" in target:
        return str(target.get("retcode")) == "0"

    if "code" in target:
        return str(target.get("code")) in {"0", "200"}

    if "success" in target:
        return bool(target.get("success"))
    if "ok" in target:
        return bool(target.get("ok"))

    return bool(target)


def does_napcat_message_exist(result: Any) -> bool:
    """判断 `get_msg` 查询结果是否仍指向一条存在的消息。"""
    wrapper_success, payload = unwrap_sdk_api_result(result)
    if wrapper_success is False:
        return False

    target = payload if wrapper_success is not None else result
    if target is None:
        return False

    if isinstance(target, dict):
        if "message_id" in target and str(target.get("message_id") or "").strip():
            return True

        data = target.get("data")
        if isinstance(data, dict) and str(data.get("message_id") or "").strip():
            return True

        nested_result = target.get("result")
        if isinstance(nested_result, dict) and str(nested_result.get("message_id") or "").strip():
            return True

    return bool(target)
