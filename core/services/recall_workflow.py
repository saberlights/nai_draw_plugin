"""插件图片的自动与手动撤回编排。"""

from __future__ import annotations

import asyncio
import json
import time
import tomllib
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout

from src.common.logger import get_logger

from ...runtime_recall import (
    MANUAL_RECALL_TTL_SECONDS,
    extract_plugin_row_message_id,
    is_napcat_action_accepted,
    load_recent_plugin_image_rows,
    load_recent_session_image_rows,
    load_recent_tracked_plugin_image_rows,
    normalize_db_timestamp,
    prune_recent_ids,
    remember_recent_id,
    resolve_db_path,
    select_recent_plugin_image_row,
    wait_for_formal_message_id,
)
from ..constants import NAI_PIC_IMAGE_DISPLAY_MARKER


logger = get_logger("nai_draw_plugin")
_PLUGIN_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = resolve_db_path(_PLUGIN_ROOT / "sdk_runtime.py")
_DEFAULT_NAPCAT_CONFIG_PATH = _PLUGIN_ROOT.parent / "MaiBot-Napcat-Adapter" / "config.toml"
_RECENT_MANUAL_RECALL_IDS: dict[str, dict[str, float]] = {}


class RecallWorkflow:
    """在一个小 Interface 后集中图片定位、平台删除与终态反馈。"""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        stream_id: str,
        context: Any,
        send_text: Callable[..., Awaitable[bool]],
        start_task: Callable[..., asyncio.Task[Any] | None],
        log_prefix: str,
        db_path: str | Path = _DEFAULT_DB_PATH,
        napcat_config_path: str | Path = _DEFAULT_NAPCAT_CONFIG_PATH,
        recent_recall_state: dict[str, dict[str, float]] | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._stream_id = str(stream_id or "")
        self._context = context
        self._send_text = send_text
        self._start_task = start_task
        self._log_prefix = str(log_prefix or "nai_draw_plugin")
        self._db_path = Path(db_path)
        self._napcat_config_path = Path(napcat_config_path)
        self._recent_recall_state = (
            _RECENT_MANUAL_RECALL_IDS if recent_recall_state is None else recent_recall_state
        )
        self._wall_clock = wall_clock

    async def schedule_auto_recall(
        self,
        *,
        enabled: bool,
        send_timestamp: float | None,
    ) -> None:
        """若当前会话启用自动撤回，创建并登记一次撤回任务。"""
        if not enabled or not self._stream_id:
            return

        delay_seconds = self._positive_config_float("auto_recall.delay_seconds", 5.0)
        id_wait_seconds = self._positive_config_float("auto_recall.id_wait_seconds", 15.0)

        async def _job() -> None:
            await asyncio.sleep(delay_seconds)
            message_id = await self._resolve_local_message_id(
                limit=120,
                target_send_timestamp=send_timestamp,
                id_wait_seconds=id_wait_seconds,
            )
            if not message_id:
                logger.warning("%s 自动撤回未命中消息", self._log_prefix)
                return
            if await self._recall_message(message_id):
                logger.info("%s 已自动撤回消息 %s", self._log_prefix, message_id)
            else:
                logger.warning("%s 自动撤回失败: %s", self._log_prefix, message_id)

        self._start_task(_job, name="nai-auto-recall")

    async def execute_manual_recall(self) -> tuple[bool, str | None, bool]:
        """按新到旧顺序撤回当前会话最近一张仍在平台时限内的插件图片。"""
        recent_excludes = self._recent_manual_recall_ids()
        attempted_ids: set[str] = set(recent_excludes)
        max_age_seconds = self._positive_config_float(
            "auto_recall.manual_max_age_seconds",
            3600.0,
        )
        current_time = self._wall_clock()
        skipped_stale_rows = False
        attempted_recall = False

        for _ in range(5):
            row = await self._find_last_plugin_image_row(
                limit=300,
                exclude_message_ids=attempted_ids,
            )
            initial_message_id = extract_plugin_row_message_id(row)
            if not initial_message_id:
                break

            target_send_timestamp = normalize_db_timestamp(row.get("timestamp")) if row else None
            if (
                max_age_seconds > 0
                and target_send_timestamp is not None
                and current_time - target_send_timestamp > max_age_seconds
            ):
                skipped_stale_rows = True
                attempted_ids.add(initial_message_id)
                logger.info(
                    "%s [手动撤回] 跳过超出撤回窗口的图片: %s age=%.1fs",
                    self._log_prefix,
                    initial_message_id,
                    current_time - target_send_timestamp,
                )
                continue

            resolved_message_id = await self._resolve_local_message_id(
                limit=300,
                target_send_timestamp=target_send_timestamp,
                exclude_message_ids=attempted_ids,
                initial_row=row,
            )
            message_id = str(resolved_message_id or initial_message_id).strip()
            current_attempt_ids = {initial_message_id, message_id}
            current_attempt_ids.discard("")
            attempted_ids.update(current_attempt_ids)

            logger.info("%s [手动撤回] 准备撤回消息: %s", self._log_prefix, message_id)
            attempted_recall = True
            if await self._recall_message(message_id):
                for recent_id in current_attempt_ids:
                    self._remember_manual_recall_id(recent_id)
                await self._send_text("✅ 已撤回", storage_message=False)
                return True, "手动撤回成功", True

            logger.warning("%s [手动撤回] 撤回失败，尝试上一条图片", self._log_prefix)

        for recent_id in attempted_ids:
            self._remember_manual_recall_id(recent_id)

        if attempted_ids == recent_excludes or (skipped_stale_rows and not attempted_recall):
            logger.info("%s [手动撤回] 未找到可撤回的图片消息", self._log_prefix)
            not_found_text = "❌ 找不到可撤回的图片（直接发送 /nai 撤回 即可按顺序撤回最近一张）"
            if skipped_stale_rows:
                not_found_text = "❌ 找不到近期可撤回的图片（图片可能已超过平台撤回时限）"
            await self._send_text(not_found_text, storage_message=False)
            return False, "找不到可撤回的消息", True

        await self._send_text(
            "❌ 撤回失败（可能消息已被删除、超过撤回时限、或 bot 无权撤回）",
            storage_message=False,
        )
        return False, "手动撤回失败", True

    def _get_config(self, key: str, default: Any = None) -> Any:
        current: Any = self._config
        for part in str(key or "").split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def _positive_config_float(self, key: str, default: float) -> float:
        raw_value = self._get_config(key, default)
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return default

    def _recent_manual_recall_ids(self) -> set[str]:
        return prune_recent_ids(
            self._recent_recall_state,
            self._stream_id,
            ttl_seconds=MANUAL_RECALL_TTL_SECONDS,
        )

    def _remember_manual_recall_id(self, message_id: str) -> None:
        remember_recent_id(
            self._recent_recall_state,
            self._stream_id,
            message_id,
        )

    async def _find_last_plugin_image_row(
        self,
        *,
        limit: int,
        target_send_timestamp: float | None = None,
        exclude_message_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if not self._stream_id:
            return None

        tracked_row = select_recent_plugin_image_row(
            load_recent_tracked_plugin_image_rows(self._stream_id, limit=limit),
            target_send_timestamp=target_send_timestamp,
            exclude_message_ids=exclude_message_ids,
        )
        if tracked_row is not None:
            return tracked_row

        marked_row = select_recent_plugin_image_row(
            load_recent_plugin_image_rows(
                self._db_path,
                self._stream_id,
                NAI_PIC_IMAGE_DISPLAY_MARKER,
                limit=limit,
            ),
            target_send_timestamp=target_send_timestamp,
            exclude_message_ids=exclude_message_ids,
        )
        if marked_row is not None:
            return marked_row

        return select_recent_plugin_image_row(
            load_recent_session_image_rows(
                self._db_path,
                self._stream_id,
                limit=limit,
            ),
            target_send_timestamp=target_send_timestamp,
            exclude_message_ids=exclude_message_ids,
        )

    async def _resolve_local_message_id(
        self,
        *,
        limit: int,
        target_send_timestamp: float | None = None,
        exclude_message_ids: set[str] | None = None,
        initial_row: dict[str, Any] | None = None,
        id_wait_seconds: float | None = None,
    ) -> str | None:
        wait_seconds = (
            self._positive_config_float("auto_recall.id_wait_seconds", 15.0)
            if id_wait_seconds is None
            else max(0.0, float(id_wait_seconds))
        )

        async def _row_loader() -> dict[str, Any] | None:
            try:
                return await self._find_last_plugin_image_row(
                    limit=limit,
                    target_send_timestamp=target_send_timestamp,
                    exclude_message_ids=exclude_message_ids,
                )
            except Exception as exc:
                logger.warning("%s 轮询本地消息库失败: %r", self._log_prefix, exc)
                return None

        return await wait_for_formal_message_id(
            _row_loader,
            initial_row=initial_row,
            id_wait_seconds=wait_seconds,
        )

    async def _recall_message(self, message_id: str) -> bool:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or not self._stream_id:
            return False
        if not normalized_message_id.isdigit():
            logger.warning("%s 撤回失败：消息ID不是纯数字: %s", self._log_prefix, normalized_message_id)
            return False
        if await self._try_direct_napcat_action(normalized_message_id):
            return True
        return await self._try_napcat_delete_api(normalized_message_id)

    def _load_napcat_server_config(self) -> dict[str, Any] | None:
        if not self._napcat_config_path.is_file():
            return None

        try:
            with self._napcat_config_path.open("rb") as fp:
                config_data = tomllib.load(fp)
        except Exception as exc:
            logger.warning("%s 读取 Napcat 配置失败: %r", self._log_prefix, exc)
            return None

        server_config = config_data.get("napcat_server")
        if not isinstance(server_config, dict):
            return None

        host = str(server_config.get("host") or "").strip()
        token = str(server_config.get("token") or "").strip()
        try:
            port = int(server_config.get("port"))
        except (TypeError, ValueError):
            return None
        try:
            timeout = max(1.0, float(server_config.get("action_timeout_sec", 15.0)))
        except (TypeError, ValueError):
            timeout = 15.0
        if not host or port <= 0:
            return None

        return {
            "ws_url": f"ws://{host}:{port}",
            "token": token,
            "action_timeout_sec": timeout,
        }

    async def _try_direct_napcat_action(self, message_id: str) -> bool:
        server_config = self._load_napcat_server_config()
        if server_config is None:
            logger.warning("%s 未找到可用的 Napcat 连接配置，无法直连撤回", self._log_prefix)
            return False

        headers = (
            {"Authorization": f"Bearer {server_config['token']}"}
            if server_config.get("token")
            else {}
        )
        timeout = float(server_config["action_timeout_sec"])
        echo_id = uuid4().hex
        payload = {
            "action": "delete_msg",
            "params": {"message_id": int(message_id)},
            "echo": echo_id,
        }

        try:
            async with ClientSession(
                headers=headers,
                timeout=ClientTimeout(total=None, connect=10),
            ) as session:
                async with session.ws_connect(str(server_config["ws_url"]), heartbeat=None) as ws:
                    await ws.send_str(json.dumps(payload, ensure_ascii=False))
                    deadline = asyncio.get_running_loop().time() + timeout
                    while True:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise TimeoutError(f"Napcat delete_msg 超时 ({timeout}s)")
                        message = await asyncio.wait_for(ws.receive(), timeout=remaining)
                        if message.type.name != "TEXT":
                            continue
                        response = json.loads(message.data)
                        if str(response.get("echo") or "").strip() != echo_id:
                            continue
                        logger.debug("%s 撤回(napcat-ws) 结果: %r", self._log_prefix, response)
                        return is_napcat_action_accepted(response)
        except Exception as exc:
            logger.warning("%s 撤回(napcat-ws) 失败: %r", self._log_prefix, exc)
            return False

    async def _try_napcat_delete_api(self, message_id: str) -> bool:
        try:
            api_proxy = getattr(self._context, "api", None)
            if api_proxy is not None and hasattr(api_proxy, "call"):
                result = await api_proxy.call(
                    "adapter.napcat.message.delete_msg",
                    message_id=int(message_id),
                )
            else:
                call_capability = getattr(self._context, "call_capability", None)
                if not callable(call_capability):
                    raise AttributeError("当前上下文不支持 API 能力调用")
                result = await call_capability(
                    "api.call",
                    api_name="adapter.napcat.message.delete_msg",
                    args={"message_id": int(message_id)},
                )
            logger.debug("%s 撤回(napcat-api) 结果: %r", self._log_prefix, result)
            return is_napcat_action_accepted(result)
        except Exception as exc:
            logger.warning("%s 撤回(napcat-api) 失败: %r", self._log_prefix, exc)
            return False
