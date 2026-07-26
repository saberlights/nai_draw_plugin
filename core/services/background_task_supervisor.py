"""插件后台任务的终端生命周期管理。"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable


JobFactory = Callable[[], Awaitable[Any]]
FailureCallback = Callable[[Exception], Any]
Finalizer = Callable[[], Any]


class BackgroundTaskSupervisor:
    """集中任务创建、异常观察、终态清理与卸载取消。"""

    def __init__(self, *, logger: Any) -> None:
        self._logger = logger
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closing = False

    @property
    def is_closing(self) -> bool:
        return self._closing

    def start(
        self,
        job_factory: JobFactory,
        *,
        name: str,
        on_failure: FailureCallback | None = None,
        finalize: Finalizer | None = None,
    ) -> asyncio.Task[Any] | None:
        """启动并监督任务；卸载开始后拒绝且不调用 factory。"""
        if self._closing:
            return None

        task_name = str(name or "background-task")

        async def _runner() -> None:
            try:
                await job_factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error(
                    "%s 后台任务异常: %r",
                    task_name,
                    exc,
                    exc_info=True,
                )
                await self._invoke_failure_callback(on_failure, exc, task_name)
            finally:
                await self._invoke_finalizer(finalize, task_name)

        task = asyncio.create_task(_runner(), name=task_name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self) -> None:
        """停止接单，取消并等待所有已登记任务终态清理。"""
        self._closing = True
        while self._tasks:
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _invoke_failure_callback(
        self,
        callback: FailureCallback | None,
        exc: Exception,
        task_name: str,
    ) -> None:
        if callback is None:
            return
        try:
            result = callback(exc)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as callback_exc:
            self._logger.error(
                "%s 后台任务失败回调异常: %r",
                task_name,
                callback_exc,
                exc_info=True,
            )

    async def _invoke_finalizer(
        self,
        finalizer: Finalizer | None,
        task_name: str,
    ) -> None:
        if finalizer is None:
            return
        try:
            result = finalizer()
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.error(
                "%s 后台任务清理异常: %r",
                task_name,
                exc,
                exc_info=True,
            )
