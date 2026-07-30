"""插件后台任务的终端生命周期管理。"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable


JobFactory = Callable[[], Awaitable[Any]]
BeforeStart = Callable[[], Awaitable[bool] | bool]
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

    async def run(self, job_factory: JobFactory, *, name: str) -> Any:
        """在当前 Task 中运行前台工作，并纳入卸载取消范围。"""
        if self._closing:
            raise RuntimeError("插件正在卸载，拒绝新的前台调用")
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("前台工作必须运行在 asyncio Task 中")
        self._tasks.add(task)
        try:
            return await job_factory()
        finally:
            self._tasks.discard(task)

    async def submit(
        self,
        job_factory: JobFactory,
        *,
        before_start: BeforeStart,
        name: str,
        on_failure: FailureCallback | None = None,
        finalize: Finalizer | None = None,
    ) -> bool:
        """先登记提交任务，再执行确认步骤；确认成功后在同一任务中运行 job。"""
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[bool] = loop.create_future()
        confirmed = False

        async def _submission() -> None:
            nonlocal confirmed
            try:
                result = before_start()
                accepted = await result if inspect.isawaitable(result) else result
                if not accepted:
                    ready.set_result(False)
                    return
                confirmed = True
                ready.set_result(True)
                await job_factory()
            except asyncio.CancelledError:
                if not ready.done():
                    ready.set_result(False)
                raise
            except Exception as exc:
                if not ready.done():
                    ready.set_exception(exc)
                raise

        async def _on_failure(exc: Exception) -> None:
            if confirmed:
                await self._invoke_failure_callback(on_failure, exc, name)

        task = self.start(
            _submission,
            name=name,
            on_failure=_on_failure,
            finalize=finalize,
        )
        if task is None:
            await self._invoke_finalizer(finalize, name)
            return False
        try:
            accepted = await ready
        except BaseException:
            await asyncio.gather(task, return_exceptions=True)
            raise
        if not accepted:
            await asyncio.gather(task, return_exceptions=True)
        return accepted

    async def shutdown(self) -> None:
        """停止接单，取消并等待所有已登记任务终态清理。"""
        self._closing = True
        while self._tasks:
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # gather 对已终态任务不会让出事件循环，而 done_callback 里的 discard
            # 靠 call_soon 排队执行；不主动移除本轮任务会在这里空转死锁
            self._tasks.difference_update(tasks)

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
