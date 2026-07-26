"""将短时阻塞 I/O 与 asyncio 默认执行器隔离。"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar


ResultT = TypeVar("ResultT")


class BlockingIORunner:
    """使用可显式关闭的线程池 Adapter 执行阻塞 I/O。"""

    def __init__(self, *, thread_name_prefix: str, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._closed = False

    async def run(
        self,
        function: Callable[..., ResultT],
        *args: Any,
        cancel_result: Callable[[ResultT], Any] | None = None,
    ) -> ResultT:
        if self._closed:
            raise RuntimeError("BlockingIORunner 已关闭")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError as cancellation:
            # Python 3.11 的已取消 Task 再次 shield 同一 executor future 可能不恢复；
            # 低频检查终态，确保底层 I/O 结束并清理结果后才传播取消。
            try:
                while not future.done():
                    await asyncio.sleep(0.01)
                result = future.result()
            except Exception:
                raise cancellation
            if cancel_result is not None:
                try:
                    cancel_result(result)
                except Exception as cleanup_error:
                    cancellation.add_note(
                        f"阻塞调用取消结果清理失败: {cleanup_error!r}"
                    )
            raise cancellation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
