"""将短时阻塞 I/O 与 asyncio 默认执行器隔离。"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar


ResultT = TypeVar("ResultT")


class BlockingIORunner:
    """使用可显式关闭的单线程 Adapter 执行短时阻塞 I/O。"""

    def __init__(self, *, thread_name_prefix: str) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )
        self._closed = False

    async def run(self, function: Callable[[], ResultT]) -> ResultT:
        if self._closed:
            raise RuntimeError("BlockingIORunner 已关闭")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, function)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
