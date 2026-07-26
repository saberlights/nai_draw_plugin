import asyncio
import sys
import threading
from pathlib import Path

import pytest


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.blocking_io_runner import BlockingIORunner


def test_runner_executes_parallel_calls_and_closes_idempotently() -> None:
    runner = BlockingIORunner(thread_name_prefix="test-blocking-io")
    parallel_runner = BlockingIORunner(
        thread_name_prefix="test-parallel-blocking-io",
        max_workers=2,
    )
    rendezvous = threading.Barrier(2, timeout=2.0)

    def blocking_call(value: int) -> int:
        rendezvous.wait()
        return value

    async def scenario() -> None:
        assert await runner.run(lambda: 42) == 42
        assert await asyncio.gather(
            parallel_runner.run(blocking_call, 1),
            parallel_runner.run(blocking_call, 2),
        ) == [1, 2]

        runner.close()
        parallel_runner.close()
        runner.close()
        with pytest.raises(RuntimeError, match="已关闭"):
            await runner.run(lambda: 0)

    try:
        asyncio.run(scenario())
    finally:
        runner.close()
        parallel_runner.close()


def test_runner_waits_for_cancelled_call_and_cleans_returned_result() -> None:
    runner = BlockingIORunner(thread_name_prefix="test-cancelled-blocking-io")
    started = threading.Event()
    release = threading.Event()
    cleaned: list[str] = []

    def blocking_call() -> str:
        started.set()
        release.wait(timeout=2.0)
        return "response"

    async def scenario() -> None:
        task = asyncio.create_task(
            runner.run(
                blocking_call,
                cancel_result=cleaned.append,
            )
        )
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(scenario())
    finally:
        runner.close()

    assert cleaned == ["response"]
