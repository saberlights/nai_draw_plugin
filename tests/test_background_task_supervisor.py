import asyncio
import sys
from pathlib import Path


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.background_task_supervisor import (
    BackgroundTaskSupervisor,
)


class _Logger:
    def __init__(self) -> None:
        self.errors: list[tuple[object, ...]] = []

    def error(self, *args: object, **_kwargs: object) -> None:
        self.errors.append(args)


def test_success_and_failure_each_run_finalizer_once() -> None:
    async def scenario() -> tuple[list[str], list[str], _Logger]:
        logger = _Logger()
        supervisor = BackgroundTaskSupervisor(logger=logger)
        finalized: list[str] = []
        failures: list[str] = []

        async def success() -> None:
            return None

        async def failure() -> None:
            raise RuntimeError("generation failed")

        async def on_failure(exc: Exception) -> None:
            failures.append(str(exc))

        success_task = supervisor.start(
            success,
            name="success",
            finalize=lambda: finalized.append("success"),
        )
        failure_task = supervisor.start(
            failure,
            name="failure",
            on_failure=on_failure,
            finalize=lambda: finalized.append("failure"),
        )
        assert success_task is not None
        assert failure_task is not None
        await asyncio.gather(success_task, failure_task)
        return finalized, failures, logger

    finalized, failures, logger = asyncio.run(scenario())

    assert finalized == ["success", "failure"]
    assert failures == ["generation failed"]
    assert len(logger.errors) == 1
    assert logger.errors[0][1] == "failure"


def test_cancellation_finalizes_without_failure_callback_or_error_log() -> None:
    async def scenario() -> tuple[list[str], list[str], _Logger]:
        logger = _Logger()
        supervisor = BackgroundTaskSupervisor(logger=logger)
        finalized: list[str] = []
        failures: list[str] = []
        started = asyncio.Event()

        async def job() -> None:
            started.set()
            await asyncio.Event().wait()

        task = supervisor.start(
            job,
            name="cancelled",
            on_failure=lambda exc: failures.append(str(exc)),
            finalize=lambda: finalized.append("cancelled"),
        )
        assert task is not None
        await started.wait()
        await supervisor.shutdown()
        await supervisor.shutdown()
        return finalized, failures, logger

    finalized, failures, logger = asyncio.run(scenario())

    assert finalized == ["cancelled"]
    assert failures == []
    assert logger.errors == []


def test_shutdown_closes_gate_before_cancelling_tasks() -> None:
    async def scenario() -> tuple[bool, int]:
        supervisor = BackgroundTaskSupervisor(logger=_Logger())
        started = asyncio.Event()
        rejected_factory_calls = 0

        async def rejected_job() -> None:
            nonlocal rejected_factory_calls
            rejected_factory_calls += 1

        async def running_job() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                assert supervisor.start(rejected_job, name="late") is None

        task = supervisor.start(running_job, name="running")
        assert task is not None
        await started.wait()
        await supervisor.shutdown()
        return supervisor.is_closing, rejected_factory_calls

    is_closing, rejected_factory_calls = asyncio.run(scenario())

    assert is_closing is True
    assert rejected_factory_calls == 0


def test_finalizer_cancellation_does_not_leave_shutdown_waiting_forever() -> None:
    async def scenario() -> int:
        supervisor = BackgroundTaskSupervisor(logger=_Logger())
        finalized = 0
        started = asyncio.Event()

        async def job() -> None:
            started.set()
            await asyncio.Event().wait()

        def finalize() -> None:
            nonlocal finalized
            finalized += 1

        task = supervisor.start(job, name="cancel-finalize", finalize=finalize)
        assert task is not None
        await started.wait()
        await supervisor.shutdown()
        return finalized

    assert asyncio.run(scenario()) == 1
