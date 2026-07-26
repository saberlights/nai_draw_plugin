import asyncio
import sys
from pathlib import Path

import pytest


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.blocking_io_runner import BlockingIORunner


def test_runner_executes_blocking_function_and_closes_idempotently() -> None:
    runner = BlockingIORunner(thread_name_prefix="test-blocking-io")

    assert asyncio.run(runner.run(lambda: 42)) == 42

    runner.close()
    runner.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        asyncio.run(runner.run(lambda: 0))
