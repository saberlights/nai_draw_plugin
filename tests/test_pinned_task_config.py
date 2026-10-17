"""使用真实宿主配置模型校验自定义任务，避免宽松 stub 掩盖 API 漂移。"""

from pathlib import Path
import subprocess
import sys


def test_pinned_task_config_matches_host_schema():
    plugin_root = Path(__file__).resolve().parents[1]
    host_root = plugin_root.parents[1]
    script = r'''
from unittest.mock import patch
import random
import runpy
import sys
import types

from pydantic import ValidationError
from src.config.model_configs import TaskConfig

# 只隔离编排器的运行时依赖；TaskConfig 必须使用宿主的真实校验规则。
module = types.ModuleType("src.llm_models.utils_model")
module.LLMOrchestrator = type("LLMOrchestrator", (), {})
with patch.dict(sys.modules, {"src.llm_models.utils_model": module}):
    build = runpy.run_path(sys.argv[1])["build_pinned_task_config"]

rng = random.Random(20261002)
for threshold in [None, 0.0, 30.0, 60.0, 3600.0] + [rng.uniform(1, 300) for _ in range(10)]:
    custom = {"model_list": [" vision-a ", "vision-b"], "max_tokens": "2048", "temperature": "0.5"}
    if threshold is not None:
        custom["slow_threshold"] = threshold
    config = build(custom, max_tokens=1024, temperature=0.7)
    assert isinstance(config, TaskConfig)
    assert config.model_list == ["vision-a", "vision-b"]
    assert config.max_tokens == 2048
    assert config.temperature == 0.5
    assert config.selection_strategy == "random"
    assert config.hard_timeout == TaskConfig().hard_timeout
    assert "slow_threshold" not in config.model_dump()

config = build({"model_list": "vision-a"}, max_tokens=1, temperature=0.0)
assert config.model_list == ["vision-a"]
assert config.max_tokens == 1
assert config.temperature == 0.0
for custom in [None, {}, {"model_list": []}, {"model_list": [" "]}]:
    assert build(custom, max_tokens=1024, temperature=0.7) is None

for field, value in [("max_tokens", 0), ("temperature", -0.1), ("temperature", 2.1)]:
    try:
        build({"model_list": ["vision-a"], field: value}, max_tokens=1024, temperature=0.7)
    except ValidationError as exc:
        assert exc.errors()[0]["loc"] == (field,)
    else:
        raise AssertionError(f"Invalid {field} was accepted")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(plugin_root / "core/services/pinned_task_orchestrator.py")],
        cwd=host_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
