"""命令 / 自动跟图发出的图跳过 VLM 识图的回写逻辑测试。

覆盖 `_send_base64_image_result` 按 `source` 分流（命令/跟图回写、action 保留识图），
以及 `_register_self_image_as_processed` 的哈希对齐、`vlm_processed=True` 置位与兜底描述。
"""

import asyncio
import base64
import hashlib
import os
import sys
import types
from pathlib import Path

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

# --- stub 主程序依赖，使 sdk_runtime 可在隔离环境下导入（与其它 sdk_runtime 测试同款）---
dummy_logger_module = types.ModuleType("src.common.logger")


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


dummy_logger_module.get_logger = lambda _name=None: _DummyLogger()
sys.modules["src.common.logger"] = dummy_logger_module

src_package = types.ModuleType("src")
src_package.__path__ = [os.path.join(MAIBOT_ROOT, "src")]
sys.modules.setdefault("src", src_package)

src_config_package = types.ModuleType("src.config")
src_config_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "config")]
sys.modules.setdefault("src.config", src_config_package)

config_module = types.ModuleType("src.config.config")
config_module.global_config = types.SimpleNamespace()
config_module.model_config = types.SimpleNamespace(
    model_task_config=types.SimpleNamespace(embedding=None)
)
sys.modules["src.config.config"] = config_module

model_configs_module = types.ModuleType("src.config.model_configs")
model_configs_module.TaskConfig = type("TaskConfig", (), {})
sys.modules["src.config.model_configs"] = model_configs_module

src_llm_models_package = types.ModuleType("src.llm_models")
src_llm_models_package.__path__ = [os.path.join(MAIBOT_ROOT, "src", "llm_models")]
sys.modules.setdefault("src.llm_models", src_llm_models_package)

utils_model_module = types.ModuleType("src.llm_models.utils_model")
utils_model_module.LLMOrchestrator = type(
    "LLMOrchestrator", (), {"__init__": lambda self, *a, **k: None}
)
sys.modules["src.llm_models.utils_model"] = utils_model_module

src_services_module = types.ModuleType("src.services")
src_services_module.llm_service = types.SimpleNamespace()
sys.modules["src.services"] = src_services_module

tag_retriever_module = types.ModuleType("plugins.nai_draw_plugin.core.services.tag_retriever")
tag_retriever_module.get_tag_retriever = lambda **_kwargs: None
sys.modules.setdefault("plugins.nai_draw_plugin.core.services.tag_retriever", tag_retriever_module)

mixins_package = types.ModuleType("plugins.nai_draw_plugin.core.mixins")
mixins_package.__path__ = [os.path.join(MAIBOT_ROOT, "plugins", "nai_draw_plugin", "core", "mixins")]
sys.modules.setdefault("plugins.nai_draw_plugin.core.mixins", mixins_package)

from plugins.nai_draw_plugin.sdk_runtime import NaiInvocation


def _build_invocation(source: str) -> NaiInvocation:
    """绕过 __init__ 构造一个只带必要字段的 NaiInvocation。"""
    invocation = object.__new__(NaiInvocation)
    invocation.source = source
    invocation.log_prefix = "test"
    invocation.stream_id = "stream-1"
    return invocation


class _FakeRecord:
    """模拟 image_manager.get_image_from_db 返回的 MaiImage 记录。"""

    def __init__(self, image_hash: str) -> None:
        self.file_hash = image_hash
        self.description = ""
        self.vlm_processed = False


def _install_fake_image_manager(record, update_results: list):
    """把 image_manager 模块替换为可观测的桩，返回捕获到的调用记录。"""
    captured: dict = {}

    class _FakeImageManager:
        def get_image_from_db(self, image_hash):
            captured["get_hash"] = image_hash
            return record

        def update_image_description(self, image):
            captured["updated"] = image
            update_results.append(image)
            return True

    sys.modules.setdefault("src.chat", types.ModuleType("src.chat"))
    sys.modules.setdefault("src.chat.image_system", types.ModuleType("src.chat.image_system"))
    module = types.ModuleType("src.chat.image_system.image_manager")
    module.image_manager = _FakeImageManager()
    sys.modules["src.chat.image_system.image_manager"] = module
    return captured


# --------- _skip_self_vlm：按 source 分流 ---------

def test_skip_self_vlm_true_for_command_and_reply_auto_draw() -> None:
    assert _build_invocation("command")._skip_self_vlm() is True
    assert _build_invocation("reply_auto_draw")._skip_self_vlm() is True


def test_skip_self_vlm_false_for_action() -> None:
    assert _build_invocation("action")._skip_self_vlm() is False


# --------- _send_base64_image_result：命令回写、action 保留识图 ---------

def test_command_image_triggers_recognition_skip_writeback() -> None:
    invocation = _build_invocation("command")
    register_calls: list = []
    send_calls: list = []

    async def fake_send_custom(message_type, content, *, display_message="", storage_message=True):
        send_calls.append((message_type, content, display_message))
        return True

    async def fake_register(image_base64, description):
        register_calls.append((image_base64, description))

    invocation.send_custom = fake_send_custom
    invocation._register_self_image_as_processed = fake_register

    ok = asyncio.run(
        invocation._send_base64_image_result("Zm9v", "disp", image_description="1girl, cat")
    )

    assert ok is True
    assert send_calls == [("image", "Zm9v", "disp")]
    assert register_calls == [("Zm9v", "1girl, cat")]


def test_action_image_keeps_vlm_and_skips_writeback() -> None:
    invocation = _build_invocation("action")
    register_calls: list = []

    async def fake_send_custom(message_type, content, *, display_message="", storage_message=True):
        return True

    async def fake_register(image_base64, description):
        register_calls.append((image_base64, description))

    invocation.send_custom = fake_send_custom
    invocation._register_self_image_as_processed = fake_register

    ok = asyncio.run(
        invocation._send_base64_image_result("Zm9v", "disp", image_description="1girl, cat")
    )

    assert ok is True
    # action 路径必须保留识图：绝不触发跳过识图回写
    assert register_calls == []


# --------- _register_self_image_as_processed：哈希对齐、置位、兜底、缺记录 ---------

def test_writeback_marks_processed_with_host_aligned_hash() -> None:
    invocation = _build_invocation("command")
    raw = b"some-real-image-bytes"
    image_b64 = base64.b64encode(raw).decode("ascii")
    expected_hash = hashlib.sha256(raw).hexdigest()

    record = _FakeRecord(expected_hash)
    updated: list = []
    captured = _install_fake_image_manager(record, updated)

    asyncio.run(invocation._register_self_image_as_processed(image_b64, "1girl, ghost"))

    # 哈希必须与宿主 _build_binary_component_from_base64 的 sha256(原始字节) 一致
    assert captured["get_hash"] == expected_hash
    assert record.vlm_processed is True
    assert record.description == "1girl, ghost"
    assert updated == [record]


def test_writeback_handles_data_uri_prefix() -> None:
    invocation = _build_invocation("command")
    raw = b"png-bytes"
    expected_hash = hashlib.sha256(raw).hexdigest()
    image_b64 = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

    record = _FakeRecord(expected_hash)
    captured = _install_fake_image_manager(record, [])

    asyncio.run(invocation._register_self_image_as_processed(image_b64, "desc"))

    # 带 data:URI 前缀时仍按纯字节算哈希，保证与入库记录命中
    assert captured["get_hash"] == expected_hash
    assert record.vlm_processed is True


def test_writeback_falls_back_to_placeholder_when_description_blank() -> None:
    invocation = _build_invocation("reply_auto_draw")
    raw = b"x"
    image_b64 = base64.b64encode(raw).decode("ascii")
    record = _FakeRecord(hashlib.sha256(raw).hexdigest())
    _install_fake_image_manager(record, [])

    asyncio.run(invocation._register_self_image_as_processed(image_b64, "   "))

    # 描述为空白会让缓存命中条件（description 非空）失效，必须兜底为占位
    assert record.description == "[由 NovelAI 生成的图片]"
    assert record.vlm_processed is True


def test_writeback_noop_when_image_not_in_db() -> None:
    invocation = _build_invocation("command")
    raw = b"missing"
    image_b64 = base64.b64encode(raw).decode("ascii")
    updated: list = []
    _install_fake_image_manager(None, updated)

    # 图未入库（imageurl 直发 / storage_message=False）时安静跳过，不得调用 update
    asyncio.run(invocation._register_self_image_as_processed(image_b64, "desc"))

    assert updated == []
