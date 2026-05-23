"""验证 on_load 注释回填：保留值、生成注释、幂等、不覆盖用户注释。"""

import sys
import types
from pathlib import Path

import pytest

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))


# ---- 复用 test_plugin_tag_retriever_registration 的 stub ----
maibot_sdk_stub = types.ModuleType("maibot_sdk")
maibot_sdk_stub.Action = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.Command = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.HookHandler = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.MaiBotPlugin = type("MaiBotPlugin", (), {})
sys.modules.setdefault("maibot_sdk", maibot_sdk_stub)

maibot_sdk_types_stub = types.ModuleType("maibot_sdk.types")
maibot_sdk_types_stub.ActivationType = type("ActivationType", (), {"ALWAYS": "ALWAYS"})
maibot_sdk_types_stub.HookMode = type("HookMode", (), {"OBSERVE": "OBSERVE"})
maibot_sdk_types_stub.HookOrder = type("HookOrder", (), {"EARLY": "EARLY", "NORMAL": "NORMAL", "LATE": "LATE"})
sys.modules.setdefault("maibot_sdk.types", maibot_sdk_types_stub)

src_config_package = types.ModuleType("src.config")
src_config_package.__path__ = [str(MAIBOT_ROOT / "src" / "config")]
sys.modules.setdefault("src.config", src_config_package)

src_chat_package = types.ModuleType("src.chat")
src_chat_package.__path__ = [str(MAIBOT_ROOT / "src" / "chat")]
sys.modules.setdefault("src.chat", src_chat_package)

src_chat_utils_package = types.ModuleType("src.chat.utils")
src_chat_utils_package.__path__ = [str(MAIBOT_ROOT / "src" / "chat" / "utils")]
sys.modules.setdefault("src.chat.utils", src_chat_utils_package)

chat_utils_module = types.ModuleType("src.chat.utils.utils")
chat_utils_module.parse_platform_accounts = lambda platforms: {}
sys.modules["src.chat.utils.utils"] = chat_utils_module

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
src_llm_models_package.__path__ = [str(MAIBOT_ROOT / "src" / "llm_models")]
sys.modules.setdefault("src.llm_models", src_llm_models_package)

src_common_data_models_package = types.ModuleType("src.common.data_models")
src_common_data_models_package.__path__ = [
    str(MAIBOT_ROOT / "src" / "common" / "data_models")
]
sys.modules.setdefault("src.common.data_models", src_common_data_models_package)

llm_service_data_models_module = types.ModuleType(
    "src.common.data_models.llm_service_data_models"
)
llm_service_data_models_module.LLMGenerationOptions = type("LLMGenerationOptions", (), {})
llm_service_data_models_module.LLMImageOptions = type("LLMImageOptions", (), {})
sys.modules["src.common.data_models.llm_service_data_models"] = (
    llm_service_data_models_module
)

utils_model_module = types.ModuleType("src.llm_models.utils_model")


class _DummyLLMOrchestrator:
    def __init__(self, *args, **kwargs):
        self.model_for_task = None
        self.model_usage = {}


utils_model_module.LLMOrchestrator = _DummyLLMOrchestrator
sys.modules["src.llm_models.utils_model"] = utils_model_module

src_services_module = types.ModuleType("src.services")
src_services_module.__path__ = [str(MAIBOT_ROOT / "src" / "services")]
src_services_module.llm_service = types.SimpleNamespace()
sys.modules["src.services"] = src_services_module

embedding_service_module = types.ModuleType("src.services.embedding_service")
embedding_service_module.EmbeddingServiceClient = type("EmbeddingServiceClient", (), {})
sys.modules["src.services.embedding_service"] = embedding_service_module

llm_service_module = types.ModuleType("src.services.llm_service")
llm_service_module.LLMServiceClient = type("LLMServiceClient", (), {})
llm_service_module.resolve_task_name = lambda preferred_task_name="": preferred_task_name or "default"
llm_service_module.resolve_task_name_from_model_config = (
    lambda model_config, preferred_task_name="": preferred_task_name or "default"
)
sys.modules["src.services.llm_service"] = llm_service_module

# 兜底：兄弟测试可能预先 stub 过 tag_retriever 但只塞了 get_tag_retriever，
# 这里补齐 reset_tag_retriever，避免 plugin.py 在 import 时挂掉
_existing_tag_stub = sys.modules.get("plugins.nai_draw_plugin.core.services.tag_retriever")
if _existing_tag_stub is not None and not hasattr(_existing_tag_stub, "reset_tag_retriever"):
    _existing_tag_stub.reset_tag_retriever = lambda *args, **kwargs: None
elif _existing_tag_stub is None:
    _tag_retriever_stub = types.ModuleType(
        "plugins.nai_draw_plugin.core.services.tag_retriever"
    )
    _tag_retriever_stub.get_tag_retriever = lambda **_kwargs: None
    _tag_retriever_stub.reset_tag_retriever = lambda *args, **kwargs: None
    sys.modules["plugins.nai_draw_plugin.core.services.tag_retriever"] = _tag_retriever_stub

import inspect as _inspect  # noqa: E402

from plugins.nai_draw_plugin.plugin import NaiPicPlugin, _resolve_existing_config_value  # noqa: E402


def _make_plugin_pointing_to(config_path: Path) -> NaiPicPlugin:
    """构造一个跳过 __init__ 的插件实例，使其 _regenerate_* 方法读到指定 config 路径。"""
    plugin = object.__new__(NaiPicPlugin)

    # _regenerate_config_with_comments_if_needed 通过 inspect.getfile(self.__class__)
    # 找到插件目录里的 config.toml。直接 monkeypatch inspect.getfile，让它指向 tmp 目录里的占位文件。
    fake_plugin_file = config_path.parent / "plugin.py"
    fake_plugin_file.write_text("", encoding="utf-8")

    real_getfile = _inspect.getfile

    def _patched_getfile(cls):
        if cls is type(plugin) or cls is NaiPicPlugin:
            return str(fake_plugin_file)
        return real_getfile(cls)

    plugin._patched_getfile = _patched_getfile  # 保活引用，避免 GC
    return plugin


def test_regenerate_writes_comments_when_file_has_none(monkeypatch, tmp_path) -> None:
    """无注释的 config.toml → 触发回填。"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[plugin]\nname = "nai_draw_plugin"\nconfig_version = "1.4.0"\nenabled = true\n',
        encoding="utf-8",
    )

    plugin = _make_plugin_pointing_to(config_path)
    monkeypatch.setattr(_inspect, "getfile", plugin._patched_getfile)
    plugin._regenerate_config_with_comments_if_needed()

    text = config_path.read_text(encoding="utf-8")
    assert "# " in text, "没有写出任何注释行"
    # 至少要把 description 写进去；挑一个明显的关键词
    assert "插件基本配置" in text or "插件配置版本号" in text
    # 用户已有的值要保留
    assert 'name = "nai_draw_plugin"' in text
    assert 'config_version = "1.4.0"' in text


def test_regenerate_is_idempotent(monkeypatch, tmp_path) -> None:
    """连续调用两次结果一致。"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[plugin]\nname = "nai_draw_plugin"\nconfig_version = "1.4.0"\nenabled = true\n',
        encoding="utf-8",
    )

    plugin = _make_plugin_pointing_to(config_path)
    monkeypatch.setattr(_inspect, "getfile", plugin._patched_getfile)
    plugin._regenerate_config_with_comments_if_needed()
    first = config_path.read_text(encoding="utf-8")

    plugin._regenerate_config_with_comments_if_needed()
    second = config_path.read_text(encoding="utf-8")

    assert first == second


def test_regenerate_skips_when_user_has_comments(monkeypatch, tmp_path) -> None:
    """文件里已经有 # 注释 → 不覆盖。"""
    config_path = tmp_path / "config.toml"
    original = (
        '# 用户的自定义注释\n'
        '[plugin]\nname = "nai_draw_plugin"\nconfig_version = "1.4.0"\nenabled = true\n'
    )
    config_path.write_text(original, encoding="utf-8")

    plugin = _make_plugin_pointing_to(config_path)
    monkeypatch.setattr(_inspect, "getfile", plugin._patched_getfile)
    plugin._regenerate_config_with_comments_if_needed()

    assert config_path.read_text(encoding="utf-8") == original


def test_regenerate_preserves_unknown_sections(monkeypatch, tmp_path) -> None:
    """schema 外的 section（如用户手加的）要原样保留。"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[plugin]\nname = "nai_draw_plugin"\nconfig_version = "1.4.0"\nenabled = true\n'
        '[my_custom]\nfoo = "bar"\n',
        encoding="utf-8",
    )

    plugin = _make_plugin_pointing_to(config_path)
    monkeypatch.setattr(_inspect, "getfile", plugin._patched_getfile)
    plugin._regenerate_config_with_comments_if_needed()

    text = config_path.read_text(encoding="utf-8")
    assert "[my_custom]" in text
    assert 'foo = "bar"' in text


def test_regenerate_missing_file_is_safe(monkeypatch, tmp_path) -> None:
    """config.toml 不存在 → 静默跳过，不抛错。"""
    config_path = tmp_path / "config.toml"
    # 没有写入文件
    plugin = _make_plugin_pointing_to(config_path)
    monkeypatch.setattr(_inspect, "getfile", plugin._patched_getfile)
    plugin._regenerate_config_with_comments_if_needed()
    assert not config_path.exists()


# ==================== _resolve_existing_config_value ====================


def test_resolve_existing_value_returns_default_when_missing() -> None:
    assert _resolve_existing_config_value({}, "sec", "key", "fallback") == "fallback"
    assert _resolve_existing_config_value({"sec": {}}, "sec", "key", 42) == 42


def test_resolve_existing_value_returns_actual_when_present() -> None:
    doc = {"sec": {"key": "actual"}}
    assert _resolve_existing_config_value(doc, "sec", "key", "fallback") == "actual"


def test_resolve_existing_value_unwraps_tomlkit_items() -> None:
    import tomlkit

    doc = tomlkit.parse('[sec]\nkey = "hello"\n')
    value = _resolve_existing_config_value(doc, "sec", "key", "fallback")
    assert value == "hello"
    assert not hasattr(value, "unwrap"), "返回值应已 unwrap"
