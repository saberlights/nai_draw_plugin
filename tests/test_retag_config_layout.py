# -*- coding: utf-8 -*-
"""验证 config 自动生成 / 重渲染时 retag 段的排版与隐藏字段策略。

这套测试在 plugin.py 内部 stub 同款 maibot_sdk，避免依赖完整 MaiBot Runner。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

# 与既有兄弟测试同款 stub —— 避免插件 import 时炸
maibot_sdk_stub = types.ModuleType("maibot_sdk")
maibot_sdk_stub.Action = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.Command = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.HookHandler = lambda *args, **kwargs: (lambda func: func)
maibot_sdk_stub.MaiBotPlugin = type("MaiBotPlugin", (), {})
sys.modules.setdefault("maibot_sdk", maibot_sdk_stub)

maibot_sdk_types_stub = types.ModuleType("maibot_sdk.types")
maibot_sdk_types_stub.ActivationType = type("ActivationType", (), {"ALWAYS": "ALWAYS"})
maibot_sdk_types_stub.HookMode = type("HookMode", (), {"OBSERVE": "OBSERVE"})
maibot_sdk_types_stub.HookOrder = type(
    "HookOrder", (), {"EARLY": "EARLY", "NORMAL": "NORMAL", "LATE": "LATE"}
)
sys.modules.setdefault("maibot_sdk.types", maibot_sdk_types_stub)

# 老朋友：补齐 src.* 的依赖空壳
for module_name in (
    "src.config",
    "src.config.config",
    "src.config.model_configs",
    "src.chat",
    "src.chat.utils",
    "src.chat.utils.utils",
    "src.llm_models",
    "src.llm_models.utils_model",
    "src.common.data_models",
    "src.common.data_models.llm_service_data_models",
    "src.services",
    "src.services.embedding_service",
    "src.services.llm_service",
):
    if module_name in sys.modules:
        continue
    module = types.ModuleType(module_name)
    module.__path__ = [str(MAIBOT_ROOT / Path(*module_name.split(".")))]
    sys.modules[module_name] = module

# 极简 stub：让 plugin.py 顶部的若干 import 通过
sys.modules["src.config.config"].global_config = types.SimpleNamespace()
sys.modules["src.config.config"].model_config = types.SimpleNamespace(
    model_task_config=types.SimpleNamespace(embedding=None)
)
sys.modules["src.config.model_configs"].TaskConfig = type("TaskConfig", (), {})
sys.modules["src.chat.utils.utils"].parse_platform_accounts = lambda platforms: {}
sys.modules["src.common.data_models.llm_service_data_models"].LLMGenerationOptions = type(
    "LLMGenerationOptions", (), {}
)
sys.modules["src.common.data_models.llm_service_data_models"].LLMImageOptions = type(
    "LLMImageOptions", (), {}
)


class _DummyLLMOrchestrator:
    pass


sys.modules["src.llm_models.utils_model"].LLMOrchestrator = _DummyLLMOrchestrator
sys.modules["src.services"].llm_service = types.SimpleNamespace()
sys.modules["src.services.embedding_service"].EmbeddingServiceClient = type(
    "EmbeddingServiceClient", (), {}
)
sys.modules["src.services.llm_service"].LLMServiceClient = type("LLMServiceClient", (), {})
sys.modules["src.services.llm_service"].resolve_task_name_from_model_config = (
    lambda model_config, preferred_task_name="": preferred_task_name or "default"
)

# 兜底：tag_retriever 服务在 stub 链路里没法真加载
tag_retriever_stub = types.ModuleType(
    "plugins.nai_draw_plugin.core.services.tag_retriever"
)
tag_retriever_stub.get_tag_retriever = lambda **_kwargs: None
tag_retriever_stub.reset_tag_retriever = lambda *args, **kwargs: None
sys.modules.setdefault(
    "plugins.nai_draw_plugin.core.services.tag_retriever", tag_retriever_stub
)


import tomlkit  # noqa: E402

from plugins.nai_draw_plugin.plugin import NaiPicPlugin  # noqa: E402


def _render_default_config_text() -> str:
    plugin = NaiPicPlugin()
    runner_dump = tomlkit.dumps(plugin.get_default_config())
    return plugin._compose_commented_config_text(tomlkit.parse(runner_dump))


def test_default_config_excludes_hidden_wd14_spaces() -> None:
    plugin = NaiPicPlugin()
    default = plugin.get_default_config()
    assert "retag" in default
    assert "wd14_spaces" not in default["retag"], (
        "wd14_spaces 默认不应出现在 get_default_config 输出里；"
        "否则 Runner 首次启动会把这一长段表数组 dump 进新生成的 config.toml"
    )
    # 同时核对其它字段都还在
    assert "wd14_proxy" in default["retag"]
    assert default["retag"]["enabled"] is True


def test_default_config_directs_nai_without_removing_wd14_proxy() -> None:
    plugin = NaiPicPlugin()
    default = plugin.get_default_config()

    assert default["model"]["nai_proxy_mode"] == "direct"
    assert "wd14_proxy" in default["retag"]


def test_webui_config_schema_exposes_configfield_metadata() -> None:
    plugin = NaiPicPlugin()
    schema = plugin.get_webui_config_schema(plugin_id="nai_draw_plugin")

    api_key = schema["sections"]["model"]["fields"]["api_key"]
    assert api_key["type"] == "string"
    assert api_key["ui_type"] == "password"
    assert api_key["input_type"] == "password"
    assert api_key["label"]

    prompt_template = schema["sections"]["prompt_generator"]["fields"]["prompt_template"]
    assert prompt_template["ui_type"] == "textarea"
    assert prompt_template["rows"] >= 8


def test_webui_config_schema_hides_advanced_wd14_spaces() -> None:
    plugin = NaiPicPlugin()
    schema = plugin.get_webui_config_schema(plugin_id="nai_draw_plugin")

    wd14_spaces = schema["sections"]["retag"]["fields"]["wd14_spaces"]
    assert wd14_spaces["hidden"] is True
    assert wd14_spaces["item_type"] == "object"
    assert set(wd14_spaces["item_fields"]) == {"name", "type", "api"}


def test_rendered_config_places_retag_between_tag_retriever_and_custom_prompt() -> None:
    text = _render_default_config_text()
    idx_tag = text.find("\n[tag_retriever]")
    idx_retag = text.find("\n[retag]")
    idx_custom = text.find("\n[custom_prompt]")
    assert idx_tag != -1 and idx_retag != -1 and idx_custom != -1
    assert idx_tag < idx_retag < idx_custom, (
        f"[retag] 段位置错误：tag_retriever={idx_tag}, retag={idx_retag}, custom_prompt={idx_custom}"
    )


def test_rendered_config_has_retag_group_header() -> None:
    text = _render_default_config_text()
    assert "========== 图片反推（/nai 反推） ==========" in text, (
        "缺少 retag 段的分组标题"
    )


def test_rendered_config_has_per_field_comments() -> None:
    text = _render_default_config_text()
    # 每个字段前都该有一行 # 注释
    must_have = [
        ("# 入站图片缓存保留时间", "cache_ttl_seconds"),
        ("# 单个 Space 请求超时", "wd14_request_timeout"),
        ("# 访问 Hugging Face Space", "wd14_proxy"),
        ("# 图片尺寸；可填 竖图 / 横图 / 方图", "nai_size"),
        ("# 随机种子；可填整数固定结果", "seed"),
    ]
    for comment_prefix, field_name in must_have:
        assert comment_prefix in text, f"缺少 {field_name} 的注释行"


def test_repository_config_toml_is_valid_and_uses_supported_draw_defaults() -> None:
    import tomllib

    config_path = Path(__file__).resolve().parents[1] / "config.toml"
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert data["model_nai4_5"]["seed"] == -1
    assert data["model_nai4_5"]["default_size"] == "832x1216"
    assert data["model_nai4"]["default_size"] == "832x1216"


def test_rendered_config_does_not_contain_wd14_spaces_block() -> None:
    text = _render_default_config_text()
    assert "[[retag.wd14_spaces]]" not in text, (
        "wd14_spaces 表数组不应被渲染（用户改不动，且默认 3 个 Space 已内置在代码中）"
    )


def test_user_can_still_override_wd14_spaces_manually() -> None:
    """schema 仍保留 wd14_spaces 字段，用户在 config.toml 手动加 [[retag.wd14_spaces]] 仍能被读取。"""
    import tomllib

    mock_config = b"""
[retag]
wd14_proxy = "http://127.0.0.1:7890"

[[retag.wd14_spaces]]
name = "mycustom/space"
type = "pixai"
api = "/predict_image"
"""
    parsed = tomllib.loads(mock_config.decode("utf-8"))
    assert parsed["retag"]["wd14_spaces"] == [
        {"name": "mycustom/space", "type": "pixai", "api": "/predict_image"}
    ]
    # 同时 schema 里得保留 wd14_spaces 字段定义，否则 Runner 可能直接丢弃用户输入
    schema_retag = NaiPicPlugin.config_schema["retag"]
    assert "wd14_spaces" in schema_retag, "schema 必须保留 wd14_spaces 以支持高级覆盖"
