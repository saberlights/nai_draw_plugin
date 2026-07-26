from pathlib import Path
import sys


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.plugin_config import PLUGIN_CONFIG


def test_config_definition_exposes_defaults_and_webui_from_one_schema() -> None:
    defaults = PLUGIN_CONFIG.default_config()
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    assert defaults["model"]["nai_proxy_mode"] == "direct"
    assert "wd14_spaces" not in defaults["retag"]
    assert webui["sections"]["model"]["fields"]["api_key"]["ui_type"] == "password"
    assert webui["sections"]["retag"]["fields"]["wd14_spaces"]["hidden"] is True


def test_runtime_config_recursively_overrides_local_values_without_dropping_siblings() -> None:
    merged = PLUGIN_CONFIG.merge(
        {
            "model": {"base_url": "https://local.example", "nai_request_timeout": 300},
            "prompt_generator": {"enabled": True},
        },
        {"model": {"base_url": "https://runtime.example"}},
    )

    assert merged == {
        "model": {"base_url": "https://runtime.example", "nai_request_timeout": 300},
        "prompt_generator": {"enabled": True},
    }
