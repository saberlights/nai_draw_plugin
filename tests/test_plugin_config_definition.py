from pathlib import Path
import sys


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.plugin_config import PLUGIN_CONFIG  # noqa: E402


def test_config_definition_exposes_defaults_and_webui_from_one_schema() -> None:
    defaults = PLUGIN_CONFIG.default_config()
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    assert defaults["model"]["nai_proxy_mode"] == "direct"
    assert "wd14_spaces" not in defaults["retag"]
    assert webui["sections"]["model"]["fields"]["api_key"]["ui_type"] == "password"
    assert webui["sections"]["retag"]["fields"]["wd14_spaces"]["hidden"] is True


def test_webui_makes_v45_prompt_fields_discoverable_in_first_tab() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    assert webui["layout"]["type"] == "tabs"
    first_tab = webui["layout"]["tabs"][0]
    assert first_tab["title"] == "生图配置"
    assert "model_nai4_5" in first_tab["sections"][:2]

    v45_section = webui["sections"]["model_nai4_5"]
    assert v45_section["title"] == "NAI V4.5 生图参数"
    assert v45_section["fields"]["custom_prompt_add"]["ui_type"] == "textarea"
    assert v45_section["fields"]["negative_prompt_add"]["ui_type"] == "textarea"


def test_webui_uses_textareas_for_all_long_prompt_fields() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    model_prompt_fields = {
        "nai_artist_prompt",
        "custom_prompt_add",
        "negative_prompt_add",
        "selfie_prompt_add",
        "selfie_negative_prompt_add",
    }
    for model_section in ("model_nai4_5", "model_nai4", "model_nai3"):
        for field_name in model_prompt_fields:
            field = webui["sections"][model_section]["fields"][field_name]
            assert field["ui_type"] == "textarea"
            assert field["rows"] >= 8

    assert webui["sections"]["prompt_generator"]["fields"]["prompt_template"][
        "ui_type"
    ] == "textarea"
    assert webui["sections"]["custom_prompt"]["fields"]["system_prompt"][
        "ui_type"
    ] == "textarea"
    assert webui["sections"]["nsfw_filter"]["fields"]["filter_tags"][
        "ui_type"
    ] == "textarea"


def test_webui_separates_artist_presets_into_collapsed_model_sections() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    expected_sections = {
        "model_nai4_5": ("model_nai4_5.artist_presets", "generation", "NAI V4.5 画师预设"),
        "model_nai4": ("model_nai4.artist_presets", "models", "NAI V4 画师预设"),
        "model_nai3": ("model_nai3.artist_presets", "models", "NAI V3 / Furry 画师预设"),
    }
    tabs = {tab["id"]: tab for tab in webui["layout"]["tabs"]}

    for source_section, (editor_section, tab_id, title) in expected_sections.items():
        assert webui["sections"][source_section]["fields"]["artist_presets"]["hidden"] is True

        detached = webui["sections"][editor_section]
        assert detached["name"] == source_section
        assert detached["title"] == title
        assert detached["collapsed"] is True
        assert "源代码" in detached["description"]
        assert "自动换行" in detached["description"]

        artist_presets = detached["fields"]["artist_presets"]
        assert artist_presets["hidden"] is False
        assert artist_presets["section"] == source_section
        assert artist_presets["item_type"] == "object"
        assert set(artist_presets["item_fields"]) == {
            "name",
            "prompt",
            "negative_prompt_add",
        }

        tab_sections = tabs[tab_id]["sections"]
        assert tab_sections.index(editor_section) == tab_sections.index(source_section) + 1


def test_webui_tabs_cover_all_sections_once_with_human_readable_titles() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    tab_sections = [
        section_name
        for tab in webui["layout"]["tabs"]
        for section_name in tab["sections"]
    ]
    assert len(tab_sections) == len(set(tab_sections))
    assert set(tab_sections) == set(webui["sections"])

    expected_titles = {
        "auto_draw_on_reply": "回复后自动跟图",
        "prompt_show": "提示词显示",
        "nsfw_filter": "NSFW 过滤",
        "auto_recall": "自动撤回",
        "admin": "管理员权限",
        "tag_retriever": "Danbooru Tag 检索",
    }
    for section_name, expected_title in expected_titles.items():
        assert webui["sections"][section_name]["title"] == expected_title


def test_webui_section_descriptions_do_not_expose_toml_decorators() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    for section in webui["sections"].values():
        description = str(section.get("description") or "")
        assert "==========" not in description
        assert "-----" not in description
    assert "当前默认模型是 V4.5" in webui["sections"]["model_nai4_5"]["description"]


def test_webui_uses_password_controls_only_for_secret_fields() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    password_fields = {
        f"{section_name}.{field_name}"
        for section_name, section in webui["sections"].items()
        for field_name, field in section["fields"].items()
        if not field["hidden"] and field["ui_type"] == "password"
    }
    assert password_fields == {"model.api_key"}
    assert webui["sections"]["model"]["fields"]["nai_max_tokens"]["ui_type"] == "number"
    assert webui["sections"]["prompt_generator"]["fields"]["max_tokens"]["ui_type"] == "number"


def test_webui_disables_plugin_managed_config_version() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    config_version = webui["sections"]["plugin"]["fields"]["config_version"]
    assert config_version["disabled"] is True


def test_webui_exposes_help_text_for_every_visible_field() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    fields_without_help = [
        f"{section_name}.{field_name}"
        for section_name, section in webui["sections"].items()
        for field_name, field in section["fields"].items()
        if not field["hidden"] and not str(field.get("hint") or "").strip()
    ]
    assert fields_without_help == []
    assert "固定追加到正向提示词" in webui["sections"]["model_nai4_5"]["fields"][
        "custom_prompt_add"
    ]["hint"]


def test_webui_edits_prompt_generator_custom_model_as_nested_fields() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    parent_field = webui["sections"]["prompt_generator"]["fields"]["custom_model"]
    assert parent_field["hidden"] is True

    nested = webui["sections"]["prompt_generator.custom_model"]
    assert nested["title"] == "提示词生成自定义模型"
    assert nested["fields"]["model_list"]["ui_type"] == "list"
    assert nested["fields"]["max_tokens"]["ui_type"] == "number"
    assert nested["fields"]["temperature"]["ui_type"] == "number"
    assert nested["fields"]["slow_threshold"]["ui_type"] == "number"

    prompting_tab = next(tab for tab in webui["layout"]["tabs"] if tab["id"] == "prompting")
    assert prompting_tab["sections"][:2] == [
        "prompt_generator",
        "prompt_generator.custom_model",
    ]


def test_webui_edits_random_scene_custom_model_as_nested_fields() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    assert webui["sections"]["random_scene"]["fields"]["custom_model"]["hidden"] is True
    nested = webui["sections"]["random_scene.custom_model"]
    assert nested["title"] == "随机场景自定义模型"
    assert set(nested["fields"]) == {
        "model_list",
        "max_tokens",
        "temperature",
        "slow_threshold",
    }

    prompting_tab = next(tab for tab in webui["layout"]["tabs"] if tab["id"] == "prompting")
    random_scene_index = prompting_tab["sections"].index("random_scene")
    assert prompting_tab["sections"][random_scene_index + 1] == "random_scene.custom_model"


def test_webui_only_exposes_controls_supported_by_dashboard() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")
    supported_controls = {
        "switch",
        "number",
        "slider",
        "select",
        "textarea",
        "password",
        "list",
        "text",
    }

    unsupported_fields = [
        f"{section_name}.{field_name}:{field['ui_type']}"
        for section_name, section in webui["sections"].items()
        for field_name, field in section["fields"].items()
        if not field["hidden"] and field["ui_type"] not in supported_controls
    ]
    assert unsupported_fields == []
    for model_section in ("model_nai4_5", "model_nai4", "model_nai3"):
        assert webui["sections"][model_section]["fields"]["nai_extra_params"]["hidden"] is True


def test_webui_uses_select_controls_for_closed_choice_fields() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    expected_choices = {
        ("model", "nai_proxy_mode"): ["direct", "inherit", "auto"],
        ("prompt_generator", "output_format"): ["json", "text"],
        ("prompt_generator", "selfie_appearance_policy"): ["auto", "never", "keep"],
        ("tag_retriever", "mode"): ["online", "local"],
        ("character_reference", "type"): ["character", "style", "character&style"],
    }
    for (section_name, field_name), choices in expected_choices.items():
        field = webui["sections"][section_name]["fields"][field_name]
        assert field["ui_type"] == "select"
        assert field["choices"] == choices

    sampler_choices = [
        "k_euler",
        "k_euler_ancestral",
        "k_dpm_2",
        "k_dpm_2_ancestral",
        "k_dpmpp_2m",
        "k_dpmpp_2s_ancestral",
        "k_dpmpp_sde",
        "ddim",
    ]
    for model_section in ("model_nai4_5", "model_nai4", "model_nai3"):
        sampler = webui["sections"][model_section]["fields"]["sampler"]
        assert sampler["ui_type"] == "select"
        assert sampler["choices"] == sampler_choices
        image_format = webui["sections"][model_section]["fields"]["image_format"]
        assert image_format["ui_type"] == "select"
        assert image_format["choices"] == ["png", "webp"]


def test_webui_numeric_controls_expose_steps_and_documented_ranges() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    numeric_fields_without_step = [
        f"{section_name}.{field_name}"
        for section_name, section in webui["sections"].items()
        for field_name, field in section["fields"].items()
        if not field["hidden"]
        and field["type"] in {"integer", "number"}
        and field.get("step") is None
    ]
    assert numeric_fields_without_step == []

    expected_ranges = {
        ("auto_draw_on_reply", "score_threshold"): (0.0, 1.0, 0.05),
        ("tag_retriever", "popularity_weight"): (0.0, 1.0, 0.01),
        ("tag_retriever", "min_score"): (0.0, 1.0, 0.01),
        ("retag", "wd14_threshold"): (0.0, 1.0, 0.01),
        ("retag", "wd14_character_threshold"): (0.0, 1.0, 0.01),
        ("i2i", "strength"): (0.01, 0.99, 0.01),
        ("i2i", "noise"): (0.0, 0.99, 0.01),
        ("vibe", "info_extracted"): (0.01, 1.0, 0.01),
        ("vibe", "reference_strength"): (0.01, 1.0, 0.01),
        ("vibe", "overall_strength"): (0.0, 1.0, 0.01),
        ("character_reference", "fidelity"): (0.0, 1.0, 0.01),
        ("character_reference", "strength"): (0.0, 1.0, 0.01),
    }
    for model_section in ("model_nai4_5", "model_nai4", "model_nai3"):
        expected_ranges[(model_section, "num_inference_steps")] = (1, 28, 1)
        expected_ranges[(model_section, "cfg_rescale")] = (0.0, 1.0, 0.05)

    for (section_name, field_name), (minimum, maximum, step) in expected_ranges.items():
        field = webui["sections"][section_name]["fields"][field_name]
        assert field["ui_type"] == "slider"
        assert (field["min"], field["max"], field["step"]) == (minimum, maximum, step)


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
