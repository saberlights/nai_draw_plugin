from pathlib import Path
import sys


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.model_capabilities import (  # noqa: E402
    ALL_MODEL_SECTIONS,
    MODEL_SPECS,
)
from plugins.nai_draw_plugin.core.plugin_config import PLUGIN_CONFIG  # noqa: E402


def test_config_definition_exposes_defaults_and_webui_from_one_schema() -> None:
    defaults = PLUGIN_CONFIG.default_config()
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    assert defaults["model"]["nai_proxy_mode"] == "direct"
    assert "nai-diffusion-5-full" in defaults["model"]["available_models"]
    assert "wd14_spaces" not in defaults["retag"]
    assert webui["sections"]["model"]["fields"]["api_key"]["ui_type"] == "password"
    assert webui["sections"]["retag"]["fields"]["wd14_spaces"]["hidden"] is True


def test_removed_reply_auto_draw_is_not_exposed_in_config() -> None:
    defaults = PLUGIN_CONFIG.default_config()
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    assert "auto_draw_on_reply" not in defaults
    assert "auto_draw_on_reply" not in webui["sections"]


def test_当前默认模型的配置节排在生图页签最前且展开() -> None:
    """有 8 个模型节，不按当前模型排序的话用户每次都要在同构折叠卡里翻找。"""
    webui = PLUGIN_CONFIG.webui_schema(
        plugin_id="nai_draw_plugin",
        active_model="nai-diffusion-4-5-full",
    )

    assert webui["layout"]["type"] == "tabs"
    tabs = {tab["id"]: tab for tab in webui["layout"]["tabs"]}
    generation = tabs["generation"]["sections"]
    # [model_defaults] 是共享基准层，必须留在模型节之前
    assert generation[0] == "model_defaults"
    assert generation[1] == "model_v45_full"

    v45_section = webui["sections"]["model_v45_full"]
    assert v45_section["title"] == "V4.5 Full 生图参数（当前默认模型）"
    assert v45_section["collapsed"] is False
    assert v45_section["fields"]["custom_prompt_add"]["ui_type"] == "textarea"
    assert v45_section["fields"]["negative_prompt_add"]["ui_type"] == "textarea"

    # 非当前模型默认折叠
    assert webui["sections"]["model_v3_furry"]["collapsed"] is True
    assert "当前默认模型" not in webui["sections"]["model_v3_furry"]["title"]


def test_切换默认模型会换掉排在最前的模型节() -> None:
    webui = PLUGIN_CONFIG.webui_schema(
        plugin_id="nai_draw_plugin",
        active_model="nai-diffusion-5-full",
    )
    tabs = {tab["id"]: tab for tab in webui["layout"]["tabs"]}

    assert tabs["generation"]["sections"][1] == "model_v5_full"
    assert tabs["presets"]["sections"][0] == "model_v5_full.artist_presets"
    assert webui["sections"]["model_v5_full"]["collapsed"] is False
    assert webui["sections"]["model_v45_full"]["collapsed"] is True


def test_webui_uses_textareas_for_all_long_prompt_fields() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    model_prompt_fields = {
        "nai_artist_prompt",
        "custom_prompt_add",
        "negative_prompt_add",
        "selfie_prompt_add",
        "selfie_negative_prompt_add",
    }
    for model_section in ALL_MODEL_SECTIONS:
        for field_name in model_prompt_fields:
            field = webui["sections"][model_section]["fields"][field_name]
            assert field["ui_type"] == "textarea", f"{model_section}.{field_name}"
            assert field["rows"] >= 8

    # 共享层的长文本字段同样要用 textarea
    for field_name in ("negative_prompt_add", "selfie_prompt_add", "selfie_negative_prompt_add"):
        assert webui["sections"]["model_defaults"]["fields"][field_name]["ui_type"] == "textarea"

    assert webui["sections"]["prompt_generator"]["fields"]["prompt_template"][
        "ui_type"
    ] == "textarea"
    assert webui["sections"]["custom_prompt"]["fields"]["system_prompt"][
        "ui_type"
    ] == "textarea"
    assert webui["sections"]["nsfw_filter"]["fields"]["filter_tags"][
        "ui_type"
    ] == "textarea"


def test_每个模型都有独立的画师预设编辑区() -> None:
    """这是本次改动的核心诉求：8 个模型各一份预设库，都能在 WebUI 里编。"""
    webui = PLUGIN_CONFIG.webui_schema(
        plugin_id="nai_draw_plugin",
        active_model="nai-diffusion-4-5-full",
    )
    tabs = {tab["id"]: tab for tab in webui["layout"]["tabs"]}
    preset_tab = tabs["presets"]["sections"]

    assert len(preset_tab) == len(MODEL_SPECS) == 8

    for spec in MODEL_SPECS:
        source_section = spec.config_section
        editor_section = f"{source_section}.artist_presets"

        # 主配置节里隐藏，避免同一个字段在两处重复编辑
        assert webui["sections"][source_section]["fields"]["artist_presets"]["hidden"] is True

        detached = webui["sections"][editor_section]
        # Dashboard 按 section.name 决定取值与保存路径，必须指回原配置节
        assert detached["name"] == source_section
        assert detached["title"].startswith(f"{spec.label} 画师预设")
        assert "切换模型不会串用" in detached["description"]
        # 长画师串编辑受 Dashboard 单行控件限制，说明里要如实指路到源代码模式
        assert "源代码" in detached["description"]

        artist_presets = detached["fields"]["artist_presets"]
        assert artist_presets["hidden"] is False
        assert artist_presets["section"] == source_section
        assert artist_presets["item_type"] == "object"
        assert set(artist_presets["item_fields"]) == {
            "name",
            "prompt",
            "negative_prompt_add",
        }
        # 预设名要能被 /nai art <名称> 调用，给出命名引导
        assert "nai art" in artist_presets["item_fields"]["name"]["placeholder"]

        assert editor_section in preset_tab


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


def test_配置说明不再硬编码某个模型是当前默认() -> None:
    """原先 group header 里写死"你当前默认模型是 V4.5"，改了 default_model 就变成谎话。

    现在"当前默认模型"只由 WebUI 按实际配置动态标注。
    """
    for section in PLUGIN_CONFIG.config_section_group_headers.values():
        assert "当前默认模型是" not in str(section)

    # 未指定 active_model 时，没有任何节被标成当前默认
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")
    assert not [s for s in webui["sections"].values() if "当前默认模型" in s["title"]]


def test_webui_uses_password_controls_only_for_secret_fields() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    password_fields = {
        f"{section_name}.{field_name}"
        for section_name, section in webui["sections"].items()
        for field_name, field in section["fields"].items()
        if not field["hidden"] and field["ui_type"] == "password"
    }
    assert password_fields == {"model.api_key", "model.chami_api_key", "model.chami_web_password"}
    assert webui["sections"]["model"]["fields"]["chami_web_username"]["ui_type"] == "text"
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
    assert "固定追加到正向提示词" in webui["sections"]["model_v45_full"]["fields"][
        "custom_prompt_add"
    ]["hint"]


def test_每个模型节的字段说明都带自己的档位名() -> None:
    """1.x 里 model_nai4 / model_nai3 两节 38 个字段的说明全是"作用同 model_nai4_5.xxx"，
    而 WebUI 的 label 取的正是 description 首行 —— 于是"其他模型"整页没有一个可读标签。
    """
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    filler_labels = [
        f"{section_name}.{field_name}={field['label']}"
        for section_name, section in webui["sections"].items()
        for field_name, field in section["fields"].items()
        if not field["hidden"] and "作用同" in str(field["label"])
    ]
    assert filler_labels == []

    # 每个模型节的可见字段标签都要带上自己的档位名，才能在 8 个同构节里分得清
    for spec in MODEL_SPECS:
        section = webui["sections"][spec.config_section]
        labelled = [
            field["label"]
            for field in section["fields"].values()
            if not field["hidden"] and spec.label in str(field["label"])
        ]
        assert labelled, f"{spec.config_section} 的字段标签没有带上 {spec.label}"


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
    # nai_extra_params 改成 [{key, value}] 数组后要能在 WebUI 里编：
    # Dashboard 没有 dict/json 编辑器，dict 形态只能靠 hidden 藏起来，等于不可编辑
    for model_section in (*ALL_MODEL_SECTIONS, "model_defaults"):
        extra = webui["sections"][model_section]["fields"]["nai_extra_params"]
        assert extra["hidden"] is False, f"{model_section}.nai_extra_params 不该被隐藏"
        assert extra["ui_type"] == "list"
        assert extra["item_type"] == "object"
        assert set(extra["item_fields"]) == {"key", "value"}


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

    # 默认模型从注册表取候选，别让用户手打模型 id
    default_model = webui["sections"]["model"]["fields"]["default_model"]
    assert default_model["ui_type"] == "select"
    assert "nai-diffusion-5-full" in default_model["choices"]
    assert "nai-diffusion-5-curated" in default_model["choices"]

    # 对齐官方 /image/sampling/ 列出的 8 个采样器：DPM++ 2M、Euler Ancestral、Euler、
    # DPM2、DPM++ 2S Ancestral、DPM++ SDE、DPM Fast、DDIM
    sampler_choices = [
        "k_euler",
        "k_euler_ancestral",
        "k_dpm_2",
        "k_dpm_2_ancestral",
        "k_dpm_fast",
        "k_dpmpp_2m",
        "k_dpmpp_2s_ancestral",
        "k_dpmpp_sde",
        "ddim",
    ]
    for model_section in ALL_MODEL_SECTIONS:
        sampler = webui["sections"][model_section]["fields"].get("sampler")
        # sampler / image_format / seed / cfg_rescale 等共享标量已收敛到 [model_defaults]，
        # 模型节不再重复定义
        assert sampler is None, f"{model_section} 不该再单独定义 sampler"
        assert "image_format" not in webui["sections"][model_section]["fields"]

    defaults_fields = webui["sections"]["model_defaults"]["fields"]
    assert defaults_fields["sampler"]["ui_type"] == "select"
    assert defaults_fields["sampler"]["choices"] == sampler_choices
    assert defaults_fields["image_format"]["ui_type"] == "select"
    assert defaults_fields["image_format"]["choices"] == ["png", "webp"]


def test_下拉候选项不含空字符串() -> None:
    """Dashboard 的 Select 基于 Radix，SelectItem 的 value 不能是空字符串。

    给下拉塞一个空选项当"未设置"会在渲染时直接抛错、把整个配置页签打崩
    （曾因此让"生图参数"页打不开）。需要"留空 = 用默认"语义的字段必须用文本框。
    """
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    offenders = [
        f"{section_name}.{field_name}"
        for section_name, section in webui["sections"].items()
        for field_name, field in section["fields"].items()
        if not field["hidden"]
        and any(choice == "" or choice is None for choice in (field.get("choices") or []))
    ]
    assert offenders == [], f"这些字段的下拉含空候选项，会让 Radix Select 崩：{offenders}"


def test_noise_schedule_保持文本框以便清空() -> None:
    """留空表示不发送该字段、让网关取默认，是用户需要能切回去的状态。"""
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")

    for model_section in ALL_MODEL_SECTIONS:
        noise = webui["sections"][model_section]["fields"]["noise_schedule"]
        assert noise["ui_type"] == "text"
        assert "karras" in noise["hint"]
        assert "留空" in noise["hint"]


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
    for model_section in ALL_MODEL_SECTIONS:
        expected_ranges[(model_section, "num_inference_steps")] = (1, 28, 1)
    expected_ranges[("model_defaults", "cfg_rescale")] = (0.0, 1.0, 0.05)

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
