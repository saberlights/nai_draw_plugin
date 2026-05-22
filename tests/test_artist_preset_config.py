# -*- coding: utf-8 -*-
import os
import sys
import types
import unittest
import importlib


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIBOT_ROOT = os.path.abspath(os.path.join(PLUGIN_DIR, "../.."))

if MAIBOT_ROOT not in sys.path:
    sys.path.insert(0, MAIBOT_ROOT)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

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


def _get_logger(_name=None):
    return _DummyLogger()


dummy_logger_module.get_logger = _get_logger
sys.modules["src.common.logger"] = dummy_logger_module

core_package = types.ModuleType("core")
core_package.__path__ = [os.path.join(PLUGIN_DIR, "core")]
sys.modules.setdefault("core", core_package)

core_services_package = types.ModuleType("core.services")
core_services_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "services")]
sys.modules.setdefault("core.services", core_services_package)

core_mixins_package = types.ModuleType("core.mixins")
core_mixins_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "mixins")]
sys.modules.setdefault("core.mixins", core_mixins_package)

session_state_module = importlib.import_module("core.services.session_state")
model_config_mixin_module = importlib.import_module("core.mixins.model_config_mixin")

session_state = session_state_module.session_state
ModelConfigMixin = model_config_mixin_module.ModelConfigMixin


class _DummyModelConfigHost(ModelConfigMixin):
    def __init__(self, config):
        self._config = config
        self.log_prefix = "test_nai_pic"

    def get_config(self, path, default=None):
        if not path:
            return default

        current = self._config
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def _get_chat_identity(self):
        return "qq", "10001", "20002"


class ArtistPresetConfigTest(unittest.TestCase):
    def setUp(self):
        session_state._init_state()

    def test_parse_artist_presets_keeps_non_empty_negative_prompt(self):
        presets = session_state._parse_artist_presets([
            {
                "name": "风格A",
                "prompt": "artist:a",
                "negative_prompt_add": " bad hands, lowres ",
            },
            {
                "name": "风格B",
                "prompt": "artist:b",
                "negative_prompt_add": "   ",
            },
        ])

        self.assertEqual(presets[0]["negative_prompt_add"], "bad hands, lowres")
        self.assertNotIn("negative_prompt_add", presets[1])

    def test_model_config_uses_artist_preset_negative_prompt_when_present(self):
        host = _DummyModelConfigHost({
            "model": {
                "base_url": "https://example.com",
                "default_model": "nai-diffusion-4-5-full",
            },
            "model_nai4_5": {
                "negative_prompt_add": "model-default-negative",
                "artist_presets": [
                    {"name": "风格A", "prompt": "artist:a"},
                    {
                        "name": "风格B",
                        "prompt": "artist:b",
                        "negative_prompt_add": "preset-negative",
                    },
                ],
                "default_artist_preset": 2,
            },
        })

        model_config = host._get_model_config()

        self.assertEqual(model_config["nai_artist_prompt"], "artist:b")
        self.assertEqual(model_config["negative_prompt_add"], "preset-negative")

    def test_model_config_falls_back_to_model_negative_prompt_when_preset_negative_is_blank(self):
        host = _DummyModelConfigHost({
            "model": {
                "base_url": "https://example.com",
                "default_model": "nai-diffusion-4-5-full",
            },
            "model_nai4_5": {
                "negative_prompt_add": "model-default-negative",
                "artist_presets": [
                    {
                        "name": "风格A",
                        "prompt": "artist:a",
                        "negative_prompt_add": "   ",
                    },
                ],
                "default_artist_preset": 1,
            },
        })

        model_config = host._get_model_config()

        self.assertEqual(model_config["nai_artist_prompt"], "artist:a")
        self.assertEqual(model_config["negative_prompt_add"], "model-default-negative")

    def test_model_config_can_apply_artist_preset_for_direct_tag_command(self):
        host = _DummyModelConfigHost({
            "model": {
                "base_url": "https://example.com",
                "default_model": "nai-diffusion-4-5-full",
            },
            "model_nai4_5": {
                "negative_prompt_add": "model-default-negative",
                "artist_presets": [
                    {
                        "name": "风格A",
                        "prompt": "artist:a",
                        "negative_prompt_add": "preset-negative",
                    },
                ],
                "default_artist_preset": 1,
            },
        })

        model_config = host._get_model_config()

        self.assertEqual(model_config["nai_artist_prompt"], "artist:a")
        self.assertEqual(model_config["negative_prompt_add"], "preset-negative")

    def test_effective_artist_index_uses_config_default_when_session_not_overridden(self):
        config = {
            "model_nai4_5": {
                "artist_presets": [
                    {"name": "风格A", "prompt": "artist:a"},
                    {"name": "风格B", "prompt": "artist:b"},
                    {"name": "风格C", "prompt": "artist:c"},
                ],
                "default_artist_preset": 2,
            }
        }

        def get_config(path, default=None):
            current = config
            for part in path.split("."):
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            return current

        index = session_state.get_effective_artist_index(
            "qq",
            "10001",
            "nai-diffusion-4-5-full",
            get_config,
        )

        self.assertEqual(index, 2)

    def test_model_config_appends_selfie_negative_prompt_when_selfie_enabled(self):
        host = _DummyModelConfigHost({
            "model": {
                "base_url": "https://example.com",
                "default_model": "nai-diffusion-4-5-full",
            },
            "model_nai4_5": {
                "negative_prompt_add": "model-default-negative",
                "selfie_negative_prompt_add": "selfie-negative",
            },
        })

        model_config = host._get_model_config(is_selfie=True)

        self.assertEqual(
            model_config["negative_prompt_add"],
            "selfie-negative, model-default-negative",
        )

    def test_model_config_does_not_append_selfie_negative_prompt_when_not_selfie(self):
        host = _DummyModelConfigHost({
            "model": {
                "base_url": "https://example.com",
                "default_model": "nai-diffusion-4-5-full",
            },
            "model_nai4_5": {
                "negative_prompt_add": "model-default-negative",
                "selfie_negative_prompt_add": "selfie-negative",
            },
        })

        model_config = host._get_model_config(is_selfie=False)

        self.assertEqual(model_config["negative_prompt_add"], "model-default-negative")

    def test_model_config_uses_selfie_negative_prompt_alone_when_base_negative_empty(self):
        host = _DummyModelConfigHost({
            "model": {
                "base_url": "https://example.com",
                "default_model": "nai-diffusion-4-5-full",
            },
            "model_nai4_5": {
                "negative_prompt_add": "",
                "selfie_negative_prompt_add": "selfie-negative",
            },
        })

        model_config = host._get_model_config(is_selfie=True)

        self.assertEqual(model_config["negative_prompt_add"], "selfie-negative")


if __name__ == "__main__":
    unittest.main()
