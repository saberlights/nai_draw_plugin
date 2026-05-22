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

src_package = types.ModuleType("src")
src_package.__path__ = [os.path.join(MAIBOT_ROOT, "src")]
sys.modules.setdefault("src", src_package)

plugin_system_module = types.ModuleType("src.plugin_system")
plugin_system_module.llm_api = types.SimpleNamespace()
sys.modules["src.plugin_system"] = plugin_system_module

plugin_system_base_package = types.ModuleType("src.plugin_system.base")
sys.modules["src.plugin_system.base"] = plugin_system_base_package

base_command_module = types.ModuleType("src.plugin_system.base.base_command")


class _DummyBaseCommand:
    pass


base_command_module.BaseCommand = _DummyBaseCommand
sys.modules["src.plugin_system.base.base_command"] = base_command_module

core_package = types.ModuleType("core")
core_package.__path__ = [os.path.join(PLUGIN_DIR, "core")]
sys.modules.setdefault("core", core_package)

core_commands_package = types.ModuleType("core.commands")
core_commands_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "commands")]
sys.modules.setdefault("core.commands", core_commands_package)

core_clients_package = types.ModuleType("core.clients")
core_clients_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "clients")]
sys.modules.setdefault("core.clients", core_clients_package)

core_mixins_package = types.ModuleType("core.mixins")
core_mixins_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "mixins")]
sys.modules.setdefault("core.mixins", core_mixins_package)

core_rules_package = types.ModuleType("core.rules")
core_rules_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "rules")]
sys.modules.setdefault("core.rules", core_rules_package)

core_services_package = types.ModuleType("core.services")
core_services_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "services")]
sys.modules.setdefault("core.services", core_services_package)

core_utils_package = types.ModuleType("core.utils")
core_utils_package.__path__ = [os.path.join(PLUGIN_DIR, "core", "utils")]
sys.modules.setdefault("core.utils", core_utils_package)

nai_web_client_module = types.ModuleType("core.clients.nai_web_client")


class _DummyNaiWebClient:
    def __init__(self, *_args, **_kwargs):
        return None


nai_web_client_module.NaiWebClient = _DummyNaiWebClient
sys.modules["core.clients.nai_web_client"] = nai_web_client_module

auto_recall_mixin_module = types.ModuleType("core.mixins.auto_recall_mixin")


class _DummyAutoRecallMixin:
    pass


auto_recall_mixin_module.AutoRecallMixin = _DummyAutoRecallMixin
sys.modules["core.mixins.auto_recall_mixin"] = auto_recall_mixin_module

image_url_helper_module = types.ModuleType("core.utils.image_url_helper")
image_url_helper_module.save_base64_image_to_file = lambda *_args, **_kwargs: ""
sys.modules["core.utils.image_url_helper"] = image_url_helper_module

model_config_mixin_module = types.ModuleType("core.mixins.model_config_mixin")


class _DummyModelConfigMixin:
    pass


model_config_mixin_module.ModelConfigMixin = _DummyModelConfigMixin
sys.modules["core.mixins.model_config_mixin"] = model_config_mixin_module

prompt_rules_module = types.ModuleType("core.rules.prompt_rules")
prompt_rules_module.PROMPT_GENERATOR_TEMPLATE = ""
prompt_rules_module.SFW_PROMPT_GENERATOR_TEMPLATE = ""
sys.modules["core.rules.prompt_rules"] = prompt_rules_module

selfie_rules_module = types.ModuleType("core.rules.selfie_rules")
selfie_rules_module.detect_selfie_from_output = lambda *_args, **_kwargs: False
selfie_rules_module.get_selfie_hint = lambda: ""
selfie_rules_module.merge_selfie_prompt = lambda *args, **kwargs: ""
sys.modules["core.rules.selfie_rules"] = selfie_rules_module

session_state_module = types.ModuleType("core.services.session_state")
session_state_module.session_state = types.SimpleNamespace()
sys.modules["core.services.session_state"] = session_state_module

tag_retriever_module = types.ModuleType("core.services.tag_retriever")
tag_retriever_module.get_tag_retriever = lambda *_args, **_kwargs: None
sys.modules["core.services.tag_retriever"] = tag_retriever_module

prompt_output_parser_module = types.ModuleType("core.utils.prompt_output_parser")
prompt_output_parser_module.parse_prompt_from_structured_output = lambda *_args, **_kwargs: None
sys.modules["core.utils.prompt_output_parser"] = prompt_output_parser_module

prompt_postprocessor_module = types.ModuleType("core.utils.prompt_postprocessor")
prompt_postprocessor_module.normalize_prompt_order = lambda prompt: prompt
prompt_postprocessor_module.remove_selfie_appearance_tags = lambda prompt: prompt
prompt_postprocessor_module.sanitize_sfw_prompt = lambda prompt: prompt
prompt_postprocessor_module.user_mentions_appearance = lambda *_args, **_kwargs: False
sys.modules["core.utils.prompt_postprocessor"] = prompt_postprocessor_module

constants_module = types.ModuleType("core.constants")
constants_module.NAI_PIC_IMAGE_DISPLAY_MARKER = ""
sys.modules["core.constants"] = constants_module

nai_draw_module = importlib.import_module("core.commands.nai_draw_command")
NaiDrawCommand = nai_draw_module.NaiDrawCommand


class RandomScenePromptTest(unittest.TestCase):
    def test_random_scene_prompt_should_not_prepend_custom_system_prompt(self):
        cmd = object.__new__(NaiDrawCommand)
        cmd.get_config = lambda key, default=None: {
            "custom_prompt.system_prompt": "SHOULD NOT APPEAR",
        }.get(key, default)

        prompt = cmd._build_random_scene_prompt()

        self.assertTrue(prompt.startswith("随机生成一个二次元 NSFW 场景"))
        self.assertNotIn("SHOULD NOT APPEAR", prompt)


if __name__ == "__main__":
    unittest.main()
