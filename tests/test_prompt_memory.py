# -*- coding: utf-8 -*-
import os
import unittest
import importlib.util


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOD_PATH = os.path.join(PLUGIN_DIR, "core", "services", "prompt_memory.py")

_spec = importlib.util.spec_from_file_location("prompt_memory", MOD_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载模块: {MOD_PATH}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

render_previous_prompt_block = _mod.render_previous_prompt_block
extract_last_prompt_from_record_display = _mod.extract_last_prompt_from_record_display
extract_last_context_from_record_display = _mod.extract_last_context_from_record_display
LAST_PROMPT_RECORD_PREFIX = _mod.LAST_PROMPT_RECORD_PREFIX


class PromptMemoryTest(unittest.TestCase):
    # ---- render_previous_prompt_block ----

    def test_render_without_last_prompt_returns_placeholder(self):
        """无上一轮提示词时返回占位块"""
        for val in (None, "", "   "):
            block = render_previous_prompt_block(val)
            self.assertIn("<previous_prompt_context>", block)
            self.assertIn("</previous_prompt_context>", block)
            self.assertIn("无上一轮提示词", block)

    def test_render_with_last_prompt_contains_xml_block(self):
        block = render_previous_prompt_block("solo, 1girl, smile")
        self.assertIn("<previous_prompt_context>", block)
        self.assertIn("</previous_prompt_context>", block)
        self.assertIn("solo, 1girl, smile", block)

    def test_render_three_tier_rules(self):
        """渲染块包含 A/B/C 三档关键词"""
        block = render_previous_prompt_block("solo, 1girl, smile")
        self.assertIn("微调", block)
        self.assertIn("换角色保场景", block)
        self.assertIn("全新主题", block)
        self.assertIn("必须遵守", block)

    def test_render_with_last_request(self):
        """传入 last_request 时注入上一轮用户请求"""
        block = render_previous_prompt_block("solo, 1girl", last_request="画一个女孩")
        self.assertIn("上一轮用户请求", block)
        self.assertIn("画一个女孩", block)

    def test_render_without_last_request(self):
        """不传 last_request 时不出现用户请求段落"""
        block = render_previous_prompt_block("solo, 1girl")
        self.assertNotIn("上一轮用户请求", block)

    def test_render_block_does_not_contain_old_compose_patterns(self):
        """render_previous_prompt_block should not contain old compose patterns"""
        block = render_previous_prompt_block("solo, 1girl, smile")
        # Old compose_prompt_generator_request patterns should be absent
        self.assertNotIn("可被丢弃", block)
        self.assertNotIn("本次用户要求", block)
        self.assertNotIn("<<USER_REQUEST>>", block)

    # ---- extract: new format ----

    def test_extract_new_format(self):
        """新格式（含 REQ: 行）正确解析"""
        display = f"{LAST_PROMPT_RECORD_PREFIX}\nREQ:画一个猫娘\n---\nsolo, 1girl, cat ears"
        prompt, request = extract_last_context_from_record_display(display)
        self.assertEqual(prompt, "solo, 1girl, cat ears")
        self.assertEqual(request, "画一个猫娘")

    def test_extract_old_format_compat(self):
        """旧格式（无 REQ: 行）仍可正常解析，request 返回 None"""
        display = f"{LAST_PROMPT_RECORD_PREFIX}\nsolo, 1girl, smile"
        prompt, request = extract_last_context_from_record_display(display)
        self.assertEqual(prompt, "solo, 1girl, smile")
        self.assertIsNone(request)

    def test_extract_empty_request(self):
        """REQ: 行存在但值为空时，request 返回 None"""
        display = f"{LAST_PROMPT_RECORD_PREFIX}\nREQ:\n---\nsolo, 1girl"
        prompt, request = extract_last_context_from_record_display(display)
        self.assertEqual(prompt, "solo, 1girl")
        self.assertIsNone(request)

    # ---- compat wrapper ----

    def test_extract_last_prompt_from_record_display(self):
        display = f"{LAST_PROMPT_RECORD_PREFIX}\nsolo, 1girl, smile"
        self.assertEqual(extract_last_prompt_from_record_display(display), "solo, 1girl, smile")

    def test_extract_returns_none_for_non_matching_prefix(self):
        self.assertIsNone(extract_last_prompt_from_record_display("other: x"))

    def test_extract_compat_new_format(self):
        """compat wrapper 也能解析新格式，只返回 prompt"""
        display = f"{LAST_PROMPT_RECORD_PREFIX}\nREQ:something\n---\ntags here"
        self.assertEqual(extract_last_prompt_from_record_display(display), "tags here")


if __name__ == "__main__":
    unittest.main()
