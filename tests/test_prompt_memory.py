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


class PromptMemoryTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
