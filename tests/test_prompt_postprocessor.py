# -*- coding: utf-8 -*-
import os
import unittest
import importlib.util


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOD_PATH = os.path.join(PLUGIN_DIR, "core", "utils", "prompt_postprocessor.py")

_spec = importlib.util.spec_from_file_location("prompt_postprocessor", MOD_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载模块: {MOD_PATH}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

normalize_prompt_order = _mod.normalize_prompt_order
remove_selfie_appearance_tags = _mod.remove_selfie_appearance_tags
sanitize_sfw_prompt = _mod.sanitize_sfw_prompt
user_mentions_appearance = _mod.user_mentions_appearance


class PromptPostprocessorTest(unittest.TestCase):
    def test_user_mentions_appearance_cn(self):
        self.assertTrue(user_mentions_appearance("自拍，黑长直"))
        self.assertTrue(user_mentions_appearance("金发蓝瞳"))

    def test_user_mentions_appearance_en(self):
        self.assertTrue(user_mentions_appearance("selfie with black hair"))
        self.assertFalse(user_mentions_appearance("自拍，微笑"))

    def test_remove_selfie_appearance_tags(self):
        s = "solo, 1girl, black hair, long hair, hair ribbon, blue eyes, smile"
        out = remove_selfie_appearance_tags(s)
        self.assertEqual(out, "solo, 1girl, hair ribbon, smile")

    def test_remove_selfie_appearance_multi(self):
        s = "2girls, street, year 2024\n|girl a, black hair, smile\n|girl b, blue eyes, smile"
        out = remove_selfie_appearance_tags(s)
        self.assertEqual(out, "2girls, street, year 2024\n| girl a, smile\n| girl b, smile")

    def test_remove_selfie_appearance_multi_single_line_pipe(self):
        s = "2girls, street, year 2024 | girl a, black hair, smile | girl b, blue eyes, smile"
        out = remove_selfie_appearance_tags(s)
        self.assertEqual(out, "2girls, street, year 2024 | girl a, smile | girl b, smile")

    def test_remove_selfie_appearance_preserves_trailing_commas_in_char_blocks(self):
        s = "2players, kissing, year 2026,\nchar1:man, black hair, smile,\nchar2:girl, blue eyes, smile,"
        out = remove_selfie_appearance_tags(s)
        self.assertEqual(out, "2players, kissing, year 2026,\nchar1:man, smile,\nchar2:girl, smile,")

    def test_normalize_prompt_order_year_last(self):
        s = "year 2024, smile, solo, pov, 1girl"
        out = normalize_prompt_order(s)
        self.assertEqual(out, "pov, solo, 1girl, smile, year 2024")

    def test_normalize_prompt_order_multi_single_line_pipe(self):
        s = "2girls, year 2024, street | smile, black hair | looking at viewer, blue eyes"
        out = normalize_prompt_order(s)
        self.assertEqual(out, "2girls, street, year 2024 | smile, black hair | looking at viewer, blue eyes")

    def test_normalize_prompt_order_preserves_trailing_commas_in_char_blocks(self):
        s = "close-up, 2players, year 2026,\nchar1:man, smile,\nchar2:girl, looking up,"
        out = normalize_prompt_order(s)
        self.assertEqual(out, "close-up, 2players, year 2026,\nchar1:man, smile,\nchar2:girl, looking up,")

    def test_sanitize_sfw_prompt_preserves_trailing_commas_in_char_blocks(self):
        s = "close-up, 2players, year 2026,\nchar1:man, nsfw, smile,\nchar2:girl, nsfw, blush,"
        out = sanitize_sfw_prompt(s)
        self.assertEqual(out, "close-up, 2players, year 2026,\nchar1:man, smile,\nchar2:girl, blush,")


if __name__ == "__main__":
    unittest.main()
