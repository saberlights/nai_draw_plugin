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
strip_cjk_and_fullwidth = _mod.strip_cjk_and_fullwidth
strip_cjk_and_fullwidth_from_characters = _mod.strip_cjk_and_fullwidth_from_characters
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


class StripCjkAndFullwidthTest(unittest.TestCase):
    """LLM 翻译后清洗：剔除中文 / 日文假名 / 韩文 / 全角符号。

    NewAPI §8 要求 prompt 必须英文，含 CJK 一律 400；本组用例覆盖各种残留形态。
    """

    def test_pure_cjk_returns_empty(self):
        """整个 prompt 全是中文 → 清完应为空字符串（行被丢弃）。"""
        self.assertEqual(strip_cjk_and_fullwidth("一个女孩, 白色连衣裙"), "")

    def test_mixed_cn_en_in_same_tag_keeps_english(self):
        """单 tag 内中英混杂：保留英文部分，剔中文部分。"""
        out = strip_cjk_and_fullwidth("1girl, 红色 hair, blue dress")
        self.assertEqual(out, "1girl, hair, blue dress")

    def test_fullwidth_comma_normalized_to_tag_separator(self):
        """全角逗号 U+FF0C 转成英文逗号，保留 tag 分隔语义。"""
        out = strip_cjk_and_fullwidth("1girl，white dress，smile")
        self.assertEqual(out, "1girl, white dress, smile")

    def test_chinese_enumeration_mark_also_normalized(self):
        """中文顿号 U+3001 也按 tag 分隔语义处理。"""
        out = strip_cjk_and_fullwidth("1girl、white dress、smile")
        self.assertEqual(out, "1girl, white dress, smile")

    def test_japanese_kana_removed(self):
        """平假名 / 片假名都应剔除。"""
        out = strip_cjk_and_fullwidth("1girl, かわいい, アニメ, anime style")
        self.assertEqual(out, "1girl, anime style")

    def test_korean_hangul_removed(self):
        out = strip_cjk_and_fullwidth("1girl, 안녕하세요, smile")
        self.assertEqual(out, "1girl, smile")

    def test_fullwidth_alphanumeric_removed(self):
        """全角字母 / 数字（ＡＢＣ / １２３）也算全角符号，应剔除。"""
        out = strip_cjk_and_fullwidth("1girl, ＡＢＣ word, ６６６, normal")
        self.assertEqual(out, "1girl, word, normal")

    def test_pure_cjk_tag_dropped_neighbors_kept(self):
        """中间一个 tag 全是中文：该 tag 丢弃，前后英文 tag 保留。"""
        out = strip_cjk_and_fullwidth("1girl, 红色, blue dress, 微笑")
        self.assertEqual(out, "1girl, blue dress")

    def test_multiline_char_blocks_preserve_role_prefix(self):
        """多行 char1:/char2: 角色前缀不应被吞，每行内部独立清洗。"""
        s = (
            "2girls, year 2024,\n"
            "char1:girl a, 黑发, smile,\n"
            "char2:girl b, blue eyes, 微笑,"
        )
        out = strip_cjk_and_fullwidth(s)
        self.assertEqual(
            out,
            (
                "2girls, year 2024,\n"
                "char1:girl a, smile,\n"
                "char2:girl b, blue eyes,"
            ),
        )

    def test_single_line_pipe_format_each_segment_cleaned(self):
        """单行 `|` 多角色格式：每段独立清洗。"""
        out = strip_cjk_and_fullwidth(
            "2girls, street | smile, 黑发 | blue eyes, 红裙"
        )
        self.assertEqual(out, "2girls, street | smile | blue eyes")

    def test_empty_and_whitespace_only_returned_as_is(self):
        self.assertEqual(strip_cjk_and_fullwidth(""), "")
        self.assertEqual(strip_cjk_and_fullwidth("   "), "   ")

    def test_pure_english_passthrough(self):
        """纯英文 prompt 不应被改变（除尾随空格等正常化外）。"""
        s = "1girl, blue dress, smile, looking at viewer"
        self.assertEqual(strip_cjk_and_fullwidth(s), s)

    def test_cjk_punctuation_removed(self):
        """中文标点（。、？！「」『』）应被剔除。"""
        out = strip_cjk_and_fullwidth("1girl, smile。 looking。, normal")
        self.assertEqual(out, "1girl, smile looking, normal")

    def test_characters_payload_drops_fully_cjk_role(self):
        """多角色 payload：某个 character 的 prompt 全 CJK → 整个 character 被丢弃。"""
        g, chars = strip_cjk_and_fullwidth_from_characters(
            "2girls, 都市",
            [
                {"prompt": "1girl, 红发, blue dress", "position": "B2"},
                {"prompt": "全是中文标签", "position": "D4"},
                {"prompt": "1boy, smile", "position": "C3"},
            ],
        )
        self.assertEqual(g, "2girls")
        self.assertEqual(
            chars,
            [
                {"prompt": "1girl, blue dress", "position": "B2"},
                {"prompt": "1boy, smile", "position": "C3"},
            ],
        )

    def test_characters_payload_cleans_negative_prompt(self):
        """character[i].negative_prompt 也走同款清洗。"""
        _, chars = strip_cjk_and_fullwidth_from_characters(
            "scene",
            [
                {
                    "prompt": "1girl, blue hair",
                    "negative_prompt": "lowres, 模糊, bad anatomy",
                    "position": "C3",
                },
                {
                    "prompt": "1boy, red hair",
                    "negative_prompt": "lowres, blurry",
                    "position": "B3",
                },
            ],
        )
        self.assertEqual(chars[0]["negative_prompt"], "lowres, bad anatomy")
        self.assertEqual(chars[1]["negative_prompt"], "lowres, blurry")


if __name__ == "__main__":
    unittest.main()
