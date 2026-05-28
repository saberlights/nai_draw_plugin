# -*- coding: utf-8 -*-
import os
import unittest
import importlib.util


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARSER_PATH = os.path.join(PLUGIN_DIR, "core", "utils", "prompt_output_parser.py")

_spec = importlib.util.spec_from_file_location("prompt_output_parser", PARSER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载模块: {PARSER_PATH}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

parse_prompt_from_structured_output = _mod.parse_prompt_from_structured_output
parse_structured_prompt_payload = _mod.parse_structured_prompt_payload
extract_multi_character_payload = _mod.extract_multi_character_payload
extract_multi_character_payload_from_text = _mod.extract_multi_character_payload_from_text
resolve_multi_character_payload = _mod.resolve_multi_character_payload
extract_last_code_block = _mod.extract_last_code_block


class PromptOutputParserTest(unittest.TestCase):
    def test_parse_single_json(self):
        text = '{"format":"single","prompt":"solo, 1girl, smile","version":1}'
        self.assertEqual(parse_prompt_from_structured_output(text), "solo, 1girl, smile")

    def test_parse_multi_json_with_newlines(self):
        text = '{"format":"multi","prompt":"2girls, rain\\\\n|girl a, hug\\\\n|girl b, hug","version":1}'
        self.assertEqual(parse_prompt_from_structured_output(text), "2girls, rain\n|girl a, hug\n|girl b, hug")

    def test_parse_json_in_code_fence(self):
        text = '```json\n{"format":"single","prompt":"a, b","version":1}\n```'
        self.assertEqual(parse_prompt_from_structured_output(text), "a, b")

    def test_parse_json_with_noise(self):
        text = 'OK\\n{"prompt":"x, y","format":"single","version":1}\\nThanks'
        self.assertEqual(parse_prompt_from_structured_output(text), "x, y")

    def test_parse_fail_returns_none(self):
        self.assertIsNone(parse_prompt_from_structured_output("not json"))

    def test_parse_v2_arrays_single(self):
        text = '{"version":2,"format":"single","global":["solo","1girl","smile"],"people":[]}'
        self.assertEqual(parse_prompt_from_structured_output(text), "solo, 1girl, smile")

    def test_parse_v2_arrays_single_with_one_person(self):
        text = '{"version":2,"format":"single","global":["solo","1girl","cityscape"],"people":[["{roxy migurdia (mushoku tensei)}","standing"]]}'
        self.assertEqual(
            parse_prompt_from_structured_output(text),
            "solo, 1girl, cityscape, {roxy migurdia (mushoku tensei)}, standing"
        )

    def test_parse_v2_arrays_multi(self):
        text = (
            '{"version":2,"format":"multi",'
            '"global":["2girls","street","day","year 2024"],'
            '"people":[["girl a","smile"],["girl b","smile"]]}'
        )
        self.assertEqual(
            parse_prompt_from_structured_output(text),
            "2girls, street, day, year 2024,\nchar1:girl a, smile,\nchar2:girl b, smile,"
        )

    def test_parse_v3_arrays_single(self):
        text = (
            '{"version":3,"format":"single","intent":"selfie","continuity":"keep",'
            '"global":["solo","1girl","selfie","full body"],'
            '"people":[["black pantyhose","loafers"]]}'
        )
        self.assertEqual(
            parse_prompt_from_structured_output(text),
            "solo, 1girl, selfie, full body, black pantyhose, loafers"
        )

    def test_parse_v3_payload_metadata(self):
        text = (
            '{"version":3,"format":"single","intent":"selfie","continuity":"adjust",'
            '"global":["selfie","mirror selfie"],"people":[]}'
        )
        payload = parse_structured_prompt_payload(text)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["intent"], "selfie")
        self.assertEqual(payload["continuity"], "adjust")

    def test_extract_multi_character_payload_with_positions(self):
        text = (
            '{"version":3,"format":"multi","intent":"normal","continuity":"new",'
            '"global":["2girls","indoor","year 2025"],'
            '"people":[["girl","blue hair","blue dress"],["girl","white hair","white kimono"]],'
            '"positions":["B2","D4"]}'
        )
        payload = extract_multi_character_payload(text)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["global_text"], "2girls, indoor, year 2025")
        self.assertEqual(len(payload["characters"]), 2)
        self.assertEqual(payload["characters"][0]["prompt"], "girl, blue hair, blue dress")
        self.assertEqual(payload["characters"][0]["position"], "B2")
        self.assertEqual(payload["characters"][1]["position"], "D4")
        self.assertTrue(payload["has_coords"])

    def test_extract_multi_character_payload_without_positions(self):
        text = (
            '{"version":3,"format":"multi",'
            '"global":["2girls","park"],'
            '"people":[["girl","laughing"],["girl","running"]]}'
        )
        payload = extract_multi_character_payload(text)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["characters"]), 2)
        self.assertEqual(payload["characters"][0]["position"], "")
        self.assertFalse(payload["has_coords"])

    def test_extract_multi_character_payload_drops_invalid_position(self):
        text = (
            '{"version":3,"format":"multi",'
            '"global":["2girls","indoor"],'
            '"people":[["girl","a"],["girl","b"]],'
            '"positions":["X9","D3"]}'
        )
        payload = extract_multi_character_payload(text)
        self.assertIsNotNone(payload)
        # X9 不匹配 [A-E][1-5]，被规整为 ""，导致 has_coords=False
        self.assertEqual(payload["characters"][0]["position"], "")
        self.assertEqual(payload["characters"][1]["position"], "D3")
        self.assertFalse(payload["has_coords"])

    def test_extract_multi_character_payload_returns_none_for_single(self):
        text = (
            '{"version":3,"format":"single",'
            '"global":["solo","1girl"],'
            '"people":[["girl","smile"]]}'
        )
        self.assertIsNone(extract_multi_character_payload(text))

    def test_extract_multi_character_payload_returns_none_when_under_two(self):
        text = (
            '{"version":3,"format":"multi",'
            '"global":["1girl"],'
            '"people":[["girl","smile"]]}'
        )
        self.assertIsNone(extract_multi_character_payload(text))

    def test_extract_multi_character_payload_returns_none_for_v1(self):
        text = '{"version":1,"format":"multi","prompt":"x"}'
        self.assertIsNone(extract_multi_character_payload(text))

    def test_extract_from_text_multiline_charN_prefix(self):
        text = (
            "2girls, nsfw, indoor, year 2026,\n"
            "char1:girl, in foreground, {hatsune miku (vocaloid)}, blush,\n"
            "char2:girl, beside girl, {luo tianyi (vocaloid)}, closed eyes,"
        )
        payload = extract_multi_character_payload_from_text(text)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["global_text"], "2girls, nsfw, indoor, year 2026")
        self.assertEqual(len(payload["characters"]), 2)
        self.assertEqual(
            payload["characters"][0]["prompt"],
            "girl, in foreground, {hatsune miku (vocaloid)}, blush",
        )
        self.assertEqual(payload["characters"][0]["position"], "")
        self.assertFalse(payload["has_coords"])

    def test_extract_from_text_pipe_format(self):
        text = "2girls, street | girl a, smile | girl b, smile"
        payload = extract_multi_character_payload_from_text(text)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["global_text"], "2girls, street")
        self.assertEqual(len(payload["characters"]), 2)
        self.assertEqual(payload["characters"][0]["prompt"], "girl a, smile")

    def test_extract_from_text_returns_none_for_single_line(self):
        self.assertIsNone(extract_multi_character_payload_from_text("solo, 1girl, smile"))

    def test_extract_from_text_tolerates_chinese_colon(self):
        text = (
            "2girls, park,\n"
            "char1:girl, smile,\n"
            "char2:girl, laugh,"  # 半角冒号
        )
        payload = extract_multi_character_payload_from_text(text)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["characters"]), 2)
        self.assertEqual(payload["characters"][1]["prompt"], "girl, laugh")

    def test_resolve_prefers_json_when_available(self):
        json_text = (
            '{"version":3,"format":"multi",'
            '"global":["2girls","park"],'
            '"people":[["girl","smile"],["girl","laugh"]],'
            '"positions":["B3","D3"]}'
        )
        rendered = "2girls, park,\nchar1:girl, smile,\nchar2:girl, laugh,"
        payload = resolve_multi_character_payload(json_text, rendered)
        self.assertIsNotNone(payload)
        # JSON 路径才有 position
        self.assertEqual(payload["characters"][0]["position"], "B3")
        self.assertTrue(payload["has_coords"])

    def test_resolve_falls_back_to_text_when_json_missing(self):
        rendered = "2girls, park,\nchar1:girl, smile,\nchar2:girl, laugh,"
        payload = resolve_multi_character_payload("not a json", rendered)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["characters"]), 2)
        self.assertFalse(payload["has_coords"])

    def test_resolve_returns_none_for_single_person(self):
        rendered = "solo, 1girl, smile"
        self.assertIsNone(resolve_multi_character_payload("not a json", rendered))


class ExtractLastCodeBlockTest(unittest.TestCase):
    """LLM 输出"思考过程 + ```prompt``` "混合格式：抠最后一个代码块内容。

    覆盖场景：完整闭合 / 未闭合截断 / 多块取末尾 / lang 标识 / 无代码块。
    """

    def test_returns_none_when_no_triple_backtick(self):
        self.assertIsNone(extract_last_code_block(""))
        self.assertIsNone(extract_last_code_block("plain text without fence"))

    def test_extracts_content_of_fully_closed_block(self):
        text = "thought stuff\n```\n1girl, blue dress, smile\n```"
        self.assertEqual(
            extract_last_code_block(text),
            "1girl, blue dress, smile",
        )

    def test_picks_last_block_when_multiple(self):
        """多个代码块时取最后一个，因为 LLM 习惯把最终 prompt 放末尾。"""
        text = (
            "```json\n"
            '{"old": true}\n'
            "```\n"
            "more thought\n"
            "```\n"
            "1girl, final tags\n"
            "```"
        )
        self.assertEqual(extract_last_code_block(text), "1girl, final tags")

    def test_strips_language_identifier_line(self):
        """```python / ```json 这种 lang 行不应进入提取内容。"""
        text = "```python\n1girl, ok\n```"
        self.assertEqual(extract_last_code_block(text), "1girl, ok")

    def test_extracts_tail_when_truncated_unclosed(self):
        """LLM 在 max_tokens 截断时尾部 ``` 未闭合：取最后一个 ``` 之后的内容。"""
        text = (
            "thought\n"
            "- 角色：路障僵尸\n"
            "- 动作：吃冰淇淋\n"
            "```\n"
            "(masterpiece, best quality), solo, 1boy"
        )
        self.assertEqual(
            extract_last_code_block(text),
            "(masterpiece, best quality), solo, 1boy",
        )

    def test_real_llm_thought_plus_unclosed_block(self):
        """真实样本：thought 5 行 + 未闭合 ``` + 纯英文 prompt。"""
        text = (
            "thought\n"
            "- 意图判定：普通画图（normal）。\n"
            "- 角色特征：路障僵尸（Conehead Zombie）。\n"
            "- 动作：吃冰淇淋（eating ice cream）。\n"
            "- 构图：中景。\n"
            "```\n"
            "(masterpiece, best quality), solo, 1boy, conehead zombie, "
            "eating ice cream, street at night"
        )
        self.assertEqual(
            extract_last_code_block(text),
            "(masterpiece, best quality), solo, 1boy, conehead zombie, "
            "eating ice cream, street at night",
        )

    def test_empty_block_falls_through_to_tail_branch(self):
        """``` 内容为空时不应误返回空串，应继续走截断兜底。"""
        # 这里测试"开头空块 + 末尾真实块"的情况：取最后真实块
        text = "```\n```\nactual prompt tags"
        # 第一个块为空，第二个 ``` 后是 "actual prompt tags"——未闭合分支兜底
        result = extract_last_code_block(text)
        self.assertEqual(result, "actual prompt tags")


if __name__ == "__main__":
    unittest.main()
