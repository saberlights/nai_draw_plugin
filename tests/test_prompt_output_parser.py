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


if __name__ == "__main__":
    unittest.main()
