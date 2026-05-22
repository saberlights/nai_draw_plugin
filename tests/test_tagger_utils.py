# -*- coding: utf-8 -*-
import os
import unittest
import importlib.util


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOD_PATH = os.path.join(PLUGIN_DIR, "core", "utils", "tagger_utils.py")

_spec = importlib.util.spec_from_file_location("tagger_utils", MOD_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载模块: {MOD_PATH}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

extract_picids = _mod.extract_picids
find_reply_message_id = _mod.find_reply_message_id
parse_json_object = _mod.parse_json_object
normalize_output = _mod.normalize_output
extract_image_base64_list = _mod.extract_image_base64_list
extract_image_base64_list_from_payload = _mod.extract_image_base64_list_from_payload
guess_image_format_from_base64 = _mod.guess_image_format_from_base64
strip_data_url = _mod.strip_data_url


class TaggerUtilsTest(unittest.TestCase):
    def test_find_reply_message_id(self):
        seg = {
            "type": "seglist",
            "data": [
                {"type": "text", "data": "hello"},
                {"type": "reply", "data": "m123"},
                {"type": "text", "data": "/打标"},
            ],
        }
        self.assertEqual(find_reply_message_id(seg), "m123")

    def test_find_reply_message_id_dict_data(self):
        seg = {"type": "reply", "data": {"message_id": "m999"}}
        self.assertEqual(find_reply_message_id(seg), "m999")

    def test_extract_picids(self):
        s = "xxx [picid:abc-12345] yyy [picid:deadbeef] zzz"
        self.assertEqual(extract_picids(s), ["abc-12345", "deadbeef"])

    def test_parse_json_object_with_fence(self):
        s = "```json\n{\"TAG\": [\"1girl\"]}\n```"
        obj = parse_json_object(s)
        self.assertIsInstance(obj, dict)
        self.assertEqual(obj.get("TAG"), ["1girl"])

    def test_normalize_output_fills_prompt(self):
        obj = {"CHARACTER_TAG": ["hatsune_miku"], "WORK_TAG": ["vocaloid"], "TAG": ["1girl"], "BAD_TAG": ["blurry"]}
        out = normalize_output(obj)
        self.assertEqual(out["PROMPT"], "hatsune_miku, vocaloid, 1girl")
        self.assertEqual(out["NEGATIVE"], "blurry")

    def test_extract_image_base64_list_dict(self):
        seg = {"type": "seglist", "data": [{"type": "emoji", "data": "R0lGODxxxx"}, {"type": "text", "data": "x"}]}
        self.assertEqual(extract_image_base64_list(seg), ["R0lGODxxxx"])

    def test_extract_image_base64_list_from_payload(self):
        payload = {
            "reply_message": {
                "message_segment": {
                    "type": "seglist",
                    "data": [{"type": "emoji", "data": "iVBORw0KGgo="}],
                }
            }
        }
        self.assertEqual(extract_image_base64_list_from_payload(payload), ["iVBORw0KGgo="])

    def test_extract_image_base64_list_from_payload_reply_segment_nested(self):
        payload = {
            "type": "seglist",
            "data": [
                {
                    "type": "reply",
                    "data": {
                        "id": "123",
                        "message": [{"type": "emoji", "data": "R0lGODlhAQABAIAAAAUEBA=="}],
                    },
                },
                {"type": "text", "data": "/打标"},
            ],
        }
        self.assertEqual(extract_image_base64_list_from_payload(payload), ["R0lGODlhAQABAIAAAAUEBA=="])

    def test_strip_data_url(self):
        b64, fmt = strip_data_url("data:image/png;base64,iVBORw0KGgo=")
        self.assertEqual(fmt, "png")
        self.assertEqual(b64, "iVBORw0KGgo=")

    def test_guess_image_format_from_base64(self):
        self.assertEqual(guess_image_format_from_base64("iVBORw0KGgo="), "png")
        self.assertEqual(guess_image_format_from_base64("/9j/4AAQSkZJRg=="), "jpeg")
        self.assertEqual(guess_image_format_from_base64("R0lGODlhAQABAIAAAAUEBA=="), "gif")


if __name__ == "__main__":
    unittest.main()
