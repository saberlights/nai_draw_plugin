# -*- coding: utf-8 -*-
import os
import unittest
import importlib.util


PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOD_PATH = os.path.join(PLUGIN_DIR, "core", "utils", "random_scene_description.py")

_spec = importlib.util.spec_from_file_location("random_scene_description", MOD_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载模块: {MOD_PATH}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

normalize_random_scene_description = _mod.normalize_random_scene_description
build_random_scene_signature = _mod.build_random_scene_signature
calculate_random_scene_repeat_score = _mod.calculate_random_scene_repeat_score
is_random_scene_too_similar = _mod.is_random_scene_too_similar


class RandomSceneDescriptionTest(unittest.TestCase):
    def test_normalize_common_danbooru_unfriendly_terms(self):
        text = "1男1女 水手服 站立后入 内射 失神 POV 天台"
        self.assertEqual(
            normalize_random_scene_description(text),
            "1个男性 1个女性 水手服 站立后入 内射 失神 第一人称视角 屋顶",
        )

    def test_normalize_punctuation_and_count_tokens(self):
        text = "2女，镜子自拍、俯视 / 床上"
        self.assertEqual(
            normalize_random_scene_description(text),
            "2个女性 镜子自拍 俯视镜头 在床上",
        )

    def test_build_signature_should_merge_medical_expansion_cluster(self):
        signature = build_random_scene_signature("1个女性 窥器 阴道扩张 插管 手术台 拘束")
        self.assertIn("簇:医疗拘束", signature)
        self.assertIn("窥器", signature)
        self.assertIn("扩张", signature)
        self.assertIn("插管", signature)
        self.assertIn("医疗场景", signature)
        self.assertIn("拘束", signature)

    def test_repeat_score_should_be_high_for_same_medical_cluster(self):
        score = calculate_random_scene_repeat_score(
            "1个女性 窥器 阴道扩张 插管 手术台 拘束",
            "1个女性 实验室 扩张器 导尿 拘束衣 俯视",
        )
        self.assertGreaterEqual(score, 0.6)

    def test_repeat_score_should_be_low_for_different_clusters(self):
        score = calculate_random_scene_repeat_score(
            "1个女性 窥器 阴道扩张 插管 手术台 拘束",
            "1个女性 修女 教堂 颜射 口交 神像",
        )
        self.assertLess(score, 0.4)

    def test_is_random_scene_too_similar_should_detect_cluster_repeat(self):
        recent = [
            "1个女性 窥器 阴道扩张 插管 手术台 拘束",
            "1个女性 地牢 拘束 窥器 扩张",
        ]
        self.assertTrue(is_random_scene_too_similar("1个女性 实验室 扩张器 导尿 拘束衣", recent))
        self.assertFalse(is_random_scene_too_similar("1个女性 修女 教堂 颜射 口交 神像", recent))


if __name__ == "__main__":
    unittest.main()
