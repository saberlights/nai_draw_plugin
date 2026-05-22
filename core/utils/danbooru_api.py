# -*- coding: utf-8 -*-
"""
Danbooru API 客户端 - 用于验证和查询画师标签
"""
import re
import asyncio
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import requests

from src.common.logger import get_logger

logger = get_logger("nai_draw_plugin")

# Danbooru API 基础配置
DANBOORU_API_BASE = "https://danbooru.donmai.us"
DANBOORU_TAGS_ENDPOINT = "/tags.json"
DANBOORU_RELATED_TAG_ENDPOINT = "/related_tag.json"
DANBOORU_ARTISTS_ENDPOINT = "/artists.json"
DANBOORU_POSTS_ENDPOINT = "/posts.json"

# 标签类别
TAG_CATEGORY_GENERAL = 0
TAG_CATEGORY_ARTIST = 1
TAG_CATEGORY_COPYRIGHT = 3
TAG_CATEGORY_CHARACTER = 4

# 最小推荐帖子数（低于此数量的画师可能不稳定）
MIN_RECOMMENDED_POST_COUNT = 100


class DanbooruAPI:
    """Danbooru API 客户端"""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NaiPicPlugin/1.0"
        })

    def search_artist(self, name: str) -> Optional[Dict]:
        """
        搜索单个画师标签

        Args:
            name: 画师名（Danbooru 格式）

        Returns:
            画师信息字典，包含 name, post_count 等，未找到返回 None
        """
        name_lower = name.lower()

        # 方法1: 使用 name_matches 精确匹配（比 search[name] 更可靠）
        try:
            params = {
                "search[category]": TAG_CATEGORY_ARTIST,
                "search[name_matches]": name_lower,
                "search[hide_empty]": "true",
                "limit": 5
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_TAGS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if data:
                    # 找完全匹配的
                    for item in data:
                        if item.get("name", "").lower() == name_lower:
                            return item
                    # 没有完全匹配，返回第一个结果（可能是近似）
                    if data[0].get("post_count", 0) > 0:
                        return data[0]
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 查询失败 (name_matches): {e}")

        # 方法2: 回退到精确搜索
        try:
            params = {
                "search[category]": TAG_CATEGORY_ARTIST,
                "search[name]": name_lower,
                "limit": 1
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_TAGS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 查询失败 (exact): {e}")

        return None

    def search_artists_batch(self, names: List[str]) -> Dict[str, Optional[Dict]]:
        """
        批量搜索画师标签

        Args:
            names: 画师名列表

        Returns:
            {画师名: 画师信息} 字典
        """
        results = {}
        for name in names:
            results[name] = self.search_artist(name)
        return results

    def get_popular_artists(self, limit: int = 50) -> List[Dict]:
        """
        获取热门画师列表（按帖子数排序）

        Args:
            limit: 返回数量

        Returns:
            画师信息列表
        """
        try:
            params = {
                "search[category]": TAG_CATEGORY_ARTIST,
                "search[order]": "count",
                "search[hide_empty]": "true",
                "limit": min(limit, 200)
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_TAGS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 获取热门画师失败: {e}")
            return []

    def search_tag(self, tag_name: str) -> Optional[Dict]:
        """
        精确搜索标签

        Args:
            tag_name: 标签名

        Returns:
            标签信息字典，或 None
        """
        try:
            params = {
                "search[name]": tag_name.lower().replace(" ", "_"),
                "limit": 1
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_TAGS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return None
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 标签搜索失败: {e}")
            return None

    def fuzzy_search_tag(self, partial_name: str, limit: int = 10) -> List[Dict]:
        """
        模糊搜索标签（通用标签，非画师）

        Args:
            partial_name: 部分标签名
            limit: 返回数量

        Returns:
            匹配的标签列表
        """
        try:
            # 优先搜索通用标签（category 0）
            params = {
                "search[name_matches]": f"*{partial_name.lower().replace(' ', '_')}*",
                "search[category]": TAG_CATEGORY_GENERAL,
                "search[order]": "count",
                "search[hide_empty]": "true",
                "limit": min(limit, 20)
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_TAGS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                results = response.json()
                if results:
                    return results

            # 如果通用标签没结果，搜索所有类别
            params.pop("search[category]")
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_TAGS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 标签模糊搜索失败: {e}")
            return []

    def fuzzy_search_artist(self, partial_name: str, limit: int = 10) -> List[Dict]:
        """
        模糊搜索画师

        Args:
            partial_name: 部分画师名
            limit: 返回数量

        Returns:
            匹配的画师列表
        """
        try:
            params = {
                "search[category]": TAG_CATEGORY_ARTIST,
                "search[name_matches]": f"*{partial_name.lower()}*",
                "search[order]": "count",
                "limit": min(limit, 50)
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_TAGS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 模糊搜索失败: {e}")
            return []

    def get_related_artists(self, artist_name: str, limit: int = 10) -> List[Dict]:
        """
        获取与指定画师相关/相似的画师

        Args:
            artist_name: 画师名
            limit: 返回数量

        Returns:
            相关画师列表，包含 tag 信息和相关度
        """
        try:
            params = {
                "query": artist_name.lower(),
                "category": TAG_CATEGORY_ARTIST
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_RELATED_TAG_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                # 返回格式可能是 {"query": "xxx", "tags": [...]} 或直接是列表
                if isinstance(data, dict) and "related_tags" in data:
                    related = data["related_tags"][:limit]
                    # 过滤掉自己
                    return [t for t in related if t.get("tag", {}).get("name") != artist_name.lower()]
                elif isinstance(data, list):
                    return data[:limit]
            return []
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 获取相关画师失败: {e}")
            return []

    def get_similar_artists_by_style(self, artist_names: List[str], limit: int = 5) -> List[str]:
        """
        根据多个画师找出可能风格相似的其他画师

        Args:
            artist_names: 画师名列表
            limit: 每个画师返回的相关画师数量

        Returns:
            相似画师名列表（去重，按出现频率排序）
        """
        from collections import Counter

        related_counter = Counter()

        for name in artist_names:
            related = self.get_related_artists(name, limit=limit)
            for item in related:
                # 处理不同的返回格式
                if isinstance(item, dict):
                    tag_info = item.get("tag", item)
                    related_name = tag_info.get("name", "")
                    if related_name and related_name not in artist_names:
                        related_counter[related_name] += 1

        # 返回出现频率最高的（说明与多个输入画师都相关）
        return [name for name, _ in related_counter.most_common(limit * 2)]

    def get_artist_info(self, artist_name: str) -> Optional[Dict]:
        """
        获取画师详细信息（包括别名 other_names）

        Args:
            artist_name: 画师名

        Returns:
            画师信息字典（含 name, other_names, group_name 等），或 None
        """
        try:
            params = {
                "search[name]": artist_name.lower(),
                "limit": 1
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_ARTISTS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return None
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 获取画师信息失败: {e}")
            return None

    def search_artist_by_other_name(self, query: str, limit: int = 5) -> List[Dict]:
        """
        通过别名搜索画师（other_names 字段）

        Args:
            query: 搜索词（如 "カントク"、"5th-year"）
            limit: 返回数量

        Returns:
            匹配的画师列表
        """
        try:
            params = {
                "search[any_other_name_like]": f"*{query}*",
                "limit": min(limit, 20)
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_ARTISTS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return [a for a in response.json() if not a.get("is_deleted", False)]
            return []
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 别名搜索失败: {e}")
            return []

    def get_artist_style_tags(self, artist_name: str, sample_size: int = 20) -> Dict[str, List[str]]:
        """
        通过采样画师的帖子，分析其风格特征标签

        Args:
            artist_name: 画师名
            sample_size: 采样帖子数

        Returns:
            {
                "common_tags": 出现频率最高的通用标签,
                "common_characters": 常画的角色,
                "common_copyrights": 常画的作品/版权
            }
        """
        from collections import Counter

        try:
            params = {
                "tags": artist_name.lower(),
                "limit": min(sample_size, 50)
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_POSTS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return {"common_tags": [], "common_characters": [], "common_copyrights": []}

            posts = response.json()
            if not posts:
                return {"common_tags": [], "common_characters": [], "common_copyrights": []}

            general_counter = Counter()
            character_counter = Counter()
            copyright_counter = Counter()

            for post in posts:
                general_tags = post.get("tag_string_general", "").split()
                character_tags = post.get("tag_string_character", "").split()
                copyright_tags = post.get("tag_string_copyright", "").split()

                general_counter.update(general_tags)
                character_counter.update(character_tags)
                copyright_counter.update(copyright_tags)

            # 过滤掉过于通用的标签（几乎所有帖子都有的）
            trivial_tags = {
                "1girl", "1boy", "solo", "highres", "absurdres",
                "commentary_request", "commentary", "translated",
                "translation_request", "simple_background", "white_background",
            }
            filtered_general = [
                tag for tag, _ in general_counter.most_common(30)
                if tag not in trivial_tags
            ]

            return {
                "common_tags": filtered_general[:15],
                "common_characters": [c for c, _ in character_counter.most_common(5)],
                "common_copyrights": [c for c, _ in copyright_counter.most_common(5)],
            }
        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 获取画师风格标签失败: {e}")
            return {"common_tags": [], "common_characters": [], "common_copyrights": []}

    def search_artists_by_tags(
        self, tags: List[str], sample_size: int = 100, min_artist_count: int = 2
    ) -> List[Dict]:
        """
        根据风格标签搜索相关画师（通过 posts 搜索）

        Args:
            tags: 风格标签列表，如 ["loli", "cute", "school_uniform"]
            sample_size: 采样帖子数
            min_artist_count: 画师最少出现次数（过滤偶然出现的）

        Returns:
            画师列表，按出现频率排序，包含风格标签
            [{"name": "kantoku", "count": 15, "post_count": 2422, "style_tags": [...]}, ...]
        """
        from collections import Counter

        if not tags:
            return []

        # 构建搜索标签（最多2个，Danbooru 匿名用户限制）
        search_tags = " ".join(tags[:2])

        try:
            params = {
                "tags": search_tags,
                "limit": min(sample_size, 200)
            }
            response = self.session.get(
                f"{DANBOORU_API_BASE}{DANBOORU_POSTS_ENDPOINT}",
                params=params,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.warning(f"[nai_pic] Danbooru posts 搜索失败: {response.status_code}")
                return []

            posts = response.json()
            if not posts:
                return []

            # 统计画师出现次数
            artist_counter = Counter()
            for post in posts:
                artist_tag = post.get("tag_string_artist", "").strip()
                if artist_tag and " " not in artist_tag:  # 单个画师
                    artist_counter[artist_tag] += 1
                elif artist_tag:  # 多个画师（collaboration）
                    for a in artist_tag.split():
                        artist_counter[a] += 1

            # 过滤出现次数太少的
            filtered_artists = [
                (name, count) for name, count in artist_counter.items()
                if count >= min_artist_count
            ]

            if not filtered_artists:
                # 如果过滤后没有，降低阈值
                filtered_artists = artist_counter.most_common(30)

            # 获取画师详细信息和风格标签（并发查询）
            top_artists = sorted(filtered_artists, key=lambda x: -x[1])[:25]

            def _fetch_artist_detail(item):
                artist_name, count = item
                artist_info = self.search_artist(artist_name)
                if not artist_info:
                    return None
                post_count = artist_info.get("post_count", 0)
                if post_count < MIN_RECOMMENDED_POST_COUNT:
                    return None
                style_info = self.get_artist_style_tags(artist_name, sample_size=15)
                style_tags = style_info.get("common_tags", [])[:6]
                return {
                    "name": artist_name,
                    "count": count,
                    "post_count": post_count,
                    "style_tags": style_tags
                }

            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(_fetch_artist_detail, top_artists))

            result = [r for r in results if r is not None]
            return result

        except Exception as e:
            logger.warning(f"[nai_pic] Danbooru API 按标签搜索画师失败: {e}")
            return []


def extract_artist_names_from_prompt(artist_prompt: str) -> List[str]:
    """
    从画师串中提取画师名

    Args:
        artist_prompt: 画师串，如 "1.2::artist:sususuyo ::, artist:fuzichoco"

    Returns:
        画师名列表
    """
    # 匹配 artist:xxx 格式
    pattern = r'artist:([a-zA-Z0-9_\-\(\)]+)'
    matches = re.findall(pattern, artist_prompt.lower())
    return list(set(matches))


def validate_artist_prompt(artist_prompt: str, api: DanbooruAPI = None) -> Tuple[bool, List[Dict], List[str]]:
    """
    验证画师串中的画师是否有效

    Args:
        artist_prompt: 画师串
        api: DanbooruAPI 实例（可选）

    Returns:
        (是否全部有效, 有效画师信息列表, 无效画师名列表)
    """
    if api is None:
        api = DanbooruAPI()

    artist_names = extract_artist_names_from_prompt(artist_prompt)
    if not artist_names:
        return False, [], []

    valid_artists = []
    invalid_artists = []

    for name in artist_names:
        info = api.search_artist(name)
        # 精确搜索失败时重试一次（应对 API 波动）
        if info is None:
            info = api.search_artist(name)
        if info and info.get("post_count", 0) > 0:
            valid_artists.append(info)
        else:
            invalid_artists.append(name)

    all_valid = len(invalid_artists) == 0
    return all_valid, valid_artists, invalid_artists


def extract_core_artist_name(artist_name: str) -> List[str]:
    """
    从画师名中提取核心部分，用于模糊搜索

    Args:
        artist_name: 完整画师名，如 "momoko_(vampire_killer)"

    Returns:
        核心名字列表，按优先级排序，如 ["momoko_(vampire_killer)", "momoko"]
    """
    results = [artist_name.lower()]

    # 提取括号前的部分：momoko_(xxx) -> momoko
    if "(" in artist_name or "_(" in artist_name:
        core = re.split(r'[_]?\(', artist_name)[0].strip("_")
        if core and core.lower() not in results:
            results.append(core.lower())

    # 提取下划线分割的第一部分：some_artist_name -> some
    parts = artist_name.split("_")
    if len(parts) > 1:
        first_part = parts[0].lower()
        if first_part and len(first_part) >= 3 and first_part not in results:
            results.append(first_part)

    return results


def try_correct_artist_name(invalid_name: str, api: DanbooruAPI = None) -> Tuple[Optional[Dict], List[Dict]]:
    """
    尝试使用模糊搜索和别名搜索纠正画师名拼写

    Args:
        invalid_name: 无效的画师名
        api: DanbooruAPI 实例

    Returns:
        (最可能的正确画师信息或None, 所有候选画师列表供LLM验证)
    """
    if api is None:
        api = DanbooruAPI()

    # 提取核心名字列表
    core_names = extract_core_artist_name(invalid_name)

    all_candidates = []
    seen_names = set()

    # 方法1: 依次用核心名字模糊搜索 tags
    for core_name in core_names:
        candidates = api.fuzzy_search_artist(core_name, limit=10)
        for c in candidates:
            name = c.get("name", "").lower()
            if name and name not in seen_names and name != invalid_name.lower():
                seen_names.add(name)
                all_candidates.append(c)

    # 方法2: 通过别名搜索（other_names），处理日文名等情况
    # 如用户输入 "カントク" → 找到 kantoku
    for core_name in core_names:
        alias_matches = api.search_artist_by_other_name(core_name, limit=5)
        for artist in alias_matches:
            # 从 /artists.json 返回的是画师信息，需要转换为 tag 格式
            artist_name = artist.get("name", "").lower()
            if artist_name and artist_name not in seen_names and artist_name != invalid_name.lower():
                # 获取对应的 tag 信息（包含 post_count）
                tag_info = api.search_artist(artist_name)
                if tag_info:
                    seen_names.add(artist_name)
                    all_candidates.append(tag_info)

    if not all_candidates:
        return None, []

    # 按帖子数排序
    all_candidates.sort(key=lambda x: x.get("post_count", 0), reverse=True)

    # 返回帖子数最多的作为首选，同时返回所有候选供 LLM 验证
    best_match = all_candidates[0] if all_candidates else None

    return best_match, all_candidates[:5]


def suggest_corrections_for_invalid(
    invalid_artists: List[str], api: DanbooruAPI = None
) -> Tuple[Dict[str, Optional[str]], Dict[str, List[Dict]]]:
    """
    为无效画师名批量建议纠正

    Args:
        invalid_artists: 无效画师名列表
        api: DanbooruAPI 实例

    Returns:
        (
            {无效名: 建议的正确名或None},
            {无效名: 候选画师列表，供LLM验证}
        )
    """
    if api is None:
        api = DanbooruAPI()

    corrections = {}
    candidates_map = {}

    for name in invalid_artists:
        best_match, candidates = try_correct_artist_name(name, api)
        if best_match:
            corrections[name] = best_match.get("name")
        else:
            corrections[name] = None
        candidates_map[name] = candidates

    return corrections, candidates_map


def get_artist_quality_score(artist_info: Dict) -> str:
    """
    根据帖子数量评估画师稳定性

    Args:
        artist_info: Danbooru 返回的画师信息

    Returns:
        质量等级: "S", "A", "B", "C", "D"
    """
    post_count = artist_info.get("post_count", 0)

    if post_count >= 5000:
        return "S"  # 非常稳定
    elif post_count >= 1000:
        return "A"  # 稳定
    elif post_count >= 500:
        return "B"  # 较稳定
    elif post_count >= 100:
        return "C"  # 一般
    else:
        return "D"  # 可能不稳定


def format_validation_result(valid_artists: List[Dict], invalid_artists: List[str]) -> str:
    """
    格式化验证结果为用户可读的文本

    Args:
        valid_artists: 有效画师信息列表
        invalid_artists: 无效画师名列表

    Returns:
        格式化的结果文本
    """
    lines = []

    if valid_artists:
        lines.append("✅ 有效画师：")
        for info in valid_artists:
            name = info.get("name", "unknown")
            count = info.get("post_count", 0)
            grade = get_artist_quality_score(info)
            lines.append(f"  • {name} [{grade}] ({count:,} posts)")

    if invalid_artists:
        lines.append("⚠️ 未找到：")
        for name in invalid_artists:
            lines.append(f"  • {name}")

    return "\n".join(lines)


def suggest_similar_artists(artist_names: List[str], api: DanbooruAPI = None) -> List[str]:
    """
    根据给定画师推荐相似风格的画师

    Args:
        artist_names: 画师名列表
        api: DanbooruAPI 实例

    Returns:
        推荐的相似画师名列表
    """
    if api is None:
        api = DanbooruAPI()

    return api.get_similar_artists_by_style(artist_names, limit=5)


def validate_and_correct_tags(tags: List[str], api: DanbooruAPI = None) -> List[str]:
    """
    验证标签是否在 Danbooru 存在，并尝试纠正不存在的标签

    Args:
        tags: LLM 提取的标签列表
        api: DanbooruAPI 实例

    Returns:
        验证/纠正后的有效标签列表
    """
    if api is None:
        api = DanbooruAPI()

    valid_tags = []
    seen = set()

    for tag in tags:
        if not tag or tag in seen:
            continue

        # 先检查标签是否存在
        tag_info = api.search_tag(tag)
        if tag_info and tag_info.get("post_count", 0) >= 100:
            valid_tags.append(tag_info["name"])
            seen.add(tag_info["name"])
            continue

        # 不存在则尝试模糊搜索
        fuzzy_results = api.fuzzy_search_tag(tag, limit=5)
        if fuzzy_results:
            # 只接受名称高度相似的纠正（原标签是纠正结果的前缀或完全包含）
            best = None
            for candidate in sorted(fuzzy_results, key=lambda x: x.get("post_count", 0), reverse=True):
                candidate_name = candidate.get("name", "")
                # 检查相似性：原标签是候选的前缀，或候选以原标签开头/结尾
                if (candidate_name.startswith(tag) or
                    candidate_name.endswith(tag) or
                    tag.replace("_", "") == candidate_name.replace("_", "")):
                    if candidate.get("post_count", 0) >= 100:
                        best = candidate
                        break

            if best and best["name"] not in seen:
                valid_tags.append(best["name"])
                seen.add(best["name"])
                logger.info(f"[nai_pic] 标签纠正: {tag} -> {best['name']}")

    return valid_tags
