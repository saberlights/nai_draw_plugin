# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

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


dummy_logger_module.get_logger = lambda _name=None: _DummyLogger()
sys.modules["src.common.logger"] = dummy_logger_module

from plugins.nai_draw_plugin.core.services.danbooru_online_retriever import DanbooruOnlineRetriever


class _FakeClient:
    async def search(self, *args, **kwargs):
        return {
            "results": [
                {
                    "tag": "hatsune_miku",
                    "cn_name": "初音未来",
                    "final_score": 0.97,
                    "category": "character",
                },
                {
                    "tag": "vocaloid",
                    "cn_name": "Vocaloid",
                    "final_score": 0.88,
                    "category": "copyright",
                },
            ]
        }

    async def related(self, *args, **kwargs):
        return {
            "results": [
                {
                    "tag": "twintails",
                    "cn_name": "双马尾",
                    "sources": ["hatsune_miku", "vocaloid"],
                    "category": "general",
                },
                {
                    "tag": "hatsune_miku",
                    "cn_name": "初音未来",
                    "sources": ["hatsune_miku"],
                    "category": "character",
                },
            ]
        }


def test_retrieve_accepts_related_result_envelope() -> None:
    retriever = DanbooruOnlineRetriever()
    retriever.client = _FakeClient()

    result = asyncio.run(retriever.retrieve("初音未来"))

    assert result == {
        "search": [
            {
                "tag": "hatsune_miku",
                "cn_name": "初音未来",
                "score": 0.97,
                "category": "character",
            },
            {
                "tag": "vocaloid",
                "cn_name": "Vocaloid",
                "score": 0.88,
                "category": "copyright",
            },
        ],
        "related": [
            {
                "tag": "twintails",
                "cn_name": "双马尾",
                "cooc_score": None,
                "source_count": 2,
                "seed_count": 2,
                "category": "general",
            }
        ],
    }


def test_format_candidates_falls_back_to_source_coverage_when_cooc_score_missing() -> None:
    retriever = DanbooruOnlineRetriever()

    formatted = retriever.format_candidates(
        {
            "search": [],
            "related": [
                {
                    "tag": "green_halo",
                    "cn_name": "绿色光环",
                    "cooc_score": None,
                    "source_count": 3,
                    "seed_count": 8,
                    "category": "general",
                }
            ],
        }
    )

    assert "绿色光环 → green_halo [general] (共现来源 3/8)" in formatted


def test_format_candidates_hides_search_score_when_upstream_omits_it() -> None:
    retriever = DanbooruOnlineRetriever()

    formatted = retriever.format_candidates(
        {
            "search": [
                {
                    "tag": "hikari_(blue_archive)",
                    "cn_name": "橘光",
                    "score": None,
                    "category": "general",
                }
            ],
            "related": [],
        }
    )

    assert "橘光 → hikari_(blue_archive) [general]" in formatted
    assert "(相关度 0.00)" not in formatted
