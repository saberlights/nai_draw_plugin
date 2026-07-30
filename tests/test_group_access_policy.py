from __future__ import annotations

import random

import pytest

from plugins.nai_draw_plugin.core.plugin_config import PLUGIN_CONFIG
from plugins.nai_draw_plugin.core.services.group_access_policy import (
    GroupAccessPolicy,
    tool_definition_name,
    without_tool,
)


def test_default_config_keeps_all_groups_available_in_blacklist_mode() -> None:
    defaults = PLUGIN_CONFIG.default_config()
    policy = GroupAccessPolicy.from_config(defaults)

    assert defaults["group_access"] == {
        "mode": "blacklist",
        "whitelist": [],
        "blacklist": [],
    }
    assert policy.has_group_restrictions is False
    assert policy.allows_scope("10001") is True
    assert policy.allows_scope("") is True
    assert policy.allows_scope(None) is True


def test_whitelist_only_allows_listed_groups_but_never_blocks_private_chat() -> None:
    policy = GroupAccessPolicy.from_config(
        {
            "group_access": {
                "mode": "whitelist",
                "whitelist": ["10001", " 10002 ", "10001", ""],
                "blacklist": ["10001"],
            }
        }
    )

    assert policy.has_group_restrictions is True
    assert policy.whitelist == frozenset({"10001", "10002"})
    assert policy.allows_scope("10001") is True
    assert policy.allows_scope("99999") is False
    assert policy.allows_scope("") is True
    assert policy.allows_scope(None) is False


def test_blacklist_rejects_exactly_the_listed_random_group_ids() -> None:
    rng = random.Random(20260731)
    group_ids = [str(rng.randrange(10**8, 10**9)) for _ in range(200)]
    blocked = set(rng.sample(group_ids, 67))
    policy = GroupAccessPolicy.from_config(
        {
            "group_access": {
                "mode": "blacklist",
                "whitelist": group_ids,
                "blacklist": sorted(blocked),
            }
        }
    )

    assert {
        group_id for group_id in group_ids if not policy.allows_scope(group_id)
    } == blocked
    assert policy.allows_scope("") is True
    assert policy.allows_scope(None) is False


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"mode": "allow"}, "group_access.mode"),
        ({"mode": "blacklist", "blacklist": "10001"}, "group_access.blacklist"),
        ({"mode": "whitelist", "whitelist": "10001"}, "group_access.whitelist"),
    ],
)
def test_invalid_access_config_fails_loudly(
    section: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GroupAccessPolicy.from_config({"group_access": section})


def test_planner_tool_filter_supports_openai_and_flattened_definitions() -> None:
    definitions = [
        {"type": "function", "function": {"name": "nai_web_draw"}},
        {"name": "nai_web_draw"},
        {"type": "function", "function": {"name": "query_memory"}},
        "invalid-but-preserved",
    ]

    filtered = without_tool(definitions, tool_name="nai_web_draw")

    assert filtered == [
        {"type": "function", "function": {"name": "query_memory"}},
        "invalid-but-preserved",
    ]
    assert tool_definition_name(filtered[0]) == "query_memory"


def test_webui_exposes_group_access_mode_as_closed_choice() -> None:
    webui = PLUGIN_CONFIG.webui_schema(plugin_id="nai_draw_plugin")
    mode = webui["sections"]["group_access"]["fields"]["mode"]

    assert mode["ui_type"] == "select"
    assert mode["choices"] == ["blacklist", "whitelist"]
    assert webui["sections"]["group_access"]["fields"]["whitelist"][
        "item_type"
    ] == "string"
