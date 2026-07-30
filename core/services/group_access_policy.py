"""Group-level access policy for the plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_BLACKLIST_MODE = "blacklist"
_WHITELIST_MODE = "whitelist"
_SUPPORTED_MODES = frozenset({_BLACKLIST_MODE, _WHITELIST_MODE})


def _normalize_group_ids(values: Any, *, field_name: str) -> frozenset[str]:
    if not isinstance(values, list):
        raise ValueError(f"group_access.{field_name} 必须是数组")

    normalized: set[str] = set()
    for value in values:
        group_id = str(value or "").strip()
        if group_id:
            normalized.add(group_id)
    return frozenset(normalized)


@dataclass(frozen=True, slots=True)
class GroupAccessPolicy:
    """Decide whether the plugin is available in a resolved chat scope."""

    mode: str = _BLACKLIST_MODE
    whitelist: frozenset[str] = frozenset()
    blacklist: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, plugin_config: dict[str, Any]) -> "GroupAccessPolicy":
        raw_section = plugin_config.get("group_access", {})
        if not isinstance(raw_section, dict):
            raise ValueError("group_access 必须是配置表")

        mode = str(raw_section.get("mode", _BLACKLIST_MODE) or "").strip().lower()
        if mode not in _SUPPORTED_MODES:
            choices = ", ".join(sorted(_SUPPORTED_MODES))
            raise ValueError(f"group_access.mode 仅支持: {choices}")

        return cls(
            mode=mode,
            whitelist=_normalize_group_ids(
                raw_section.get("whitelist", []),
                field_name="whitelist",
            ),
            blacklist=_normalize_group_ids(
                raw_section.get("blacklist", []),
                field_name="blacklist",
            ),
        )

    @property
    def has_group_restrictions(self) -> bool:
        """Whether resolving the current chat scope is required."""
        return self.mode == _WHITELIST_MODE or bool(self.blacklist)

    def allows_scope(self, group_id: str | None) -> bool:
        """Allow private chats, evaluate groups, and reject unresolved restricted scopes."""
        if group_id is None:
            return not self.has_group_restrictions

        normalized_group_id = str(group_id).strip()
        if not normalized_group_id:
            return True
        if self.mode == _WHITELIST_MODE:
            return normalized_group_id in self.whitelist
        return normalized_group_id not in self.blacklist


def tool_definition_name(tool_definition: Any) -> str:
    """Extract a tool name from OpenAI-style or flattened tool definitions."""
    if not isinstance(tool_definition, dict):
        return ""
    function = tool_definition.get("function")
    if isinstance(function, dict):
        return str(function.get("name", "") or "").strip()
    return str(tool_definition.get("name", "") or "").strip()


def without_tool(
    tool_definitions: Iterable[Any],
    *,
    tool_name: str,
) -> list[Any]:
    """Return tool definitions without the named tool, preserving all other entries."""
    return [
        tool_definition
        for tool_definition in tool_definitions
        if tool_definition_name(tool_definition) != tool_name
    ]
