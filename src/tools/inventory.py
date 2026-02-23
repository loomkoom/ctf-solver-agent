from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


INVENTORY_PATH = Path(__file__).resolve().parent / "toolbox" / "tool_inventory.yaml"


@lru_cache(maxsize=1)
def load_tool_inventory() -> list[dict[str, Any]]:
    if not INVENTORY_PATH.exists():
        return []
    raw = INVENTORY_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        tools = data.get("tools") or []
        return tools if isinstance(tools, list) else []
    return []


@lru_cache(maxsize=1)
def build_tool_index() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    tools = load_tool_inventory()
    by_tool: dict[str, dict[str, Any]] = {}
    by_alias: dict[str, str] = {}
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        name = entry.get("tool_name")
        if not name:
            continue
        by_tool[name] = entry
        for key in entry.get("binaries") or []:
            if key:
                by_alias[str(key)] = name
        for key in entry.get("aliases") or []:
            if key:
                by_alias[str(key)] = name
    return by_tool, by_alias


def resolve_tool_manual_key(name: str) -> str | None:
    if not name:
        return None
    by_tool, by_alias = build_tool_index()
    if name in by_tool:
        return name
    return by_alias.get(name)
