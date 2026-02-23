from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


DESCRIPTION_FILES = [
    "description.txt",
    "description.md",
    "challenge.txt",
    "readme.md",
    "readme.txt",
]

SKIP_FILES = {
    "writeup.md",
    "writeup.txt",
    "solution.md",
    "solution.txt",
}

DEFAULT_SKIP_DIRS = {
    "writeup",
    "writeups",
    "solution",
    "solutions",
    "images",
    "img",
    "assets",
    ".git",
    ".venv",
    "__pycache__",
}


@dataclass
class LocalChallenge:
    challenge_id: str
    name: str
    category: str
    description: str
    url: str
    files: list[Path]
    root: Path
    value: int
    flag: str | None
    raw: dict


def discover_challenges(root: Path, limit: int = 0, skip_dirs: set[str] | None = None) -> list[LocalChallenge]:
    root = root.resolve()
    skip = {d.lower() for d in (skip_dirs or DEFAULT_SKIP_DIRS)}
    challenges: list[LocalChallenge] = []

    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if any(part.lower() in skip for part in path.parts):
            continue
        if _has_description(path) or (path / "challenge.json").exists():
            if _has_child_descriptions(path, skip):
                continue
            challenge = load_challenge_dir(path)
            if challenge:
                challenges.append(challenge)
        if limit and len(challenges) >= limit:
            break

    return challenges


def load_challenge_dir(
    root: Path,
    name_override: str = "",
    category_override: str = "",
    url_override: str = "",
    skip_dirs: set[str] | None = None,
) -> LocalChallenge | None:
    root = root.resolve()
    metadata = _read_metadata(root)
    description = metadata.get("description") or _read_description(root)
    if not description and not metadata:
        return None

    name = name_override or metadata.get("name") or root.name
    category = category_override or metadata.get("category") or "unknown"
    url = url_override or metadata.get("url") or metadata.get("connection_info") or ""
    value = int(metadata.get("value") or 100)
    flag = metadata.get("flag")
    files = _collect_local_files(root, skip_dirs=skip_dirs)

    return LocalChallenge(
        challenge_id=_slugify(root.name),
        name=name,
        category=category,
        description=description or "",
        url=url,
        files=files,
        root=root,
        value=value,
        flag=flag,
        raw=metadata,
    )


def _read_metadata(root: Path) -> dict:
    challenge_json = root / "challenge.json"
    if not challenge_json.exists():
        return {}
    try:
        return json.loads(challenge_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _collect_local_files(root: Path, skip_dirs: set[str] | None = None) -> list[Path]:
    skip = {d.lower() for d in (skip_dirs or DEFAULT_SKIP_DIRS)}
    attachments = root / "attachments"
    if attachments.exists():
        return [p for p in attachments.rglob("*") if p.is_file()]

    skip_names = {name.lower() for name in DESCRIPTION_FILES} | {"challenge.json"} | SKIP_FILES
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.lower() in skip for part in path.parts):
            continue
        if path.name.lower() in skip_names:
            continue
        files.append(path)
    return files


def _read_description(root: Path) -> str:
    for name in DESCRIPTION_FILES:
        path = root / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _has_description(root: Path) -> bool:
    return any((root / name).exists() for name in DESCRIPTION_FILES)


def _has_child_descriptions(root: Path, skip: set[str]) -> bool:
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if any(part.lower() in skip for part in child.parts):
            continue
        if _has_description(child) or (child / "challenge.json").exists():
            return True
    return False


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "local"
