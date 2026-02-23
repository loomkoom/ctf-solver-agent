from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from src.challenges import LocalChallenge, discover_challenges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a local CTFd instance with challenges")
    parser.add_argument("--ctfd-url", required=True, help="Base URL for CTFd (e.g., http://localhost:8000)")
    parser.add_argument("--ctfd-token", default="", help="CTFd API token")
    parser.add_argument("--ctfd-username", default="", help="CTFd username")
    parser.add_argument("--ctfd-password", default="", help="CTFd password")
    parser.add_argument("--challenge-root", default="data/test_bench", help="Folder containing challenge dirs")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of challenges to import")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without creating anything")
    parser.add_argument("--default-value", type=int, default=100, help="Default challenge value if not specified")
    parser.add_argument("--default-category", default="misc", help="Default category if missing")
    parser.add_argument("--state", default="visible", choices=["visible", "hidden"], help="Challenge state")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if a challenge with the same name exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = _build_session(args.ctfd_url, args.ctfd_token, args.ctfd_username, args.ctfd_password)
    existing = {c.get("name") for c in _list_challenges(session, args.ctfd_url)}

    root = Path(args.challenge_root)
    challenges = discover_challenges(root, limit=args.limit)
    if not challenges:
        raise SystemExit(f"No challenges found under {root}")

    print(f"Found {len(challenges)} challenge(s) under {root}")
    for challenge in challenges:
        if args.skip_existing and challenge.name in existing:
            print(f"Skipping existing challenge: {challenge.name}")
            continue
        _create_challenge(
            session,
            args.ctfd_url,
            challenge,
            args.default_value,
            args.default_category,
            args.state,
            dry_run=args.dry_run,
        )


def _build_session(base_url: str, token: str, username: str, password: str) -> requests.Session:
    session = requests.Session()
    session.verify = True
    if token:
        session.headers.update({"Authorization": f"Token {token}"})
        return session
    if not username or not password:
        raise SystemExit("Provide --ctfd-token or --ctfd-username/--ctfd-password")
    login_api = f"{base_url.rstrip('/')}/api/v1/users/login"
    resp = session.post(login_api, json={"name": username, "password": password}, timeout=15)
    if resp.ok:
        data = resp.json().get("data", {})
        token_value = data.get("token")
        if token_value:
            session.headers.update({"Authorization": f"Token {token_value}"})
            return session
    raise SystemExit("Failed to authenticate to CTFd.")


def _list_challenges(session: requests.Session, base_url: str) -> list[dict[str, Any]]:
    resp = session.get(f"{base_url.rstrip('/')}/api/v1/challenges", timeout=15)
    if not resp.ok:
        return []
    return resp.json().get("data", [])


def _create_challenge(
    session: requests.Session,
    base_url: str,
    challenge: LocalChallenge,
    default_value: int,
    default_category: str,
    state: str,
    dry_run: bool = False,
) -> None:
    payload = {
        "name": challenge.name,
        "category": challenge.category or default_category,
        "description": challenge.description,
        "value": challenge.value or default_value,
        "type": "standard",
        "state": state,
        "connection_info": challenge.url or "",
    }

    if dry_run:
        print(json.dumps({"create_challenge": payload, "files": [f.name for f in challenge.files]}, indent=2))
        return

    resp = session.post(f"{base_url.rstrip('/')}/api/v1/challenges", json=payload, timeout=30)
    resp.raise_for_status()
    challenge_id = resp.json().get("data", {}).get("id")
    if not challenge_id:
        raise RuntimeError(f"Failed to create challenge: {challenge.name}")

    if challenge.flag:
        _create_flag(session, base_url, challenge_id, challenge.flag)

    for file in challenge.files:
        _upload_file(session, base_url, challenge_id, file)
    print(f"Imported challenge: {challenge.name} (id={challenge_id})")


def _create_flag(session: requests.Session, base_url: str, challenge_id: int, flag: str) -> None:
    payload = {
        "challenge_id": challenge_id,
        "content": flag,
        "type": "static",
    }
    resp = session.post(f"{base_url.rstrip('/')}/api/v1/flags", json=payload, timeout=15)
    resp.raise_for_status()


def _upload_file(session: requests.Session, base_url: str, challenge_id: int, path: Path) -> None:
    with path.open("rb") as handle:
        files = {"file": handle}
        resp = session.post(
            f"{base_url.rstrip('/')}/api/v1/challenges/{challenge_id}/files",
            files=files,
            timeout=60,
        )
    resp.raise_for_status()


if __name__ == "__main__":
    main()
