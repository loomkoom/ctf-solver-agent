from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.agent import build_graph
from src.config import settings
from src.ctfd import CTFdConnector
from src.state import DCipherState
from src.tools.rag import ingest_knowledge_base
from src.tools.toolbox import Toolbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI CTF Solver (CTFd + sandboxed Kali)")
    parser.add_argument("--ctfd-url", default=settings.ctfd_url, help="Base URL for CTFd")
    parser.add_argument("--ctfd-token", default=_secret(settings.ctfd_token), help="CTFd API token")
    parser.add_argument("--ctfd-username", default=settings.ctfd_username, help="CTFd username")
    parser.add_argument("--ctfd-password", default=_secret(settings.ctfd_password), help="CTFd password")
    parser.add_argument("--no-tls-verify", action="store_true", help="Disable TLS verification")
    parser.add_argument("--challenge-id", action="append", help="Specific challenge ID(s) to solve")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of challenges")
    parser.add_argument("--knowledge-base", default=settings.knowledge_base_path)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--reindex", action="store_true", help="Force reindex of the knowledge base")
    parser.add_argument("--flag-format", default=settings.flag_format)
    parser.add_argument("--max-iterations", type=int, default=settings.max_iterations)
    parser.add_argument("--sandbox-container", default=settings.sandbox_container)
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ctfd_url:
        raise SystemExit("CTFd URL is required (--ctfd-url or CTFD_URL).")

    settings.sandbox_container = args.sandbox_container
    settings.max_iterations = args.max_iterations
    settings.flag_format = args.flag_format

    if not args.skip_rag:
        kb_path = Path(args.knowledge_base).resolve()
        print(f"Indexing knowledge base from {kb_path} ...")
        ingest_knowledge_base(str(kb_path), force=args.reindex)

    connector = CTFdConnector(
        base_url=args.ctfd_url,
        token=args.ctfd_token or None,
        username=args.ctfd_username or None,
        password=args.ctfd_password or None,
        verify_tls=not args.no_tls_verify,
        timeout_s=settings.ctfd_timeout_s,
    )

    toolbox = Toolbox()
    graph = build_graph(toolbox, connector=connector)

    if args.challenge_id:
        challenges = [{"id": int(cid)} for cid in args.challenge_id]
    else:
        challenges = connector.list_challenges()

    if args.limit:
        challenges = challenges[: args.limit]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    for challenge in challenges:
        challenge_id = int(challenge["id"])
        details = connector.get_challenge(challenge_id)
        result = solve_challenge(details, connector, toolbox, graph)

        out_path = log_dir / f"ctfd_{challenge_id}_{run_id}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Challenge {challenge_id} done: flags={result.get('flag_candidates')}")


def solve_challenge(details: dict, connector: CTFdConnector, toolbox: Toolbox, graph) -> dict:
    challenge_id = int(details.get("id"))
    name = details.get("name") or f"challenge_{challenge_id}"
    category = details.get("category") or "unknown"
    description = details.get("description") or ""

    local_dir = Path("data/ctfd_cache") / str(challenge_id)
    files = connector.download_challenge_files(challenge_id, local_dir)
    container_dir = f"{settings.sandbox_workdir}/ctfd/{challenge_id}"
    if files:
        toolbox.run(f"mkdir -p {container_dir}")
        for file in files:
            toolbox.copy_to_container(str(file), f"{container_dir}/{file.name}")

    file_list = [f.name for f in files]
    context = _build_challenge_context(name, category, description, container_dir, file_list)

    state: DCipherState = {
        "challenge_id": str(challenge_id),
        "challenge_name": name,
        "category": category,
        "description": description,
        "current_files": file_list,
        "attempt_history": [],
        "reasoning_log": [],
        "messages": [],
        "challenge_context": context,
        "flag_format": settings.flag_format,
        "current_objective": "Initial reconnaissance",
        "plan": "",
        "last_command": "",
        "last_output": "",
        "last_error": "",
        "flag_candidates": [],
        "iteration": 0,
        "submitted_flags": [],
        "done": False,
    }

    final_state = graph.invoke(state)
    return {
        "challenge_id": challenge_id,
        "challenge_name": name,
        "category": category,
        "flag_candidates": final_state.get("flag_candidates"),
        "submitted_flags": final_state.get("submitted_flags"),
        "iteration": final_state.get("iteration"),
        "done": final_state.get("done"),
        "attempts": final_state.get("attempt_history"),
        "reasoning_log": final_state.get("reasoning_log"),
    }


def _build_challenge_context(
    name: str,
    category: str,
    description: str,
    container_dir: str,
    file_list: list[str],
) -> str:
    files = "\n".join(f"- {fname}" for fname in file_list) if file_list else "None"
    return (
        f"CHALLENGE: {name}\n"
        f"CATEGORY: {category}\n"
        f"DESCRIPTION:\n{description}\n\n"
        f"FILES (container path {container_dir}):\n{files}\n"
    )


def _secret(value) -> str:
    if value is None:
        return ""
    try:
        return value.get_secret_value()
    except AttributeError:
        return str(value)


if __name__ == "__main__":
    main()
