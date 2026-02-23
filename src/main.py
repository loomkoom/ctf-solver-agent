from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.challenges import LocalChallenge, load_challenge_dir
from src.config import settings
from src.ctfd import CTFdConnector
from src.solver import run_solver
from src.tools.rag import ingest_knowledge_base
from src.tools.toolbox import Toolbox
from src.llm_tiers import load_tiers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI CTF Solver (CTFd + sandboxed Kali)")
    parser.add_argument("--ctfd-url", default=settings.ctfd_url, help="Base URL for CTFd")
    parser.add_argument("--ctfd-token", default=_secret(settings.ctfd_token), help="CTFd API token")
    parser.add_argument("--ctfd-username", default=settings.ctfd_username, help="CTFd username")
    parser.add_argument("--ctfd-password", default=_secret(settings.ctfd_password), help="CTFd password")
    parser.add_argument("--no-tls-verify", action="store_true", help="Disable TLS verification")
    parser.add_argument("--challenge-id", action="append", help="Specific challenge ID(s) to solve")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of challenges")
    parser.add_argument("--challenge-dir", help="Local challenge folder (overrides CTFd)")
    parser.add_argument("--challenge-name", default="", help="Override local challenge name")
    parser.add_argument("--challenge-category", default="", help="Override local challenge category")
    parser.add_argument("--challenge-url", default="", help="Explicit URL for web challenges")
    parser.add_argument("--knowledge-base", default=settings.knowledge_base_path)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--reindex", action="store_true", help="Force reindex of the knowledge base")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output/logging")
    parser.add_argument(
        "--rag-mode",
        default=settings.rag_mode,
        choices=["all", "no_writeups", "methodology"],
        help="RAG ingestion mode: all, no_writeups, or methodology",
    )
    parser.add_argument("--flag-format", default=settings.flag_format)
    parser.add_argument("--planner-provider", default=settings.planner_provider)
    parser.add_argument("--planner-model", default=settings.planner_model)
    parser.add_argument("--planner-tiers", default=settings.planner_tiers, help="Planner model tiers (JSON or CSV).")
    parser.add_argument("--planner-max-tokens", type=int, default=settings.planner_max_tokens)
    parser.add_argument("--executor-provider", default=settings.executor_provider)
    parser.add_argument("--executor-model", default=settings.executor_model)
    parser.add_argument("--executor-tiers", default=settings.executor_tiers, help="Executor model tiers (JSON or CSV).")
    parser.add_argument("--executor-max-tokens", type=int, default=settings.executor_max_tokens)
    parser.add_argument("--tier-phase-cycles", default=settings.tier_phase_cycles, help="Tier thresholds for phase cycles (comma-separated).")
    parser.add_argument("--tier-tool-calls", default=settings.tier_tool_calls, help="Tier thresholds for tool calls (comma-separated).")
    parser.add_argument("--tier-category-pivots", default=settings.tier_category_pivots, help="Tier thresholds for category pivots (comma-separated).")
    parser.add_argument("--tier-recent-failures", default=settings.tier_recent_failures, help="Tier thresholds for recent failures (comma-separated).")
    parser.add_argument("--tier-failure-window", type=int, default=settings.tier_failure_window, help="Number of recent attempts considered for failure escalation.")
    parser.add_argument("--max-iterations", type=int, default=settings.max_iterations)
    parser.add_argument("--max-tool-calls", type=int, default=settings.max_tool_calls)
    parser.add_argument("--max-phase-cycles", type=int, default=settings.max_phase_cycles)
    parser.add_argument("--max-wall-seconds", type=int, default=settings.max_wall_seconds_per_challenge)
    parser.add_argument("--max-category-pivots", type=int, default=settings.max_category_pivots)
    parser.add_argument("--tool-timeout-seconds", type=int, default=settings.tool_timeout_seconds)
    parser.add_argument("--sandbox-container", default=settings.sandbox_container)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--runs-dir", default=settings.runs_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_phase_cycles = settings.max_phase_cycles
    default_iterations = settings.max_iterations
    settings.sandbox_container = args.sandbox_container
    settings.max_iterations = args.max_iterations
    settings.flag_format = args.flag_format
    settings.max_tool_calls = args.max_tool_calls
    settings.max_phase_cycles = args.max_phase_cycles
    settings.max_wall_seconds_per_challenge = args.max_wall_seconds
    settings.max_category_pivots = args.max_category_pivots
    settings.tool_timeout_seconds = args.tool_timeout_seconds
    settings.sandbox_timeout_s = args.tool_timeout_seconds
    settings.runs_dir = args.runs_dir
    settings.rag_mode = args.rag_mode
    settings.rag_include_writeups = args.rag_mode not in {"no_writeups", "methodology"}
    settings.planner_provider = args.planner_provider
    settings.planner_model = args.planner_model
    settings.planner_tiers = args.planner_tiers
    settings.planner_max_tokens = args.planner_max_tokens
    settings.executor_provider = args.executor_provider
    settings.executor_model = args.executor_model
    settings.executor_tiers = args.executor_tiers
    settings.executor_max_tokens = args.executor_max_tokens
    settings.tier_phase_cycles = args.tier_phase_cycles
    settings.tier_tool_calls = args.tier_tool_calls
    settings.tier_category_pivots = args.tier_category_pivots
    settings.tier_recent_failures = args.tier_recent_failures
    settings.tier_failure_window = args.tier_failure_window
    settings.debug = args.debug
    if args.max_iterations != default_iterations and args.max_phase_cycles == default_phase_cycles:
        settings.max_phase_cycles = args.max_iterations

    _preflight_llm()

    settings.rag_enabled = not args.skip_rag
    if not args.skip_rag:
        kb_path = Path(args.knowledge_base).resolve()
        print(f"Indexing knowledge base from {kb_path} ...")
        ingest_knowledge_base(str(kb_path), force=args.reindex, mode=args.rag_mode)

    toolbox = Toolbox()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.challenge_dir:
        if not Path(args.challenge_dir).exists():
            raise SystemExit(f"Challenge dir not found: {args.challenge_dir}")
        challenge = load_challenge_dir(
            Path(args.challenge_dir),
            name_override=args.challenge_name,
            category_override=args.challenge_category,
            url_override=args.challenge_url,
        )
        if not challenge:
            raise SystemExit("No challenge description or metadata found in --challenge-dir.")
        result = _solve_local_challenge(challenge, toolbox, run_id)
        out_path = log_dir / f"local_{challenge.challenge_id}_{run_id}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _print_result(result)
        if settings.debug:
            _print_debug_info(result, run_id)
        return

    if not args.ctfd_url:
        raise SystemExit("Provide --challenge-dir or --ctfd-url/CTFD_URL.")

    connector = CTFdConnector(
        base_url=args.ctfd_url,
        token=args.ctfd_token or None,
        username=args.ctfd_username or None,
        password=args.ctfd_password or None,
        verify_tls=not args.no_tls_verify,
        timeout_s=settings.ctfd_timeout_s,
    )

    if args.challenge_id:
        challenges = [{"id": int(cid)} for cid in args.challenge_id]
    else:
        challenges = connector.list_challenges()

    if args.limit:
        challenges = challenges[: args.limit]

    for challenge in challenges:
        challenge_id = int(challenge["id"])
        details = connector.get_challenge(challenge_id)
        result = _solve_ctfd_challenge(details, connector, toolbox, run_id)

        out_path = log_dir / f"ctfd_{challenge_id}_{run_id}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _print_result(result)
        if settings.debug:
            _print_debug_info(result, run_id)


def _solve_ctfd_challenge(details: dict, connector: CTFdConnector, toolbox: Toolbox, run_id: str) -> dict:
    challenge_id = int(details.get("id"))
    name = details.get("name") or f"challenge_{challenge_id}"
    category = details.get("category") or "unknown"
    description = details.get("description") or ""
    url = details.get("connection_info") or ""

    local_dir = Path("data/ctfd_cache") / str(challenge_id)
    files = connector.download_challenge_files(challenge_id, local_dir)
    container_dir = f"{settings.sandbox_workdir}/ctfd/{challenge_id}"
    if files:
        toolbox.run(f"mkdir -p {json.dumps(container_dir)}")
        for file in files:
            toolbox.copy_to_container(str(file), f"{container_dir}/{file.name}")

    return run_solver(
        challenge_id=str(challenge_id),
        name=name,
        category=category,
        description=description,
        files=files,
        container_dir=container_dir,
        url=url,
        connector=connector,
        toolbox=toolbox,
        run_id=run_id,
    )


def _solve_local_challenge(challenge: LocalChallenge, toolbox: Toolbox, run_id: str) -> dict:
    challenge_id = challenge.challenge_id
    container_dir = f"{settings.sandbox_workdir}/local/{run_id}/{challenge_id}"
    files: list[Path] = challenge.files
    if files:
        toolbox.run(f"mkdir -p {json.dumps(container_dir)}")
        for file in files:
            rel = file.relative_to(challenge.root)
            dest = f"{container_dir}/{rel.as_posix()}"
            toolbox.run(f"mkdir -p {json.dumps(Path(dest).parent.as_posix())}")
            toolbox.copy_to_container(str(file), dest)

    return run_solver(
        challenge_id=challenge_id,
        name=challenge.name,
        category=challenge.category,
        description=challenge.description,
        files=files,
        container_dir=container_dir,
        url=challenge.url,
        connector=None,
        toolbox=toolbox,
        run_id=run_id,
    )


def _preflight_llm() -> None:
    for role in ("planner", "executor"):
        tiers = load_tiers(role)
        for tier in tiers:
            provider = tier.provider
            model = tier.model
            if provider == "stub":
                continue
            if provider == "openai" and not settings.openai_api_key:
                raise SystemExit(f"Missing OPENAI_API_KEY for {role} provider=openai.")
            if provider == "anthropic" and not settings.anthropic_api_key:
                raise SystemExit(f"Missing ANTHROPIC_API_KEY for {role} provider=anthropic.")
            if provider == "gemini" and not settings.gemini_api_key:
                raise SystemExit(f"Missing GEMINI_API_KEY for {role} provider=gemini.")
            if provider == "ollama":
                _check_ollama_model(model, settings.ollama_base_url, role)


def _check_ollama_model(model: str, base_url: str, role: str) -> None:
    import requests

    if not base_url:
        raise SystemExit(f"Missing OLLAMA_BASE_URL for {role} provider=ollama.")

    url = base_url.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=5)
    except Exception as exc:
        raise SystemExit(
            f"Ollama not reachable at {base_url} for {role}. "
            f"Run 'ollama serve' and set OLLAMA_BASE_URL. ({exc})"
        )

    if resp.status_code != 200:
        raise SystemExit(
            f"Ollama returned HTTP {resp.status_code} for {url}. "
            "Check OLLAMA_BASE_URL or whether Ollama is running."
        )

    try:
        data = resp.json()
    except ValueError:
        raise SystemExit(f"Ollama returned non-JSON for {url}. Check OLLAMA_BASE_URL.")

    models = {m.get("name") or m.get("model") for m in data.get("models", [])}
    models.discard(None)
    if model not in models:
        example = ", ".join(sorted(models)[:10]) if models else "none"
        raise SystemExit(
            f"Ollama model '{model}' not found for {role}. "
            f"Run 'ollama pull {model}' or set --{role}-model. "
            f"Available (first 10): {example}"
        )


def _print_result(result: dict) -> None:
    flags = result.get("flag_hits") or []
    if flags:
        print(f"Verified flag: {flags[0].get('flag')}")
    else:
        print(f"Challenge {result.get('challenge_id')} done: flags={result.get('flag_candidates')}")


def _print_debug_info(result: dict, run_id: str) -> None:
    challenge_id = result.get("challenge_id", "unknown")
    run_dir = Path(settings.runs_dir) / str(challenge_id)
    log_dir = run_dir / "logs"
    latest_log = ""
    if log_dir.exists():
        logs = sorted(log_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            latest_log = str(logs[0])
    print("DEBUG:")
    print(f"  run_id: {run_id}")
    print(f"  run_dir: {run_dir}")
    print(f"  trajectory: {run_dir / 'trajectory.jsonl'}")
    print(f"  latest_log: {latest_log or 'none'}")


def _secret(value) -> str:
    if value is None:
        return ""
    try:
        return value.get_secret_value()
    except AttributeError:
        return str(value)


if __name__ == "__main__":
    main()
