from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.challenges import LocalChallenge, discover_challenges
from src.config import settings
from src.solver import run_solver
from src.tools.rag import ingest_knowledge_base
from src.tools.toolbox import Toolbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local bench runner for data/test_bench")
    parser.add_argument("--bench", default="data/test_bench", help="Bench root folder")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of challenges")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG indexing")
    parser.add_argument(
        "--rag-mode",
        default="no_writeups",
        choices=["all", "no_writeups", "methodology"],
        help="RAG ingestion mode (default: no_writeups)",
    )
    parser.add_argument("--flag-format", default=settings.flag_format)
    parser.add_argument("--planner-provider", default=settings.planner_provider)
    parser.add_argument("--planner-model", default=settings.planner_model)
    parser.add_argument("--executor-provider", default=settings.executor_provider)
    parser.add_argument("--executor-model", default=settings.executor_model)
    parser.add_argument("--max-iterations", type=int, default=settings.max_iterations)
    parser.add_argument("--output", default="", help="Output JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.max_iterations = args.max_iterations
    settings.max_phase_cycles = args.max_iterations
    settings.max_phase_cycles = args.max_iterations
    settings.flag_format = args.flag_format
    settings.rag_mode = args.rag_mode
    settings.rag_include_writeups = args.rag_mode == "all"
    settings.rag_enabled = not args.skip_rag
    settings.planner_provider = args.planner_provider
    settings.planner_model = args.planner_model
    settings.executor_provider = args.executor_provider
    settings.executor_model = args.executor_model

    bench_root = Path(args.bench)
    if not bench_root.exists():
        raise SystemExit(f"Bench root not found: {bench_root}")

    if not args.skip_rag:
        ingest_knowledge_base(str(settings.knowledge_base_path), mode=args.rag_mode)

    toolbox = Toolbox()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    challenges = discover_challenges(bench_root, limit=args.limit)
    if not challenges:
        raise SystemExit(f"No challenges found under {bench_root}")

    start = time.time()
    results = []
    for challenge in challenges:
        container_dir = _stage_challenge(toolbox, challenge, run_id)
        result = run_solver(
            challenge_id=challenge.challenge_id,
            name=challenge.name,
            category=challenge.category,
            description=challenge.description,
            files=challenge.files,
            container_dir=container_dir,
            url=challenge.url,
            connector=None,
            toolbox=toolbox,
            run_id=run_id,
        )
        expected_flag = challenge.flag
        found_flags = result.get("flag_candidates") or []
        success = expected_flag in found_flags if expected_flag else bool(found_flags)
        results.append(
            {
                "challenge": _challenge_to_dict(challenge),
                "expected_flag": expected_flag,
                "success": success,
                "tool_calls": result.get("tool_calls") or len(result.get("attempts") or []),
                "iterations": result.get("iteration"),
                "result": result,
            }
        )

    elapsed = time.time() - start
    report = _build_report(results, elapsed)
    out_path = Path(args.output or f"runs/bench_{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_summary(report, out_path)


def _stage_challenge(toolbox: Toolbox, challenge: LocalChallenge, run_id: str) -> str:
    container_base = f"{settings.sandbox_workdir}/bench/{run_id}"
    toolbox.run(f"mkdir -p {json.dumps(container_base)}")
    container_dir = f"{container_base}/{challenge.challenge_id}"
    if challenge.files:
        toolbox.run(f"mkdir -p {json.dumps(container_dir)}")
        for file in challenge.files:
            rel = file.relative_to(challenge.root)
            dest = f"{container_dir}/{rel.as_posix()}"
            toolbox.run(f"mkdir -p {json.dumps(Path(dest).parent.as_posix())}")
            toolbox.copy_to_container(str(file), dest)
    return container_dir


def _build_report(results: list[dict], elapsed: float) -> dict:
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    tool_calls = [r.get("tool_calls", 0) for r in results]
    iterations = [r.get("iterations", 0) for r in results]
    avg_tool_calls = sum(tool_calls) / total if total else 0
    avg_iterations = sum(iterations) / total if total else 0

    return {
        "summary": {
            "total": total,
            "successes": successes,
            "success_rate": (successes / total) if total else 0,
            "avg_tool_calls": avg_tool_calls,
            "avg_iterations": avg_iterations,
            "total_time_s": elapsed,
        },
        "results": results,
    }


def _challenge_to_dict(challenge: LocalChallenge) -> dict:
    data = asdict(challenge)
    data["root"] = str(challenge.root)
    data["files"] = [str(p) for p in challenge.files]
    return data


def _print_summary(report: dict, out_path: Path) -> None:
    summary = report.get("summary", {})
    print(f"Bench report: {out_path}")
    print(
        "Summary: total={total} success_rate={rate:.2f} avg_tool_calls={calls:.2f} avg_iterations={iters:.2f} time_s={time:.1f}".format(
            total=summary.get("total", 0),
            rate=summary.get("success_rate", 0.0),
            calls=summary.get("avg_tool_calls", 0.0),
            iters=summary.get("avg_iterations", 0.0),
            time=summary.get("total_time_s", 0.0),
        )
    )


if __name__ == "__main__":
    main()
