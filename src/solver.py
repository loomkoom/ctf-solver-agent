from __future__ import annotations

import time
from pathlib import Path

from src.agent.react_agent import build_react_graph
from src.config import settings
from src.ctfd import CTFdConnector
from src.debug_log import debug_log
from src.state import DCipherState
from src.trajectory import TrajectoryLogger
from src.tools.toolbox import Toolbox


def build_challenge_context(
    name: str,
    category: str,
    description: str,
    container_dir: str,
    file_list: list[str],
    url: str = "",
) -> str:
    files = "\n".join(f"- {fname}" for fname in file_list) if file_list else "None"
    url_line = f"URL: {url}\n" if url else ""
    return (
        f"CHALLENGE: {name}\n"
        f"CATEGORY: {category}\n"
        f"DESCRIPTION:\n{description}\n\n"
        f"{url_line}"
        f"FILES (container path {container_dir}):\n{files}\n"
    )


def run_solver(
    challenge_id: str,
    name: str,
    category: str,
    description: str,
    files: list[Path],
    container_dir: str,
    url: str,
    connector: CTFdConnector | None,
    toolbox: Toolbox,
    run_id: str,
) -> dict:
    file_list = [f.name for f in files]
    context = build_challenge_context(name, category, description, container_dir, file_list, url=url)

    run_dir = Path(settings.runs_dir) / str(challenge_id)
    trajectory_logger = TrajectoryLogger(
        run_dir / "trajectory.jsonl",
        challenge_id=str(challenge_id),
        run_id=run_id,
        enabled=settings.debug,
    )
    graph = build_react_graph(toolbox, connector=connector, trajectory_logger=trajectory_logger)

    state: DCipherState = {
        "challenge_id": str(challenge_id),
        "challenge_name": name,
        "category": category,
        "description": description,
        "current_files": file_list,
        "attempt_history": [],
        "reasoning_log": [],
        "messages": [],
        "run_id": run_id,
        "challenge_context": context,
        "flag_format": settings.flag_format,
        "current_objective": "Initial reconnaissance",
        "plan": "",
        "candidate_categories": [],
        "selected_category": category,
        "selected_pipeline": "",
        "research_summary": "",
        "artifact_inventory": [],
        "triage_done": False,
        "url": url or "",
        "container_dir": container_dir,
        "last_command": "",
        "last_output": "",
        "last_error": "",
        "last_exit_code": 0,
        "last_log_path": "",
        "verifier_hint": "",
        "last_decode": {},
        "flag_candidates": [],
        "flag_hits": [],
        "command_history": [],
        "iteration": 0,
        "submitted_flags": [],
        "done": False,
        "tool_calls": 0,
        "phase_cycles": 0,
        "category_pivots": 0,
        "started_at": time.monotonic(),
        "run_dir": str(run_dir),
        "tool_manuals_seen": [],
    }

    debug_log(
        settings.debug,
        run_id,
        str(challenge_id),
        name,
        "graph_invoke_start",
        run_dir=str(run_dir),
        container_dir=container_dir,
    )
    start = time.monotonic()
    final_state = graph.invoke(state)
    duration = time.monotonic() - start
    debug_log(
        settings.debug,
        run_id,
        str(challenge_id),
        name,
        "graph_invoke_end",
        duration_s=round(duration, 3),
        done=final_state.get("done"),
        iteration=final_state.get("iteration"),
        tool_calls=final_state.get("tool_calls"),
        phase_cycles=final_state.get("phase_cycles"),
    )
    return {
        "challenge_id": challenge_id,
        "challenge_name": name,
        "category": category,
        "flag_candidates": final_state.get("flag_candidates"),
        "flag_hits": final_state.get("flag_hits"),
        "submitted_flags": final_state.get("submitted_flags"),
        "iteration": final_state.get("iteration"),
        "done": final_state.get("done"),
        "attempts": final_state.get("attempt_history"),
        "reasoning_log": final_state.get("reasoning_log"),
        "run_dir": str(run_dir),
        "tool_calls": final_state.get("tool_calls"),
        "phase_cycles": final_state.get("phase_cycles"),
        "category_pivots": final_state.get("category_pivots"),
    }
