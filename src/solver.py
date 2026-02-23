from __future__ import annotations

import time
from pathlib import Path

from src.agent import build_graph
from src.config import settings
from src.ctfd import CTFdConnector
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
    graph = build_graph(toolbox, connector=connector, trajectory_logger=trajectory_logger)

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
        "candidate_categories": [],
        "selected_category": category,
        "selected_pipeline": "",
        "research_summary": "",
        "artifact_inventory": [],
        "url": url or "",
        "container_dir": container_dir,
        "last_command": "",
        "last_output": "",
        "last_error": "",
        "last_exit_code": 0,
        "last_log_path": "",
        "flag_candidates": [],
        "flag_hits": [],
        "iteration": 0,
        "submitted_flags": [],
        "done": False,
        "tool_calls": 0,
        "phase_cycles": 0,
        "category_pivots": 0,
        "started_at": time.monotonic(),
        "run_dir": str(run_dir),
    }

    final_state = graph.invoke(state)
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
