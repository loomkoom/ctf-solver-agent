from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from operator import add


class AttemptRecord(TypedDict):
    step: int
    phase: str
    objective: str
    tool: str
    command: str
    stdout: str
    stderr: str
    exit_code: int
    log_path: str


class FlagHit(TypedDict):
    flag: str
    step: int
    tool: str
    command: str
    evidence: str
    evidence_type: str
    log_path: str


class CommandRecord(TypedDict):
    cmd: str
    stdout: str
    stderr: str
    exit_code: int
    tool: str
    log_path: str


class CTFState(TypedDict):
    challenge_id: str
    challenge_name: str
    category: str
    description: str
    current_files: list[str]
    attempt_history: Annotated[list[AttemptRecord], add]
    reasoning_log: Annotated[list[str], add]


class DCipherState(CTFState):
    messages: Annotated[list[BaseMessage], add]
    run_id: str
    challenge_context: str
    flag_format: str
    current_objective: str
    plan: str
    candidate_categories: list[str]
    selected_category: str
    selected_pipeline: str
    research_summary: str
    artifact_inventory: list[dict]
    triage_done: bool
    url: str
    container_dir: str
    last_command: str
    last_output: str
    last_error: str
    last_exit_code: int
    last_log_path: str
    verifier_hint: str
    last_decode: dict
    flag_candidates: list[str]
    flag_hits: Annotated[list[FlagHit], add]
    command_history: Annotated[list[dict], add]
    iteration: int
    submitted_flags: list[str]
    done: bool
    tool_calls: int
    phase_cycles: int
    category_pivots: int
    started_at: float
    run_dir: str
    tool_manuals_seen: list[str]
