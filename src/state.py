from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from operator import add


class AttemptRecord(TypedDict):
    step: int
    objective: str
    tool: str
    command: str
    stdout: str
    stderr: str
    exit_code: int


class FlagHit(TypedDict):
    flag: str
    step: int
    tool: str
    command: str


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
    challenge_context: str
    flag_format: str
    current_objective: str
    plan: str
    last_command: str
    last_output: str
    last_error: str
    last_exit_code: int
    flag_candidates: list[str]
    flag_hits: Annotated[list[FlagHit], add]
    iteration: int
    submitted_flags: list[str]
    done: bool
