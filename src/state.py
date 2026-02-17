from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from operator import add

class AgentState(TypedDict):
    # Use 'add' so that messages are appended rather than overwritten
    messages: Annotated[list[BaseMessage], add]
    challenge_context: str
    flag_format: str  # e.g. "flag{...}"
    current_objective: str  # The high-level task set by Tier 1
    subgoals_completed: list[str] # Track what we've already done
    remaining_bottlenecks: list[str] # Identify current technical hurdles
    logs: Annotated[list[str], add] # History of actions and outcomes