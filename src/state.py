from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from operator import add

class AgentState(TypedDict):
    # Use 'add' so that messages are appended rather than overwritten
    messages: Annotated[list[BaseMessage], add]
    challenge_context: str
    current_objective: str  # The high-level task set by Tier 1
    logs: Annotated[list[str], add] # History of actions and outcomes