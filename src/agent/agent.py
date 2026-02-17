from src.config import settings
from src.state import AgentState
from src.tools.bash import run_bash_in_sandbox
from src.tools.artifacts import list_challenge_artifacts
from src.tools.artifacts import extract_path_from_text
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re

from src.tools.rag import search_knowledge


# Initialize your tiered models using your existing provider logic
def init_tiered_models():
    # Tier 1: Architect (Always high-level reasoning)
    architect_llm = ChatOpenAI(
        model=settings.planner_model,
        api_key=settings.openai_api_key.get_secret_value()
    )

    # Tier 2: Executor (Usually the "Primary" from your old code)
    executor_llm = ChatOpenAI(
        model=settings.executor_model,
        api_key=settings.openai_api_key.get_secret_value()
    )

    return architect_llm, executor_llm

architect_llm, executor_llm = init_tiered_models()

# --- Nodes ---
def architect_node(state: AgentState):
    """Tier 1: Strategic Planning with RAG Retrieval."""
    # Step 1: Retrieve context from RAG
    retrieved_info = search_knowledge(state['challenge_context'])

    system_msg = SystemMessage(content=(
        "You are the Strategic Architect. Analyze the challenge and the following write-ups: "
        f"\n\n{retrieved_info}\n\n"
        "If a write-up has a SOURCE PATH, tell the Executor: 'check artifacts for [PATH]'. "
        "Otherwise, set a technical objective for the next step. Do not write code."
    ))

    context_msg = HumanMessage(content=f"CORE CHALLENGE CONTEXT: {state['challenge_context']}")
    response = architect_llm.invoke([system_msg, context_msg] + state['messages'])

    return {
        "current_objective": response.content,
        "logs": [f"Architect objective: {response.content[:100]}..."]
    }

def executor_node(state: AgentState):
    """Tier 2: Tactical Execution with cleaner output formatting."""
    objective = state.get("current_objective", "")

    # Artifact Check Logic
    if "check artifacts" in objective.lower():
        path = extract_path_from_text(objective)
        files = list_challenge_artifacts(path)
        msg = f"Found artifacts: {files}"
        return {
            "messages": [AIMessage(content=msg)],
            "logs": [f"Executor: Inspected artifacts in {path}"]
        }

    # Bash Execution Logic
    prompt = f"OBJECTIVE: {objective}\nProvide one bash command in ```bash blocks."
    response = executor_llm.invoke(state['messages'] + [HumanMessage(content=prompt)])

    cmd_match = re.search(r"```bash\n(.*?)\n```", response.content, re.DOTALL)
    cmd = cmd_match.group(1).strip() if cmd_match else response.content.strip()

    result = run_bash_in_sandbox(cmd)

    # Format result as a clean string for AI readability
    readable_res = f"STDOUT: {result.get('stdout')}\nSTDERR: {result.get('stderr')}"

    return {
        "messages": [
            response,
            HumanMessage(content=f"COMMAND OUTPUT:\n{readable_res}\n\nIf this failed, fix it.")
        ],
        "logs": [f"Executor ran: {cmd[:50]}"]
    }

# --- Compile Workflow ---
builder = StateGraph(AgentState)
builder.add_node("architect", architect_node)
builder.add_node("executor", executor_node)

builder.add_edge(START, "architect")
builder.add_edge("architect", "executor")
# Add a loop back to architect for re-planning
builder.add_edge("executor", "architect")

graph = builder.compile()