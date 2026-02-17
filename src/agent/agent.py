from src.config import settings
from src.state import AgentState
from src.tools.bash import run_bash_in_sandbox
from src.tools.artifacts import list_challenge_artifacts
from src.tools.artifacts import extract_path_from_text
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re


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
    """Tier 1: Analyzes context and sets the objective."""
    system_msg = SystemMessage(content=(
        "You are the Strategic Architect. Your job is to analyze the CTF challenge and any retrieved write-ups"
        "If a relevant challenge write-upvhas a SOURCE PATH, always instruct the Executor to list artifacts "
        "for that specific path to find source code or binaries."
        "and set the single most important technical objective for the next step."
        "Do not write code; just define the goal."
        "Based on the context, what is the single next technical goal?"
    ))

    # We provide the specific challenge context as a HumanMessage
    # This ensures the 'target' is always fresh in its mind
    context_msg = HumanMessage(content=f"CORE CHALLENGE CONTEXT: {state['challenge_context']}")

    # We invoke the model with: [Instructions] + [The Challenge] + [The History]
    response = architect_llm.invoke([system_msg, context_msg] + state['messages'])

    return {
        "current_objective": response.content,
        "logs": [f"Architect: Strategic goal updated based on context -> {response.content[:100]}..."]
    }



def executor_node(state: AgentState):
    """Tier 2: Tactical Execution with Self-Healing logic."""
    objective = state.get("current_objective", "Initial Recon")

    # If the Architect said 'check artifacts for [path]', the Executor runs this:
    # (Pseudo-logic: your LLM prompt should handle the extraction)
    if "check artifacts" in objective.lower():
        path = extract_path_from_text(objective) # Implement simple regex or let LLM do it
        files = list_challenge_artifacts(path)
        return {"logs": [f"Executor found files: {files}"]}

    # 1. Focus the LLM on generating ONLY the command
    prompt = (
        f"OBJECTIVE: {objective}\n"
        "Provide a single bash command to advance this goal. "
        "Wrap the command in ```bash blocks."
    )
    response = executor_llm.invoke(state['messages'] + [HumanMessage(content=prompt)])

    # 2. Extract command using Regex (Cleaning Step)
    # This handles the LLM's tendency to add conversational fluff
    cmd_match = re.search(r"```bash\n(.*?)\n```", response.content, re.DOTALL)
    cmd = cmd_match.group(1).strip() if cmd_match else response.content.strip()

    # 3. Execution (The Action)
    result = run_bash_in_sandbox(cmd)

    # 4. Self-Healing Logic
    # If the command failed, we add a specific 'Nudge' to the log
    status = "Success" if "STDERR:" not in result or len(result.split("STDERR:")[1].strip()) == 0 else "Failed"

    log_entry = f"Executor ran: `{cmd}` | Status: {status}"

    # We return the response AND the result to the history
    return {
        "messages": [
            response,
            HumanMessage(content=f"COMMAND OUTPUT:\n{result}\n\nINSTRUCTION: If this failed, analyze why and suggest a fix.")
        ],
        "logs": [log_entry]
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