import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.state import DCipherState
from src.tools.artifacts import extract_path_from_text, list_challenge_artifacts
from src.tools.rag import search_knowledge
from src.tools.toolbox import ToolResult, Toolbox

MAX_TOOL_OUTPUT_CHARS = 4000
MAX_RAG_CHARS = 3500
DEFAULT_FLAG_REGEX = r"(?:IGCTF|flag|CSCBE|UCTF|ctf)\{.*?\}"


def _init_llm(provider: str, model: str):
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for provider=openai.")
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key.get_secret_value()
        )
    if provider == "ollama":
        return ChatOllama(model=model, base_url=settings.ollama_base_url)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for provider=anthropic.")
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key.get_secret_value()
        )

    raise ValueError(f"Unsupported provider: {provider}")


def build_graph(toolbox: Toolbox, connector=None):
    planner_llm = _init_llm(settings.planner_provider, settings.planner_model)
    executor_llm = _init_llm(settings.executor_provider, settings.executor_model)

    tool_registry = _tool_registry(toolbox)

    def plan_node(state: DCipherState):
        retrieved_info = search_knowledge(state["challenge_context"])
        retrieved_info = _truncate_text(retrieved_info, MAX_RAG_CHARS)
        attempts = _summarize_attempts(state.get("attempt_history", [])[-3:])

        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Planner. Produce a concise plan and one next objective. "
            "Use retrieved checklists/writeups when available. "
            "If errors occurred, adapt the plan based on stderr. "
            "Reply with:\nPLAN:\n- ...\nOBJECTIVE:\n..."
        ))
        human_msg = HumanMessage(content=(
            f"CHALLENGE:\n{state['challenge_context']}\n\n"
            f"RAG:\n{retrieved_info}\n\n"
            f"RECENT ATTEMPTS:\n{attempts}"
        ))

        response = planner_llm.invoke([system_msg, human_msg])
        plan, objective = _parse_plan_objective(response.content)

        return {
            "plan": plan,
            "current_objective": objective,
            "messages": [response],
            "reasoning_log": [f"Plan set: {objective}"]
        }

    def execute_node(state: DCipherState):
        objective = state.get("current_objective") or "Initial reconnaissance"

        if "check artifacts" in objective.lower():
            path = extract_path_from_text(objective)
            artifact_msg = list_challenge_artifacts(path)
            tool_result = ToolResult(
                tool="artifacts",
                command=path or "",
                stdout=artifact_msg,
                stderr="",
                exit_code=0
            )
            ai_msg = AIMessage(content=f"Artifacts lookup: {artifact_msg}")
            return _record_attempt(state, tool_result, [ai_msg])

        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Executor. Pick exactly one tool call. "
            "Respond with JSON: {\"tool\": \"name\", \"args\": {..}} "
            "Use tool \"bash\" for raw shell commands. "
            f"Tools: {', '.join(tool_registry.keys())}"
        ))
        human_msg = HumanMessage(content=f"OBJECTIVE:\n{objective}")

        response = executor_llm.invoke([system_msg, human_msg])
        tool_name, args, raw_cmd = _parse_tool_call(response.content)

        if tool_name == "bash" and "command" not in args:
            fallback_cmd = raw_cmd or response.content.strip()
            args = {"command": fallback_cmd}

        if tool_name in tool_registry:
            tool_result = tool_registry[tool_name](**args)
        else:
            tool_result = toolbox.run(raw_cmd)

        output_msg = HumanMessage(
            content=_format_tool_output(tool_result, max_chars=MAX_TOOL_OUTPUT_CHARS)
        )
        return _record_attempt(state, tool_result, [response, output_msg])

    def observe_node(state: DCipherState):
        output = state.get("last_output", "")
        error = state.get("last_error", "")
        flags = _extract_flags(output + "\n" + error, state.get("flag_format", settings.flag_format))
        merged_flags = _merge_flags(state.get("flag_candidates", []), flags)
        new_hits = _new_flag_hits(state.get("flag_hits", []), flags, state)

        observation = _summarize_observation(state)
        return {
            "flag_candidates": merged_flags,
            "flag_hits": new_hits,
            "reasoning_log": [observation]
        }

    def reflect_node(state: DCipherState):
        updates = {}
        if connector:
            hit_flags = {hit["flag"] for hit in state.get("flag_hits", [])}
            new_flags = [
                flag for flag in state.get("flag_candidates", [])
                if flag not in state.get("submitted_flags", []) and flag in hit_flags
            ]
            for flag in new_flags:
                if not _is_flag_like(flag, state.get("flag_format", settings.flag_format)):
                    continue
                resp = connector.submit_flag(int(state["challenge_id"]), flag)
                status = _submission_status(resp)
                updates.setdefault("submitted_flags", state.get("submitted_flags", []) + [flag])
                updates.setdefault("reasoning_log", []).append(
                    f"Submission {flag}: {status}"
                )
                if status == "correct":
                    updates["done"] = True
                    return updates

        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Reflector. Analyze the last command output. "
            "If it failed, explain why using stderr and propose the next fix. "
            "Be concise and avoid assuming success without evidence."
        ))
        stdout = _truncate_text(state.get("last_output", ""), MAX_TOOL_OUTPUT_CHARS)
        stderr = _truncate_text(state.get("last_error", ""), MAX_TOOL_OUTPUT_CHARS)
        human_msg = HumanMessage(content=(
            f"OBJECTIVE: {state.get('current_objective')}\n"
            f"LAST COMMAND: {state.get('last_command')}\n"
            f"EXIT CODE: {state.get('last_exit_code')}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}\n"
            f"FLAGS: {state.get('flag_candidates')}\n"
        ))

        response = planner_llm.invoke([system_msg, human_msg])
        updates.setdefault("messages", []).append(response)
        updates.setdefault("reasoning_log", []).append(f"Reflection: {response.content[:200]}")
        return updates

    def should_continue(state: DCipherState) -> Literal["plan", END]:
        if state.get("done"):
            return END
        if state.get("iteration", 0) >= settings.max_iterations:
            return END
        if state.get("flag_candidates") and not connector:
            return END
        return "plan"

    builder = StateGraph(DCipherState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("observe", observe_node)
    builder.add_node("reflect", reflect_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "observe")
    builder.add_edge("observe", "reflect")
    builder.add_conditional_edges("reflect", should_continue)

    return builder.compile()


def _tool_registry(toolbox: Toolbox) -> dict:
    return {
        "bash": lambda command: toolbox.run(command),
        "read_file": toolbox.read_file,
        "write_file": toolbox.write_file,
        "list_dir": toolbox.list_dir,
        "grep": toolbox.grep,
        "find": toolbox.find,
        "binwalk": toolbox.binwalk,
        "checksec": toolbox.checksec,
        "strings": toolbox.strings,
        "exiftool": toolbox.exiftool,
        "ghidra_headless": toolbox.ghidra_headless,
        "gdb_pwndbg": toolbox.gdb_pwndbg,
        "pwntools": toolbox.pwntools,
        "python": toolbox.python,
        "curl": toolbox.curl,
        "nmap": toolbox.nmap,
        "sqlmap": toolbox.sqlmap,
    }


def _parse_tool_call(text: str) -> tuple[str, dict, str]:
    json_block = _extract_json_block(text)
    if json_block:
        try:
            data = json.loads(json_block)
            tool = data.get("tool", "").strip()
            args = data.get("args") or {}
            if tool == "bash":
                if "command" in args:
                    return tool, args, args.get("command", "")
                if "command" in data:
                    args = {"command": data["command"]}
                    return tool, args, data.get("command", "")
            return tool, args, data.get("command", "")
        except json.JSONDecodeError:
            pass

    cmd = _extract_bash_command(text)
    if cmd:
        return "bash", {"command": cmd}, cmd
    return "bash", {"command": text.strip()}, text.strip()


def _extract_json_block(text: str) -> str | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    return match.group(1) if match else None


def _extract_bash_command(text: str) -> str | None:
    match = re.search(r"```bash\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _record_attempt(state: DCipherState, tool_result: ToolResult, messages: list):
    attempt = {
        "step": state.get("iteration", 0) + 1,
        "objective": state.get("current_objective", ""),
        "tool": tool_result.tool,
        "command": tool_result.command,
        "stdout": tool_result.stdout,
        "stderr": tool_result.stderr,
        "exit_code": tool_result.exit_code,
    }

    updates = {
        "attempt_history": [attempt],
        "messages": messages,
        "last_command": tool_result.command,
        "last_output": tool_result.stdout,
        "last_error": tool_result.stderr,
        "last_exit_code": tool_result.exit_code,
        "iteration": state.get("iteration", 0) + 1,
    }
    if tool_result.parsed and "files" in tool_result.parsed:
        updates["current_files"] = tool_result.parsed["files"]
    return updates


def _format_tool_output(result: ToolResult) -> str:
    return (
        f"TOOL: {result.tool}\n"
        f"COMMAND: {result.command}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
        f"EXIT CODE: {result.exit_code}"
    )


def _parse_plan_objective(text: str) -> tuple[str, str]:
    plan_match = re.search(r"PLAN:\s*(.*?)(?:OBJECTIVE:|$)", text, re.DOTALL)
    obj_match = re.search(r"OBJECTIVE:\s*(.*)", text, re.DOTALL)
    plan = plan_match.group(1).strip() if plan_match else text.strip()
    objective = obj_match.group(1).strip() if obj_match else plan.splitlines()[0]
    return plan, objective


def _summarize_attempts(attempts: list[dict]) -> str:
    if not attempts:
        return "None"
    lines = []
    for attempt in attempts:
        lines.append(
            f"{attempt.get('step')}: {attempt.get('tool')} {attempt.get('command')} "
            f"(exit {attempt.get('exit_code')})"
        )
    return "\n".join(lines)


def _summarize_observation(state: DCipherState) -> str:
    if not state.get("attempt_history"):
        return "Observation: no attempts yet."
    last = state["attempt_history"][-1]
    return (
        "Observation: "
        f"exit={last.get('exit_code')} "
        f"stderr={'yes' if last.get('stderr') else 'no'} "
        f"stdout_len={len(last.get('stdout', ''))}"
    )


def _extract_flags(text: str, flag_format: str) -> list[str]:
    patterns = [
        re.escape(flag_format.rstrip("{")) + r"\{[^}]{3,}\}",
        r"[A-Za-z0-9]{2,}\{[^}]{3,}\}",
    ]
    matches = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            if match not in matches:
                matches.append(match)
    return matches


def _merge_flags(existing: list[str], new_flags: list[str]) -> list[str]:
    merged = list(existing)
    for flag in new_flags:
        if flag not in merged:
            merged.append(flag)
    return merged


def _is_flag_like(flag: str, flag_format: str) -> bool:
    return re.search(re.escape(flag_format.rstrip("{")) + r"\{[^}]+\}", flag, re.IGNORECASE) is not None


def _submission_status(resp: dict) -> str:
    if not isinstance(resp, dict):
        return "unknown"
    if resp.get("success") is False:
        return "rejected"
    data = resp.get("data") or {}
    status = data.get("status") or data.get("message") or "unknown"
    if isinstance(status, str) and "correct" in status.lower():
        return "correct"
    if isinstance(status, str) and "incorrect" in status.lower():
        return "incorrect"
    return status
