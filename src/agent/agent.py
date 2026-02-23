import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.llm_tiers import load_tiers, select_tier_index
from src.state import DCipherState
from src.tools.artifacts import extract_path_from_text, list_challenge_artifacts
from src.tools.rag import search_knowledge
from src.tools.toolbox import ToolResult, Toolbox
from src.trajectory import TrajectoryLogger

MAX_RAG_CHARS = 3500
DEFAULT_FLAG_REGEX = r"\b[A-Za-z0-9_\-]{0,24}\{[^\n\r]{3,200}\}\b"
CTF_PREFIX_FLAG_REGEX = r"(?:IGCTF|flag|CSCBE|UCTF|ctf)\{.*?\}"


def _init_llm(provider: str, model: str, role: str, max_tokens: int | None = None):
    if provider == "stub":
        return StubLLM(role)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for provider=openai.")
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key.get_secret_value(),
            max_tokens=max_tokens,
        )
    if provider == "ollama":
        kwargs = {}
        if max_tokens:
            kwargs["num_predict"] = max_tokens
        return ChatOllama(model=model, base_url=settings.ollama_base_url, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for provider=anthropic.")
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_tokens=max_tokens or 1024,
        )

    raise ValueError(f"Unsupported provider: {provider}")


class StubLLM:
    def __init__(self, role: str):
        self.role = role

    def invoke(self, messages):
        if self.role == "planner":
            content = (
                "CATEGORIES:\nforensics\n"
                "PIPELINE:\ntriage\n"
                "PLAN:\n- list files\n"
                "OBJECTIVE:\nList files in the container directory."
            )
        else:
            content = json.dumps({"tool": "bash", "args": {"command": "ls -la"}})
        return AIMessage(content=content)


def build_graph(toolbox: Toolbox, connector=None, trajectory_logger: TrajectoryLogger | None = None):
    planner_tiers = load_tiers("planner")
    executor_tiers = load_tiers("executor")
    llm_cache: dict[tuple[str, str, str, int | None], object] = {}

    def _get_llm(role: str, state: DCipherState):
        tiers = planner_tiers if role == "planner" else executor_tiers
        idx = select_tier_index(state, tiers, role)
        tier = tiers[idx]
        key = (role, tier.provider, tier.model, tier.max_tokens)
        llm = llm_cache.get(key)
        if llm is None:
            llm = _init_llm(tier.provider, tier.model, role=role, max_tokens=tier.max_tokens)
            llm_cache[key] = llm
        return llm, tier, idx

    tool_registry = _tool_registry(toolbox)
    web_tools = {"curl", "http_request", "ffuf"}

    def plan_node(state: DCipherState):
        budget_reason = _budget_exceeded(state)
        if budget_reason:
            return _mark_done(state, f"Budget exceeded: {budget_reason}", trajectory_logger)

        retrieved_info = search_knowledge(state.get("challenge_context", ""))
        retrieved_info = _truncate_text(retrieved_info, MAX_RAG_CHARS)
        attempts = _summarize_attempts(state.get("attempt_history", [])[-3:])
        research_summary = state.get("research_summary", "")

        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Planner (Architect). "
            "Pick at most 2 candidate categories and exactly one pipeline to try next. "
            "Produce a concise plan and one next objective. "
            "Use retrieved checklists/writeups when available. "
            "If errors occurred, adapt the plan based on stderr. "
            "Reply with:\nCATEGORIES:\nPIPELINE:\nPLAN:\n- ...\nOBJECTIVE:\n..."
        ))
        human_msg = HumanMessage(content=(
            f"CHALLENGE:\n{state['challenge_context']}\n\n"
            f"RESEARCH:\n{research_summary or 'none'}\n\n"
            f"RAG:\n{retrieved_info}\n\n"
            f"RECENT ATTEMPTS:\n{attempts}"
        ))

        planner_llm, tier, tier_idx = _get_llm("planner", state)
        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Planner (Architect). "
            "Pick at most 2 candidate categories and exactly one pipeline to try next. "
            "Produce a concise plan and one next objective. "
            "Use retrieved checklists/writeups when available. "
            "If errors occurred, adapt the plan based on stderr. "
            f"Keep the response under {tier.max_tokens or settings.planner_max_tokens} tokens. "
            "Reply with:\nCATEGORIES:\nPIPELINE:\nPLAN:\n- ...\nOBJECTIVE:\n..."
        ))
        verifier_llm, tier, tier_idx = _get_llm("planner", state)
        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Verifier. Analyze the last command output. "
            "If it failed, explain why using stderr and propose the next fix. "
            "Be concise and avoid assuming success without evidence. "
            f"Keep the response under {tier.max_tokens or settings.planner_max_tokens} tokens."
        ))
        response = verifier_llm.invoke([system_msg, human_msg])
        plan, objective, categories, pipeline = _parse_plan_objective(response.content)
        categories = [c for c in categories if c] or [state.get("category", "unknown")]
        categories = categories[:2]
        selected_category = categories[0] if categories else state.get("category", "unknown")

        category_pivots = state.get("category_pivots", 0)
        prev_category = state.get("selected_category")
        if prev_category and selected_category and prev_category != selected_category:
            category_pivots += 1

        phase_cycles = state.get("phase_cycles", 0) + 1
        updates = {
            "plan": plan,
            "current_objective": objective,
            "candidate_categories": categories,
            "selected_category": selected_category,
            "selected_pipeline": pipeline,
            "phase_cycles": phase_cycles,
            "category_pivots": category_pivots,
            "messages": [response],
            "reasoning_log": [f"Architect: {objective}"]
        }
        if trajectory_logger:
            trajectory_logger.log(
                "plan",
                {
                    "objective": objective,
                    "categories": categories,
                    "pipeline": pipeline,
                    "phase_cycles": phase_cycles,
                },
            )
        return updates

    def research_node(state: DCipherState):
        if state.get("done"):
            return {}
        budget_reason = _budget_exceeded(state)
        if budget_reason:
            return _mark_done(state, f"Budget exceeded: {budget_reason}", trajectory_logger)

        container_dir = state.get("container_dir") or _infer_container_dir(state.get("challenge_context", ""))
        files = state.get("current_files", []) or []
        if not container_dir or not files:
            summary = "No files provided." if not files else "No container directory found."
            if trajectory_logger:
                trajectory_logger.log("research", {"summary": summary})
            return {
                "research_summary": summary,
                "artifact_inventory": [],
                "reasoning_log": [f"Research: {summary}"],
            }

        cmd = _build_research_command(container_dir)
        tool_result = toolbox.run(cmd)
        log_path = _persist_tool_output(state, tool_result, phase="research")
        tool_result.log_path = log_path
        summary, inventory = _summarize_file_inventory(tool_result.stdout)

        output_msg = HumanMessage(
            content=_format_tool_output(tool_result, log_path)
        )
        updates = _record_attempt(state, tool_result, [output_msg], phase="research", log_path=log_path)
        updates["research_summary"] = summary
        updates["artifact_inventory"] = inventory
        updates.setdefault("reasoning_log", []).append(f"Research: {summary}")
        if trajectory_logger:
            trajectory_logger.log("research", {"summary": summary, "log_path": log_path})
        return updates

    def execute_node(state: DCipherState):
        if state.get("done"):
            return {}
        budget_reason = _budget_exceeded(state)
        if budget_reason:
            return _mark_done(state, f"Budget exceeded: {budget_reason}", trajectory_logger)

        objective = state.get("current_objective") or "Initial reconnaissance"
        url = state.get("url") or ""
        research_summary = state.get("research_summary", "")
        inventory = state.get("artifact_inventory", [])

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
            log_path = _persist_tool_output(state, tool_result, phase="execute")
            tool_result.log_path = log_path
            ai_msg = AIMessage(content=f"Artifacts lookup: {artifact_msg}")
            return _record_attempt(state, tool_result, [ai_msg], phase="execute", log_path=log_path)

        executor_llm, tier, tier_idx = _get_llm("executor", state)
        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Executor. Pick exactly one tool call. "
            "Respond with JSON: {\"tool\": \"name\", \"args\": {..}} "
            "Use tool \"bash\" for raw shell commands. "
            "Only use web tools if an explicit URL is provided. "
            f"Keep the response under {tier.max_tokens or settings.executor_max_tokens} tokens. "
            f"Tools: {', '.join(tool_registry.keys())}"
        ))
        human_msg = HumanMessage(content=(
            f"OBJECTIVE:\n{objective}\n\n"
            f"URL:\n{url or 'none'}\n\n"
            f"RESEARCH SUMMARY:\n{research_summary or 'none'}\n\n"
            f"ARTIFACT INVENTORY:\n{inventory or 'none'}"
        ))

        response = executor_llm.invoke([system_msg, human_msg])
        tool_name, args, raw_cmd = _parse_tool_call(response.content)

        if tool_name == "bash" and "command" not in args:
            fallback_cmd = raw_cmd or response.content.strip()
            args = {"command": fallback_cmd}

        args = _normalize_tool_args(tool_name, args)

        if tool_name in web_tools and not url:
            tool_result = ToolResult(
                tool=tool_name,
                command=json.dumps(args),
                stdout="",
                stderr="Web tools require an explicit URL in the challenge context.",
                exit_code=2,
            )
        elif tool_name in web_tools and url and _violates_url_scope(json.dumps(args), url):
            tool_result = ToolResult(
                tool=tool_name,
                command=json.dumps(args),
                stdout="",
                stderr="Tool targets a URL outside the provided endpoint scope.",
                exit_code=2,
            )
        elif tool_name == "bash" and not url and _contains_network_command(args.get("command", "")):
            tool_result = ToolResult(
                tool="bash",
                command=args.get("command", ""),
                stdout="",
                stderr="Network access requires an explicit URL in the challenge context.",
                exit_code=2,
            )
        elif tool_name == "bash" and url and _violates_url_scope(args.get("command", ""), url):
            tool_result = ToolResult(
                tool="bash",
                command=args.get("command", ""),
                stdout="",
                stderr="Command targets a URL outside the provided endpoint scope.",
                exit_code=2,
            )
        elif tool_name == "bash" and _is_disallowed_command(args.get("command", "")):
            tool_result = ToolResult(
                tool="bash",
                command=args.get("command", ""),
                stdout="",
                stderr="Disallowed command (port scanning or prohibited tooling).",
                exit_code=2,
            )
        elif tool_name in tool_registry:
            missing = _missing_required_args(tool_name, args)
            if missing:
                tool_result = ToolResult(
                    tool=tool_name,
                    command=json.dumps(args),
                    stdout="",
                    stderr=f"Missing required args for {tool_name}: {', '.join(missing)}",
                    exit_code=2,
                )
            else:
                try:
                    tool_result = tool_registry[tool_name](**args)
                except TypeError as exc:
                    tool_result = ToolResult(
                        tool=tool_name,
                        command=json.dumps(args),
                        stdout="",
                        stderr=f"Tool argument error: {exc}",
                        exit_code=2,
                    )
        else:
            tool_result = ToolResult(
                tool=tool_name or "unknown",
                command=raw_cmd,
                stdout="",
                stderr="Unknown tool requested.",
                exit_code=127,
            )

        log_path = _persist_tool_output(state, tool_result, phase="execute")
        tool_result.log_path = log_path
        output_msg = HumanMessage(content=_format_tool_output(tool_result, log_path))
        updates = _record_attempt(state, tool_result, [response, output_msg], phase="execute", log_path=log_path)
        if trajectory_logger:
            trajectory_logger.log("execute", {"tool": tool_result.tool, "command": tool_result.command, "log_path": log_path})
        return updates

    def verify_node(state: DCipherState):
        if state.get("done"):
            return {}
        output = state.get("last_output", "")
        error = state.get("last_error", "")
        evidence_hits = _extract_flags_with_evidence(
            output,
            error,
            state.get("flag_format", settings.flag_format),
            state.get("last_log_path", ""),
        )
        flags = [hit["flag"] for hit in evidence_hits]
        merged_flags = _merge_flags(state.get("flag_candidates", []), flags)
        new_hits = _new_flag_hits(state.get("flag_hits", []), evidence_hits, state)

        observation = _summarize_observation(state)
        updates = {
            "flag_candidates": merged_flags,
            "flag_hits": new_hits,
            "reasoning_log": [observation]
        }

        if connector:
            all_hits = list(state.get("flag_hits", [])) + new_hits
            hit_flags = {hit["flag"] for hit in all_hits}
            new_flags = [
                flag for flag in merged_flags
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
                    if trajectory_logger:
                        trajectory_logger.log("verify", {"status": status, "flag": flag})
                    return updates

        verifier_llm, tier, tier_idx = _get_llm("planner", state)
        system_msg = SystemMessage(content=(
            "You are the D-CIPHER Verifier. Analyze the last command output. "
            "If it failed, explain why using stderr and propose the next fix. "
            "Be concise and avoid assuming success without evidence. "
            f"Keep the response under {tier.max_tokens or settings.planner_max_tokens} tokens."
        ))
        stdout = _truncate_text(state.get("last_output", ""), settings.output_large_char_threshold)
        stderr = _truncate_text(state.get("last_error", ""), settings.output_large_char_threshold)
        auto_hint = _auto_failure_hint(state)
        human_msg = HumanMessage(content=(
            f"OBJECTIVE: {state.get('current_objective')}\n"
            f"LAST COMMAND: {state.get('last_command')}\n"
            f"EXIT CODE: {state.get('last_exit_code')}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}\n"
            f"AUTO_HINT: {auto_hint or 'none'}\n"
            f"FLAGS: {state.get('flag_candidates')}\n"
        ))

        response = verifier_llm.invoke([system_msg, human_msg])
        updates.setdefault("messages", []).append(response)
        updates.setdefault("reasoning_log", []).append(f"Verifier: {response.content[:200]}")
        if trajectory_logger:
            trajectory_logger.log("verify", {"reflection": response.content[:200]})
        return updates

    def should_continue(state: DCipherState) -> Literal["plan", END]:
        if state.get("done"):
            return END
        if _budget_exceeded(state):
            return END
        if state.get("flag_candidates") and not connector:
            return END
        return "plan"

    builder = StateGraph(DCipherState)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("execute", execute_node)
    builder.add_node("verify", verify_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "execute")
    builder.add_edge("execute", "verify")
    builder.add_conditional_edges("verify", should_continue)

    return builder.compile()


def _tool_registry(toolbox: Toolbox) -> dict:
    return {
        "bash": lambda command: toolbox.run(command),
        "read_file": toolbox.read_file,
        "write_file": toolbox.write_file,
        "list_dir": toolbox.list_dir,
        "grep": toolbox.grep,
        "find": toolbox.find,
        "hash_file": toolbox.hash_file,
        "extract_archive": toolbox.extract_archive,
        "file_info": toolbox.file_info,
        "binwalk": toolbox.binwalk,
        "checksec": toolbox.checksec,
        "strings": toolbox.strings,
        "exiftool": toolbox.exiftool,
        "stegseek": toolbox.stegseek,
        "zsteg": toolbox.zsteg,
        "ciphey": toolbox.ciphey,
        "lemmeknow": toolbox.lemmeknow,
        "ghidra_headless": toolbox.ghidra_headless,
        "radare2_json": toolbox.radare2_json,
        "objdump": toolbox.objdump,
        "readelf": toolbox.readelf,
        "gdb_pwndbg": toolbox.gdb_pwndbg,
        "pwntools": toolbox.pwntools,
        "pwntools_template": toolbox.pwntools_template,
        "pwntools_ret2win": toolbox.pwntools_ret2win,
        "pwntools_fmt_leak": toolbox.pwntools_fmt_leak,
        "pwntools_rop_system": toolbox.pwntools_rop_system,
        "bash_template": toolbox.bash_template,
        "python_template": toolbox.python_template,
        "python": toolbox.python,
        "base_decode": toolbox.base_decode,
        "curl": toolbox.curl,
        "http_request": toolbox.http_request,
        "ffuf": toolbox.ffuf,
    }


def _missing_required_args(tool_name: str, args: dict) -> list[str]:
    required = {
        "bash": ["command"],
        "read_file": ["path"],
        "write_file": ["path", "content"],
        "grep": ["pattern"],
        "find": ["path", "name"],
        "hash_file": ["path"],
        "extract_archive": ["path"],
        "file_info": ["path"],
        "binwalk": ["path"],
        "checksec": ["path"],
        "strings": ["path"],
        "exiftool": ["path"],
        "stegseek": ["path"],
        "zsteg": ["path"],
        "ciphey": ["text_or_path"],
        "ghidra_headless": ["project_dir", "project_name", "binary_path"],
        "radare2_json": ["binary_path", "commands"],
        "objdump": ["binary_path"],
        "readelf": ["binary_path"],
        "gdb_pwndbg": ["binary_path"],
        "pwntools": ["script"],
        "pwntools_template": ["binary_path"],
        "pwntools_ret2win": ["binary_path"],
        "pwntools_fmt_leak": ["binary_path"],
        "pwntools_rop_system": ["binary_path"],
        "python": ["script"],
        "base_decode": ["value"],
        "curl": ["url"],
        "http_request": ["url"],
        "ffuf": ["url", "wordlist"],
        "bash_template": ["template_name"],
        "python_template": ["template_name"],
    }
    needed = required.get(tool_name, [])
    return [key for key in needed if key not in args or args[key] in (None, "")]


def _normalize_tool_args(tool_name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        return {}

    if tool_name == "ghidra_headless" and "project_name" not in args and "name" in args:
        args["project_name"] = args["name"]
    if tool_name in {"bash_template", "python_template", "pwntools_template"} and "template_name" not in args and "name" in args:
        args["template_name"] = args["name"]

    alias_map = {
        "path": ["file", "filepath", "filename", "target", "input", "input_path"],
        "binary_path": ["binary", "bin", "exe", "elf", "program"],
        "command": ["cmd", "shell"],
        "pattern": ["regex", "query", "search"],
        "wordlist": ["wordlist_path", "wordlist_file", "wl"],
        "dest_dir": ["dest", "out", "output", "output_dir"],
        "project_dir": ["project", "project_path"],
        "project_name": ["proj_name"],
        "script": ["code", "python", "py"],
        "value": ["text", "data", "payload"],
        "encoding": ["codec", "enc"],
        "gdb_commands": ["gdb_cmds", "gdb"],
        "template_name": ["template", "tpl"],
        "url": ["endpoint", "uri", "target_url"],
        "args": ["options", "flags"],
        "text_or_path": ["path", "file", "text", "value", "input"],
    }

    for canonical, aliases in alias_map.items():
        if canonical in args:
            continue
        for alias in aliases:
            if alias in args:
                args[canonical] = args[alias]
                break

    if tool_name in {"bash_template", "python_template", "pwntools_template"}:
        return args

    if tool_name in {"ciphey", "lemmeknow"}:
        return _filter_args(args, {"text_or_path", "args", "json_output"})

    allowed_keys = {
        "bash": {"command"},
        "read_file": {"path"},
        "write_file": {"path", "content"},
        "list_dir": {"path"},
        "grep": {"pattern", "path"},
        "find": {"path", "name"},
        "hash_file": {"path", "algo"},
        "extract_archive": {"path", "dest_dir"},
        "file_info": {"path"},
        "binwalk": {"path"},
        "checksec": {"path"},
        "strings": {"path"},
        "exiftool": {"path"},
        "stegseek": {"path", "wordlist"},
        "zsteg": {"path"},
        "ghidra_headless": {"project_dir", "project_name", "binary_path"},
        "radare2_json": {"binary_path", "commands"},
        "objdump": {"binary_path", "args"},
        "readelf": {"binary_path", "args"},
        "gdb_pwndbg": {"binary_path", "gdb_commands"},
        "pwntools": {"script"},
        "pwntools_ret2win": {"binary_path", "offset", "win_symbol", "win_addr"},
        "pwntools_fmt_leak": {"binary_path", "leak_index"},
        "pwntools_rop_system": {"binary_path", "offset", "system_addr"},
        "python": {"script"},
        "base_decode": {"value", "encoding"},
        "curl": {"url", "args"},
        "http_request": {"url", "method", "headers", "params", "data", "timeout_s"},
        "ffuf": {"url", "wordlist", "args", "max_requests"},
    }

    return _filter_args(args, allowed_keys.get(tool_name, set()))


def _filter_args(args: dict, allowed: set[str]) -> dict:
    if not allowed:
        return args
    return {key: value for key, value in args.items() if key in allowed}


def _budget_exceeded(state: DCipherState) -> str | None:
    started_at = state.get("started_at")
    if started_at is None:
        return None
    elapsed = time.monotonic() - started_at
    if elapsed > settings.max_wall_seconds_per_challenge:
        return f"wall time {elapsed:.1f}s > {settings.max_wall_seconds_per_challenge}s"
    if state.get("tool_calls", 0) >= settings.max_tool_calls:
        return f"tool calls {state.get('tool_calls', 0)} >= {settings.max_tool_calls}"
    if state.get("phase_cycles", 0) >= settings.max_phase_cycles:
        return f"phase cycles {state.get('phase_cycles', 0)} >= {settings.max_phase_cycles}"
    if state.get("category_pivots", 0) >= settings.max_category_pivots:
        return f"category pivots {state.get('category_pivots', 0)} >= {settings.max_category_pivots}"
    return None


def _is_disallowed_command(command: str) -> bool:
    lowered = (command or "").lower()
    banned = ["nmap", "masscan", "zmap", "rustscan", "sqlmap"]
    return any(re.search(rf"\\b{re.escape(tool)}\\b", lowered) for tool in banned)


def _contains_network_command(command: str) -> bool:
    lowered = (command or "").lower()
    patterns = [r"\bcurl\b", r"\bwget\b", r"https?://", r"\bffuf\b"]
    return any(re.search(pat, lowered) for pat in patterns)


def _violates_url_scope(command: str, base_url: str) -> bool:
    base_host = _extract_host(base_url)
    if not base_host:
        return False
    for url in _extract_urls(command):
        if _extract_host(url) != base_host:
            return True
    return False


def _extract_urls(command: str) -> list[str]:
    return re.findall(r"https?://[^\s'\"]+", command or "")


def _extract_host(url: str) -> str:
    match = re.match(r"https?://([^/]+)", url or "")
    return match.group(1).lower() if match else ""


def _mark_done(state: DCipherState, reason: str, logger: TrajectoryLogger | None) -> dict:
    if logger:
        logger.log("budget", {"reason": reason})
    return {"done": True, "reasoning_log": [reason]}


def _infer_container_dir(context: str) -> str:
    match = re.search(r"FILES\s+\(container path ([^)]+)\):", context)
    return match.group(1).strip() if match else ""


def _build_research_command(container_dir: str) -> str:
    escaped_dir = json.dumps(container_dir)
    return (
        f"ls -la {escaped_dir} && "
        f"find {escaped_dir} -maxdepth 1 -type f -print0 | "
        "xargs -0 -I{} sh -c 'echo \"=== {}\"; file -b \"{}\"'"
    )


def _summarize_file_inventory(stdout: str) -> tuple[str, list[dict]]:
    inventory: list[dict] = []
    current_path = ""
    for line in stdout.splitlines():
        if line.startswith("==="):
            current_path = line.replace("===", "").strip()
            continue
        if current_path and line.strip():
            inventory.append({"path": current_path, "type": line.strip()})
            current_path = ""
    type_counts: dict[str, int] = {}
    for entry in inventory:
        ftype = entry.get("type", "unknown")
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
    summary_parts = [f"{t} x{c}" for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])][:5]
    summary = "Files analyzed: " + str(len(inventory))
    if summary_parts:
        summary += f" ({', '.join(summary_parts)})"
    return summary, inventory


def _persist_tool_output(state: DCipherState, result: ToolResult, phase: str) -> str:
    challenge_id = state.get("challenge_id", "unknown")
    run_dir = Path(state.get("run_dir") or Path(settings.runs_dir) / str(challenge_id))
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = log_dir / f"{result.tool}_{ts}.txt"
    payload = (
        f"TOOL: {result.tool}\n"
        f"PHASE: {phase}\n"
        f"COMMAND: {result.command}\n"
        f"EXIT CODE: {result.exit_code}\n"
        "STDOUT:\n"
        f"{result.stdout}\n"
        "STDERR:\n"
        f"{result.stderr}\n"
    )
    log_path.write_text(payload, encoding="utf-8", errors="replace")
    return str(log_path)

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


def _record_attempt(state: DCipherState, tool_result: ToolResult, messages: list, phase: str, log_path: str):
    attempt = {
        "step": state.get("iteration", 0) + 1,
        "phase": phase,
        "objective": state.get("current_objective", ""),
        "tool": tool_result.tool,
        "command": tool_result.command,
        "stdout": tool_result.stdout,
        "stderr": tool_result.stderr,
        "exit_code": tool_result.exit_code,
        "log_path": log_path,
    }

    next_tool_calls = state.get("tool_calls", 0) + 1
    updates = {
        "attempt_history": [attempt],
        "messages": messages,
        "last_command": tool_result.command,
        "last_output": tool_result.stdout,
        "last_error": tool_result.stderr,
        "last_exit_code": tool_result.exit_code,
        "iteration": state.get("iteration", 0) + 1,
        "tool_calls": next_tool_calls,
        "last_log_path": log_path,
    }
    if tool_result.parsed and "files" in tool_result.parsed:
        updates["current_files"] = tool_result.parsed["files"]
    return updates


def _truncate_text(text: str, max_chars: int, tail_chars: int = 800) -> str:
    if not text or len(text) <= max_chars:
        return text
    notice = "\n[...snip...]\n"
    head_chars = max(max_chars - tail_chars - len(notice), 0)
    if head_chars <= 0:
        return text[:max_chars]
    return f"{text[:head_chars]}{notice}{text[-tail_chars:]}"


def _truncate_lines(text: str, head: int, tail: int) -> str:
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text
    return "\n".join(lines[:head] + ["[...snip...]"] + lines[-tail:])


def _extract_indicators(text: str) -> dict:
    flags = _find_flag_matches(text, CTF_PREFIX_FLAG_REGEX) + _find_flag_matches(text, DEFAULT_FLAG_REGEX)
    flags = _dedupe_list(flags)[:5]
    paths = re.findall(r"/[A-Za-z0-9_\-./]{3,}", text)
    paths = _dedupe_list(paths)[:5]
    return {"flags": flags, "paths": paths}


def _format_tool_output(result: ToolResult, log_path: str) -> str:
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined = stdout + "\n" + stderr
    line_count = len(combined.splitlines())
    char_count = len(combined)
    large = (
        line_count > settings.output_large_line_threshold
        or char_count > settings.output_large_char_threshold
    )
    if large:
        stdout_view = _truncate_lines(stdout, settings.output_head_lines, settings.output_tail_lines)
        stderr_view = _truncate_lines(stderr, settings.output_head_lines, settings.output_tail_lines)
    else:
        stdout_view = stdout
        stderr_view = stderr

    indicators = _extract_indicators(combined)
    summary = f"lines={line_count} chars={char_count}"
    return (
        f"TOOL: {result.tool}\n"
        f"COMMAND: {result.command}\n"
        f"LOG: {log_path}\n"
        f"SUMMARY: {summary}\n"
        f"INDICATORS: {indicators}\n"
        f"STDOUT:\n{stdout_view}\n"
        f"STDERR:\n{stderr_view}\n"
        f"EXIT CODE: {result.exit_code}"
    )


def _parse_plan_objective(text: str) -> tuple[str, str, list[str], str]:
    plan_match = re.search(r"PLAN:\s*(.*?)(?:OBJECTIVE:|$)", text, re.DOTALL)
    obj_match = re.search(r"OBJECTIVE:\s*(.*)", text, re.DOTALL)
    cat_match = re.search(r"CATEGORIES:\s*(.*)", text)
    pipe_match = re.search(r"PIPELINE:\s*(.*)", text)

    plan = plan_match.group(1).strip() if plan_match else text.strip()
    objective = obj_match.group(1).strip() if obj_match else (plan.splitlines()[0] if plan else "")

    categories: list[str] = []
    if cat_match:
        raw = cat_match.group(1)
        categories = [c.strip() for c in re.split(r"[,/]", raw) if c.strip()]

    pipeline = pipe_match.group(1).strip() if pipe_match else ""
    return plan, objective, categories, pipeline


def _summarize_attempts(attempts: list[dict]) -> str:
    if not attempts:
        return "None"
    lines = []
    for attempt in attempts:
        lines.append(
            f"{attempt.get('step')}[{attempt.get('phase', 'unknown')}]: "
            f"{attempt.get('tool')} {attempt.get('command')} "
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
        f"stdout_len={len(last.get('stdout', ''))} "
        f"log={last.get('log_path') or state.get('last_log_path', '')}"
    )


def _auto_failure_hint(state: DCipherState) -> str | None:
    exit_code = state.get("last_exit_code")
    stderr = (state.get("last_error") or "").lower()
    if exit_code == 127 or "command not found" in stderr:
        return "Command not found (exit 127). Check the tool name or install it in the Kali container."
    if exit_code == 2 and "disallowed" in stderr:
        return "Disallowed command. Avoid port scanning or prohibited tooling."
    if exit_code == 124 or "timed out" in stderr:
        return "Command timed out. Consider narrowing scope, adding filters, or increasing timeout."
    return None


def _extract_flags(text: str, flag_format: str) -> list[str]:
    matches: list[str] = []
    if flag_format:
        prefix = flag_format.rstrip("{").strip()
        if prefix:
            matches.extend(_find_flag_matches(text, re.escape(prefix) + r"\{[^}]{3,}\}"))

    if not matches:
        matches.extend(_find_flag_matches(text, CTF_PREFIX_FLAG_REGEX))
        matches.extend(_find_flag_matches(text, DEFAULT_FLAG_REGEX))

    return _dedupe_list(matches)


def _extract_flags_with_evidence(stdout: str, stderr: str, flag_format: str, log_path: str) -> list[dict]:
    hits: list[dict] = []
    hits.extend(_find_flags_in_text(stdout or "", "stdout", flag_format, log_path))
    hits.extend(_find_flags_in_text(stderr or "", "stderr", flag_format, log_path))
    deduped = []
    seen = set()
    for hit in hits:
        flag = hit["flag"]
        if flag in seen:
            continue
        seen.add(flag)
        deduped.append(hit)
    return deduped


def _find_flags_in_text(text: str, source: str, flag_format: str, log_path: str) -> list[dict]:
    patterns = []
    if flag_format:
        prefix = flag_format.rstrip("{").strip()
        if prefix:
            patterns.append(re.escape(prefix) + r"\{[^}]{3,}\}")
    patterns.append(CTF_PREFIX_FLAG_REGEX)
    patterns.append(DEFAULT_FLAG_REGEX)

    hits: list[dict] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            flag = match.group(0)
            line = _extract_line(text, match.start(), match.end())
            hits.append({
                "flag": flag,
                "evidence": line,
                "evidence_type": source,
                "log_path": log_path or "",
            })
    return hits


def _extract_line(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    snippet = text[line_start:line_end].strip()
    if len(snippet) > 200:
        snippet = snippet[:200] + "...(truncated)"
    return snippet


def _merge_flags(existing: list[str], new_flags: list[str]) -> list[str]:
    merged = list(existing)
    for flag in new_flags:
        if flag not in merged:
            merged.append(flag)
    return merged


def _new_flag_hits(existing_hits: list[dict], new_hits: list[dict], state: DCipherState) -> list[dict]:
    if not new_hits:
        return []
    last_attempt = state.get("attempt_history", [])[-1] if state.get("attempt_history") else {}
    step = int(last_attempt.get("step", state.get("iteration", 0)))
    tool = str(last_attempt.get("tool", state.get("last_command", "")))
    command = str(last_attempt.get("command", state.get("last_command", "")))

    existing_set = {hit.get("flag") for hit in existing_hits}
    out = []
    for hit in new_hits:
        flag = hit.get("flag")
        if not flag or flag in existing_set:
            continue
        out.append({
            "flag": flag,
            "step": step,
            "tool": tool,
            "command": command,
            "evidence": hit.get("evidence", ""),
            "evidence_type": hit.get("evidence_type", ""),
            "log_path": hit.get("log_path", state.get("last_log_path", "")),
        })
    return out


def _is_flag_like(flag: str, flag_format: str) -> bool:
    if flag_format:
        prefix = flag_format.rstrip("{").strip()
        if prefix and re.search(re.escape(prefix) + r"\{[^}]+\}", flag, re.IGNORECASE):
            return True
    if re.search(CTF_PREFIX_FLAG_REGEX, flag, re.IGNORECASE):
        return True
    return re.search(DEFAULT_FLAG_REGEX, flag, re.IGNORECASE) is not None


def _find_flag_matches(text: str, pattern: str) -> list[str]:
    return re.findall(pattern, text, flags=re.IGNORECASE)


def _dedupe_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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
