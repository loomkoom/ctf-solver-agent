import json
import re
import time
import shlex
from collections import Counter
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
from src.tools.inventory import resolve_tool_manual_key
from src.tools.toolbox import ToolResult, Toolbox
from src.trajectory import TrajectoryLogger
from src.debug_log import debug_log, head_lines, preview_text, redact

MAX_RAG_CHARS = 3500
DEFAULT_FLAG_REGEX = r"\b[A-Za-z0-9_\-]{0,24}\{[^\n\r]{3,200}\}\b"
CTF_PREFIX_FLAG_REGEX = r"(?:flag|IGCTF|ctf|picoCTF|HTB|TBTL)\{[^}]{1,200}\}"
DEBUG_TOOL_PREVIEW_LINES = 8
HEX_CANDIDATE_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
BASE64_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9+/]{12,}={0,2}\b")

_TOOL_ALIASES = {
    "cat": "read_file",
    "type": "read_file",
    "ls": "list_dir",
    "dir": "list_dir",
    "file": "file_info",
}

_PATH_PLACEHOLDER_MARKERS = (
    "path/to",
    "<path>",
    "your/path",
    "yourfile",
    "your_file",
    "file_here",
    "path_here",
)


def _debug_context(state: DCipherState) -> dict:
    return {
        "run_id": state.get("run_id"),
        "challenge_id": state.get("challenge_id"),
        "challenge_name": state.get("challenge_name"),
    }


def _log_event(state: DCipherState, event: str, **fields) -> None:
    debug_log(
        settings.debug,
        state.get("run_id"),
        state.get("challenge_id"),
        state.get("challenge_name"),
        event,
        **fields,
    )


def _llm_temperature(llm) -> float | int | str | None:
    value = getattr(llm, "temperature", None)
    if value is not None:
        return value
    model_kwargs = getattr(llm, "model_kwargs", None)
    if isinstance(model_kwargs, dict) and "temperature" in model_kwargs:
        return model_kwargs.get("temperature")
    kwargs = getattr(llm, "kwargs", None)
    if isinstance(kwargs, dict) and "temperature" in kwargs:
        return kwargs.get("temperature")
    return None


def _summarize_messages(messages) -> dict:
    roles: list[str] = []
    lengths: list[int] = []
    combined_parts: list[str] = []
    total_chars = 0
    for msg in messages:
        role = getattr(msg, "type", msg.__class__.__name__)
        content = getattr(msg, "content", "") or ""
        roles.append(str(role))
        lengths.append(len(content))
        total_chars += len(content)
        combined_parts.append(content)
    combined = "\n".join(combined_parts)
    return {
        "msg_count": len(messages),
        "msg_chars": total_chars,
        "msg_lengths": lengths,
        "msg_roles": roles,
        "msg_preview": preview_text(redact(combined), 200),
    }


def _format_inventory_for_prompt(inventory: list[dict], limit: int = 8) -> str:
    if not inventory:
        return "none"
    lines = []
    for entry in inventory[:limit]:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path", "")
        if not path:
            continue
        ftype = entry.get("type") or entry.get("mime") or ""
        if ftype:
            lines.append(f"- {path} ({ftype})")
        else:
            lines.append(f"- {path}")
    if not lines:
        return "none"
    if len(inventory) > limit:
        lines.append(f"...({len(inventory) - limit} more)")
    return "\n".join(lines)


def _extract_message_text(message) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block and block.get("text"):
                    parts.append(str(block.get("text")))
                elif "refusal" in block and block.get("refusal"):
                    parts.append(str(block.get("refusal")))
            else:
                if hasattr(block, "text") and getattr(block, "text"):
                    parts.append(str(getattr(block, "text")))
                elif hasattr(block, "refusal") and getattr(block, "refusal"):
                    parts.append(str(getattr(block, "refusal")))
        text = "\n".join([part for part in parts if part])
    else:
        text = str(content) if content is not None else ""

    if not text:
        additional = getattr(message, "additional_kwargs", {}) or {}
        refusal = additional.get("refusal")
        if refusal:
            return str(refusal)
    return text


def _extract_refusal_data(message) -> tuple[bool, str]:
    if message is None:
        return False, ""
    refusal_present = False
    refusal_text = ""

    additional = getattr(message, "additional_kwargs", {}) or {}
    if "refusal" in additional:
        refusal_present = True
        if additional.get("refusal"):
            refusal_text = str(additional.get("refusal"))

    if hasattr(message, "refusal"):
        refusal_present = True
        val = getattr(message, "refusal")
        if val:
            refusal_text = str(val)

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if "refusal" in block:
                    refusal_present = True
                    if block.get("refusal"):
                        refusal_text = str(block.get("refusal"))
                        break
                if block.get("type") == "refusal" and block.get("text"):
                    refusal_present = True
                    refusal_text = str(block.get("text"))
                    break
            else:
                if hasattr(block, "refusal"):
                    refusal_present = True
                    val = getattr(block, "refusal")
                    if val:
                        refusal_text = str(val)
                        break

    return refusal_present, refusal_text


def _response_needs_retry(message) -> tuple[bool, str]:
    text = _extract_message_text(message)
    refusal_present, refusal_text = _extract_refusal_data(message)
    if refusal_present:
        return True, "refusal" if refusal_text else "refusal-empty"
    if not (text or "").strip():
        return True, "empty"
    return False, ""


def _summarize_llm_response(message) -> dict:
    content = getattr(message, "content", None)
    additional = getattr(message, "additional_kwargs", {}) or {}
    metadata = getattr(message, "response_metadata", {}) or {}
    refusal_present, refusal_text = _extract_refusal_data(message)
    summary = {
        "content_type": type(content).__name__,
        "content_preview": preview_text(redact(str(content)), 200) if content is not None else "",
        "additional_keys": list(additional.keys()),
        "response_metadata_keys": list(metadata.keys()),
        "refusal_present": refusal_present,
        "refusal_len": len(refusal_text or "") if refusal_present else 0,
        "refusal_preview": preview_text(redact(refusal_text), 120) if refusal_present else "",
    }
    if isinstance(content, list):
        block_types: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_types.append(str(block.get("type") or "dict"))
            else:
                block_types.append(type(block).__name__)
        summary["content_block_types"] = block_types[:8]
        if len(block_types) > 8:
            summary["content_block_types_truncated"] = len(block_types)
    if "status" in metadata:
        summary["response_status"] = metadata.get("status")
    if "incomplete_details" in metadata:
        summary["incomplete_details"] = metadata.get("incomplete_details")
    if "id" in metadata:
        summary["response_id"] = metadata.get("id")
    return summary


def _fallback_llm_response(role: str, state: DCipherState) -> AIMessage:
    if role in {"planner", "executor"}:
        return StubLLM(role).invoke([])
    hint = _auto_failure_hint(state) or "No verifier response. Inspect stdout/stderr and retry."
    return AIMessage(content=hint)


def _invoke_llm(
    llm,
    messages,
    state: DCipherState,
    role: str,
    tier,
    purpose: str,
    fallback_on_empty: bool = True,
):
    summary = _summarize_messages(messages)
    max_tokens = tier.max_tokens if tier.max_tokens is not None else "unset"
    num_predict = tier.max_tokens if tier.provider == "ollama" else "unset"
    temperature = _llm_temperature(llm)
    if temperature is None:
        temperature = "unset"
    _log_event(
        state,
        "llm_call_start",
        role=role,
        purpose=purpose,
        provider=tier.provider,
        model=tier.model,
        max_tokens=max_tokens,
        num_predict=num_predict,
        temperature=temperature,
        **summary,
    )
    _log_event(
        state,
        "llm_stream_start",
        role=role,
        purpose=purpose,
        streaming=False,
    )
    start = time.monotonic()
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        duration = time.monotonic() - start
        _log_event(
            state,
            "llm_stream_end",
            role=role,
            purpose=purpose,
            streaming=False,
            duration_s=round(duration, 3),
            status="error",
        )
        _log_event(
            state,
            "llm_call_error",
            role=role,
            purpose=purpose,
            error=repr(exc),
            retry=0,
            duration_s=round(duration, 3),
        )
        raise
    duration = time.monotonic() - start
    response_text = _extract_message_text(response)
    raw_response_chars = len(response_text or "")
    refusal_present, refusal_text = _extract_refusal_data(response)
    if not response_text.strip() or refusal_present:
        reason = "refusal" if refusal_present else "empty"
        _log_event(
            state,
            "llm_empty_response",
            role=role,
            purpose=purpose,
            provider=tier.provider,
            model=tier.model,
            reason=reason,
        )
        _log_event(
            state,
            "llm_empty_response_detail",
            role=role,
            purpose=purpose,
            reason=reason,
            **_summarize_llm_response(response),
        )
        if fallback_on_empty:
            response = _fallback_llm_response(role, state)
            response_text = _extract_message_text(response)
    try:
        if response_text and response_text != getattr(response, "content", ""):
            response = response.model_copy(update={"content": response_text})
    except Exception:
        if response_text:
            response = AIMessage(content=response_text)
    _log_event(
        state,
        "llm_stream_end",
        role=role,
        purpose=purpose,
        streaming=False,
        duration_s=round(duration, 3),
        status="ok",
    )
    _log_event(
        state,
        "llm_call_end",
        role=role,
        purpose=purpose,
        duration_s=round(duration, 3),
        response_chars=len(response_text or ""),
        raw_response_chars=raw_response_chars,
    )
    return response


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
    verifier_tiers = load_tiers("verifier")
    llm_cache: dict[tuple[str, str, str, int | None], object] = {}
    tiers_by_role = {
        "planner": planner_tiers,
        "executor": executor_tiers,
        "verifier": verifier_tiers,
    }

    def _get_llm(role: str, state: DCipherState, force_idx: int | None = None):
        tiers = tiers_by_role.get(role) or []
        if not tiers:
            tiers = [load_tiers(role)[0]]
        if force_idx is None:
            idx = select_tier_index(state, tiers, role)
        else:
            idx = max(0, min(force_idx, len(tiers) - 1))
        tier = tiers[idx]
        key = (role, tier.provider, tier.model, tier.max_tokens)
        llm = llm_cache.get(key)
        if llm is None:
            llm = _init_llm(tier.provider, tier.model, role=role, max_tokens=tier.max_tokens)
            llm_cache[key] = llm
        return llm, tier, idx

    def _invoke_with_fallback(
        role: str,
        messages: list,
        purpose: str,
        llm,
        tier,
        tier_idx: int,
    ) -> AIMessage:
        response = _invoke_llm(
            llm,
            messages,
            state,
            role,
            tier,
            purpose,
            fallback_on_empty=False,
        )
        needs_retry, reason = _response_needs_retry(response)
        if not needs_retry:
            return response

        tiers = tiers_by_role.get(role, [])
        _log_event(
            state,
            "llm_retry",
            role=role,
            purpose=purpose,
            reason=reason,
            from_tier=tier_idx,
            to_tier=min(tier_idx + 1, max(len(tiers) - 1, 0)),
        )
        if tier_idx < len(tiers) - 1:
            next_llm, next_tier, next_idx = _get_llm(role, state, force_idx=tier_idx + 1)
            retry_response = _invoke_llm(
                next_llm,
                messages,
                state,
                role,
                next_tier,
                purpose,
                fallback_on_empty=False,
            )
            needs_retry_again, reason_again = _response_needs_retry(retry_response)
            if needs_retry_again:
                _log_event(
                    state,
                    "llm_retry_failed",
                    role=role,
                    purpose=purpose,
                    reason=reason_again,
                    from_tier=next_idx,
                )
                _log_event(
                    state,
                    "llm_fallback_stub",
                    role=role,
                    purpose=purpose,
                    reason=reason_again,
                    from_tier=next_idx,
                )
                return _fallback_llm_response(role, state)
            return retry_response

        _log_event(
            state,
            "llm_fallback_stub",
            role=role,
            purpose=purpose,
            reason=reason,
            from_tier=tier_idx,
        )
        return _fallback_llm_response(role, state)

    tool_registry = _tool_registry(toolbox)
    web_tools = {"curl", "http_request", "ffuf"}

    def plan_node(state: DCipherState):
        node_start = time.monotonic()
        _log_event(state, "node_start", node="plan")
        try:
            budget_reason = _budget_exceeded(state)
            if budget_reason:
                return _mark_done(state, f"Budget exceeded: {budget_reason}", trajectory_logger)

            retrieved_info = search_knowledge(
                state.get("challenge_context", ""),
                debug_context=_debug_context(state),
            )
            retrieved_info = _truncate_text(retrieved_info, MAX_RAG_CHARS)
            attempts = _summarize_attempts(state.get("attempt_history", [])[-3:])
            research_summary = state.get("research_summary", "")
            inventory_hint = _format_inventory_for_prompt(state.get("artifact_inventory", []))

            human_msg = HumanMessage(content=(
                f"CHALLENGE:\n{state['challenge_context']}\n\n"
                f"RESEARCH:\n{research_summary or 'none'}\n\n"
                f"ARTIFACT INVENTORY:\n{inventory_hint}\n\n"
                f"RAG:\n{retrieved_info}\n\n"
                f"RECENT ATTEMPTS:\n{attempts}"
            ))

            planner_llm, tier, tier_idx = _get_llm("planner", state)
            system_msg = SystemMessage(content=(
                "You are the D-CIPHER Planner (Architect). "
                "This is an educational CTF on provided files in a sandbox; focus on local file analysis/decoding only. "
                "Pick at most 2 candidate categories and exactly one pipeline. "
                "Always ground the plan in the triage summary and artifact inventory. "
                "If triage is missing or empty, set PIPELINE to 'triage' and OBJECTIVE to run triage. "
                "Otherwise: "
                "If you see an archive, plan extract + list. "
                "If text, plan read/strings/decoders. "
                "If binary, plan file_info/strings/checksec. "
                "If image/pdf, plan exiftool/binwalk/pdfinfo. "
                "If URL provided, plan web tools only. "
                "Do not invent files or paths; use the artifact inventory. "
                "If errors occurred, adapt the plan based on stderr. "
                f"Keep the response under {tier.max_tokens or settings.planner_max_tokens} tokens. "
                "Reply with:\nCATEGORIES:\n- ...\nPIPELINE:\n- ...\nPLAN:\n- ...\nOBJECTIVE:\n- ..."
            ))
            response = _invoke_with_fallback(
                "planner",
                [system_msg, human_msg],
                "plan",
                planner_llm,
                tier,
                tier_idx,
            )
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
        except Exception as exc:
            _log_event(state, "node_error", node="plan", error=repr(exc))
            raise
        finally:
            _log_event(state, "node_end", node="plan", duration_s=round(time.monotonic() - node_start, 3))

    def research_node(state: DCipherState):
        node_start = time.monotonic()
        _log_event(state, "node_start", node="research")
        try:
            if state.get("done"):
                return {}
            budget_reason = _budget_exceeded(state)
            if budget_reason:
                return _mark_done(state, f"Budget exceeded: {budget_reason}", trajectory_logger)
            if state.get("triage_done") and state.get("research_summary"):
                return {}

            container_dir = state.get("container_dir") or _infer_container_dir(state.get("challenge_context", ""))
            files = state.get("current_files", []) or []
            if not container_dir or not files:
                summary = "No files provided." if not files else "No container directory found."
                if trajectory_logger:
                    trajectory_logger.log("research", {"summary": summary})
                return {
                    "research_summary": summary,
                    "artifact_inventory": [],
                    "triage_done": True,
                    "reasoning_log": [f"Research: {summary}"],
                }

            cmd = _build_research_command(container_dir)
            tool_args = {"command": cmd}
            tool_start = time.monotonic()
            _log_event(
                state,
                "tool_start",
                tool="bash",
                phase="research",
                args=redact(tool_args),
            )
            tool_result = toolbox.run(cmd)
            tool_duration = time.monotonic() - tool_start
            _log_event(
                state,
                "tool_end",
                tool=tool_result.tool,
                phase="research",
                duration_s=round(tool_duration, 3),
                exit_code=tool_result.exit_code,
                stdout_head=redact(head_lines(tool_result.stdout, DEBUG_TOOL_PREVIEW_LINES)),
                stderr_head=redact(head_lines(tool_result.stderr, DEBUG_TOOL_PREVIEW_LINES)),
            )
            log_path = _persist_tool_output(state, tool_result, phase="research")
            tool_result.log_path = log_path
            summary, inventory = _summarize_file_inventory(tool_result.stdout)

            output_msg = HumanMessage(
                content=_format_tool_output(tool_result, log_path)
            )
            updates = _record_attempt(state, tool_result, [output_msg], phase="research", log_path=log_path)
            updates["research_summary"] = summary
            updates["artifact_inventory"] = inventory
            updates["triage_done"] = True
            updates.setdefault("reasoning_log", []).append(f"Research: {summary}")
            if trajectory_logger:
                trajectory_logger.log("research", {"summary": summary, "log_path": log_path})
            return updates
        except Exception as exc:
            _log_event(state, "node_error", node="research", error=repr(exc))
            raise
        finally:
            _log_event(state, "node_end", node="research", duration_s=round(time.monotonic() - node_start, 3))

    def execute_node(state: DCipherState):
        node_start = time.monotonic()
        _log_event(state, "node_start", node="execute")
        try:
            if state.get("done"):
                return {}
            budget_reason = _budget_exceeded(state)
            if budget_reason:
                return _mark_done(state, f"Budget exceeded: {budget_reason}", trajectory_logger)

            objective = state.get("current_objective") or "Initial reconnaissance"
            url = state.get("url") or ""
            research_summary = state.get("research_summary", "")
            inventory = state.get("artifact_inventory", [])
            container_dir = state.get("container_dir", "")
            avoid_tools, recent_tools = _recent_tool_guard(state, inventory)
            avoid_note = ", ".join(avoid_tools) if avoid_tools else "none"
            recent_note = ", ".join(recent_tools) if recent_tools else "none"
            last_output_summary = _truncate_text(state.get("last_output", ""), 1200, tail_chars=400)
            last_error_summary = _truncate_text(state.get("last_error", ""), 800, tail_chars=300)
            verifier_hint = state.get("verifier_hint", "") or "none"

            if "check artifacts" in objective.lower():
                path = extract_path_from_text(objective)
                tool_args = {"path": path or ""}
                tool_start = time.monotonic()
                _log_event(
                    state,
                    "tool_start",
                    tool="artifacts",
                    phase="execute",
                    args=redact(tool_args),
                )
                artifact_msg = list_challenge_artifacts(path)
                tool_result = ToolResult(
                    tool="artifacts",
                    command=path or "",
                    stdout=artifact_msg,
                    stderr="",
                    exit_code=0
                )
                tool_duration = time.monotonic() - tool_start
                _log_event(
                    state,
                    "tool_end",
                    tool=tool_result.tool,
                    phase="execute",
                    duration_s=round(tool_duration, 3),
                    exit_code=tool_result.exit_code,
                    stdout_head=redact(head_lines(tool_result.stdout, DEBUG_TOOL_PREVIEW_LINES)),
                    stderr_head=redact(head_lines(tool_result.stderr, DEBUG_TOOL_PREVIEW_LINES)),
                )
                log_path = _persist_tool_output(state, tool_result, phase="execute")
                tool_result.log_path = log_path
                ai_msg = AIMessage(content=f"Artifacts lookup: {artifact_msg}")
                return _record_attempt(state, tool_result, [ai_msg], phase="execute", log_path=log_path)

            auto_decode = _auto_decode_suggestion(state)
            if auto_decode:
                tool_name = "base_decode"
                args = {
                    "value": auto_decode["value"],
                    "encoding": auto_decode["encoding"],
                }
                tool_start = time.monotonic()
                _log_event(
                    state,
                    "tool_start",
                    tool=tool_name,
                    phase="execute",
                    args=redact(args),
                )
                try:
                    tool_result = tool_registry[tool_name](**args)
                except Exception as exc:
                    tool_result = ToolResult(
                        tool=tool_name,
                        command=json.dumps(args),
                        stdout="",
                        stderr=f"Auto-decode failed: {exc}",
                        exit_code=2,
                    )
                tool_duration = time.monotonic() - tool_start
                _log_event(
                    state,
                    "tool_end",
                    tool=tool_result.tool,
                    phase="execute",
                    duration_s=round(tool_duration, 3),
                    exit_code=tool_result.exit_code,
                    stdout_head=redact(head_lines(tool_result.stdout, DEBUG_TOOL_PREVIEW_LINES)),
                    stderr_head=redact(head_lines(tool_result.stderr, DEBUG_TOOL_PREVIEW_LINES)),
                )
                log_path = _persist_tool_output(state, tool_result, phase="execute")
                tool_result.log_path = log_path
                output_msg = HumanMessage(content=_format_tool_output(tool_result, log_path))
                updates = _record_attempt(state, tool_result, [output_msg], phase="execute", log_path=log_path)
                updates["last_decode"] = {
                    "value": auto_decode["value"],
                    "encoding": auto_decode["encoding"],
                }
                updates.setdefault("reasoning_log", []).append(
                    f"Auto-decode: {auto_decode['reason']} ({auto_decode['encoding']})"
                )
                if trajectory_logger:
                    trajectory_logger.log("execute", {"tool": tool_result.tool, "command": tool_result.command, "log_path": log_path})
                return updates

            fallback = _fallback_tool_for_failure(state, inventory, avoid_tools)
            if fallback:
                tool_name = fallback["tool"]
                args = fallback["args"]
                tool_start = time.monotonic()
                _log_event(
                    state,
                    "tool_start",
                    tool=tool_name,
                    phase="execute",
                    args=redact(args),
                )
                try:
                    tool_result = tool_registry[tool_name](**args)
                except Exception as exc:
                    tool_result = ToolResult(
                        tool=tool_name,
                        command=json.dumps(args),
                        stdout="",
                        stderr=f"Fallback tool failed: {exc}",
                        exit_code=2,
                    )
                tool_duration = time.monotonic() - tool_start
                _log_event(
                    state,
                    "tool_end",
                    tool=tool_result.tool,
                    phase="execute",
                    duration_s=round(tool_duration, 3),
                    exit_code=tool_result.exit_code,
                    stdout_head=redact(head_lines(tool_result.stdout, DEBUG_TOOL_PREVIEW_LINES)),
                    stderr_head=redact(head_lines(tool_result.stderr, DEBUG_TOOL_PREVIEW_LINES)),
                )
                log_path = _persist_tool_output(state, tool_result, phase="execute")
                tool_result.log_path = log_path
                output_msg = HumanMessage(content=_format_tool_output(tool_result, log_path))
                updates = _record_attempt(state, tool_result, [output_msg], phase="execute", log_path=log_path)
                updates.setdefault("reasoning_log", []).append(fallback.get("reason", "Fallback tool invoked."))
                if trajectory_logger:
                    trajectory_logger.log("execute", {"tool": tool_result.tool, "command": tool_result.command, "log_path": log_path})
                return updates

            executor_llm, tier, tier_idx = _get_llm("executor", state)
            system_msg = SystemMessage(content=(
                "You are the D-CIPHER Executor (Operator). Pick exactly one tool call. "
                "Respond with raw JSON only: {\"tool\": \"name\", \"args\": {..}} "
                "Do not wrap the JSON in code fences or add commentary. "
                "Prefer toolbox tools (read_file, list_dir, extract_archive, file_info, strings, base_decode) over bash. "
                "Use tool \"bash\" only for short, concrete shell commands or small pipelines. "
                "Only use web tools if an explicit URL is provided. "
                "Use real paths from the ARTIFACT INVENTORY; never use placeholders like /path/to/file. "
                "If the objective is to inspect an archive, use extract_archive with the archive path. "
                "Tool manuals exist in the local KB; use them to choose correct flags/args. "
                f"Recent tools: {recent_note}. Avoid repeating tools: {avoid_note} unless a new path is explicitly required. "
                f"Keep the response under {tier.max_tokens or settings.executor_max_tokens} tokens. "
                f"Tools: {', '.join(tool_registry.keys())}"
            ))
            human_msg = HumanMessage(content=(
                f"OBJECTIVE:\n{objective}\n\n"
                f"URL:\n{url or 'none'}\n\n"
                f"CONTAINER_DIR:\n{container_dir or 'unknown'}\n\n"
                f"RESEARCH SUMMARY:\n{research_summary or 'none'}\n\n"
                f"LAST OUTPUT (summary):\n{last_output_summary or 'none'}\n\n"
                f"LAST ERROR (summary):\n{last_error_summary or 'none'}\n\n"
                f"VERIFIER HINT:\n{verifier_hint}\n\n"
                f"ARTIFACT INVENTORY:\n{_format_inventory_for_prompt(inventory)}\n\n"
                f"RECENT TOOLS:\n{recent_note}\n"
                f"AVOID REPEATS:\n{avoid_note}"
            ))

            response = _invoke_with_fallback(
                "executor",
                [system_msg, human_msg],
                "execute_select",
                executor_llm,
                tier,
                tier_idx,
            )
            tool_name, args, raw_cmd = _parse_tool_call(response.content)
            tool_name = _normalize_tool_name(tool_name)
            executor_messages = [response]

            if tool_name == "bash" and "command" not in args:
                if "commands" in args:
                    args = {"command": _join_commands(args.get("commands"))}
                else:
                    fallback_cmd = raw_cmd or response.content.strip()
                    args = {"command": fallback_cmd}

            args = _normalize_tool_args(tool_name, args)
            args = _resolve_artifact_paths(args, inventory)
            manual_key = _resolve_tool_manual_key_for_call(tool_name, args)
            manual_text = ""
            manual_seen = list(state.get("tool_manuals_seen", []))
            if manual_key and manual_key not in manual_seen:
                manual_text = search_knowledge(
                    f"TOOL MANUAL: {manual_key}",
                    debug_context=_debug_context(state),
                )
                manual_text = _truncate_text(manual_text, MAX_RAG_CHARS)
                manual_seen.append(manual_key)

            if manual_text and tool_name != "bash":
                refine_system_msg = SystemMessage(content=(
                    f"You already chose tool '{tool_name}'. Use the manual below to refine args. "
                    "Respond with JSON: {\"tool\": \"name\", \"args\": {..}} and keep the same tool."
                ))
                refine_human_msg = HumanMessage(content=(
                    f"OBJECTIVE:\n{objective}\n\n"
                    f"CURRENT ARGS:\n{json.dumps(args)}\n\n"
                    f"TOOL MANUAL:\n{manual_text}\n"
                ))
                refine_response = _invoke_llm(
                    executor_llm,
                    [refine_system_msg, refine_human_msg],
                    state,
                    "executor",
                    tier,
                    "execute_refine",
                )
                executor_messages.append(refine_response)
                refined_tool, refined_args, refined_cmd = _parse_tool_call(refine_response.content)
                if refined_tool and refined_tool != tool_name:
                    refined_tool = tool_name
                if refined_args:
                    tool_name = refined_tool or tool_name
                    args = _normalize_tool_args(tool_name, refined_args)
                    args = _resolve_artifact_paths(args, inventory)
                    raw_cmd = refined_cmd

            missing = _missing_required_args(tool_name, args)
            if missing:
                inventory_hint = _format_inventory_for_prompt(inventory)
                repair_system_msg = SystemMessage(content=(
                    f"You selected tool '{tool_name}' but required args are missing: {', '.join(missing)}. "
                    "Provide corrected JSON: {\"tool\": \"name\", \"args\": {..}}. "
                    "Keep the same tool unless it cannot satisfy the objective. "
                    "Use real paths from ARTIFACT INVENTORY; avoid placeholders like path/to/file."
                ))
                repair_human_msg = HumanMessage(content=(
                    f"OBJECTIVE:\n{objective}\n\n"
                    f"URL:\n{url or 'none'}\n\n"
                    f"CURRENT ARGS:\n{json.dumps(args)}\n\n"
                    f"MISSING:\n{', '.join(missing)}\n"
                    f"ARTIFACT INVENTORY:\n{inventory_hint}\n"
                ))
                repair_response = _invoke_llm(
                    executor_llm,
                    [repair_system_msg, repair_human_msg],
                    state,
                    "executor",
                    tier,
                    "execute_repair",
                )
                executor_messages.append(repair_response)
                repaired_tool, repaired_args, repaired_cmd = _parse_tool_call(repair_response.content)
                if repaired_tool:
                    tool_name = repaired_tool
                if repaired_args:
                    args = _normalize_tool_args(tool_name, repaired_args)
                    args = _resolve_artifact_paths(args, inventory)
                if repaired_cmd:
                    raw_cmd = repaired_cmd

            tool_start = time.monotonic()
            _log_event(
                state,
                "tool_start",
                tool=tool_name or "unknown",
                phase="execute",
                args=redact(args),
            )
            if _should_block_redundant_tool(tool_name, args, avoid_tools):
                tool_result = ToolResult(
                    tool=tool_name,
                    command=json.dumps(args),
                    stdout="",
                    stderr="Redundant list_dir blocked (triage already done). Choose a different tool or a new path.",
                    exit_code=2,
                )
            elif tool_name in web_tools and not url:
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
                        stderr=(
                            f"Missing required args for {tool_name}: {', '.join(missing)}\n"
                            f"Provided args: {json.dumps(args)}"
                        ),
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

            tool_duration = time.monotonic() - tool_start
            _log_event(
                state,
                "tool_end",
                tool=tool_result.tool,
                phase="execute",
                duration_s=round(tool_duration, 3),
                exit_code=tool_result.exit_code,
                stdout_head=redact(head_lines(tool_result.stdout, DEBUG_TOOL_PREVIEW_LINES)),
                stderr_head=redact(head_lines(tool_result.stderr, DEBUG_TOOL_PREVIEW_LINES)),
            )
            log_path = _persist_tool_output(state, tool_result, phase="execute")
            tool_result.log_path = log_path
            output_msg = HumanMessage(content=_format_tool_output(tool_result, log_path))
            updates = _record_attempt(state, tool_result, executor_messages + [output_msg], phase="execute", log_path=log_path)
            if manual_seen != state.get("tool_manuals_seen", []):
                updates["tool_manuals_seen"] = manual_seen
            if trajectory_logger:
                trajectory_logger.log("execute", {"tool": tool_result.tool, "command": tool_result.command, "log_path": log_path})
            return updates
        except Exception as exc:
            _log_event(state, "node_error", node="execute", error=repr(exc))
            raise
        finally:
            _log_event(state, "node_end", node="execute", duration_s=round(time.monotonic() - node_start, 3))

    def verify_node(state: DCipherState):
        node_start = time.monotonic()
        _log_event(state, "node_start", node="verify")
        try:
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
            auto_decode = None
            auto_decode_hint = "none"
            if not flags:
                auto_decode = _auto_decode_candidate(output)
                if auto_decode:
                    auto_decode_hint = f"{auto_decode['encoding']}: {preview_text(auto_decode['value'], 120)}"
                    updates.setdefault("reasoning_log", []).append(
                        f"Auto-decode hint: {auto_decode['reason']} ({auto_decode['encoding']})"
                    )

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

            verifier_llm, tier, tier_idx = _get_llm("verifier", state)
            system_msg = SystemMessage(content=(
                "You are the D-CIPHER Verifier (Critic). Analyze the last command output. "
                "If it failed, state the exact cause from stderr/exit code and propose ONE concrete next action. "
                "If it succeeded but no flag is found, propose the smallest next step grounded in triage (decode, extract, strings, etc.). "
                "Use real paths from the artifact inventory; do not use placeholders. "
                "Be concise and avoid assuming success without evidence. "
                f"Keep the response under {tier.max_tokens or settings.verifier_max_tokens} tokens."
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
                f"AUTO_DECODE_HINT: {auto_decode_hint}\n"
                f"FLAGS: {state.get('flag_candidates')}\n"
                f"ARTIFACT INVENTORY: {_format_inventory_for_prompt(state.get('artifact_inventory', []))}\n"
            ))

            response = _invoke_with_fallback(
                "verifier",
                [system_msg, human_msg],
                "verify",
                verifier_llm,
                tier,
                tier_idx,
            )
            verifier_preview = preview_text(response.content or "", 220)
            if auto_decode and auto_decode_hint != "none":
                if verifier_preview:
                    verifier_preview = f"{verifier_preview} | AutoDecode: {auto_decode_hint}"
                else:
                    verifier_preview = f"AutoDecode: {auto_decode_hint}"
            updates["verifier_hint"] = verifier_preview
            updates.setdefault("messages", []).append(response)
            updates.setdefault("reasoning_log", []).append(f"Verifier: {response.content[:200]}")
            if trajectory_logger:
                trajectory_logger.log("verify", {"reflection": response.content[:200]})
            return updates
        except Exception as exc:
            _log_event(state, "node_error", node="verify", error=repr(exc))
            raise
        finally:
            _log_event(state, "node_end", node="verify", duration_s=round(time.monotonic() - node_start, 3))

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
        "foremost": toolbox.foremost,
        "tshark": toolbox.tshark,
        "yara": toolbox.yara,
        "pdfinfo": toolbox.pdfinfo,
        "pdftotext": toolbox.pdftotext,
        "qpdf": toolbox.qpdf,
        "pdf_parser": toolbox.pdf_parser,
        "stegseek": toolbox.stegseek,
        "zsteg": toolbox.zsteg,
        "zbarimg": toolbox.zbarimg,
        "qrencode": toolbox.qrencode,
        "apktool": toolbox.apktool,
        "jadx": toolbox.jadx,
        "aapt": toolbox.aapt,
        "dex2jar": toolbox.dex2jar,
        "ciphey": toolbox.ciphey,
        "hashcat": toolbox.hashcat,
        "john": toolbox.john,
        "hashid": toolbox.hashid,
        "name_that_hash": toolbox.name_that_hash,
        "hashdeep": toolbox.hashdeep,
        "ghidra_headless": toolbox.ghidra_headless,
        "ropgadget": toolbox.ropgadget,
        "pwninit": toolbox.pwninit,
        "one_gadget": toolbox.one_gadget,
        "radare2_json": toolbox.radare2_json,
        "objdump": toolbox.objdump,
        "readelf": toolbox.readelf,
        "slither": toolbox.slither,
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
        "foremost": ["path"],
        "tshark": ["path"],
        "yara": ["rule_path", "target_path"],
        "pdfinfo": ["path"],
        "pdftotext": ["path"],
        "qpdf": ["path"],
        "pdf_parser": ["path"],
        "stegseek": ["path"],
        "zsteg": ["path"],
        "zbarimg": ["path"],
        "qrencode": ["value"],
        "apktool": ["path"],
        "jadx": ["path"],
        "aapt": ["path"],
        "dex2jar": ["path"],
        "ciphey": ["text_or_path"],
        "ghidra_headless": ["project_dir", "project_name", "binary_path"],
        "hashcat": ["hash_file"],
        "john": ["hash_file"],
        "hashid": ["value_or_path"],
        "name_that_hash": ["value_or_path"],
        "hashdeep": ["path"],
        "radare2_json": ["binary_path", "commands"],
        "objdump": ["binary_path"],
        "readelf": ["binary_path"],
        "ropgadget": ["binary_path"],
        "one_gadget": ["path"],
        "slither": ["target"],
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
        "path": ["file", "filepath", "file_path", "filename", "target", "input", "input_path"],
        "binary_path": ["binary", "bin", "exe", "elf", "program"],
        "command": ["cmd", "shell"],
        "pattern": ["regex", "query", "search"],
        "wordlist": ["wordlist_path", "wordlist_file", "wl"],
        "dest_dir": ["dest", "out", "output", "output_dir"],
        "out_dir": ["dest_dir", "output_dir", "out_dir", "out", "output"],
        "out_path": ["output_path", "out_path", "out_file", "output_file"],
        "hash_file": ["hashes", "hash_path", "hashfile"],
        "rule_path": ["rule", "rules", "yara_rule"],
        "target_path": ["target_path", "target", "target_file", "path"],
        "value_or_path": ["value", "text", "data", "hash", "path"],
        "target": ["contract", "project", "path"],
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

    if tool_name in {"hashcat", "john"} and "hash_file" not in args and "path" in args:
        args["hash_file"] = args["path"]
    if tool_name in {"hashid", "name_that_hash"} and "value_or_path" not in args:
        for key in ("value", "path", "hash"):
            if key in args:
                args["value_or_path"] = args[key]
                break
    if tool_name == "yara":
        if "rule_path" not in args and "rule" in args:
            args["rule_path"] = args["rule"]
        if "target_path" not in args and "path" in args:
            args["target_path"] = args["path"]
    if tool_name == "dex2jar" and "out_path" not in args and "out_dir" in args:
        args["out_path"] = args["out_dir"]

    if tool_name == "extract_archive":
        path = args.get("path")
        if path:
            default_dest = f"{path}_extracted"
            dest = args.get("dest_dir")
            if not dest or dest in {"extracted", "extract", "output", "out"}:
                args["dest_dir"] = default_dest

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
        "foremost": {"path", "out_dir", "args"},
        "tshark": {"path", "args"},
        "yara": {"rule_path", "target_path", "args"},
        "pdfinfo": {"path"},
        "pdftotext": {"path", "out_path", "args"},
        "qpdf": {"path", "out_path", "args"},
        "pdf_parser": {"path", "args"},
        "stegseek": {"path", "wordlist"},
        "zsteg": {"path"},
        "zbarimg": {"path", "args"},
        "qrencode": {"value", "out_path", "args"},
        "apktool": {"path", "out_dir", "args"},
        "jadx": {"path", "out_dir", "args"},
        "aapt": {"path", "args"},
        "dex2jar": {"path", "out_path", "args"},
        "ghidra_headless": {"project_dir", "project_name", "binary_path"},
        "radare2_json": {"binary_path", "commands"},
        "objdump": {"binary_path", "args"},
        "readelf": {"binary_path", "args"},
        "hashcat": {"hash_file", "wordlist", "args"},
        "john": {"hash_file", "args"},
        "hashid": {"value_or_path", "args"},
        "name_that_hash": {"value_or_path", "args"},
        "hashdeep": {"path", "args"},
        "ropgadget": {"binary_path", "args"},
        "pwninit": {"binary_path", "args"},
        "one_gadget": {"path", "args"},
        "slither": {"target", "args"},
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


def _resolve_artifact_paths(args: dict, inventory: list[dict]) -> dict:
    if not isinstance(args, dict) or not inventory:
        return args

    paths = [entry.get("path") for entry in inventory if isinstance(entry, dict) and entry.get("path")]
    if not paths:
        return args

    name_to_paths: dict[str, list[str]] = {}
    for path in paths:
        name = Path(path).name
        name_to_paths.setdefault(name, []).append(path)
    name_map = {name: vals[0] for name, vals in name_to_paths.items() if len(vals) == 1}

    def resolve_value(value):
        if not isinstance(value, str):
            return value
        stripped = value.strip().strip("\"'")
        if not stripped:
            return value
        lowered = stripped.lower()
        if any(marker in lowered for marker in _PATH_PLACEHOLDER_MARKERS):
            base = Path(stripped).name
            if base in name_map:
                return name_map[base]
            if len(name_map) == 1:
                return next(iter(name_map.values()))
            return stripped
        if stripped.startswith("/"):
            if stripped not in paths:
                base = Path(stripped).name
                if base in name_map:
                    return name_map[base]
            return stripped
        base = Path(stripped).name
        if base in name_map:
            return name_map[base]
        return stripped

    path_keys = {
        "path",
        "binary_path",
        "text_or_path",
        "rule_path",
        "target_path",
        "hash_file",
        "wordlist",
        "value_or_path",
        "project_dir",
    }
    for key in path_keys:
        if key in args:
            val = args[key]
            if isinstance(val, list):
                args[key] = [resolve_value(item) for item in val]
            else:
                args[key] = resolve_value(val)
    return args


def _filter_args(args: dict, allowed: set[str]) -> dict:
    if not allowed:
        return args
    return {key: value for key, value in args.items() if key in allowed}


def _resolve_tool_manual_key_for_call(tool_name: str, args: dict) -> str | None:
    manual_key = resolve_tool_manual_key(tool_name)
    if manual_key:
        return manual_key
    if tool_name == "bash":
        cmd = (args or {}).get("command", "")
        head = _extract_first_command_token(cmd)
        if head:
            return resolve_tool_manual_key(head)
    return None


def _extract_first_command_token(command: str) -> str:
    if not command:
        return ""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return ""
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"sudo", "env"}:
            idx += 1
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            idx += 1
            continue
        return token
    return ""


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
    banned = ["nmap", "masscan", "zmap", "rustscan"]
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
        "xargs -0 -I{} sh -c '"
        "path=\"{}\"; "
        "echo \"=== $path\"; "
        "echo TYPE: $(file -b \"$path\"); "
        "echo MIME: $(file -bi \"$path\"); "
        "size=$(stat -c %s \"$path\" 2>/dev/null || wc -c < \"$path\"); "
        "echo SIZE: $size; "
        "echo STRINGS:; "
        "strings -n 4 \"$path\" 2>/dev/null | head -n 20 | sed \"s/^/STR: /\""
        "'"
    )


def _summarize_file_inventory(stdout: str) -> tuple[str, list[dict]]:
    inventory: list[dict] = []
    current: dict | None = None
    for line in stdout.splitlines():
        if line.startswith("==="):
            if current:
                inventory.append(current)
            current = {
                "path": line.replace("===", "").strip(),
                "type": "",
                "mime": "",
                "encoding": "",
                "size": "",
                "strings_sample": [],
            }
            continue
        if not current:
            continue
        if line.startswith("TYPE:"):
            current["type"] = line.replace("TYPE:", "", 1).strip()
            continue
        if line.startswith("MIME:"):
            mime = line.replace("MIME:", "", 1).strip()
            current["mime"] = mime
            match = re.search(r"charset=([^;\\s]+)", mime)
            if match:
                current["encoding"] = match.group(1)
            continue
        if line.startswith("SIZE:"):
            current["size"] = line.replace("SIZE:", "", 1).strip()
            continue
        if line.startswith("STR:"):
            sample = line.replace("STR:", "", 1).strip()
            if len(sample) > 120:
                sample = sample[:120] + "...(truncated)"
            if len(current["strings_sample"]) < 3:
                current["strings_sample"].append(sample)
            continue
    if current:
        inventory.append(current)
    type_counts: dict[str, int] = {}
    for entry in inventory:
        ftype = entry.get("type") or entry.get("mime") or "unknown"
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
    summary_parts = [f"{t} x{c}" for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])][:5]
    summary = "Files analyzed: " + str(len(inventory))
    if summary_parts:
        summary += f" ({', '.join(summary_parts)})"
    if inventory:
        detail_lines = []
        for entry in inventory[:5]:
            name = Path(entry.get("path", "")).name
            parts = []
            if entry.get("type"):
                parts.append(entry["type"])
            if entry.get("mime"):
                parts.append(entry["mime"])
            if entry.get("strings_sample"):
                parts.append("strings: " + "; ".join(entry["strings_sample"]))
            if parts:
                detail_lines.append(f"{name}: " + " | ".join(parts))
        if detail_lines:
            summary += "\n" + "\n".join(detail_lines)
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

def _join_commands(value) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " && ".join(parts)
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_bash_args(args: dict, data: dict) -> tuple[dict, str]:
    if "command" in args:
        return args, str(args.get("command", "") or "")
    if "commands" in args:
        cmd = _join_commands(args.get("commands"))
        if cmd:
            return {"command": cmd}, cmd
    for key in ("command", "cmd", "shell"):
        if key in data and isinstance(data.get(key), str):
            return {"command": data.get(key)}, str(data.get(key))
    if "commands" in data:
        cmd = _join_commands(data.get("commands"))
        if cmd:
            return {"command": cmd}, cmd
    return args, str(data.get("command") or "")


def _normalize_tool_name(tool_name: str) -> str:
    if not tool_name:
        return tool_name
    stripped = tool_name.strip()
    if stripped in _TOOL_ALIASES:
        return _TOOL_ALIASES[stripped]
    lowered = stripped.lower()
    return _TOOL_ALIASES.get(lowered, stripped)


def _parse_tool_call(text: str) -> tuple[str, dict, str]:
    candidates: list[str] = []
    json_block = _extract_json_block(text)
    if json_block:
        candidates.append(json_block)
    fallback_block = _extract_first_json_object(text)
    if fallback_block and fallback_block not in candidates:
        candidates.append(fallback_block)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        tool = str(data.get("tool", "") or "").strip()
        args = data.get("args") or {}
        if tool == "bash":
            args, cmd = _coerce_bash_args(args, data)
            return tool, args, cmd
        return tool, args, str(data.get("command") or "")

    cmd = _extract_bash_command(text)
    if cmd:
        return "bash", {"command": cmd}, cmd
    return "bash", {"command": text.strip()}, text.strip()


def _extract_json_block(text: str) -> str | None:
    for pattern in (
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return _extract_first_json_object(text)


def _extract_first_json_object(text: str) -> str | None:
    if not text:
        return None
    start_idx = None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    return text[start_idx:idx + 1]
    return None


def _extract_bash_command(text: str) -> str | None:
    match = re.search(r"```(?:bash|sh|shell)\s*(.*?)\s*```", text, re.DOTALL)
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


def _recent_tool_guard(state: DCipherState, inventory: list[dict], window: int = 3) -> tuple[list[str], list[str]]:
    attempts = state.get("attempt_history") or []
    recent = attempts[-window:]
    tools = [attempt.get("tool") for attempt in recent if attempt.get("tool")]
    avoid: list[str] = []
    counts = Counter(tools)
    for tool, count in counts.items():
        if count >= 2:
            avoid.append(tool)
    if state.get("triage_done") and inventory:
        avoid.append("list_dir")
    return _dedupe_list(avoid), tools


def _should_block_redundant_tool(tool_name: str, args: dict, avoid_tools: list[str]) -> bool:
    if tool_name not in avoid_tools:
        return False
    if tool_name == "list_dir":
        path = (args or {}).get("path") or "."
        return str(path).strip() in {"", ".", "./"}
    return False


def _extract_path_from_command(command: str) -> str:
    if not command:
        return ""
    quoted = re.findall(r"['\"](/[^'\"]+)['\"]", command)
    if quoted:
        return quoted[-1]
    bare = re.findall(r"(/[^\\s]+)", command)
    return bare[-1] if bare else ""


def _fallback_tool_for_failure(
    state: DCipherState,
    inventory: list[dict],
    avoid_tools: list[str],
) -> dict | None:
    if state.get("last_exit_code", 0) == 0:
        return None
    attempts = state.get("attempt_history") or []
    last_attempt = attempts[-1] if attempts else {}
    if last_attempt.get("phase") != "execute":
        return None
    last_tool = str(last_attempt.get("tool") or "")
    if not last_tool or last_tool in {"bash", "artifacts"}:
        return None

    fallback_map = {
        "strings": ["file_info", "read_file"],
        "file_info": ["strings", "read_file"],
        "read_file": ["strings", "file_info"],
        "binwalk": ["strings", "file_info"],
        "exiftool": ["strings", "binwalk"],
        "pdfinfo": ["pdftotext", "strings"],
        "pdftotext": ["pdfinfo", "strings"],
        "checksec": ["strings", "file_info"],
    }
    options = fallback_map.get(last_tool, [])
    if not options:
        return None

    command = str(last_attempt.get("command") or state.get("last_command") or "")
    path = _extract_path_from_command(command)
    if not path and inventory:
        paths = [entry.get("path") for entry in inventory if isinstance(entry, dict) and entry.get("path")]
        if len(paths) == 1:
            path = paths[0]
    if not path:
        return None

    for candidate in options:
        if candidate == last_tool:
            continue
        if candidate in avoid_tools:
            continue
        return {
            "tool": candidate,
            "args": {"path": path},
            "reason": f"Fallback after {last_tool} failure.",
        }
    return None


def _find_candidate(
    pattern: re.Pattern,
    text: str,
    *,
    prefer_flag: bool = True,
    even_length: bool = False,
    mod4: bool = False,
    max_len: int = 4096,
) -> str | None:
    if not text:
        return None

    def valid(value: str) -> bool:
        if max_len and len(value) > max_len:
            return False
        if even_length and len(value) % 2 != 0:
            return False
        if mod4 and len(value) % 4 != 0:
            return False
        return True

    if prefer_flag:
        for line in text.splitlines():
            if "flag" in line.lower():
                for match in pattern.finditer(line):
                    value = match.group(0)
                    if valid(value):
                        return value

    candidates = []
    for match in pattern.finditer(text):
        value = match.group(0)
        if valid(value):
            candidates.append(value)
    if not candidates:
        return None
    return max(candidates, key=len)


def _auto_decode_candidate(text: str) -> dict | None:
    if not text:
        return None
    hex_value = _find_candidate(HEX_CANDIDATE_RE, text, even_length=True)
    if hex_value:
        return {
            "encoding": "hex",
            "value": hex_value,
            "reason": "Detected hex string in output.",
        }
    b64_value = _find_candidate(BASE64_CANDIDATE_RE, text, mod4=True)
    if b64_value:
        return {
            "encoding": "base64",
            "value": b64_value,
            "reason": "Detected base64 string in output.",
        }
    return None


def _auto_decode_suggestion(state: DCipherState) -> dict | None:
    suggestion = _auto_decode_candidate(state.get("last_output", ""))
    if not suggestion:
        return None
    last_decode = state.get("last_decode") or {}
    if (
        last_decode.get("value") == suggestion.get("value")
        and last_decode.get("encoding") == suggestion.get("encoding")
    ):
        return None
    return suggestion


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
