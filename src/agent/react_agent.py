from __future__ import annotations

import json
import shlex
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from src.config import settings
from src.state import DCipherState
from src.tools.toolbox import Toolbox, ToolResult, make_bash_tool, make_submit_flag_tool
from src.trajectory import TrajectoryLogger
from src.agent.agent import (
    _extract_flags_with_evidence,
    _init_llm,
    _is_disallowed_command,
    _mark_done,
    _new_flag_hits,
    _persist_tool_output,
    _violates_url_scope,
)


COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_WHITE = "\033[97m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}\033[0m"


def _live(text: str) -> None:
    print(text, flush=True)


SYSTEM_PROMPT = (
    "You are a CTF solver with access to a Kali Linux shell via a tool named 'bash'.\n"
    "Rules:\n"
    "- Run commands immediately without explanation.\n"
    "- Never ask for clarification.\n"
    "- If stuck after 3 attempts on one approach, change strategy completely.\n"
    "- All files are in /challenge/ inside the sandbox.\n"
    "- Use the bash tool for all shell commands.\n"
    "- If a URL is provided, interact only with that base URL.\n"
    "- Do not use port scanners or mass scanners (nmap, masscan, zmap, rustscan).\n"
    "- Submit the flag immediately using submit_flag when found.\n"
    "Methodology by category:\n"
    "- crypto: identify encoding/cipher, try base64/hex/rot, frequency, known-plaintext; use ciphey/python.\n"
    "- forensics: file/strings/binwalk/exiftool/foremost/tshark to extract hidden data.\n"
    "- stego: exiftool/binwalk/steghide/stegseek/zsteg, check LSB and embedded files.\n"
    "- rev: file/strings/objdump/r2, then ghidra analyzeHeadless for decompilation.\n"
    "- pwn: file/checksec/strings/ldd, then gdb+pwndbg and pwntools exploit.\n"
    "- web: curl/view source, enumerate paths within base URL; ffuf/sqlmap only within scope.\n"
    "Available tools include: python3, pwntools, analyzeHeadless (Ghidra), r2, gdb+pwndbg, checksec, "
    "binwalk, steghide, stegseek, zsteg, exiftool, tshark, foremost, hashcat, john, RsaCtfTool, ciphey, "
    "ffuf, sqlmap, strings, objdump, file.\n"
    "CRITICAL: You MUST always respond with a tool call. NEVER output plain text. "
    "If you see the answer, call submit_flag immediately. "
    "If you need more work, call bash. "
    "A text-only response means you failed.\n"
)


def _log_trajectory(trajectory_logger: TrajectoryLogger | None, event: str, payload: dict) -> None:
    if trajectory_logger and settings.debug:
        trajectory_logger.log(event, payload)


def _response_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("text"):
                    parts.append(str(block.get("text")))
                elif block.get("refusal"):
                    parts.append(str(block.get("refusal")))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join([part for part in parts if part])
    return str(content) if content is not None else ""


def _head(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:limit] if len(text) > limit else text


def _inline_preview(text: str, limit: int, replace_newlines: bool = False) -> str:
    preview = _head(text or "", limit)
    if replace_newlines:
        preview = preview.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ↵ ")
    return preview


def _truncate_command(command: str, limit: int = 120) -> str:
    if not command:
        return ""
    if len(command) <= limit:
        return command
    if limit <= 3:
        return command[:limit]
    return command[:limit - 3] + "..."


def _budget_reason(tool_calls: int, iteration: int, started_at: float) -> str | None:
    elapsed = time.monotonic() - started_at
    if elapsed > settings.max_wall_seconds_per_challenge:
        return f"wall time {elapsed:.1f}s > {settings.max_wall_seconds_per_challenge}s"
    if tool_calls >= settings.max_tool_calls:
        return f"tool calls {tool_calls} >= {settings.max_tool_calls}"
    if iteration >= settings.max_iterations:
        return f"iterations {iteration} >= {settings.max_iterations}"
    return None


def _tool_call_fields(call: Any) -> tuple[str, dict, str | None]:
    name = ""
    args: dict | str | None = None
    call_id = None
    if isinstance(call, dict):
        name = str(call.get("name") or call.get("tool") or "")
        args = call.get("args") or call.get("arguments") or {}
        call_id = call.get("id") or call.get("tool_call_id")
    else:
        name = str(getattr(call, "name", "") or getattr(call, "tool", "") or "")
        args = getattr(call, "args", None) or getattr(call, "arguments", None) or {}
        call_id = getattr(call, "id", None) or getattr(call, "tool_call_id", None)

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"command": args} if name == "bash" else {}

    if not isinstance(args, dict):
        args = {}

    return name, args, call_id


def _append_attempt(
    state: DCipherState,
    attempt_history: list[dict],
    tool_result: ToolResult,
    phase: str,
    log_path: str,
) -> dict:
    attempt = {
        "step": len(attempt_history) + 1,
        "phase": phase,
        "objective": state.get("current_objective", ""),
        "tool": tool_result.tool,
        "command": tool_result.command,
        "stdout": tool_result.stdout,
        "stderr": tool_result.stderr,
        "exit_code": tool_result.exit_code,
        "log_path": log_path,
    }
    attempt_history.append(attempt)
    state["last_command"] = tool_result.command
    state["last_output"] = tool_result.stdout
    state["last_error"] = tool_result.stderr
    state["last_exit_code"] = tool_result.exit_code
    state["last_log_path"] = log_path
    if tool_result.parsed and "files" in tool_result.parsed:
        state["current_files"] = tool_result.parsed["files"]
    return attempt


def _truncate_tail(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[-limit:] if len(text) > limit else text


def _format_tool_message(result: ToolResult) -> str:
    stdout = _truncate_tail(result.stdout or "", 6000)
    stderr = _truncate_tail(result.stderr or "", 1000)
    return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}\nEXIT_CODE: {result.exit_code}"


def _file_report_command(container_dir: str) -> str:
    escaped = shlex.quote(container_dir)
    return f"find {escaped} -maxdepth 1 -type f -print0 | xargs -0 -r file"


def _initial_prompt(state: DCipherState, file_report: str) -> str:
    flag_format = state.get("flag_format") or "unknown"
    url = state.get("url") or ""
    url_line = f"URL: {url}\n" if url else ""
    container_dir = state.get("container_dir") or settings.sandbox_workdir
    description = (state.get("description") or "").strip()
    return (
        f"CHALLENGE: {state.get('challenge_name', '')}\n"
        f"CATEGORY: {state.get('category', '')}\n"
        f"DESCRIPTION:\n{description}\n\n"
        f"{url_line}"
        f"FILES (file output from {container_dir}):\n{file_report or 'None'}\n"
        f"FLAG FORMAT: {flag_format}\n"
        "Start solving immediately."
    )


def build_react_graph(toolbox: Toolbox, connector=None, trajectory_logger: TrajectoryLogger | None = None):
    bash_tool = make_bash_tool(toolbox)
    submit_flag_tool = make_submit_flag_tool()
    tools = [bash_tool, submit_flag_tool]

    def agent_node(state: DCipherState):
        if state.get("done"):
            return {"done": True}

        messages = list(state.get("messages", []))
        attempt_history = list(state.get("attempt_history", []))
        flag_hits = list(state.get("flag_hits", []))
        reasoning_log = list(state.get("reasoning_log", []))
        command_history = list(state.get("command_history", []))

        base_message_len = len(messages)
        base_attempt_len = len(attempt_history)
        base_flag_hit_len = len(flag_hits)
        base_reason_len = len(reasoning_log)
        base_command_len = len(command_history)

        tool_calls = int(state.get("tool_calls", 0))
        iteration = int(state.get("iteration", 0))
        started_at = float(state.get("started_at", time.monotonic()))
        flag_candidates = list(state.get("flag_candidates", []))
        submitted_flags = list(state.get("submitted_flags", []))
        last_command = str(state.get("last_command") or "")
        last_output = str(state.get("last_output") or "")
        last_error = str(state.get("last_error") or "")
        last_exit_code = int(state.get("last_exit_code", 0))
        last_log_path = str(state.get("last_log_path") or "")
        current_files = list(state.get("current_files", []))
        done = False
        done_reason: str | None = None
        nudge_count = 0

        challenge_label = state.get("challenge_name") or str(state.get("challenge_id") or "unknown")
        _live(_c(
            COLOR_CYAN,
            (
                f"[REACT] Challenge: {challenge_label} | Model: {settings.executor_model} | "
                f"Budget: {settings.max_tool_calls} calls / {settings.max_wall_seconds_per_challenge}s"
            ),
        ))
        _log_trajectory(
            trajectory_logger,
            "react_start",
            {
                "challenge_id": state.get("challenge_id"),
                "model": settings.executor_model,
                "max_tool_calls": settings.max_tool_calls,
                "max_iterations": settings.max_iterations,
            },
        )

        def _emit_done(reason: str) -> None:
            elapsed = time.monotonic() - started_at
            _log_trajectory(
                trajectory_logger,
                "react_done",
                {
                    "reason": reason,
                    "tool_calls": tool_calls,
                    "iteration": iteration,
                    "duration_s": round(elapsed, 3),
                },
            )
            _live(_c(
                COLOR_CYAN,
                f"[DONE] {reason} | tool_calls={tool_calls} | iterations={iteration} | elapsed={elapsed:.1f}s",
            ))

        budget_reason = _budget_reason(tool_calls, iteration, started_at)
        if budget_reason:
            reasoning_log.append(f"Budget exceeded: {budget_reason}")
            _log_trajectory(trajectory_logger, "budget_exceeded", {"reason": budget_reason})
            _live(_c(COLOR_RED, f"[BUDGET] {budget_reason}"))
            done_reason = "budget"
            _emit_done(done_reason)
            updates = _mark_done(
                state,
                f"Budget exceeded: {budget_reason}",
                trajectory_logger if settings.debug else None,
            )
            updates.update({
                "messages": messages[base_message_len:],
                "attempt_history": attempt_history[base_attempt_len:],
                "flag_hits": flag_hits[base_flag_hit_len:],
                "reasoning_log": reasoning_log[base_reason_len:],
                "command_history": command_history[base_command_len:],
                "flag_candidates": flag_candidates,
                "submitted_flags": submitted_flags,
                "tool_calls": tool_calls,
                "iteration": iteration,
                "last_command": last_command,
                "last_output": last_output,
                "last_error": last_error,
                "last_exit_code": last_exit_code,
                "last_log_path": last_log_path,
                "current_files": current_files,
            })
            return updates

        file_report = ""
        if not messages:
            container_dir = state.get("container_dir") or settings.sandbox_workdir
            command = _file_report_command(container_dir)
            tool_calls += 1
            if tool_calls > settings.max_tool_calls:
                reasoning_log.append(
                    f"Budget exceeded: tool calls {tool_calls} > {settings.max_tool_calls}"
                )
                budget_reason = f"tool calls {tool_calls} > {settings.max_tool_calls}"
                _log_trajectory(trajectory_logger, "budget_exceeded", {"reason": budget_reason})
                _live(_c(COLOR_RED, f"[BUDGET] {budget_reason}"))
                done_reason = "budget"
                _emit_done(done_reason)
                updates = _mark_done(
                    state,
                    f"Budget exceeded: tool calls {tool_calls}",
                    trajectory_logger if settings.debug else None,
                )
                updates.update({
                    "messages": messages[base_message_len:],
                    "attempt_history": attempt_history[base_attempt_len:],
                    "flag_hits": flag_hits[base_flag_hit_len:],
                    "reasoning_log": reasoning_log[base_reason_len:],
                    "command_history": command_history[base_command_len:],
                    "flag_candidates": flag_candidates,
                    "submitted_flags": submitted_flags,
                    "tool_calls": tool_calls,
                    "iteration": iteration,
                    "last_command": last_command,
                    "last_output": last_output,
                    "last_error": last_error,
                    "last_exit_code": last_exit_code,
                    "last_log_path": last_log_path,
                    "current_files": current_files,
                })
                return updates

            _log_trajectory(
                trajectory_logger,
                "tool_start",
                {"tool": "bash", "command": command, "iteration": iteration},
            )
            _live(
                f"{_c(COLOR_YELLOW, f'[BASH #{tool_calls}] $ ')}"
                f"{_c(COLOR_WHITE, _truncate_command(command))}"
            )
            tool_started = time.monotonic()
            result = toolbox.run(command, tool_name="bash")
            tool_duration = time.monotonic() - tool_started
            log_path = _persist_tool_output(state, result, phase="triage")
            _append_attempt(state, attempt_history, result, "triage", log_path)
            last_command = result.command
            last_output = result.stdout
            last_error = result.stderr
            last_exit_code = result.exit_code
            last_log_path = log_path
            if result.parsed and "files" in result.parsed:
                current_files = list(result.parsed["files"])

            command_history.append({
                "cmd": result.command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "tool": result.tool,
                "log_path": log_path,
            })

            _log_trajectory(
                trajectory_logger,
                "tool_end",
                {
                    "tool": result.tool,
                    "exit_code": result.exit_code,
                    "stdout_head": _head(result.stdout or "", 200),
                    "stderr_head": _head(result.stderr or "", 100),
                    "duration_s": round(tool_duration, 3),
                },
            )

            if result.exit_code == 0:
                stdout_preview = _inline_preview(result.stdout or "", 200, replace_newlines=True)
                _live(_c(COLOR_GREEN, f"[OK] exit=0 | {stdout_preview}"))
            else:
                stderr_preview = _inline_preview(result.stderr or "", 150, replace_newlines=True)
                _live(_c(COLOR_RED, f"[FAIL] exit={result.exit_code} | stderr: {stderr_preview}"))

            _log_trajectory(
                trajectory_logger,
                "execute",
                {
                    "tool": result.tool,
                    "command": result.command,
                    "log_path": log_path,
                    "exit_code": result.exit_code,
                },
            )

            file_report = (result.stdout or "").strip()
            if not file_report:
                err_text = (result.stderr or "").strip()
                file_report = f"ERROR: {err_text}" if err_text else "None"

            hits = _extract_flags_with_evidence(
                result.stdout,
                result.stderr,
                state.get("flag_format", ""),
                log_path,
            )
            new_hits = _new_flag_hits(flag_hits, hits, state)
            if new_hits:
                flag_hits.extend(new_hits)
                for hit in new_hits:
                    flag = hit.get("flag")
                    source = hit.get("evidence_type") or hit.get("source") or ""
                    if flag:
                        _log_trajectory(
                            trajectory_logger,
                            "flag_autodetect",
                            {"flag": flag, "source": source},
                        )
                        _live(_c(COLOR_MAGENTA, f"[FLAG FOUND] {flag}"))
                    if flag and flag not in flag_candidates:
                        flag_candidates.append(flag)
                    if connector and flag and flag not in submitted_flags:
                        try:
                            result = connector.submit_flag(int(state.get("challenge_id")), flag)
                            status = result.get("data", {}).get("status") or result.get("message") or str(result)
                            _live(_c("\033[93m", f"[CTFd] Response: {status}"))
                            submitted_flags.append(flag)
                        except Exception as exc:
                            reasoning_log.append(f"Flag submission error: {exc}")
                done = True
                done_reason = "flag_found"

            if done:
                if done_reason:
                    _emit_done(done_reason)
                return {
                    "messages": messages[base_message_len:],
                    "attempt_history": attempt_history[base_attempt_len:],
                    "flag_hits": flag_hits[base_flag_hit_len:],
                    "reasoning_log": reasoning_log[base_reason_len:],
                    "command_history": command_history[base_command_len:],
                    "flag_candidates": flag_candidates,
                    "submitted_flags": submitted_flags,
                    "tool_calls": tool_calls,
                    "iteration": iteration,
                    "last_command": last_command,
                    "last_output": last_output,
                    "last_error": last_error,
                    "last_exit_code": last_exit_code,
                    "last_log_path": last_log_path,
                    "current_files": current_files,
                    "done": True,
                }

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_initial_prompt(state, file_report)),
            ]

        llm = _init_llm(
            settings.executor_provider,
            settings.executor_model,
            role="executor",
            max_tokens=settings.executor_max_tokens,
        )
        llm_with_tools = llm.bind_tools(tools)

        while True:
            budget_reason = _budget_reason(tool_calls, iteration, started_at)
            if budget_reason:
                reasoning_log.append(f"Budget exceeded: {budget_reason}")
                _log_trajectory(trajectory_logger, "budget_exceeded", {"reason": budget_reason})
                _live(_c(COLOR_RED, f"[BUDGET] {budget_reason}"))
                done_reason = "budget"
                _emit_done(done_reason)
                updates = _mark_done(
                    state,
                    f"Budget exceeded: {budget_reason}",
                    trajectory_logger if settings.debug else None,
                )
                updates.update({
                    "messages": messages[base_message_len:],
                    "attempt_history": attempt_history[base_attempt_len:],
                    "flag_hits": flag_hits[base_flag_hit_len:],
                    "reasoning_log": reasoning_log[base_reason_len:],
                    "command_history": command_history[base_command_len:],
                    "flag_candidates": flag_candidates,
                    "submitted_flags": submitted_flags,
                    "tool_calls": tool_calls,
                    "iteration": iteration,
                    "last_command": last_command,
                    "last_output": last_output,
                    "last_error": last_error,
                    "last_exit_code": last_exit_code,
                    "last_log_path": last_log_path,
                    "current_files": current_files,
                })
                return updates

            llm_iteration = iteration + 1
            _log_trajectory(
                trajectory_logger,
                "llm_call_start",
                {
                    "role": "executor",
                    "model": settings.executor_model,
                    "iteration": llm_iteration,
                    "message_count": len(messages),
                },
            )
            _live(_c(COLOR_YELLOW, f"[LLM #{llm_iteration}] Calling {settings.executor_model}..."))
            llm_started = time.monotonic()
            response = llm_with_tools.invoke(messages)
            llm_duration = time.monotonic() - llm_started
            response_text = _response_text(response)
            tool_calls_data = getattr(response, "tool_calls", None) or []
            _log_trajectory(
                trajectory_logger,
                "llm_call_end",
                {
                    "duration_s": round(llm_duration, 3),
                    "response_chars": len(response_text or ""),
                    "had_tool_calls": bool(tool_calls_data),
                },
            )

            messages.append(response)
            iteration = llm_iteration

            if tool_calls_data:
                _live(_c(COLOR_YELLOW, f"[LLM #{llm_iteration}] \u2192 {len(tool_calls_data)} tool call(s)"))
            else:
                response_preview = _inline_preview(response_text or "", 150, replace_newlines=True)
                _live(_c(
                    COLOR_YELLOW,
                    f"[LLM #{llm_iteration}] \u26a0 No tool call \u2014 text: {response_preview}",
                ))

            if not tool_calls_data:
                # Model responded with text instead of a tool call — push it back
                reasoning_log.append("Model stopped without tool calls — nudging.")
                _log_trajectory(
                    trajectory_logger,
                    "llm_no_tool_calls",
                    {
                        "iteration": llm_iteration,
                        "response_preview": _inline_preview(response_text or "", 200),
                    },
                )
                nudge_count += 1
                if nudge_count >= 3:
                    done = True
                    done_reason = "nudge_limit"
                    break
                messages.append(HumanMessage(
                    content="You must call a tool. Use bash to continue, or submit_flag if you have the answer."
                ))
                _log_trajectory(
                    trajectory_logger,
                    "llm_nudge",
                    {"nudge_count": nudge_count},
                )
                _live(_c(COLOR_YELLOW, f"[NUDGE #{nudge_count}] Pushing model back to tool use..."))
                continue  # Loop again instead of breaking

            nudge_count = 0

            for call in tool_calls_data:
                tool_name, args, call_id = _tool_call_fields(call)
                call_id = call_id or tool_name or "tool"

                if tool_name == "submit_flag":
                    flag = str(args.get("flag", "") or "").strip()
                    if flag:
                        _log_trajectory(trajectory_logger, "flag_submit", {"flag": flag})
                        _live(_c(COLOR_MAGENTA, f"[SUBMIT] {flag}"))
                        if flag not in submitted_flags:
                            submitted_flags.append(flag)
                        if connector:
                            try:
                                connector.submit_flag(int(state.get("challenge_id")), flag)
                            except Exception as exc:
                                reasoning_log.append(f"Flag submission error: {exc}")
                        messages.append(ToolMessage(content=f"Submitted flag: {flag}", tool_call_id=call_id))
                        done = True
                        done_reason = "flag_found"
                        break

                    messages.append(ToolMessage(content="submit_flag called with empty flag.", tool_call_id=call_id))
                    continue

                if tool_name != "bash":
                    messages.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=call_id))
                    continue

                command = str(args.get("command", "") or "").strip()
                tool_calls += 1
                if tool_calls > settings.max_tool_calls:
                    reasoning_log.append(
                        f"Budget exceeded: tool calls {tool_calls} > {settings.max_tool_calls}"
                    )
                    budget_reason = f"tool calls {tool_calls} > {settings.max_tool_calls}"
                    _log_trajectory(trajectory_logger, "budget_exceeded", {"reason": budget_reason})
                    _live(_c(COLOR_RED, f"[BUDGET] {budget_reason}"))
                    done = True
                    done_reason = "budget"
                    break

                _log_trajectory(
                    trajectory_logger,
                    "tool_start",
                    {"tool": "bash", "command": command, "iteration": iteration},
                )
                _live(
                    f"{_c(COLOR_YELLOW, f'[BASH #{tool_calls}] $ ')}"
                    f"{_c(COLOR_WHITE, _truncate_command(command))}"
                )

                tool_duration = 0.0
                if not command:
                    result = ToolResult(
                        tool="bash",
                        command="",
                        stdout="",
                        stderr="Empty command.",
                        exit_code=2,
                    )
                elif _is_disallowed_command(command):
                    result = ToolResult(
                        tool="bash",
                        command=command,
                        stdout="",
                        stderr="Disallowed command (port scanning or prohibited tooling).",
                        exit_code=2,
                    )
                elif state.get("url") and _violates_url_scope(command, state.get("url", "")):
                    result = ToolResult(
                        tool="bash",
                        command=command,
                        stdout="",
                        stderr="Command violates URL scope. Use only the provided base URL.",
                        exit_code=2,
                    )
                else:
                    tool_started = time.monotonic()
                    result = toolbox.run(command, tool_name="bash")
                    tool_duration = time.monotonic() - tool_started

                log_path = _persist_tool_output(state, result, phase="execute")
                formatted = _format_tool_message(result)
                _append_attempt(state, attempt_history, result, "execute", log_path)
                last_command = result.command
                last_output = result.stdout
                last_error = result.stderr
                last_exit_code = result.exit_code
                last_log_path = log_path
                if result.parsed and "files" in result.parsed:
                    current_files = list(result.parsed["files"])

                command_history.append({
                    "cmd": result.command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "tool": result.tool,
                    "log_path": log_path,
                })

                _log_trajectory(
                    trajectory_logger,
                    "tool_end",
                    {
                        "tool": result.tool,
                        "exit_code": result.exit_code,
                        "stdout_head": _head(result.stdout or "", 200),
                        "stderr_head": _head(result.stderr or "", 100),
                        "duration_s": round(tool_duration, 3),
                    },
                )

                if result.exit_code == 0:
                    stdout_preview = _inline_preview(result.stdout or "", 200, replace_newlines=True)
                    _live(_c(COLOR_GREEN, f"[OK] exit=0 | {stdout_preview}"))
                else:
                    stderr_preview = _inline_preview(result.stderr or "", 150, replace_newlines=True)
                    _live(_c(COLOR_RED, f"[FAIL] exit={result.exit_code} | stderr: {stderr_preview}"))

                _log_trajectory(
                    trajectory_logger,
                    "execute",
                    {
                        "tool": result.tool,
                        "command": result.command,
                        "log_path": log_path,
                        "exit_code": result.exit_code,
                    },
                )

                hits = _extract_flags_with_evidence(
                    result.stdout,
                    result.stderr,
                    state.get("flag_format", ""),
                    log_path,
                )
                new_hits = _new_flag_hits(flag_hits, hits, state)
                if new_hits:
                    flag_hits.extend(new_hits)
                    for hit in new_hits:
                        flag = hit.get("flag")
                        source = hit.get("evidence_type") or hit.get("source") or ""
                        if flag:
                            _log_trajectory(
                                trajectory_logger,
                                "flag_autodetect",
                                {"flag": flag, "source": source},
                            )
                            _live(_c(COLOR_MAGENTA, f"[FLAG FOUND] {flag}"))
                        if flag and flag not in flag_candidates:
                            flag_candidates.append(flag)
                        if connector and flag and flag not in submitted_flags:
                            try:
                                connector.submit_flag(int(state.get("challenge_id")), flag)
                                submitted_flags.append(flag)
                            except Exception as exc:
                                reasoning_log.append(f"Flag submission error: {exc}")
                    done = True
                    done_reason = "flag_found"

                messages.append(ToolMessage(content=formatted, tool_call_id=call_id))
                if done:
                    break

            if done:
                break

        if done and done_reason:
            _emit_done(done_reason)

        return {
            "messages": messages[base_message_len:],
            "attempt_history": attempt_history[base_attempt_len:],
            "flag_hits": flag_hits[base_flag_hit_len:],
            "reasoning_log": reasoning_log[base_reason_len:],
            "command_history": command_history[base_command_len:],
            "flag_candidates": flag_candidates,
            "submitted_flags": submitted_flags,
            "tool_calls": tool_calls,
            "iteration": iteration,
            "last_command": last_command,
            "last_output": last_output,
            "last_error": last_error,
            "last_exit_code": last_exit_code,
            "last_log_path": last_log_path,
            "current_files": current_files,
            "done": done,
        }

    graph = StateGraph(DCipherState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()
