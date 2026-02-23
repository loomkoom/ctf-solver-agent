from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any


_SECRET_KEYS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
)

_TEXT_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(--?(?:api[-_]?key|token|password|passwd|secret|authorization|bearer|cookie))(?:=|\s+)([^\s'\";]+)"), r"\1=REDACTED"),
    (re.compile(r"(?i)((?:api[-_]?key|token|password|passwd|secret)\s*[:=]\s*)([^\s'\";]+)"), r"\1REDACTED"),
    (re.compile(r"(?i)(authorization:\s*bearer\s+)([^\s'\";]+)"), r"\1REDACTED"),
    (re.compile(r"(?i)(authorization:\s*basic\s+)([^\s'\";]+)"), r"\1REDACTED"),
    (re.compile(r"(?i)(cookie:\s*)([^\r\n]+)"), r"\1REDACTED"),
    (re.compile(r"(?i)(\b[A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD)\b=)([^\s'\";]+)"), r"\1REDACTED"),
]


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return json.dumps(str(value), ensure_ascii=True)


def _is_secret_key(key: str) -> bool:
    lowered = (key or "").lower()
    return any(token in lowered for token in _SECRET_KEYS)


def redact_text(text: str) -> str:
    if not text:
        return text
    redacted = text
    for pattern, repl in _TEXT_REDACTIONS:
        redacted = pattern.sub(repl, redacted)
    return redacted


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, val in value.items():
            if _is_secret_key(str(key)):
                out[key] = "REDACTED"
            else:
                out[key] = redact(val)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def preview_text(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def head_lines(text: str, count: int = 8) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[:count])


def debug_log(
    enabled: bool,
    run_id: str | None,
    challenge_id: str | None,
    challenge_name: str | None,
    event: str,
    **fields: Any,
) -> None:
    if not enabled:
        return
    payload: dict[str, Any] = {
        "ts": _now_ts(),
        "event": event,
        "run_id": run_id or "unknown",
        "challenge_id": challenge_id or "unknown",
    }
    if challenge_name:
        payload["challenge_name"] = challenge_name
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value
    line = " ".join(f"{key}={_format_value(value)}" for key, value in payload.items())
    print(line)
