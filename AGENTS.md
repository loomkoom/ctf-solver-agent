# AGENTS.md

Autonomous CTF Solver — Operational Contract & Execution Rules (Jeopardy-Style, File/Service Challenges)

## 0. Scope & Non-Goals
This agent solves **Jeopardy-style CTF challenges** (crypto, forensics/stego, rev, pwn, misc, and “web” where a URL is explicitly provided).
(OSINT challenges can also be possible, to search the web for specific information).

Non-goals:
- No network pentesting / machine hacking / port scanning (NO nmap/masscan).
- No “autonomous red team” behavior.
- No heavy multi-agent infrastructure in the 48h prototype (use phased single-agent loop).

## 1. Primary Objective
> Reliably solve challenges end-to-end within strict time/tool budgets, producing verifiable flags and structured logs suitable for competition submission.

Success = working pipeline: ingest → analyze → run tools → extract → verify → (optionally) submit.

The system must transition from a "chatbot" to a **Recursive Problem Solver**. Success is defined by a functional pipeline that autonomously navigates from challenge ingestion to flag submission.

---

## 2. Core Methodology: "Think, Decompose, Execute, Critique"

The agent logic should follow a **Dynamic Research & Development (R&D) cycle**. 

While implementation can vary (LangGraph, Headless SWE-agent, or custom state machines), the logic must follow these phases:

**D-CIPHER-style phases** on top of a **SWE-agent-like Agent-Computer Interface (ACI)** seems most aligned with our goals.

### Phases (must be explicit in logs/state)
1. **Architect (Strategy):** Breaks the problem into sub-tasks (e.g., "Enumerate services," "Reverse binary"). It identifies expected flag formats (e.g., `IGCTF{}`, `CSCBE{}`).
2. **Researcher (Discovery):** Gathers live data. If a write-up isn't found in RAG, it uses tools (`nmap`, `strings`, `checksec`) to populate the `challenge_context`.
3. **Executor (Tactical):** Generates and runs exact commands within the sandboxed environment.
4. **Verifier (Critique):** Analyzes `STDOUT/STDERR`. It identifies failures, prevents hallucinations, and feeds back to the Architect to pivot.

---

## 3. Challenge Lifecycle & Ingestion

* **CTFd Integration:** Use a dedicated `CTFdConnector` to fetch challenge descriptions, download attachments, and submit flags autonomously.
* **Challenges provide**: description + optional attachments + optional explicit URL/endpoint.
* **Data Structure Navigation:** * **Knowledge Base:** Strictly `.md` write-ups and wikis (HackTricks, CTF-Wiki) for RAG.
* **Artifacts:** Binaries, images, and non-textual files live in `data/artifacts/`.
* **Mapping:** When RAG identifies a methodology from a write-up, the agent must cross-reference the corresponding artifact folder to find functional source code or binaries.

---

## 4. RAG Strategy (Memory & Knowledge)

* **Dual-Stream RAG:** * **Methodology Stream:** Accesses wikis and checklists to provide the agent with specific bypass techniques or exploitation steps.
* **Experience Stream:** Accesses past write-ups to find historical matches for similar challenges.

* **Source Mapping:** Ingested documents must contain `source_path` metadata to allow the agent to move from "reading a solution" to "executing on artifacts."

---

## 5. Global Constraints & Tactical Rules

* **Do Not Reinvent the Wheel:** Leverage established patterns from **D-CIPHER**, **SWE-agent**, and **Deadend-CLI** etc (see `RESOURCES.md`) for terminal interfaces and feedback-driven iteration.
* **Sandbox Isolation:** All commands MUST run in the **Dockerized Kali Linux** environment. Never execute challenge binaries on the host.
* **Dynamic Flag Detection:** Use the `flag_format` variable. If unknown, use a broad regex fallback: `(?:IGCTF|flag|CSCBE|UCTF|ctf)\{.*?\}`.
* **Timeouts & Bloat:** Tool calls must have hard 30s timeouts. Large outputs must be summarized before being passed back to the LLM to preserve context.
* **Self-Healing:** If a command fails (e.g., `Exit Code 127`), the agent must analyze why and suggest a fix (e.g., installing a missing tool or changing parameters) rather than repeating the error.
* If an endpoint is given, interact **only with that endpoint** (no scanning other hosts/ports).
* All tool execution happens inside the **Dockerized Kali sandbox**.
* Outputs are logged and replayable.

---

## 6. Tool Output Control (Prevent Context Death)

Rules:

* If stdout/stderr > 300 lines OR > 20k chars:

  * Save full output to `runs/<challenge_id>/logs/<tool>_<ts>.txt`
  * Pass to LLM only:

    * first 100 lines + last 100 lines
    * plus a short summary + extracted indicators (paths, strings, matches)

avoid dumping giant logs into the model.

---

## 7. Example Tooling (No Port Scans)

All tools must run in the sandbox.

### Core filesystem / inspection
* `ls`, `find`, `file`, `stat`, `cat`, `head`, `tail`, `grep`, `ripgrep (rg)`, `xxd`, `hexdump`, `tar`, `unzip`, `7z`

### Forensics / Stego
* `exiftool`, `binwalk`, `strings`, `stegseek`, `zsteg`, `pdfinfo` (if relevant)

### Crypto / Encoding
* `ciphey` (best-effort), python scripts, `openssl` (basic), `base64`/`base32` utilities

### Reverse engineering
* `radare2` **in JSON mode** (e.g., function list/strings), `objdump`, `readelf`

### Pwn
* `checksec` (or equivalent), `pwntools` scripting, `gdb` **batch mode only** (`-q -ex ... -ex quit`)
* No interactive debugging sessions.

### Web ( when URL is provided)
* `curl`, python `requests`, small targeted fuzzing allowed ONLY within provided base URL.
* Optional: `ffuf` with strict caps (<= 200 requests, small wordlist, single depth).
* NO scanning other hosts or ports.

Disallowed:
* `nmap`, `masscan`, uncontrolled brute force, long-running fuzzers.

---
## 8. Flag Detection & Verification (No Hallucinations)

* Prefer the CTF-provided format if known (e.g., `IGCTF{...}`).
* Otherwise use a general regex:

  * `\\b[A-Za-z0-9_\\-]{0,24}\\{[^\\n\\r]{3,200}\\}\\b`
* Candidate flags MUST have proof:

  * file path + line number OR exact tool output snippet source
* Never “guess” a flag.

---

## 9. Failure Handling (Self-Healing)

If a command fails:

* Explain why (missing tool, wrong path, permission, format).
* Apply ONE corrective action (install tool if allowed, adjust args, change pipeline).
* Do not repeat identical commands.

After 2 failed attempts in the same pipeline → pivot or terminate.

---

## 10. Pipeline Priorities (Win Fast)

Default priority if category is unclear:

1. forensics/stego
2. crypto/encoding
3. rev (strings/r2-json)
4. pwn (checksec + quick pwntools template)
5. web (only if URL provided)

---
## 11. Required minimal State Model (Single Source of Truth)
Maintain a structured `CTFState` for every challenge:
```json
{
  "challenge_id": "string",
  "title": "string",
  "description": "string",
  "provided_flag_format": "string|null",
  "flag_regex": "string",
  "category_guess": ["crypto|forensics|stego|rev|pwn|web|misc"],
  "confidence": 0.0,
  "artifacts": [],
  "attempts": [],
  "candidates": [],
  "flags_found": [],
  "budgets": {
    "wall_seconds_max": 600,
    "tool_calls_max": 15,
    "phase_cycles_max": 20,
    "tool_timeout_seconds": 30
  }
}
```
---
## 12. minimal Logging Contract (Replayable)

Every tool call emits a JSONL event:

```json
{
  "ts": "ISO-8601",
  "challenge_id": "string",
  "phase": "plan|research|execute|verify",
  "cmd": "string",
  "cwd": "string",
  "timeout_s": 30,
  "exit_code": 0,
  "duration_s": 1.23,
  "stdout_path": "runs/.../stdout.txt",
  "stderr_path": "runs/.../stderr.txt",
  "summary": "short summary",
  "extractions": {
    "paths": [],
    "urls": [],
    "matches": []
  }
}
```
---


## 13. Hard Budgets (Demo-Safe)

Mandatory defaults (override via config if needed):

* Max wall time per challenge: **10 minutes**
* Max tool calls per challenge: **15**
* Max phase cycles: **20**
* Max tool runtime: **30s** per call
* Max category pivots: **3**
* Max submissions per candidate flag: **3**

If budget exceeded → terminate gracefully with summary.

---

## 14. Performance & Evaluation (Metrics)

The system is optimized for:

* **ESR (Exploitation Success Rate):** Percentage of flags successfully captured independently.
* **AST (Average Steps per Task):** Efficiency in reaching the flag. Prefer specialized tools (`ciphey`, `sqlmap`, `stegseek`) over generic scripts to reduce token usage.
* **Convergent Reasoning:** Avoiding infinite loops by terminating gracefully when the budget is exceeded or all pipelines are exhausted.

---

## 15. Reference Benchmarks

Validate performance against one or more of the following benchmarks:

* **InterCode-CTF:** For interactive bash/terminal performance.
* **XBOW:** For labor-relevant, high-difficulty exploitation.
* **NYU CTF Bench:** For baseline security reasoning comparison.