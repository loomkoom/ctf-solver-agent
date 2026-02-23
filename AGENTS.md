# AGENTS.md

**Autonomous CTF Solver — Operational Contract & Execution Rules**

## 1. Primary Objective

> **Reliably solve Jeopardy-style CTF challenges end-to-end within strict time and tool budgets, producing verifiable flags and structured execution logs for competition submission.**

The system must transition from a "chatbot" to a **Recursive Problem Solver**. Success is defined by a functional pipeline that autonomously navigates from challenge ingestion to flag submission.

---

## 2. Core Philosophy: "Think, Decompose, Execute, Critique"

The agent logic should follow a **Dynamic Research & Development (R&D) cycle**. While implementation can vary (LangGraph, Headless SWE-agent, or custom state machines), the logic must follow these phases:

1. **Architect (Strategy):** Breaks the problem into sub-tasks (e.g., "Enumerate services," "Reverse binary"). It identifies expected flag formats (e.g., `IGCTF{}`, `CSCBE{}`).
2. **Researcher (Discovery):** Gathers live data. If a write-up isn't found in RAG, it uses tools (`nmap`, `strings`, `checksec`) to populate the `challenge_context`.
3. **Executor (Tactical):** Generates and runs exact commands within the sandboxed environment.
4. **Verifier (Critique):** Analyzes `STDOUT/STDERR`. It identifies failures, prevents hallucinations, and feeds back to the Architect to pivot.

---

## 3. Challenge Lifecycle & Ingestion

* **CTFd Integration:** Use a dedicated `CTFdConnector` to fetch challenge descriptions, download attachments, and submit flags autonomously.
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

---

## 6. Performance & Evaluation (Metrics)

The system is optimized for:

* **ESR (Exploitation Success Rate):** Percentage of flags successfully captured independently.
* **AST (Average Steps per Task):** Efficiency in reaching the flag. Prefer specialized tools (`ciphey`, `sqlmap`, `stegseek`) over generic scripts to reduce token usage.
* **Convergent Reasoning:** Avoiding infinite loops by terminating gracefully when the budget is exceeded or all pipelines are exhausted.

---

## 7. Reference Benchmarks

Validate performance against industry standards:

* **InterCode-CTF:** For interactive bash/terminal performance.
* **XBOW / HTB MCP:** For labor-relevant, high-difficulty exploitation.
* **NYU CTF Bench:** For baseline security reasoning comparison.