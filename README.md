## AI CTF Solver (LangGraph + Kali Sandbox)

Demo-focused, Jeopardy-style CTF solver with a strict Plan → Research → Execute → Verify loop,
hard budgets, and replayable logs. All tooling executes inside a Dockerized Kali sandbox.

**Key guarantees**
- No port scanning or machine pentesting (`nmap`, `masscan`, etc. are blocked).
- Web tools only run when an explicit URL is provided by the challenge.
- Every tool call is logged with stdout/stderr and bounded timeouts.
- Flags are only accepted with evidence (output snippet + log path).

**Quickstart (A→Z)**
1) Install deps:
```bash
uv sync
```

2) Start the Kali sandbox:
```bash
docker compose build ctf-sandbox
docker compose up -d ctf-sandbox
```

3) Run against a local challenge folder:
```bash
uv run python -m ig_ctf_solver --challenge-dir data/challenges/sample --challenge-url https://example.com
```
Or:
```bash
uv run ig-ctf-solver --challenge-dir data/challenges/sample --challenge-url https://example.com
```

4) Run against a CTFd instance:
```bash
uv run python -m ig_ctf_solver --ctfd-url https://ctfd.example.com --ctfd-token <token>
```

You can override model providers/models with:
```bash
uv run python -m ig_ctf_solver --planner-provider openai --planner-model gpt-4o --executor-provider openai --executor-model gpt-4o-mini
```

Offline stub mode (deterministic, no API calls):
```bash
uv run python -m ig_ctf_solver --planner-provider stub --executor-provider stub
```
Stub mode is for smoke tests and demo plumbing; it does not solve challenges.

5) View logs:
- `runs/<challenge_id>/trajectory.jsonl`
- `runs/<challenge_id>/logs/*.txt`

**CLI help**
```bash
uv run python -m ig_ctf_solver --help
```

### Environment variables
Create `.env` with your model and CTFd credentials:
```
OPENAI_API_KEY=...
CTFD_URL=https://ctfd.example.com
CTFD_TOKEN=...
```

## Local CTFd Bootstrap (Demo)
Start a local CTFd:
```bash
docker compose -f docker-compose.ctfd.yml up -d
```

Optional: set a persistent secret key for CTFd:
```
CTFD_SECRET_KEY=change-me
```

Create an admin user in the UI, then generate an API token. Bootstrap challenges from a local folder:
```bash
uv run python -m src.ctfd_bootstrap --ctfd-url http://localhost:8000 --ctfd-token <token> --challenge-root data/test_bench --dry-run
uv run python -m src.ctfd_bootstrap --ctfd-url http://localhost:8000 --ctfd-token <token> --challenge-root data/test_bench
```

CTFd bootstrap supports `--dry-run`, `--skip-existing`, and `--limit`.

## Local Bench Testing (data/test_bench)
Runs challenges from `data/test_bench/**` using description + attachments only.
Writeups are ignored by default.

```bash
uv run python -m src.benchmarks.local_bench --bench data/test_bench --limit 5
```

Outputs a JSON report (default `runs/bench_<timestamp>.json`) with:
- success rate
- avg tool calls / iterations
- total time

## RAG Indexing (with progress + cache)
RAG indexing shows progress and caches results in `chroma_db/ingest_manifest.json`.
Re-runs skip indexing if nothing changed.

Modes:
- `all` (default): writeups + references
- `no_writeups`: references only
- `methodology`: small methodology-only subset (best for huge corpora)

Examples:
```bash
uv run python -m ig_ctf_solver --rag-mode no_writeups
uv run python -m ig_ctf_solver --rag-mode methodology
```

Optional link-based KB ingestion:
```bash
uv run python src/rag/links_rag.py ingest --yaml src/rag/links.yaml
```

## How It Works (Architecture Summary)
**Components**
- `src/agent/agent.py`: Plan → Research → Execute → Verify loop and budgeting
- `src/tools/*`: sandboxed tool wrappers (all commands run in Docker)
- `src/solver.py`: solver orchestration + trajectory logging
- `src/challenges.py`: local challenge parsing
- `src/ctfd.py` + `src/ctfd_bootstrap.py`: CTFd API client + loader

**Data flow**
1. Ingest challenge (description + attachments + optional URL).
2. Planner selects category + objective.
3. Research scans files in the sandbox.
4. Execute runs exactly one tool call.
5. Verify extracts flags with evidence and validates.

**Autonomy limits**
- No port scanning or host exploitation.
- Web requests only to explicit challenge URLs.
- Bounded by hard budgets (tool calls, phase cycles, wall time).

## Tests
```bash
uv run pytest
```

## Demo Bench Metrics (latest)
Last run (stub provider, 1 challenge):
- total: 1
- success_rate: 0.00
- avg_tool_calls: 15.00
- avg_iterations: 15.00
- total_time_s: 38.1

Reproduce:
```bash
uv run python -m src.benchmarks.local_bench --bench data/test_bench --limit 1 --skip-rag --planner-provider stub --executor-provider stub
```

Metrics are written to `runs/bench_<timestamp>.json`.
