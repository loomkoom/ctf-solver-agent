## AI CTF Solver (LangGraph + Kali Sandbox)

Jeopardy-style CTF solver with a strict Plan → Research → Execute → Verify loop,
hard budgets, and replayable logs. All tooling executes inside a Dockerized Kali sandbox.

**Key guarantees**
- No port scanning or machine pentesting (`nmap`, `masscan`, etc. are blocked).
- Web tools only run when an explicit URL is provided by the challenge.
- Every tool call is logged with stdout/stderr and bounded timeouts.
- Flags are only accepted with evidence (output snippet + log path).

**Key autonomy features**
- The executor receives last output + verifier hint to choose the next step.
- Auto-decode heuristic for obvious hex/base64 output.
- Non-zero exit fallback map (e.g., `strings → file_info`) to reduce stalls.
- Tiered model fallback on empty/refusal responses.

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
uv run ig-ctf-solver --challenge-dir data/challenges/sample --challenge-url https://example.com
```

4) Run against a CTFd instance:
```bash
uv run ig-ctf-solver --ctfd-url https://ctfd.example.com --ctfd-token <token>
```

5) View logs:
- `runs/<challenge_id>/trajectory.jsonl`
- `runs/<challenge_id>/logs/*.txt`

**Model selection**
Override providers/models:
```bash
uv run ig-ctf-solver --planner-provider openai --planner-model o3 --executor-provider openai --executor-model o4-mini
```

Add tier fallbacks (JSON or CSV format):
```bash
uv run ig-ctf-solver --planner-tiers "openai:o3,openai:o4-mini" --verifier-tiers "openai:o4-mini,openai:o3"
```

Offline stub mode (deterministic, no API calls):
```bash
uv run ig-ctf-solver --planner-provider stub --executor-provider stub
```
Stub mode is for smoke tests and demo plumbing; it does not solve challenges.

**CLI help**
```bash
uv run ig-ctf-solver --help
```

### Environment variables
Create `.env` with your model and CTFd credentials:
```
OPENAI_API_KEY=...
CTFD_URL=https://ctfd.example.com
CTFD_TOKEN=...
OLLAMA_BASE_URL=http://localhost:11434
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
uv run ctfd-bootstrap --ctfd-url http://localhost:8000 --ctfd-token <token> --challenge-root data/test_bench --dry-run
uv run ctfd-bootstrap --ctfd-url http://localhost:8000 --ctfd-token <token> --challenge-root data/test_bench
```

CTFd bootstrap supports `--dry-run`, `--skip-existing`, and `--limit`.

## Local Bench Testing (data/test_bench)
Runs challenges from `data/test_bench/**` using description + attachments only.
Writeups are ignored by default.

```bash
uv run ctf-local-bench --bench data/test_bench --limit 5
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
uv run ig-ctf-solver --rag-mode no_writeups
uv run ig-ctf-solver --rag-mode methodology
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
