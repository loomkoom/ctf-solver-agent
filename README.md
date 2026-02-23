## AI CTF Solver (LangGraph + Kali Sandbox)

### Quick start
1) Build and run the Kali sandbox:

```bash
docker compose build ctf-sandbox
docker compose up -d ctf-sandbox
```

2) Run the solver against a CTFd instance:

```bash
uv run python src/main.py --ctfd-url https://ctfd.example.com --ctfd-token <token>
```

### Environment variables
Create a `.env` with the model and CTFd credentials you want to use:

```
OPENAI_API_KEY=...
CTFD_URL=https://ctfd.example.com
CTFD_TOKEN=...
```

### Knowledge base (RAG)
- Drop writeups, HackTricks, or CTF-Wiki markdowns under:
  - `data/knowledge_base/writeups`
  - `data/knowledge_base/wikis/hacktricks`
  - `data/knowledge_base/wikis/ctf-wiki`
- The solver will ingest on startup unless `--skip-rag` is passed.

### Benchmark runner (NYU CTF Bench style)
Run against a local dataset tree:

```bash
uv run python src/benchmarks/nyu_ctf_bench.py --bench data/test_bench/ulyssisctf --limit 25
```
