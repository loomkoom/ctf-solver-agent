# RAG Sources Overview

## Local KB vs Links RAG
- Local KB (`data/knowledge_base/**`) is authoritative, deterministic, and preferred for tool manuals and internal methodologies.
- Links RAG (`src/rag/links.yaml`) is for external documentation and references when local manuals are insufficient.

## Ingest Local KB
The solver ingests the local KB at startup unless `--skip-rag` is provided.
To force a rebuild:
```bash
python src/main.py --reindex
```
You can also call ingestion directly:
```bash
python - <<'PY'
from src.tools.rag import ingest_knowledge_base
ingest_knowledge_base("data/knowledge_base", force=True)
PY
```

## Ingest Links RAG
The links RAG pipeline crawls URLs in `src/rag/links.yaml` and stores a separate Chroma DB.
```bash
python src/rag/links_rag.py ingest --yaml src/rag/links.yaml
```
Notes:
- `meta_collections` are stored but not embedded by default.
- Use links RAG only when local manuals do not cover a tool.
