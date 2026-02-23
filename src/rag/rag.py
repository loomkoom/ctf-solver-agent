from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

WRITEUPS_COLLECTION_NAME = "ctf_writeups"
REFERENCE_COLLECTION_NAME = "ctf_reference"

# local vector DB
client = chromadb.PersistentClient(path=DB_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL_NAME
)
writeups_collection = client.get_or_create_collection(
    name=WRITEUPS_COLLECTION_NAME,
    embedding_function=embedding_function,
)
reference_collection = client.get_or_create_collection(
    name=REFERENCE_COLLECTION_NAME,
    embedding_function=embedding_function,
)


def ingest_knowledge_base(base_dir: str, force: bool = False) -> int:
    path = Path(base_dir)
    if not path.exists():
        return 0

    total = 0
    if force or writeups_collection.count() == 0:
        total += _ingest_markdown_dir(
            path / "writeups",
            source_type="writeup",
            base_dir=path,
            collection=writeups_collection,
            store="writeups",
        )

    if force or reference_collection.count() == 0:
        total += _ingest_markdown_dir(
            path / "wikis",
            source_type="wiki",
            base_dir=path,
            collection=reference_collection,
            store="reference",
        )
        total += _ingest_markdown_dir(
            path / "cheat_sheets",
            source_type="cheatsheet",
            base_dir=path,
            collection=reference_collection,
            store="reference",
        )
        total += _ingest_markdown_dir(
            path / "tool_manuals",
            source_type="tool",
            base_dir=path,
            collection=reference_collection,
            store="reference",
        )
        total += _ingest_markdown_dir(
            path / "methodologies",
            source_type="methodology",
            base_dir=path,
            collection=reference_collection,
            store="reference",
        )

    return total


def _ingest_markdown_dir(
    directory: Path,
    source_type: str,
    base_dir: Path,
    collection,
    store: str,
) -> int:
    if not directory.exists():
        return 0

    count = 0
    for file in directory.rglob("*.md"):
        try:
            content = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file.read_text(encoding="latin-1", errors="ignore")

        rel_path = str(file.relative_to(base_dir))
        metadata = {
            "source_type": source_type,
            "source": file.parent.name,
            "source_path": rel_path,
            "category": file.parts[-2] if len(file.parts) > 2 else "unknown",
            "kb_store": store,
        }

        collection.upsert(
            documents=[content],
            metadatas=[metadata],
            ids=[rel_path],
        )
        count += 1
    return count


def search_knowledge(query: str, n_results: int = 3) -> str:
    results: list[dict[str, Any]] = []
    results.extend(_query_collection(writeups_collection, "writeups", query, n_results))
    results.extend(_query_collection(reference_collection, "reference", query, n_results))

    if not results:
        return ""

    results.sort(key=lambda r: r.get("distance") if r.get("distance") is not None else 1e9)
    top = results[:n_results]

    formatted_results = []
    for item in top:
        metadata = item.get("metadata") or {}
        rel_path = metadata.get("source_path", "unknown")
        source_type = metadata.get("source_type", "unknown")
        store = item.get("store", "unknown")
        formatted_results.append(
            "STORE: {store}\nSOURCE TYPE: {source_type}\nSOURCE PATH: {rel_path}\nCONTENT: {doc}".format(
                store=store,
                source_type=source_type,
                rel_path=rel_path,
                doc=item.get("doc", ""),
            )
        )

    return "\n\n---\n\n".join(formatted_results)


def _query_collection(collection, store: str, query: str, n_results: int) -> list[dict[str, Any]]:
    if collection.count() == 0:
        return []

    res = collection.query(query_texts=[query], n_results=n_results)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs)

    out: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "doc": doc,
            "metadata": meta,
            "distance": dist,
            "store": store,
        })
    return out
