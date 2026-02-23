from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import time
import json

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from src.config import settings

def _rag_db_path() -> Path:
    return Path(settings.rag_db_path)

def _links_rag_paths() -> list[Path]:
    candidates = [
        Path(settings.rag_links_db_path),
        Path(__file__).resolve().parent / "data" / "chroma",
    ]
    seen: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
    return seen

def _manifest_path() -> Path:
    return _rag_db_path() / "ingest_manifest.json"

# local vector DB (lazy)
@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(_rag_db_path()))


@lru_cache(maxsize=1)
def _get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.rag_embed_model)


@lru_cache(maxsize=1)
def _get_writeups_collection():
    return _get_client().get_or_create_collection(
        name=settings.rag_writeups_collection,
        embedding_function=_get_embedding_function(),
    )


@lru_cache(maxsize=1)
def _get_reference_collection():
    return _get_client().get_or_create_collection(
        name=settings.rag_reference_collection,
        embedding_function=_get_embedding_function(),
    )


def ingest_knowledge_base(base_dir: str, force: bool = False, mode: str | None = None) -> int:
    path = Path(base_dir)
    if not path.exists():
        return 0

    mode = (mode or settings.rag_mode or "all").lower()
    targets = _build_ingest_targets(path, mode)
    all_files: list[Path] = []
    for target in targets:
        all_files.extend(target["files"])

    if not all_files:
        print("RAG: no markdown files found for ingestion.")
        return 0

    fingerprint = _fingerprint_files(all_files, path, mode)
    if not force and _manifest_matches(path, fingerprint, mode) and _collections_ready():
        print("RAG: cache hit, skipping ingestion.")
        return 0

    total = 0
    total_files = len(all_files)
    print(f"RAG: scanning {total_files} markdown files (mode={mode}).")
    start = time.time()
    for target in targets:
        total += _ingest_markdown_files(
            target["files"],
            source_type=target["source_type"],
            base_dir=path,
            collection=target["collection"],
            store=target["store"],
        )
    elapsed = time.time() - start
    _write_manifest(path, fingerprint, mode, total_files)
    print(f"RAG: indexed {total} files in {elapsed:.1f}s.")
    return total


def _collections_ready() -> bool:
    return _get_writeups_collection().count() > 0 or _get_reference_collection().count() > 0


def _build_ingest_targets(base_dir: Path, mode: str) -> list[dict]:
    mode = mode.lower()
    targets: list[dict] = []

    def _add(directory: Path, source_type: str, collection, store: str):
        files = list(directory.rglob("*.md")) if directory.exists() else []
        targets.append({
            "files": files,
            "source_type": source_type,
            "collection": collection,
            "store": store,
        })

    if mode in {"all", "writeups"}:
        _add(base_dir / "writeups", "writeup", _get_writeups_collection(), "writeups")

    if mode in {"all", "no_writeups", "methodology"}:
        if mode != "methodology":
            _add(base_dir / "wikis", "wiki", _get_reference_collection(), "reference")
        _add(base_dir / "cheat_sheets", "cheatsheet", _get_reference_collection(), "reference")
        _add(base_dir / "tool_manuals", "tool", _get_reference_collection(), "reference")
        _add(base_dir / "methodologies", "methodology", _get_reference_collection(), "reference")

    return targets


def _ingest_markdown_files(
    files: list[Path],
    source_type: str,
    base_dir: Path,
    collection,
    store: str,
) -> int:
    if not files:
        return 0

    count = 0
    total = len(files)
    last_report = 0
    start = time.time()
    for idx, file in enumerate(files, start=1):
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

        progress = int(idx / total * 100)
        if progress - last_report >= 10:
            elapsed = time.time() - start
            rate = elapsed / idx if idx else 0
            eta = rate * (total - idx)
            print(f"RAG: {source_type} {progress}% ({idx}/{total}) ETA {eta:.1f}s")
            last_report = progress
    return count


def _fingerprint_files(files: list[Path], base_dir: Path, mode: str) -> str:
    sha = hashlib.sha1()
    sha.update(mode.encode("utf-8"))
    for file in files:
        try:
            stat = file.stat()
            rel = file.relative_to(base_dir)
        except FileNotFoundError:
            continue
        sha.update(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}\n".encode("utf-8"))
    return sha.hexdigest()


def _manifest_matches(base_dir: Path, fingerprint: str, mode: str) -> bool:
    manifest_path = _manifest_path()
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("fingerprint") == fingerprint and data.get("mode") == mode and data.get("base_dir") == str(base_dir)


def _write_manifest(base_dir: Path, fingerprint: str, mode: str, total_files: int) -> None:
    manifest_path = _manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "mode": mode,
        "base_dir": str(base_dir),
        "total_files": total_files,
        "updated_at": time.time(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _truncate_doc(text: str, max_chars: int | None = None) -> str:
    limit = max_chars or settings.rag_doc_max_chars
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "\n[...snip...]\n"


@lru_cache(maxsize=1)
def _links_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.rag_embed_model)


def _query_links_rag(query: str, n_results: int) -> list[dict[str, Any]]:
    paths = _links_rag_paths()
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        return []
    try:
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(name=settings.rag_links_collection)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    embedder = _links_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    res = collection.query(query_embeddings=[q_vec], n_results=n_results)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs)

    out: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "doc": doc,
            "metadata": meta,
            "distance": dist,
            "store": "links_rag",
        })
    return out


def search_knowledge(query: str, n_results: int = 3, include_writeups: bool | None = None) -> str:
    results: list[dict[str, Any]] = []
    if not settings.rag_enabled:
        return ""
    use_writeups = settings.rag_include_writeups if include_writeups is None else include_writeups
    if use_writeups:
        results.extend(_query_collection(_get_writeups_collection(), "writeups", query, n_results))
    results.extend(_query_collection(_get_reference_collection(), "reference", query, n_results))
    results.extend(_query_links_rag(query, n_results))

    if not results:
        return ""

    results.sort(key=lambda r: r.get("distance") if r.get("distance") is not None else 1e9)
    top = results[:n_results]

    formatted_results = []
    for item in top:
        metadata = item.get("metadata") or {}
        rel_path = metadata.get("source_path") or metadata.get("source_url") or metadata.get("title") or "unknown"
        source_type = metadata.get("source_type") or metadata.get("source_kind") or "unknown"
        store = item.get("store", "unknown")
        doc = _truncate_doc(item.get("doc", ""), settings.rag_doc_max_chars)
        formatted_results.append(
            "STORE: {store}\nSOURCE TYPE: {source_type}\nSOURCE PATH: {rel_path}\nCONTENT: {doc}".format(
                store=store,
                source_type=source_type,
                rel_path=rel_path,
                doc=doc,
            )
        )

    output = "\n\n---\n\n".join(formatted_results)
    if len(output) > settings.rag_total_max_chars:
        output = output[:settings.rag_total_max_chars] + "\n[...snip...]\n"
    return output


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
