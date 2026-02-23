#!/usr/bin/env python3
"""
RAG ingestion pipeline for a security/CTF knowledge base.

Supports:
- YAML format:
    domains:
      <domain>:
        <section>: [urls...]
    meta_collections:
      <group>: [urls...]
- Web pages / docs / wikis
- GitHub repos
- GitHub blob URLs (single files)
- Gists (as web pages)

Outputs:
- data/raw_html/
- data/raw_repo/
- data/cleaned_md/
- data/chunks/
- data/chroma/   (vector store)

Default behavior:
- Embeds everything under `domains`
- Skips embedding `meta_collections` (link directories / awesome repos) but stores manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Any
from urllib.parse import urlparse, urljoin, urldefrag

import requests
import yaml
import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm
from git import Repo, GitCommandError

import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb

from src.config import settings


# -----------------------------
# Config (tune these)
# -----------------------------
DATA_DIR = pathlib.Path(settings.links_rag_data_dir)
RAW_HTML_DIR = DATA_DIR / "raw_html"
RAW_REPO_DIR = DATA_DIR / "raw_repo"
CLEAN_MD_DIR = DATA_DIR / "cleaned_md"
CHUNKS_DIR = DATA_DIR / "chunks"
MANIFEST_DIR = DATA_DIR / "manifests"
CHROMA_DIR = pathlib.Path(settings.rag_links_db_path)

USER_AGENT = settings.links_rag_user_agent
REQUEST_TIMEOUT = settings.links_rag_request_timeout_s
HTTP_RETRIES = settings.links_rag_http_retries
CRAWL_DELAY_S = settings.links_rag_crawl_delay_s

# Crawl defaults
DEFAULT_MAX_PAGES = settings.links_rag_default_max_pages
DEFAULT_MAX_DEPTH = settings.links_rag_default_max_depth
WIKI_MAX_PAGES = settings.links_rag_wiki_max_pages
WIKI_MAX_DEPTH = settings.links_rag_wiki_max_depth

# Chunking
CHUNK_TARGET_TOKENS = settings.links_rag_chunk_target_tokens
CHUNK_OVERLAP_TOKENS = settings.links_rag_chunk_overlap_tokens
MIN_CHUNK_TOKENS = settings.links_rag_min_chunk_tokens

# Repo extraction
INDEX_CODE_FILES_FROM_REPOS = settings.links_rag_index_code_files  # docs-only by default = better signal/noise
MAX_REPO_FILE_BYTES = settings.links_rag_max_repo_file_bytes       # skip huge files
MAX_TEXT_CHARS_PER_FILE = settings.links_rag_max_text_chars_per_file  # truncate very large text files
REPO_DOC_EXTS = {".md", ".markdown", ".rst", ".txt", ".adoc"}
REPO_CODE_EXTS = {
    ".py", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".java", ".kt", ".js", ".ts",
    ".sh", ".ps1", ".php", ".rb", ".lua", ".sol", ".yml", ".yaml", ".json", ".toml",
    ".ini", ".cfg", ".conf", ".asm", ".s", ".smali"
}

# Embeddings / Vector DB
EMBED_MODEL_NAME = settings.rag_embed_model
CHROMA_COLLECTION_NAME = settings.rag_links_collection

# If True, try Playwright fallback when requests returns empty/blocked pages
ENABLE_PLAYWRIGHT_FALLBACK = settings.links_rag_enable_playwright

# Domain/section-specific crawl heuristics
# (host-level overrides are possible later if you want)
CRAWL_POLICY_BY_SECTION = {
    "references": {"crawl": True, "max_pages": DEFAULT_MAX_PAGES, "max_depth": DEFAULT_MAX_DEPTH},
    "learning": {"crawl": True, "max_pages": DEFAULT_MAX_PAGES, "max_depth": DEFAULT_MAX_DEPTH},
    "cheat_sheets": {"crawl": False, "max_pages": 1, "max_depth": 0},
    "tools": {"crawl": False, "max_pages": 1, "max_depth": 0},
    "blogs": {"crawl": False, "max_pages": 1, "max_depth": 0},
}

WIKI_LIKE_DOMAINS = {"wiki"}  # your domain bucket "wiki" gets deeper crawl
NO_EMBED_TOPLEVELS = {"meta_collections"}  # keep for discovery, not for semantic retrieval

# URL substrings that are usually login/search hubs or not worth crawling deeper
BLOCK_CRAWL_PATH_SUBSTRINGS = [
    "/login", "/signin", "/signup", "/register", "/search", "/account",
    "/privacy", "/terms", "/contact", "/about", "/cookie", "/cookies",
]

# Skip binary-ish extensions when crawling
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".zip", ".7z", ".gz", ".tgz", ".tar", ".xz", ".bz2",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".mp3", ".wav", ".mp4", ".mov", ".avi", ".mkv",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".jar",
    ".css", ".js", ".map", ".woff", ".woff2", ".ttf", ".eot",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Seed:
    url: str
    toplevel: str           # "domains" or "meta_collections"
    domain: str             # e.g. "web", "crypto", or "meta_collections"
    section: str            # e.g. "references", "tools", "link_directories"
    subgroup: str           # usually same as section; reserved for future nesting
    embed: bool             # False for meta_collections by default


@dataclass
class SourceDoc:
    doc_id: str
    seed_url: str
    source_url: str
    domain: str
    section: str
    subgroup: str
    toplevel: str
    source_kind: str        # webpage|repo_file|github_blob|raw_text
    title: str
    content: str            # cleaned markdown/text
    extra: Dict[str, Any]


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    seed_url: str
    source_url: str
    domain: str
    section: str
    subgroup: str
    toplevel: str
    source_kind: str
    title: str
    chunk_index: int
    text: str


# -----------------------------
# Filesystem helpers
# -----------------------------
def ensure_dirs() -> None:
    for d in [RAW_HTML_DIR, RAW_REPO_DIR, CLEAN_MD_DIR, CHUNKS_DIR, MANIFEST_DIR, CHROMA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def safe_slug(s: str, max_len: int = 180) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:max_len] if s else "item"

def canonical_url(url: str) -> str:
    # strip fragment only
    clean, _ = urldefrag(url.strip())
    return clean.rstrip("/") if clean.endswith("/") and len(clean) > len("https://x") else clean

def path_ext_from_url(url: str) -> str:
    try:
        return pathlib.Path(urlparse(url).path).suffix.lower()
    except Exception:
        return ""

def is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


# -----------------------------
# YAML parsing (your exact schema)
# -----------------------------
def load_kb_yaml(yaml_path: str) -> List[Seed]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    seeds: List[Seed] = []

    # domains -> domain -> section -> [urls]
    domains = data.get("domains", {}) or {}
    if not isinstance(domains, dict):
        raise ValueError("`domains` must be a mapping")

    for domain_name, domain_map in domains.items():
        if not isinstance(domain_map, dict):
            raise ValueError(f"`domains.{domain_name}` must be a mapping")
        for section_name, urls in domain_map.items():
            if not isinstance(urls, list):
                raise ValueError(f"`domains.{domain_name}.{section_name}` must be a list")
            for u in urls:
                if not isinstance(u, str):
                    continue
                u = canonical_url(u)
                if not is_http_url(u):
                    continue
                seeds.append(Seed(
                    url=u,
                    toplevel="domains",
                    domain=domain_name,
                    section=section_name,
                    subgroup=section_name,
                    embed=True,
                ))

    # meta_collections -> group -> [urls]
    meta_collections = data.get("meta_collections", {}) or {}
    if not isinstance(meta_collections, dict):
        raise ValueError("`meta_collections` must be a mapping")
    for group_name, urls in meta_collections.items():
        if not isinstance(urls, list):
            raise ValueError(f"`meta_collections.{group_name}` must be a list")
        for u in urls:
            if not isinstance(u, str):
                continue
            u = canonical_url(u)
            if not is_http_url(u):
                continue
            seeds.append(Seed(
                url=u,
                toplevel="meta_collections",
                domain="meta_collections",
                section=group_name,
                subgroup=group_name,
                embed=False,  # critical: don't embed link directories by default
            ))

    # de-duplicate exact URLs while preserving multiple categories if present
    # (here we keep duplicates across different metadata on purpose; exact duplicate same metadata removed)
    dedup = {}
    for s in seeds:
        key = (s.url, s.toplevel, s.domain, s.section, s.subgroup, s.embed)
        dedup[key] = s
    return list(dedup.values())


# -----------------------------
# Source type detection (GitHub repo/blob/gist/web)
# -----------------------------
def parse_github_url(url: str) -> Dict[str, Optional[str]]:
    """
    Returns dict with:
      kind: repo | blob | tree | gist | other
      owner, repo, branch, subpath
    """
    p = urlparse(url)
    host = p.netloc.lower()
    parts = [x for x in p.path.split("/") if x]

    result = {"kind": "other", "owner": None, "repo": None, "branch": None, "subpath": None}

    if host in {"gist.github.com", "www.gist.github.com"}:
        result["kind"] = "gist"
        return result

    if host not in {"github.com", "www.github.com"}:
        return result

    if len(parts) < 2:
        return result

    owner, repo = parts[0], parts[1]
    result["owner"] = owner
    result["repo"] = repo

    if len(parts) >= 5 and parts[2] == "blob":
        result["kind"] = "blob"
        result["branch"] = parts[3]
        result["subpath"] = "/".join(parts[4:])
        return result

    if len(parts) >= 5 and parts[2] == "tree":
        result["kind"] = "tree"
        result["branch"] = parts[3]
        result["subpath"] = "/".join(parts[4:])
        return result

    # plain repo URL
    result["kind"] = "repo"
    return result


def github_blob_to_raw(url: str) -> Optional[str]:
    info = parse_github_url(url)
    if info["kind"] != "blob":
        return None
    owner, repo, branch, subpath = info["owner"], info["repo"], info["branch"], info["subpath"]
    if not all([owner, repo, branch, subpath]):
        return None
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{subpath}"


# -----------------------------
# HTTP / Fetching
# -----------------------------
def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def fetch_url_text(url: str, session: requests.Session) -> Optional[str]:
    last_err = None
    for _ in range(HTTP_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}"
                time.sleep(0.3)
                continue
            # best effort decoding
            if r.encoding is None:
                r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception as e:
            last_err = str(e)
            time.sleep(0.3)
    logging.debug("fetch failed %s -> %s", url, last_err)
    return None


def fetch_url_bytes(url: str, session: requests.Session) -> Optional[bytes]:
    last_err = None
    for _ in range(HTTP_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}"
                time.sleep(0.3)
                continue
            return r.content
        except Exception as e:
            last_err = str(e)
            time.sleep(0.3)
    logging.debug("fetch bytes failed %s -> %s", url, last_err)
    return None


def fetch_with_playwright(url: str) -> Optional[str]:
    if not ENABLE_PLAYWRIGHT_FALLBACK:
        return None
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        logging.warning("Playwright not installed; skipping fallback for %s", url)
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=60_000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        logging.debug("Playwright fetch failed %s -> %s", url, e)
        return None


# -----------------------------
# Crawling (same-site BFS)
# -----------------------------
def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[str] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        abs_url = canonical_url(urljoin(base_url, href))
        if not is_http_url(abs_url):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        out.append(abs_url)
    return out


def same_site(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()


def should_skip_crawl_link(url: str) -> bool:
    ext = path_ext_from_url(url)
    if ext in SKIP_EXTENSIONS:
        return True
    lower = url.lower()
    if any(x in lower for x in BLOCK_CRAWL_PATH_SUBSTRINGS):
        return True
    return False


def crawl_site(seed_url: str, session: requests.Session, max_pages: int, max_depth: int) -> List[Tuple[str, str]]:
    """
    Returns list[(url, html)] for same-site pages.
    """
    seed_url = canonical_url(seed_url)
    queue: List[Tuple[str, int]] = [(seed_url, 0)]
    seen = {seed_url}
    pages: List[Tuple[str, str]] = []

    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)
        if should_skip_crawl_link(url):
            continue

        html = fetch_url_text(url, session)
        if html is None and ENABLE_PLAYWRIGHT_FALLBACK:
            html = fetch_with_playwright(url)
        if html is None:
            continue

        pages.append((url, html))

        if depth >= max_depth:
            time.sleep(CRAWL_DELAY_S)
            continue

        try:
            links = extract_links(html, url)
        except Exception:
            links = []

        for link in links:
            if not same_site(seed_url, link):
                continue
            if should_skip_crawl_link(link):
                continue
            if link not in seen:
                seen.add(link)
                queue.append((link, depth + 1))

        time.sleep(CRAWL_DELAY_S)

    return pages


# -----------------------------
# Cleaning / normalization
# -----------------------------
def html_to_clean_markdown(url: str, html: str) -> str:
    # Prefer trafilatura for boilerplate removal
    extracted = None
    try:
        extracted = trafilatura.extract(
            html,
            include_links=True,
            include_comments=False,
            output_format="markdown",
        )
    except TypeError:
        # Older trafilatura may not support output_format
        extracted = trafilatura.extract(
            html,
            include_links=True,
            include_comments=False,
        )
    except Exception:
        extracted = None

    if extracted and len(extracted.strip()) >= 200:
        body = extracted.strip()
    else:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.text.strip() if soup.title and soup.title.text else ""
        main = soup.find("main") or soup.body or soup
        body = md(str(main), heading_style="ATX").strip()
        if title and title.lower() not in body.lower()[:500]:
            body = f"# {title}\n\n{body}"

    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    # provenance header
    return f"# Source\n\n{url}\n\n---\n\n{body}\n"


def guess_title_from_markdown(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    if m:
        t = m.group(1).strip()
        return t[:180]
    return "Untitled"


def save_raw_html(url: str, html: str) -> pathlib.Path:
    uhash = sha1(url)
    name = safe_slug(urlparse(url).netloc + "_" + (urlparse(url).path.strip("/") or "index"))
    out = RAW_HTML_DIR / f"{name}__{uhash}.html"
    out.write_text(html, encoding="utf-8", errors="ignore")
    return out


def save_clean_doc(doc: SourceDoc) -> pathlib.Path:
    domain_dir = CLEAN_MD_DIR / safe_slug(doc.domain) / safe_slug(doc.section)
    domain_dir.mkdir(parents=True, exist_ok=True)
    out = domain_dir / f"{safe_slug(doc.title)}__{doc.doc_id}.md"

    frontmatter = {
        "doc_id": doc.doc_id,
        "seed_url": doc.seed_url,
        "source_url": doc.source_url,
        "domain": doc.domain,
        "section": doc.section,
        "subgroup": doc.subgroup,
        "toplevel": doc.toplevel,
        "source_kind": doc.source_kind,
        **(doc.extra or {}),
    }

    fm = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in frontmatter.items()) + "\n---\n\n"
    out.write_text(fm + doc.content, encoding="utf-8")
    return out


def _kb_bucket(doc: SourceDoc) -> str:
    section = (doc.section or "").lower()
    domain = (doc.domain or "").lower()
    if domain == "writeups" or section in {"blogs", "writeups"}:
        return "writeups"
    if section in {"cheat_sheets", "cheatsheets", "cheat-sheets", "cheatsheet"}:
        return "cheat_sheets"
    if section in {"tools", "tooling", "tool"}:
        return "tool_manuals"
    if section in {"methodologies", "methodology"}:
        return "methodologies"
    if section in {"references", "learning", "practice"} or domain in {"wiki", "wikis"}:
        return "wikis"
    return "wikis"


def save_kb_doc(doc: SourceDoc) -> pathlib.Path:
    base_dir = pathlib.Path(settings.knowledge_base_path)
    bucket = _kb_bucket(doc)
    out_dir = base_dir / bucket / safe_slug(doc.domain) / safe_slug(doc.section)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{safe_slug(doc.title)}__{doc.doc_id}.md"
    out.write_text(doc.content, encoding="utf-8")
    return out


# -----------------------------
# Repo handling
# -----------------------------
def repo_clone_url_from_github(url: str) -> Optional[str]:
    info = parse_github_url(url)
    if info["kind"] not in {"repo", "tree", "blob"}:
        return None
    owner, repo = info["owner"], info["repo"]
    if not owner or not repo:
        return None
    return f"https://github.com/{owner}/{repo}.git"

def clone_or_update_repo(repo_url: str) -> pathlib.Path:
    # repo_url must be github repo url or *.git
    info = parse_github_url(repo_url)
    if info["kind"] == "other":
        # fallback if repo_url already ends .git or not parseable
        p = urlparse(repo_url)
        parts = [x for x in p.path.split("/") if x]
        if len(parts) < 2:
            raise ValueError(f"Not a GitHub repo URL: {repo_url}")
        owner, repo = parts[0], parts[1].removesuffix(".git")
    else:
        owner, repo = info["owner"], info["repo"]
        assert owner and repo

    target = RAW_REPO_DIR / f"{safe_slug(owner)}__{safe_slug(repo)}"

    if (target / ".git").exists():
        try:
            r = Repo(str(target))
            r.remotes.origin.fetch()
            try:
                default_ref = r.git.symbolic_ref("refs/remotes/origin/HEAD")
                # e.g. refs/remotes/origin/main
                default_branch = default_ref.split("/")[-1]
            except Exception:
                default_branch = "main"
            r.git.checkout(default_branch)
            r.git.pull("--ff-only")
        except Exception as e:
            logging.warning("Repo update failed (%s), keeping local copy: %s", target.name, e)
        return target

    if target.exists():
        # clean stale folder
        for p in sorted(target.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except Exception:
                pass
        try:
            target.rmdir()
        except Exception:
            pass

    clone_url = repo_url if repo_url.endswith(".git") else repo_clone_url_from_github(repo_url)
    if not clone_url:
        raise ValueError(f"Cannot derive clone URL from {repo_url}")

    Repo.clone_from(clone_url, str(target))
    return target


def read_text_file(path: pathlib.Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None


def looks_binary_or_skip(path: pathlib.Path) -> bool:
    ext = path.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return True
    return False


def should_index_repo_file(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    if looks_binary_or_skip(path):
        return False
    try:
        if path.stat().st_size > MAX_REPO_FILE_BYTES:
            return False
    except Exception:
        return False

    ext = path.suffix.lower()
    if ext in REPO_DOC_EXTS:
        return True
    if INDEX_CODE_FILES_FROM_REPOS and ext in REPO_CODE_EXTS:
        return True

    # Include common files without extensions
    if path.name.lower() in {"readme", "license", "copying", "changelog", "makefile"}:
        return True

    return False


def repo_files_to_docs(repo_root: pathlib.Path, seed: Seed, seed_repo_url: str) -> List[SourceDoc]:
    docs: List[SourceDoc] = []
    for path in repo_root.rglob("*"):
        if not should_index_repo_file(path):
            continue

        text = read_text_file(path)
        if not text:
            continue
        if len(text.strip()) < 50:
            continue
        if len(text) > MAX_TEXT_CHARS_PER_FILE:
            text = text[:MAX_TEXT_CHARS_PER_FILE] + "\n\n[TRUNCATED]\n"

        rel = path.relative_to(repo_root).as_posix()
        lang = (path.suffix.lower().lstrip(".") or "text")
        wrapped = (
            f"# Source\n\n{seed_repo_url}\n\n"
            f"## File\n\n{rel}\n\n---\n\n"
            f"```{lang}\n{text.strip()}\n```\n"
        )

        source_url = f"{seed_repo_url}#file={rel}"
        doc_id = sha1(f"{seed_repo_url}::{rel}::{sha1(text)}")
        doc = SourceDoc(
            doc_id=doc_id,
            seed_url=seed.url,
            source_url=source_url,
            domain=seed.domain,
            section=seed.section,
            subgroup=seed.subgroup,
            toplevel=seed.toplevel,
            source_kind="repo_file",
            title=f"{repo_root.name}:{rel}",
            content=wrapped,
            extra={"repo_root": repo_root.name, "repo_relpath": rel},
        )
        docs.append(doc)
    return docs


# -----------------------------
# Chunking
# -----------------------------
ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    try:
        return len(ENCODER.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def split_markdown_heading_aware(text: str) -> List[str]:
    # Split on markdown headings, preserving the heading as start of each section
    parts = re.split(r"\n(?=#{1,6}\s)", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [text.strip()]

def pack_sections_into_chunks(sections: List[str]) -> List[str]:
    chunks: List[str] = []
    buf = ""

    def append_chunk(s: str):
        s = s.strip()
        if s:
            chunks.append(s)

    for sec in sections:
        if not buf:
            buf = sec
            continue

        candidate = f"{buf}\n\n{sec}"
        if count_tokens(candidate) <= CHUNK_TARGET_TOKENS:
            buf = candidate
        else:
            append_chunk(buf)
            # overlap from previous buffer
            if CHUNK_OVERLAP_TOKENS > 0 and buf:
                toks = ENCODER.encode(buf)
                tail = ENCODER.decode(toks[-CHUNK_OVERLAP_TOKENS:]) if len(toks) > CHUNK_OVERLAP_TOKENS else buf
                buf = f"{tail}\n\n{sec}"
            else:
                buf = sec

    if buf.strip():
        append_chunk(buf)

    # final split for any oversized chunks (rare)
    final_chunks: List[str] = []
    for ch in chunks:
        if count_tokens(ch) <= CHUNK_TARGET_TOKENS * 1.3:
            final_chunks.append(ch)
            continue
        # fallback by paragraphs
        paras = [p for p in re.split(r"\n\s*\n", ch) if p.strip()]
        b = ""
        for p in paras:
            cand = f"{b}\n\n{p}" if b else p
            if count_tokens(cand) <= CHUNK_TARGET_TOKENS:
                b = cand
            else:
                if b.strip():
                    final_chunks.append(b.strip())
                b = p
        if b.strip():
            final_chunks.append(b.strip())

    return [c for c in final_chunks if count_tokens(c) >= MIN_CHUNK_TOKENS or len(c.strip()) > 80]


def doc_to_chunks(doc: SourceDoc) -> List[ChunkRecord]:
    sections = split_markdown_heading_aware(doc.content)
    texts = pack_sections_into_chunks(sections)
    out: List[ChunkRecord] = []

    for i, text in enumerate(texts):
        chunk_id = sha1(f"{doc.doc_id}::{i}::{sha1(text)}")
        out.append(ChunkRecord(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            seed_url=doc.seed_url,
            source_url=doc.source_url,
            domain=doc.domain,
            section=doc.section,
            subgroup=doc.subgroup,
            toplevel=doc.toplevel,
            source_kind=doc.source_kind,
            title=doc.title,
            chunk_index=i,
            text=text,
        ))
    return out


# -----------------------------
# Chroma indexing
# -----------------------------
def get_chroma_collection(name: str = CHROMA_COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=name)

def upsert_chunks_chroma(chunks: List[ChunkRecord], embedder: SentenceTransformer, collection_name: str = CHROMA_COLLECTION_NAME) -> None:
    if not chunks:
        return
    col = get_chroma_collection(collection_name)
    BATCH = 64

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        docs = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metas = []
        for c in batch:
            metas.append({
                "doc_id": c.doc_id,
                "seed_url": c.seed_url,
                "source_url": c.source_url,
                "domain": c.domain,
                "section": c.section,
                "subgroup": c.subgroup,
                "toplevel": c.toplevel,
                "source_kind": c.source_kind,
                "title": c.title[:500],  # safety
                "chunk_index": int(c.chunk_index),
            })
        vectors = embedder.encode(docs, normalize_embeddings=True).tolist()
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)


# -----------------------------
# Per-seed pipeline
# -----------------------------
def pick_crawl_policy(seed: Seed) -> Dict[str, int | bool]:
    base = CRAWL_POLICY_BY_SECTION.get(seed.section, {"crawl": False, "max_pages": 1, "max_depth": 0}).copy()

    # deeper crawl for wiki domain references/learning
    if seed.domain in WIKI_LIKE_DOMAINS and seed.section in {"references", "learning"}:
        base["crawl"] = True
        base["max_pages"] = WIKI_MAX_PAGES
        base["max_depth"] = WIKI_MAX_DEPTH

    # Practice platforms often have login-heavy pages; keep shallow
    if seed.domain == "practice":
        base["crawl"] = False
        base["max_pages"] = 1
        base["max_depth"] = 0

    # Writeup blogs: single page
    if seed.domain == "writeups":
        base["crawl"] = False
        base["max_pages"] = 1
        base["max_depth"] = 0

    # Meta collections: store seed only / optionally clone but don't embed
    if seed.toplevel in NO_EMBED_TOPLEVELS:
        # still process repo/page lightly for discovery if desired
        base["crawl"] = False
        base["max_pages"] = 1
        base["max_depth"] = 0

    return base


def save_manifest(name: str, rows: List[Dict[str, Any]]) -> pathlib.Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"{safe_slug(name)}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def process_web_seed(seed: Seed, session: requests.Session) -> List[SourceDoc]:
    policy = pick_crawl_policy(seed)
    docs: List[SourceDoc] = []

    pages: List[Tuple[str, str]]
    if policy["crawl"]:
        pages = crawl_site(
            seed.url, session=session,
            max_pages=int(policy["max_pages"]),
            max_depth=int(policy["max_depth"])
        )
    else:
        html = fetch_url_text(seed.url, session)
        if html is None and ENABLE_PLAYWRIGHT_FALLBACK:
            html = fetch_with_playwright(seed.url)
        pages = [(seed.url, html)] if html else []

    seen_content = set()
    for url, html in pages:
        if not html:
            continue

        # raw save
        save_raw_html(url, html)

        cleaned = html_to_clean_markdown(url, html)
        c_hash = sha1(cleaned)
        if c_hash in seen_content:
            continue
        seen_content.add(c_hash)

        title = guess_title_from_markdown(cleaned)
        doc_id = sha1(f"{seed.url}::{url}::{c_hash}")
        doc = SourceDoc(
            doc_id=doc_id,
            seed_url=seed.url,
            source_url=url,
            domain=seed.domain,
            section=seed.section,
            subgroup=seed.subgroup,
            toplevel=seed.toplevel,
            source_kind="webpage",
            title=title,
            content=cleaned,
            extra={},
        )
        docs.append(doc)

    return docs


def process_github_blob_seed(seed: Seed, session: requests.Session) -> List[SourceDoc]:
    raw_url = github_blob_to_raw(seed.url)
    if not raw_url:
        return []

    content_bytes = fetch_url_bytes(raw_url, session)
    if content_bytes is None:
        # fallback: treat original blob page as webpage
        return process_web_seed(seed, session)

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = content_bytes.decode("utf-8", errors="ignore")

    info = parse_github_url(seed.url)
    rel = info.get("subpath") or "blob_file"
    ext = pathlib.Path(rel).suffix.lower().lstrip(".") or "text"
    wrapped = (
        f"# Source\n\n{seed.url}\n\n"
        f"## File\n\n{rel}\n\n---\n\n"
        f"```{ext}\n{text.strip()}\n```\n"
    )
    doc_id = sha1(f"{seed.url}::{rel}::{sha1(text)}")
    doc = SourceDoc(
        doc_id=doc_id,
        seed_url=seed.url,
        source_url=seed.url,
        domain=seed.domain,
        section=seed.section,
        subgroup=seed.subgroup,
        toplevel=seed.toplevel,
        source_kind="github_blob",
        title=f"{info.get('repo') or 'repo'}:{rel}",
        content=wrapped,
        extra={"github_raw_url": raw_url},
    )
    return [doc]


def process_github_repo_seed(seed: Seed) -> List[SourceDoc]:
    repo_url = repo_clone_url_from_github(seed.url) or seed.url
    try:
        repo_root = clone_or_update_repo(repo_url)
    except Exception as e:
        logging.warning("Repo clone/update failed for %s: %s", seed.url, e)
        return []

    return repo_files_to_docs(repo_root, seed, seed_repo_url=seed.url)


def process_seed(seed: Seed, session: requests.Session) -> List[SourceDoc]:
    gh = parse_github_url(seed.url)

    if gh["kind"] == "blob":
        return process_github_blob_seed(seed, session)

    if gh["kind"] in {"repo", "tree"}:
        return process_github_repo_seed(seed)

    # gist and everything else -> web page handling
    return process_web_seed(seed, session)


# -----------------------------
# End-to-end run
# -----------------------------
def ingest_from_yaml(yaml_path: str, embed: bool = True) -> None:
    ensure_dirs()

    seeds = load_kb_yaml(yaml_path)
    logging.info("Loaded %d seeds from %s", len(seeds), yaml_path)

    session = requests_session()

    # Manifest of seeds
    manifest_rows = [asdict(s) for s in seeds]
    save_manifest("seeds", manifest_rows)

    all_docs: List[SourceDoc] = []
    per_seed_counts: List[Dict[str, Any]] = []
    seen_doc_hashes = set()

    for seed in tqdm(seeds, desc="Processing seeds"):
        try:
            docs = process_seed(seed, session)
        except Exception as e:
            logging.exception("Seed failed: %s (%s)", seed.url, e)
            docs = []

        kept_docs = []
        for d in docs:
            dh = sha1(d.content)
            if dh in seen_doc_hashes:
                continue
            seen_doc_hashes.add(dh)
            save_clean_doc(d)
            save_kb_doc(d)
            kept_docs.append(d)
            all_docs.append(d)

        per_seed_counts.append({
            "seed_url": seed.url,
            "toplevel": seed.toplevel,
            "domain": seed.domain,
            "section": seed.section,
            "embed": seed.embed,
            "docs_extracted": len(kept_docs),
        })

    save_manifest("ingest_results", per_seed_counts)
    logging.info("Extracted %d cleaned docs", len(all_docs))

    # Chunk
    all_chunks: List[ChunkRecord] = []
    for d in tqdm(all_docs, desc="Chunking docs"):
        all_chunks.extend(doc_to_chunks(d))

    # Save chunks manifest
    chunks_path = CHUNKS_DIR / f"chunks__{int(time.time())}.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    logging.info("Created %d chunks -> %s", len(all_chunks), chunks_path)

    if not embed:
        logging.info("Embedding skipped (--no-embed).")
        return

    # Embed only chunks from embed=True seeds / non-meta
    embeddable_chunks = [c for c in all_chunks if c.toplevel not in NO_EMBED_TOPLEVELS]
    logging.info("Embeddable chunks: %d (meta_collections excluded)", len(embeddable_chunks))

    if embeddable_chunks:
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        upsert_chunks_chroma(embeddable_chunks, embedder, CHROMA_COLLECTION_NAME)
        logging.info("Indexed into Chroma collection '%s' at %s", CHROMA_COLLECTION_NAME, CHROMA_DIR.resolve())
    else:
        logging.info("No embeddable chunks found.")


# -----------------------------
# Query helper (optional)
# -----------------------------
def query_kb(query: str, k: int = 5, domain: Optional[str] = None, section: Optional[str] = None) -> None:
    col = get_chroma_collection(CHROMA_COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    q_vec = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    where: Dict[str, Any] = {}
    if domain:
        where["domain"] = domain
    if section:
        where["section"] = section
    if not where:
        where = None  # type: ignore

    res = col.query(query_embeddings=[q_vec], n_results=k, where=where)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    for i, (_id, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
        print("\n" + "=" * 100)
        print(f"[{i}] {meta.get('title', 'Untitled')} | {meta.get('domain')}/{meta.get('section')} | {meta.get('source_kind')}")
        print(f"Source URL: {meta.get('source_url')}")
        print("-" * 100)
        print((doc or "")[:1800])


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG scraper/ingestor for security KB YAML")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Run full ingestion (crawl -> clean -> chunk -> embed)")
    p_ingest.add_argument("--yaml", default="kb_sources.yaml", help="Path to YAML config")
    p_ingest.add_argument("--no-embed", action="store_true", help="Skip embeddings/indexing")

    p_query = sub.add_parser("query", help="Query local Chroma KB")
    p_query.add_argument("text", help="Search query")
    p_query.add_argument("-k", type=int, default=5, help="Top-k results")
    p_query.add_argument("--domain", default=None, help="Optional domain filter")
    p_query.add_argument("--section", default=None, help="Optional section filter")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "ingest":
        ingest_from_yaml(args.yaml, embed=(not args.no_embed))
    elif args.cmd == "query":
        query_kb(args.text, k=args.k, domain=args.domain, section=args.section)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
