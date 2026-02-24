#!/bin/bash
set -e  # Exit on error

echo "Starting AI CTF Solver Setup..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Docker is not running or installed. Please fix this and retry."
    exit 1
fi

# 3. Check for Ollama and Model
if command -v ollama &> /dev/null; then
    echo "Pulling Qwen model (if not present)..."
    ollama pull qwen2.5-coder:14b-instruct
else
    echo "Ollama not found. If using a local model, please install it first."
fi

# Sync dependencies
echo "Syncing Python dependencies..."
uv sync

# Build and Start Sandbox
echo "Initializing Kali Sandbox..."
docker compose build ctf-sandbox
docker compose up -d ctf-sandbox

# Initialize Knowledge Base (RAG)
echo "Initializing Knowledge Base..."
if [ -d "./data/knowledge_base" ]; then
    uv run python -c "from src.rag.rag import ingest_knowledge_base; ingest_knowledge_base('./data/knowledge_base', force=False, mode='all')"
else
    echo "No knowledge_base found in ./data/knowledge_base. Skipping indexing."
fi

echo "Setup complete! Run your agent with: uv run ig-ctf-solver --help"
