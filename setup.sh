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
    ollama pull qwen2.5-coder:7b
else
    echo "Ollama not found. If using a local model, please install it first."
fi

# Sync dependencies
echo "Syncing Python dependencies..."
uv sync

# Build and Start Sandbox
echo "Initializing Kali Sandbox..."
# Ensure any old container is removed
docker rm -f ctf-sandbox 2>/dev/null || true
docker build -t ctf-sandbox -f src/sandbox/DOCKERFILE .
docker run -d --name ctf-sandbox ctf-sandbox sleep infinity

# Initialize Knowledge Base (RAG)
echo "Initializing Knowledge Base..."
if [ -d "./data/knowledge_base/writeups" ]; then
    uv run python -c "from src.tools.rag import ingest_writeups; ingest_writeups('./data/knowledge_base/writeups')"
else
    echo "No writeups found in ./data/knwoledge_base/writeups. Skipping indexing."
fi

echo "Setup complete! Run your agent with: uv run python src/main.py"