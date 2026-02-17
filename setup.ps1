Write-Host "Starting AI CTF Solver Setup..." -ForegroundColor Cyan

# Check for uv
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# Check for Docker
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker Desktop." -ForegroundColor Red
    exit
}

# Check for Ollama and Model
if (!(Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama CLI not found. Please install from ollama.com" -ForegroundColor Red
} else {
    Write-Host "Checking Ollama service and Qwen model..." -ForegroundColor Green
    # Pull Qwen if it doesn't exist (ollama pull is idempotent)
    ollama pull qwen2.5-coder:7b
}

# Sync dependencies
Write-Host "Syncing Python dependencies..." -ForegroundColor Green
uv sync

# Build and Start Sandbox
Write-Host "Initializing Kali Sandbox..." -ForegroundColor Green
docker rm -f ctf-sandbox 2>$null
docker build -t ctf-sandbox -f src/sandbox/DOCKERFILE .
docker run -d --name ctf-sandbox ctf-sandbox sleep infinity

# Initialize Knowledge Base (RAG)
Write-Host "Initializing Knowledge Base..." -ForegroundColor Green
if (Test-Path "./data/knowledge_base/writeups") {
    uv run python -c "from src.tools.rag import ingest_writeups; ingest_writeups('./data/knowledge_base/writeups')"
} else {
    Write-Warning "No writeups found in './data/knowledge_base/writeups'. Skipping indexing."
}

Write-Host "Setup complete! Run your agent with: uv run python src/main.py" -ForegroundColor Cyan