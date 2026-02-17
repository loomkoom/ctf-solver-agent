docker run -d --name ctf-sandbox kalilinux/kali-rolling sleep infinity

irm https://ollama.com/install.ps1 | iex
ollama pull deepseek-v3
(400GB)
ollama pull qwen2.5-coder:7b
(4gb)

# Core security & reversing
uv add sympy pwntools z3-solver ROPGadget

# Mathematics & Crypto (Heavy)
uv add gmpy2 pycryptodomex primefac factordb-pycli

# Media & Forensics
uv add opencv-python Pillow png-parser morse-audio-decoder

# Web & Data
uv add requests beautifulsoup4 Flask pandas numpy

uv add angr sagemath