import subprocess
import re
import chromadb
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# --- CONFIG ---
MODEL = "qwen2.5-coder:7b"
CONTAINER = "ctf-sandbox"
DB_PATH = "./chroma_db"

llm = ChatOllama(model=MODEL, temperature=0)

# --- RAG SETUP ---
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("ctf_knowledge")

def search_knowledge(query: str):
    """Retrieves context from the knowledge base."""
    results = collection.query(query_texts=[query], n_results=2)
    if results['documents'] and results['documents'][0]:
        # Merge source path into the output so agent can find artifacts
        context = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context.append(f"Source: {meta['source_path']}\nContent: {doc}")
        return "\n\n".join(context)
    return "No relevant write-ups found."

# --- TOOLS ---
def run_in_sandbox(cmd: str):
    docker_cmd = ["docker", "exec", CONTAINER, "bash", "-c", cmd]
    try:
        res = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=15)
        return f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN LOOP ---
def agent_loop(challenge: str):
    # 1. RAG STEP: Look up similar challenges first
    print(f"🔍 Searching knowledge base for: {challenge[:50]}...")
    hints = search_knowledge(challenge)

    system_prompt = (
        "You are an elite CTF solver. Output ONLY a bash command inside ```bash blocks. "
        "Use the provided knowledge base hints to inform your strategy. "
        "If you see a SOURCE PATH in the hints, remember you can check 'data/artifacts/[path]' if needed. "
        "If you find the flag, output 'FLAG_FOUND: [flag]'."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"KNOWLEDGE BASE HINTS:\n{hints}\n\nCHALLENGE: {challenge}")
    ]

    for i in range(7): # Slightly more rounds for research-heavy tasks
        print(f"\n--- Round {i+1} ---")
        response = llm.invoke(messages)
        content = response.content
        print(f"AI Thought: {content}")

        if "FLAG_FOUND" in content:
            print("🎯 Success!")
            break

        # Extract command
        match = re.search(r"```bash\n(.*?)\n```", content, re.DOTALL)
        cmd = match.group(1).strip() if match else None

        if cmd:
            print(f"🛠️ Executing: {cmd}")
            result = run_in_sandbox(cmd)
            messages.append(response)
            messages.append(HumanMessage(content=f"Output: {result}"))
        else:
            print("⚠️ No command found. AI might be stuck or explaining.")
            messages.append(HumanMessage(content="Please provide a bash command to proceed."))

if __name__ == "__main__":
    # Ensure your sandbox is running!
    # docker run -d --name ctf-sandbox kalilinux/kali-rolling sleep infinity

    test_chall = "I have a file at /tmp/task.bin. Analyze it and find the flag."

    # POC Setup: Create a fake binary and flag
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", "echo 'flag{rag_is_working}' > /tmp/flag.txt"])

    agent_loop(test_chall)