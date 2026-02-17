import subprocess
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# --- CONFIG ---
MODEL = "qwen2.5-coder:7b"
CONTAINER = "ctf-sandbox"

llm = ChatOllama(model=MODEL, temperature=0)

def run_in_sandbox(cmd: str):
    """Executes command in the Docker sandbox."""
    docker_cmd = ["docker", "exec", CONTAINER, "bash", "-c", cmd]
    res = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=10)
    return f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"

def agent_loop(challenge: str):
    messages = [
        SystemMessage(content="You are a CTF solver. Output ONLY a bash command inside ```bash blocks to solve the challenge. If you find the flag, output 'FLAG_FOUND: [flag]'."),
        HumanMessage(content=f"Challenge: {challenge}")
    ]

    for i in range(5):  # Limit to 5 attempts for POC
        print(f"\n--- Round {i+1} ---")
        response = llm.invoke(messages)
        content = response.content
        print(f"AI Thought: {content}")

        if "FLAG_FOUND" in content:
            print("Success!")
            break

        # Extract command
        match = re.search(r"```bash\n(.*?)\n```", content, re.DOTALL)
        cmd = match.group(1).strip() if match else None

        if cmd:
            print(f"🛠Executing: {cmd}")
            result = run_in_sandbox(cmd)
            messages.append(response)
            messages.append(HumanMessage(content=f"Output: {result}"))
        else:
            print("No command found.")
            break

if __name__ == "__main__":
    test_chall = "Look in /tmp/flag.txt and tell me what is inside."
    # Setup the file first for the POC test
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", "echo 'flag{poc_success}' > /tmp/flag.txt"])
    agent_loop(test_chall)