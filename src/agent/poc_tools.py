import subprocess
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Setup
llm = ChatOllama(model="qwen2.5-coder:7b")
CONTAINER = "ctf-sandbox"

def run_bash(cmd):
    print(f"Executing in Sandbox: {cmd}")
    process = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-c", cmd],
        capture_output=True, text=True
    )
    return f"STDOUT: {process.stdout}\nSTDERR: {process.stderr}"

# Challenge Input
challenge = "Check the /etc/os-release file and tell me which Linux distro is running."

# Simple Loop
msg = [SystemMessage(content="You are a helper. Output ONLY the bash command."),
       HumanMessage(content=challenge)]

response = llm.invoke(msg)
cmd = response.content.strip().strip('`')
result = run_bash(cmd)

print(f"AI Result: \n{result}")