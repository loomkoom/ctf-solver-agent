import subprocess

def run_bash_in_sandbox(command: str):
    """Executes a command inside the 'ctf-sandbox' container."""
    # We use 'docker exec' to run the command inside the already-running container
    docker_cmd = [
        "docker", "exec", "ctf-sandbox",
        "bash", "-c", command
    ]
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=30 # Safety timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 30 seconds."}