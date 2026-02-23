import subprocess
from dataclasses import dataclass

from src.config import settings


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


class SandboxRunner:
    def __init__(self, container: str | None = None, workdir: str | None = None, timeout_s: int | None = None):
        self.container = container or settings.sandbox_container
        self.workdir = workdir or settings.sandbox_workdir
        self.timeout_s = timeout_s or settings.tool_timeout_seconds

    def run(self, command: str) -> CommandResult:
        shell_cmd = f"cd {self.workdir} && {command}"
        docker_cmd = [
            "docker", "exec", self.container,
            "bash", "-lc", shell_cmd
        ]
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired:
            return CommandResult(stdout="", stderr=f"Command timed out after {self.timeout_s} seconds.", exit_code=124)

        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode
        )

    def copy_to_container(self, source_path: str, dest_path: str) -> CommandResult:
        docker_cmd = ["docker", "cp", source_path, f"{self.container}:{dest_path}"]
        return self._run_docker_cmd(docker_cmd)

    def copy_from_container(self, source_path: str, dest_path: str) -> CommandResult:
        docker_cmd = ["docker", "cp", f"{self.container}:{source_path}", dest_path]
        return self._run_docker_cmd(docker_cmd)

    def _run_docker_cmd(self, docker_cmd: list[str]) -> CommandResult:
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired:
            return CommandResult(stdout="", stderr=f"Command timed out after {self.timeout_s} seconds.", exit_code=124)

        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode
        )


def run_bash_in_sandbox(command: str):
    runner = SandboxRunner()
    result = runner.run(command)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code
    }
