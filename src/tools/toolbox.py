import base64
import json
from dataclasses import dataclass

from src.tools.bash import SandboxRunner


@dataclass
class ToolResult:
    tool: str
    command: str
    stdout: str
    stderr: str
    exit_code: int
    parsed: dict | None = None


class Toolbox:
    def __init__(self, runner: SandboxRunner | None = None):
        self.runner = runner or SandboxRunner()

    def run(self, command: str) -> ToolResult:
        result = self.runner.run(command)
        return ToolResult(
            tool="bash",
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    def copy_to_container(self, source_path: str, dest_path: str) -> ToolResult:
        result = self.runner.copy_to_container(source_path, dest_path)
        return ToolResult(
            tool="docker_cp_to",
            command=f"{source_path} -> {dest_path}",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    def copy_from_container(self, source_path: str, dest_path: str) -> ToolResult:
        result = self.runner.copy_from_container(source_path, dest_path)
        return ToolResult(
            tool="docker_cp_from",
            command=f"{source_path} -> {dest_path}",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    # --- Filesystem ---
    def read_file(self, path: str) -> ToolResult:
        return self.run(f"cat {self._shell_escape(path)}")

    def write_file(self, path: str, content: str) -> ToolResult:
        payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = (
            "python3 - <<'PY'\n"
            "import base64\n"
            "from pathlib import Path\n"
            f"Path({json.dumps(path)}).write_bytes(base64.b64decode({json.dumps(payload)}))\n"
            "PY"
        )
        return self.run(cmd)

    def list_dir(self, path: str = ".") -> ToolResult:
        result = self.run(f"ls -la {self._shell_escape(path)}")
        files = [line.split()[-1] for line in result.stdout.splitlines() if line and not line.startswith("total")]
        result.parsed = {"files": files}
        return result

    def grep(self, pattern: str, path: str = ".") -> ToolResult:
        cmd = (
            "command -v rg >/dev/null 2>&1 && "
            f"rg -n --no-heading {self._shell_escape(pattern)} {self._shell_escape(path)} "
            "|| "
            f"grep -R -n {self._shell_escape(pattern)} {self._shell_escape(path)}"
        )
        return self.run(cmd)

    def find(self, path: str, name: str) -> ToolResult:
        return self.run(f"find {self._shell_escape(path)} -name {self._shell_escape(name)}")

    # --- Analysis ---
    def binwalk(self, path: str) -> ToolResult:
        return self.run(f"binwalk {self._shell_escape(path)}")

    def checksec(self, path: str) -> ToolResult:
        return self.run(f"checksec --file={self._shell_escape(path)}")

    def strings(self, path: str) -> ToolResult:
        return self.run(f"strings {self._shell_escape(path)}")

    def exiftool(self, path: str) -> ToolResult:
        return self.run(f"exiftool {self._shell_escape(path)}")

    def ghidra_headless(self, project_dir: str, project_name: str, binary_path: str) -> ToolResult:
        cmd = (
            f"analyzeHeadless {self._shell_escape(project_dir)} "
            f"{self._shell_escape(project_name)} -import {self._shell_escape(binary_path)}"
        )
        return self.run(cmd)

    # --- Execution ---
    def gdb_pwndbg(self, binary_path: str, gdb_commands: list[str] | None = None) -> ToolResult:
        commands = gdb_commands or ["set pagination off", "info files", "quit"]
        gdb_args = " ".join([f"-ex {self._shell_escape(cmd)}" for cmd in commands])
        return self.run(f"gdb -q {gdb_args} --args {self._shell_escape(binary_path)}")

    def pwntools(self, script: str) -> ToolResult:
        return self._python(script)

    def python(self, script: str) -> ToolResult:
        return self._python(script)

    # --- Web ---
    def curl(self, url: str, args: str = "") -> ToolResult:
        return self.run(f"curl {args} {self._shell_escape(url)}")

    def nmap(self, target: str, args: str = "-sV -sC") -> ToolResult:
        return self.run(f"nmap {args} {self._shell_escape(target)}")

    def sqlmap(self, args: str) -> ToolResult:
        return self.run(f"sqlmap {args}")

    def _python(self, script: str) -> ToolResult:
        cmd = (
            "python3 - <<'PY'\n"
            f"{script}\n"
            "PY"
        )
        return self.run(cmd)

    def _shell_escape(self, value: str) -> str:
        return json.dumps(value)
