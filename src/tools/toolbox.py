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
    log_path: str | None = None


class Toolbox:
    def __init__(self, runner: SandboxRunner | None = None):
        self.runner = runner or SandboxRunner()

    def run(self, command: str, tool_name: str = "bash") -> ToolResult:
        result = self.runner.run(command)
        return ToolResult(
            tool=tool_name,
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
        return self.run(f"cat {self._shell_escape(path)}", tool_name="read_file")

    def write_file(self, path: str, content: str) -> ToolResult:
        payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = (
            "python3 - <<'PY'\n"
            "import base64\n"
            "from pathlib import Path\n"
            f"Path({json.dumps(path)}).write_bytes(base64.b64decode({json.dumps(payload)}))\n"
            "PY"
        )
        return self.run(cmd, tool_name="write_file")

    def list_dir(self, path: str = ".") -> ToolResult:
        result = self.run(f"ls -la {self._shell_escape(path)}", tool_name="list_dir")
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
        return self.run(cmd, tool_name="grep")

    def find(self, path: str, name: str) -> ToolResult:
        return self.run(f"find {self._shell_escape(path)} -name {self._shell_escape(name)}", tool_name="find")

    def hash_file(self, path: str, algo: str = "sha256") -> ToolResult:
        algo = (algo or "sha256").lower()
        cmd = "sha256sum"
        if algo in {"sha1", "sha"}:
            cmd = "sha1sum"
        elif algo in {"md5", "md5sum"}:
            cmd = "md5sum"
        return self.run(f"{cmd} {self._shell_escape(path)}", tool_name="hash_file")

    def extract_archive(self, path: str, dest_dir: str | None = None) -> ToolResult:
        dest = dest_dir or f"{path}_extracted"
        cmd = (
            f"mkdir -p {self._shell_escape(dest)} && "
            f"7z x -y -o{self._shell_escape(dest)} {self._shell_escape(path)}"
        )
        return self.run(cmd, tool_name="extract_archive")

    # --- Analysis ---
    def file_info(self, path: str) -> ToolResult:
        return self.run(f"file {self._shell_escape(path)}", tool_name="file_info")

    def binwalk(self, path: str) -> ToolResult:
        return self.run(f"binwalk {self._shell_escape(path)}", tool_name="binwalk")

    def checksec(self, path: str) -> ToolResult:
        return self.run(f"checksec --file={self._shell_escape(path)}", tool_name="checksec")

    def strings(self, path: str) -> ToolResult:
        return self.run(f"strings {self._shell_escape(path)}", tool_name="strings")

    def exiftool(self, path: str) -> ToolResult:
        return self.run(f"exiftool {self._shell_escape(path)}", tool_name="exiftool")

    def stegseek(self, path: str, wordlist: str | None = None) -> ToolResult:
        if wordlist:
            return self.run(
                f"stegseek {self._shell_escape(path)} {self._shell_escape(wordlist)}",
                tool_name="stegseek",
            )
        return self.run(f"stegseek {self._shell_escape(path)}", tool_name="stegseek")

    def zsteg(self, path: str) -> ToolResult:
        return self.run(f"zsteg {self._shell_escape(path)}", tool_name="zsteg")

    def ciphey(self, text_or_path: str, args: str = "") -> ToolResult:
        base_args = args or "-f"
        cmd = f"ciphey {base_args} {self._shell_escape(text_or_path)}".strip()
        return self.run(cmd, tool_name="ciphey")

    def lemmeknow(self, text_or_path: str, json_output: bool = True, args: str = "") -> ToolResult:
        flags = "--json" if json_output else ""
        cmd = f"lemmeknow {flags} {args} {self._shell_escape(text_or_path)}".strip()
        return self.run(cmd, tool_name="lemmeknow")

    def ghidra_headless(self, project_dir: str, project_name: str, binary_path: str) -> ToolResult:
        cmd = (
            f"analyzeHeadless {self._shell_escape(project_dir)} "
            f"{self._shell_escape(project_name)} -import {self._shell_escape(binary_path)}"
        )
        return self.run(cmd, tool_name="ghidra_headless")

    def radare2_json(self, binary_path: str, commands: str | list[str]) -> ToolResult:
        if isinstance(commands, list):
            cmd_str = ";".join(commands)
        else:
            cmd_str = commands
        return self.run(
            f"r2 -q -c {self._shell_escape(cmd_str)} -j {self._shell_escape(binary_path)}",
            tool_name="radare2_json",
        )

    def objdump(self, binary_path: str, args: str = "-d") -> ToolResult:
        return self.run(f"objdump {args} {self._shell_escape(binary_path)}", tool_name="objdump")

    def readelf(self, binary_path: str, args: str = "-a") -> ToolResult:
        return self.run(f"readelf {args} {self._shell_escape(binary_path)}", tool_name="readelf")

    # --- Execution ---
    def gdb_pwndbg(self, binary_path: str, gdb_commands: list[str] | None = None) -> ToolResult:
        commands = gdb_commands or ["set pagination off", "info files", "quit"]
        gdb_args = " ".join([f"-ex {self._shell_escape(cmd)}" for cmd in commands])
        return self.run(
            f"gdb -q {gdb_args} --args {self._shell_escape(binary_path)}",
            tool_name="gdb_pwndbg",
        )

    def pwntools(self, script: str) -> ToolResult:
        return self._python(script, tool_name="pwntools")

    def pwntools_template(
        self,
        binary_path: str = "",
        offset: int = 0,
        target_addr: str = "0x0",
        template_name: str = "basic",
        **kwargs,
    ) -> ToolResult:
        from src.tools.templates import render_template
        payload = render_template(
            template_name,
            binary_path=binary_path,
            offset=offset,
            target_addr=target_addr,
            **kwargs,
        )
        return self._python(payload, tool_name="pwntools_template")

    def pwntools_ret2win(
        self,
        binary_path: str,
        offset: int = 0,
        win_symbol: str = "win",
        win_addr: str = "0x0",
    ) -> ToolResult:
        from src.tools.templates import render_template
        payload = render_template(
            "ret2win",
            binary_path=binary_path,
            offset=offset,
            win_symbol=win_symbol,
            win_addr=win_addr,
        )
        return self._python(payload, tool_name="pwntools_ret2win")

    def pwntools_fmt_leak(self, binary_path: str, leak_index: int = 6) -> ToolResult:
        from src.tools.templates import render_template
        payload = render_template(
            "fmt_leak",
            binary_path=binary_path,
            leak_index=leak_index,
        )
        return self._python(payload, tool_name="pwntools_fmt_leak")

    def pwntools_rop_system(
        self,
        binary_path: str,
        offset: int = 0,
        system_addr: str = "0x0",
    ) -> ToolResult:
        from src.tools.templates import render_template
        payload = render_template(
            "rop_system",
            binary_path=binary_path,
            offset=offset,
            system_addr=system_addr,
        )
        return self._python(payload, tool_name="pwntools_rop_system")

    def bash_template(self, template_name: str, **kwargs) -> ToolResult:
        from src.tools.templates import render_bash_template
        command = render_bash_template(template_name, **kwargs)
        return self.run(command, tool_name="bash_template")

    def python_template(self, template_name: str, **kwargs) -> ToolResult:
        from src.tools.templates import render_python_template
        script = render_python_template(template_name, **kwargs)
        return self._python(script, tool_name="python_template")

    def python(self, script: str) -> ToolResult:
        return self._python(script, tool_name="python")

    def base_decode(self, value: str, encoding: str = "base64") -> ToolResult:
        encoding = (encoding or "base64").lower()
        script = (
            "import base64, binascii\n"
            f"value = {json.dumps(value)}\n"
            f"encoding = {json.dumps(encoding)}\n"
            "raw = value.strip().encode()\n"
            "try:\n"
            "    if encoding in ('base64', 'b64'):\n"
            "        data = base64.b64decode(raw)\n"
            "    elif encoding in ('base32', 'b32'):\n"
            "        data = base64.b32decode(raw)\n"
            "    elif encoding in ('hex', 'base16', 'b16'):\n"
            "        data = binascii.unhexlify(raw)\n"
            "    else:\n"
            "        raise ValueError('Unsupported encoding')\n"
            "    print(data.decode('utf-8', errors='replace'))\n"
            "except Exception as exc:\n"
            "    print(f'ERROR: {exc}')\n"
        )
        return self._python(script, tool_name="base_decode")

    # --- Web ---
    def curl(self, url: str, args: str = "") -> ToolResult:
        return self.run(f"curl {args} {self._shell_escape(url)}", tool_name="curl")

    def http_request(
        self,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        params: dict | None = None,
        data: str | None = None,
        timeout_s: int = 10,
    ) -> ToolResult:
        script = (
            "import json, requests\n"
            f"method = {json.dumps(method)}\n"
            f"url = {json.dumps(url)}\n"
            f"headers = {json.dumps(headers or {})}\n"
            f"params = {json.dumps(params or {})}\n"
            f"data = {json.dumps(data)}\n"
            f"timeout = {json.dumps(timeout_s)}\n"
            "resp = requests.request(method, url, headers=headers, params=params, data=data, timeout=timeout)\n"
            "print(resp.status_code)\n"
            "print(resp.text)\n"
        )
        return self._python(script, tool_name="http_request")

    def ffuf(
        self,
        url: str,
        wordlist: str,
        args: str = "",
        max_requests: int = 200,
    ) -> ToolResult:
        capped = max(1, min(int(max_requests), 200))
        tmp_list = "/tmp/ffuf_wordlist.txt"
        cmd = (
            f"head -n {capped} {self._shell_escape(wordlist)} > {self._shell_escape(tmp_list)} && "
            f"ffuf -u {self._shell_escape(url)} -w {self._shell_escape(tmp_list)} {args}"
        )
        return self.run(cmd, tool_name="ffuf")

    def _python(self, script: str, tool_name: str = "python") -> ToolResult:
        cmd = (
            "python3 - <<'PY'\n"
            f"{script}\n"
            "PY"
        )
        return self.run(cmd, tool_name=tool_name)

    def _shell_escape(self, value: str) -> str:
        return json.dumps(value)
