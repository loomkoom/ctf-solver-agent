import base64
import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool

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
    def __init__(self, runner: SandboxRunner | None = None, workdir: str | None = None):
        if runner is None:
            runner = SandboxRunner(workdir=workdir)
        elif workdir:
            runner.workdir = workdir
        self.runner = runner

    def set_workdir(self, workdir: str) -> None:
        if workdir:
            self.runner.workdir = workdir

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

    def foremost(self, path: str, out_dir: str | None = None, args: str = "") -> ToolResult:
        dest = out_dir or f"{path}_foremost"
        extra = (args or "").strip()
        cmd = f"foremost -i {self._shell_escape(path)} -o {self._shell_escape(dest)}"
        if extra:
            cmd = f"{cmd} {extra}"
        return self.run(cmd, tool_name="foremost")

    def tshark(self, path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"tshark -r {self._shell_escape(path)}"
        if extra:
            cmd = f"{cmd} {extra}"
        return self.run(cmd, tool_name="tshark")

    def yara(self, rule_path: str, target_path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"yara {extra} {self._shell_escape(rule_path)} {self._shell_escape(target_path)}".strip()
        return self.run(cmd, tool_name="yara")

    def pdfinfo(self, path: str) -> ToolResult:
        return self.run(f"pdfinfo {self._shell_escape(path)}", tool_name="pdfinfo")

    def pdftotext(self, path: str, out_path: str | None = None, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        if out_path:
            cmd = f"pdftotext {extra} {self._shell_escape(path)} {self._shell_escape(out_path)}".strip()
        else:
            cmd = f"pdftotext {extra} {self._shell_escape(path)} -".strip()
        return self.run(cmd, tool_name="pdftotext")

    def qpdf(self, path: str, out_path: str | None = None, args: str = "--check") -> ToolResult:
        extra = (args or "").strip()
        if out_path:
            cmd = f"qpdf {extra} {self._shell_escape(path)} {self._shell_escape(out_path)}".strip()
        else:
            cmd = f"qpdf {extra} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="qpdf")

    def pdf_parser(self, path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"pdf-parser {extra} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="pdf_parser")

    def steghide(self, path: str, args: str = "info") -> ToolResult:
        clean_args = (args or "info").strip()
        if clean_args.startswith("extract"):
            extra = clean_args[len("extract"):].strip()
            cmd = f"steghide extract -sf {self._shell_escape(path)}"
            if extra:
                cmd = f"{cmd} {extra}"
        elif clean_args.startswith("info"):
            cmd = f"steghide info {self._shell_escape(path)}"
        else:
            cmd = f"steghide {clean_args} {self._shell_escape(path)}"
        return self.run(cmd, tool_name="steghide")

    def stegsnow(self, path: str, args: str = "") -> ToolResult:
        cmd = f"stegsnow {args} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="stegsnow")

    def pngcheck(self, path: str, args: str = "") -> ToolResult:
        cmd = f"pngcheck {args} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="pngcheck")

    def zbarimg(self, path: str, args: str = "") -> ToolResult:
        cmd = f"zbarimg {args} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="zbarimg")

    def qrencode(self, value: str, out_path: str | None = None, args: str = "") -> ToolResult:
        dest = out_path or "qrcode.png"
        extra = (args or "").strip()
        cmd = f"qrencode {extra} -o {self._shell_escape(dest)} {self._shell_escape(value)}".strip()
        return self.run(cmd, tool_name="qrencode")

    def stegseek(self, path: str, wordlist: str | None = None) -> ToolResult:
        if wordlist:
            return self.run(
                f"stegseek {self._shell_escape(path)} {self._shell_escape(wordlist)}",
                tool_name="stegseek",
            )
        return self.run(f"stegseek {self._shell_escape(path)}", tool_name="stegseek")

    def zsteg(self, path: str) -> ToolResult:
        return self.run(f"zsteg {self._shell_escape(path)}", tool_name="zsteg")

    def apktool(self, path: str, out_dir: str | None = None, args: str = "d") -> ToolResult:
        extra = (args or "d").strip()
        if out_dir:
            cmd = f"apktool {extra} -o {self._shell_escape(out_dir)} {self._shell_escape(path)}"
        else:
            cmd = f"apktool {extra} {self._shell_escape(path)}"
        return self.run(cmd, tool_name="apktool")

    def jadx(self, path: str, out_dir: str | None = None, args: str = "") -> ToolResult:
        dest = out_dir or f"{path}_jadx"
        extra = (args or "").strip()
        cmd = f"jadx {extra} -d {self._shell_escape(dest)} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="jadx")

    def aapt(self, path: str, args: str = "dump badging") -> ToolResult:
        extra = (args or "dump badging").strip()
        cmd = f"aapt {extra} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="aapt")

    def dex2jar(self, path: str, out_path: str | None = None, args: str = "") -> ToolResult:
        dest = out_path or str(Path(path).with_suffix(".jar"))
        extra = (args or "").strip()
        cmd = f"d2j-dex2jar {extra} -o {self._shell_escape(dest)} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="dex2jar")

    def ciphey(self, text_or_path: str, args: str = "") -> ToolResult:
        base_args = args or "-f"
        cmd = f"ciphey {base_args} {self._shell_escape(text_or_path)}".strip()
        return self.run(cmd, tool_name="ciphey")

    def lemmeknow(self, text_or_path: str, json_output: bool = True, args: str = "") -> ToolResult:
        flags = "--json" if json_output else ""
        cmd = f"lemmeknow {flags} {args} {self._shell_escape(text_or_path)}".strip()
        return self.run(cmd, tool_name="lemmeknow")

    def hashcat(self, hash_file: str, wordlist: str | None = None, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"hashcat {extra} {self._shell_escape(hash_file)}".strip()
        if wordlist:
            cmd = f"{cmd} {self._shell_escape(wordlist)}"
        return self.run(cmd, tool_name="hashcat")

    def john(self, hash_file: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"john {extra} {self._shell_escape(hash_file)}".strip()
        return self.run(cmd, tool_name="john")

    def hashid(self, value_or_path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"hashid {extra} {self._shell_escape(value_or_path)}".strip()
        return self.run(cmd, tool_name="hashid")

    def name_that_hash(self, value_or_path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"nth {extra} {self._shell_escape(value_or_path)}".strip()
        return self.run(cmd, tool_name="name_that_hash")

    def hashdeep(self, path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"hashdeep {extra} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="hashdeep")

    def ghidra_headless(self, project_dir: str, project_name: str, binary_path: str) -> ToolResult:
        cmd = (
            f"analyzeHeadless {self._shell_escape(project_dir)} "
            f"{self._shell_escape(project_name)} -import {self._shell_escape(binary_path)}"
        )
        return self.run(cmd, tool_name="ghidra_headless")

    def ropgadget(self, binary_path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"ROPgadget --binary {self._shell_escape(binary_path)} {extra}".strip()
        return self.run(cmd, tool_name="ropgadget")

    def pwninit(self, binary_path: str | None = None, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = "pwninit"
        if binary_path:
            cmd = f"{cmd} --bin {self._shell_escape(binary_path)}"
        if extra:
            cmd = f"{cmd} {extra}"
        return self.run(cmd, tool_name="pwninit")

    def one_gadget(self, path: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"one_gadget {extra} {self._shell_escape(path)}".strip()
        return self.run(cmd, tool_name="one_gadget")

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

    def slither(self, target: str, args: str = "") -> ToolResult:
        extra = (args or "").strip()
        cmd = f"slither {extra} {self._shell_escape(target)}".strip()
        return self.run(cmd, tool_name="slither")

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


_DEFAULT_TOOLBOX: Toolbox | None = None


def set_default_toolbox(toolbox: Toolbox) -> None:
    global _DEFAULT_TOOLBOX
    _DEFAULT_TOOLBOX = toolbox


@tool("bash")
def bash_tool(command: str) -> dict:
    """Execute any shell command in the Kali CTF sandbox.
    All CTF tools are available: python3, pwntools, ghidra (analyzeHeadless),
    r2, gdb+pwndbg, checksec, binwalk, steghide, stegseek, zsteg, exiftool,
    tshark, foremost, hashcat, john, RsaCtfTool, ciphey, ffuf, strings,
    objdump, file, xxd, 7z, curl, nc, and all standard Kali tools.
    Chain commands with pipes. Files are in /challenge/."""
    toolbox = _DEFAULT_TOOLBOX or Toolbox()
    result = toolbox.run(command, tool_name="bash")
    stdout = (result.stdout or "")
    stderr = (result.stderr or "")
    if len(stdout) > 6000:
        stdout = stdout[-6000:]
    if len(stderr) > 1000:
        stderr = stderr[-1000:]
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.exit_code,
    }


def make_bash_tool(toolbox: Toolbox):
    set_default_toolbox(toolbox)
    return bash_tool


@tool("submit_flag")
def submit_flag_tool(flag: str) -> str:
    """Submit a CTF flag when found."""
    return flag


def make_submit_flag_tool(connector=None, challenge_id=None):
    return submit_flag_tool
