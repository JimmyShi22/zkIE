import struct
import sys

def header(path):
    with open(path, "rb") as f:
        tokens = struct.unpack("<Q", f.read(8))[0]
        f.seek(tokens * 4, 1)
        assets = struct.unpack("<Q", f.read(8))[0]
        return tokens, assets

for path in sys.argv[1:]:
    tokens, assets = header(path)
    print(f"{path}: tokens={tokens} assets={assets}")
