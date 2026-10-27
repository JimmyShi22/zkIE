import concurrent.futures
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
BASE = "https://hf-mirror.com"
OUT = Path("/data/zkIE/models/deepseek-v4.1-flash/raw")
API_FILE = Path("/tmp/v41.json")
FILE_WORKERS = int(os.environ.get("V41_FILE_WORKERS", "6"))
CHUNKS = int(os.environ.get("V41_CHUNKS", "4"))


def manifest_items():
    data = json.loads(API_FILE.read_text())
    items = [
        item
        for item in data["siblings"]
        if item["rfilename"].endswith(".safetensors")
    ]
    return [(item["rfilename"], int(item.get("size") or item["lfs"]["size"])) for item in items]


def curl_range(name, start, end, target):
    tmp = target.with_suffix(target.suffix + ".new")
    if tmp.exists():
        tmp.unlink()
    url = f"{BASE}/{REPO}/resolve/main/{name}"
    cmd = [
        "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "2",
        "--retry-all-errors", "--connect-timeout", "20", "-sS",
        "-r", f"{start}-{end}", "-o", str(tmp), url,
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"{name} range {start}-{end} failed rc={rc}")
    if target.exists():
        with target.open("ab") as out, tmp.open("rb") as inp:
            shutil.copyfileobj(inp, out, 8 << 20)
        tmp.unlink()
    else:
        tmp.replace(target)


def download_file(item):
    name, expected = item
    path = OUT / name
    base = path.stat().st_size if path.exists() else 0
    if base == expected:
        print(f"SKIP {name}", flush=True)
        return True
    if base > expected:
        raise RuntimeError(f"{name} larger than expected")
    remaining = expected - base
    chunk_size = math.ceil(remaining / CHUNKS)
    parts = path.with_name(path.name + ".parts")
    parts.mkdir(exist_ok=True)
    ranges = []
    for i in range(CHUNKS):
        start = base + i * chunk_size
        if start >= expected:
            break
        end = min(expected - 1, start + chunk_size - 1)
        ranges.append((i, start, end))

    def fetch_range(r):
        i, start, end = r
        target = parts / f"chunk-{i}"
        pending = target.with_suffix(target.suffix + ".new")
        if pending.exists():
            if target.exists():
                with target.open("ab") as out, pending.open("rb") as inp:
                    shutil.copyfileobj(inp, out, 8 << 20)
                pending.unlink()
            else:
                pending.replace(target)
        expected_len = end - start + 1
        if target.exists() and target.stat().st_size == expected_len:
            return True
        if target.exists() and target.stat().st_size > expected_len:
            target.unlink()
        current = target.stat().st_size if target.exists() else 0
        begin = start + current
        if begin > end:
            return target.stat().st_size == expected_len
        curl_range(name, begin, end, target)
        actual = target.stat().st_size
        if actual != expected_len:
            raise RuntimeError(f"{name} chunk {i}: {actual} != {expected_len}")
        return True

    print(f"START {name} base={base} chunks={len(ranges)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNKS) as pool:
        list(pool.map(fetch_range, ranges))

    temp = path.with_name(path.name + ".assembling")
    with temp.open("wb") as out:
        if base:
            with path.open("rb") as inp:
                shutil.copyfileobj(inp, out, 8 << 20)
        for i, _, _ in ranges:
            with (parts / f"chunk-{i}").open("rb") as inp:
                shutil.copyfileobj(inp, out, 8 << 20)
    if temp.stat().st_size != expected:
        raise RuntimeError(f"{name} assembled size mismatch")
    temp.replace(path)
    shutil.rmtree(parts)
    print(f"DONE {name} {expected}", flush=True)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = manifest_items()
    pending = []
    for name, size in items:
        p = OUT / name
        if not p.exists() or p.stat().st_size != size:
            pending.append((name, size))
    print(f"pending={len(pending)} file_workers={FILE_WORKERS} chunks={CHUNKS}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        futures = [pool.submit(download_file, item) for item in pending]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
