import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
BASE = "https://hf-mirror.com"
OUT = Path("/data/zkIE/models/deepseek-v4.1-flash/raw")
API_FILE = Path("/tmp/v41.json")
WORKERS = int(os.environ.get("V41_WORKERS", "12"))

WANTED = {
    "README.md",
    "LICENSE",
    ".gitattributes",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
}


def load_manifest():
    if API_FILE.exists():
        data = json.loads(API_FILE.read_text())
        if data.get("id") == REPO:
            return data
    import urllib.request

    with urllib.request.urlopen(f"{BASE}/api/models/{REPO}?blobs=true", timeout=30) as r:
        data = json.load(r)
    API_FILE.write_text(json.dumps(data))
    return data


def wanted(name):
    return name.endswith(".safetensors") or name in WANTED


def expected_size(item):
    return int(item.get("size") or item.get("lfs", {}).get("size") or 0)


def download(item):
    name = item["rfilename"]
    size = expected_size(item)
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and size and path.stat().st_size == size:
        print(f"SKIP {name} {size}", flush=True)
        return name, True
    if path.exists() and size and path.stat().st_size > size:
        print(f"DROP oversized {name} {path.stat().st_size} > {size}", flush=True)
        path.unlink()

    url = f"{BASE}/{REPO}/resolve/main/{name}"
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "-C",
        "-",
        "-sS",
        "-o",
        str(path),
        url,
    ]
    print(f"START {name} {size}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"FAIL {name} rc={proc.returncode}", flush=True)
        return name, False
    actual = path.stat().st_size if path.exists() else 0
    if size and actual != size:
        print(f"BAD_SIZE {name} actual={actual} expected={size}", flush=True)
        return name, False
    print(f"DONE {name} {actual}", flush=True)
    return name, True


def main():
    manifest = load_manifest()
    items = [
        item
        for item in manifest.get("siblings", [])
        if wanted(item.get("rfilename", ""))
    ]
    total = sum(expected_size(item) for item in items)
    print(f"repo={REPO} files={len(items)} bytes={total} workers={WORKERS}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    okay = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            _, result = future.result()
            okay = okay and result
    print("ALL_COMPLETE" if okay else "INCOMPLETE", flush=True)
    return 0 if okay else 1


if __name__ == "__main__":
    sys.exit(main())
