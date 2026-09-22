#!/usr/bin/env python3
"""Export TimesFM 1.0-200M to models/timesfm_1_0_200m.onnx (+ .onnx.data).

Reproduces the procedure documented in
docs/superpowers/specs/2026-07-26-zkie-timesfm-onnx-export-attempt.md:
a vendored PyTorch-only subset of google-research/timesfm's v1 source loads
the checkpoint from `google/timesfm-1.0-200m-pytorch` (the repo id from the
task notes only hosts a JAX/Orbax checkpoint), then torch's dynamo exporter
(dynamo=True, opset 18) exports one static forward pass with
context_len=512 / horizon_len=128.

Run inside a Python >= 3.10 environment with torch, onnx, onnxscript,
pandas, utilsforecast and huggingface_hub installed, e.g.:

  docker run --rm -v $PWD:/zkie -w /zkie python:3.12 bash -c '
      pip install -q torch --index-url https://download.pytorch.org/whl/cpu
      pip install -q onnx onnxscript onnxruntime pandas utilsforecast huggingface_hub
      python scripts/export_timesfm_200m.py'

The vendored sources land in .spike-test/timesfm_v1/ (gitignored).
"""

import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIKE = os.path.join(ROOT, ".spike-test", "timesfm_v1")
MODELS = os.path.join(ROOT, "models")

FILES = ["pytorch_patched_decoder.py", "timesfm_base.py", "timesfm_torch.py"]


def fetch_sources():
    os.makedirs(SPIKE, exist_ok=True)
    with urllib.request.urlopen(
        "https://api.github.com/repos/google-research/timesfm/commits/master",
        timeout=30,
    ) as resp:
        head = json.loads(resp.read().decode())
    sha = head["sha"]
    print(f"vendoring google-research/timesfm v1 sources at {sha[:12]}")
    base = f"https://raw.githubusercontent.com/google-research/timesfm/{sha}/v1/src/timesfm/"
    for name in FILES:
        dest = os.path.join(SPIKE, name)
        if os.path.exists(dest):
            print(f"  {name}: already vendored")
            continue
        urllib.request.urlretrieve(base + name, dest)
        print(f"  {name}: fetched")
    # The vendored files use intra-package relative imports, so they must be
    # imported as a package. The rewrite below also covers the July-era
    # sources, which imported `from timesfm import timesfm_base` (and would
    # collide with an installed `timesfm` distribution).
    with open(os.path.join(SPIKE, "__init__.py"), "w") as f:
        f.write("")
    torch_path = os.path.join(SPIKE, "timesfm_torch.py")
    with open(torch_path) as f:
        src = f.read()
    src = src.replace("from timesfm import timesfm_base", "from . import timesfm_base")
    with open(torch_path, "w") as f:
        f.write(src)


def main():
    fetch_sources()
    sys.path.insert(0, os.path.dirname(SPIKE))  # .spike-test (package parent)

    import torch  # noqa: E402

    from timesfm_v1.timesfm_base import TimesFmCheckpoint, TimesFmHparams  # noqa: E402
    from timesfm_v1.timesfm_torch import TimesFmTorch  # noqa: E402

    hparams = TimesFmHparams(
        context_len=512,
        horizon_len=128,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        num_heads=16,
        model_dims=1280,
        per_core_batch_size=1,
        backend="cpu",
    )
    checkpoint = TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
    print("loading checkpoint (this downloads ~814MB on first run)...")
    tfm = TimesFmTorch(hparams=hparams, checkpoint=checkpoint)
    tfm.load_from_checkpoint(checkpoint)
    model = tfm._model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {type(model).__name__} with {n_params:,} parameters")

    os.makedirs(MODELS, exist_ok=True)
    input_ts = torch.randn(1, 512)
    input_padding = torch.zeros(1, 512)
    freq = torch.zeros(1, 1, dtype=torch.int64)
    out_path = os.path.join(MODELS, "timesfm_1_0_200m.onnx")
    print("exporting (torch dynamo, opset 18)...")
    torch.onnx.export(
        model,
        (input_ts, input_padding, freq),
        out_path,
        input_names=["input_ts", "input_padding", "freq"],
        output_names=["output_ts"],
        opset_version=18,
        dynamo=True,
    )
    size = os.path.getsize(out_path)
    data_size = os.path.getsize(out_path + ".data")
    print(f"done: {out_path} ({size} bytes) + {out_path}.data ({data_size} bytes)")


if __name__ == "__main__":
    main()
