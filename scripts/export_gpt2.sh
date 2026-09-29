#!/usr/bin/env bash
set -e
cd /data/jimmyshi/ie
docker run --rm -v "$PWD:/zkie" -w /zkie -e GPT2_SEQ="${GPT2_SEQ:-16}" -e HF_HUB_DISABLE_PROGRESS_BARS=1 -e TQDM_DISABLE=1 python:3.12 bash -c 'pip install -q torch && pip install -q onnx onnxscript transformers && python scripts/export_gpt2.py'