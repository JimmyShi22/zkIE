#!/usr/bin/env bash
set -e
cd /data/jimmyshi/ie
docker run --rm -v "$PWD:/zkie" -w /zkie python:3.12 bash -c 'pip install -q onnx onnxruntime numpy && python scripts/extract_gpt2.py'