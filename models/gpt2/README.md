# GPT-2 124M

Proving target: GPT-2 124M (`openai-community/gpt2`), 12 layers, 12 heads,
pre-norm, causal LM head.

## Layout

- `src/lib.rs` — op-graph builder (`build_gpt2`).
- `examples/` — `prove` (end-to-end), `bench_sharded`, `bench_autotune`,
  `bench_16`, and legacy per-layer micro-benchmarks.
- `weights/` — extracted fixed-point weights + lookup tables (gitignored).

## Getting the ONNX + weights (not committed)

The ONNX and the extracted `.bin` weights are large and are not committed.
Run from the repo root:

```bash
# 1. export ONNX (needs torch + transformers)
python3 scripts/export_gpt2.py

# 2. extract fixed-point weights + tables into models/gpt2/weights/
bash scripts/run_extract_gpt2.sh
```

Verify the exported ONNX before extracting:

| file | sha256 |
| --- | --- |
| `gpt2.onnx` | `d120efd4d920721891f2afdf3f00231fef72f68f14e324a69f680dbefc183543` |
| `gpt2.onnx.data` | `8f78c45e3fe4438a956f5d85357ffc3c8164e517b7df128fa0648fd5e8b5ac27` |

Then prove / benchmark:

```bash
cargo run --release -p zkie-models-gpt2 --example prove
cargo run --release -p zkie-models-gpt2 --example bench_sharded
```
