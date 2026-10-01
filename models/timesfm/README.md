# TimesFM 1.0 200M

Proving target: TimesFM 1.0-200M (`google/timesfm-1.0-200m-pytorch`), 20 layers,
16 heads x 80, hidden 1280, horizon head.

## Layout

- `src/lib.rs` — op-graph builder (`build_timesfm`).
- `examples/` — `prove` (end-to-end), `bench` (shard-granularity sweep).
- `weights/` — extracted fixed-point weights + lookup tables (gitignored).

## Getting the ONNX + weights (not committed)

The ONNX and the extracted `.bin` weights are large and are not committed.
Run from the repo root (needs torch + onnx + onnxscript + huggingface_hub):

```bash
# 1. export ONNX
python3 scripts/export_timesfm_200m.py

# 2. extract fixed-point weights + tables into models/timesfm/weights/
python3 scripts/extract_200m_full.py
python3 scripts/extract_200m.py
```

The shared `exp` / `rsqrt` lookup tables are the generic ones also produced by
the GPT-2 extraction (`models/gpt2/weights/exp_table_i32.bin`,
`models/gpt2/weights/rsqrt_table_i32.bin`).

Verify the exported ONNX before extracting:

| file | sha256 |
| --- | --- |
| `timesfm_1_0_200m.onnx` | `a9a577f627fa357a90283c1925d657d9e1c57d6f148d62fb3ff8b15457d3dd0f` |
| `timesfm_1_0_200m.onnx.data` | `ab2e2789373f46469484665d143d7afc97e41feb63558acad9213183e010cb85` |

Then prove / benchmark:

```bash
cargo run --release -p zkie-models-timesfm --example prove
cargo run --release -p zkie-models-timesfm --example bench
```
