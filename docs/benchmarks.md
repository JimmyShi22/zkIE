# Benchmark method

All performance figures for the current implementation are **TBD**. The
numbers measured for earlier designs are not comparable with the current code
and are intentionally not carried over: the proof format, the commitment
layout and the executors have changed.

## What to measure

Measure the three stages separately, because they have different cost and
security profiles:

| Stage | What is measured |
| --- | --- |
| commit | wall clock, peak resident memory, public commitment size |
| prove | wall clock, peak resident memory, `proof.bin` size |
| verify | wall clock, peak resident memory |

Correctness gates for every row:

- `verify` accepts the produced proof;
- the final argmax or output tensor equals the case's expected value;
- a tampered proof or tampered public input is rejected.

## Cases

| Model | Input | commit | prove | verify | proof size | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 124M | 16 tokens | TBD | TBD | TBD | TBD | TBD |
| GPT-2 124M | 512 tokens | TBD | TBD | TBD | TBD | TBD |
| Gemma 3 270M | 16 tokens | TBD | TBD | TBD | TBD | TBD |
| Gemma 3 270M | 512 tokens | TBD | TBD | TBD | TBD | TBD |
| TimesFM 1.0 200M | 16 patches | TBD | TBD | TBD | TBD | TBD |

## Method

1. Build in release mode on the target machine.
2. Pin the thread count and record it with every result.
3. Run commit once per case. Reuse the commit artifact for the prove runs.
4. Run one cold prove and verify, then at least three warm runs. Report the
   median warm wall clock and the maximum peak resident memory.
5. Record the exact byte size of `proof.bin`; do not estimate it from field
   counts or from an older run.
6. Keep the run directory and the produced `result.json` files so the numbers
   can be audited.

The separate stages can be driven directly:

```bash
cargo run --release -p zkie -- commit \
  --case <case.toml> --assets-manifest <manifest> --parameters <parameters> \
  --out <commit-dir> --run-dir <run-dir>

cargo run --release -p zkie -- prove \
  --commit <commit-dir> --input <input> --model-id <model-id> \
  --out <proof-dir> --run-dir <run-dir>

cargo run --release -p zkie -- verify \
  --public <commit-dir>/public --statement <proof-dir>/statement.bin \
  --proof <proof-dir>/proof.bin --public-inputs <proof-dir>/public-inputs.bin \
  --model-id <model-id> --policy <policy> --run-dir <run-dir>
```

Batch runners:

```bash
# cold + warm stages over a case list
cargo run --release -p zkie -- matrix \
  --cases <case.toml>... --cold-runs 1 --warm-runs 3 --out <dir>

# per-thread-count samples
cargo run --release -p zkie -- profile-matrix \
  --cases <case.toml>... --threads 1 8 64 --runs 3 --out <dir>
```

The run report records wall clock and peak resident memory per stage. Proof
size is the byte length of `proof.bin`.

## Reporting rules

- Do not mix numbers from different builds, thread counts or asset versions.
- Do not report a number that was not produced by the current code on the
  stated case.
- Mark an unavailable measurement as **TBD**, not as zero or as an estimate.
- Report commit, prove and verify separately; an end-to-end total hides which
  stage dominates.
