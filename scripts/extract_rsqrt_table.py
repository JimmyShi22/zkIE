#!/usr/bin/env python3
"""Generate the quantized rsqrt lookup table for the LayerNorm proof.

LayerNorm needs `rstd = 1/sqrt(var + eps)`; this is the single non-arithmetic
scalar in the norm (everything else — mean, variance, the affine product — is
plain field arithmetic the verifier recomputes from the committed input).  The
table maps a fixed-point `s = var + eps` to the corresponding `rstd`, so the
prover can bind `rstd` to the committed `x` through a LogUp lookup instead of
supplying it as a trusted scalar.

Quantization:
  * index scale 2^14, table size 2^19 -> covers s up to 32.0 (TimesFM's var
    spans ~1.32 .. ~27.9, so this is comfortably inside range).
  * rstd stored at scale 2^16 (matches the weight/activation scale).
  * `eps` (1e-6) is absorbed into the table entries; at 2^14 it is < 1/2 index
    unit, so it does not shift the index.

Usage: .venv-timesfm/bin/python scripts/extract_rsqrt_table.py
"""
import os

import numpy as np

INDEX_SCALE = 1 << 14
TABLE_SIZE = 1 << 19
RSTD_SCALE = 1 << 16
EPS = 1e-6


def main() -> None:
    s = np.arange(TABLE_SIZE, dtype=np.float64) / INDEX_SCALE + EPS
    rstd = 1.0 / np.sqrt(s)
    table = np.clip(np.round(rstd * RSTD_SCALE), 0, 2**31 - 1).astype(np.int32)
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/rsqrt_table_i32.bin"
    table.tofile(path)
    print(f"wrote {TABLE_SIZE} entries (max s = {TABLE_SIZE / INDEX_SCALE:.1f}) to {path}")


if __name__ == "__main__":
    main()
