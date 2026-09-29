#!/usr/bin/env python3
"""Generate the piecewise LayerNorm rsqrt lookup table (shared with GPT-2).

LayerNorm needs `rstd = 1/sqrt(var + eps)`. To cover GPT-2's outlier-dimension
variances (per-position variance up to ~13294) without a 2^28-entry table, the
index is piecewise: indices below 2^20 are stored at full resolution (step 1,
index scale 2^14), and larger indices are quantized with step 2^8 (the rsqrt
curve is nearly flat there, so this loses no meaningful precision). The first
2^20 entries are identical to the original uniform table, so this is
backward-compatible with TimesFM (whose variance stays below 32).

Usage: python scripts/extract_rsqrt_piecewise.py
"""
import numpy as np

SCALE = 65536.0
INDEX_SCALE = 1 << 14
EPS = 1e-6
FINE = 1 << 20          # step-1 region
COARSE_STEP = 1 << 8
COARSE = 1 << 20        # number of coarse entries; total = 2^21

def main() -> None:
    fine_idx = np.arange(FINE, dtype=np.float64)
    fine_s = fine_idx / INDEX_SCALE + EPS
    fine_rstd = 1.0 / np.sqrt(fine_s)

    coarse_k = np.arange(COARSE, dtype=np.float64)
    coarse_raw = FINE + coarse_k * COARSE_STEP
    coarse_s = coarse_raw / INDEX_SCALE + EPS
    coarse_rstd = 1.0 / np.sqrt(coarse_s)

    table = np.concatenate([fine_rstd, coarse_rstd])
    table = np.clip(np.round(table * SCALE), 0, 2**31 - 1).astype(np.int32)
    table.tofile("models/rsqrt_table_i32.bin")
    max_var = (FINE + (COARSE - 1) * COARSE_STEP) / INDEX_SCALE
    print(f"wrote {len(table)} piecewise rsqrt entries; max var = {max_var:.1f}")

if __name__ == "__main__":
    main()