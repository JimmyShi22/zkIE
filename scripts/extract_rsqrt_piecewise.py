#!/usr/bin/env python3
"""Generate the piecewise LayerNorm rsqrt lookup table (shared with GPT-2).

LayerNorm needs `rstd = 1/sqrt(var + eps)`. The index is piecewise:
- indices below 2^20 are full resolution (step 1, index scale 2^14);
- raw in [2^20, 2^30) are quantized with step 2^8;
- raw >= 2^30 are quantized with step 2^16 (rsqrt is nearly flat there).
This covers GPT-2's ~13294 variance and Gemma 3's much larger ~1e6 variance.

Usage: python scripts/extract_rsqrt_piecewise.py
"""
import numpy as np

SCALE = 65536.0
INDEX_SCALE = 1 << 14
EPS = 1e-6
FINE = 1 << 20          # step-1 region
COARSE_STEP = 1 << 8
COARSE = 1 << 22        # coarse-1 entries (step 2^8)
COARSE_STEP2 = 1 << 16
COARSE2 = 1 << 21       # coarse-2 entries (step 2^16)


def main() -> None:
    fine_idx = np.arange(FINE, dtype=np.float64)
    fine_s = fine_idx / INDEX_SCALE + EPS
    fine_rstd = 1.0 / np.sqrt(fine_s)

    coarse_k = np.arange(COARSE, dtype=np.float64)
    coarse_raw = FINE + coarse_k * COARSE_STEP
    coarse_s = coarse_raw / INDEX_SCALE + EPS
    coarse_rstd = 1.0 / np.sqrt(coarse_s)

    coarse2_start = FINE + COARSE * COARSE_STEP
    coarse2_k = np.arange(COARSE2, dtype=np.float64)
    coarse2_raw = coarse2_start + coarse2_k * COARSE_STEP2
    coarse2_s = coarse2_raw / INDEX_SCALE + EPS
    coarse2_rstd = 1.0 / np.sqrt(coarse2_s)

    table = np.concatenate([fine_rstd, coarse_rstd, coarse2_rstd])
    table = np.clip(np.round(table * SCALE), 0, 2**31 - 1).astype(np.int32)
    table.tofile("models/gpt2/weights/rsqrt_table_i32.bin")
    max_var = (coarse2_start + (COARSE2 - 1) * COARSE_STEP2) / INDEX_SCALE
    print(f"wrote {len(table)} piecewise rsqrt entries; max var = {max_var:.1f}")


if __name__ == "__main__":
    main()
