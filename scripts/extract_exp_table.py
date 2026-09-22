#!/usr/bin/env python3
"""Generate the quantized exp lookup table for the attention softmax.

The softmax is computed in the numerically-stable form `softmax(x) =
softmax(x - max)`, so the exp input lives in `(-inf, 0]` and the exp output is
in `(0, 1]`. The table maps `x` in [-32, 0] (scale 2^16) to `exp(x)` at scale
2^16; below -32 the value is clipped to 0.

Usage: .venv-timesfm/bin/python scripts/extract_exp_table.py
"""
import os

import numpy as np

SCALE = 65536.0
OFFSET = 1 << 21       # x in [-32, 0] at scale 2^16
TABLE_SIZE = 1 << 21


def main() -> None:
    idx = np.arange(TABLE_SIZE, dtype=np.float64)
    x = (idx - OFFSET) / SCALE
    table = np.clip(np.round(np.exp(x) * SCALE), 0, 2**31 - 1).astype(np.int32)
    os.makedirs("models", exist_ok=True)
    path = "models/exp_table_i32.bin"
    table.tofile(path)
    print(f"wrote {TABLE_SIZE} exp entries (x in [-32, 0]) to {path}")


if __name__ == "__main__":
    main()
