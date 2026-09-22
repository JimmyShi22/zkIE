#!/usr/bin/env python3
"""Generate the quantized SiLU lookup table for the prologue/epilogue FFNs.

`SiLU(x) = x * sigmoid(x)` is the one non-arithmetic activation in TimesFM's
input and horizon FFNs. It is a table lookup exactly like GELU/softmax: the
input `x` is quantized at scale 2^16 and offset to a non-negative index, and the
table holds `SiLU(x)` at scale 2^16. The table covers `x` in [-8, 8] (SiLU
saturates to `x`/`0` outside that), which comfortably contains the observed
activation range (~[-1.6, 1.6]).

Usage: .venv-timesfm/bin/python scripts/extract_silu_table.py
"""
import os

import numpy as np

SCALE = 65536.0
OFFSET = 1 << 19       # x in [-8, 8] at scale 2^16
TABLE_SIZE = 1 << 20


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def main() -> None:
    idx = np.arange(TABLE_SIZE, dtype=np.float64)
    x = (idx - OFFSET) / SCALE
    table = np.clip(np.round(silu(x) * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
    os.makedirs("models", exist_ok=True)
    path = "models/silu_table_i32.bin"
    table.tofile(path)
    print(f"wrote {TABLE_SIZE} SiLU entries (x in [-8, 8]) to {path}")


if __name__ == "__main__":
    main()
