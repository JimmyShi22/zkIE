#!/usr/bin/env python3
"""Generate the two activation lookup tables DeepSeek-V4.1-Flash MoE needs.

Both tables follow the exact fixed-point convention of the existing GELU table
(`extract_gpt2.py`): index = (x * 2^16) + OFFSET, table[idx] = round(f(x) * 2^16),
OFFSET = 2^23 (x in [-128, 128]) and SIZE = 2^24.

- sqrt_softplus(x) = sqrt(log1p(exp(x))) for the routing-gate scores.
- silu(x) = x * sigmoid(x) for the shared/routed expert gate activation.
"""
import os

import numpy as np

SCALE = 65536.0
OFFSET = 1 << 23
SIZE = 1 << 24


def sqrt_softplus(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.log1p(np.exp(x)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def main() -> None:
    out = os.environ.get("V41_TABLE_OUT", "models")
    os.makedirs(out, exist_ok=True)
    idx = np.arange(SIZE, dtype=np.float64)
    x = (idx - OFFSET) / SCALE

    sp = np.clip(np.round(sqrt_softplus(x) * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
    sp.tofile(os.path.join(out, "sqrt_softplus_table_i32.bin"))

    sl = np.clip(np.round(silu(x) * SCALE), -(2**31), 2**31 - 1).astype(np.int32)
    sl.tofile(os.path.join(out, "silu_table_v41_i32.bin"))

    print(f"wrote {out}/sqrt_softplus_table_i32.bin ({SIZE} entries)")
    print(f"wrote {out}/silu_table_v41_i32.bin ({SIZE} entries)")


if __name__ == "__main__":
    main()
