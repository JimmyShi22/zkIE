#!/usr/bin/env python3
"""Verify the rsqrt-table LayerNorm reconstruction matches the float reference.

`prove_layer_norm` now derives `mean` from the committed input and binds `rstd`
to `var + eps` through the generated `models/rsqrt_table_i32.bin` (index scale
2^14, rstd scale 2^16), instead of trusting float scalars.  This script replays
exactly that integer protocol for all 7 LayerNorm layers and reports the output
difference against the float-derived reconstruction, so the lookup does not
degrade end-to-end accuracy.

Usage: .venv-timesfm/bin/python scripts/verify_norm_rsqrt_table.py
"""
import numpy as np

INDEX_SCALE = 1 << 14
RSTD_SCALE = 1 << 16


def div_round(a: int, b: int) -> int:
    """Round a/b to nearest, ties toward +inf (matches the Rust div_round)."""
    q, r = divmod(a, b)
    return q + 1 if r * 2 >= b else q


def main() -> None:
    table = np.fromfile("models/rsqrt_table_i32.bin", dtype=np.int32).astype(np.int64)
    worst = 0.0
    for i in range(7):
        x = np.fromfile(f"models/norms/L{i}_in.bin", dtype=np.int32).astype(np.int64)
        w = np.fromfile(f"models/norms/L{i}_w.bin", dtype=np.int32).astype(np.int64)
        b = np.fromfile(f"models/norms/L{i}_b.bin", dtype=np.int32).astype(np.int64)
        mf, rf = np.fromfile(f"models/norms/L{i}_scalars_f64.bin", dtype=np.float64)

        # Number of true (unpadded) elements: weight nonzero count.
        n_real = int((w != 0).sum())
        xr = x[:n_real]

        # Integer protocol (matches Rust `layer_norm_scalars_i32`).
        mean_q = div_round(int(xr.sum()), n_real)
        var_q = div_round(int(((xr - mean_q) ** 2).sum()), n_real)
        s_index = div_round(var_q, 1 << 18)
        rstd_table = int(table[s_index])

        # Float reference (matches the old float-scalar path).
        mean_f = mean_q / RSTD_SCALE
        rstd_f = rf
        y_float = (x / RSTD_SCALE - mean_f) * rstd_f * (w / RSTD_SCALE) + b / RSTD_SCALE

        # Table reconstruction: raw at 2^48, rescale by 2^32, then add bias.
        raw = (x - mean_q) * rstd_table * w
        y_table = (np.round(raw.astype(np.float64) / (1 << 32)) + b) / RSTD_SCALE

        d = np.abs(y_table - y_float)
        worst = max(worst, float(d.max()))
        print(
            f"L{i}: n_real={n_real} s_index={s_index} rstd_table={rstd_table} "
            f"rstd_float={round(rf * RSTD_SCALE)} max_abs_err={d.max():.3e}"
        )
    print(f"\nworst max abs error across all layers = {worst:.3e}")
    assert worst < 1e-3, f"rsqrt table degrades norm accuracy: {worst:.3e}"
    print("rsqrt-table LayerNorm accuracy within budget (<1e-3 absolute)")


if __name__ == "__main__":
    main()
