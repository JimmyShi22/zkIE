#!/usr/bin/env python3
"""Run the full 7-layer transformer stack in quantized fixed-point, end to end.

This is the "compiler" half: it replays the whole TimesFM transformer stack
(per layer: RMSNorm -> V matmul -> o_proj -> residual -> LayerNorm -> gate ->
ReLU -> down -> residual) with the exact fixed-point rules the Rust prover uses
(round half-to-even, the rsqrt table, padding to powers of two). The residual
stream is chained across layers, so each layer's output *is* the next layer's
input. The final residual is compared against the ONNX float reference.

Usage: .venv-timesfm/bin/python scripts/simulate_full_stack.py
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SCALE = 65536.0


def div_round(a: int, b: int) -> int:
    q, r = divmod(a, b)
    return q + 1 if r * 2 >= b else q


def pad1(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    out[: arr.size] = arr
    return out


def pad2(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.int64)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def rescale(raw: np.ndarray, shift: int) -> np.ndarray:
    return np.array([div_round(int(v), 1 << shift) for v in raw])


def rms_norm(x: np.ndarray, w: np.ndarray, rsqrt: np.ndarray, n_real: int) -> np.ndarray:
    sq = int((x[:n_real] ** 2).sum())
    s_index = div_round(div_round(sq, n_real), 1 << 18)
    rstd = int(rsqrt[s_index])
    raw = x * rstd * w
    return rescale(raw, 32)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, rsqrt: np.ndarray, n_real: int) -> np.ndarray:
    xr = x[:n_real]
    mean = div_round(int(xr.sum()), n_real)
    var = div_round(int(((xr - mean) ** 2).sum()), n_real)
    s_index = div_round(var, 1 << 18)
    rstd = int(rsqrt[s_index])
    raw = (x - mean) * rstd * w
    return rescale(raw, 32) + b


def main() -> None:
    base = "models"
    m = onnx.load(f"{base}/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
    rsqrt = np.fromfile(f"{base}/rsqrt_table_i32.bin", dtype=np.int32).astype(np.int64)

    weights = [
        ["val_94", "val_122", "val_126", "val_128"],
        ["val_134", "val_159", "val_163", "val_165"],
        ["val_171", "val_196", "val_200", "val_202"],
        ["val_208", "val_233", "val_237", "val_239"],
        ["val_245", "val_270", "val_274", "val_276"],
        ["val_282", "val_307", "val_311", "val_313"],
        ["val_319", "val_344", "val_348", "val_350"],
    ]
    N_REAL = 264

    # Reference: add_2 (layer-0 input) and add_30 (layer-6 residual output).
    for name in ("add_2", "add_30"):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    m2 = helper.make_model(g, opset_imports=m.opset_import)
    onnx.save(m2, "/tmp/tsfm_stack.onnx")
    sess = ort.InferenceSession("/tmp/tsfm_stack.onnx", providers=["CPUExecutionProvider"])
    inp = np.random.RandomState(0).randn(1, 32).astype(np.float32)
    pad_in = np.zeros((1, 32), dtype=np.float32)
    res = sess.run(["add_2", "add_30"], {
        "input_ts": inp, "input_padding": pad_in, "freq": np.array([[0]], dtype=np.int64),
    })
    add2_float = res[0].reshape(-1)
    final_float = res[1].reshape(-1)

    x = pad1(np.round(add2_float * SCALE), 512)
    for li, (qw_name, ow_name, gw_name, dw_name) in enumerate(weights):
        qkv = inits[qw_name]                    # [264, 792]
        v_w = pad2(np.round(qkv[:, 528:792] * SCALE), 512, 512)
        op_w = pad2(np.round(inits[ow_name] * SCALE), 512, 512)
        gate_w = pad2(np.round(inits[gw_name] * SCALE), 512, 1024)
        down_w = pad2(np.round(inits[dw_name] * SCALE), 1024, 512)

        qb = inits[f"stacked_transformer.layers.{li}.self_attn.qkv_proj.bias"]
        ob = inits[f"stacked_transformer.layers.{li}.self_attn.o_proj.bias"]
        gb = inits[f"stacked_transformer.layers.{li}.mlp.gate_proj.bias"]
        db = inits[f"stacked_transformer.layers.{li}.mlp.down_proj.bias"]
        v_b = pad1(np.round(qb[528:792] * SCALE), 512)
        op_b = pad1(np.round(ob * SCALE), 512)
        gate_b = np.round(gb * SCALE).astype(np.int64)   # 1024, already pow2
        down_b = pad1(np.round(db * SCALE), 512)

        lnw = pad1(np.round(inits[f"stacked_transformer.layers.{li}.input_layernorm.weight"] * SCALE), 512)
        mlp_w = pad1(np.round(inits[f"stacked_transformer.layers.{li}.mlp.layer_norm.weight"] * SCALE), 512)
        mlp_b = pad1(np.round(inits[f"stacked_transformer.layers.{li}.mlp.layer_norm.bias"] * SCALE), 512)

        # Attention half.
        mul9 = rms_norm(x, lnw, rsqrt, N_REAL)
        v_raw = (mul9.reshape(1, 512) @ v_w).reshape(-1)
        lin3v = rescale(v_raw, 16) + v_b
        op_raw = (lin3v.reshape(1, 512) @ op_w).reshape(-1)
        lin4 = rescale(op_raw, 16) + op_b
        add5 = x + lin4

        # FFN half.
        ln_out = layer_norm(add5, mlp_w, mlp_b, rsqrt, N_REAL)
        gate_raw = (ln_out.reshape(1, 512) @ gate_w).reshape(-1)   # 1024
        relu = np.maximum(rescale(gate_raw, 16) + gate_b, 0)
        down_raw = (relu.reshape(1, 1024) @ down_w).reshape(-1)    # 512
        ffn_out = rescale(down_raw, 16) + down_b
        x = add5 + ffn_out

    err = float(np.abs(x[: final_float.size] / SCALE - final_float).max())
    print(f"7-layer stack simulated; final residual max abs error vs ONNX float = {err:.3e}")
    assert err < 5e-3, f"quantized stack drifted too far from float: {err:.3e}"


if __name__ == "__main__":
    main()
