#!/usr/bin/env python3
"""Map the TimesFM ONNX graph onto GKR primitives.

This is the first half of the ONNX->GKR compiler: it walks the graph in
topological order and classifies every node as a GKR primitive (matmul, norm,
softmax, relu, split, arithmetic, or metadata), recording the weight tensors a
matmul reads. The second half (executing the plan as actual GKR proofs) lives in
the Rust crates; this script produces the plan it consumes.

Usage:
    .venv-timesfm/bin/python scripts/onnx_gkr_plan.py
"""

import json

import onnx


def classify(op_type: str) -> str:
    if op_type == "MatMul":
        return "MATMUL"
    if op_type == "LayerNormalization":
        return "NORM"
    if op_type == "Softmax":
        return "SOFTMAX"
    if op_type == "Relu":
        return "RELU"
    if op_type == "Split":
        return "SPLIT"
    if op_type in ("Add", "Mul", "Sub", "Div"):
        return "ARITH"
    if op_type in ("Pow", "Sqrt", "ReduceMean", "Reciprocal"):
        return "NORM_DECOMP"
    if op_type in ("Reshape", "Transpose", "Squeeze", "Unsqueeze", "Concat", "Cast", "Clip"):
        return "META"
    if op_type in ("Sigmoid", "Gelu"):
        return "NONLIN"
    return "OTHER:" + op_type


def main() -> None:
    m = onnx.load("models/timesfm_8m_fintext_ctx32.onnx")
    g = m.graph
    inits = {i.name: [d for d in i.dims] for i in g.initializer}

    plan = []
    for n in g.node:
        kind = classify(n.op_type)
        weights = [
            inp for inp in n.input if inp in inits and len(inits[inp]) == 2
        ]
        plan.append(
            {
                "op": n.op_type,
                "kind": kind,
                "outputs": list(n.output),
                "weights": weights,
                "weight_shapes": {w: inits[w] for w in weights},
            }
        )

    core = [p for p in plan if not p["kind"].startswith(("OTHER", "META", "NORM_DECOMP", "NONLIN"))]
    summary = {}
    for p in plan:
        summary[p["kind"]] = summary.get(p["kind"], 0) + 1

    print("=== GKR op plan summary ===")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")
    print()
    print("=== core layer ops (first 40) ===")
    for p in core[:40]:
        w = p["weights"][0] if p["weights"] else ""
        shape = p["weight_shapes"].get(w, "")
        print(f"{p['kind']:<8} {p['op']:<8} {w:<28} {shape}")

    with open("models/timesfm_8m_gkr_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    print()
    print("wrote models/timesfm_8m_gkr_plan.json")


if __name__ == "__main__":
    main()
