#!/usr/bin/env python3
"""Dump GPT-2 ONNX graph structure: I/O, initializers, node order."""
import os
import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = onnx.load(os.path.join(ROOT, "models", "gpt2/gpt2.onnx"), load_external_data=False)
g = m.graph
inits = {i.name: [d for d in i.dims] for i in g.initializer}

print("=== graph inputs ===")
for i in g.input:
    dims = []
    for d in i.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else "?")
    print("  in  %s %s" % (i.name, dims))

print("=== graph outputs ===")
for o in g.output:
    dims = []
    for d in o.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else "?")
    print("  out %s %s" % (o.name, dims))

print("=== initializers (%d) ===" % len(inits))
for name in sorted(inits):
    print("  %s %s" % (name, inits[name]))

print("=== nodes (%d) ===" % len(g.node))
for idx, n in enumerate(g.node):
    w = [i for i in n.input if i in inits]
    wtag = ",".join(w) if w else "-"
    print("  [%d] %-18s w=%s" % (idx, n.op_type, wtag))