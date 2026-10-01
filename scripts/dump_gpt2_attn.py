import onnx
m = onnx.load("models/gpt2.onnx")
nodes = [n for n in m.graph.node if "attn" in "".join(list(n.input) + list(n.output))]
for n in nodes:
    if n.op_type in ("Div", "Sqrt", "Mul", "MatMul", "Softmax", "Sub", "Add", "Where", "Constant"):
        print(n.op_type, "in:", [i[:50] for i in n.input], "out:", [o[:50] for o in n.output])
