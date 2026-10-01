import onnx
from onnx import numpy_helper
m = onnx.load("models/gpt2/gpt2/gpt2.onnx")
inits = {i.name: i for i in m.graph.initializer}
interesting = ("Div", "Sqrt", "Mul", "MatMul", "Softmax", "Sub", "Add", "Where", "Constant", "ReduceMax", "ReduceSum", "Erf", "Tanh", "Pow", "Slice", "Concat", "Reshape", "Transpose", "Gather", "Unsqueeze")
for i, n in enumerate(m.graph.node):
    if n.op_type in interesting:
        consts = []
        for inp in n.input:
            if inp in inits:
                try:
                    a = numpy_helper.to_array(inits[inp])
                    if a.size <= 8:
                        consts.append(f"{inp}={a.reshape(-1)[:8].tolist()}")
                    else:
                        consts.append(f"{inp}=shape{a.shape}")
                except Exception:
                    consts.append(f"{inp}=?")
        print(i, n.op_type, "out:", list(n.output)[:1], "in:", list(n.input)[:4], "const:", consts)
