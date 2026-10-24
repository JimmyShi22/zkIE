import os
import numpy as np
import onnx
from onnx import numpy_helper

MODELS = "models"
OUT = "models/gpt2/weights"
SCALE = 65536.0
H_PAD = 1024
SEQ = 512

def q(arr):
    return np.clip(np.round(np.asarray(arr, dtype=np.float64) * SCALE), -(2**31), 2**31 - 1).astype(np.int32)

def pad2(arr, r, c):
    o = np.zeros((r, c), dtype=np.int32)
    a = np.asarray(arr)
    o[: a.shape[0], : a.shape[1]] = a
    return o

m = onnx.load(os.path.join(MODELS, "gpt2/gpt2.onnx"))
inits = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
wpe = inits["model.transformer.wpe.weight"]  # [1024, 768]

input_ids = np.load(os.path.join(OUT, "input_ids_512.npy")).astype(np.int64)
tokens = input_ids[:SEQ].astype(np.int32)
tokens.tofile(os.path.join(OUT, "tokens_i32.bin"))

pos_emb = wpe[:SEQ]
pad2(q(pos_emb), SEQ, H_PAD).tofile(os.path.join(OUT, "pos_emb_i32.bin"))
print("SEQ512 regenerated: tokens", tokens.shape, "pos_emb", pos_emb.shape)
