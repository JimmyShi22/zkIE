import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import onnx
from onnx import numpy_helper

OUT = "models/gpt2/weights"
SEQ = 512
SCALE = 65536.0
H_PAD = 1024

def q(arr):
    return np.clip(np.round(np.asarray(arr, dtype=np.float64) * SCALE), -(2**31), 2**31 - 1).astype(np.int32)

def pad2(arr, r, c):
    o = np.zeros((r, c), dtype=np.int32)
    a = np.asarray(arr)
    o[: a.shape[0], : a.shape[1]] = a
    return o

tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2/tokenizer", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = tokenizer.encode("The sky is")
ids = prompt[:]
with torch.no_grad():
    for _ in range(SEQ - len(prompt)):
        out = model(torch.tensor([ids])).logits[0, -1]
        ids.append(int(out.argmax()))

np.asarray(ids[:SEQ], dtype=np.int32).tofile(os.path.join(OUT, "tokens_i32.bin"))

m = onnx.load("models/gpt2/gpt2.onnx")
inits = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
wpe = inits["model.transformer.wpe.weight"]
pad2(q(wpe[:SEQ]), SEQ, H_PAD).tofile(os.path.join(OUT, "pos_emb_i32.bin"))

with torch.no_grad():
    logits = model(torch.tensor([ids[:SEQ]])).logits[0]
gt = logits.argmax(dim=1).numpy().astype(np.int32)
gt.tofile(os.path.join(OUT, "gt_argmax_512_i32.bin"))

print("generated", len(ids), "tokens; first 12:", ids[:12])
print("text head:", repr(tokenizer.decode(ids[:40])))
