import os
import numpy as np
import torch

SEQ = int(os.environ.get("M", "16"))
H = 2048

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/work/models/deepseek-v2-lite/raw", local_files_only=True)
emb = model.model.embed_tokens.weight.float().detach().numpy()  # [vocab, H]

tokens = np.fromfile("/work/models/deepseek-v2-lite/weights/tokens_i32.bin", dtype=np.int32).astype(np.int64)
o = np.zeros((SEQ, H), dtype=np.int32)
x0 = np.clip(np.round(emb[tokens] * 65536.0), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
o[:SEQ, :H] = x0
o.tofile("/work/models/deepseek-v2-lite/weights/x0_i32.bin")
print("x0 saved", o.shape)

