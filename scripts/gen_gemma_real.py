import os
import numpy as np
import torch

SEQ = int(os.environ.get("GEMMA_SEQ", "16"))

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("/work/models/gemma3/raw", local_files_only=True)
model.eval()

prompt = os.environ.get("GEMMA_PROMPT", "The sky is")
prompt_ids = tok.encode(prompt, add_special_tokens=False)
assert len(prompt_ids) < SEQ, (len(prompt_ids), SEQ)

gen_ids = prompt_ids[:]
with torch.no_grad():
    for _ in range(SEQ - len(prompt_ids)):
        cur = torch.tensor([gen_ids], dtype=torch.long)
        lg = model(cur).logits
        gen_ids.append(int(lg[0, -1].argmax().item()))

input_ids = np.array([gen_ids], dtype=np.int64)
input_ids[0].astype(np.int32).tofile("/work/models/gemma3/weights/tokens_i32.bin")

with torch.no_grad():
    lg = model(torch.tensor([gen_ids], dtype=torch.long)).logits
gt = lg[0].argmax(-1).numpy().astype(np.int32)
gt.tofile("/work/models/gemma3/weights/gt_argmax_i32.bin")

print("prompt_ids:", prompt_ids)
print("input_ids:", gen_ids)
print("gt_argmax:", gt.tolist())

