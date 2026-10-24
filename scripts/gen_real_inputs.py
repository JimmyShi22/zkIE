#!/usr/bin/env python3
"""Generate real prompt+response inputs and GT for GPT-2 and Gemma 3."""

import json
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
PROMPT = os.environ.get("REAL_PROMPT", "The sky is")
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "64"))


def generate_sequence(model, tokenizer, prompt_ids, seq_len):
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=seq_len - len(prompt_ids),
            min_new_tokens=seq_len - len(prompt_ids),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
        )[0]
        input_ids = torch.tensor([generated.tolist()], dtype=torch.long)
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[0]
    gt = logits.argmax(dim=-1).numpy().astype(np.int32)
    tokens = generated.numpy().astype(np.int32)
    assert len(tokens) == seq_len
    start = len(prompt_ids) - 1
    assert np.array_equal(gt[start:seq_len - 1], tokens[start + 1:seq_len])
    return tokens, gt


def save_model_inputs(name, weights_dir, model, tokenizer):
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    assert len(prompt_ids) < 16

    tokens16, gt16 = generate_sequence(model, tokenizer, prompt_ids, 16)
    tokens512, gt512 = generate_sequence(model, tokenizer, prompt_ids, 512)
    assert np.array_equal(tokens16, tokens512[:16])
    assert np.array_equal(gt16, gt512[:16])

    weights_dir.mkdir(parents=True, exist_ok=True)
    tokens16.tofile(weights_dir / "tokens_16_i32.bin")
    gt16.tofile(weights_dir / "gt_argmax_16_i32.bin")
    tokens512.tofile(weights_dir / "tokens_512_i32.bin")
    gt512.tofile(weights_dir / "gt_argmax_512_i32.bin")
    tokens512.tofile(weights_dir / "tokens_i32.bin")
    gt16.tofile(weights_dir / "gt_argmax_i32.bin")

    metadata = {
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "tokens16": tokens16.tolist(),
        "gt16": gt16.tolist(),
        "tokens512_sha256": hashlib.sha256(tokens512.tobytes()).hexdigest(),
    }
    (weights_dir / "real_inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{name}: prompt={prompt_ids} tokens16={tokens16.tolist()} "
        f"tokens512_head={tokens512[:20].tolist()}"
    )


def save_gemma_rope(weights_dir, seq_len=512):
    head_dim = 256
    scale = 65536.0
    position = np.arange(seq_len, dtype=np.float64)[:, None]
    index = np.arange(head_dim // 2, dtype=np.float64)[None, :]
    for base, tag in ((10000.0, "local"), (1_000_000.0, "global")):
        theta = position * np.power(base, -2.0 * index / head_dim)
        cos = np.clip(np.round(np.cos(theta) * scale), -(2**31), 2**31 - 1)
        sin = np.clip(np.round(np.sin(theta) * scale), -(2**31), 2**31 - 1)
        cos.astype(np.int32).tofile(weights_dir / f"rope_cos_{tag}_i32.bin")
        sin.astype(np.int32).tofile(weights_dir / f"rope_sin_{tag}_i32.bin")


def main():
    torch.set_num_threads(TORCH_THREADS)

    gpt2_dir = ROOT / "models/gpt2/weights"
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
        ROOT / "models/gpt2/tokenizer", local_files_only=True
    )
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    gpt2_model.eval()
    save_model_inputs("gpt2", gpt2_dir, gpt2_model, gpt2_tokenizer)

    gemma_hf = ROOT / "models/gemma3/hf"
    weights_link = gemma_hf / "model.safetensors"
    if not weights_link.exists():
        weights_link.symlink_to("../raw/model.safetensors")
    gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_hf, local_files_only=True)
    gemma_model = AutoModelForCausalLM.from_pretrained(
        gemma_hf, local_files_only=True, dtype=torch.float32
    )
    gemma_model.eval()
    gemma_weights = ROOT / "models/gemma3/weights"
    save_model_inputs("gemma3", gemma_weights, gemma_model, gemma_tokenizer)
    save_gemma_rope(gemma_weights)

    print("real prompt+response inputs generated")


if __name__ == "__main__":
    main()
