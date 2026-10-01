#!/usr/bin/env python3
"""Export GPT-2 (124M) to models/gpt2/gpt2/gpt2.onnx (+ .onnx.data) and print the op histogram.

Loads HuggingFace gpt2 (GPT2LMHeadModel), runs one static forward pass with
use_cache=False, and torch.onnx.export (dynamo, opset 18) writes a fixed-shape
graph. Context length is parameterised by GPT2_SEQ (default 16) so the op set
is identical while the graph stays small for analysis.

Run inside Python >= 3.10 with torch, transformers, onnx, onnxscript installed.
"""
import os
from collections import Counter

import torch
from transformers import GPT2LMHeadModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
SEQ = int(os.environ.get("GPT2_SEQ", "16"))


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def main():
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded GPT2 with {n:,} parameters, vocab {model.config.vocab_size}")

    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ))
    attention_mask = torch.ones(1, SEQ, dtype=torch.long)
    position_ids = torch.arange(0, SEQ, dtype=torch.long).unsqueeze(0)

    os.makedirs(MODELS, exist_ok=True)
    out_path = os.path.join(MODELS, "gpt2/gpt2.onnx")
    print(f"exporting seq={SEQ} (torch dynamo, opset 18)...")
    torch.onnx.export(
        Wrapper(model),
        (input_ids, attention_mask, position_ids),
        out_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
    )
    size = os.path.getsize(out_path)
    data_size = os.path.getsize(out_path + ".data") if os.path.exists(out_path + ".data") else 0
    print(f"done: {out_path} ({size} bytes) + .data ({data_size} bytes)")

    import onnx

    m = onnx.load(out_path, load_external_data=False)
    hist = Counter(n.op_type for n in m.graph.node)
    print("=== op histogram ===")
    for k, v in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")
    print("=== node count ===", len(m.graph.node))


if __name__ == "__main__":
    main()