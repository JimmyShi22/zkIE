import os
import shutil
import sys

import numpy as np

ROOT = "/data/zkIE"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def save(path, values):
    np.asarray(values, dtype=np.int32).tofile(path)
    print(f"wrote {path} {np.asarray(values).shape}")


def fill_gpt2():
    import onnx
    from onnx import numpy_helper

    model_path = "models/gpt2/gpt2.onnx"
    if not os.path.exists(model_path):
        return
    model = onnx.load(model_path)
    inits = {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in model.graph.initializer
    }
    wpe = inits["model.transformer.wpe.weight"]
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50257, (512,), dtype=np.int64)
    out = "models/gpt2/weights"
    save(f"{out}/tokens_i32.bin", tokens.astype(np.int32))
    pos = np.zeros((512, 1024), dtype=np.float64)
    pos[:, :wpe.shape[1]] = wpe[:512]
    save(f"{out}/pos_emb_i32.bin", np.clip(np.round(pos * 65536.0), -(2**31), 2**31 - 1))
    if os.path.exists(f"{out}/gt_argmax_i32.bin"):
        shutil.copyfile(f"{out}/gt_argmax_i32.bin", f"{out}/gt_argmax_512_i32.bin")


def fill_gemma():
    from extract_gemma3 import H_PAD, SEQ, VOCAB, Loader, pad2, q

    raw = "models/gemma3/raw/model.safetensors"
    if not os.path.exists(raw):
        return
    loader = Loader(raw)
    emb = loader.arr("model.embed_tokens.weight").astype(np.float64)
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, VOCAB, (SEQ,)).astype(np.int32)
    out = "models/gemma3/weights"
    save(f"{out}/tokens_i32.bin", tokens)
    save(f"{out}/embed_tokens_i32.bin", pad2(q(emb * np.sqrt(640.0)), VOCAB, H_PAD))


def fill_deepseek():
    from extract_deepseek import H, SEQ, VOCAB, Sharded, pad2, q

    raw = "models/deepseek-v2-lite/raw"
    if not all(os.path.exists(f"{raw}/model-0000{i}-of-000004.safetensors") for i in range(1, 5)):
        return
    loader = Sharded()
    emb = loader.arr("model.embed_tokens.weight").astype(np.float64)
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, VOCAB, (SEQ,)).astype(np.int32)
    out = "models/deepseek-v2-lite/weights"
    os.makedirs(out, exist_ok=True)
    save(f"{out}/tokens_i32.bin", tokens)
    save(f"{out}/embed_tokens_i32.bin", pad2(q(emb), VOCAB, H))


fill_gpt2()
fill_gemma()
fill_deepseek()
