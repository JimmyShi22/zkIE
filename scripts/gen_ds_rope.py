import os
import numpy as np

SEQ = int(os.environ.get("M", "16"))
QK_ROPE = 64
ROPE_BASE = 10000.0

pos = np.arange(SEQ, dtype=np.float64)[:, None]
i = np.arange(QK_ROPE // 2, dtype=np.float64)[None, :]
theta = pos * np.power(ROPE_BASE, -2.0 * i / QK_ROPE)


def q(x):
    return np.clip(np.round(x * 65536.0), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)


q(np.cos(theta)).tofile("/work/models/deepseek-v2-lite/weights/rope_cos_i32.bin")
q(np.sin(theta)).tofile("/work/models/deepseek-v2-lite/weights/rope_sin_i32.bin")
print("rope saved", theta.shape)

