# zkIE Op Fixed-Point Semantics Contract (Authoritative)

Date: 2026-09-28
Status: Authoritative definition that every numeric op MUST follow
Scope: The core numeric ops required by TimesFM 200M, plus layout and head/tail op handling

## 1. Global conventions

- Field: Goldilocks (64-bit)
- Signed-integer embedding: negative x maps to P minus abs(x) (twos-complement-style)
- Rounding: round-half-up everywhere (the div_round rule)
- Padding: every tensor length must be a power of two (WHIR MLE requirement); the real dimension is passed explicitly as n_real, and norm reductions only sum the first n_real entries, ignoring padding
- Three scale tiers:
  - activations / weights = 2^16 (i32 embedding)
  - matmul accumulation output = 2^32 (i64 embedding, 2^16 times 2^16)
  - norm raw output = 2^48 (three 2^16 factors multiplied)

## 2. Per-op contracts

### MatMul
- inputs A at scale 2^16, B at scale 2^16; output C = A times B at scale 2^32
- no rounding (exact in the field); accumulation in i64 (2^32), no overflow for TimesFM shapes
- m / k / n must be powers of two

### Add
- elementwise, equal length, scale unchanged (2^16 plus 2^16 = 2^16)

### Affine (rescale + bias, optional ReLU)
- input at 2^32; output out = round(in / 2^shift) + bias at scale 2^16
- round-half-up; shift = 16 brings matmul output back to activation scale
- relu = true: out = max(out, 0)

### Scale (multiply by constant + bias)
- inputs at 2^16, bias at 2^16, integer scale at 2^16
- output out = round(in times scale / 2^16) + bias
- round-half-up

### Softmax
- input shifted logits at 2^16 (scores minus max, <= 0)
- exp lookup: idx = shifted + offset (clamped to table range), e = exp_table[idx]
- sum = sum of all e; output = e / sum at 2^16
- table: exp_table; offset is a per-op public parameter

### LayerNorm
- inputs x at 2^16, w at 2^16, n_real
- mean = sum(x) / n_real; var = sum((x - mean)^2) / n_real (integer reduction over the first n_real entries)
- rstd = rsqrt_table[var_idx] (quantized 1 / sqrt(var + eps))
- raw = (x - mean) times rstd times w at 2^48, then rescaled twice back to 2^16

### RMSNorm
- inputs x at 2^16, w at 2^16, n_real
- s = mean(x^2) (integer reduction over the first n_real entries)
- rstd = rsqrt_table[s_idx]
- raw = x times rstd times w at 2^48, then rescaled twice back to 2^16

### ReLU
- input at 2^32; output out = max(round(in / 2^16) + bias, 0) at 2^16
- round-half-up, SHIFT = 16

## 3. Layout ops (no proof, layout rules only)

Reshape / Transpose / Split / Concat / Squeeze / Unsqueeze / Cast / Clip:
these do not change values, only reshape or re-view. The proving layer treats them as free; the compiler applies layout rules (Split for QKV, Transpose for K, and so on).

## 4. Head/tail special ops (precomputed at extraction time, not proved)

Sin / Cos (positional encoding), Where / Less / GreaterOrEqual / Equal / Not / Min / Max (mask logic),
ArgMax / Gather / GatherND / GatherElements / Pad (indexing and gathering), ReduceSum / ReduceMin / ReduceMax (reductions),
Sigmoid / Abs / Clip / Cast / Sub / Div / Mod (misc head/tail logic).
These are precomputed into constants by the extract scripts and do not enter the GKR proof.
