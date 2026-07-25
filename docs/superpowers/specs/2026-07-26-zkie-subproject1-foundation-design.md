# zkIE Sub-Project 1 — Foundation Layer + First Chip (End-to-End Proof Roundtrip)

**Date**: 2026-07-26
**Status**: Approved (design), pending implementation
**Parent idea**: `2026-07-26-zkie-design.md` (original zkIE concept doc)

---

## 1. Context and Goal

zkIE is a modular, chip-based ZK proof system for AI inference: SP1 is to RISC-V what
zkIE is to AI inference. Each AI primitive (matmul, softmax, layer norm, etc.) is
implemented as its own Halo2 chip; an ONNX graph compiles to a sequence of these
chip-instructions; the assembled circuit produces an EVM-verifiable KZG proof.

**Near-term product goal**: a complete, end-to-end working proof of inference for
**TimesFM 1.0** (200M params), verified on-chain on a local Anvil node. **Follow-on
goal**: extend chip/compiler coverage to support **Gemma 4**.

That goal is too large for one implementation pass. This spec covers only the first
step in a chain of sub-projects that together deliver the TimesFM goal.

### Sub-project roadmap (for context — only #1 is in scope here)

1. **Foundation + first chip, real proof roundtrip** (this spec) — validate that the
   fixed-point number system and Halo2 chip abstraction actually work, with a real
   (non-mocked) KZG prove/verify pipeline.
2. Remaining chips needed for TimesFM (SOFTMAX, GELU, LAYER_NORM, DOT_GENERAL,
   PATCH_EMBED, REDUCE, EMBED_LOOKUP).
3. ONNX compiler (graph parsing, op→instruction mapping, register allocation).
4. TimesFM end-to-end integration + EVM verifier contract + Anvil deployment.
5. (Post-TimesFM) Gemma 4 support — likely needs new/adapted chips (e.g. fused or
   different attention variants).

---

## 2. Scope of This Sub-Project

**In scope:**
- `zkie-core` crate (new Cargo workspace) containing:
  - `fixed_point.rs` — signed fixed-point number system over the BN254 scalar field
  - `tensor.rs` — minimal `Tensor<T>` (shape + flat data; no broadcasting)
  - `isa.rs` — `Instruction` enum with all 8 ISA variants defined (only ELTWISE
    implemented behind it — the rest are placeholders for sub-project 2)
  - `chip.rs` — `Chip` trait that all chip implementations will follow
  - `chips/eltwise.rs` — ELTWISE chip supporting `Add` and `Mul` only
- One integration test that performs a **real KZG trusted setup → proof generation →
  verification** roundtrip for a small ELTWISE circuit (not `MockProver` only).
- Unit tests for fixed-point arithmetic, especially the requantization step.

**Out of scope (deferred to later sub-projects):**
- ONNX parsing, CLI, Solidity/EVM verifier, Anvil deployment
- ELTWISE's `Relu` variant (needs a sign/comparison gate — separate concern)
- SOFTMAX, GELU, LAYER_NORM, DOT_GENERAL, REDUCE, EMBED_LOOKUP, PATCH_EMBED
  implementations (ISA variants are declared, not implemented)
- TimesFM/Gemma 4 model-specific work

**Success criteria for this sub-project:**
1. `cargo test` passes, including one test that runs a real (non-mock) KZG
   setup→prove→verify roundtrip for an ELTWISE(Add) and an ELTWISE(Mul) circuit.
2. Fixed-point arithmetic has unit tests covering: representable range boundaries,
   negative values, and multiplication requantization rounding behavior.
3. The `Chip` trait and `Instruction` enum are shaped so that sub-project 2 can add
   new chips without changing the trait's shape (validated by having at least one
   stub/placeholder variant exercised, even if unimplemented).

---

## 3. Architecture

### 3.1 Why PSE's `halo2_proofs`, not zcash/halo2

The original concept doc listed `ark-bn254`/`ark-ff` (arkworks) as dependencies. This
is corrected here: the halo2 ecosystem's own `halo2curves` crate is used instead,
because:
- PSE's (`privacy-scaling-explorations/halo2`) fork adds **KZG** polynomial
  commitment support over BN254, which zcash/halo2 (IPA-only, no trusted setup) does
  not have.
- KZG-over-BN254 is what downstream EVM verifier generation
  (`halo2-solidity-verifier`-style tooling) targets in sub-project 4. Choosing IPA now
  would force a rewrite later.
- Exact crate names/versions are **not pinned in this spec** — implementation will
  resolve current, actively-maintained versions via crates.io / GitHub search per
  the project's research-first convention, rather than guessing at version numbers.

### 3.2 Fixed-point number system

- **I18**: values are quantized to 18 decimal places (scale = 10^18), stored as
  signed integers on the host side and as range-checked field elements (`Fr` from
  `halo2curves::bn256`) inside the circuit.
- **Signed representation in a prime field**: witnessed values are constrained (via
  a range-check gadget) to stay within a bounded interval (e.g. `±2^63`), far smaller
  than the BN254 scalar field's ~254-bit modulus. As long as a value never leaves
  that bound, field addition/subtraction behaves exactly like signed integer
  arithmetic — no modular "wraparound" can occur. This range check is itself a
  circuit constraint, not just a host-side assertion, since a malicious prover must
  not be able to submit out-of-range witnesses.
- **I36 (multiplication accumulator)**: multiplying two Q18 values yields a Q36
  intermediate (scale = 10^36). To get back to a Q18 result, the circuit witnesses a
  quotient `q` and remainder `r` such that `q18_a * q18_b = q * 10^18 + r`, with `r`
  range-checked to `0 <= r < 10^18`. `q` is the requantized product. This
  quotient/remainder decomposition is the standard "fixed-point rescale" gadget and
  is the piece most likely to hide a soundness bug (e.g. if the remainder range
  check is missing or wrong, a prover could choose an invalid `q`/`r` pair and forge
  a result) — it gets the most unit-test attention.
- Non-linear functions (softmax, GELU) will use a lookup-table approach in later
  sub-projects; not needed for ELTWISE Add/Mul.

### 3.3 `Chip` trait and `Instruction` enum

- `isa.rs` defines an `Instruction` enum with all 8 variants from the original
  design (`DotGeneral`, `Softmax`, `Gelu`, `LayerNorm`, `Eltwise`, `Reduce`,
  `EmbedLookup`, `PatchEmbed`), matching the parameter shapes documented in the
  parent design doc. Only `Eltwise` is backed by a working chip in this
  sub-project; others exist so the enum's shape is validated early and sub-project 2
  doesn't need to redesign the ISA.
- `chip.rs` defines a `Chip` trait (configure/synthesize-style, following Halo2's
  usual chip pattern: a `Config` struct produced by `configure(meta)`, and a
  `synthesize`/`assign` method that lays out the circuit given a `Layouter`).
- `chips/eltwise.rs` implements this trait for `Add` and `Mul`, using a custom gate
  over `Fr` (Add) and the quotient/remainder rescale gadget described above (Mul).

### 3.4 Data flow (this sub-project only)

```
test harness
   │  build small ELTWISE circuit (Add or Mul) with witnessed I18 inputs
   ▼
zkie-core::chips::eltwise::EltwiseChip
   │  configure() → gate/lookup config
   │  synthesize() → assigns witnesses, enforces constraints
   ▼
halo2_proofs (PSE, KZG backend)
   │  setup(k) → SRS params
   │  keygen_vk / keygen_pk
   │  create_proof(params, pk, circuit, instances) → proof bytes
   │  verify_proof(params, vk, instances, proof) → Ok(()) / Err
   ▼
test asserts verification succeeds (and, for a negative test, that a tampered
witness/proof fails verification)
```

### 3.5 Error handling

- Circuit-level: malformed witnesses (out-of-range fixed-point values, wrong
  remainder in the rescale gadget) must fail constraint satisfaction — this is
  enforced by the circuit itself, not by host-side validation. Tests must include at
  least one negative case per gadget (out-of-range witness, wrong quotient/remainder)
  to confirm the constraints actually reject bad witnesses.
- Host-level (Rust API): fixed-point conversion functions (`f64` → `I18` etc.)
  return `Result` and reject values that don't fit the representable range, rather
  than silently truncating.

### 3.6 Testing

- Unit tests in `fixed_point.rs`: range boundaries (max/min representable I18),
  negative value round-tripping, multiplication requantization rounding (including
  a case exercising the remainder boundary).
- Unit tests in `chips/eltwise.rs` using `MockProver` for fast constraint-level
  iteration during development.
- One integration test (`tests/` in `zkie-core`) doing the real KZG
  setup→prove→verify roundtrip described in 3.4, for both `Add` and `Mul`, including
  a negative case (tampered proof/instance must fail verification).

---

## 4. Non-Goals (this sub-project)

Same non-goals as the parent design (no private weights, no training proofs, no GPU
acceleration, no distributed proving) plus: no ONNX, no CLI, no Solidity, no
Relu/other chips beyond Add/Mul ELTWISE.

---

## 5. Open Items Resolved During Implementation

- Exact `halo2_proofs`/`halo2curves` crate source and version (research via
  crates.io/GitHub at implementation time).
- Exact range-check gadget construction (e.g. bit-decomposition + lookup vs. other
  standard Halo2 range-check patterns) — chosen based on what's idiomatic in the
  resolved halo2 fork's ecosystem.
