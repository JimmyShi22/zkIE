# zkIE — Zero-Knowledge Inference Engine

## Design Spec v1.0

**Date**: 2026-07-26
**Status**: Approved
**Project**: zkIE (Zero-Knowledge Inference Engine)

---

## 1. Project Overview

### 1.1 Positioning

SP1 is to RISC-V what zkIE is to AI inference: a modular, chip-based ZK proof system
where each "instruction" is an AI primitive (matmul, softmax, layer norm, etc.) rather
than a CPU opcode. The system produces EVM-verifiable proofs via Halo2 + KZG (BN254).

### 1.2 Deliverables (Phase 1)

1. **zkie-core** — AI primitive ISA with modular Halo2 chip implementations
2. **zkie-compiler** — ONNX graph → ISA instruction sequence compiler
3. **zkie-cli** — `zkie prove` / `zkie verify` / `zkie evm-verifier`
4. **ZkIEVerifier.sol** — ERC-8274 `IProofVerifier` compatible on-chain verifier
5. **End-to-end proof**: TimesFM 1.0 (200M params) inference verified on local Anvil chain

### 1.3 Non-Goals (Phase 1)

- Private weights/model (weights are public in the circuit)
- Training proofs (inference only)
- GPU acceleration (CPU proving first)
- Distributed proving
- MLIR integration (ONNX path only)
- Fused ATTENTION chip (v1 uses DOT_GENERAL + SOFTMAX + ELTWISE composition)

---

## 2. Architecture

### 2.1 Two-Layer Design

```
Layer 2: Model Compiler (zkie-compiler)
  - Parses ONNX graph via oxionnx-ops
  - Maps ONNX ops 1:1 to ISA instructions
  - Topological sort + virtual register allocation
  - Output: CompiledProgram { instructions, weights }

Layer 1: AI Primitives (zkie-core)
  - Each ISA instruction = a Halo2 Chip
  - Each Chip = Plonkish custom gates + lookup arguments
  - Shared: I18/I36 fixed-point number system
  - Output: Halo2 circuit ready for proving
```

### 2.2 Data Flow

```
model.onnx + input.json
       │
       ▼  zkie-compiler
CompiledProgram { instructions: Vec<Instruction>, weights: HashMap }
       │
       ▼  zkie-core (chip assembly)
Halo2 circuit (Plonkish matrix)
       │
       ▼  Halo2 prover
proof.json + vk.key
       │
       ▼  halo2-solidity-verifier
verifier.sol ──→ deploy to Anvil
       │
       ▼  ERC-8274 wrapper
on-chain verify(proof) → ✅ true
```

---

## 3. ISA Instruction Set

### 3.1 Instruction Definitions

| Instruction    | Parameters                              | TimesFM Usage              |
|----------------|-----------------------------------------|----------------------------|
| DOT_GENERAL    | m, n, k, batch_dims, trans_a, trans_b   | Q/K/V/O projections, FFN   |
| SOFTMAX        | axis_dim                                | Attention softmax          |
| GELU           | —                                       | FFN activation             |
| LAYER_NORM     | dim, epsilon                            | Pre/post attention norm    |
| ELTWISE        | op: Add \| Mul                          | Residual, RoPE scale       |
| REDUCE         | op: Sum \| Mean, axis                   | Statistics                 |
| EMBED_LOOKUP   | table_size, embed_dim                   | Position encoding          |
| PATCH_EMBED    | patch_len, embed_dim                    | TimesFM patch → token      |

### 3.2 Fixed-Point Number System

- Input/Output: I18 (18 decimal places, ~10^18 precision)
- Accumulator: I36 (intermediate matmul results, overflow protection)
- Non-linear functions (GELU, softmax): Lookup table with 1024 entries
  - Table is a Halo2 lookup argument → O(N) constraints, not O(N^2)
- All values are BN254 field elements

### 3.3 Example: FFN Block Compilation

```
GELU(input=@0, output=@1)
DOT_GENERAL(m=seq, n=hidden, k=hidden, input_a=@1, input_b=W2, output=@2)
DOT_GENERAL(m=seq, n=hidden*4, k=hidden, input_a=@0, input_b=W1, output=@3)
ELTWISE(op=Mul, a=@2, b=@3, output=@4)
ELTWISE(op=Add, a=@0, b=@4, output=@5)   // residual connection
```

`@N` = virtual register for intermediate tensors. `W` = model weights (fixed).

---

## 4. Compilation Pipeline

### 4.1 ONNX → ISA Mapping

| ONNX Operator               | ISA Instruction              |
|-----------------------------|------------------------------|
| MatMul / Gemm               | DOT_GENERAL                  |
| Softmax                     | SOFTMAX                      |
| Gelu / Relu                 | GELU / ELTWISE(op=Relu)      |
| LayerNormalization          | LAYER_NORM                   |
| ReduceMean / ReduceSum      | REDUCE                       |
| Add / Mul                   | ELTWISE(Add\|Mul)            |
| Gather                      | EMBED_LOOKUP                 |
| Reshape / Transpose         | No instruction (metadata)    |
| Concat                      | Buffer splicing              |

### 4.2 Compiler Stages

1. **Parse**: Read ONNX graph via `oxionnx-ops`
2. **Map**: Each ONNX node → corresponding ISA instruction
3. **Schedule**: Topological sort ensuring input dependencies resolved
4. **Allocate**: Assign virtual register `@id` to each intermediate tensor
5. **Register**: Extract weights into dedicated lookup table
6. **Output**: `CompiledProgram { instructions, weights, shapes }`

### 4.3 Attention Handling (v1)

Attention is NOT recognized as a fused pattern. The ONNX attention subgraph
decomposes naturally into individual ops that map to:
- 4× DOT_GENERAL (Q, K, V projections + output)
- 1× SOFTMAX
- 2× ELTWISE (scale, residual add)

This is correct but not optimal. A fused ATTENTION chip can be added in v2.

---

## 5. Technology Stack

### 5.1 Languages

| Layer              | Language   | Reason                              |
|--------------------|------------|-------------------------------------|
| Core (chips, ISA)  | Rust       | Halo2 ecosystem is pure Rust        |
| Compiler           | Rust       | Same ecosystem, oxionnx-ops crate   |
| CLI                | Rust       | Unified binary                      |
| Contracts           | Solidity   | EVM target                          |
| Test scripts       | Python     | PyTorch → ONNX export              |

### 5.2 Key Dependencies

```toml
halo2_proofs = "0.3"              # Plonkish constraint system
halo2_solidity_verifier = "..."   # EVM verifier generator
ark-bn254 = "0.4"                 # BN254 elliptic curve
ark-ff = "0.4"                    # Finite field arithmetic
oxionnx-ops = "0.1"               # ONNX operator parsing (147 ops)
```

### 5.3 Project Structure

```
zkie/
├── Cargo.toml                    # workspace
├── crates/
│   ├── zkie-core/                # ISA + Chip trait + all chip impls
│   │   └── src/
│   │       ├── isa.rs            # Instruction enum
│   │       ├── chip.rs           # Chip trait
│   │       ├── tensor.rs         # Tensor<I18>, Tensor<I36>
│   │       ├── fixed_point.rs    # I18/I36 types + arithmetic
│   │       ├── lookup.rs         # Shared lookup table utilities
│   │       └── chips/
│   │           ├── dot_general.rs
│   │           ├── softmax.rs
│   │           ├── gelu.rs
│   │           ├── layer_norm.rs
│   │           ├── eltwise.rs
│   │           ├── embed_lookup.rs
│   │           ├── reduce.rs
│   │           └── patch_embed.rs
│   │
│   ├── zkie-compiler/            # ONNX → ISA compiler
│   │   └── src/
│   │       ├── onnx_parser.rs
│   │       ├── op_mapper.rs
│   │       └── graph_compiler.rs
│   │
│   └── zkie-cli/                 # CLI tool
│       └── src/
│           ├── main.rs
│           ├── prover.rs
│           └── verifier_gen.rs
│
├── contracts/                    # Solidity (Foundry)
│   ├── foundry.toml
│   ├── src/
│   │   └── ZkIEVerifier.sol     # ERC-8274 IProofVerifier
│   ├── script/
│   │   └── Deploy.s.sol
│   └── test/
│       └── VerifyProof.t.sol
│
├── py-tools/                     # Python helpers
│   ├── export_timesfm.py
│   ├── gen_test_input.py
│   └── test_pipeline.py
│
└── models/                       # ONNX model files (gitignored)
```

---

## 6. Smart Contract

### 6.1 ERC-8274 IProofVerifier Interface

```solidity
interface IProofVerifier {
    function verify(
        bytes32 inputHash,
        bytes32 outputHash,
        bytes calldata metadata,
        bytes calldata proof
    ) external returns (bool);

    function proofSystem() external view returns (string memory);
    function proofProfile() external view returns (bytes32);
}
```

### 6.2 ZkIEVerifier Implementation

- Wraps the Halo2 verifier generated by `halo2-solidity-verifier`
- `proofSystem()` returns `"zk/halo2"`
- `proofProfile()` returns keccak256 of circuit identifying info (model hash, chip versions)
- `metadata` encodes circuit configuration (circuit params, verifying key commitment)
- `inputHash` = keccak256(input_tensor)
- `outputHash` = keccak256(output_tensor)
- `proof` = encoded Halo2 proof (KZG openings + accumulator)

### 6.3 Deployment

- Target: Anvil (local testnet)
- Tool: Foundry (`forge script Deploy.s.sol --rpc-url anvil --broadcast`)
- Gas: Halo2 KZG verification is ~300k gas (pairing check)

---

## 7. Implementation Plan

### Phase 1: Foundation (core abstractions)
- `fixed_point.rs` — I18/I36 fixed-point system
- `tensor.rs` — Tensor type with shape tracking
- `chip.rs` + `isa.rs` — ISA definition + Chip trait
- `lookup.rs` — Shared lookup table utilities
- Unit tests for all

### Phase 2: Chip Implementation (easiest to hardest)
- ELTWISE (simplest, validates Halo2 pipeline)
- REDUCE
- GELU (lookup-based)
- EMBED_LOOKUP
- SOFTMAX (lookup + DOT_GENERAL composition)
- LAYER_NORM
- DOT_GENERAL (core, most complex)
- PATCH_EMBED
- Independent unit tests per chip

### Phase 3: Compiler
- ONNX parsing
- Op → instruction mapping
- Graph compilation (topological sort + register allocation)
- End-to-end test with a tiny model

### Phase 4: TimesFM Integration
- Export TimesFM 1.0 to ONNX
- Compile TimesFM ONNX → ISA
- Circuit assembly + proof generation
- Verify inference correctness (compare with PyTorch output)

### Phase 5: On-Chain Verification
- Generate verifier.sol via `halo2-solidity-verifier`
- Implement ERC-8274 ZkIEVerifier wrapper
- Deploy to Anvil with Foundry
- Submit proof on-chain → verify passes

---

## 8. Success Criteria (Phase 1)

1. `zkie prove timesfm.onnx input.json` produces a valid Halo2 proof
2. `zkie verify proof.json vk.key` returns `true`
3. `zkie evm-verifier vk.key` generates `verifier.sol`
4. `ZkIEVerifier.sol` deploys to Anvil
5. On-chain `verify(proof)` returns `true`
6. Inference output matches reference PyTorch output (within quantization tolerance)

---

## 9. Future Directions (Post Phase 1)

- Fused ATTENTION chip (efficiency optimization)
- MLIR integration (PyTorch → Torch-MLIR → zkie dialect)
- Model weight privacy (weights as private witnesses)
- Recursive proof composition (proof of multiple inferences)
- GPU-accelerated proving
- Distributed proving across machines
- Gemma 4 / GPT-2 / Llama model support
