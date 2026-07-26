# zkIE EVM Verifier Attempt — Solidity/EVM Verification of a halo2 KZG Proof

Date: 2026-07-26

## TL;DR

- **Approach 1 (feed a real zkie-core v0.4.0 proof through a v0.3.0-pinned
  verifier-generation crate) does NOT work.** Confirmed by direct source
  comparison, not by trial and error: `VerifyingKey`'s binary format changed
  incompatibly between halo2 `v0.3.0` and `v0.4.0` (different version byte,
  different struct fields, different `read()` signature entirely). This is a
  hard, structural incompatibility, not a minor version skew.
- **Approach 2 (a fully decoupled, standalone toy circuit built directly
  against halo2 `v0.3.0`, matching what `halo2-solidity-verifier` needs)
  worked completely**, including full on-chain deployment and verification on
  a local Anvil node via Foundry, plus a negative (tampered-proof) test.
- **This is a genuine, partial result.** It proves the EVM-verification
  toolchain and pipeline work end-to-end for a halo2 KZG proof. It does
  **NOT** prove that any of zkie-core's real v0.4.0 model-inference circuits
  (the `EltwiseAdd`/`EltwiseMul`/`DotGeneral`/`Softmax`/etc. chips with the
  151 passing tests) can be verified on-chain — that remains blocked by the
  same halo2-solidity-verifier / halo2 v0.4.0 incompatibility described below,
  and would require either a halo2-solidity-verifier fork updated for the
  v0.4.0 frontend/backend/middleware split, or downgrading zkie-core itself
  (not recommended — explicitly out of scope, not attempted, and zkie-core
  was not modified).

## Why this gap exists (recap, verified against source)

- `halo2-solidity-verifier`'s `Cargo.toml` (cloned at commit `9e99730a`,
  2024-10-01, from `privacy-scaling-explorations/halo2-solidity-verifier`,
  which now lives at `privacy-ethereum/halo2-solidity-verifier`) pins:
  ```toml
  halo2_proofs = { git = "https://github.com/privacy-scaling-explorations/halo2", tag = "v0.3.0" }
  ```
- zkie-core/zkie-compiler pin `halo2_proofs` to git tag `v0.4.0` (unchanged,
  not touched in this attempt).
- PR #21 on halo2-solidity-verifier ("Bump halo2") to move to v0.4.0's
  frontend/backend/middleware split was never merged, because it depended on
  halo2's own PR #396 ("Expose required types and functions for external
  verifier generation"), which merged into halo2's `main` branch *after*
  `v0.4.0` was tagged — no halo2 tag has been cut since that included it. So
  halo2-solidity-verifier's main branch genuinely cannot build against exact
  tag `v0.4.0` today.

## Approach 1: attempted, ruled out by source inspection (not by a failing test run)

Per the plan, I checked whether halo2's own binary serialization format for
`VerifyingKey`/`ParamsKZG` is stable across v0.3.0 → v0.4.0, by reading both
versions' actual serialization code side by side (cloned both tags locally
under `.spike-test/halo2-v0.3.0-src` and `.spike-test/halo2-v0.4.0-src`,
gitignored, not committed).

**Result: incompatible.** Concretely, in
`halo2_proofs/src/plonk.rs` (v0.3.0) vs `halo2_backend/src/plonk.rs` (v0.4.0):

| | v0.3.0 | v0.4.0 |
|---|---|---|
| VK version byte | `const VERSION: u8 = 0x03;`, written first, checked with strict `==` on read | `const VERSION: u8 = 0x04;` |
| VK struct fields | includes `selectors: Vec<Vec<bool>>`, `compress_selectors: bool` (written to the byte stream when selector compression is off) | those fields don't exist — selector handling moved into the frontend/backend split |
| `VerifyingKey::read()` signature | `read<R, ConcreteCircuit: Circuit<C::Scalar>>(reader, format, #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params)` — internally re-runs `ConcreteCircuit::configure()` to rebuild the `ConstraintSystem` | `read<R>(reader, format, cs: ConstraintSystemBack<C::Scalar>)` — takes an already-compiled `ConstraintSystemBack` as an argument instead of a `Circuit` type param |

Reading v0.4.0-written VK bytes with v0.3.0's `VerifyingKey::read()` would
fail at the very first byte (`VERSION != version_byte[0]` → immediate
`io::Error`), before even reaching the deeper structural mismatch (different
field layout, and a completely different, non-interchangeable function
signature that wouldn't even type-check against a v0.4.0 caller). This is
conclusive: no byte-compat shim is possible without rewriting the read/write
logic itself, which is out of scope (would mean re-implementing chunks of
halo2's internals, not "feeding serialized data through").

I did **not** additionally write and run a throwaway "deserialize v0.4.0
bytes with v0.3.0 code" experiment to watch it panic — the version-byte
check alone, plus the incompatible `read()` signatures (which wouldn't even
compile in a shared caller), make the outcome unambiguous from source
inspection alone, and I judged that an actual failing-test dramatization
would burn time without adding information. If a future session wants that
literal artifact, it's a 20-line addition.

For completeness: `ParamsKZG` (the full prover parameters — `k, n, g,
g_lagrange, g2, s_g2`) appear structurally unchanged between the two
versions and its `write_custom`/`read_custom` byte layout looks compatible.
This doesn't help, though, since the VK format (and the `VerifyingKey::read`
signature specifically) is what actually blocks the pipeline — halo2-solidity-verifier's own `read_params`
helper for the *verifier*-side params (`ParamsVerifierKZG`) is itself new in
v0.4.0 (didn't exist as a distinct type in v0.3.0; v0.3.0's verifier consumes
`&ParamsKZG` directly), so even the params side has API-shape differences,
just not byte-format ones.

**Conclusion: Approach 1 is not viable as described.** Moved to Approach 2.

## Approach 2: standalone toy circuit on halo2 v0.3.0 — SUCCEEDED

### What was built

A new, fully standalone Rust crate:

- `/Users/jimmyshi/code/zkie/zkie-evm-verifier-poc/`
  - `Cargo.toml` — has its own empty `[workspace]` table so it does **not**
    join the repo root's workspace (which pins `halo2_proofs` v0.4.0). This
    avoids any Cargo dependency-resolution conflict between the two
    different git revisions of `halo2_proofs` — I did not even need to test
    whether Cargo could resolve both in one workspace, since keeping them
    fully separate sidesteps the question entirely (this matches the "or its
    own separate `[workspace]` block" option from the task brief).
  - `halo2_proofs` pinned to git tag `v0.3.0` (same as
    halo2-solidity-verifier).
  - `halo2_solidity_verifier` as a plain git dependency (default features —
    the `evm` feature, which pulls in `revm` for local-EVM testing, was
    **not** enabled, since we deploy to real Anvil instead).
  - `src/main.rs` — reimplements halo2's own
    `halo2_proofs/examples/simple-example.rs` (tag v0.3.0) circuit verbatim
    (a single multiplication gate: `c = constant * a² * b²`, one instance
    column exposing `c`), generalized from `pasta::Fp` to `bn256::Fr` since
    KZG-on-BN254 is what the EVM verifier needs. This does **not** reuse any
    zkie-core chip — those remain untouched and v0.4.0-pinned.

### Pipeline executed (all real, no mocks)

1. `MockProver::run(k=4, ...).verify()` — sanity check, passed.
2. `ParamsKZG::<Bn256>::setup(k, &mut OsRng)` — real (insecure, POC-only)
   KZG trusted setup.
3. `keygen_vk` / `keygen_pk` — real key generation.
4. `create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK, ...>` using
   halo2-solidity-verifier's own `Keccak256Transcript` (required — the EVM
   verifier does Fiat-Shamir with Keccak, not Blake2b like zkie-core's
   internal tests use). Produced a **real 1376-byte KZG proof**.
5. `verify_proof::<..., VerifierSHPLONK, ...>` off-chain — passed, confirming
   the proof is valid before touching Solidity.
6. `halo2_solidity_verifier::SolidityGenerator::new(&params, &vk, Bdfg21, 1).render()`
   — rendered a real 883-line, 44,482-byte `Halo2Verifier.sol` with the
   verifying key embedded.
7. `halo2_solidity_verifier::encode_calldata(None, &proof, &instances)` —
   produced real ABI calldata for `verifyProof(bytes,uint256[])` (selector
   `0x1e8e1e13`), 1540 bytes / 3082 hex chars, for both the valid proof and a
   tampered copy (one byte flipped at the midpoint).

Command used: `cd /Users/jimmyshi/code/zkie/zkie-evm-verifier-poc && cargo run`
(from inside the project tree, per the repo's hard constraint on build-script
execution paths).

Artifacts written to `zkie-evm-verifier-poc/out/` (small, kept in place, not
gitignored — see below):
- `Halo2Verifier.sol` (44,482 bytes)
- `valid_calldata.hex` (0x-prefixed, 3082 hex chars)
- `tampered_calldata.hex` (0x-prefixed, 3082 hex chars)
- `instances.json` (human-readable summary: k=4, constant=7, a=2, b=3,
  c_instance=0xfc=252, proof_len_bytes=1376, num_instances=1)

### Foundry + Anvil deployment and on-chain verification

Foundry project at `zkie-evm-verifier-poc/contracts/` (`forge init --no-git --force .`,
then removed the default `Counter.sol`/`Counter.s.sol`/`Counter.t.sol`
boilerplate; copied the generated verifier into `contracts/src/Halo2Verifier.sol`).

**Gotcha**: plain `forge build` failed with `Stack too deep` (the generated
verifier is one big function with heavy inline assembly). Fixed by adding to
`contracts/foundry.toml`:
```toml
via_ir = true
optimizer = true
optimizer_runs = 200
```
(This mirrors what halo2-solidity-verifier's own test helper does — it
invokes `solc --optimize --via-ir` directly.) After that, `forge build`
succeeded (Solc 0.8.30, two harmless "unused parameter" warnings only).

Foundry/Anvil/cast versions: `1.3.1-stable` (commit `08d3a4ad4d`,
2025-08-15), from `~/.foundry/bin/`.

Steps and exact results:

1. Started `anvil --port 8545` in the background (default well-known
   test accounts/keys — public, safe, no real funds).
2. Deployed with:
   ```
   forge create --rpc-url http://127.0.0.1:8545 \
     --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
     --broadcast src/Halo2Verifier.sol:Halo2Verifier
   ```
   Deployed to `0x5FbDB2315678afecb367f032d93F642f64180aa3`
   (tx `0xeae2ca1659df889dbaea225216acbed52aa6e001e14b8f756806aa0a3dbf557c`).
3. **Valid proof — `cast call`** (passing the pre-encoded raw calldata from
   `encode_calldata` directly, no ABI signature needed since it's already
   fully encoded):
   ```
   cast call --rpc-url http://127.0.0.1:8545 <addr> <valid_calldata.hex>
   ```
   → returned `0x0000...0001` — **`true`**, matching exactly what
   halo2-solidity-verifier's own internal test asserts
   (`[vec![0; 31], vec![1]].concat()`).
4. **Valid proof — `cast send`** (real mined transaction, not just a call):
   status `1 (success)`, `gasUsed 298990`, block 2,
   tx `0x0d64c9e20d5c03468d77f61b61f7c719961e187c51183ea9653afdeafb235a11`.
5. **Tampered proof (one byte flipped) — `cast call`**:
   `Error: server returned an error response: error code 3: execution reverted, data: "0x"`
   — **rejected**.
6. **Tampered proof — `cast send`**: gas estimation itself fails with the
   same "execution reverted" — the transaction cannot even be broadcast,
   confirming rejection at the strongest level Foundry can show.

**On-chain result: valid proof verifies as `true` (both as a call and as a
mined transaction); tampered proof is rejected (reverts) both as a call and
at gas-estimation time for a real transaction.** This is a real, complete,
positive-and-negative on-chain verification of a real halo2 KZG proof
end-to-end through Foundry/Anvil.

Anvil was stopped after testing (`pkill -f "anvil --port 8545"`); it is not
left running.

## Explicit honesty note

**This demonstrates the EVM-verification infrastructure works in principle,
decoupled from zkie-core's actual v0.4.0-based circuits.** The circuit
verified on-chain here is a trivial single-gate multiplication circuit
copied from halo2's own examples, built against a different `halo2_proofs`
revision (v0.3.0) than zkie-core uses (v0.4.0). It is **not** the same as
verifying one of zkie-core's real model-inference proofs
(`EltwiseAddChip`/`DotGeneralChip`/`SoftmaxChip`/etc., the 151-passing-test
suite) on-chain. No zkie-core or zkie-compiler files were modified; both
remain exactly as they were (v0.4.0-pinned, 151 tests passing, verified
separately with `cargo test --workspace` before starting this work).

## What would be needed to go further

To get a *real* zkie-core v0.4.0 circuit verified on-chain, one of the
following would be required (none attempted here — explicitly out of scope
per the task brief):

1. **Fork/patch halo2-solidity-verifier to target halo2 v0.4.0.** This means
   picking up the intent of the abandoned PR #21 and re-deriving the
   `SolidityGenerator`'s codegen (in `src/codegen.rs`/`src/evm.rs`) against
   the new `VerifyingKey`/`ConstraintSystemBack` shapes from
   `halo2_backend`/`halo2_frontend`/`halo2_middleware`. Nontrivial —
   `SolidityGenerator::new` reads deeply into `vk.cs()` internals
   (`num_advice_columns()`, `instance_queries()`, degree, etc.) whose exact
   shape changed with the split.
2. **Wait for/help land a new halo2 tag** that includes halo2 PR #396 merged
   to `main`, then re-attempt a bump of halo2-solidity-verifier to depend on
   that tag instead of `v0.4.0`'s pre-#396 state — the cleanest long-term
   fix, but outside this session's control (upstream release process).
3. Alternatively, use a *different* EVM-verifier generator that already
   targets v0.4.0/newer halo2 architectures, if one exists and is
   trustworthy (not investigated in this session — Approach 1/2 as
   specified were followed first).

## Files and paths

- Report (this file): `/Users/jimmyshi/code/zkie/docs/superpowers/specs/2026-07-26-zkie-evm-verifier-attempt.md`
- POC crate: `/Users/jimmyshi/code/zkie/zkie-evm-verifier-poc/`
  - `Cargo.toml`, `src/main.rs` — proof generation + Solidity/calldata export
  - `out/Halo2Verifier.sol`, `out/valid_calldata.hex`,
    `out/tampered_calldata.hex`, `out/instances.json` — generated artifacts,
    kept (small, useful for review/re-running the on-chain steps without
    regenerating)
  - `contracts/` — Foundry project (`foundry.toml`, `src/Halo2Verifier.sol`,
    default `lib/forge-std`, empty `script/`/`test/`)
  - `logs/anvil.log` — Anvil session log from the run above
- `.gitignore` updated (additively) to exclude this crate's build artifacts:
  `zkie-evm-verifier-poc/target`, `contracts/out`, `contracts/cache`,
  `contracts/lib`, `logs` (target/ alone was 539MB; contracts/lib is the
  cloned forge-std dependency).
- Reference clones used for source comparison (gitignored, not committed,
  under the existing `.spike-test/` which the root `.gitignore` already
  excludes): `.spike-test/halo2-solidity-verifier-src`,
  `.spike-test/halo2-v0.3.0-src`, `.spike-test/halo2-v0.4.0-src`.

Nothing was committed to git — left in place for supervisor review, per
instructions.
