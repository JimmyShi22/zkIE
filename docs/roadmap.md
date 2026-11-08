# Roadmap

Current state: the three-stage commit / prove / verify pipeline runs for
GPT-2 124M, Gemma 3 270M and TimesFM 1.0 200M over a shared model commitment.
Weights and public LUTs are committed as Merkle-backed assets; each operator
step produces a relation receipt; the verifier checks openings, equations and
data-flow edges without re-running the model. Proofs are non-ZK.

## Next

### 1. Measure and publish current performance

Run the benchmark method in `docs/benchmarks.md` for every supported model and
input shape. Report commit, prove and verify separately, plus proof size and
peak resident memory. All current numbers are **待测** until this is done.
Do not reuse figures from removed designs.

### 2. Expand operator and model coverage

- Add new fixed-point operators only together with their algebraic relation and
  rejection tests.
- Add a new model as a configuration plus a shared-trace executor over the
  existing operator set; do not add model-specific proof logic to the verifier.
- Keep the public LUT path uniform: a new table is a committed asset, not a
  verifier constant.

### 3. Reduce proof size and verification cost

- Measure the contribution of asset openings, relation vectors and Merkle paths
  to `proof.bin`.
- Deduplicate repeated asset intervals and row openings where the trace permits
  it, without weakening the anchor or Merkle checks.
- Keep verification free of any model re-execution or weight read.

### 4. Witness hiding

The current proof exposes weights, LUTs and activations. A future zero-knowledge
variant must hide witness values while preserving the same public statement,
commitment binding and relation checks. This is a protocol change, not a
configuration switch; it needs its own specification and benchmark suite.

### 5. Hardware acceleration

The engine currently uses deterministic CPU kernels. Any accelerated kernel
must produce byte-identical results to the CPU reference and must be covered by
the same correctness gates before it is enabled in the production path.

### 6. Operational hardening

- Validate the public bundle and policy against the case configuration before
  accepting a run.
- Keep artifact manifests and digests checked end to end.
- Add negative tests for every rejection path described in `docs/spec.md`.
- Keep `zkie-check` able to validate assets, contracts and run reports without
  the prover.
