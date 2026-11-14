# The request W's tree in the witness

## Scheme

- Scheme: `witness-request-w`.
- Current code: `cbefc8fb`, following `8aa853e3`.
- Previous code: `8aa853e3`.

## Purpose

The prover's witness now carries the request-level W's committed value message:
its tree, value messages and roots, kept for the later openings. Every existing
path constructs the witness with the empty value, so old runs are byte- and
shape-identical.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: the workspace's hidden-session tests.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| workspace | 51 suites, 1077 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The witness keeps the request W's tree and messages | witness rule | the witness field | the workspace builds and passes |
| Existing paths carry the empty value | stability rule | constructed as empty | hard gate unchanged |

## Conclusions

- The witness can hold the request W's commitment.
- No proof path fills it yet; the first part does next.

## Open items

- The first part commits the derived W, absorbs its root and fills the witness;
  the verifier absorbs the same root.
