# The request W's sampling values in the witness

## Scheme

- Scheme: `request-w-witness`.
- Current code: `beabca5c`, following `66f06b42`.
- Previous code: `66f06b42`.

## Purpose

The prover's witness keeps the request W's tail and salts beside its committed
message, so the own group's sampling can build the member's sample rows from the
same private values the commit used. The field starts empty in every existing
path, so old runs are unchanged.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: the workspace's hidden-session tests.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| workspace | 51 suites, 1078 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The witness keeps the tail and salts the sampling needs | witness rule | `RequestWWitness` | the workspace builds and passes |
| Existing paths carry the empty value | stability rule | constructed as empty | hard gate unchanged |

## Conclusions

- The sampling values of the request W travel with its commitment.
- No existing path changes; the toy proof is byte-identical.

## Open items

- The own group's member takes the commit's message, tree and sampling values;
  the callers pass the request rows; then the shared and second tests run.
