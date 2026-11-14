# Committing and absorbing the request W in the first part

## Scheme

- Scheme: `request-w-first`.
- Current code: `2e65871c`, following `cbefc8fb`.
- Previous code: `cbefc8fb`.

## Purpose

The first part now commits the request-level W when the plan carries it: the
member claim's row weights become the vector, the W is derived on the witness
stream, committed with its own private randomness and oracle, its root travels
in the message and is absorbed in the first chain before the request seals and
the samples are drawn. The verifier absorbs the same root and refuses a
plan/message mismatch. Every existing plan carries none, so old runs are
byte-identical.

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
| The first part commits the request W from the member claim's weights | object rule | `pending_row_weights`, `derive_request_w`, `commit_request_w` | the workspace builds and passes |
| The W root is absorbed before the request seals and the samples | order rule | `absorb_request_w_root` (tag 17) after the reduction root | the verifier mirrors the absorb and refuses a mismatch |
| Existing plans are byte-identical | stability rule | gated on the plan's field | hard gate unchanged |

## Conclusions

- The request W's commit/absorb is wired in both the prover and the verifier.
- No existing path changes; the toy proof is byte-identical.

## Open items

- The own group takes the request W's member and the two request rows join it;
  a session test with a member-carrying plan follows.
