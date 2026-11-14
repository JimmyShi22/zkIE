# The request chain's transcript section

## Scheme

- Scheme: `chain-section`.
- Current code: `613f5c0d`, following `f38904f8`.
- Previous code: `f38904f8`.

## Purpose

A request's sub-proof leaves three masked sumchecks: the kept token sumcheck,
the batched stack and the weight merge. Their proofs travel as one transcript
section, placed after the sub-proof's other messages. The section carries its
own version; the round and shown end-value counts of each proof come from the
plan's claim shapes, so the encoder refuses proofs of another shape and the
decoder refuses another version, another stored count or truncated bytes
instead of reinterpreting them.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_chain`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused section tests | 2 passed |
| workspace | 51 suites, 1055 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The three proofs travel as one section with its own version | section rule | `RequestChainMessages`, `REQUEST_CHAIN_VERSION` | focused test round-trips a section |
| The counts come from the plan's claim shapes | plan-derived shape rule | `RequestChainShape`, encoder and decoder checks | focused test refuses another round or end-value count and a section of another shape |
| A section of another version or truncated bytes is refused | version and mismatch rule | decoder checks | focused test |

## Conclusions

- The section's codec, version and shape checks are in place and tested.
- No session message is touched yet; the toy proof is byte-identical.

## Open items

- The plan flag, the message field and the chain's claim checks enter the
  hidden session next; the old paths must stay byte-identical.
