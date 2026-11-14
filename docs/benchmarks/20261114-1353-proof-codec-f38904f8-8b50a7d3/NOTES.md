# Canonical encoding of the general sumcheck proof

## Scheme

- Scheme: `proof-codec`.
- Current code: `f38904f8`, following `8b50a7d3`.
- Previous code: `8b50a7d3`.

## Purpose

The request-level sub-proofs carry general sumcheck proofs in the session
transcript. This block gives the proof one canonical encoding — the hiding sum,
the rounds, the last round and the end values, each with its count — and a
decoder that checks the expected round and end-value counts, so a message that
belongs to another claim shape is refused instead of being reinterpreted.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib psc_general`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused general-sumcheck tests | 11 passed |
| workspace | 51 suites, 1053 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The proof encodes canonically with its counts | proof encoding rule | `GeneralProof::encode_into` | focused test round-trips a proof |
| The decoder checks the round count and the end-value count | version and mismatch rule | `GeneralProof::decode` | focused test refuses another round count and another end-value count |
| A round carries at most its bounded coefficients | round shape rule | decoder bound | focused test refuses an over-wide round |

## Conclusions

- The general sumcheck proof has one canonical encoding and a checking decoder.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The request chain's transcript section and its claims enter the hidden
  session next.
