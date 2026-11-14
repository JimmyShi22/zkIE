# Deriving the request W

## Scheme

- Scheme: `request-w-derive`.
- Current code: `8aa853e3`, following `e5f70423`.
- Previous code: `e5f70423`.

## Purpose

The first part needs the request-level W = g^T M. The session derives it on the
same table stream the witness uses: one pass over the matrix rows, each column
accumulating the row's cell times the row's weight, with a weight vector of the
matrix's row count required. The W's own private randomness — one pad per
answer, one tail per message row and tail coefficient, one salt per tree
column — is drawn from the same private source only when the plan carries the
request W, under its own tree.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_request_w_is_the_weighted_table_rows`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused derive test | 1 passed |
| workspace | 51 suites, 1077 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| W is the weighted combination of the table's rows | object rule | `derive_request_w` on the witness stream | focused test matches the committed messages |
| The weight vector covers the matrix's rows | layout rule | length check | focused test refuses another length |
| The W's randomness is drawn only when the plan carries it | gating rule | `RequestWRandomness` drawn from the same private source | the field is absent in every existing plan |
| Old paths are unchanged | stability rule | gated draw and commit | hard gate unchanged |

## Conclusions

- The request W can be derived from the committed table on the witness stream.
- No proof path commits it yet; that closes the first part next.

## Open items

- The first part commits the derived vector, absorbs its root in the chain and
  keeps the tree; the verifier absorbs the same root.
