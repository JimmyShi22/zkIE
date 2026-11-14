# Chain source negatives

## Scheme

- Scheme: `source-negatives`.
- Current code: `30145541`, following `8cf677a4`.
- Previous code: `8cf677a4`.

## Purpose

The chain's verifier reads the first side and the read panels' second side from
the planner's cells. This block adds the source negatives at the chain: another
first-side cell, swapped second-side rows and a changed second-side cell each
fail the group's source ties against the proof. The weight cells are the
prover's maps; their changed-cell negative remains the pending matrix check.
The fixture's second side is made distinct per row so a swap is observable.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_chain`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused chain tests | 3 passed |
| workspace | 51 suites, 1066 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Another first-side cell fails the chain's source tie | source rule | changed left read | focused chain test |
| Swapped second-side rows or a changed cell are refused | source rule | distinct rows; swap and cell change | focused chain test |
| Weight cells remain bound through the pending matrix | source rule | matrix negative | pending-check block test |
| No new mechanism | same-object rule | the verifier reads the planned cells | the negatives reuse the existing ties |

## Conclusions

- The chain refuses another read cell and swapped read rows; changed weight
  cells stay bound by the pending matrix check.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The member/W + second chain, at_start ledger, request-level parameters and
  the batch report follow.
