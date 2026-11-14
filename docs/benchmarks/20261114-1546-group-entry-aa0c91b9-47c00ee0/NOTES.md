# The stack group's chain entry

## Scheme

- Scheme: `group-entry`.
- Current code: `aa0c91b9`, following `47c00ee0`.
- Previous code: `47c00ee0`.

## Purpose

The stack's whole end group now has its chain entry. The reduction packs the A
side, the read-only B side and one map per weight-panel row; the group's inner
sumcheck folds those packed maps for the rounds, records the challenges,
recomputes each row's end coefficient at the end and sends the unpacked row
values with the rest values the anchor sends. The entry proves and verifies
with the panel-product summand, and the received end values derive the hidden
group the ties check.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_stack`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused stack tests | 10 passed |
| workspace | 51 suites, 1062 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The group's maps are the A side, the read-only B side and one map per weight-panel row | group map rule | `StackReduction::group_maps` | focused test packs the maps and matches the claim sum |
| The inner sends the unpacked row values with the rest values | anchor send rule | `GroupInner` | focused test derives the hidden group from the received ends |
| The terminal second side is the packed B side at the point | terminal rule | `group_bases` from the received ends | focused test compares it with the packed B side's multilinear value |
| A changed last round or weight value is refused | negatives | prove/verify checks | focused test |

## Conclusions

- The whole group is provable and verifiable at the chain's entry, on the same
  convention as the group ties.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The chain entry's switch to the group, the merge's slot references and the
  session entry follow.
