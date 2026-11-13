# Output tail equation bounds below the no-wrap half modulus

- Scheme: `norm-output-tail-nowrap`
- Change: code commit `e006dfaf` (worst-case magnitudes of the image,
  reconstruction, shifted, bias addition and clamp comparison equations with a
  half-modulus verdict).
- Baseline: previous commit `a191a1ec` (trace of the tail slot commitments
  `dca1b628`).
- Purpose: establish the no-wrap side of the tail equations before the
  relations are wired; if any equation could reach half the base modulus the
  segment must stop and report.

## What changed

- `NormTailBounds` and `output_tail_bounds`: the maximum of `|image|`,
  `|q| * 2^shift + |r|`, `|q| + selector`, `|shifted| + max |bias|` and
  `|pre_clamp| + max(|low|, |high|)` over every position and column, with
  overflow checked.
- `output_tail_no_wrap_fault`: rejects any bound that reaches half the base
  modulus.

## Evidence

- Fixture values (weight 2/-1/3/1, bias 1/2/-1/2, rstd 100, shift 3, clamp
  -20..20) give: image 1200, reconstruction 1200, shifted 150, bias addition
  152, clamp comparison 169 — all far below half the modulus
  (`0xFFFF_FFFF_0000_0001 / 2`).
- The test also rejects a bound at exactly half the modulus and structurally
  oversized reconstruction and bias bounds.
- Focused: hidden_norm 35, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 911 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| no-wrap with the real fixture bounds | per-equation worst-case magnitudes | bounds test | the relation wiring reuses the same numbers |
| every equation below p/2 | half-modulus verdict | over-bound rejections | states the fixture domain, not the model |
| overflow safety | checked arithmetic in the bounds | checked paths | constants only |

## Key numbers

- Fixture bounds: image 1200, reconstruction 1200, shifted 150, bias sum 152,
  clamp sum 169; half modulus 9,223,372,030,322,688,000.
- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 911 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The rounding, bias, clamp and padding relations over the committed slots
  (digit reconstruction, half selector, slack rows); the rows must reuse the
  bounds above.
- The hd binding; the ledger categories at the zkie path; production source
  closure.
- No push in this round.
