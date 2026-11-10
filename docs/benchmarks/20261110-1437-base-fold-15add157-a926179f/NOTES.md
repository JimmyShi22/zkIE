# base-fold - 15add157 / a926179f

Trace of the fixture BASE rework that checks the FOLD output rows against the
original member commitments and the FOLD fresh mask commitment at the drawn
positions. The previous package `t002-s6b3j-r2-review.tar.gz` sha256
`a23a7618994db795dbbee63112d5da3ffcf16d3dbd88ce8c362eb930d0ac1569` is
superseded and retained unchanged.

## Basis and scope

- Based on `15add157`; the previous state is `a926179f`.
- Isolated fixture only; the production verification entry is unchanged.
- The code-switch stage is not connected. This is a conditional local
  experiment, not the reference standard IOPP path. The reference path is
  FOLD -> CS -> final SingleRows -> BASE; skipping CS is not claimed to be
  equivalent to it, and the result cannot be used to claim a complete
  proximity test or a production pass.
- The fixture does not use a zero-fold degenerate input to prove general
  conclusions.

## What changed

- The BASE no longer commits rebuilt old main, old mask or fresh rows. Only the
  two star row commitments and target* are fixed before lambda; M'/R' and the
  fresh rows are fixed before the position draw.
- At the drawn positions the verifier opens every original member commitment
  and the FOLD fresh mask tree, derives each member's group-row placement from
  the bound fold layout, folds the opened columns with the transcript's fold
  challenges, and checks: the combined main row equals the star opening plus
  lambda times the folded original main columns; the combined mask row equals
  the star mask opening plus lambda times the folded original mask columns;
  and the fresh row equals the folded fresh columns.
- The folded-row encoding tails are derived by folding the committed member
  tails with the same challenges; they are not new commitments.
- The fresh coefficient vector is recomputed from the transcript challenges
  and compared with the FOLD output vector.
- A single fold-layout placement helper is shared by BAT and the BASE verifier
  instead of duplicated.

## Positive check, item by item

- No kept rows: the handoff kept count is zero and every layout member is
  unkept; the layout's kept-row count also rechecks to zero.
- Codes and mask: both codes rebuild, the mask fits the main code, the mask
  length covers the hiding polynomial, and the fresh tree plan matches the
  fresh root/tree number, mask row count, mask code depth and leaf class.
- Transcript order: commitments and target* are absorbed before lambda; M'/R'
  and the fresh rows before the position draw. The original and fresh openings
  are checked locally afterwards; the recorded check round equals the rows
  round, so no extra opening round was inserted into the challenge window.
- Folded main columns: every original member column is authenticated against
  its bound root at the drawn positions and folded with the main challenges;
  the combined row matches the star opening plus lambda times that fold.
- Folded mask columns: same with the mask challenges and the star mask row.
- Fresh columns: the fresh mask tree is opened against the FOLD fresh root at
  the drawn positions and folded; the fresh row matches that fold.
- Linear: the extended linear relation on the two star rows, the fresh row and
  the FOLD claim/target holds.
- Not authenticated here: the fold binding is sampled at the drawn positions
  only, not every column; it is not a complete fold binding. The original
  roots and the fresh tree plan are fixture inputs bound through the group
  session, not SEND receipts, asset roots or an independent trust anchor. The
  star rows have no independent trusted source. CS, proximity, hiding and
  production wiring remain unclosed. The result stays pending.

## Negative samples

- Changed folded main tail: rejected at `FoldMain`.
- Changed folded mask tail: rejected at `FoldMask`.
- Changed fresh tail: rejected at `FoldFresh`.
- Replaced fold challenge with recomputed fresh coefficients and a target
  adjusted so the linear check still holds: rejected at `FoldMain`; the linear
  relation passing shows the fold layer, not the linear layer, caught it.
- Replaced FOLD identity, star root, member root, fresh root or layout, and
  replaced a member opening or the fresh opening: fail closed as a stage error.
- Fixed wrong upstream answer (first received answer plus one): carried through
  BAT and FOLD and rejected at `TargetLinear` on the same chain.
- Fixed target* shift: rejected at `TargetLinear`.
- These are fixed local samples, not a claim that every wrong declaration is
  rejected; wrong declarations can cancel through aggregation.

## Tests and measurements

| run | result | wall clock | VmHWM |
| --- | --- | --- | --- |
| BAT-to-FOLD unit | 16 passed | 0.05 s | 29,732 kB |
| BAT-to-FOLD integration | 5 passed | 0.04 s | 30,744 kB |
| Engine package | 234 unit + 57 integration passed | 0.53 s | 202,764 kB |
| Ops package | 89 unit + 23 integration passed | 0.20 s | 188,556 kB |
| Workspace | 487 passed, 0 failed, 11 ignored | 1.07 s | 200,484 kB |

The workspace run used the debug test profile. No proof bytes were produced;
no performance gain, hiding or soundness bound is claimed.

## Fixture values

| item | value |
| --- | --- |
| passing FOLD identity | `2f0cd6024e6fd269d3c80c135e2908f64f9299d5fc8e6f28a7c6c7f9d30b3fbe` |
| passing star main root | `b7161ab94b2a6bc48276b48601019e6d00204776e2ce3845976d015fdb402158` |
| passing star mask root | `8576851c98802e899876799ada67e36a919db161d91e31c08b6554a48aa28a60` |
| passing member root 0 | `499430bf7b5dc5dc0ca84549817ce1f742ff88a024917cdea4d5c139decb1b9b` |
| passing member root 1 | `ab9a1163a095ecb4b1476634d499270f6977dfd9aa9960c680c98c7b72203187` |
| passing member root 2 | `cfd895b99a9594f1342fc9d72b411d29d674a54b464ad008051ada06c90cc44e` |
| passing fresh root | `2b94c1018ee7fb23b5a8e329c595563f10f854fdab04345098b4ffbb2557cd02` |
| passing lambda | `Fp3([430870282149263012, 11647216507932674768, 15081478647295954624])` |
| passing target* | `Fp3([16868277434214652845, 13470965661350279654, 12186804821915069562])` |
| passing positions | `[9, 3]` |
| wrong upstream lambda | `Fp3([14110942288656323744, 17286076374538583336, 12881489594459434530])` |
| wrong upstream target* | `Fp3([7005279915648414391, 11225751876979421554, 348300791375232952])` |
| wrong upstream positions | `[12, 8]` |
| wrong upstream layer | `TargetLinear` |
| target-shift lambda | `Fp3([4181077404533880468, 16786332104266696149, 3456930394359919983])` |
| target-shift positions | `[7, 3]` |
| target-shift layer | `TargetLinear` |

The positive fixture uses tag 9, chain seed `fixture base final`; the wrong
upstream sample uses tag 11 and seed `fixture base wrong upstream`; the target
shift sample uses tag 10 and seed `fixture base target`.

## Unclosed items

- A complete (not sampled) fold binding and a reference-verifiable source for
  the original and star commitments.
- The CS stage and the reference FOLD -> CS -> SingleRows -> BASE path; no
  equivalence claim for skipping CS.
- A complete BASE/IOPP result, complete proximity and a valid FOLD-to-BASE
  handoff.
- Real assets, real SEND/G_u receipts, hiding, folded tails, single-shot
  semantics.
- A deterministic wrong-declaration guarantee; single-equality cancellation
  remains.
- Production wiring and performance acceptance; the production verification
  entry is unchanged.

## Artifacts

- `tests/bat-fold-unit.txt`
- `tests/bat-fold-integration.txt`
- `tests/bat-fold-integration-nocapture.txt`
- `tests/regression-engine.txt`
- `tests/regression-ops.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
