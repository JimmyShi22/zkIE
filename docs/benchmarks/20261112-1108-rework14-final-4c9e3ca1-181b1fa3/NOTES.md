# rework14 final: L1 affine corrections, L2 wording and the multi-stage shape blocker

## Scheme and baseline

- Scheme: `rework14-final`.
- Code: document commit `4c9e3ca1` (v12); previous round head `181b1fa3`.
- This round changes only `docs/lin-bind-closure-security.md` (v11 -> v12).
  No production code, test path, plan, parameters, commitments, challenge
  order or proof format change was made. The multi-stage IOPP implementation
  is blocked on a plan-shape condition (below), not silently skipped.
- Fixture for the reused performance numbers: `/data/lin-fixture`, formal
  hidden session, GPT-2 output head, 16 positions.

## Purpose

Record the M1 L1 corrections (affine constraint space, precise conditional
lemma, same-source conditioning, hidden definition, root programming history),
the M3 L2 wording corrections, and the M2 structural blocker for the
reference-style multi-stage IOPP.

## Findings

- M1: the B6 star map is not surjective onto the whole public space; the
  public values satisfy the affine constraint
  `a*main' + b*mask' - target* = lambda*target`, and the distribution is
  uniform on that affine image with equal fibers. The conditional block lemma
  is stated in affine form (`P_b = A X_b + c`, uniform on the image, fiber
  `|ker A|`), not "surjective implies uniform". Same-source commitment/opening
  conditioning and the root-programming split (public setup roots vs prover
  commitments) are written. The hidden property is stated as
  `view ⊥ witness | statement+public inputs`; a simulation-based witness-free
  simulator is not achieved and is marked as the missing piece.
- M3: `0.3164`/155 is a single conditional event (two independent index draws
  missing a word at relative distance `delta`), not a proximity-gap bound, not
  a complete-proof acceptance probability and not an overall security bound;
  the finding is a major security-bound gap/risk, not a demonstrated attack.
  The candidate direction is corrected: fixing `n`/`ell` and increasing `r`
  raises the rate and lowers the distance; lowering the rate needs larger `n`
  or smaller `ell + r`.
- M2 blocker: the reference-style chain requires power-of-two message lengths,
  `l = 2^k' l'` and a per-stage `k/h`/mask-room list. The actual plan's main
  code is `ell = 65,539 = 65,536 + 3` (not a power of two) and
  `HiddenProximityPlan` has one code/mask code/fold shape and no stage list.
  The reference shape check rejects the plan at the first stage. Candidate
  mechanisms: change the main `ell` to a power of two and move the three pads
  out of the code message; add separate power-of-two IOPP stage codes with a
  CS bridge; or obtain the reference-compatible stage list/codes from the
  trusted planner. This is a plan/parameter change (seven-category), reported
  to the TL/expert; no parameter was chosen here.

## Workspace

`cargo test --workspace`: 48 test binaries, 751 passed, 0 failed, 17 ignored.
The complete log is `workspace-test.txt`.

## Reused seq16x3 numbers

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 | accepts | rejects |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 359,802 | 50,538 | 646.80 | 169,394,600 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 2 | 358,777 | 50,224 | 644.51 | 169,315,672 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 3 | 358,806 | 49,829 | 644.82 | 169,404,380 | f8f512c0…3215f5 | 276,343,251 | 0 |

These are the rework10 measurements, reused with provenance. They were not
re-measured in this round because the protocol was not changed. If the M2
plan change is approved and implemented, seq16x3 must be rerun and the 5-case
decision is for the TL to make and record in the new trace.

## Open items

- `GAP`: the direct FOLD-to-BASE proximity is ineffective at the current
  parameters (main miss 0.3164234164; mask miss 3.424292663e-5); a single
  conditional event, not a demonstrated attack or an overall bound.
- `GAP`: the reference-style multi-stage IOPP is not implemented; the plan's
  non-power-of-two `ell` and missing stage list block it. Candidate plan
  changes are in the report; the choice is a seven-category decision.
- `UNPROVEN`: kernel coverage, a witness-free simulator, concrete-hash RO/CR,
  deployment randomness, exact final-verifier negative.
- No push: origin remains `831e7134`.

## Ledger

- Tests passed: 751 workspace tests, 0 failed, 17 ignored.
- Proof bytes, wall time and VmHWM: not re-measured; reused rework10 figures
  with provenance above.
