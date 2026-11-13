# Output tail groups committed in the norm first message

- Scheme: `norm-output-commit`
- Change: code commit `1cefcd15` (the four output tail groups committed in the
  norm first-stage message; roots, sample rounds and column openings drawn
  after the parameter claims; prover checks the committed messages against
  the exact tail rows and the verifier redraws the same transcript).
- Baseline: previous commit `74c99ac8` (trace of the exact output tail
  witness `0a09f845`).
- Purpose: fix the broadcast, product, image and out commitments in the
  first stage so the pointwise product instances and the output relations can
  bind their claims to committed messages.

## What changed

- `NormOutputMessage`: the four committed output groups; plan-gated encoding,
  decoding and a declaration check (group id, shape, position, oracle, tree,
  non-zero root).
- `NormFirstMessages.output`: the field after the parameter message, omitted
  entirely when the plan has no output section; the bind version moves to 9.
- `NormOutputWitness`: the exact output rows plus the four committed group
  witnesses.
- Prover: validates every committed output group and every message value
  against the tail rows, absorbs the roots, draws the sample seeds and
  answers, then draws the column positions and opens them.
- Verifier: checks the message declarations, absorbs the same roots, redraws
  the seeds and answers, redraws the positions and verifies the openings.
- The four groups use their own root, seed, answer and position tags.

## Evidence

- `norm_output_groups_prove_and_verify`: a full two-stage chain with the
  output section passes prove and verify; tampering the product root, an
  image sample answer, an out opening position, removing the output message
  and changing the out group number in the plan are all rejected through the
  real verifier.
- Focused: hidden_norm 34, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 909 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| output rows are committed objects | four committed groups with roots, samples and openings | prove/verify test | the pointwise products are not yet proven over them |
| commitment fixed before dependent challenges | output roots absorbed after the parameter claims in the first stage | transcript order in both sides | the product challenges are next |
| openings authenticated | verifier redraws and verifies every opening | position/root negatives | the ledger categories stay at the zkie path |
| non-valid columns | committed rows carry image zero and the clamped bias output | witness check in the prover | no relation yet |
| hd object | the out group is the hd commitment | group declaration check | MatMul convergence in 2c |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 79 tests across the five filters.
- Workspace: 909 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The two pointwise product instances with equality weights and their
  claim-to-message rows; then the rounding, bias, clamp and padding
  relations; then the hd binding to the same-position parameters.
- The no-wrap real-bound re-derivation for the product, rounding and bias
  equations.
- The output group ledger categories at the zkie path; production source
  closure.
- No push in this round.
