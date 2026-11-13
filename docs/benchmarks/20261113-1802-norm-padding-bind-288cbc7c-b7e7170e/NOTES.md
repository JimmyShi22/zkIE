# Padding bindings on the hidden chain

## Scheme and baseline

- Scheme: `norm-padding-bind`.
- Code: `288cbc7c`; previous commit `b7e7170e`.
- This is partial progress on the norm output tail: the padding rule of the
  non-valid columns is exercised through the real hidden chain with the clamp
  relations and the slack digit lookup attached.

## What changed

- New negatives on the real hidden path:
  - a changed output on a padded column;
  - a changed pre-clamp value on a padded column (the padding row reads the
    unit-point bias, so the relation no longer closes);
  - a valid/padding partition that moves a padded column into the valid set;
  - a bias claim that points at the gain table row (the plan rejects the kind
    mismatch).
- The honest chain still proves and verifies with the padding rows active: the
  fixture has two valid and two padded columns per position, and every boundary
  and lookup test runs with those rows.

## Results

- `cargo test --workspace`: 48 suites, 928 passed, 0 failed, 17 ignored, exit 0.
- Focused: hidden_norm 37, norm_output 1, rstd 16, lookup 31, norm_param_table 4,
  norm_tail_slack 6, norm_tail_clamp 5, norm_tail_padding 4 passed
  (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: the change is a module and fixture block on the hidden
  verifier path and does not touch seq512; the norm-free byte gate is the
  required byte gate for this step. No end-to-end seq512 case was run because no
  model-level path changed.

## Positive and negative coverage

- The padding row enumerates exactly the non-valid columns from the trusted
  plan and reads the committed bias message at the unit point of each column.
  A positive chain with the padding rows active proves and verifies.
- The partition, output, pre-clamp and claim-swap negatives are rejected in the
  real hidden path. The bias opening, opening symbol, missing bias section and
  table root negatives are covered by the existing parameter-claim tests; the
  changed bias value is the pre-clamp negative above.
- The plaintext boundary is the fixed-width message and claim encoding already
  checked by the parameter-claim tests; no bias plaintext vector enters the
  proof.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| A padded column's output is clamp(bias) | padding rows read the committed G_u at the unit point; output tamper negative | positive chain; output negative | local tail section |
| The same physical G_u serves both reads | the bias claim is the only bias source; claim-swap plan gate | plan gate; parameter-claim tests | fixture table identity |
| One trusted valid partition covers every column | the rows enumerate the non-valid columns from the plan | partition negative | trusted plan |
| Every declaration enters the chain | the rows join the same BAT/FOLD/CS/BASE path | positive chain; negatives | local tail section |

## Scope and next

- No claim that the complete norm or exact rounding is proven; no production
  source closure for the gain and bias tables.
- Open: the hidden-output object binding and the model-level assembly.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
