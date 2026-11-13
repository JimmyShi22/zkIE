# Tail-keyed clamp slack range lookup

## Scheme and baseline

- Scheme: `norm-tail-slack-lookup`.
- Code: `515732a1`; previous commit `71f2b279`.
- This is partial progress on the norm output tail. The byte-table lookup now
  reads its keys from the committed clamp slack digits of the tail group; the
  multiplicity bits, commitment messages, samples, openings and key-claim rows
  run through the real hidden chain.

## What changed

- `NormLookupPlan::from_tail_slack` sizes the lookup from the strict and
  non-strict sign-consistency slack widths, requires the public byte table
  identity, and binds the tail group as the key group (first oracle, message
  shape, column count).
- The norm plan validates that a non-empty key map addresses the rounding group
  or the output tail group; the plan digest binds the key group and the key map.
- The tail witness now commits digit bytes as unsigned field elements: the
  public byte table holds `0..255`, and the reconstruction rows multiply the
  committed digit rows by positive radix-256 constants.
- The hidden fixture attaches the lookup to the output-tail fixture, counts the
  multiplicity bits from the committed tail messages, adds the lookup mask rows
  through the same own-mask derivation, and proves and verifies through the real
  chain. The norm plan version rises to 13.

## Results

- `cargo test --workspace`: 48 suites, 919 passed, 0 failed, 17 ignored, exit 0.
- Focused: hidden_norm 37, norm_output 1, rstd 16, lookup 31, norm_param_table 4,
  norm_tail_slack 6 passed (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: the change is a module and fixture block on the hidden
  verifier path; it does not touch the seq512 path. The norm-free byte gate is
  the required byte gate for this step; no end-to-end seq512 case was run
  because no model-level path changed.

## Positive and negative coverage

- An honest output tail with the attached lookup proves and verifies through
  the real hidden chain; the key group equals the tail group, the multiplicity
  bits come from the committed tail messages, and the lookup claims appear in
  the second-stage message and flow through the same BAT/FOLD/CS/BASE steps.
- Rejected at the plan gates: a non-byte table identity, equal bit and mask
  oracles, a zero group, a missing clamp section, and a key group that is
  neither the rounding group nor the tail group.
- Rejected in the real hidden path: a committed slack digit changed in the tail
  messages, a slack value that does not fit its byte width, a committed slack
  cell replaced under a consistent tree while the honest key claim is proven,
  and tampered or missing lookup claims and messages at the verifier.
- These are structural and relation negatives for the local tail section; no
  soundness claim for the complete norm is made.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| The key is the committed slack cell, read from the committed message | key map addresses the tail slack cells; the key group is the tail group | positive chain test; key-group plan gates | the tail-keyed instance only |
| Table identity is fixed before the dependent challenges | the byte table identity is a plan requirement; the plan digest binds the key group | plan gates; lookup transcript tests | public byte table only |
| Claims enter the same BAT/FOLD/CS/BASE | key-claim rows are built from the key group; the full hidden chain runs | positive chain test; claim and message tampering | local tail section |
| One budget derivation | the lookup mask rows enter the existing own-mask derivation | the fixture uses the same derivation; workspace pass | no new budget source |
| Digit bytes match the public table | the tail witness commits unsigned byte digits | positive lookup; existing reconstruction tests | changes the tail witness encoding |

## Scope and next

- No claim that the complete norm or exact rounding is proven; no production
  source closure for the gain and bias tables.
- Open: the clamp five-point acceptance and its negative matrix, the padding
  negatives, the hidden-output binding, and the model-level assembly.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
