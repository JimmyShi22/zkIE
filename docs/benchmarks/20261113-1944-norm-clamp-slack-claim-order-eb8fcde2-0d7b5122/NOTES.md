# Clamp slack digits in the byte table and the masked claim transcript order

## Scheme and baseline

- Scheme: `norm-clamp-slack-claim-order`.
- Code: `eb8fcde2`; previous commit `0d7b5122`.
- This delivery merges the two rework items: the four clamp slack digit
  groups enter the byte table key map (with the selector attack and slack
  negatives), and the masked pointwise claim is absorbed before the first
  sumcheck challenge on both sides.

## What changed

- `tail_digit_groups` now lists all twelve tail digit groups: quotient,
  remainder, doubled remainder, tie slack, shifted value, pre-clamp value,
  the two clamp distance rows and the four clamp slack rows (low strict, low
  non strict, high strict, high non strict). The half selector and the two
  clamp selectors stay outside the lookup; the sign consistency relations
  force them to zero or one once every slack word is a byte.
- The key count, the multiplicity plane count and the committed message
  sizes are derived from the same enumeration, and the canonical key map
  check in the plan validation recomputes it. Sizing for the fixture is
  unchanged by the four groups (the key bit count and plane count stay the
  same, only the key sources grow).
- The pointwise general sumchecks absorb the masked claim into the chain
  before the first challenge, per instance, on both the prover and the
  verifier side. The challenge source carries a claim tag; the absorption
  happens before the hiding sum is absorbed and before the first challenge
  that multiplies the target. The mask commitment, the original column
  point and the later BAT and unit rows are unchanged.

## Results

- `cargo test --workspace`: 48 suites, 956 passed, 0 failed, 17 ignored,
  exit 0 (`workspace.txt`).
- Focused: `hidden_norm_psc` 27, `hidden_norm` 9, `lookup` 14,
  `committed_psc` 8, merged norm chain 56 passed (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: the change is at fixture scale on the hidden verifier
  chain and does not touch any model-level path; the norm-free byte gate is
  the required byte gate for this round. No seq512 case was run because no
  model-level path changed.

## Positive and negative coverage

- Positive: the integrated lookup declares two key groups, at least one
  source addresses each, and the tail sources address every cell of the
  twelve groups (positions × columns × digit width). The clamp boundary
  matrix, including the collapsed interval, proves and verifies.
- Slack negatives on the real path: a flipped low selector and a flipped
  high selector, each with the sign relation satisfied by a field element
  slack cell and the committed output shifted; the same attack on a padding
  column; an out-of-byte digit in each of the four slack groups. Every case
  is rejected by the prover, or by the verifier when the prover accepts.
- Lookup-level negative: a tail message with a non-byte slack cell is
  resolved into the key vector and run through the lookup prover and
  verifier; the non-byte key cannot pass the byte table.
- Claim order: a proof produced with the claim absorption verifies and
  reproduces the prover's output; a replay without the absorption draws
  different challenges and derives a different output claim set, which the
  committed claim binding rejects. A changed claim is still rejected.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| Every clamp slack digit is byte-range proven | four slack groups in `tail_digit_groups`; canonical key map check recomputes the enumeration | tail source count assertion; slack digit out-of-byte negatives; non-byte slack key lookup negative | selectors excluded; honest slack values are non-negative and fit the span check |
| The selector attack is rejected | flipped selector with alpha/beta field cells and shifted output | low and high attack negatives, padding attack negative | rejected at the real prover entry; a bypassing prover is countered by the lookup |
| The masked claim is fixed before the first challenge | claim tag and `send_claim`; both sides absorb per instance before the general sumcheck | order test (old form derives a different output set); changed claim negative | `verify_general` transforms claims; the committed binding rejects a mismatched set |
| Mask commitment, original point, BAT and unit rows stay | untouched code paths | existing claim, mask and point negatives | unchanged |
| Hard gate and workspace | no change to the norm-free toy path | 45,080 bytes identical; 48 suites green | — |

## Scope and next

- The full-chain selector attack and the old-order proof form cannot be
  produced through the honest prover entry: the prover checks the committed
  tail message against the exact row encoding and always absorbs the claim
  before the first challenge. The attack is rejected at that entry; a
  prover that bypasses it is countered by the byte lookup, exercised
  directly at the lookup prover and verifier level.
- The fixture premise, the A references, the gain and bias source closure,
  the rstd-to-rsqrt linkage, the complete norm relation and the single-shot
  whole chain remain open, unchanged boundaries.
- Open: the model-level entry convergence, the real-path ledger budget and
  the expert's remaining review items.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
