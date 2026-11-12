# Claim masks and the halving line

## Scheme and baseline

- Scheme: `psc-claim-masks-line`.
- Code: `5f20e9f1`; previous commit `228ac1c7`.
- This is partial progress on the shared lookup foundation. It adds the claim
  mask form to the general sumcheck and the line reduction that combines two
  half-map claims. It does not add a lookup message, commitment or transcript
  order.

## Changes

- `MaskTerm` and `GeneralClaim` state a point claim with a linear form in mask
  entries.
- `prove_general` accepts claim mask terms and a mask-entry provider; the
  prover adds the claim's mask form to every round. `verify_general` accepts the
  same terms for the output row.
- `GeneralOutput.claim_masks` carries the claim terms scaled by the output
  claim scale.
- `line` draws the last coordinate, appends it to the point and combines the
  two half claims and their mask terms with the new coordinate's weights.
- Unit tests cover a claim mask term through the prover/verifier and the line
  combination.

## Results

- `cargo test --workspace`: 48 suites, 838 passed, 0 failed, 17 ignored.
- Focused `psc_general`: 8 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.

## Scope and next

- No lookup table, key or multiplicity message, mask slot allocation or
  challenge order is wired. The lookup driver builds on these claims next.
- No round-half-up, soundness or complete-norm claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
