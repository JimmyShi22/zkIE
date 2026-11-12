# Norm product-sumcheck endpoints and row binding

## Scheme and baseline

- Scheme: `norm-psc-row-binding`.
- Code: `a9fbfc01`; previous commit `66c60353`.
- This is partial progress. It wires the sum and squares norm relations into
  the existing output-head hidden transcript and binds their product-sumcheck
  endpoints to committed member rows. It does not bind the mean and variance
  relations, the carry/range digits, the reciprocal-square-root lookup, the
  weight product, the affine/activation tail, or the output-head consumer.

## Changes

- `NormPlan` fixes the active relation domain. Only the relations whose row
  binding exists can be listed; the integrated domain in this round is the sum
  and squares relations.
- `NormMaskPlan` allocates per-instance hiding and shown entries, reserves the
  claim slots, and validates the mask row capacity, code and runtime tree.
- `CommittedNormMask`, `NormMaskWitness`, `NormGroupWitness` and `NormWitness`
  carry the prover's committed rows, tails, salts, pads, integer rows, exact
  integer core and the mask entries.
- The first stage fixes the norm group roots and the mask root, draws the norm
  seeds and answers, draws the opening positions, verifies every opening, and
  absorbs the norm material into the existing first chain before the hidden W
  seeds.
- The second stage runs one product-sumcheck instance per position and active
  relation after the closure instance, with the committed mask entries. The
  verifier rebuilds the same challenges and checks the shown point values.
- `NormRowBinding` and `NormPositionBinding` carry the checked endpoint claims.
  `norm_plain_rows` maps each sum endpoint to the committed x and sum messages
  and each squares endpoint to the committed x, mean and squares messages; the
  right sum endpoint is a public constant plus its committed shown mask. The
  `(U)` row binds the hiding coefficients to the committed mask entries. The
  claim target is zero.
- The norm plan digest enters the hidden parameters digest; the norm messages
  enter the existing first and second transcripts. The BAT/FOLD/CS/BASE stage
  sequence and types are unchanged.
- The norm mask rows join the existing fresh mask row budget.

## Results

- `cargo test --workspace`: 48 suites, 827 passed, 0 failed, 17 ignored.
- Focused `norm_`: 12 passed, 0 failed.
- Norm-free byte gate: the toy output-head proof with the norm absent is
  45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.

## Positive coverage

- A hidden toy fixture with the norm present runs through
  `prove_hidden_lin_hidden_with_norm` and the public
  `verify_hidden_lin_hidden_with_witness` entry. Both sum and squares
  endpoints are accepted only with the committed x, sum, squares and mean
  messages, the committed mask entries and the existing BAT/FOLD/BASE path.

## Negative coverage

- The verifier rejects a changed shown point value, a zero mask root, a changed
  sum sample answer, a changed product-sumcheck end value, and a plan whose
  active relation set differs from the proof.
- An inconsistent committed sum witness is rejected by the prover before any
  proof is emitted; this is recorded as a prover-side rejection, not as a
  verifier-path negative.

## Explicitly deferred

- Mean and variance endpoint binding; their relation indices are rejected from
  the active domain until their maps and rows exist.
- Carry/range digits, the reciprocal-square-root lookup, the weight product,
  shift, bias, clamp and padding, and the output-head consumer.
- A fully adversarial prover that assembles a transcript with mismatched
  committed maps; the current negatives cover message tampering after a
  serialized round trip, not a hand-built adversarial prover.
- No soundness, privacy, round-half-up, complete-norm or batch-completion
  claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
