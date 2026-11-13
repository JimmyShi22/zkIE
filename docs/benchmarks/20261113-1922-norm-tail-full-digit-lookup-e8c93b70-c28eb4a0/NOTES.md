# Full tail digit lookup and plan-level output tail bounds

## Scheme and baseline

- Scheme: `norm-tail-full-digit-lookup`.
- Code: `e8c93b70`; previous commit `c28eb4a0`.
- This is the rework round for the batch review: every committed tail digit
  cell is in the byte-table lookup together with the rounding digits, the
  clamp distance rows carry a semantic binding, the no-wrap derivation is
  plan level, and the malicious-witness matrix covers the review list.

## What changed

- The integrated byte-table lookup is one key map with two key groups: the
  committed rounding digit cells and every committed tail digit cell of the
  quotient, remainder, doubled remainder, tie slack, shifted value, pre-clamp
  value and the two clamp distance rows. The half selector and the clamp
  selectors stay outside the lookup because the booleanity and
  sign-consistency relations force them to zero or one. Both groups are bound
  in the lookup plan digest, and the plan validation recomputes the canonical
  key map so a lookup that drops or redirects a digit cell is rejected.
- The clamp distance rows are plain relation rows, one per position and
  column and per side: the committed distance word minus the committed
  pre-clamp word equals the public half ranges minus the clamp bound. The rows
  address the same committed tail cells as the other tail relations.
- `NormPlan::tail_no_wrap` derives every reconstruction and relation magnitude
  from the trusted digit widths, the shift, the rounding input bound, the
  rsqrt bound and the published gain and bias ranges, and rejects a plan whose
  worst case reaches half the base modulus. A plan that carries the output
  tail must now carry the rounding section and the parameter tables. The
  witness-level bound duplicate is removed.
- The malicious-witness matrix adds an out-of-byte digit in each class
  (quotient, remainder, pre-clamp, mean and variance), a tampered tie slack
  digit, a tampered distance cell, a shifted output, a shifted output on the
  collapsed clamp interval, a value one below the low bound, a value one above
  the high bound and a value inside a collapsed interval.
- The tail-only lookup constructor is removed; the integrated constructor is
  the single way to build the key map.

## Results

- `cargo test --workspace`: 48 suites, 951 passed, 0 failed, 17 ignored,
  exit 0 (`workspace.txt`).
- Focused: `hidden_norm_psc` 27, `hidden_norm` 9, `lookup` 14, merged norm
  chain 52 passed (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: the change is at fixture scale on the hidden verifier
  chain and does not touch any model-level path; the norm-free byte gate is
  the required byte gate for this round. No seq512 case was run because no
  model-level path changed.

## Positive and negative coverage

- Positive: the integrated fixture proves and verifies with the mixed lookup;
  the key map declares two groups, at least one source addresses each group,
  and the second-stage lookup claims are non-empty. The clamp boundary matrix
  includes the value at each end, one step outside each end and the collapsed
  interval, and every case proves and verifies.
- Negative on the real hidden path: a committed tail digit that is not a byte
  in the quotient, remainder, pre-clamp, mean or variance class; a tampered
  tie slack digit; a tampered distance cell; a shifted hidden output; a
  shifted hidden output on the collapsed interval; an output that ignores the
  low bound, the high bound or a collapsed interval. Every case is rejected by
  the prover, or by the verifier when the prover accepts.
- Plan gates: a lookup that pops a source, redirects a rounding source or
  redirects a tail source fails validation; a plan whose input bound and gain
  range push the normalized image past the safe magnitude fails validation.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| Every tail digit cell is byte-range proven | integrated key map over q/r/u/w/shifted/pre/low/high cells; canonical key map check in plan validation | two key groups and source coverage asserted; digit-class negatives | selectors excluded by the forcing relations; one lookup with two key groups |
| The integrated chain keeps both digit lookups | the same key map covers the rounding and tail groups; both groups in the plan digest | round digit (mean/variance) and tail digit negatives; key-group assertions | one mixed lookup instance, not two transcripts |
| The clamp distance slot has a semantic binding | one plain BAT row per position, column and side: bound word minus pre word equals half ranges minus the clamp bound | distance cell tamper rejected; clamp boundary matrix proves | rows exist only when the clamp interval exists |
| No-wrap is derived at plan level | `NormPlan::tail_no_wrap` from widths, shift, input bound, rsqrt bound and parameter ranges; plan must carry round and params | plan validation test with an over-large input bound and gain range fails | conservative integer bound, not a formal proof |
| Malicious-witness negatives | out-of-byte digits, tie slack, distance, out shift, collapsed interval, both boundaries | every case rejected on the real path | fixture scale |
| Honest boundaries preserved | fixture premise; A-table references; weight/bias source closure; rstd not tied to the rsqrt chain; no complete norm or single-shot claim | notes and code paths unchanged | not claimed complete |

## Scope and next

- No claim that the complete norm relation or the single-shot whole chain is
  proven; the production source closure for the gain and bias tables and the
  rstd-to-rsqrt linkage remain open dependencies.
- The no-wrap derivation is a conservative bound check, not a formal proof;
  the canonical key map check is plan-level structural validation.
- Open: the model-level entry convergence, the real-path ledger budget and the
  expert's remaining review items.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
