# range-limbs

- Scheme: range-limbs (balanced byte limbs and one range object per limb).
- Based on: the range activation source (code 4b9a2ac4, trace dba596a0).
- Purpose: the original computation matrix's cells are the tensor's balanced
  base-256 limb bytes, each limb is a cell matrix of the smallness check, the
  reduced read table does the same functional per limb, and the digit object
  covers every limb at its offset.
- Change:
  - `packed_matrix::balanced_limbs` / `balanced_limb_count`: the reference's
    limb split (`b = ((v + 128) mod 256) - 128`, `v = (v - b) / 256`), so the
    integer is `sum_l 256^l b_l`; a value outside the declared count is
    refused, never reduced. Unit test over both signs and the boundaries.
  - `RangePlan` carries one `RangeObject` per limb (its packed symbol array's
    oracle, shape and matrix geometry), one reduced read table message and one
    digit object whose planes are the summed cell rows; each limb's rows start
    at its offset. `fault()` checks one geometry, distinct oracle and tree
    identities across every object, the digit coverage and the message
    capacities. The plan documents the supported combination counts (powers of
    two in 2..=256) and the draw order (carrier before the digit object, the
    batching element after its sample answers).
  - The value functions build one packed array per limb and one flat digit
    object at the limbs' offsets; the member carries one read table per limb
    and the reduced entries are the per-limb functionals in table order.
  - The segment commits and samples every array message before the section and
    mirrors it on the verifier; the wire form carries one array message per
    limb and refuses another count, a missing message, a changed root or a
    changed sample answer.
  - The tie and read rows: one tie per limb matrix at
    `first = mu * Np * 2^ceil_log2(groups)`, the reads follow all ties from
    `eqFirst[NM]`; the coefficients keep `theta^(claims + first)` and the
    per-row `vt`/`rowX` weights.
  - New tests: the two-limb objects recompose the signed cells (arrays, digits
    at the offsets, the reduced table's `256^l` recomposition into the
    functional's integer, all tie and read rows evaluated on the committed
    messages, a changed second-limb digit breaking a row, a value outside the
    declared limbs refused, a table count that disagrees with the limbs
    refused).
- Key numbers: workspace 1106 passed / 0 failed (workspace.log); hard
  gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the limb mechanism and the per-limb range objects are carried by
  real SEND messages with the ruled order and coefficients; the two-limb
  relation algebra holds on the committed messages and the U3 entries recompose
  by `256^l`.
- Boundaries: BZ and the digit lookup are absent, so no smallness or range
  closure is claimed and the tie/read equations remain equality relations
  only; the consumer-side `256^l` recombination into the panel claim is outside
  this round; the real activation tensor's limb count is not yet wired into the
  diagnostic plan, and the digit object's capacity with the group code
  (`tau * (ell - answers)` entries) bounds the supported limb count; the real
  entry has not been run. Not pushed.
