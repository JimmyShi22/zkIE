# range-activation

- Scheme: range-activation (the request segment's original computation matrix
  is taken from the activation tensor).
- Based on: the range objects and their relations (code 766de2d9).
- Purpose: the request segment's range objects were built from the weight
  table read source, which is the weight-class object with its own commitment
  and consumption path. This round binds the original object's supply, layout
  and owner back to the computation matrix the functional reads: the committed
  activation tensor in the functional's read orientation.
- Change:
  - `RangePlan.matrix` is the activation in the read orientation (cell rows are
    the hidden features, cell columns the positions); the session binds it to
    the plan's activation shape.
  - The symbol array, the digit object and the reduced read tables derive their
    cells from the committed activation, never from the weight row source. The
    cells are balanced base-256 limbs (`b = ((v + 128) mod 256) - 128`,
    `v = (v - b) / 256`); each read table reduces one limb and a value that does
    not fit the declared limb count is refused, never reduced.
  - `RangePlan.limbs`, one read table per limb; this step carries a single
    limb and refuses another count until the per-limb message layout lands.
  - The member and request validation now tie the read tables to the activation
    geometry; the weight table commitment and its paths are unchanged.
  - The diagnostic entry reduces the activation tensor: its read tables cut the
    hidden dimension into runs, the pending claim stays the weight-side row.
- Key numbers: workspace 1104 passed / 0 failed (workspace.log); hard gate
  45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the range objects' source is the computation matrix the
  functional reads; the object identity error of the previous round is
  corrected. The tie and read algebra is unchanged and still holds on the
  committed messages.
- Boundaries: the multi-limb message layout (one symbol array per limb, the
  summed digit object with per-matrix offsets) is not carried yet; the limb
  count of the real activation tensor has not been derived from its range; BZ
  and the digit lookup are absent, so no smallness/range closure is claimed;
  the consumer-side `256^l` recomposition is outside this round; the real
  entry has not been run. Not pushed.
