# Hidden output object bound to its position

## Scheme and baseline

- Scheme: `norm-hd-object`.
- Code: `43dc5e75`; previous commit `2cb03944`.
- This is partial progress on the norm output tail: the output object the next
  layer consumes is bound to the same position's committed values and the same
  operation output, through the real hidden chain.

## What changed

- `NormOutputPlan::hd_group` exposes the out group as the hidden output object;
  it is the same committed group the tail relation rows address, not a copy and
  not a second root.
- The hidden fixture checks that every committed out cell equals the operation
  output of its position and column, and that the first-stage out entries carry
  the plan's oracles and shapes.
- Negatives on the real hidden path:
  - another position's committed output object in this position (the prover's
    own fold and base check reject it);
  - a changed out root or out sample answer at the verifier;
  - an output shift that differs from the committed tail rows.

## Results

- `cargo test --workspace`: 48 suites, 932 passed, 0 failed, 17 ignored, exit 0.
- Focused: hidden_norm 37, norm_output 5, rstd 16, lookup 31, norm_param_table 4,
  norm_tail_slack 6, norm_tail_clamp 5, norm_tail_padding 4, norm_output_hd 4
  passed (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: the change is a module and fixture block on the hidden
  verifier path and does not touch seq512; the norm-free byte gate is the
  required byte gate for this step. No end-to-end seq512 case was run because no
  model-level path changed.

## Positive and negative coverage

- The honest chain proves and verifies with the output object exposed; the
  committed out cells equal the operation outputs per position and column.
- The cross-position output substitution, the changed root and sample, and the
  shift mismatch are rejected in the real hidden path. The valid partition
  negative from the padding trace covers the same-position valid binding; the
  parameter-claim tests cover the bias root and opening binding.
- The same-position mean, rstd and gain inputs are bound by the per-position
  relation rows and their existing negatives; this step does not add a new input
  commitment.
- No soundness claim and no convergence with the next layer's consumer is
  claimed here; the consumer must read this same object when it is built.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| The out group is the hidden output object | explicit accessor; the plan digest already binds the group | positive chain; root tamper | local tail section |
| The consumer reads the same value, not two roots | committed out cells equal the operation output; substitution negative | cell equality; cross-position negative | next-layer consumer not built |
| Same position x/mean/rstd/weight/bias/valid/shift/clamp | per-position relation and padding rows; shift and valid negatives | shift negative; padding partition negative | existing input commitments |
| Every declaration enters the chain | the out group joins the same BAT/FOLD/CS/BASE path | positive chain; negatives | local tail section |

## Scope and next

- No claim that the complete norm or exact rounding is proven; no production
  source closure for the gain and bias tables.
- Open: the next-layer consumer convergence and the model-level assembly.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
