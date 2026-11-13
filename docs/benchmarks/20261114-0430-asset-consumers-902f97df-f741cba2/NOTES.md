# Asset consumer inventory and the consumer-closure gap

## Scheme and baseline

- Scheme: `asset-consumers` (B2 pre-work).
- Code: `902f97df` (B1 trace); previous commit `f741cba2` (B1 code); branch
  `refactor/v2`. Read-only analysis: no prover, verifier, key material or code
  change.

## Inventory

An external read-only tool walks the production `TracePlan` of the three
models and, for every step input, collects `PlanSource::Asset` paths
recursively (through layouts and concatenations). The full per-asset list is
in `consumers.tsv`; the tool and its build/run log are `tool-main.rs`,
`tool-Cargo.toml` and `build-run.txt`.

- 683 asset paths with at least one plan consumer; 13,873 consumer entries;
  every asset in the three models' trusted asset specs has at least one plan
  consumer (0 unused).
- Consumer entries by operator class (entries / distinct assets):
  Projection 1,754 / 284; Add 1,347 / 163; ProfileRmsNorm 1,018 / 131;
  ProfileLayerNorm 210 / 92; ProfileRope 360 / 4; Hadamard 320 / 1;
  Softmax 8,576 / 2 (the exp table); SoftmaxRows 216 / 2;
  TableLookup 66 / 5; MatMul 4 / 2 (the lm_head weight); ProfileStandardize
  2 / 2.
- Current verification destination of these consumers: the asset is opened in
  full at the consumer port and the operator relation recomputes the operator
  from the full rows (`AssetView::materialize_full` and
  `verify_op_relation_with_external`); the softmax exp table and the lookup
  tables are read per index in the relation. There is no LIN weight claim for
  any of these consumers today.
- The inventory counts plan-level asset sources. At run time the shared trace
  opens the same assets through the `sources`/`external` ports; the consumer
  positions are the ones listed.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Per-asset consumer list | production `TracePlan` inputs (`PlanSource::Asset`) | read-only inventory tool | `consumers.tsv`: 683 assets, 13,873 entries, 0 unused |
| Current verification destination per consumer | `plan::AssetView::materialize_full`, `relations::verify_op_relation_with_external` | class mapping above | code paths and counts |
| Operator weight declaration into LIN | reference `vf_layers_check.inc` matrix panel, `wsPending` → `member_pending_row` → `lin_first_rows`/prove/verify | **not delivered** | gap below |
| Negative matrix for the closure | — | **not delivered** | gap below |

## Gap: the B2 wiring is not delivered

The reference consumer closure requires a sub-proof weight claim to become a
`wsPending`, then `member_pending_row`, then `lin_first_rows`/`lin_first_prove`
/`lin_first_verify_on_shape`, with the shared chain and the second chain
closing the operator's actual input, output and fixed-point arithmetic against
the same matrix, while the original integer range, rounding, bias, edge and
output checks stay verified.

The current tree has `lin_session.rs` (LIN request messages, authenticated LIN
table, control checks) and the specialized `output_head_lin.rs` pipeline. It
does **not** have a general weights-pending claim, a `member_pending_row`
equivalent, or fork/sub-proof weight-claim plumbing; the only pending-row path
is the specialized one in the output-head key code.

Building the general binder is a structural addition (proof relation and
verification conditions; commitment object and binding). Per the seven-class
rule I am stopping here for the expert's structural ruling instead of
inventing a binder or a second LIN path. No operator declaration, no
consumer-mode removal and no negative matrix is claimed in this trace.

For the ruling, the proposed B2b order is:

1. Port the weights-pending claim (`wsPending`) and `member_pending_row` over
   the B1 encoded table identity (table record and root already enter the
   anchor);
2. Add LIN first/shared/second requests one operator class at a time
   (Projection first, then MatMul/RegressionHead, then Add and the norm/bias
   parameters), with coefficient, target and mask sources fixed before LIN;
3. Wire the prover trace and the verifier, keeping the existing relation
   checks until each consumer closure is verified; remove plaintext per asset
   only afterwards;
4. Embedding psi coefficients through the same weight mechanism and the F.lin
   binding to the actual activation; private lookup parameters keep their
   weight read and consumer relation;
5. Negative matrix per class and the five-case acceptance.

## Boundaries

- This trace is the consumer inventory and the gap statement only. No
  operator wiring, no negatives, no plaintext removal and no proof-byte claim.
- No code changed; the 45,080 B gate is untouched.
- No push.

## Artifacts

`consumers.tsv`, `tool-main.rs`, `tool-Cargo.toml`, `build-run.txt`,
`NOTES.md`.
