# column-session - 69535418 / 0cfa4bf4

Trace of the isolated column verification component demonstration, reworked
after review.

## Rework

Code commit `69535418` ("Rework the column session onto the FOLDEQ plan"),
based on `0cfa4bf4`. The production verify entry is unchanged.

This rework supersedes the earlier round `ba9e32ee` / `0cfa4bf4`:

- the earlier package `t002-s6b2e-review.tar.gz` with sha256
  `d68ffd012469aa382f462704ee4c6a37c4427d1d913c364d3c3fcd9578ec83ee` is
  superseded and is retained unchanged;
- the session order, plan use, gamma use, and the opening-cost description are
  changed below.

The four corrections:

1. The G_u message is now fixed and sent through `FoldeqSession` before any
   column draw. The session order is first query, seeds, gamma, G_u SEND, shared
   seeds, answer round, record join, column draw, column read/verification.
   The same immutable root reference is passed to the G_u SEND and carried into
   the run result. A changed-root test asserts both the query transcript and
   the query order change.
2. The query path now uses `FoldeqPlan`/`FoldeqSession::draw_queries`; the plan
   fixes the column count, the index domain, the statement/group/tree identity
   and the gamma path. The earlier bare draw count/position call is gone.
   The plan construction attempt succeeded; no interface blocker remains for
   this step.
3. The equation weights are `foldeq_weights(gamma, v)` on the session's real
   gamma, not `[1, 2, 3, 4]`. The G_u record and verified-column rows are built
   by the session with those weights. The separate target control uses the same
   weights against the full re-encoded table.
4. The module and trace now state that the complete asset opening is verified
   row by row and is resident; only the first four rows and the first plain
   column participate in the table. This is an expensive isolation control and
   does not claim the production mmap/ranged or memory path is closed.

## Encoding table

The table is built only after the trusted model anchor and the asset opening
are verified. The full opening is resident; the asset file is not reopened or
materialized whole by this path.

| item | demonstration value |
| --- | --- |
| source asset | `gpt2/weights/lm_head_w_i32.bin`, pool entry 192 |
| slice | first 4 asset rows, first 1 value of each row |
| conversion | signed `i32` little-endian word -> two's-complement `u64` -> base-field element (negative values negated) |
| table rows | `2^v = 4` |
| message length | `ell = 4` |
| code | Reed-Solomon, `log_n = 3` (8 columns), `ell = 4`, `r = 1` |
| plain entries | `ell - 3 * answers = 1` |
| answer pads | 3 zero entries per row |
| tails | 1 zero entry per row |
| salts | 40 bytes per column; byte `0xA5 ^ (position mod 256)` repeated |
| table/chain configuration | existing binding candidate's first-query digest |
| commit primitive | `commit_table` test foundation; it materializes the small table, not the asset |
| record | derived by `commit_table` from the message, pads and deterministic salts |

The full opening is verified by `verify_model_anchor` and
`ModelCommitmentPublic::verify_opening` before the table is built. All opening
rows are checked and resident; only the first four rows and the first plain
column are read into the table.

## Selected object

The real-artifact test uses the GPT-2 sequence-16 final MatMul step `4444`.
The selected-step full-recompute control is run first; it is not replaced by
this demonstration. The table slice is the first four rows and first column of
the already-opened `lm_head_w_i32.bin` asset entry.

## Three roots

The roots are recorded separately and are not claimed equal.

| root | value in the real run | source |
| --- | --- | --- |
| asset meta root | `9a8be4edbbc390fd6844710399441788506c2f1f3a1fc9c9c95e19e067971832` | `AssetMeta.root` from the verified asset commitment |
| table root | `fc74299956624518e337eb3f82cd3b693bed6b635f12535b4e14f7f717364498` | `IndexReader.root`, the column-tree root |
| G_u root | `4747474747474747474747474747474747474747474747474747474747474747` | `FoldeqSession` G_u SEND message, fixed demo root |

## Session order and data flow

| stage | actual call flow |
| --- | --- |
| plan | `FoldeqPlan` holds the committed reader, FOLD shape, `VerifiedStatement`, registered fresh-mask `RuntimeOracles`, read ledger, group/tree ids, gamma path, G_u shape and FOLD layout |
| first query | `FoldeqBinder` builds the first-query bytes; `FoldeqSession::open` checks them; `bind_first_query` binds commitment and record before any challenge |
| gamma | session `draw_gamma` with `GammaPath::Direct { columns: v }`; real run has 2 extension coordinates |
| G_u SEND | session `send_gu(root)` absorbs the G_u root and records the notice in the plan's G_u shape; the same immutable root reference is carried into the run result |
| G_u answers | `shared_seeds` draws the seed set after the root; `receive_gu_answer` absorbs the fixed answer round and builds the sample row |
| record join | `join_gu_record` uses the session gamma and `foldeq_weights`, builds the record rows, joins the trusted group and checks every pending row's oracle/group |
| column query | `draw_queries` steps the positions prefix and calls the plan's `draw_queries`; real run queries `[7, 0, 5, 2]`, unique openings `[0, 2, 5, 7]` |
| column read | one charged opening per unique position is supplied per logical query; `read_queries` verifies each opening against the trusted reader root and the read ledger |
| column rows | `join_gu_read_columns` builds one relation row per logical query from the verified columns with the same real equation weights |
| full comparison | every opened cell is compared with the re-encoded table |
| FOLD | `fold_bind_row` fixes the self-reported fixture; `start_fold` verifies the fresh-mask message and draws epsilon; `fold_rounds_start`; two rounds receive fixed component values and draw the round challenges; `fold_finish` stores the output; `fold_next_stage` fails closed |

The fixed test values are not a production transcript. The table/chain
configuration comes from the existing binding candidate's first-query digest;
the binding's protocol identity and request parameters are demo placeholders.
The G_u runtime-tree starts and the FOLD fresh-mask root are demo placeholders.

## Gamma and equation weights

The real run uses the session gamma (2 coordinates) and the public
`foldeq_weights(gamma, 2)` to produce the four row weights. The recorded
`eq_gamma` is exactly that function's output. G_u record rows and verified
column rows are built by the session with these weights; the separate target
control recomputes each column target from the full re-encoded table with the
same weights. There is no fixed `[1, 2, 3, 4]` weight path.

## Four layers and value sources

| layer | calls | what is actually checked | what is not claimed |
| --- | --- | --- | --- |
| 1. asset and column authentication | trusted anchor, `verify_opening`, `commit_table`, charged table open, `FoldeqSession::read_queries` with `VerifierReads`/read ledger | anchor identity, complete opening identity/row order/width and every Merkle path; each drawn column's salt, symbols and path against the fixed table root; unique physical positions are charged | not proximity; the complete opening is resident, so this is an expensive isolation control, not a production streaming path |
| 2. G_u registration and join | session `send_gu`, `shared_seeds`, `receive_gu_answer`, `join_gu_record`, `join_gu_read_columns`, `GuMembership::check_row` | the SEND notice matches the plan's G_u shape; the seed/answer round is absorbed in order; one dynamic group join; every sample, record and column row's oracle/group matches | the G_u root and sample answer are public/self-reported; `check_row` does not evaluate targets; record and column targets are checked only by the separate full-table control |
| 3. full reconstruction and target control | `encode_row_fp`, `check_full_columns`, `check_merge_targets` | every opened cell equals the recomputed codeword; every G_u column target equals the eq-weighted full-table column value | deterministic full-value comparison and local target control; not proximity or soundness |
| 4. FOLD component | `FoldeqSession` `fold_bind_row`, `start_fold`, `fold_rounds_start`, `receive_fold_round`, `fold_finish`, `fold_next_stage` | public FOLD start/message shape, group/layout/membership, epsilon/round challenge order, final public output, and the fail-closed next stage | FOLD input row is a self-reported fixture; FOLD round values are fixed component values, not a prover session; the next IOPP/CS stage is not connected |

## Negative coverage and rejection sources

Every negative case returns an error or a false layer result; none panics.

| negative | layer and source |
| --- | --- |
| asset opening root changed | trusted anchor/opening check, deterministic |
| column Merkle path extended | `read_queries`/`verify_columns` Merkle path check, deterministic |
| verified opening path with a changed merge value | session `join_gu_columns` accepts the row for membership; separate target control against the full re-encoded table fails |
| G_u row with wrong oracle or wrong group | `GuMembership::check_row` |
| one opened cell changed | full table comparison |
| FOLD input coefficient length changed | `FoldInputRow::check` |
| FOLD next stage | `fold_next_stage` returns the expected fail-closed error |
| changed G_u message root | session query transcript and query order both change; the fixed/used root is the same immutable reference |

The path-valid/merge-wrong case uses a column from an authenticated opening
(the path is valid) and then changes the column value supplied to the merge
relation; membership still passes and the target control fails.

## Tests and measurements

- 9 synthetic unit tests: four-layer success, reproducible fixed fixture,
  changed-G_u query/transcript sensitivity, asset root tamper, column path
  tamper, path-valid/merge-wrong, wrong G_u oracle/group, wrong opened cell,
  and the FOLD component boundary.
- Real gpt2-seq16 test: authenticated full-recompute control 16,822 ms; the
  isolated column session 364 ms; the whole test 72.97 s.
- Timed process: wall clock 1:13.13, maximum resident set size 6,377,040 kB.
- Workspace: 417 passed, 0 failed, 5 ignored.
- No proof bytes were produced by this component demonstration; no proof file
  exists, so no byte count is claimed. No optimization comparison is claimed.

## Validation scope

Only the `gpt2-seq16` case is used. No five-case three-stage run was made. No
claim is made about the full model or performance from this single case. The
production verify entry remains unchanged and is not switched to this demo.

## Unclosed items

- Asset to FOLD witness derivation: there is no public construction path from
  an authenticated asset table to the FOLD input coefficients, mask, target or
  round messages. The FOLD row here is an explicitly labelled self-reported
  fixture; the round messages are fixed component values. The
  `FoldRoundSession` mask coin source is only a private test implementation in
  the current tree and is not used here.
- FOLD next stage: `FoldOutput::next_stage` is deliberately fail-closed; IOPP
  and the completion stage are not connected.
- G_u root: fixed demo message value, not derived from the asset or table root.
- G_u sample answer: fixed self-reported value; no G_u oracle table is
  connected.
- G_u targets: membership does not evaluate them; the record target comes from
  the reader record, and the column target control is local full-table
  comparison only.
- G_u generator/group/layout and the fresh-mask tree: trusted test fixtures;
  production statement/group source and mask opening/authentication are not
  connected.
- Table commitment: `commit_table` is the test foundation, not the production
  commit path; the complete opening is resident and only a slice builds the
  table.
- Binding: the binding candidate's first-query digest is used as the table and
  chain configuration; its protocol identity and request parameters are demo
  placeholders.
- Proximity: not connected. The table comparison is full-value, not proximity.
- Hiding: the charged column open uses a hiding charge, but no hiding security
  claim is made; provider/read ledgers are not a durable closure.
- Single-shot: not claimed.
- Upstream graph source and downstream graph binding: not connected.
- Commitment consistency: table root, record and G_u root are only locally
  bound in this transcript.
- Real prefix and production protocol identity: not used.
- Production wiring: the production verify entry is unchanged and the demo is
  not an accepted proof format.

## Artifacts

- `tests/column-session-unit.txt`: the nine module tests.
- `tests/column-session-real.txt`: the real gpt2-seq16 test and its timed
  process output. The local fixture path in the command line is replaced with
  `<fixture>`; no measured value is changed.
- `logs/cargo-test-workspace.txt`: the full workspace log.
- No binary artifacts were produced or deleted.
