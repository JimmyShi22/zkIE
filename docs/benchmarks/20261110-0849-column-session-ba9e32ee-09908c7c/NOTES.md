# column-session - ba9e32ee / 09908c7c

Trace of the isolated column verification component demonstration.

## Change

Code commit `ba9e32ee` ("Add isolated column verification component demo"),
based on `09908c7c`. The production verify entry is unchanged.

- New isolated module `crates/zkie-ops/src/proof/column_session.rs` and one
  real-artifact integration test in `zkie/tests/column_session.rs`.
- The module builds a small encoding table deterministically from an
  authenticated asset slice and exercises four component layers: column
  authentication, G_u registration and membership, a full-value table
  comparison, and the public FOLD input/output component.
- This is not a proof verifier. Its result must not be used to accept a proof.

## Encoding table

The table is built only after the trusted model anchor and the asset opening
are verified. The asset file is read only through the verified opening; it is
never materialized whole.

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
| table configuration | fixed demo digest `0x5a` repeated |
| commit primitive | `commit_table` test foundation; it materializes the small table, not the asset |
| record | derived by `commit_table` from the message, pads and deterministic salts |

The table's column root and record are absorbed into the transcript before any
column draw. The table's `IndexReader.root` is the column-tree root.

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
| table root | `77f94d0bbba9ed1ad9c291277ad4da5f05467e3e2ab6e85862b4a3df5fa3c9bd` | `IndexReader.root`, the column-tree root |
| G_u notice root | `4747474747474747474747474747474747474747474747474747474747474747` | fixed demo `GuNotice.root`; self-reported, not derived from the asset or table |

## Four layers

| layer | calls | what is actually checked | what is not claimed |
| --- | --- | --- | --- |
| 1. asset and column authentication | `verify_model_anchor`, `ModelCommitmentPublic::verify_opening`, `HidingState::begin_charge`, `CommittedTable::open_columns_charged`, `verify_columns` | anchor identity, asset opening identity/row order/width and Merkle paths; each drawn column's salt, symbols and Merkle path against the fixed table root; unique physical positions are opened once | not proximity; no claim that the asset table is the output of a trusted graph predecessor |
| 2. G_u registration | `GuRegistry::receive`, `GuMembership::new`, `GuMembership::join`, `column_plan_rows`, `GuMembership::check_row` | the self-reported notice matches the fixed G_u shape; one dynamic join; each row's oracle and group match the joined membership | `check_row` does not evaluate the row target; the notice root is self-reported; the G_u tree/group/parts are demo fixtures |
| 3. full table comparison | `encode_row_fp` re-encoding, comparison of every opened cell | every opened cell equals the recomputed codeword symbol for the same message and tail | deterministic full-value comparison on the small in-memory table; not a proximity or soundness test |
| 4. FOLD component | `FoldInputRow::check`, `fold_output`, `FoldOutput::next_stage` | the public FOLD input row has the shape/layout lengths and member layout; the public output is computed for fixed challenges; `next_stage` fails closed | the FOLD input row is a self-reported fixture, not derived from the asset or G_u table; the full FOLD gate and the next IOPP/CS stage are not connected |

The G_u column targets are recomputed from the full re-encoded table and the
fixed equation weights and compared with the registered row targets by a
separate control. This is the only target check; the membership check itself
only checks the oracle and group.

## Challenge and query order

| stage | order |
| --- | --- |
| table binding | asset meta root, table root, configuration, shape and complete record are absorbed before any draw |
| column draws | 4 logical draws from the existing chain; real run `[0, 6, 6, 3]`; unique opened positions `[0, 3, 6]` |
| column authentication | unique positions are opened with one hiding charge and verified against the table root |
| G_u registration | the self-reported notice and fixed G_u shape are absorbed after the draws; the membership join and column rows follow |
| full comparison | all opened cells are compared with the re-encoded table |
| FOLD binding | the self-reported FOLD input row is absorbed before the FOLD challenges |
| FOLD challenges | epsilon and the single gamma are drawn from the chain after the FOLD row; the public output is then computed |
| FOLD boundary | `FoldOutput::next_stage` is called and must fail closed |

The binding parameters, protocol identity, table configuration, G_u notice
root, runtime-tree starts and FOLD fresh root are demo placeholders. The
transcript is deterministic for the fixed test inputs and is not a production
seed.

## Negative coverage and rejection sources

Every negative case returns an error or a false layer result; none panics. The
source of the rejection is recorded so a local control is not presented as a
soundness proof.

| negative | layer and source |
| --- | --- |
| asset opening root changed | trusted anchor/opening check, deterministic |
| column Merkle path extended | `verify_columns` Merkle path check, deterministic |
| verified opening path with a changed merge value | membership registration still passes; separate target control against the full re-encoded table fails |
| G_u row with wrong oracle or wrong group | `GuMembership::check_row` |
| one opened cell changed | full table comparison |
| FOLD input coefficient length changed | `FoldInputRow::check` |
| FOLD next stage | `FoldOutput::next_stage` returns the expected fail-closed error |

The "verified opening path with a changed merge value" case is the requested
path-valid/merge-wrong case: the opening was authenticated, but the column
value supplied to the merge relation was changed; the membership check does not
see it and the target control does.

## Tests and measurements

- 8 synthetic unit tests: four-layer success, reproducible fixed fixture,
  asset root tamper, column path tamper, path-valid/merge-wrong, wrong G_u
  oracle/group, wrong opened cell, and the FOLD component boundary.
- Real gpt2-seq16 test: authenticated full-recompute control 16,622 ms; the
  isolated column session 363 ms; the whole test 73.24 s.
- Timed process: wall clock 1:13.41, maximum resident set size 6,378,456 kB.
- Workspace: 416 passed, 0 failed, 5 ignored.
- No proof bytes were produced by this component demonstration; no proof file
  exists, so no byte count is claimed. No memory or wall-time comparison is
  claimed as an optimization; the numbers are a single observation on this
  machine.

The workspace total includes the existing earlier tests and ignored
multi-GiB tests; the new module adds 8 unit tests and 1 ignored real-artifact
test.

## Validation scope

Only the `gpt2-seq16` case is used. No five-case three-stage run was made. No
claim is made about the full model or performance from this single case. The
production verify entry remains unchanged and is not switched to this demo.

## Unclosed items

- Asset to FOLD witness derivation: there is no public construction path from
  an authenticated asset table to the FOLD input coefficients, mask, target or
  challenges. The FOLD row here is an explicitly labelled self-reported
  fixture; adding such a derivation is a new interface and is not part of this
  demonstration.
- FOLD next stage: `FoldOutput::next_stage` is deliberately fail-closed; IOPP
  and the completion stage are not connected. The `FoldRoundSession` mask coin
  source is only a private test implementation in the current tree and is not
  used here.
- G_u notice root: self-reported and fixed; it is not derived from the asset
  root or the table root.
- G_u relation targets: the membership check does not evaluate them; only the
  local full-table target control does.
- G_u generator/group/layout: trusted test fixtures; the production source and
  statement binding are not connected.
- Table commitment: `commit_table` is the test foundation, not the production
  commit path; the asset is not materialized, but production streaming is not
  exercised.
- Proximity: not connected. The table comparison is full-value, not proximity.
- Hiding: the hiding charge is used for the column opening budget, but no
  hiding security claim is made.
- Single-shot: not claimed.
- Hiding/read ledgers: no cumulative provider ledger or durable hiding budget
  is claimed.
- Upstream graph source and downstream graph binding: not connected.
- Commitment consistency: the table root, record and G_u root are only
  locally bound in this transcript.
- Real prefix and production protocol identity: not used.
- Production wiring: the production verify entry is unchanged and the demo is
  not an accepted proof format.

## Artifacts

- `tests/column-session-unit.txt`: the eight module tests.
- `tests/column-session-real.txt`: the real gpt2-seq16 test and its timed
  process output. The local fixture path in the command line is replaced with
  `<fixture>`; no measured value is changed.
- `logs/cargo-test-workspace.txt`: the full workspace log.
- No binary artifacts were produced or deleted.
