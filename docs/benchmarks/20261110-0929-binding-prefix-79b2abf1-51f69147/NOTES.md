# binding-prefix - 79b2abf1 / 51f69147

Trace of the typed 2b statement prefix on the isolated column session.

## Change

Code commit `79b2abf1` ("Bind the column session prefix to the typed
statement"), based on `51f69147`. The production verify entry is unchanged.

- New required `FoldeqBinding` on `FoldeqPlan`; no unbound plan form.
- The canonical typed `BindingStatement` bytes enter the first query after the
  reader root/record and before the verified statement id and all challenges.
- The parameters digest is independently constructed from the version word,
  candidate protocol identity, request parameters and plan identifier; it is
  the table/chain configuration and never contains the table root/record
  derived from it.
- The statement table identity is the actual reader root, shape,
  configuration and record; construction and plan fault both reject a
  mismatch.
- `FoldeqBinder` and `FoldeqSession::open`/`bind_first_query` consume the same
  extended first query; plan, binding, reader or advanced-chain replacement is
  rejected.

The earlier packages are retained unchanged and are superseded for this step:

- `t002-s6b2e-review.tar.gz` sha256
  `d68ffd012469aa382f462704ee4c6a37c4427d1d913c364d3c3fcd9578ec83ee`;
- `t002-s6b2e-r2-review.tar.gz` sha256
  `1a80556fc40d0db0f708ba85d59569e6c1f2ab7a079084001014931aed92dd04`.

## Binding order

| stage | content and source |
| --- | --- |
| independent parameters | version word, candidate protocol identity, `RequestParams` and plan identifier; `BindingParameters::digest()` is the table/chain configuration |
| table construction | `commit_table` with the parameters digest as configuration; produces the column root and record after the configuration |
| statement construction | typed 2b `BindingStatement` with placeholder case id/request id and `reader_table_identity(reader)` as table identity; `FoldeqBinding::new` checks version/protocol and table identity, then canonically encodes the statement |
| first query | reader name (commitment + record), then the canonical statement bytes, then table/fold shapes, verified statement id, group/tree/budget values, gamma path and G_u shape |
| binder | parameters digest is the binder parameter value; `first_query` includes the statement bytes; start is derived from both |
| session | `open` checks binder first query and start; `bind_first_query` requires exact equality; gamma is drawn only after binding |

The statement is never placed after the first challenge. `request_id` and
`case_id` are explicit placeholders: this does not claim a real request
identity or that different real requests are distinguished. The protocol
identity is a candidate value.

## Consumption sensitivity

The engine tests change five inputs one at a time and assert the actual
`FoldeqSession` first challenge/prefix changes:

| changed input | observed in test |
| --- | --- |
| parameters/configuration | different `first_query`, binder start and session gamma |
| case id placeholder | different `first_query`, session gamma |
| request id placeholder | different `first_query`, session gamma |
| table root | different `first_query`, session gamma |
| reader record | different `first_query`, session gamma |

This is a consumption-chain sensitivity test, not a proof that any input
source is trusted. The tests also reject a wrong table identity, protocol
mismatch, empty case, zero request id, oversized statement, and plan/binding
or reader substitution.

## Table identity

`reader_table_identity` hashes the actual reader root, configuration, shape and
`IndexName` (commitment plus complete record). `FoldeqBinding::new` rejects a
statement naming another identity, and `FoldeqPlan::fault` recomputes the same
identity. The configuration does not contain the root or record it derives:
the table is committed only after the independent parameters digest is fixed.

The binding is typed and length-bounded. The statement encoding is produced
from `BindingStatement::canonical_bytes`; no raw prover-supplied statement
bytes are accepted. The statement byte limit is 1 MiB.

## Sources

| item | independent verifier configuration | fixture / prover-provided |
| --- | --- | --- |
| parameters | version word, candidate protocol identity, `RequestParams`, plan identifier are supplied by the module configuration | no proof bytes |
| case id / request id | explicit placeholders | no real request identity; not distinguished |
| table configuration | parameters digest | no proof bytes |
| table root and record | produced by `commit_table` from the verified asset opening and fixed configuration | opening fixture |
| statement table identity | derived from the actual reader | no separate caller value |
| G_u root | fixed demo SEND root | public message value |
| G_u sample answer | fixed demo value | self-reported |
| FOLD input row and round values | fixed demo fixture | self-reported; not asset-derived |
| FOLD fresh mask root/tree | fixed demo plan values | fixture |

## Four layers and data flow

| layer | calls | checked | not claimed |
| --- | --- | --- | --- |
| plan binding | `FoldeqPlan`, `FoldeqBinding::new`, `reader_table_identity`, `FoldeqBinder::new`, `FoldeqSession::open`, `bind_first_query` | typed statement bytes and table identity in the actual first query; parameters digest as binder configuration; exact binder/plan/session equality | statement source is independently trusted; request identity is real |
| asset and column authentication | trusted anchor, `verify_opening`, `commit_table`, charged open, `read_queries`/`VerifierReads` | complete opening row/path checks, fixed column root, unique physical positions | complete opening is resident; this is an expensive isolation control, not production mmap/ranged memory closure |
| G_u and gamma | `draw_gamma`, `send_gu`, `shared_seeds`, `receive_gu_answer`, `join_gu_record`, `draw_queries`, `join_gu_read_columns`, `foldeq_weights` | G_u message before query draw; real session gamma; record/column rows with real equation weights; membership oracle/group checks | G_u root/sample answer self-reported; membership does not evaluate targets |
| full comparison and FOLD | `check_full_columns`, `check_merge_targets`, `fold_bind_row`, `start_fold`, FOLD rounds, `fold_finish`, `fold_next_stage` | full re-encoded cell comparison, column target control, public FOLD shape/order/output, fail-closed next stage | full comparison is not proximity; FOLD row/rounds are fixed component fixtures; no full FOLD or IOPP/CS closure |

The actual session still follows: first query and binding, gamma, G_u SEND and
answer round, record join, plan-constrained column draw, verified read,
verified-column join, full comparison, FOLD bind/start/rounds/finish, closed
next stage.

## Real run values

| item | value |
| --- | --- |
| asset meta root | `9a8be4edbbc390fd6844710399441788506c2f1f3a1fc9c9c95e19e067971832` |
| table root | `27bc1861bf38eb2b098bd216868e0472cd7aa233996e7476aaa2c4a227780630` |
| G_u root | `4747474747474747474747474747474747474747474747474747474747474747` |
| queries | `[4, 3, 7, 7]` |
| opened positions | `[3, 4, 7]` |
| gamma coordinates | 2 |
| equation weights | 4, equal to `foldeq_weights(gamma, 2)` |
| column cells | 16 |
| G_u rows | 6 (1 sample + 1 record + 4 column) |

## Negative coverage

| negative | rejection source |
| --- | --- |
| statement table identity changed | `FoldeqBinding::new`, deterministic |
| protocol identity mismatch | `FoldeqBinding::new` |
| empty case id / zero request id | typed `BindingStatement` validation |
| oversized statement | binding length bound |
| plan binding replaced after binder | `FoldeqSession::open` first-query equality |
| reader replaced after binder | `FoldeqSession::open` first-query/plan fault |
| first query changed after open | `bind_first_query` exact equality |
| asset opening root changed | trusted anchor/opening check |
| column Merkle path extended | `read_queries`/`verify_columns` |
| verified opening path with changed merge value | membership passes; separate full-table target control fails |
| wrong G_u oracle/group | `GuMembership::check_row` |
| opened cell changed | full table comparison |
| FOLD input coefficient length changed | `FoldInputRow::check` |
| FOLD next stage | expected fail-closed |

## Tests and measurements

- Engine: 227 passed, 0 failed, including the five-input consumption
  sensitivity, identity/canonical-field negatives, and plan/binding/reader
  substitution.
- Column session: 9 unit tests passed; real gpt2-seq16 test passed.
- Workspace: 420 passed, 0 failed, 5 ignored.
- Real gpt2-seq16: full-recompute control 15,993 ms; column session 362 ms;
  whole test 71.90 s.
- Timed process: wall clock 1:12.07, maximum resident set size 6,377,248 kB.
- No proof bytes were produced; no byte count, soundness bound or performance
  improvement is claimed.

## Validation scope

Only the `gpt2-seq16` case is used. No five-case three-stage run was made. The
production verify entry remains unchanged and is not switched to this demo.

## Unclosed items

- `request_id` real source: only an explicit placeholder; real request identity
  and distinction are not closed.
- `protocol_identity` freeze: candidate value only.
- Remaining security parameters: not connected; no complete trusted setup.
- Graph edges, upstream source and downstream binding: not connected.
- Real request: not represented.
- Proximity: not connected; full comparison is not proximity.
- Hiding: a charge is used for column opening, but no hiding security claim or
  durable provider ledger closure is made.
- Single-shot: not claimed.
- Asset to FOLD witness derivation: no public construction path from the
  authenticated asset table to the FOLD coefficients/mask/target/rounds; the
  FOLD row and round values are self-reported/component fixtures.
- FOLD next stage: `FoldOutput::next_stage` is fail-closed; IOPP/CS is not
  connected. `FoldRoundSession`'s mask coin source remains a private test
  implementation and is not used.
- G_u root/sample answer: fixed public/self-reported demo values; no G_u
  oracle table.
- G_u target evaluation: membership does not evaluate targets; record target
  comes from the reader record, column target is local full-table control.
- G_u generator/group/tree/layout and fresh-mask tree: test fixtures, not a
  production statement/group or mask-opening closure.
- Table commitment: `commit_table` is the test foundation, not the production
  streaming/ranged commit; complete opening is resident.
- Binding statement/case/request source: typed 2b construction and candidate
  placeholders, not a production request identity or trusted external
  statement source.
- Production wiring: `verify_request` unchanged; this demo is not an accepted
  proof format.

## Artifacts

- `tests/engine-binding.txt`: engine tests, including the five-input
  consumption sensitivity.
- `tests/column-session-unit.txt`: the nine module tests.
- `tests/column-session-real.txt`: the real gpt2-seq16 test and timed process
  output. The local fixture path in the command line is replaced with
  `<fixture>`; no measured value is changed.
- `logs/cargo-test-workspace.txt`: full workspace log.
- No binary artifacts were produced or deleted.
