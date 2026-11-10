# binding-prefix - 5c51d904 / 9e9b2be8

Trace of the real selected-plan and case binding for the isolated column
session.

## Change

Code commit `5c51d904` ("Derive the column binding from the selected plan and
real case"), based on `9e9b2be8`. The production verify entry is unchanged.

- The real gpt2-seq16 fixture no longer calls the fixed demo input.
- `selected_plan_identifier` extends the prior selected-step encoding with
  every input port, source, and the identity of every asset named by an asset
  source.
- The real case id is read from the selected policy case and enters the typed
  statement bytes.
- `ColumnBindingInput::demo` is test-only and explicitly fixed; it is not used
  by the real fixture.
- The interface guarantees that the supplied plan/case bytes enter the
  challenge prefix. It does not make the plan, case, parameters or protocol
  trusted.

The earlier package `t002-s6b2f-review.tar.gz` sha256
`68a2e4bb06b07753c76f2f550ae807033a1b376e024ff50fbdbeea978ac29d12` is
retained unchanged and is superseded by this round. The earlier `r2` and
initial packages remain retained unchanged as well.

## Plan identifier

`selected_plan_identifier(public, trace_plan, selection)` is rebuilt by the
verifier from its independently held static `trace_plan` and selected step.
The encoding covers:

| covered item | encoding |
| --- | --- |
| schema version | `SESSION_SCHEMA_VERSION` |
| model version | plan field |
| selected step | `step_index` |
| opcode | `MatMul` |
| config | all selected config values |
| input port count | selected inputs length |
| input ports | explicit port index |
| previous-step source | source kind, step, output, flat layout |
| asset source | source kind, path, asset id, start, root, rows, row width, element count, leaf id/generation/table identity/leaf root |
| output count | one selected MatMul output |

Gaps, explicitly: the identifier does not cover the rest of the trace graph,
other step plans, the case id, the request id, the protocol/security
parameters, the full plan authenticity, or plan/source authorization. The
selected step is local to the observation, not a complete trusted graph.

## Real case and binding

The real fixture obtains the case id from the selected policy request
(`gpt2-seq16`) and the plan identifier from
`selected_plan_identifier(&public, &plan, &selection)`. It builds
`ColumnBindingInput::from_plan(plan_identifier, case_id)`. The request id is
the explicit placeholder; the protocol identity remains a candidate; the
security parameters remain unconfigured.

`ColumnBindingInput::demo` is `#[cfg(test)]` and used only by module unit tests.
The real fixture does not call it.

## Order and consumption

The actual order remains:

1. independent parameters digest (version word, candidate protocol identity,
   `RequestParams`, plan identifier) is the table/chain configuration;
2. `commit_table` produces the reader root and record;
3. the typed statement is built with the real case and the reader table
   identity; `FoldeqBinding` canonically encodes it;
4. the statement bytes enter the first query after reader root/record and
   before all challenges;
5. gamma is drawn; G_u SEND/answer and record join precede the column draw;
6. query draw/read, verified-column join, full comparison, FOLD component.

No statement is placed after the first challenge.

## Consumption tests

The real fixture asserts:

| check | observed |
| --- | --- |
| plan identifier source | `input.parameters.plan_identifier == selected_plan_identifier(...)` |
| case source | `input.case_id == trusted_request.case_id` |
| case entering the same session prefix | changed real case id keeps the table root but changes `transcript_after_queries` |
| plan entering the same session prefix | changed plan identifier changes both table root and `transcript_after_queries` |
| binding parameters digest | outcome digest equals the input parameters digest |
| binding table identity | outcome identity equals `reader_table_identity(reader)` |

The changed plan/case cases are prefix-change negatives, not rejection: the
interface has no trusted expected plan/case to compare against, so an
inconsistent input changes the binding rather than being rejected. This is
stated rather than presenting prefix sensitivity as source trust.

The engine tests additionally cover the five-input consumption sensitivity
(parameters/configuration, case, request placeholder, table root, record),
wrong table identity, protocol mismatch, empty case, zero request, oversized
statement, plan/binding/reader substitution, and changed first query.

## Source report

| field | actual source | verification path |
| --- | --- | --- |
| plan identifier | selected `trace_plan` rebuilt by the verifier plus selected step/ports/sources/asset identity | `selected_plan_identifier` encoding; real fixture equality assertion; changed-plan prefix test |
| case id | real selected policy case (`gpt2-seq16`) | typed statement bytes; real fixture equality and changed-case transcript test |
| table identity | reader actual root/shape/configuration/record | `reader_table_identity`; `FoldeqBinding::new` and plan fault; real fixture assertion |
| request id | explicit placeholder | not a real request identity; no request distinction claimed |
| protocol identity | candidate value | not frozen; no production protocol claim |
| security parameters | unconfigured | not connected |

This report describes sources, not trust: the interface ensures the bytes are
consumed in the challenge path, not that the plan/case/parameters are
independently trusted.

## Four layers and data flow

| layer | calls | checked | not claimed |
| --- | --- | --- | --- |
| plan binding | selected plan identifier, typed `ColumnBindingInput`, `FoldeqBinding`, `FoldeqBinder`, `FoldeqSession::open`/`bind_first_query` | real plan/case bytes in first query; table identity actual; parameters digest as configuration; exact first-query consumption | plan/case/parameters source trust |
| asset and column authentication | anchor, complete opening verification, `commit_table`, charged open, `read_queries` | full opening rows/paths, fixed column root, unique positions | complete opening is resident; not production streaming |
| G_u and gamma | real session gamma, `send_gu`, seeds/answers, record join, plan draw, verified-column join, `foldeq_weights` | G_u before draw; real equation weights; membership oracle/group | G_u root/sample self-reported; membership does not evaluate target |
| full comparison and FOLD | `check_full_columns`, `check_merge_targets`, FOLD bind/start/rounds/finish/next | full re-encoded cells, target control, FOLD order and fail-closed boundary | not proximity; FOLD row/rounds are fixtures; no full FOLD/IOPP closure |

## Real run values

| item | value |
| --- | --- |
| asset meta root | `9a8be4edbbc390fd6844710399441788506c2f1f3a1fc9c9c95e19e067971832` |
| table root | `83a62c3995b33823d29b5465f7a271d69b2cc9a4f73d4c403ee9e96c01f623b3` |
| G_u root | `4747474747474747474747474747474747474747474747474747474747474747` |
| queries | `[4, 7, 7, 4]` |
| opened positions | `[4, 7]` |
| gamma coordinates | 2 |
| equation weights | 4, equal to `foldeq_weights(gamma, 2)` |
| column cells | 16 |
| G_u rows | 6 (1 sample + 1 record + 4 column) |

## Negative coverage

| negative | rejection/source |
| --- | --- |
| wrong selected-plan identifier | prefix changes; table root and transcript differ (prefix-change negative) |
| wrong case id | table root unchanged; statement bytes and transcript differ (prefix-change negative) |
| statement table identity changed | `FoldeqBinding::new` |
| protocol mismatch | `FoldeqBinding::new` |
| empty case / zero request | typed statement validation |
| oversized statement | binding length bound |
| plan binding replaced after binder | `FoldeqSession::open` first-query equality |
| reader replaced after binder | `FoldeqSession::open` |
| changed first query after open | `bind_first_query` |
| asset opening root changed | trusted anchor/opening check |
| column path extended | `read_queries`/`verify_columns` |
| verified path with changed merge value | membership passes; target control fails |
| wrong G_u oracle/group | `GuMembership::check_row` |
| opened cell changed | full comparison |
| FOLD coefficient length | `FoldInputRow::check` |
| FOLD next stage | expected fail-closed |

## Tests and measurements

- Engine foldeq: 38 passed, 0 failed.
- Column session: 9 unit tests passed; real gpt2-seq16 test passed.
- Workspace: 420 passed, 0 failed, 5 ignored.
- Real gpt2-seq16: full-recompute control 16,489 ms; column session 365 ms;
  whole test 75.03 s.
- Timed process: wall clock 1:15.19, maximum resident set size 6,376,672 kB.
- No proof bytes were produced; no byte count, soundness bound or performance
  gain is claimed.

## Validation scope

Only `gpt2-seq16` is used. No five-case three-stage run was made. The
production verify entry remains unchanged and is not switched to this demo.

## Unclosed items

- request id real source; real request identity and distinction are not closed.
- protocol identity freeze; candidate only.
- remaining security parameters and complete trusted setup.
- graph edges, upstream source and downstream binding.
- real request representation.
- proximity; full comparison is not proximity.
- hiding; a charge is used but no hiding security or durable ledger closure.
- single-shot.
- asset to FOLD witness derivation; FOLD row/round values are fixed fixtures.
- FOLD next stage/IOPP/CS; `FoldRoundSession` mask coin source remains private
  test code and is not used.
- G_u root/sample answer self-reported; membership does not evaluate targets.
- G_u generator/group/tree/layout and fresh-mask tree are fixtures.
- table commitment is the test foundation; complete opening is resident.
- production `verify_request` is unchanged; this is not an accepted proof
  format.

## Artifacts

- `tests/engine-foldeq.txt`: 38 foldeq tests.
- `tests/column-session-unit.txt`: 9 module tests.
- `tests/column-session-real.txt`: real gpt2-seq16 test and timed output. The
  local fixture path in the command line is replaced with `<fixture>`; no
  measured value is changed.
- `logs/cargo-test-workspace.txt`: full workspace log.
- No binary artifacts were produced or deleted.
