# edge-session - 0f80cccc / cad747a1

Trace of the isolated two-step graph-edge demonstration.

## Change

Code commit `0f80cccc` ("Add a typed two-step graph edge demonstration"),
based on `cad747a1`. The production verify entry is unchanged.

- Step `4443` is verified by the real `ProfileLayerNorm` relation branch.
- Step `4444` input-0 is authenticated and compared with the producer output.
- A typed previous-step edge is bound into the candidate column-session first
  query before any challenge.
- The previous-step and asset-reference edge forms are separate typed forms;
  a previous-step edge has no asset identity.
- The selected `MatMul` 2c control remains separate; the consumer relation is
  not re-run here.

The earlier package `t002-s6b2f-r2-review.tar.gz` sha256
`f7ec7083448ff95b69138d985d44dd10c6b0c2ecd8fa8866a4a3f8d18c213b47` is
retained unchanged and is superseded by this round. The earlier `2f`, `2e-r2`
and initial packages also remain retained unchanged.

## Step 4443 inspection (R0)

The verifier's independent `trace_plan(16)` was inspected directly; no
assumption of MatMul was made:

| item | observed value |
| --- | --- |
| opcode | `ProfileLayerNorm` |
| config | `[16, 1024, 768, 32, 0, 0, 0]` |
| output count | 1 |
| input 0 | `Prev{step=4442, output=0, layout=Flat}` |
| input 1 | `Layout(Asset ln_f_w_i32.bin, Broadcast{rows=16})` |
| input 2 | `Layout(Asset ln_f_b_i32.bin, Broadcast{rows=16})` |
| input 3 | `Asset rsqrt_table_i32.bin, Flat`, external pool 197 |
| relation support | `OpCode::ProfileLayerNorm` branch in `verify_op_relation_with_external` |

Step `4444` is the selected MatMul `[16, 65536, 1024, 0, 0]`, with input 0
`Prev{step=4443, output=0, layout=Flat}` and input 1 the `lm_head` asset.

No R0 blocker: the ProfileLayerNorm relation is available and is executed.

## Typed edge

`FoldeqEdgeBinding` has three explicit forms:

| form | fields |
| --- | --- |
| `NotAnEdge` | no edge; used by the 2f single-step session |
| `Prev(PrevEdge)` | producer step/output, consumer step/input port, direction, layout/slice; no asset identity |
| `Asset(AssetEdge)` | consumer step/input port, direction, asset id/path/start/root/rows/width/count, layout/slice; no producer step/output |

The edge is built from the verifier's independent plan by
`edge_from_plan`. `check_edge_against_plan` rebuilds it and requires exact
equality, so a wrong producer, port, layout/slice, direction, source kind or
missing field is rejected. The bound edge bytes enter the candidate first
query after the 2f statement bytes and before its challenge fields. A
`NotAnEdge` session writes no edge bytes, so the 2f single-step first query is
unchanged.

## Value flow

| stage | implementation |
| --- | --- |
| producer plan check | exact opcode/config/input-port/source check for step 4443; input 0 must be the planned `Prev 4442/output 0/flat` edge |
| producer asset input | external `rsqrt` opening is checked against the trusted asset identity and complete Merkle opening; its values are supplied externally to the relation |
| producer relation | `verify_op_relation_with_external` with real ProfileLayerNorm config, input openings, output/raw/aux groups, and external input 3 |
| producer output | decoded from the authenticated output group after the relation succeeds; its oracle root is recorded |
| consumer input | step 4444 input 0 group is authenticated against its trace roots; decoded values are the consumer side |
| edge compare | exact length and value equality; producer output and consumer input roots are recorded separately |
| prefix binding | the typed edge is added to `ColumnBindingInput`; the candidate column session first query contains it before challenges; `edge_bound` is true |

Real values:

| item | value |
| --- | --- |
| producer step | `4443` |
| consumer step/input | `4444` / `0` |
| producer output length | `16384` |
| consumer input length | `16384` |
| first four values | `[-3457, -733, -13827, -4884]` on both sides |
| producer root | `3be32de90cc64c017ba6bfef44e1e704091fc8dec7d7d3d7365b878c0cd9bb2d` |
| consumer root | `22c81586ed0ac7794892ea73d15fbb5a954303f1aa06413c7b167cf36743c83c` |
| edge form | previous-step |
| edge bound | `true` |
| column table root | `83a62c3995b33823d29b5465f7a271d69b2cc9a4f73d4c403ee9e96c01f623b3` |
| column queries | `[2, 3, 4, 5]` |
| G_u rows | `6` |

The producer and consumer roots differ; equality is established after both
openings are authenticated to their own roots, not by treating the roots as
equal.

## Rejection layers

| negative | layer |
| --- | --- |
| producer output value changed | producer relation/opening authentication |
| producer output root changed | producer relation/opening authentication |
| consumer input opening value changed | consumer input opening authentication |
| edge value changed after authentication | edge value comparison |
| wrong producer step | independent-plan edge identity |
| wrong consumer port | independent-plan edge identity |
| wrong layout/slice | independent-plan edge identity |
| missing producer/asset field | typed edge validation |
| same values under a wrong producer | independent-plan edge identity rejects before value equality |
| edge present vs absent | candidate first-query transcript changes; table root stays the same |

These are separate calls in the test, so a relation/authentication failure is
not reported as an edge comparison failure.

## Tests and measurements

- Engine foldeq: 38 passed, 0 failed.
- Column session: 9 unit tests passed.
- Edge session: 2 unit tests passed.
- Real gpt2-seq16: 1 two-step edge test passed.
- Workspace: 422 passed, 0 failed, 6 ignored.
- The 2c selected MatMul control passed in 16,968 ms.
- The edge session passed in 2,596 ms; whole test 90.66 s.
- Timed process: wall clock 1:31.13, maximum resident set size 16,916,584 kB.
- No proof bytes were produced; no byte count, soundness bound or performance
  gain is claimed.

## Validation scope

Only `gpt2-seq16` is used. No five-case three-stage run was made. The
production verify entry is unchanged and is not switched to this demo.

## Unclosed items

- Upstream of step 4443 (step 4442 and earlier) is a boundary; no upstream
  relation is verified.
- Downstream of step 4444 / final graph output: not connected.
- The consumer MatMul relation is not re-run in this module; the 2c selected
  MatMul control is the separate regression.
- Inputs 1/2 of step 4443 are plan asset sources but the trace supplies
  relation openings, not asset-commitment openings; their relation openings
  are verified, but the asset-commitment source is not closed.
- The complete graph, graph edge set and graph acceptance are not claimed.
- Proximity is not connected.
- Hiding is not claimed; the column session's hiding charge is a local budget.
- Single-shot is not claimed.
- G_u root/sample answer, G_u target evaluation, FOLD row/round fixtures,
  asset-to-FOLD witness derivation, and the closed FOLD next stage remain as
  in the earlier column session.
- Real request identity, protocol identity freeze, remaining security
  parameters and production wiring remain unclosed.
- `commit_table` remains the test foundation; the complete opening is
  resident.
- Production `verify_request` is unchanged; this is not an accepted proof
  format.

## Artifacts

- `tests/engine-foldeq.txt`: 38 foldeq tests.
- `tests/column-session-unit.txt`: 9 column tests.
- `tests/edge-unit.txt`: 2 edge tests.
- `tests/edge-real.txt`: real gpt2-seq16 two-step test and timed output. The
  local fixture path in the command line is replaced with `<fixture>`; no
  measured value is changed.
- `logs/cargo-test-workspace.txt`: full workspace log.
- No binary artifacts were produced or deleted.
