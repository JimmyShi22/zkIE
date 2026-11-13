# Projection reduction core on the production diagnostic path

## Scheme and baseline

- Scheme: `projection-diagnostic` (execution-layer transition; the diagnostic
  entry is recorded as a transition, not as the hidden-input acceptance).
- Code: `76de1cc0`; previous commit `94265372`; branch `refactor/v2`.
- Scope: one real projection consumer (the layer-0 key projection). All
  existing full-weight relation checks stay in place; no protocol object or
  message is added; no plaintext is removed.

## What the test does

1. Rebuilds the registered model commitment from the saved private table
   randomness and asserts its anchor equals the registered public anchor
   (`91397c7e…0cc060`).
2. Verifies the real gpt2-seq16 trace with `verify_shared_trace_outputs`,
   takes the verified activation rows that feed the layer-0 key projection,
   and the verified output rows of the twelve head consumers, concatenated to
   the 1024-wide consumer output.
3. Recomputes the raw integer accumulator from the verified activation and
   the trusted weight rows, requantizes with the consumer's fixed-point shift
   from the plan, and asserts equality with the verified trace output. This is
   the deterministic engineering check on already authenticated trace values
   that the ruling allows; it is not a hidden-source closure claim.
4. Arranges the real weight rows in the hidden session's table layout (table
   row = output column, entries = input rows, i.e. the transpose of the
   exported input-major weight file), commits the table with the registered
   table geometry, and runs `prove_hidden_lin`/`verify_hidden_lin` over all
   sixteen positions with the actual activation and raw output. The reduced
   witness is committed by the session's first-part path; the pending claim is
   the session's own single claim (J=1 pass-through).
5. Checks the projection reduction core (`projection_claims` and
   `check_projection_reduction`) against the same table values; this is the
   plaintext diagnostic comparison, not a declaration layer.
6. Rejects a changed output, a changed coefficient and a changed table
   identity through the same session.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Real consumer input/output values | verified production trace outputs; trusted plan step | test takes the verified rows of the step | test pass, 16 positions |
| Reduced witness from the committed table | session first part; W committed there | session reduced rows over the real table | prove/verify pass |
| Fixed-point semantics kept | plan shift; integer raw accumulator | requantize equality against the verified output | test assertion |
| Source-layer engineering check | verified activation + trusted weight rows | raw recomputation | test assertion |
| Session rejects changed values | session verifier | changed output / coefficient / identity negatives | rejected |
| Pending/member formal layer, index-mask message, hidden-input entry | next step | **not in this transition** | gap |

## Evidence

- Diagnostic test over all sixteen positions: 26.17 s test time, 27.19 s
  wall, VmHWM 6,295,992 KiB (release). Two-position run: 24.71 s / 24.87 s /
  6,297,496 KiB. Logs `diagnostic-16pos.log`, `diagnostic-2pos.log` and the
  time files.
- `cargo test --workspace`: 51 suites, 991 passed, 0 failed, 18 ignored (the
  new test is explicitly invoked against the registered trace artifacts and
  skips without them).
- Hard gate: norm-free toy output-head proof 45,080 bytes, sha256
  `7a5da6eb…de4cfb`, unchanged; `proof-bytes.txt` and `proof-45080.bin`.
- The test is explicitly invoked (`invocation.txt`); it is ignored by default
  like the other real-asset stage tests.
- No push.

## Boundaries

- This is the diagnostic transition only. It does not replace the three
  acceptance points (pending/member identity, two-layer identity checks,
  index-mask message wiring) and does not claim the hidden-input consumer
  closure.
- The diagnostic table is the transposed view of the exported input-major
  weight rows and is not the registered table instance. The layout mapping
  between the registered instance and the session's reduction convention
  remains a trusted-plan decision for the formal channel.
- The session's activation commitment here is a diagnostic object with fixed
  test salts; it is not connected to the trace objects by a protocol relation
  (the values are the verified trace values, checked by recomputation). The
  formal hidden-input activation binding and the output/raw cross-object
  binding remain the next step.
- No index-mask message, no BATT (single claim), no mask ledger wiring in this
  step.
- Bias stays in its separate Add step with the existing parameter checks; no
  bias is folded into a target or mask.
- The asset plaintext is not removed; every existing asset and relation check
  remains. This is not evidence that plaintext can be deleted for this asset.
- gemma3 and timesfm are not touched; the toy hard gate is unchanged.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `diagnostic-16pos.log`, `diagnostic-16pos.time.txt`,
`diagnostic-2pos.log`, `diagnostic-2pos.time.txt`, `invocation.txt`,
`NOTES.md`.
