# Projection consumer in the hidden-input session mode

## Scheme and baseline

- Scheme: `projection-hidden-input` (session composition on verified
  production values; still not the production proof wiring).
- Code: `66929bc8`; previous commit `53c27123`; branch `refactor/v2`.
- Scope: the layer-0 key projection consumer on the real registered instance,
  with the hidden-input session entry and the sampled index-mask message. All
  existing full-weight checks stay; no protocol object or message is added.

## What the test does

1. Same verified production values as the diagnostic transition: the real
   gpt2-seq16 trace verified, the layer-0 key projection activation rows, the
   real weight rows of the registered asset, all sixteen positions.
2. Builds the hidden-input session plan: empty row weights, empty per-position
   masks, the sampled index-mask plan (independent oracle and tree, mask code
   of the group, entry allocation for the matrix row bits and the position
   count), the shown-map message, and the own fold-mask tree sized to the
   index-mask rows. The plan carries no output relation or closure (the
   allowed hidden combination for this intermediate consumer).
3. Runs `prove_hidden_lin_hidden` and `verify_hidden_lin_hidden`: the session
   commits its reduced rows and the evaluation targets, and the verifier
   replays the first, shared and second chains with both proximity tests.
4. Rejects a plan without the sampled index-mask message (validation) and a
   changed table identity (prover) through the same session.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Hidden-input entry over the real consumer | session hidden entries; real verified activation and weight table | session plan without plaintext weights or outputs | prove/verify pass |
| Sampled index-mask message | `IndexMaskPlan`; independent oracle and tree; mask code | plan carries the mask message and the shown map | hidden run pass |
| Mask slot ledger shape | mask entry allocation for row bits and positions | allocation in the plan | plan validation pass |
| Missing index-mask message rejected | hidden-mode plan validation | negative | rejected |
| Changed table identity rejected | prover table identity check | negative | rejected |
| Cross-object source binding, two-way negatives, bias identity negative | next step | **not in this composition** | gap |

## Evidence

- Both explicitly invoked tests over all sixteen positions: 2 passed, 27.19 s
  (release); diagnostics log `projection-hidden-16pos.log` and timing file.
- `cargo test --workspace`: 51 suites, 991 passed, 0 failed, 19 ignored.
- Hard gate: norm-free toy output-head proof 45,080 bytes, sha256
  `7a5da6eb…de4cfb`, unchanged; `proof-bytes.txt` and `proof-45080.bin`.
- Invocation in `invocation.txt`; no push.

## Boundaries

- Still a session composition in the test, not embedded in the production
  proof message stream: the index-mask message has no request position yet
  and the trace carries no session field.
- The activation commitment root is pinned by the plan with fixed salts; it
  is not yet connected to the production step's input object by a protocol
  relation. The raw/output cross-object binding (the two-layer identity
  checks) is not implemented here.
- The two-way negatives (change the trace without the session, change the
  session reduction without the trace) and the bias identity negative are not
  in this delivery.
- Keeps every existing full-weight relation check; no plaintext removed; the
  asset plaintext deletion claim stays open.
- Bias remains in its separate Add step; no bias is folded into a target or
  mask. gemma3 and timesfm are not touched.
- Not the first-batch acceptance: this is one incremental delivery of the
  session-side pieces.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `projection-hidden-16pos.log`,
`projection-hidden-16pos.time.txt`, `invocation.txt`, `NOTES.md`.
