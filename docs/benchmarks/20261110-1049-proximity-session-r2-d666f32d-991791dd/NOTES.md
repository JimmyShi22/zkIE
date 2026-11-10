# proximity-session r2 - d666f32d / 991791dd

Trace of the isolated CS -> BASE proximity component after the prover/verifier
split.

## Change

Code commit `d666f32d` ("Split the proximity component into a prover and a
verifier"), based on `991791dd`. The production verify entry is unchanged.

- New verifier view: `ProximityStatement` carries codes, claim data, target and
  row lengths only; old plaintext rows are not part of it.
- The prover owns the plaintext rows, the switched message, all commitments and
  all openings. The verifier owns the independent plan, its own registry built
  from the public commitments, all challenges and all checks.
- The verifier receives openings only through `ProximityProverChannel` and
  authenticates each one with `verify_opening` against its own registry. Old
  symbols used in claim registration are the authenticated opening values; the
  verifier no longer recomputes them from rows.
- CS registration keeps the original claim and every per-position main and mask
  generator claim separate; targets are never summed across main and mask.
- BASE checks the absorbed target* against the target* carried by M'/R', checks
  the linear relation with the verifier's combined claim and target, and checks
  sampled symbols against authenticated final-row and star openings.
- The plan binds its stages, including the fold flag, and rejects fold
  configurations (including non-final k >= 1) as unsupported; adjacent code,
  mask capacity and switch/base message-shape constraints are checked.
- The chain prefix binds the plan, the statement (codes, lengths, claim, mask
  claims, target) and all six commitments before any challenge.

The earlier package `t002-s6b3a-review.tar.gz` sha256
`17e862d1863fe18b53cafe41366d5b5dceda28a1f3810b6755c03118e20606ca` is void and
retained unchanged. The earlier `2g`, `2f` and `2e` packages remain retained
unchanged.

## Prover and verifier boundary

| item | prover | verifier |
| --- | --- | --- |
| plan | receives the verifier plan | owns the plan |
| statement | holds the prover rows | holds `ProximityStatement` only |
| commitments | commits rows, holds roots/openings | rebuilds a registry from public commitments |
| sample | declares the OOD answer | draws the seed, absorbs the answer, compares it with the explicit plaintext oracle |
| CS positions/theta | receives the challenges | draws them |
| old symbols | can open its own rows | uses only authenticated openings |
| target* | computes it from its own rows | absorbs it, then requires M'/R' to carry the same value |
| lambda | receives it | draws it |
| M'/R' | sends them in the clear | absorbs all M' then all R' and checks the relations |
| BASE positions | receives them | draws them |
| checks | none | linear, symbol, opening, count, shape and order checks |

The verifier session is a separate object from the prover. A test channel that
refuses every message fails closed; a caller cannot form `BaseComplete` from the
public booleans because `verified` is private and only the successful session
sets it.

## Plan constraints

- exactly one code-switch stage followed by one BASE stage;
- input/switch/base main and mask codes are valid, mask code not longer than its
  main code, switch code not longer than input, base not longer than switch;
- switch message holds the old message and tail; base message shape equals the
  switched message shape;
- one answer per stage, positive sample counts, mask row has room for the
  answer, `h` within mask capacity;
- non-final folding requires k >= 1; folding is then rejected as unsupported,
  so the isolated subset never silently treats a fold stage as no-fold;
- final stage has k = h = 0 and no folding;
- choices come from the verifier plan only.

## Negative coverage

| negative | layer | detection |
| --- | --- | --- |
| missing/extra/swapped stage, wrong role | plan | deterministic |
| wrong code adjacency, base/switch shape, mask capacity | plan | deterministic |
| folding stage with k = 0 or k >= 1 | plan | deterministic (unsupported) |
| wrong statement code | verifier construction | deterministic |
| valid commitment with a different row count | verifier construction | deterministic |
| wrong statement target with valid commitments | BASE linear | deterministic |
| wrong target* message | target* equality and linear | deterministic |
| wrong M' value | BASE linear | deterministic |
| wrong sample answer | sample declaration vs plaintext oracle | deterministic component check |
| old opening value/path/index | Merkle opening authentication | deterministic |
| same-shape valid switched commitment with old openings | opening authentication | deterministic |
| consistently replaced switched message | BASE symbol check | every position differs in this witness; in general sampled |
| missing or erroring prover message | channel error | deterministic |
| public booleans without the private flag | outcome constructor | deterministic |
| truncated plan, wrong challenge order | plan and transcript tests | deterministic |

No test relies on a random sample missing a tamper. The consistently replaced
message test changes every entry, so every drawn BASE position detects it. The
symbol check remains a sampled check in the protocol.

## Tests and measurements

- Proximity plan: 2 passed.
- CS stage: 2 passed.
- BASE stage: 1 passed.
- Proximity session unit: 10 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 component: 1 passed.
- Workspace: 437 passed, 0 failed, 7 ignored.
- Real component run: session 0 ms (reported precision); whole test 57.01 s;
  timed process wall clock 57.81 s; VmHWM 4,233,948 kB.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

Real fixture values (gpt2-seq16):

| item | value |
| --- | --- |
| old main root | `34f80cc3c66f5965bc55c4911623d0d7bdf01c1ab4eed51df7b6ea54ecd72acd` |
| old mask root | `96d24a904a93adf7422ee65050f62b09801cffa7e80b9a9d0b83761a0ead4647` |
| switched root | `5450e4a78c7ac6abe7e1c299c83282532bc96005a21427047337e3f28cce81d4` |
| star root | `d8353181aa8a2bab97e6ea9070d053c2afad5e33a7cce981b4743236fda96009` |
| CS positions | `[0, 5, 1]` |
| BASE positions | `[5, 1, 1, 7]` |
| registration count | `7` |

The roots are local component commitments, not a real oracle group.

## Validation scope

Only `gpt2-seq16` is used. The old rows come from authenticated asset opening
values in the integration test; the fixtures are local. No five-case
three-stage run was made. The production verify entry remains unchanged.

## Unclosed items

- CS pending claims: registered but not proved by a later proximity test. Only
  the isolated BASE overlap is checked; a CS output that is consistently
  re-committed is not rejected by CS algebra alone.
- Kept rows, fresh mask rows and a second mask row: unsupported and rejected;
  the handoff is no-kept, one-mask only.
- Row folding: unsupported and rejected. The non-final k >= 1 constraint is
  enforced before the unsupported rejection, but no folding implementation
  exists.
- Sample answer: checked against an explicit plaintext oracle in the component
  test. This is not a protocol soundness check; a production OOD claim needs a
  proximity/claim check that this subset does not implement.
- Real oracle group: old, switched and star rows are local commitments; no real
  oracle group, SEND or answer provider.
- Real random masks/hiding: fixed test masks; no hiding or zero-knowledge claim.
- All stage counts, error accounting and production security parameters: not
  implemented.
- Random checks are conditional on drawn positions; the deterministic negatives
  above do not rely on sampling, but the symbol check itself is sampled.
- Single-shot is not claimed.
- Production wiring: `verify_request` unchanged; this is not an accepted proof
  format.

## Artifacts

- `tests/proximity-plan.txt`
- `tests/cs-stage.txt`
- `tests/base-stage.txt`
- `tests/proximity-session-unit.txt`
- `tests/proximity-session-real.txt` (local fixture path replaced with
  `<fixture>`; no measured value changed)
- `tests/regression-foldeq.txt`
- `tests/regression-column.txt`
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
