# proximity-session - 5446085c / cfdfd4b4

Trace of the isolated CS -> BASE proximity component.

## Change

Code commit `5446085c` ("Add an isolated CS-to-BASE proximity component"),
based on `cfdfd4b4`. The production verify entry is unchanged.

- New engine modules: `proximity` (plan and single-row handoff), `cs_stage`
  (claim registration), `base_stage` (BASE algebra).
- New isolated session: `proximity_session` with local commitments/openings.
- New real gpt2-seq16 component test.
- `ProximityOutcome::next_stage` returns the component marker
  `BaseComplete` only after the isolated checks pass. The earlier 2e/2f
  `FoldOutput::next_stage` remains independently fail-closed; it was not
  changed. This is a component result, not a production proof acceptance.

The earlier package `t002-s6b2g-review.tar.gz` sha256
`f28697df4a67484bf21a4112977afdc542431d494d62449c1f02f970f2222e71` is
retained unchanged and is superseded by this round. Earlier packages remain
retained unchanged as well.

## Proximity plan

`ProximityPlan` is verifier-side and independent of the proof. The isolated
subset has:

- input main/mask codes;
- exactly one switch stage and one BASE stage, in that order;
- each stage code, mask code, k/h, answer count, sample count and choice
  reference;
- adjacent code-length checks, mask-capacity checks, last-stage k=h=0 and
  positive sample counts;
- `ChoiceReference::VerifierPlan` only, so a proof cannot select choices.

Unimplemented or incompatible constraints fail closed. The plan is a
component plan, not a production security parameter set.

## CS component

The session follows the fixed order:

1. commit the old main and mask rows and the switched main/mask messages;
2. absorb the switched roots, draw the sample seed and absorb the fixed sample
   answer;
3. draw the old positions, then draw theta;
4. authenticate the old row openings at the drawn positions;
5. register the original claim and one generator-position claim per position,
   with mask positions modulo the mask code's length;
6. combine the registered claims by powers of theta.

The switched message stores the old main/mask messages and their old random
tails; new tails are zero in this subset. `register_switch` only registers
claims. It does not prove that the new message is a correct re-encoding, and
the session does not call a claim "verified" because it was registered.

## BASE component

`base_stage::prepare` computes target* and the M'/R' plaintext rows. The session:

1. fixes the star main/mask commitments and target*, then absorbs them;
2. draws lambda from the same chain;
3. computes and absorbs all M' then all R' values (uncommitted);
4. draws positions from the same chain;
5. authenticates the final switched-row and star openings;
6. checks `<c, M'> + <d, mask M'> = target* + lambda * target`;
7. checks each sampled symbol of Enc(M',R') against the star symbol plus
   lambda times the final-row symbol, using mask positions for mask rows.

No M'/R' commitment is created. The final `SingleRows` handoff is the checked
no-kept-row form (`base_ready`) and is not an arbitrary CS group conversion.

## Source and fixture semantics

The reference semantics used for the isolated equations are the approved CS
and BASE semantics: the switched message carries old messages/tails, old
positions become generator claims, theta batches the group claim, BASE sends
M'/R' in the clear, and the final check is the linear/symbol form above.

Real/fixture classification:

| item | source |
| --- | --- |
| old main/mask rows | real gpt2 asset opening values in the integration test; local fixture for the unit test |
| old row/g/star commitments | local component commitments, not a real G_u/AET group |
| sample seed and answer | chain seed plus fixed answer; no real SEND/answer provider |
| G_u group/sample, FOLD input | not used here; the earlier 2e/2f G_u/FOLD data remains fixture |
| star main/mask rows | fixed local rows |
| theta/lambda/positions | same local test chain; fixed seed test only |
| M'/R' | plaintext local vectors, uncommitted |

The component does not claim a real asset-to-proximity closure, a real oracle
group, hiding, zero knowledge, soundness, or a security bound.

## Negative coverage

| negative | layer |
| --- | --- |
| plan has missing/extra/swapped stage | plan |
| wrong code/mask length adjacency or no sample | plan |
| wrong input code vs rows | session plan/row binding |
| wrong target with valid commitments | BASE linear check |
| wrong BASE linear target* | BASE linear check |
| wrong encoded symbol at a sampled position | BASE symbol check |
| wrong CS code capacity | CS registration |
| truncated plan | plan |
| repeated deterministic runs | same component result |

The real test covers a wrong target, truncated plan and input-code mismatch;
unit tests cover the plan edges and CS/BASE checks. The tests do not claim
that any random sample must reject every tamper; detection is conditional on
the sampled positions. The component result is local and does not upgrade to a
complete proof.

## Tests and measurements

- Proximity plan: 2 passed.
- CS stage: 2 passed.
- BASE stage: 1 passed.
- Proximity session unit: 2 passed.
- Regressions: foldeq 38, edge 2, column 9.
- Real gpt2-seq16 component: 1 passed.
- Workspace: 429 passed, 0 failed, 7 ignored.
- Real component run: session 1 ms; whole test 55.62 s; timed process 55.77 s;
  VmHWM 4,235,924 kB.
- No proof bytes were produced; no performance gain, hiding or soundness
  bound is claimed.

## Validation scope

Only `gpt2-seq16` is used. The rows are derived from the authenticated asset
opening values, not from a real G_u/AET group. No five-case three-stage run was
made. The production verify entry remains unchanged.

## Unclosed items

- Kept-row re-send/CS path: not exercised in this minimal subset; the
  `SingleRows` handoff is no-kept only.
- Full proximity test after CS: registered claims are not proved by a complete
  later proximity test; only the isolated BASE overlap is checked.
- Real oracle group: old/g/star rows are local commitments; no real G_u/AET
  group or sample provider.
- Real random masks/hiding: fixed test masks and a fixed sample answer; no
  hiding or ZK claim.
- All stage counts/error accounting and the production security parameters:
  not implemented.
- The earlier 2e/2f `FoldOutput::next_stage` remains fail-closed; this
  component's `next_stage` is not a production stage.
- Random checks are conditional on the actual drawn positions; no per-run
  rejection guarantee or security bound is claimed.
- Single-shot: not claimed.
- Production wiring: `verify_request` unchanged; this is not an accepted proof
  format.
- Commitment/opening coverage: local roots and openings only; no real asset to
  proximity closure.

## Artifacts

- `tests/proximity-plan.txt`
- `tests/cs-stage.txt`
- `tests/base-stage.txt`
- `tests/proximity-session-unit.txt`
- `tests/proximity-session-real.txt` (local fixture path replaced with
  `<fixture>`; no value changed)
- `tests/regression-foldeq.txt`
- `tests/regression-edge.txt`
- `tests/regression-column.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
