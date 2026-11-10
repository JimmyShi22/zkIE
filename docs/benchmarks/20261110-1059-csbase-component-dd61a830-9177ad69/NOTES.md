# csbase-component - dd61a830 / 9177ad69

Trace of the CS/BASE local algebraic component after the expert review rework.

## Change

Code commit `dd61a830` ("Position the proximity session as a local CS/BASE
algebraic component"), based on `9177ad69`. The production verify entry is
unchanged.

- The sample comparison is now an auxiliary test-side check. `SampleOracle` is
  constructed from fixed fixture data, not from the prover channel.
- The result marker is `ProximityComponent::ComponentComplete`; it covers only
  the local algebraic checks and does not prove the pending CS claims.
- The outcome speaks only of local checks; the plan, CS and BASE module docs
  state the implemented-subset boundary and do not call it iopp_fault
  equivalent.
- Wrong-commitment and valid-commitment-wrong-claim negatives are separate
  tests and assert the error of the check they reach.
- The plan binds the fold flag and rejects fold configurations (including
  non-final k >= 1) as unsupported.

The earlier package `t002-s6b3a-r2-review.tar.gz` sha256
`9674e06cc671bb61b5073203b796e4e3d5ab462c0f87fd1b46cf62e686a93abc` is void and
retained unchanged. The initial package `t002-s6b3a-review.tar.gz` sha256
`17e862d1863fe18b53cafe41366d5b5dceda28a1f3810b6755c03118e20606ca` remains void
and retained. The `2g`, `2f` and `2e` packages remain retained unchanged.

## Positioning and coverage

This is a local algebraic component, not a complete BASE, IOPP or proof
acceptance. It checks:

- the CS pending-claim registration from authenticated old openings;
- the switch-to-BASE shape handoff;
- the local BASE linear relation;
- the local BASE sampled symbol relations;
- opening authentication and commitment counts.

It does not check the pending CS claims with a later proximity test, does not
connect SEND, does not implement kept rows, fresh mask rows, a second mask row
or row folding, and does not produce proof bytes.

## Sample comparison

The declared sample answer comes from `ProximityProverChannel::sample_answer`.
The expected value comes from a separate test-side `SampleOracle`:

- the fixed fixture rows and the switch code are used to build the switched
  message before the prover object is constructed;
- the oracle stores that code and message/tail and evaluates the OOD sample at
  the seed drawn by the verifier;
- it never reads or calls the prover channel;
- it is deterministic for the fixed fixture data;
- it is not a fixed constant and not a verifier trust source.

The comparison is an auxiliary component check. There is no SEND sample
declaration, no answer provider and no later protocol check of the OOD claim in
this subset; that remains unclosed. The component does not present this
comparison as verification of a protocol sample.

## Data flow

| step | input | operation | output |
| --- | --- | --- | --- |
| original claim | statement claim, mask claim, target | place at new main/mask message coordinates; theta^0 | original `SwitchClaim` |
| main position | authenticated old main opening at p | target = authenticated symbol; main coefficient vector from the old code generator at p, with tail coefficients after the old ell | main `SwitchClaim` |
| mask position | authenticated old mask opening at p mod mask_n | target = authenticated symbol; mask coefficient vector from the mask code generator at p mod mask_n | mask `SwitchClaim` |
| theta combination | original and per-position claims | sum theta^i * claim_i over the fixed claim order | combined main, mask and target |
| local BASE | combined claim, target*, M'/R' | absorb all M' then all R'; linear check; sampled symbol checks against authenticated switched and star openings | local BASE checks |

The original claim, main-position claims and mask-position claims stay separate
through registration; main and mask targets are never summed into one target.

## Plan boundary

The plan checks are the implemented subset, not an iopp_fault-equivalent check.
The k = 0, ell = 6 fixture is an algebraic fixture and is not claimed to satisfy
iopp_fault. The non-final folding k >= 1 constraint is checked before folding is
rejected as unsupported.

Real IOPP still needs a real FOLD output and witness source, every applicable
plan constraint, and a real SEND sample declaration with its follow-up checks.
These stay on the project unclosed list and are not removed by this local
result.

## Negative coverage

Class A: a wrong commitment object.

| negative | check reached | detection |
| --- | --- | --- |
| valid commitment with a different row count | verifier commitment count check | deterministic, `StatementMismatch` |
| same-shape valid switched commitment with old openings | opening authentication | deterministic, `InvalidProof` |
| old opening value changed | opening authentication | deterministic |
| old opening path changed | opening authentication | deterministic |
| old opening index changed | position/index check | deterministic |

Class B: valid commitments, wrong claim or prover message.

| negative | check reached | detection |
| --- | --- | --- |
| wrong statement target | local BASE linear check | deterministic, `InvalidProof` |
| wrong statement claim | local BASE linear check | deterministic, `InvalidProof` |
| wrong target* message | absorbed target* equality and linear check | deterministic, `StatementMismatch` |
| wrong M' value | local BASE linear check | deterministic, `InvalidProof` |
| wrong sample answer | auxiliary test-side sample comparison | deterministic, `InvalidProof` |
| consistently replaced switched message | local BASE symbol check | every position differs in this witness; sampled in general |

Other negatives: plan/statement shape mismatch, missing or erroring channel,
public booleans without the private flag, truncated plan and fixed challenge
order. No test relies on a random sample missing a tamper; the consistently
replaced message differs at every position, so every drawn BASE position
detects it. The symbol check itself remains a sampled check.

## Tests and measurements

- Proximity plan: 2 passed.
- CS stage: 2 passed.
- BASE stage: 1 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 component: 1 passed.
- Workspace: 439 passed, 0 failed, 7 ignored.
- Real component run: session 0 ms (reported precision); whole test 55.00 s;
  timed process wall clock 55.15 s; VmHWM 4,232,808 kB.
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

- Pending CS claims: registered but not proved by a later proximity test. A
  consistently re-committed CS output is rejected only by the sampled local BASE
  symbol check, not by CS algebra alone.
- Kept rows, fresh mask rows and a second mask row: unsupported and rejected;
  the handoff is no-kept, one-mask only.
- Row folding: unsupported and rejected. The non-final k >= 1 constraint is
  enforced before the unsupported rejection, but no folding implementation
  exists.
- Real FOLD output and witness source: not connected.
- All applicable IOPP plan constraints: not implemented; the current checks are
  the implemented subset only.
- Real SEND sample declaration and its follow-up checks: not connected. The
  sample comparison here is a test-side auxiliary check, not a verifier trust
  source.
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
