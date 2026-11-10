# bat-fold-r2 - 001b99c1 / 111f8ec7

Trace of the isolated algebraic/chain interface demonstration from a BAT group
row to the FOLD stage, without a group commitment. It supersedes the previous
trace, whose claim is void.

## Claim status

- This is not a same-commitment positive case. The original condition that the
  same group oracle root, prover witness and declarations be established and
  mutually consistent is **not completed** in this step.
- `group_main` and `group_mask` are plaintext fixture tables with no oracle
  root. The same-tag inner products establish only fixture arithmetic
  consistency; they do not show that a witness corresponds to a group
  commitment, and no declaration comes from a bound message.
- The BAT sample-class rows are plaintext inner-product fixture rows, not send
  receipts.
- The fresh-mask tree root commits only the fresh mask rows and their encoding
  tail. It is not a group oracle root. The root is registered in the verifier's
  runtime registry but not authenticated by openings.
- A witness replacement or mismatch after a fixed group root cannot be tested
  here because no group root exists. That negative is not applicable in this
  step and is listed as unclosed.
- The verified local interface progress kept from the previous round: the BAT
  and FOLD codes and layout are native and converted without a width adapter;
  the combined row enters the FOLD input row unchanged; the chain is moved from
  the BAT session into the FOLD-only entry with round/value continuity; and the
  FOLD output remains pending.

## What the step shows

| area | evidence |
| --- | --- |
| native codes and layout | shared BAT/FOLD codes, equal layout, non-kept members |
| unchanged handoff | main coefficients, mask coefficients and target equal |
| chain move | BAT session consumed; FOLD entry rounds/value equal the BAT exit values; start adds one round and each of `k` rounds adds one |
| verifier order | start after the bound row, rounds after epsilon, message before gamma |
| pending output | no next stage; a wrong BAT target completes all rounds and stays pending |
| wrong inputs | shape, code, layout, row, oracle, group, tree, root and forged runtime root rejected before epsilon |
| no group commitment | outcome flags report no group root, no witness bound to a group commitment, no send receipts and no bound-message declarations; only the fresh mask root is committed |

The arithmetic consistency flags (`combined_matches_witness` and
`fold_matches_witness`) use the plaintext fixture tables. They are not a
commitment check and are not evidence for the missing group commitment.

## Test coverage

- New unit tests: 9 passed, including
  `the_demonstration_has_no_group_commitment`.
- New integration tests: 2 passed.
- Engine regression: 291 passed.
- Ops regression: 98 passed.
- Workspace: 470 passed, 0 failed, 11 ignored.

| run | wall clock | VmHWM |
| --- | --- | --- |
| new unit | 0.04 s | 29,728 kB |
| new integration | 0.03 s | 30,616 kB |
| workspace | 1.37 s | 328,976 kB |

No proof bytes were produced; no performance gain, hiding or soundness bound is
claimed. The workspace run used the debug test profile.

## Fixture values

Fixture tag 7, as used by the integration run.

| item | value |
| --- | --- |
| committed fresh-mask root | `7467fade6982a92edc5900c08f0f2dd4325bb0c65dff577b3ee7a384427de54e` |
| start sum | `Fp3([6920283429447377251, 17809523116397550467, 2483515671979276613])` |
| BAT x | `Fp3([5190656392923401655, 18291762898753100460, 15963090726719551220])` |
| epsilon | `Fp3([13172321604147895412, 566987724245929467, 16222885632263852598])` |
| FOLD target and auxiliary value | `Fp3([14213439689004257719, 12143128358762237907, 5909419943386611493])` |
| BAT order | plain, plain, sample, sample, sample, generator |

The root is a fixture mask-tree commitment, not an authenticated asset or group
root.

## Validation scope

This step is a self-contained fixture component. No three-stage run and no model
asset read were made, so the default three-case set and the five-case set do not
apply; the shared three-stage path and seq512 logic are unchanged.

## Unclosed items

- Establishing a group oracle root and a witness bound to it; the group-root
  witness replacement/mismatch negative is not applicable in this step.
- Real SEND/G_u and real send receipts.
- Authenticated assets.
- Fresh-mask openings and proximity.
- Folded-tail derivation.
- A relation/proximity connection that can reject a pending FOLD output.
- Hiding and a security bound; the fixture source is public and deterministic.
- Single-shot semantics and production wiring; `verify_request` is unchanged.
- Kept forms are not part of this fixture.

## Artifacts

- `tests/bat-fold-unit.txt`
- `tests/bat-fold-integration.txt`
- `tests/regression-engine.txt`
- `tests/regression-ops.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The previous package `t002-s6b3g-review.tar.gz` sha256
`301d6de9fd94d910b1d41f4138dd47c8c4b8f7d2c11d25be3bf9cdd97f6b4aa3` is void and
retained unchanged.
