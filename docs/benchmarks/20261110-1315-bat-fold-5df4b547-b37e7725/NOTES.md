# bat-fold - 5df4b547 / b37e7725

Trace of the isolated fixture BAT-to-FOLD component handoff.

## Change

Code commit `5df4b547` ("Hand a native fixture BAT row to the FOLD stage"),
based on `b37e7725`. The production verify entry is unchanged.

- One native fixture group is shared by BAT and FOLD: the BAT group base and
  mask codes are the FOLD shape codes, the member mask width is already a power
  of two, and the BAT layout is checked against the FOLD fixture layout. No
  width adapter is used and no field is copied from one.
- The real BAT combined row enters the FOLD input row unchanged: the main
  coefficients, mask coefficients and target are equal, and the run rejects a
  mismatch. There is no post-BAT target rewrite and no witness chosen after a
  challenge.
- The fixture group message tables are fixed before the run. Each BAT claim
  target is the true inner product of that claim's coefficients, mapped through
  BAT's own combination, with the same group tables. The sample-class rows are
  fixture claims, not authenticated send receipts.
- The fresh-mask witness uses a deterministic numbered test source. Its
  coefficients, encoding tail and mask rows are committed through the mask
  tree, the start message's root and sum come from the same witness, the
  fixture re-commits the stored rows and tails and checks the same root, and the
  runtime registry receives that root before the epsilon challenge. The prover
  sum equals the start sum. The values are public, reproducible and provide no
  hiding.
- Chain ownership is a move: the BAT session is consumed by `into_chain`, the
  same advanced chain is moved into the FOLD-only fixture entry, and the entry
  round count and value equal the BAT exit values. The start message adds one
  round and each of the `k` round messages adds one round. No pointer identity
  is used.
- The verifier advances only on the bound row and the checked start/round
  messages. The FOLD output is pending and has no next stage. A wrong BAT target
  still completes every round and stays pending; only a later relation or
  proximity check could reject it.
- The FOLD-only fixture entry is a constructor on the existing FOLD session, so
  this step does not add a second FOLD implementation. The plan's reader and
  binding are a valid but inert scaffold and are not a witness or an asset
  binding.

## Test coverage

| area | evidence |
| --- | --- |
| native group, codes and layout | shared codes, equal BAT/FOLD layout, non-kept members, committed witness root and sum |
| unchanged handoff | BAT row coefficients, mask and target equal the FOLD input row; positive run matches the witness |
| one fixture source | same tag and different chains keep the same start root/sum; both runs match the witness |
| frozen group | add, join and modified frozen input are rejected; a completed session cannot run again |
| wrong shape/code/layout/row | wrong `k`/`h`, wrong main code, wrong layout and a truncated row are rejected |
| wrong oracle/root | wrong id, group, tree, root and a forged runtime root are rejected before epsilon; the correct start then succeeds |
| order | an unadvanced chain, finish before rounds, a round before start, a repeated bind, a repeated start and an extra round are rejected |
| pending declaration | a shifted BAT target completes all rounds, differs from the prover's end value and remains pending |

## Tests and measurements

- New unit tests: 8 passed, 0 failed.
- New integration tests: 2 passed, 0 failed.
- Engine regression: 291 passed, 0 failed.
- Ops regression: 97 passed, 0 failed.
- Workspace: 469 passed, 0 failed, 11 ignored.

| run | wall clock | VmHWM |
| --- | --- | --- |
| new unit | 0.04 s | 30,004 kB |
| new integration | 0.03 s | 30,808 kB |
| workspace | 1.58 s | 348,340 kB |

No proof bytes were produced; no performance gain, hiding or soundness bound is
claimed. The workspace run used the debug test profile.

## Fixture values

Fixture tag 7, as used by the integration run.

| item | value |
| --- | --- |
| committed mask root | `7467fade6982a92edc5900c08f0f2dd4325bb0c65dff577b3ee7a384427de54e` |
| start sum | `Fp3([6920283429447377251, 17809523116397550467, 2483515671979276613])` |
| BAT x | `Fp3([5190656392923401655, 18291762898753100460, 15963090726719551220])` |
| epsilon | `Fp3([13172321604147895412, 566987724245929467, 16222885632263852598])` |
| FOLD target and auxiliary value | `Fp3([14213439689004257719, 12143128358762237907, 5909419943386611493])` |
| BAT order | plain, plain, sample, sample, sample, generator |

The root is a fixture mask-tree commitment, not an authenticated asset root.
The main group message tables are plaintext fixture data and are not bound to an
asset root. The fixture rebuilds the verifier statement per run; single-shot
mask leasing across runs is not modeled.

## Validation scope

This step is a self-contained fixture component. No three-stage run and no model
asset read were made, so the default three-case set and the five-case set do not
apply; the shared three-stage path and seq512 logic are unchanged. The new tests
use only the fixture.

## Unclosed items

- Real SEND/G_u and a real provider.
- Authenticated asset roots, fresh-mask openings and proximity.
- Hiding and a security bound; the fixture source is public and deterministic.
- Folded-tail derivation.
- A relation/proximity connection that can reject a pending FOLD output.
- Single-shot semantics and production wiring; `verify_request` is unchanged.
- Kept forms are not part of this fixture; the earlier isolated conversion tests
  remain the coverage for that mapping.

## Artifacts

- `tests/bat-fold-unit.txt`
- `tests/bat-fold-integration.txt`
- `tests/regression-engine.txt`
- `tests/regression-ops.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
