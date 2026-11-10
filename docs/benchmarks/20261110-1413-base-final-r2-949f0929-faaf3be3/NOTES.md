# base-final-r2 - 949f0929 / faaf3be3

Trace of the fixture BASE final-state wiring rework: root-binding status,
upstream wrong-declaration rejection, and fail-closed verification semantics.
The previous package `t002-s6b3j-review.tar.gz` sha256
`2d6fd7687c6875eafd0096aafe7c250239a4d669fca6137ec13e49ac578b09e0` is void and
retained unchanged.

## Root-binding status (F1)

- The BASE creates a new fixture fresh-row root, rebuilds the old main/mask
  roots from the same prover copies, and uses a same-prover expectation with a
  second independent commit. These roots are approved only for this local
  fixture experiment. They are not a production mechanism.
- **Root binding is not completed.** The fresh-row root is not authenticated by
  the FOLD fresh root, and the rebuilt old roots are not the reference
  commitment source. The passed result stays pending; `projection_only` and the
  unauthenticated fresh term remain explicit. No document or result claims
  completed root binding or reference equivalence.
- The reference path is FOLD -> CS -> final SingleRows -> BASE. The extra
  commitment and the zero-star third row implemented in this fixture are not
  the reference protocol and are not an equivalence proof. Later wiring must use
  a reference-verifiable commitment source and the reference folding/conversion
  relation, not this fixture extension.

## Upstream wrong-declaration sample (F2)

- A well-formed wrong SEND answer is produced upstream and carried through the
  same chain: wrong SEND answer -> BAT sample claim target -> BAT combined
  target -> FOLD output target -> BASE target/linear check.
- The verification entry consumes the failed BASE check and returns a typed
  rejection at the target/linear layer; the exact BASE lambda, target* and
  positions are recorded below. The FOLD identity binds the upstream target and
  the same FOLD continuation. This is one fixed sample; it is not a claim that
  every wrong declaration is rejected. Wrong declarations can cancel through
  aggregation and the single-equality boundary remains.

## Failure semantics (F3)

- The verification entry chooses the fail-closed path. `BaseFinalSession::finish`
  returns an explicit `BaseFinalCheck::Passed` or `BaseFinalCheck::Rejected`;
  the run entry consumes the rejection and returns
  `Err(BaseFinalRunError::Rejected)` with the layer and fixed conditions. It
  does not return `Ok` with only diagnostic fields for a failed check.
- A passed local check remains pending and is not a complete BASE, IOPP result
  or valid handoff.

## Tests and measurements

- BASE final unit: 5 passed.
- BAT-to-FOLD unit: 14 passed.
- BAT-to-FOLD integration: 5 passed.
- Engine regression: 291 passed.
- Ops regression: 115 passed.
- Workspace: 490 passed, 0 failed, 11 ignored.

| run | wall clock | VmHWM |
| --- | --- | --- |
| BASE final unit | 0.03 s | 30,000 kB |
| BAT-to-FOLD unit | 0.04 s | 29,476 kB |
| BAT-to-FOLD integration | 0.04 s | 30,552 kB |
| workspace | 0.96 s | 200,328 kB |

No proof bytes were produced; no performance gain, hiding or soundness bound is
claimed. The workspace run used the debug test profile.

## Fixture values

| item | value |
| --- | --- |
| passing FOLD identity | `2f0cd6024e6fd269d3c80c135e2908f64f9299d5fc8e6f28a7c6c7f9d30b3fbe` |
| passing old main root | `f1e4766e8e49d8ec777eb29446997f8b77348b682a7fe4419c9bc0e49b5d2124` |
| passing old mask root | `754fccca6b0886e88c88dbd5bea472acc0f4e8b60a8b8b126c40e60a863338ef` |
| passing star root | `b7161ab94b2a6bc48276b48601019e6d00204776e2ce3845976d015fdb402158` |
| passing star mask root | `8576851c98802e899876799ada67e36a919db161d91e31c08b6554a48aa28a60` |
| passing fresh row root | `bcccfdddb6140e579f668bdf2d9a4fd30a0c76fb69d01edb4ac87ede26a2dc6b` |
| passing lambda | `Fp3([17123776299782491163, 11529012978629097546, 14864888526832527820])` |
| passing target* | `Fp3([16868277434214652845, 13470965661350279654, 12186704821915069562])` |
| passing positions | `[0, 11]` |
| target-shift sample lambda | `Fp3([11603395840025481809, 16880675433117640569, 5089752682697088842])` |
| target-shift sample target* | `Fp3([17652280374797064835, 8161331915830456396, 7358458655439356460])` |
| target-shift sample positions | `[0, 3]` |
| target-shift layer | `TargetLinear` |
| wrong upstream sample lambda | `Fp3([16081494546638240291, 8876820690161062943, 2832377814735984705])` |
| wrong upstream sample target* | `Fp3([7005279915648414391, 11225751876979421554, 348300791375232952])` |
| wrong upstream sample positions | `[2, 2]` |
| wrong upstream layer | `TargetLinear` |
| wrong upstream path | wrong SEND answer -> BAT target -> FOLD output -> BASE |

The wrong upstream sample uses fixture tag 11, chain seed
`fixture base wrong upstream`, and adds `Fp3::ONE` to the first received answer
in the fixed answer order. The target-shift sample uses fixture tag 10, chain
seed `fixture base target`, and adds `Fp3::ONE` to target*. Both are
reproducible fixed samples.

## Boundary classes

This round changes only the fixture verification semantics and the statement of
the root-binding boundary. Sampling, the challenge order, the public/secret
split and the trusted-source position are not relaxed. The new BASE roots are
explicitly outside the approved production commitment source and are not
claimed as reference-equivalent. Fixed pads remain public and non-hiding; the
multi-answer order remains by answer number, then member order.

## Unclosed items

- Completed root binding and a reference-verifiable commitment source for the
  BASE rows; the fresh-row root is not authenticated by the FOLD fresh root.
- A complete BASE/IOPP result and a valid FOLD-to-BASE handoff.
- Real assets, full proximity and authenticated SEND/G_u sources.
- A deterministic wrong-declaration guarantee; single-equality cancellation
  remains.
- Hiding, fresh openings, folded tails, single-shot semantics.
- Production wiring and performance acceptance; `verify_request` is unchanged.

## Artifacts

- `tests/base-final-unit.txt`
- `tests/bat-fold-unit.txt`
- `tests/bat-fold-integration.txt`
- `tests/regression-engine.txt`
- `tests/regression-ops.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
