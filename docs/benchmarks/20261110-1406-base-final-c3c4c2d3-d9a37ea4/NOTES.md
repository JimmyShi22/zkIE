# base-final - c3c4c2d3 / d9a37ea4

Trace of the fixture BASE final-state wiring on the FOLD continuation chain.

## Claim status

- The local BASE final-state component now runs on the same verified FOLD chain
  produced by the previous step. The claim, target and FOLD identity come from
  the same FOLD output; the old main and old mask rows are the FOLD prover's
  end rows; the fresh row is the FOLD prover's folded fresh message; the star
  rows are the BASE prover's own fixture rows.
- Five row commitments and target* are fixed before lambda; M', R' and the
  fresh combined row are fixed before the position draw; every opening is
  authenticated; the extended linear check and the encoded-symbol checks are
  recorded. A missing precondition fails closed.
- The result stays pending. This is not a complete BASE, IOPP or proof
  acceptance and not a valid FOLD-to-BASE handoff. The FOLD fresh root does not
  authenticate the folded fresh row; the BASE fixture commitment does. A single
  equality at one fixed lambda does not show that every wrong declaration is
  rejected and gives no IOPP security bound; wrong declarations can cancel
  through aggregation.

## Source chain

| item | source |
| --- | --- |
| claim, mask claim, target | the same FOLD output through the FOLD handoff identity check |
| FOLD identity | canonical digest of the handoff shape, identity, challenges, claims and target; absorbed before lambda |
| old main row | the FOLD prover's end main message |
| old mask row | the FOLD prover's end mask message |
| fresh row | the FOLD prover's end fresh message; coefficients are the FOLD output fresh coefficients |
| tails | deterministic fixture tails for the old, star and fresh rows |
| star main/mask rows | the BASE prover's fixture rows |
| row commitments | fixture expectations independently committed from the same rows; roots are not SEND, asset or FOLD roots |

## Order and algebra

Order: FOLD output and chain transfer -> five commitments and target* -> lambda
-> M', R' and the fresh combined row -> positions -> authenticated openings ->
linear and symbol checks.

The fresh row uses a zero star row, so its combined message is lambda times the
fresh message. The linear check is the two-row BASE relation extended by the
fresh row:

`<claim, M'> + <mask claim, mask M'> + <fresh claim, fresh M'> = target* + lambda * target`

This is the same FOLD target identity split into old and fresh components, not a
rewritten target. The symbol checks cover the main, mask and fresh encodings
against the authenticated openings. A fixed target* shift is rejected at the
target/linear layer.

## Acceptance and negatives

| case | result |
| --- | --- |
| correct fixture, full wiring | linear, symbol and opening checks pass; pending; not complete BASE |
| fixed target* shift | rejected at the target/linear layer; fixed lambda, target* and positions are recorded below |
| changed authenticated opening | rejected at the opening check |
| changed encoding tail | symbol layer rejects; the linear check does not use tails |
| skipped precondition | session stage fails closed |
| swapped FOLD identity or commitment | expectation check rejects |

The negative cases drive the `BaseFinalProver`/`BaseFinalSession` stage machine,
not only the helper functions. The fixed inconsistent sample is crafted to fail
at the target layer; it is not a general wrong-declaration rejection claim.

## Tests and measurements

- BASE final unit: 5 passed.
- BAT-to-FOLD unit: 13 passed.
- BAT-to-FOLD integration: 4 passed.
- Engine regression: 291 passed.
- Ops regression: 114 passed.
- Workspace: 488 passed, 0 failed, 11 ignored.

| run | wall clock | VmHWM |
| --- | --- | --- |
| BASE final unit | 0.04 s | 29,476 kB |
| BAT-to-FOLD unit | 0.04 s | 29,760 kB |
| BAT-to-FOLD integration | 0.04 s | 30,576 kB |
| workspace | 1.49 s | 334,448 kB |

No proof bytes were produced; no performance gain, hiding or soundness bound is
claimed. The workspace run used the debug test profile.

## Fixture values

Fixture tag 9 for the passing BASE integration run; tag 10 for the fixed
target-shift sample.

| item | value |
| --- | --- |
| FOLD identity | `2f0cd6024e6fd269d3c80c135e2908f64f9299d5fc8e6f28a7c6c7f9d30b3fbe` |
| old main root | `f1e4766e8e49d8ec777eb29446997f8b77348b682a7fe4419c9bc0e49b5d2124` |
| old mask root | `754fccca6b0886e88c88dbd5bea472acc0f4e8b60a8b8b126c40e60a863338ef` |
| star root | `b7161ab94b2a6bc48276b48601019e6d00204776e2ce3845976d015fdb402158` |
| star mask root | `8576851c98802e899876799ada67e36a919db161d91e31c08b6554a48aa28a60` |
| fresh row root | `bcccfdddb6140e579f668bdf2d9a4fd30a0c76fb69d01edb4ac87ede26a2dc6b` |
| passing lambda | `Fp3([17123776299782491163, 11529012978629097546, 14864888526832527820])` |
| passing target* | `Fp3([16868277434214652845, 13470965661350279654, 12186704821915069562])` |
| passing positions | `[0, 11]` |
| rejected sample lambda | `Fp3([11603395840025481809, 16880675433117640569, 5089752682697088842])` |
| rejected sample target* | `Fp3([17652280374797064835, 8161331915830456396, 7358458655439356460])` |
| rejected sample positions | `[0, 3]` |
| rejection layer | target/linear |

The roots are fixture BASE commitments, not SEND, asset or FOLD roots.

## Boundary classes

This step confirms the proof relation and the challenge order within the
fixture. It does not relax the trusted-source, commitment-object,
challenge-order or public/secret boundaries: the FOLD claim/target/identity and
old rows come from the same FOLD continuation; the fresh row is an existing FOLD
output row committed by the BASE fixture; lambda and the positions use the same
chain draws as the local component. Sampling is unchanged. The fixed pads from
the previous step remain public and non-hiding; the answer order remains by
answer number, then member order.

## Unclosed items

- Real assets, full proximity and authenticated SEND/G_u sources.
- A complete BASE/IOPP result and a valid FOLD-to-BASE handoff; the fresh row is
  not authenticated by the FOLD fresh root.
- A deterministic wrong-declaration guarantee; the single fixed equality can
  cancel through aggregation.
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
