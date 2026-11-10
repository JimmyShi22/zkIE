# fold-cs - 97962523 / d2f2973c

Trace of the isolated fixture code-switch continuation that connects the FOLD
output through a switch into BASE on the same transcript. The previous package
`t002-s6b3k-review.tar.gz` sha256
`d31bdfad40f80a4db94fac8ecdc0e72f14c03a97db14cc1f89f5a8ba6df79b25` is
superseded and retained unchanged.

## Basis and scope

- Based on `97962523`; the previous state is `d2f2973c`.
- Isolated fixture only; the production verification entry is unchanged.
- This is a conditional local experiment, not the reference proof, not a
  complete proximity test, not zero knowledge and not a production pass. The
  fixture is public, the fresh rows are plaintext and the star row is fixed;
  there is no hiding.

## Input source and conversions

- The code-switch session consumes the same FOLD chain, declaration, layout,
  original member identities, transcript challenges and FOLD identity. It does
  not create a second plan and does not rebuild unrelated rows.
- The old columns are authenticated at the code-switch positions from the
  original member commitments and the FOLD fresh mask tree, using the same
  folded-column relation as the BASE check. The relation remains sampled at the
  drawn positions; the original commitments are not described as completely
  bound.
- The new group declaration is derived by the verifier from its own
  authenticated old symbols and the transcript theta; no prover-supplied
  target is accepted for it.

## New oracles and output

- The two switched-row commitments (main and mask) are fixed before the SEND
  seed and bound in the message chain through the FOLD identity.
- The switched mask message carries the FOLD fresh row after the old mask and
  its tail. The combined mask claim carries the FOLD fresh coefficients at the
  same offset, so the original FOLD target term is preserved without changing
  the target.
- The final single-row view is derived from the combined group row and the
  switch layout. BASE consumes that view and opens the new switch commitments;
  the old FOLD objects are not relabelled as the switch output.
- The switch claims are pending. The code-switch stage does not prove the new
  group declaration true on the spot and no proximity test runs here.

## Order and transcript

1. The two new switched roots and the FOLD identity are absorbed (round after
   the FOLD exit).
2. The SEND seed is drawn.
3. The complete answer is received and absorbed.
4. One challenge round absorbs the position-round marker, draws the old
   positions and theta.
5. The old columns are authenticated from the original commitments and the
   new group declaration is registered and combined with theta.
6. BASE fixes its star commitments and target*, draws lambda, receives M'/R'
   and draws its own positions, then checks the openings.

No new message is inserted into an already frozen challenge window. The
recorded rounds prove the order: declaration = FOLD exit + 1, positions =
answer + 1, BASE commitments = positions + 1, BASE rows = commitments + 2, and
the final round equals the BASE rows round.

## Acceptance

Positive path, item by item:

- The FOLD identity in the declaration equals the identity of the same FOLD
  output.
- The two new roots are nonzero and their counts equal the switch code lengths.
- The old columns open against the original member roots and the FOLD fresh
  root; the fold relation stays sampled.
- The verifier-derived claims equal the prover-side registration over the
  plaintext old rows (fixture consistency check).
- The combined mask claim covers the FOLD fresh coefficients at the switched
  fresh offset.
- The BASE linear relation and the symbol checks over the opened switch rows
  hold.
- The result remains pending; the switch claims are pending and the fold
  binding is not complete.

Negative coverage and the rejecting stage:

- Replaced declaration root: rejected when the declaration is received.
- Replaced original member opening: rejected during old-column authentication.
- Replaced switched opening: rejected during BASE opening verification.
- Replaced theta (prover-side registration differs): rejected by the BASE
  target/linear path.
- Missing stage (seed before declaration, answer before seed, positions before
  answer): rejected as an invalid session state.
- Unsupported switch code length: rejected as an unsupported primitive.
- Kept layout: rejected as an invalid configuration.
- Different code lengths are exercised by the positive path (old codes of
  length 8 with depth 4; switch codes of length 20 with depth 5). No kept
  member is supported.
- The fixed wrong upstream answer sample is unchanged and is not claimed to be
  a general rejection guarantee.

## Tests and measurements

| run | result | wall clock | VmHWM |
| --- | --- | --- | --- |
| BAT-to-FOLD and code-switch unit | 19 passed | 0.05 s | 29,476 kB |
| BAT-to-FOLD integration | 5 passed | 1.16 s | 400,452 kB |
| Engine package | 234 unit + 57 integration passed | 0.63 s | 200,732 kB |
| Ops package | 92 unit + 23 integration passed | 0.20 s | 193,108 kB |
| Workspace | 490 passed, 0 failed, 11 ignored | 1.77 s | 387,596 kB |

The workspace run used the debug test profile. No proof bytes were produced;
no performance gain, hiding, zero-knowledge or soundness bound is claimed.

## Fixture values

| item | value |
| --- | --- |
| FOLD identity | `0023131013ab74b868374530b8670d4d72ba7d7d8dec0336768c422dba253faa` |
| switched main root | `b413e76da82bbad6668f996bc18f2726acde0f9a44246ed4e15e4dcbd2c2ab0f` |
| switched mask root | `4fe47bdfbd9b71866c05fa1a081eb190ceea5ec6d438e74348076332be901e00` |
| theta | `Fp3([12337188300315816533, 2744956256541347282, 12204341619432039098])` |
| lambda | `Fp3([11909099744949334567, 12417158858109033649, 17972193412664376142])` |
| target* | `Fp3([4001372689779521232, 17323326352227369308, 1754963609652451628])` |
| code-switch positions | `[4, 9]` |
| BASE positions | `[25, 18]` |
| registered claims | 5 (one original plus two per position) |

The positive fixture uses tag 50 and seed `fold cs base`.

## Unclosed items

- Production-trusted sources for the original, star and switch commitments.
- A complete (not sampled) fold binding and a complete proximity test; the
  switch claims remain pending.
- Hiding and zero knowledge; the fresh rows and the star row are public.
- Real assets, folded tails and single-shot semantics.
- Reference-protocol equivalence and production wiring; the production
  verification entry is unchanged.

## Artifacts

- `tests/fold-cs-unit.txt`
- `tests/fold-cs-positive.txt`
- `tests/bat-fold-integration.txt`
- `tests/regression-engine.txt`
- `tests/regression-ops.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
