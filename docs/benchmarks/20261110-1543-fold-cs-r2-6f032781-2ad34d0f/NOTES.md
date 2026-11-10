# fold-cs-r2 - 6f032781 / 2ad34d0f

Trace of the code-switch rework that registers a fixture SEND sample and the
fresh-column correspondence in the same theta-combined group row. The previous
package `t002-s6b3l-review.tar.gz` sha256
`0161dbf58c08cf0910a839af15d5c77fe8d21cf068775142192a0479f9798471` is void and
retained unchanged.

## Basis and scope

- Based on `6f032781`; the previous state is `2ad34d0f`.
- Isolated fixture only; the production verification entry is unchanged.
- Conditional local experiment, not a proof, not complete proximity, not zero
  knowledge and not a production pass. The fixture is public, the fresh rows are
  plaintext and the star row is fixed; there is no hiding.

## Fixture SEND sample registration

- The switch oracle is one runtime tree carrying the switched main row and the
  switched mask row in one part. Its shape fixes one main row, one mask row and
  one answer; the root and shape are fixed before the seed.
- The switched mask row carries the old mask row, its tail, the FOLD fresh mask
  row and the FOLD fresh tail, followed by one public deterministic pad. The pad
  is reserved in the committed message before the seed.
- The degenerate SEND phase draws a real sample seed with the shape's cube
  `z` values and a nonzero `beta`; the seed length and nonzero beta are checked
  and the shared-seed validation runs. The answer is received and absorbed.
- The sample claim is built on the same oracle shape and commitment and is
  appended to the same theta-combined group row as the switch claims. The claim
  is pending; no proximity test runs here. Unsupported answer counts and shapes
  fail closed.
- This is not a production SEND receipt and not the reference standard
  SEND-to-code-switch wiring; the result reports `send_standard_wiring=false`.

## Fresh-column correspondence

- At the drawn positions the verifier opens the FOLD fresh mask tree and folds
  the opened columns exactly as the fold relation does.
- For each drawn fresh column, one pending claim maps the authenticated fresh
  column to the switched mask row's fresh block and tail at the fixed offset;
  the coefficients are derived by the verifier and the target is the
  authenticated fresh column value. The prover's plaintext fresh row is not a
  verifier source for this mapping.
- These claims are appended to the same theta combination; the registration
  does not prove them true. The result reports the correspondence as registered
  and sampled, with the fold binding still incomplete.

## Order and transcript

1. The switch root and FOLD identity are absorbed (round after the FOLD exit).
2. The sample seed is drawn.
3. The answer is received and absorbed.
4. One challenge round absorbs the position marker and draws the old positions
   and theta.
5. The old columns are authenticated; the switch claims, fresh correspondence
   claims and sample claim are registered and combined with theta.
6. BASE fixes its star commitments and target*, draws lambda, receives M'/R',
   draws its own positions, and checks the openings.

No new message is inserted into an already frozen challenge window. Recorded
rounds: declaration = FOLD exit + 1, positions = answer + 1, BASE commitments =
positions + 1, BASE rows = commitments + 2, final = BASE rows.

## Acceptance and negatives

Positive path: the registration count is two plus three per drawn position (one
original, two per old position, one fresh correspondence per position, one
sample); the verifier registration equals the prover-side registration; the BASE
linear and symbol checks hold; the result is pending.

Adversarial samples:

- Only the SEND answer is changed (all bindings unchanged): the registration
  stage completes and the altered relation is rejected by the later BASE check,
  not by an early format check.
- Only the committed switch fresh block is changed while the original FOLD
  fresh opening and the correspondence targets stay unchanged: the registration
  stage completes and the altered relation is rejected by the later BASE check.
- Replaced declaration root, member opening, switched opening or theta: rejected
  in the named stage (declaration, authentication, opening verification, or the
  BASE target/linear check).
- Missing stage (seed before declaration, answer before seed, positions before
  answer), unsupported switch code length and kept layout: rejected.
- Different code lengths are exercised by the positive path (old codes of length
  8 with depth 4; switch codes of length 20 with depth 5); kept is unsupported.
- The fixed wrong upstream answer sample is unchanged and is not a general
  rejection guarantee.

## Tests and measurements

| run | result | wall clock | VmHWM |
| --- | --- | --- | --- |
| BAT-to-FOLD and code-switch unit | 20 passed | 0.05 s | 29,760 kB |
| BAT-to-FOLD integration | 5 passed | 0.70 s | 364,672 kB |
| Engine package | 234 unit + 57 integration passed | 0.63 s | 202,328 kB |
| Ops package | 93 unit + 23 integration passed | 0.21 s | 193,152 kB |
| Workspace | 491 passed, 0 failed, 11 ignored | 1.52 s | 331,224 kB |

The workspace run used the debug test profile. No proof bytes were produced; no
performance gain, hiding, zero knowledge or soundness bound is claimed.

## Fixture values

| item | value |
| --- | --- |
| FOLD identity | `0023131013ab74b868374530b8670d4d72ba7d7d8dec0336768c422dba253faa` |
| switch root | `e01147e5ac2c9b820fde6ae06d69b79e1d6c4309d60c7df10361bfca159695d6` |
| theta | `Fp3([16120239178938937620, 13650324339513277316, 15815150777425288769])` |
| lambda | `Fp3([11898693514605470438, 15999598354947106097, 14143747956050975823])` |
| target* | `Fp3([7127459302351547731, 2457062071420164091, 12605833891154970583])` |
| code-switch positions | `[12, 11]` |
| BASE positions | `[13, 29]` |
| registered claims | 8 (two plus three per position) |
| sample answer | `Fp3([2298245731154766907, 4048333599279767746, 3816564099335330975])` |

The positive fixture uses tag 50 and seed `fold cs base`.

## Unclosed items

- Production-trusted sources for the original, star and switch commitments, and
  a real SEND receipt.
- A complete (not sampled) fold binding and a complete proximity test; the
  switch claims, fresh correspondence and sample claim remain pending.
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
