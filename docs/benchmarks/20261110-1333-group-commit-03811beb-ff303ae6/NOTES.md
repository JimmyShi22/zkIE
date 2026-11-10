# group-commit - 03811beb / ff303ae6

Trace of the fixture group commitment and the BAT-to-FOLD component handoff.

## Claim status

- Each member oracle now has its own fixture runtime-tree commitment, root and
  opening positions. The ordered root message is absorbed into the session
  chain before the BAT challenge, the drawn member openings are checked against
  those roots, and the same chain is then moved to BAT.
- The roots are fixture expectations checked against the fixture's own
  independent re-encoding and the openings. They are not SEND receipts, not
  G_u or asset roots, and not a production trust anchor.
- This is not a production verification and not a complete soundness result.
  The openings cover only the drawn positions; the FOLD output remains pending.
  Building the expected roots and the prover witness from the same deterministic
  fixture values is construction sharing, not a verifier commitment proof.
- The member roots are kept separate; the ordered list of roots is not called
  one group root, and the chain transcript value after the root message is not
  a group commitment root. If a caller replaces both the witness and the
  fixture expectation, no independent trusted binding exists in this fixture;
  that remains unclosed.

## Group source table

| item | source and layout |
| --- | --- |
| members | three oracles; each main and mask code is `log_n = 4, ell = 8, r = 1`; two main rows and two mask rows |
| member messages | deterministic fixture values per tag/oracle; main and mask rows of the code's `ell` entries |
| member tails | deterministic fixture values, `r` entries per row; encoded with the row and not part of the FOLD relation view |
| member trees | one part per member; `main_symbols = main_rows * 3`, `mask_symbols = mask_rows * 3`, `log_n = 4`, `log_n_zk = 4`, 16 columns; each leaf holds the main symbols, then the mask symbols, then a 40-byte salt |
| pad | none; the BAT sample-class rows are fixture claims, not SEND receipts, and no reserved sample-pad slot is read |
| roots | one root per member from its runtime tree; tree numbers 10, 11 and 12, leaf class 4 |

The group's base and mask codes are the FOLD shape's codes, the member mask
width is a power of two, and the BAT layout is the FOLD fixture layout. The BAT
claim targets are computed by BAT's own combination against the group tables
reconstructed from the committed member messages. The sample rows are plaintext
fixture inner products; no declaration is labelled as derived from
protocol-bound messages.

## Comparison semantics

- The fixture's expected roots and the prover witnesses come from the same
  deterministic values; that is construction sharing, not a verifier proof.
  The verifier checks the received root, plan and codes against the fixture
  expectation, then verifies each member opening against the received root.
- Tests independently re-encode every member message and tail, re-commit the
  tree, compare the root with the member plan, construct the drawn openings,
  compare the opened symbols with the member layout, and check the group tables
  at the BAT offsets.
- A root fixed and then a different witness: a second root message is rejected
  by the session stage; a root from another witness or a changed root/plan
  entry is rejected by the expectation check; openings from another witness
  fail the opening check against the fixed root.
- If both the witness and the fixture expectation are replaced, no independent
  trusted binding exists in this fixture. This is listed as unclosed, not
  claimed as an immediate rejection.

## FOLD row source table

| item | source |
| --- | --- |
| main coefficients | BAT combined main coefficients |
| old-mask coefficients | BAT combined mask coefficients |
| kept forms | none in this fixture |
| main messages | group table reconstructed from the committed member main messages at the BAT layout offsets |
| old-mask messages | group table reconstructed from the committed member mask messages at the BAT layout offsets |
| fresh mask | deterministic fresh-mask witness committed under its own mask tree; root and sum fixed at the FOLD start after BAT |
| tails | committed member tails; not part of the FOLD relation view |

The FOLD verifier consumes the fixed row and the prover messages; it does not
re-open the member roots, which the group session checks before `x`. The single
G_u witness from the SEND fixture is not applied to this group: each member has
its own shape-checked fixture commitment. Unsupported plans and shapes fail
closed.

## Order and negative layering

Order: ordered group roots (seal round) and opening positions (positions round),
then the member openings are checked, the chain moves to BAT, BAT draws `x`, the
chain moves to the FOLD-only entry, the fresh root and sum enter the start
message, epsilon is drawn, each round message precedes its gamma, and the FOLD
output stays pending.

| layer | negative | rejection source |
| --- | --- | --- |
| opening | changed symbol, salt or path; changed position list | opening check against the fixed root |
| witness | openings from another commitment | opening check against the fixed root |
| declaration | wrong root order, oracle id, codes, rows, tree, or replaced root | session expectation check |
| stage | second root bind, bind after positions, openings before positions, chain transfer before openings | session stage checks |
| BAT frozen input | changed frozen row, layout or target | BAT frozen-input check |
| FOLD | wrong oracle/group/tree/root; wrong shape/code/layout/row; order violations | FOLD start/row/round checks |
| pending | a legal wrong BAT target completes all rounds | stays pending; FOLD does not decide truth |

## Tests and measurements

- Group commitment unit: 5 passed.
- BAT-to-FOLD unit: 9 passed.
- BAT-to-FOLD integration: 2 passed.
- Engine regression: 291 passed.
- Ops regression: 103 passed.
- Workspace: 475 passed, 0 failed, 11 ignored.

| run | wall clock | VmHWM |
| --- | --- | --- |
| group commitment unit | 0.04 s | 29,780 kB |
| BAT-to-FOLD unit | 0.04 s | 29,476 kB |
| BAT-to-FOLD integration | 0.03 s | 30,800 kB |
| workspace | 1.78 s | 376,168 kB |

No proof bytes were produced; no performance gain, hiding or soundness bound is
claimed. The workspace run used the debug test profile.

## Fixture values

Fixture tag 7 for the integration run.

| item | value |
| --- | --- |
| member roots | `946e7b0c0738c48a931708abae6f903efdd807a5a368d8dd1a49c87a60169775`, `78917b6cbf41c2ee6bf27a57455ad4f87e484d7a1ef3b3809b01b2dd33856ce9`, `b3f7d8854601763edcd68fe7c7af372a6d864f031569694d8acab3578f6a5889` |
| opening positions | `[[2, 12], [2, 11], [3, 6]]` |
| fresh-mask root | `7467fade6982a92edc5900c08f0f2dd4325bb0c65dff577b3ee7a384427de54e` |
| start sum | `Fp3([6920283429447377251, 17809523116397550467, 2483515671979276613])` |
| BAT x | `Fp3([12200302307732533841, 14196379388032193109, 18327534833922446880])` |
| epsilon | `Fp3([8249439135391915515, 9158930159721092282, 22530662469755384])` |
| FOLD target and auxiliary value | `Fp3([11919649772941284836, 15517183844130328055, 1612395900922610245])` |

The member roots are fixture commitments, not SEND or asset roots. The fresh
mask root commits only the fresh mask.

## Validation scope

This step is a self-contained fixture component. No three-stage run and no model
asset read were made, so the default three-case set and the five-case set do not
apply; the shared three-stage path and seq512 logic are unchanged.

## Unclosed items

- Real SEND/G_u and send receipts.
- Authenticated assets.
- A trust anchor independent of the fixture expectation; replacing both the
  witness and the fixture expectation is not detected here.
- Fresh-mask openings and proximity; the openings cover only the drawn
  positions, not every row.
- Folded-tail derivation.
- A relation/proximity connection that can reject a pending FOLD output.
- Hiding and a security bound; the fixture source is public and deterministic.
- Single-shot semantics and production wiring; `verify_request` is unchanged.
- Kept forms are not part of this fixture.

## Artifacts

- `tests/group-commit-unit.txt`
- `tests/bat-fold-unit.txt`
- `tests/bat-fold-integration.txt`
- `tests/regression-engine.txt`
- `tests/regression-ops.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
