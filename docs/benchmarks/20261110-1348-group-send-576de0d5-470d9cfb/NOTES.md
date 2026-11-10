# group-send - 576de0d5 / 470d9cfb

Trace of the fixture group commitment, SEND sample registration and
BAT-to-FOLD component handoff.

## Claim status

- The committed group members now feed a SEND-style sample phase. The ordered
  root message is absorbed as the root/seed round, the shared seed is drawn from
  that round, the complete ordered answers are received and absorbed, the
  sampled registry and sample-form claims are registered, and only then are the
  member opening positions drawn and checked and the chain moved to BAT. The
  extra positions are never between the seed and the answers.
- Each member reserves one public deterministic pad per answer in its committed
  message before the seed. The seed supplies the pad coefficient, not the pad
  value. The pads are demo values and provide no hiding.
- The sample claims are pending. The verifier registers them from the bound
  roots, the drawn seed and the received answers; it never compares them with
  the provider's plaintext evaluation. SEND and BAT do not decide truth.
- The roots and plans remain fixture expectations checked against the fixture's
  own independent re-encoding and the openings. They are not real SEND receipts,
  not G_u or asset roots, and not a production trust anchor.
- The shared-seed interface exists in the relation layer and is used here: one
  seed per answer number with the largest cube's coordinates, and each member
  reads its own prefix. No interface blocker was hit; the session order is
  implemented in the isolated module. Unsupported plans, shapes, answer counts
  and seed lengths fail closed.

## Group and sample source table

| item | source and layout |
| --- | --- |
| members | three oracles; main and mask codes `log_n = 4, ell = 8, r = 1`; two main and two mask rows; one answer each |
| member messages | deterministic fixture values per tag/oracle; main and mask rows of the code's `ell` entries |
| member tails | deterministic fixture values, `r` entries per row; encoded with the row and not part of the FOLD relation view |
| pads | one public deterministic pad per answer in the last mask row at `ell - answers + answer`; encoded and committed before the seed; the seed's `beta` only supplies the coefficient; not hiding |
| member trees | one part; `main_symbols = 6`, `mask_symbols = 6`, `log_n = 4`, `log_n_zk = 4`, 16 columns; each leaf holds the main symbols, then the mask symbols, then a 40-byte salt |
| roots | one root per member from its runtime tree; tree numbers 10, 11 and 12; the ordered list is not one group root |
| shared seed | drawn from the root/seed round after the fixed roots; one seed for this fixture (largest answer count one, largest cube five coordinates), with a nonzero `beta` |
| answer order | by answer number, then member order; for this fixture oracle 0, oracle 1, oracle 2 at answer 0 |
| sample form | the existing sample cube and pad form over the member's own seed prefix |
| registration | one pending sample claim per answer in the sampled registry, then the same rows enter the corresponding BAT group |

## Binding and comparison semantics

- The sample claim binds the member identity, root, shape, sample form and
  received answer. It is pending; a later relation or proximity check is the
  only place a wrong answer could be rejected.
- The provider evaluates the same fixed member witness with an auxiliary
  plaintext sample form. That evaluation is a test oracle, not a verifier
  comparison and not a protocol-validated answer.
- The verifier builds the BAT sample rows from the received answers and the
  drawn seed. It does not use a second provider answer as a standard. A
  well-formed wrong answer is registered, changes the BAT target and stays
  pending through BAT and FOLD.
- The roots and plans are fixture values; replacing both the witness and the
  fixture expectation remains undetected without an independent trusted binding.
  This is unclosed.

## Order and negative layering

Order: ordered roots and shared seed (root/seed round) -> complete ordered
answers and answer-round absorption -> sampled/sample claim registration ->
member opening positions -> member openings -> chain move to BAT -> BAT `x` ->
FOLD start root/sum -> epsilon -> round messages and gamma -> pending output.

| layer | negative | rejection source |
| --- | --- | --- |
| root/plan | wrong order, oracle id, group, code, rows, answers, tree, root | session expectation check |
| seed/order | answers before seed, positions before answers, wrong oracle order, extra answer, truncated answer list | session stage and answer-order checks |
| pad | changed pad changes the committed root and the fixed-seed answer | fixture re-encode and auxiliary sample evaluation |
| opening | changed symbol, salt or path; witness from another commitment | opening check against the fixed root |
| BAT | changed frozen row, layout or target | BAT frozen-input check |
| FOLD | wrong oracle/group/tree/root; wrong shape/code/layout/row; order violations | FOLD start/row/round checks |
| pending | well-formed wrong answer or wrong BAT target completes all rounds | stays pending; SEND/BAT/FOLD do not decide truth |

## Tests and measurements

- Group commitment/SEND unit: 7 passed.
- BAT-to-FOLD unit: 11 passed.
- BAT-to-FOLD integration: 2 passed.
- Engine regression: 291 passed.
- Ops regression: 107 passed.
- Workspace: 479 passed, 0 failed, 11 ignored.

| run | wall clock | VmHWM |
| --- | --- | --- |
| group/SEND unit | 0.04 s | 29,760 kB |
| BAT-to-FOLD unit | 0.04 s | 29,804 kB |
| BAT-to-FOLD integration | 0.04 s | 30,808 kB |
| workspace | 1.27 s | 323,772 kB |

No proof bytes were produced; no performance gain, hiding or soundness bound is
claimed. The workspace run used the debug test profile.

## Fixture values

Fixture tag 7 for the integration run.

| item | value |
| --- | --- |
| member roots | `4f56581e5af898c3b33bc75611ee306e1536c6e5e2d39ae17978b76f0302e4f3`, `6a59c4565b4ed4a4321f393465cae4e9a7634fc2e6265d59c9dc16f2dea81021`, `46c3320280f91cd0c100b838c5f2590d0fef2539e19ae7ab962880cae597d356` |
| opening positions | `[[8, 14], [5, 7], [10, 14]]` |
| shared seed | `z` five coordinates and `beta`; exact values are in `tests/bat-fold-integration.txt` |
| received answers | three values; exact values are in `tests/bat-fold-integration.txt` |
| fresh-mask root | `7467fade6982a92edc5900c08f0f2dd4325bb0c65dff577b3ee7a384427de54e` |
| start sum | `Fp3([6920283429447377251, 17809523116397550467, 2483515671979276613])` |
| BAT x | `Fp3([14280002280861239336, 18046767938536086875, 6823183131953967170])` |
| epsilon | `Fp3([717177151104958925, 2739090653223031486, 11765671224129447086])` |
| FOLD target and auxiliary value | `Fp3([17372188479993875748, 18320622313472264981, 15353606433726305722])` |

The member roots are fixture commitments, not SEND or asset roots. The fresh
mask root commits only the fresh mask.

## Validation scope

This step is a self-contained fixture component. No three-stage run and no model
asset read were made, so the default three-case set and the five-case set do not
apply; the shared three-stage path and seq512 logic are unchanged.

## Unclosed items

- Real SEND/G_u, send receipts and authenticated assets.
- A trust anchor independent of the fixture expectation; replacing both the
  witness and the fixture expectation is not detected here.
- Fresh-mask openings and proximity.
- Folded-tail derivation.
- A relation/proximity connection that can reject a pending sample claim.
- Random hiding and a security bound; the fixture pads and seeds are public and
  deterministic.
- Single-shot semantics.
- Production wiring and performance acceptance; `verify_request` is unchanged.
- Kept forms are not part of this fixture.

## Artifacts

- `tests/group-send-unit.txt`
- `tests/bat-fold-unit.txt`
- `tests/bat-fold-integration.txt`
- `tests/regression-engine.txt`
- `tests/regression-ops.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
