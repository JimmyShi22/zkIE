# shared-budget-owner

- Scheme: shared-budget-owner (B3-3 rework: the hiding charge and the read
  ledger owned by the session objects, and the reservation taken before the
  first part).
- Based on: shared-read-ledger (fef589c2); previous baseline commit fef589c2
  (the rework code commit is aa8bd982).
- Purpose: align the budget and order with the ruled chain: the provider
  hiding reservation is taken before the first part, the keeper owns the table
  hiding state, and the verifier read ledger belongs to the plan rather than
  to one shared-stage call.
- Change:
  - The table keeper trait carries the hiding state. `prove_first` reserves the
    plan column count against it before any commit; the charge token rides the
    witness and the shared stage consumes the unique positions. A repeated
    position is free, the reservation has no refund, and a second proof on the
    same keeper charges the same state again.
  - The plan owns the verifier read ledger (`ReadLedgerCell`): its maximum is
    the plan column budget, a verification reserves and commits each supplied
    column read, a repeated position must return the same column, a plan clone
    starts a fresh ledger with the same maximum, and the same plan object keeps
    one budget across calls.
  - Each table keeper in the fixtures and call sites provides the state. A
    fixture that reuses one committed table for several proofs now builds a
    fresh keeper over the same commitment per proof (a separate table copy
    object), and the second verification of one proof uses a plan clone.
  - The plan still checks draws within the budget and the budget within the
    table code tail; the shared and own one-time mask ids must differ; the
    shared and own Omega openings are counted together against one code tail,
    and the own sample rows reuse the shared sample record.
- First-run record: the first workspace run after the owner change failed two
  output-head tests with `ResourceLimit` (unit-output-head-first-failure.log):
  the fixture reused one keeper across two proofs while the toy table tail has
  one column. The fix is a fresh keeper per proof over the same commitment,
  plus a plan clone for the repeated verification; both tests then pass
  (unit-output-head-fix.log).
- Numbers:
  - focused: shared 4; keeper reservation 1; plan ledger 1; one-time masks 1;
    chain segment 1; norm shared 1; digit lookup 5
  - real gpt2-seq16 L0 K segment evidence: limbs 3; segment message bytes
    2,031,224; segment prove 26.41 s; segment verify 11.16 s; process VmHWM
    8,956,892 kB; tamper negatives 9/9 rejected
  - workspace: 1117 passed / 0 failed over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundary mapping: trusted table identity/root/record/layout/code and the
  one-time column test are in the chain; a later test that consumes a tested
  table with a matching setting is not in this single-request session. The
  order is budget/table identity first, then the first seal, the shared point,
  G_u root and answers, the table columns, Omega and its relation point, the
  shared proximity test and the shared seal. Column authentication is shared by
  unique position, while one column relation is added per draw, repeats
  included; all record, OOD, column and Fold rows enter the proximity test.
  A split parameter shape is not supported and is refused.
- Remaining open items (recorded, not claimed closed): the durable
  cross-process read/hiding ledger of the reference table keeper is not wired;
  this session keeps the in-memory owner. A separate runtime per-tree opening
  accumulator for Omega is not added: the plan counts the shared and own
  openings together against one code tail, and the sample record is shared.
  The full diagnostic entry still fails with the placeholder claim and zero
  maps; request-source limits from the earlier batch keep their list. Local
  evidence only, not R5. Not pushed.

## Correction (TURN 566)

The "fresh keeper over the same commitment" wording in this scheme is a
test-only isolated-world boundary, not a production reset path. A new hiding
budget needs new random material and the established authentication source;
the same table root and random source must follow the old ledger. The
production asset-table budget is durable and is wired separately (persistent
count under the table identity, serialized reservations, reserve before use,
no refund on failure, no reset on restart).
