# shared-read-ledger

- Scheme: shared-read-ledger (B3-3: the hiding charge and the read ledger for
  the shared column reads).
- Based on: shared-fold-binding (4a38566d); previous baseline commit 4a38566d
  (the B3-3 code commit is 0c5888dc).
- Purpose: charge the provider hiding state before any shared column is
  exposed, and verify the shared columns through one read ledger with a
  reserved/committed budget. The plan's column draws and budget and the
  per-chain one-time mask ids remain the trusted bounds.
- Change:
  - Prover: the shared stage builds the table hiding state from the reader
    (table root, one lifecycle, the code tail as the column capacity), charges
    the unique columns before the opening, and consumes the same unique list
    against the charge. A repeated position is free; the reservation has no
    refund path. An over-capacity charge fails before any opening.
  - Verifier: the shared columns are verified through `VerifierReads` over one
    `ReadLedger`: each supplied column read is reserved and committed, a
    repeated position must return the same column, and the plan column budget
    is the ledger maximum. The existing `verify_columns` leaf check (salt and
    the whole column of symbols; path separate) still runs inside the ledger
    read.
  - The plan checks stay: draws nonzero and within the budget, the budget
    within the table code tail, the drawn list equal to the plan, the unique
    openings within the budget, and the shared and own one-time fresh mask ids
    distinct (a new test names the same id for both chains and expects a plan
    refusal).
- Open items (recorded, not claimed closed):
  - The durable cross-batch holder is not wired: the charge and the ledger are
    per run. The reference keeps the provider capacity in the table keeper
    (`IndexHolder`-like) and the read budget in the statement; the engine
    candidates mark durable storage and cross-process recovery open.
  - The cross-chain opened-position count for the adopted Omega
    (`open_oracles` over the shared tree plus every request test) is not wired
    into this session; the shared and second chains use the same Omega root and
    tree, but the accumulated per-tree opened-position ledger is an engine
    facility not yet routed here.
- Numbers:
  - focused: `the_shared_` 4 passed; one-time mask chain test 1 passed; chain
    segment 1 passed; norm shared carrier 1 passed
  - real gpt2-seq16 L0 K segment evidence: limbs 3; segment message bytes
    2,031,224; segment prove 26.40 s; segment verify 11.16 s; process VmHWM
    8,946,568 kB; tamper negatives 9/9 rejected
  - workspace: 1115 passed / 0 failed over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries and memo items (not claimed closed): B3-3 is the third group of
  B3; the message order group and the parameters/negatives group are pending,
  together with the durable holder and the cross-chain Omega count above. The
  full diagnostic entry still fails with the placeholder claim and zero maps;
  real claim/maps derivation, kept owner, the weight unit class and the
  shared-forest lookup shape keep their expert memo boundaries. Local evidence
  only, not R5. Not pushed.
