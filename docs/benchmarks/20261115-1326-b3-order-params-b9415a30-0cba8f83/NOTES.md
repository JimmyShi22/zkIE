# b3-order-params

- Scheme: b3-order-params (B3-4 message order and B3-5 parameters and
  negatives; closes the B3 column authentication chain).
- Based on: shared-budget-owner (0cba8f83); previous baseline commit
  0cba8f83 (the close-out code commit is b9415a30).
- Purpose: pin the ruled message order and the parameter and relation
  obligations on the actual verification path, and close the B3 chain.
- Change (tests only):
  - The column reservation is taken at the first part start: a proof that
    fails after that point does not get the reservation back, and the next
    proof on the same keeper charges again. This is the budget-before-first
    order and the no-refund rule.
  - The shared transcript binds each message to the one before it: a changed
    first seal, G_u seed or answer, Omega root, seed or answer, column nonce,
    fold start, fold round, shared tail or shared seal is refused.
  - Budget reuse: a keeper whose hiding capacity is the batch column count
    refuses a second proof, and a plan whose read ledger is the batch size
    refuses a second verification.
  - Parameters and relations: another table identity, fewer shared base
    positions, a budget that no longer matches the ledger, and a missing
    column opening, fold round, G_u answer or Omega answer are refused.
    The plan keeps the draw/budget/tail checks.
- Query counts: the plan's draw count, budget and ledger are the same values
  on both sides; the verifier draws exactly the plan counts, and the plan
  fault rejects zero or over-budget draws and a budget beyond the table code
  tail. Reducing the shared base positions while keeping the same proof is
  refused. A derivation of the query count from an error-bound solver (the
  reference plan counterpart) is not in this model and stays recorded as
  open; no number is claimed as final.
- Numbers:
  - focused: reservation/order/budget/parameters 4; shared 7; chain segment 1;
    norm shared 1; digit lookup 5
  - real gpt2-seq16 L0 K segment evidence: limbs 3; segment message bytes
    2,031,224; segment prove 26.41 s; segment verify 11.17 s; process VmHWM
    8,945,092 kB; tamper negatives 9/9 rejected
  - workspace: 1121 passed / 0 failed over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries: single-request one-time table test; later-test qualification,
  the durable asset-table hiding budget and the cross-chain Omega opening
  accumulator follow the durability ruling and are not part of this group.
  The full diagnostic entry still fails with the placeholder claim and zero
  maps; request-source limits keep their list. Local evidence only, not R5.
  Not pushed.
