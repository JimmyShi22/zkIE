# asset-budget-durable

- Scheme: asset-budget-durable (TURN 566 item 1: the asset table hiding budget
  is a durable count).
- Based on: the B3 order and parameters close-out (35ca07ef) and the wording
  correction (bc85b983); previous baseline commit bc85b983.
- Purpose: replace the per-call in-memory reservation of the asset table
  columns with a durable table-life count: reserve before use, no refund, no
  reset on restart, serialized reservations.
- Change:
  - The engine has a `DurableColumnBudget` over the persistent count store:
    one count file under the provider's trusted directory, a lock over the
    live files, a room check for every live count, an append of the fixed batch
    columns, a recheck and one sync before a receipt is returned. A rejected or
    failed call returns no receipt; there is no cancel or refund. The file
    length is the table-life count, and a new owner over the same file
    continues from it.
  - The table keeper can provide the budget. The first part reserves one batch
    against it before any commit when the keeper owns one; the batch size must
    equal the budget's fixed column count, and the in-memory state still checks
    the unique positions of this batch against the same reserved count.
  - Test: a durable budget over a temporary table directory. One proof counts
    one batch; a second proof that fails after the reservation keeps its count;
    a new owner over the same files continues from the count and an exhausted
    budget is refused.
- Corrected wording: the earlier "fresh keeper over the same commitment" is a
  test-only isolated world, not a production reset path; this durable count is
  the production authority for the asset table budget (correction commit
  bc85b983 and the correction section of the earlier trace).
- Numbers:
  - focused: durable budget 1; shared 7; chain segment 1; norm shared 1; digit
    lookup 5
  - real gpt2-seq16 L0 K segment evidence: limbs 3; segment message bytes
    2,031,224; segment prove 26.27 s; segment verify 11.15 s; process VmHWM
    8,946,020 kB; tamper negatives 9/9 rejected
  - workspace: 1122 passed / 0 failed over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries (recorded, not claimed closed): the durable store is the engine's
  single-machine count adapter with its documented limits (trusted same-machine
  lock and directory; not rollback-proof or cross-machine; a copied directory
  has an independent count). The production keeper construction at the real
  asset table directory is the remaining wiring: the keeper hook is in place,
  the model/CLI path is not yet connected. LATER qualification is in-session
  only per the ruling; this session supports the one-time table test and says
  so. The Omega shared and own openings are counted together by the plan
  against one code tail, and the own sample rows reuse the shared sample
  record; a per-tree runtime opening accumulator is not yet wired and stays
  recorded. The full diagnostic entry still fails with the placeholder claim
  and zero maps; local evidence only, not R5. Not pushed.
