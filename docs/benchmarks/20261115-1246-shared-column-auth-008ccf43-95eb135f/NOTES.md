# shared-column-auth

- Scheme: shared-column-auth (B3-1: the shared FOLDEQ column authentication and
  its same-parameter checks over the encoded table asset).
- Based on: diagnostic-segment-evidence (95eb135f); previous baseline commit
  95eb135f.
- Purpose: bind the shared fold, the verifier's table reader and the drawn
  columns to one parameter set, and keep the ruled order: G_u root and its
  sampled answers first, then the table's encoded columns, then Omega and its
  relation challenge.
- Change:
  - The verifier now checks its reader against the plan's table in the first
    part: the same shape (including sampling distances) and the same table
    identity the prover committed. The prover already did this; the verifier
    did not.
  - The shared stage requires the fold shape (`G_u.index`) and the reader to
    be the same parameter set (code, row count, answers, salt, distances), and
    the column draws to come from the plan: a nonzero draw count, draws within
    the budget, the budget within the table code tail, the drawn list length
    equal to the plan, and the unique openings within the budget. Both sides
    apply the same checks.
  - Existing column authentication stays: `verify_columns` checks the stored
    column leaf (salt and the whole column of symbols; the authentication path
    is separate opening material) against the reader root, and one
    `column_row` per drawn position ties the opened column to G_u at the
    shared point. Omega is committed and sampled before its relation point is
    drawn.
- Numbers:
  - focused: chain segment 1 passed; shared reader binding 1 passed; norm
    shared carrier 1 passed
  - real gpt2-seq16 L0 K segment evidence: limbs 3; segment message bytes
    2,031,224; segment prove 26.35 s; segment verify 11.20 s; process VmHWM
    8,939,272 kB; tamper negatives 9/9 rejected
  - workspace: 1112 passed / 0 failed over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Negatives: a verifier reader with another sampling distance is refused as a
  statement mismatch, and another salt or root is refused; a plan with other
  fold distances is refused by the first-statement binding and with another
  fold salt by plan validation; a changed column position, nonce, path, salt
  or symbol is refused; an empty or over-budget draw list and a budget beyond
  the table code tail are refused by plan validation.
- Boundaries and memo items (not claimed closed): B3-1 is the first group of
  B3; the remaining B3 groups (lin_shared fold/Omega wiring, ledger and budget,
  message order, parameters and negatives) are pending. The full diagnostic
  entry still fails with the placeholder claim and zero maps; real claim/maps
  derivation, kept owner, the weight unit class and the shared-forest lookup
  shape keep their expert memo boundaries. Local evidence only, not R5. Not
  pushed.
