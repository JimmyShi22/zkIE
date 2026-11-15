# digit-lookup-remaining-rows

- Scheme: digit-lookup-remaining-rows (the segment consumes every remaining
  relation the two single-call digit lookups leave; each lookup output lands
  on its own committed mask member).
- Based on: diagnostic-segment-evidence (9eeed10c); previous baseline commit
  9eeed10c.
- Purpose: the lookup proofs leave tie rows and key-level and table-level rows
  besides the leaf and bit rows. Only the leaf and bit rows were consumed
  before, so (planes - 1) + key_bits + table_log relations per call were not
  checked against the committed mask table.
- Change: `digit_lookup_rows` now walks every output in
  `claims.rows[0..leaf_index]` and maps its `hiding_slots`/`hiding_form`,
  `claim_masks` and `row_target` onto the lookup's committed mask member. The
  key registration, multiplicity, leaf, bit and bit-end rows are unchanged.
  The same builder serves the low (0..1023) and the high (0..255) plane.
- Parameters and ledger: the added rows consume the existing mask allocation;
  no new mask slots, entries or trees are added. The frozen request layout
  values (mask rows/slots, end values, rounds, opening budget, sample plan,
  tree layout, stage plan) are unchanged, and the proximity runtime plans the
  actual row count. Real 3-limb activation: the added rows are 31 + 17 + 10 =
  58 for the low call and 31 + 17 + 8 = 56 for the high call (114 in total).
- Numbers:
  - chain segment unit test: 1 passed (unit-chain.log)
  - digit lookup tests: 5 passed, including the new both-plane test
    (unit-digit.log)
  - segment evidence, real gpt2-seq16 L0 K activation: limbs 3; pending
    discovery 26.83 s (separate); segment message bytes 2,031,224 (was
    2,032,888, no growth); segment prove 26.40 s; segment verify 11.23 s;
    process VmHWM 8,935,792 kB; tamper negatives 9/9 rejected
    (segment-evidence.log)
  - full diagnostic entry: unchanged, still `hidden prove: InvalidProof`
    (full-failure.log)
  - workspace: 1111 passed / 0 failed over 51 suites (workspace.log)
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
    (hard-gate.log, hard-gate.sha256, hard-gate.size)
- Distinguishing negative: the new test builds a lookup proof whose PSC
  hiding entries differ from the committed mask table. The outer lookup
  verification accepts that proof (the mask values are private and the roots
  are the committed table's), while the added remaining row, evaluated on the
  committed mask member, no longer equals its target. The honest row holds.
  Both the low and the high plane are covered; the row count is
  `leaf_index + 5` for each, so no remaining output is skipped.
- Boundaries and memo items (not claimed closed): the full consumer entry is
  unchanged and still fails with the placeholder claim and zero maps; local
  segment evidence only, not R5; real claim/maps derivation for the isolated
  L0 K projection, kept owner mapping, weight unit class and weight bit bound,
  shared-forest lookup shape (two single-call lookups), tau restricted to
  powers of two, openings_per_oracle=2 transitional W read.
- Not pushed.
