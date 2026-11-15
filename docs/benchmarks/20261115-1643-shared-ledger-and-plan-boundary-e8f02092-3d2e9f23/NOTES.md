# shared-ledger-and-plan-boundary

- Scheme: shared-ledger-and-plan-boundary (TURN 570 F2a/F2b/F3a/F3b as one
  rework batch; F1 stays closed).
- Based on: the shared linear plan check (622c8a4f) and its trace
  (3d2e9f23); the rework code and bookkeeping commit is e8f02092; previous
  baseline commit 3d2e9f23.
- Purpose: the real asset/shared-seal entry uses one ledger stable per table
  identity and the existing durable-budget protocol; the lifetime total is the
  table's own hiding capacity; production boundaries require the checked
  shared linear plan; nonzero point/zeta work bits are refused while the
  actual rounds perform no work.
- Change:
  - `AssetTableKeeper::open` refuses a lifetime total above
    `table.reader.shape.code.hiding_positions()`; the diagnostic table's total
    is its table code tail and its HidingState uses the same total.
  - The diagnostic table opens a ledger keyed by the table root under a
    stable root; the first run initializes it, later runs open the same files,
    and a missing count is refused instead of reset. The shared-seal
    production entry requires the keeper's durable budget.
  - The production asset and shared-seal entries require
    `plan.proximity.lin_check`; the fixture entry `prove_hidden_lin_hidden`
    remains for component tests. `check_shared_lin` accepts only zero point
    and zeta work bits, matching the fixed rounds.
  - Repeated verifications use a fresh plan-owned read ledger (a plan clone),
    so the tamper checks are genuine and do not pass by ledger exhaustion.
  - Evidence mode: the real first+shared chains are proven and verified, five
    tampered messages are rejected, a second keeper in the same process
    reopens the persistent ledger file and proves again (the count continues;
    this is not a measured process restart, and the code fact is that opening
    an existing ledger never resets it), the remaining lifetime is drained,
    and the next proof is refused at the reservation. Negatives: a plan
    without the check is refused at the production entry; a plan with one
    fewer column or without the distance is refused at validation.
- Evidence iterations (all logs kept):
  - attempt 1, table tail 256: rerun reservation hit the lifetime because the
    setup (discovery, chain stack, first prefix) already spent three batches
    (log shared-seal-attempt1.log).
  - attempt 2, table tail 512: the rerun proof succeeded but its verification
    hit the plan-owned read ledger shared by repeated verifications (log
    shared-seal-attempt2.log); fixed by verifying with a plan clone and by
    making the tamper checks use one, then raising the tail to 1024 so the
    stable ledger holds the setup, the first prefix and the rerun.
  - attempt 3 (final): passed (log shared-seal.log).
- Numbers (final, table tail 1024):
  - profile: debug (target/debug); the times and peak memory below are debug
    diagnostics, not release performance and not comparable across builds.
  - derived: table distance 530; shared column draws 129; lifetime 1024
  - real first+shared chains: seal bytes 3,365,736; prove 323.76 s; verify
    106.63 s; negatives 5/5; ledger count before rerun 387 and after 516;
    lifetime 1024; the next proof refused after draining; process VmHWM
    8,943,072 kB; the 227.25 s discovery is separate from the seal prove
  - workspace: 1140 passed / 0 failed / 20 ignored over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries: the stable ledger persists across runs by table identity; a
  deleted whole ledger directory is indistinguishable from a new table (the
  store's documented limit). The second stage still fails on the placeholder
  claim/zero maps (not R5). The diagnostic table tail rose 128 to 1024 and
  the draws from 52 to 129, so these bytes and times replace the earlier
  shared-chain values. LATER stays ONE_TIME-only. Not pushed.
