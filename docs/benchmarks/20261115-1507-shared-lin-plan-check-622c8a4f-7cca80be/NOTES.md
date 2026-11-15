# shared-lin-plan-check

- Scheme: shared-lin-plan-check (TURN 568 F3: the shared linear plan is checked
  against its table distance and actual column count).
- Based on: the asset table keeper entry (a79431a8) and its trace (7cca80be);
  previous baseline commit 7cca80be.
- Purpose: the production shared plan's column draws, table distance, instance
  count, work parameters and error terms come from the same trusted plan and
  the verifier rejects a plan that does not meet the targets, not only a
  derived API.
- Change:
  - The hidden plan can carry `SharedLinCheck` (the point and comparison-point
    work bits; the column round uses the plan's column work bits).
  - `validate()` runs the shared linear plan check: it derives the table's
    one-time distance proximity, the fold-point error, the minimum column
    count and the comparison-point field error, and refuses a plan that draws
    fewer columns than the minimum, carries no distance (m below 3), or fails
    the point, column or comparison-point targets. One request is one
    instance; there is no instance field.
  - The authenticated diagnostic plan derives its table distance and column
    draws from the same parameters and carries the check; its table's random
    tail is 128, the derived distance is 284 and the derived column count 52.
  - A shared-seal prefix entry proves and verifies the first part and the
    shared stage up to their seals; the diagnostic evidence mode measures that
    prefix on the real plan and table and runs five tamper negatives.
  - Wiring negatives: a plan with one column fewer and a plan without the
    one-time distance are refused at validation.
- Numbers:
  - profile: debug (target/debug); the times and peak memory below are debug
    diagnostics, not release performance and not comparable across builds.
  - derived: table distance 284; shared column draws 52; table code tail 128
  - real first+shared chains: seal bytes 2,742,312; prove 326.72 s; verify
    106.22 s; negatives 5/5; process VmHWM 8,954,768 kB; the 231.46 s
    discovery is separate from the seal prove
  - the earlier first-only segment evidence (2,031,224 B) is retained for the
    first-part record; the shared chain's re-measurement is the seal number
    above
  - workspace: 1137 passed / 0 failed / 20 ignored over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Note: the shared-seal evidence run carried temporary trace markers that
  were removed before this commit; they printed step markers only and did not
  change any value or ordering.
- Boundaries: the check derives only the table distance and the
  query-count/error-bound layer; the fold round count and the hiding
  polynomial are not derived. The real second stage still fails on the
  placeholder claim and zero maps (not R5). The diagnostic column draws rise
  from 2 to 52 and the table tail from 8 to 128, so the new seal bytes and
  times replace the old shared-chain values. LATER stays ONE_TIME-only; the
  check uses the one-time distance. Not pushed.
