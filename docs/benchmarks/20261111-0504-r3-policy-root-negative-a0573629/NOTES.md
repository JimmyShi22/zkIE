# Prover-reported table root rejected by the policy pin

## Scheme

- Scheme: `r3-policy-root-negative`.
- Previous code: `a0573629`.

## Purpose

Add the stage-level negative required by the trusted-policy identity: the
prover replaces the new encoded table root in the public handoff (statement and
reader kept consistent) and the stage verifier must reject it because the
operator policy does not name that root. The proof's own statement/roots cannot
stand in for the policy identity.

## Results

- The stage harness passes: commit 32.604 s / 461,332 KiB; prove 9.862 s /
  630,832 KiB / artifact 5,581,944 B; verify 5.537 s / 72,672 KiB / accept with
  the same seal.
- The new negative assertion runs the verify worker on the re-encoded handoff
  with the replaced root and requires a non-zero exit. The other negatives
  (spent budget, missing/replaced private state, truncated artifact, changed
  handoff, replaced plan identity, stale lock, tampered artifact) still pass.
- The policy identity is rebuilt from the actual reader/statement; a bare root
  replacement fails the multi-field identity comparison.

## Boundary

- Hiding remains unclosed: the shared trace's asset openings carry plaintext
  private weights to the verifier. This negative does not change that.
- The operator policy distribution and anti-rollback are deployment trust
  assumptions; a prover-generated policy is not an independent anchor.

## Logs

- `stage-processes.txt`: the full positive and negative run.
