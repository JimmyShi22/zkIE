# Norm segment whole chain - checkpoint round (gpt2-seq16)

## Scheme and baseline

- Scheme: `norm-segment-whole-chain` (first acceptance attempt).
- Code: `3cc93db2`; previous commit `d1812052`.
- Purpose: run the real 1 cold + 3 warm acceptance of the final-norm ->
  lm_head segment after the byte-width fix.

## Result

- Round-1 commit PASS: 239,826 ms, VmHWM 27,296,668 KiB.
- Round-1 prove PASS: 797,547 ms, VmHWM 338,379,444 KiB, proof
  3,589,244,952 B.
- Round-1 verify FAIL: `statement_mismatch` after 82,021 ms, VmHWM
  7,439,960 KiB. The driver stopped after the first round; no warm round ran.

## Cause

- The carrier table identity built by the prover
  (root `1cbc9c0b...`, configuration `b89b6eaf...`, record
  `7a3e0325...`) differed from the policy identity block
  (`6b7bedfa...` / `b45ea4f7...` / `5765142c...`); the shapes matched.
- The policy block names the consumer table, not the lm_head table. The
  follow-up commit binds the segment table identity in the statement and
  derives the session from it (`cf182085`).

## Deletion note

- The text traces of this round are kept. Its gitignored proof binaries were
  deleted when the follow-up scheme started benching.
