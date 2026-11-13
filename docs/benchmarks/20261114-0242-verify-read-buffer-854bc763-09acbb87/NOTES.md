# Buffered proof decoding and production verify timing

## Scheme and baseline

- Scheme: `verify-read-buffer`.
- Code: `854bc763`; previous commit `09acbb87`; branch `refactor/v2`.
- Trigger: review of the five-case acceptance package found a production CLI
  verify regression against the pre-work baseline (+182.6% on gpt2-seq16,
  +34.9% on gpt2-seq512, +296.4% on gemma3-seq16, +169.4% on timesfm). The
  independent harness is not a substitute for the pre-work -> current
  production CLI comparison.
- Scope: the proof read path only. No wire format, sampling parameter,
  memory budget, verification condition, anchor or output binding change.

## Stage split

Temporary instrumentation in the production provider timed the proof decode
(`read_proof_exact`) against everything after it (anchor check, relation
verification and output binding). The instrumentation was removed before the
fix commit; its raw logs are preserved.

| Stage (cold) | gpt2-seq16 | gpt2-seq512 | gemma3-seq16 | gemma3-seq512 | timesfm-16patch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode before fix | 54,717 | not measured | not measured | not measured | not measured |
| Rest before fix | 16,357 | not measured | not measured | not measured | not measured |
| Total before fix | 71,074 | not measured | not measured | not measured | not measured |
| Decode after fix | 2,046 | 10,169 | 5,735 | 15,283 | 3,640 |
| Rest after fix | 16,249 | 32,822 | 26,329 | 64,754 | 19,836 |
| Total after fix | 18,295 | 42,991 | 32,064 | 80,037 | 23,476 |
| Decode share after fix | 11.2% | 23.7% | 17.9% | 19.1% | 15.5% |

- The pre-fix split was measured on the same instrumented build immediately
  before the decoder change (`stage-timing-prefix-gpt2-seq16.log`). Decode was
  77.0% of the verify wall time on gpt2-seq16.
- The after-fix split is from the same instrumentation on the fixed decoder
  (`stage-timing-postfix-*.log`). The clean, uninstrumented build is used for
  the production timings below.
- The instrumentation did not change verification behavior; it only printed
  the two timings to stderr.

## Fix

`bounded_trace.rs::read_file` now wraps the proof file in
`BufReader::with_capacity(64 * 1024, file)` before the single streaming decode
entry. The wire format is decoded word by word, so a plain `File` turned every
word into a small read; a fixed 64 KiB window removes that overhead without
buffering the proof.

Preserved unchanged:

- one production decoding path (`read_from` / `read_from_measured`);
- the file size versus decode budget check before decoding, and the
  per-collection budget charged before its backing allocation;
- overflow, truncation and trailing-byte rejection;
- the trusted plan source and identity checks, root and path authentication,
  and the output binding;
- proof bytes and sampling parameters: the same proof files are verified.

The decoder unit tests (11) pass, including the budget, truncation, trailing
and read-only-stream cases.

## Production CLI verify after the fix

Same machine, 64 logical CPUs, default rayon pool, no thread override, 1 cold
+ 3 warm per case, clean release build. The proof files are the ones produced
by the five-case acceptance matrix; no proof was regenerated.

| Case | Pre-work cold/warm ms | Before fix cold/warm ms | After fix cold/warm ms | Delta vs pre-work (cold / warm) | Verify HWM after KiB | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gpt2-seq16 | 25,325 / 25,339 | 71,627 / 71,609 | 17,626 / 17,562 | -30.4% / -30.7% | 4,102,648 | PASS |
| gpt2-seq512 | 233,576 / 232,400 | 318,670 / 313,535 | 39,963 / 39,596 | -82.9% / -83.0% | 18,850,544 | PASS |
| gemma3-seq16 | 46,184 / 46,312 | 182,674 / 183,594 | 29,875 / 29,895 | -35.3% / -35.4% | 10,949,652 | PASS |
| gemma3-seq512 | 855,864 / 830,363 | 487,948 / 486,512 | 75,425 / 75,554 | -91.2% / -90.9% | 30,511,796 | PASS |
| timesfm-16patch | 43,144 / 43,141 | 117,847 / 116,243 | 22,355 / 22,303 | -48.2% / -48.3% | 4,188,216 | PASS |

- All five cases are now below the pre-work CLI verify baseline; the
  regression is eliminated. Against the pre-fix acceptance state the reduction
  is 75.4-87.5% (cold).
- Verify peak memory is also below the pre-work baseline for all five cases
  (23.9-74.7% lower); the 64 KiB buffer itself is negligible.
- The authenticated output digest of every run equals the expected output
  digest bound by the case policy; the digests are identical to the acceptance
  matrix.
- No explicit cache drop is performed; cold/warm is run order, the same basis
  as the acceptance matrix. CLI verify timing is separate from the independent
  harness timing.
- This table replaces the verify column of the five-case acceptance trace. The
  proof bytes, prove and commit numbers there are unaffected because this fix
  only changes the read path.
- An initial rerun attempt used the new trace directory instead of the matrix
  trace directory for the input paths and produced immediate
  `incomplete_artifact` records; those invalid runs were deleted and the table
  above was produced with the matrix trace directory as the input root.

## Workspace

`cargo test --workspace`: 48 suites, 973 passed, 0 failed, 17 ignored
(`workspace.txt`).

## Boundaries

- The remaining verify time is the actual verification work: decoding is now
  11.2-23.7% of the total. No further read-path claim is made.
- The regression was read overhead, not a proof or verification semantics
  change; no verification check was relaxed.
- No push.

## Acceptance closure

- The buffered decode repair and the work package are accepted for closure
  (expert review TURN 520).
- After-fix production CLI verify warm medians: gpt2-seq16 17,562 ms;
  gpt2-seq512 39,596 ms; gemma3-seq16 29,895 ms; gemma3-seq512 75,554 ms;
  timesfm-16patch 22,303 ms. All are below the pre-work baseline
  (25,339 / 232,400 / 46,312 / 830,363 / 43,141 ms).
- The regression numbers measured before the fix (warm medians 71,609 /
  313,535 / 183,594 / 486,512 / 116,243 ms) remain in the table above and in
  the five-case acceptance trace. They are retained as history and are not
  overwritten.
- Push target: `https://github.com/JimmyShi22/zkIE.git`, branch `refactor/v2`.
  The upstream `trustless-ai` repository is not pushed.

## Artifacts

`identity.txt`, `fix-stat.txt`, `fix.diff`,
`stage-timing-prefix-gpt2-seq16.log`, `stage-timing-postfix-*.log`,
`verify/<case>/<run>/result.json`, `progress.jsonl` and `*.log`, `results.md`,
`workspace.txt`, `NOTES.md`.
