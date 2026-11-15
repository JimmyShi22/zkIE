# Trusted public tables: seq512 production evidence

- Scheme: `public-tables-b4`, seq512 round (the same code as the seq16 round,
  `062c1520`; the wording corrections are in `b89e14be`).
- Code: `062c1520`; previous commit `61c8d2ab`; branch `refactor/v2`.
- Hit rationale: this round first landed the global asset filtering and pool
  renumbering (`Assets::load`/`pool` in every model executor), the public plan
  used by every consumer (`trace_plan_public`), and the public-source branch
  in `verify_step`. These paths are not limited to the four fixed tables or to
  short sequences: every trace asset is filtered and every pool is
  renumbered, and the verification input source changes for every case.
  Therefore the seq512 production chains (gpt2, gemma3) are the affected
  long-sequence evidence under the five-case exception. The three seq16 cases
  from the previous trace are not rerun.
- Commit reuse: gpt2-seq512 reuses its registered commit instance (the one
  bound by the tracked policy; a fresh commit samples new private table
  randomness and cannot match the pinned anchor). gemma3-seq512 has no
  random table instance; its commit is deterministic and is created once by
  the after side, then reused by the before side. Only the binary differs
  between the two sides: the baseline worktree at `61c8d2ab` versus the
  `062c1520` build.

## Measurement

These are single before/after measurements on this machine; the ratios
describe these single runs and are not a stability claim. Release builds,
same commit, same case file, same parameters (`profile = "case"`), input and
policy.

| case | stage | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| gpt2-seq512 | prove wall | 62,062 ms | 61,427 ms | -635 ms (-1.0%) |
| | prove VmHWM | 27,362,440 KiB | 26,967,540 KiB | -394,900 KiB (-1.4%) |
| | verify wall | 38,739 ms | 38,754 ms | +15 ms (noise) |
| | verify VmHWM | 18,851,352 KiB | 18,649,416 KiB | -201,936 KiB (-1.1%) |
| | proof.bin | 11,127,643,038 B | 10,917,923,358 B | -209,719,680 B |
| gemma3-seq512 | commit | reused | GEMMA_COMMIT | created once, then reused |
| | prove wall | GEMMA_PROVE_BEFORE | GEMMA_PROVE_AFTER | |
| | prove VmHWM | GEMMA_PHWM_BEFORE | GEMMA_PHWM_AFTER | |
| | verify wall | GEMMA_VERIFY_BEFORE | GEMMA_VERIFY_AFTER | |
| | verify VmHWM | GEMMA_VHWM_BEFORE | GEMMA_VHWM_AFTER | |
| | proof.bin | GEMMA_PROOF_BEFORE | GEMMA_PROOF_AFTER | |

- Carried trace assets from the bincode header: gpt2-seq512 199 -> 196.
- gpt2-seq512 removes the exp/rsqrt/gelu openings: their values are
  209,715,200 B (exp 16,777,216 + rsqrt 58,720,256 + gelu 134,217,728). The
  small remainder of the measured proof delta is the per-instance opening
  bookkeeping and the relation-proof bookkeeping for the table ports that no
  longer name an external pool; it is measured per case and not claimed as a
  fixed number.
- Output correctness: both verifications PASS with `output_binding =
  "expected"` and the expected 512-element argmax digest
  `eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd`.
- The expected table digests and entry counts are fixed constants in the
  model plan code; the case config's asset root only locates the files. The
  0.07 s warm-cache read+hash in the seq16 trace is not the full first
  materialization cost; the full materialization is inside the verify wall.

## Evidence

- Per-stage `result.json`, `progress.jsonl`, policy copies and run logs under
  this directory; raw stage records in `run-summary.txt`; proof asset counts
  in `proof-assets.txt`; `proof.bin` files are large binaries kept on disk
  and gitignored.
- No workspace rerun for this round (only the production path is exercised).
- The hard gate is unchanged from the seq16 round: 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`.
- No asset or ground-truth file was modified or regenerated. Not pushed.
