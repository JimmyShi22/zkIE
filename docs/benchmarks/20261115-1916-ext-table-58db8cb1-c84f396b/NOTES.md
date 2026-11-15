# Gemma3 rsqrt extension read from the trusted public file

- Scheme: `ext-table`, the gemma3 rsqrt extension table moved to the existing
  trusted public-file channel. The file is a fixed public file; its identity
  is fixed in the model plan code and the runtime directory only locates it.
- Code: `58db8cb1`; follow-up negative completion commit `45c4d6ed`;
  consumption-negative rework on top of `45c4d6ed`; previous commit
  `c84f396b`; branch `refactor/v2`.
- Baseline binary: the before side was built from `062c1520`. Between
  `062c1520` and `c84f396b` only the diagnostic test file changed, so the
  before binary is production-code equivalent to the previous commit.
- Hit rationale: the gemma3 rms helper, plan and asset paths are shared by the
  seq16 and seq512 cases, so this is a seq512-related hit; the evidence batch
  therefore covers the five production cases (gpt2/gemma3/timesfm seq16 and
  gpt2/gemma3 seq512).
- Fixed identity, byte-checked this round:
  `cases/gemma3/rsqrt_ext_i32.bin`, 14,849,263 entries, 59,397,052 bytes,
  signed i32 little-endian, SHA-256
  `59fc2554a5d0f13abb265ed2eb5b56d5e75d82616561c60545013be7aa6dcf50`.
- Change:
  - the gemma3 plan adds the extension path to the public table list with the
    fixed entry count and digest; the public plan promotes every consumer of
    that path to the trusted public source;
  - in the public trace the executor no longer opens the extension asset and
    the extension port carries no external pool; every other asset keeps its
    opening and its pool remapping;
  - the verifier reads the extension once per run through the existing
    public-table reader, which checks `bytes == entries * 4`, the fixed
    SHA-256, decodes signed i32 and caches by the trusted identity;
  - the ProfileRmsNorm arithmetic, the base table source, the commitment,
    anchor and policy are unchanged; the extension remains registered in the
    commitment and its asset identity is not deleted.

## Evidence

Single before/after runs on this machine, release builds, same commit per
case, same parameters/input/policy; gpt2 uses its registered commits, gemma3
and timesfm use deterministic commits created once by the after side and
reused by the before side. Percentages are arithmetic ratios of these single
runs and are not a stability claim.

| case | stage | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| gpt2 seq16 | proof.bin | 1,980,401,278 | 1,980,401,278 | 0 |
| | prove wall / VmHWM | 14,243 ms / 4,087,708 KiB | 14,150 ms / 4,088,476 KiB | noise |
| | verify wall / VmHWM | 12,156 ms / 3,895,820 KiB | 12,216 ms / 3,896,356 KiB | noise |
| | trace assets | 196 | 196 | 0 |
| gemma3 seq16 | proof.bin | 6,013,555,254 | 5,894,758,014 | -118,797,240 |
| | prove wall / VmHWM | 34,713 ms / 12,379,140 KiB | 34,349 ms / 12,262,544 KiB | -364 ms / -116,596 KiB |
| | verify wall / VmHWM | 27,041 ms / 10,714,696 KiB | 26,176 ms / 10,600,848 KiB | -865 ms / -113,848 KiB |
| | trace assets | 242 | 241 | -1 |
| timesfm 16-patch | proof.bin | 3,740,667,388 | 3,740,667,388 | 0 |
| | prove wall / VmHWM | 23,646 ms / 8,043,976 KiB | 23,395 ms / 8,043,260 KiB | noise |
| | verify wall / VmHWM | 21,324 ms / 4,115,012 KiB | 21,481 ms / 4,115,672 KiB | noise |
| | trace assets | 238 | 238 | 0 |
| gpt2 seq512 | proof.bin | 10,917,923,358 | 10,917,923,358 | 0 |
| | prove wall / VmHWM | 61,331 ms / 26,969,340 KiB | 60,797 ms / 26,967,808 KiB | noise |
| | verify wall / VmHWM | 38,944 ms / 18,649,988 KiB | 38,718 ms / 18,653,672 KiB | noise |
| | trace assets | 196 | 196 | 0 |
| gemma3 seq512 | proof.bin | 16,368,601,254 | 16,249,804,014 | -118,797,240 |
| | prove wall / VmHWM | 113,537 ms / 52,093,376 KiB | 113,099 ms / 51,861,704 KiB | -438 ms / -231,672 KiB |
| | verify wall / VmHWM | 75,132 ms / 30,295,756 KiB | 74,752 ms / 30,178,440 KiB | -380 ms / -117,316 KiB |
| | trace assets | 242 | 241 | -1 |

- Whole-artifact metric, listed separately from the proof.bin sizes: gemma3
  seq16 `proof_bytes` 6,013,559,226 -> 5,894,761,986; gemma3 seq512
  16,368,611,194 -> 16,249,813,944. The other three cases are unchanged. The
  metric's difference from proof.bin is the statement/public-input counting
  and is not part of the segment totals.
- All ten verifications pass with `output_binding = "expected"` and the
  unchanged expected output digests: gpt2 seq16 `135c4d1c…`, gpt2 seq512
  `eac0949d…`, gemma3 seq16 `8683f5e2…`, gemma3 seq512 `6d1df6b5…`, timesfm
  `cc72805b…`.
- The measured gemma3 proof drop is 118,797,240 B. The earlier segment account
  gives the extension opening segment as 118,794,632 B; the remaining 2,608 B
  is the 163 extension ports' external-pool and index bookkeeping
  (163 x 16 B), consistent with the measured value.
- The gpt2 and timesfm pairs are controls: identical proof bytes, identical
  digests and unchanged asset counts.

## Valid examples and negatives

Valid examples (accepted through the full verification path after a serialize
round trip):
- the base boundary: an activation whose derived index is 3 consumes the last
  base-table entry;
- the extension entry: an activation whose derived index equals the base
  length consumes the first extension entry.

Negatives (rejected through the same path; there is no plaintext fallback):
- swapped, missing, truncated and padded extension file;
- wrong fixed entry count and wrong fixed digest;
- a prover-supplied opening with an external port on the extension input;
- an activation whose derived index passes the extension end while the file
  identity and the input lengths stay legal. The legal boundary instance in
  the same test verifies first, and the past-end instance is rejected in the
  consumption relation check, not by a length check;
- a committed record whose norm output, raw product, mean-square or rstd
  declaration is altered before the relation proof is produced. The prover
  commits the altered record through the existing relation path, so the
  message roots are the roots of the altered declarations and the opening
  checks accept the material: the test first re-verifies every committed
  group against the registered root with the production opening routine.
  The full verification then rejects the declaration at the ProfileRmsNorm
  numerical relation. The verifier checks mean square, rstd, raw product
  and norm output in that order, and each case alters exactly one of those
  fields, so the rejection is forced at that field's constraint. The honest
  instance built through the same path verifies;
- the existing full-path asset-reference negatives are reused:
  `negative_external_and_missing_ports` and
  `prover_supplied_table_opening_is_rejected` in the same plan_check suite;
  their lines are in `extension-negatives.log`.

## Boundaries

- RoPE, the private weights and every other table keep their original
  sources. No hidden, commitment or challenge protocol is added.
- The extension asset stays registered in the commitment; the anchor,
  policy, asset identities and ground truth are unchanged and no asset is
  regenerated.
- The 45,080 B gate object is untouched. The workspace suite is green at
  the evidence commit (1156 passed / 0 failed / 20 ignored); the later
  consumption-negative rework changes only the test file and was verified
  with the targeted filter (6 passed / 0 failed). Not pushed.
- Logs and scripts: the ten `before-*`/`after-*` run logs; the workspace
  suite log `workspace.log`; the release build log `release-build.log`; the
  run script `run-script.sh`; the targeted negative suite log
  `extension-negatives.log`; the consumption-negative rework log
  `extension-consumption-tamper.log` (targeted `rms_extension` filter after
  the rework: 6 passed / 0 failed, with the four
  authentication-accepted/relation-rejected lines). Per-stage `result.json`
  and `progress.jsonl` stay under each run directory; proof binaries stay
  on disk and are gitignored.

## Round closure

- Expert batch review: the whole round is accepted, no further rework.
- Final commit list, in order: `58db8cb1` (production change: the extension
  read through the trusted public-file channel), `b5579ef4` (evidence
  record), `45c4d6ed` (negative matrix completion), `0d00804e` (trace
  archive), `ec41db34` (authentication-consistent consumption negatives),
  and this closing commit (round flag, NOTES record only; no code change).
- State at closure: hard gate 45,080 B untouched; no anchor, policy, asset,
  ground truth or commitment registration changed; source and evidence are
  on the branch, not pushed.
