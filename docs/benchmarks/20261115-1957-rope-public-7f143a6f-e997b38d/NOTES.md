# Gemma3 rope coefficient tables read from the trusted public files

- Scheme: `rope-public`, the four Gemma3 RoPE coefficient files moved to the
  trusted public-file channel. Each file is fixed; its identity is fixed in
  the model plan code and the runtime directory only locates it.
- Code: `7f143a6f`; previous commit `e997b38d`; branch `refactor/v2`.
- Baseline binary: built from `58db8cb1`, the last commit that changed the
  production code (only tests and trace notes changed between it and
  `e997b38d`), so the before side is production-code equivalent to the
  previous commit; saved as `/data/rope-baseline-zkie`, SHA-256
  `f1415b10a39189784b735ad869ec7e032a9db75f02d37cf5a1b23e83cf5d8f37`.
- Hit rationale: the four files are consumed only by the gemma3 plan and the
  same prefix logic serves seq16 and seq512, so the five-case batch covers
  gemma3 seq16 and seq512 plus the gpt2/timesfm controls.
- Fixed identity, byte-checked this round, each 65,536 entries,
  262,144 bytes, shape `[512, 128]`, signed i32 little-endian:
  - `models/gemma3/weights/rope_cos_global_i32.bin`
    `f1274259717ea29f4377d1b3a9510f44104007708de3f3a8bb15f35afe772bd1`
  - `models/gemma3/weights/rope_cos_local_i32.bin`
    `59215ba7da66e69a9209ab0a39c068229aa974dd533752104e6075c4118b10c6`
  - `models/gemma3/weights/rope_sin_global_i32.bin`
    `47c3d75d9b4a0f66476a20252e73cc3a0de33a68c422b36561755ce7f5dd5730`
  - `models/gemma3/weights/rope_sin_local_i32.bin`
    `3096bad8797eed01cd733d508d726fff764630bce8fb3740b71746ad8e80f91d`
- Change:
  - the gemma3 public table list gains the four paths with the fixed entry
    counts and digests; the public plan promotes every consumer of those
    paths to the trusted public source;
  - the per-sequence prefix layout `Range(0, m * 128)` and the global/local
    layer pairing are unchanged; the plan keeps port 1 on cos and port 2 on
    sin, and the executor reads the same prefix;
  - in the public trace the four full-table asset openings are no longer
    carried and the rope ports still take no external pool; every other
    asset keeps its original binding and is remapped by asset id;
  - the verifier reads each table once per run through the existing
    public-table reader, which checks `bytes == entries * 4`, the fixed
    SHA-256, decodes signed i32 and caches by the trusted identity;
  - the ProfileRope arithmetic (two-half vectors times cos/sin, raw
    comparison, rounded output), the commitment, anchor and policy are
    unchanged; no shape field is added to the generic public table and the
    serialization format is not changed; the shape is enforced by the fixed
    entry count, the prefix range and the ProfileRope geometry.

## Evidence

Single before/after runs on this machine, release builds, same commit per
case, same parameters/input/policy; gemma3 and timesfm use deterministic
commits created once by the after side and reused by the before side; gpt2
uses its registered commits. The wall and memory numbers are single runs and
are not a stability claim.

| case | stage | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| gpt2 seq16 | proof.bin | 1,980,401,278 | 1,980,401,278 | 0 |
| | prove wall / VmHWM | 14,232 ms / 4,088,652 KiB | 14,207 ms / 4,086,148 KiB | noise |
| | verify wall / VmHWM | 12,211 ms / 3,896,064 KiB | 11,875 ms / 3,896,772 KiB | noise |
| | trace assets | 196 | 196 | 0 |
| gemma3 seq16 | proof.bin | 5,894,758,014 | 5,892,609,694 | -2,148,320 |
| | prove wall / VmHWM | 33,836 ms / 12,263,336 KiB | 34,361 ms / 12,259,932 KiB | +525 ms / -3,404 KiB |
| | verify wall / VmHWM | 26,113 ms / 10,601,116 KiB | 25,897 ms / 10,601,140 KiB | -216 ms / +24 KiB |
| | trace assets | 241 | 237 | -4 |
| timesfm 16-patch | proof.bin | 3,740,667,388 | 3,740,667,388 | 0 |
| | prove wall / VmHWM | 23,547 ms / 8,044,608 KiB | 23,395 ms / 8,043,612 KiB | noise |
| | verify wall / VmHWM | 21,454 ms / 4,114,892 KiB | 21,232 ms / 4,118,656 KiB | noise |
| | trace assets | 238 | 238 | 0 |
| gpt2 seq512 | proof.bin | 10,917,923,358 | 10,917,923,358 | 0 |
| | prove wall / VmHWM | 60,763 ms / 26,969,400 KiB | 60,586 ms / 26,968,360 KiB | noise |
| | verify wall / VmHWM | 38,321 ms / 18,650,760 KiB | 38,684 ms / 18,649,732 KiB | noise |
| | trace assets | 196 | 196 | 0 |
| gemma3 seq512 | proof.bin | 16,249,804,014 | 16,247,655,694 | -2,148,320 |
| | prove wall / VmHWM | 112,855 ms / 51,861,796 KiB | 113,368 ms / 51,853,984 KiB | +513 ms / -7,812 KiB |
| | verify wall / VmHWM | 74,208 ms / 30,179,184 KiB | 74,394 ms / 30,174,816 KiB | +186 ms / -4,368 KiB |
| | trace assets | 241 | 237 | -4 |

- Whole-artifact metric, listed separately from the proof.bin sizes: this
  metric counts the whole proof output directory, including the statement,
  public inputs and manifest files. gemma3 seq16 `proof_bytes`
  5,894,761,986 -> 5,892,613,669 and gemma3 seq512 16,249,813,944 ->
  16,247,665,637; the manifest file grew by 3 B and 13 B respectively, which
  is the difference between this metric's delta and the proof.bin delta. The
  other three cases are unchanged (gpt2 seq16 1,980,405,259, gpt2 seq512
  10,917,933,294, timesfm 3,740,935,379).
- The measured gemma3 proof.bin drop is 2,148,320 B, equal to the pre-round
  estimate for the four full-table openings, 537,080 B per table. The value
  payload is 65,536 entries x 8 B = 524,288 B per table; the rest is opening
  bookkeeping (row records and length prefixes). No port bookkeeping is
  removed this round.
- All ten verifications pass with `output_binding = "expected"` and the
  unchanged expected output digests: gpt2 seq16 `135c4d1c…`, gpt2 seq512
  `eac0949d…`, gemma3 seq16 `8683f5e2…`, gemma3 seq512 `6d1df6b5…`, timesfm
  `cc72805b…`.
- The gpt2 and timesfm pairs are controls: identical proof bytes, identical
  digests and unchanged asset counts. Their wall and memory differences are
  run-to-run noise; no performance gain or regression is claimed for any
  case.

## Valid examples and negatives

Valid examples (accepted through the full verification path after a serialize
round trip):
- the rope step consumes the trusted cos and sin public tables through the
  fixed prefix range and reproduces the committed raw and output values;
- the production gemma3 plan promotes both global and local layer pairs and
  keeps cos on port 1 and sin on port 2 (plan test).

Negatives (rejected through the same path; split by what each shows):
- file identity, verified in the public-table reader: missing file (`io`),
  truncated and padded file (`shape_mismatch`), wrong fixed digest
  (`asset_mismatch`), wrong fixed entry count (`shape_mismatch`);
- rotation consumption with consistent authentication: the proof is produced
  honestly over the cos prefix, and the trusted plan maps the cos port to
  the sin table (a legal file identity) or to a different legal prefix slice
  of the cos table. Lengths and identities stay legal, the proof material is
  untouched, and the verifier rejects in the ProfileRope numerical relation
  (`invalid_proof`); this is not a broken-Merkle-path case;
- public port injection: a prover-supplied opening with an external port on
  a public input is rejected (`statement_mismatch`);
- other asset remapping: the existing full-path negatives
  `negative_wrong_asset`, `negative_external_and_missing_ports` and
  `negative_external_opening_must_cover_the_full_asset` are reused; their
  lines are in `plan-check.log`.

## Boundaries

- No table or ground truth is regenerated or modified; other models and
  tables keep their sources; private weights, hidden protocol, challenges,
  sampling and budgets are unchanged.
- No raw value, proof format or serialization format change; the 45,080 B
  gate object is untouched. The workspace suite is green (1163 passed /
  0 failed / 20 ignored, 51 suites). Not pushed.
- Logs and scripts: the ten `before-*`/`after-*` run logs; the release build
  log `release-build.log`; the targeted plan_check log `plan-check.log`; the
  run script `run-script.sh`; the script console log `run-console.log`; the
  run summary `run-summary.txt`; `proof-assets.py` and `proof-assets.txt`.
  Per-stage `result.json` and `progress.jsonl` stay under each run
  directory; proof binaries stay on disk and are gitignored.

## Round closure

- Expert batch review: the whole round is accepted, no further rework.
- Final commit list, in order: `7f143a6f` (production change: the four rope
  coefficient tables read through the trusted public files), `339adc55`
  (test tidy), `ec0ac723` (evidence record), and this closing commit (round
  flag, NOTES record only; no code change).
- State at closure: hard gate 45,080 B untouched; no anchor, policy, asset,
  ground truth or commitment registration changed; the workspace suite is
  green (1163 passed / 0 failed / 20 ignored); source and evidence are on
  the branch, not pushed.
