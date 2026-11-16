# table-key: fixed R19 declaration and the shape-handoff investigation

## Scheme and provenance

- Scheme: table-key. Committed code under test at the start of this trace:
  `41090c52` (fixed public R19 range declaration) and `c7b7795b` (the
  evidence driver takes the range from the public configuration). The
  diagnostic builds add temporary shape probes on top of `c7b7795b`; the
  probes are not committed. Previous scheme close: `e46f5efb`.
- The findings of this trace are fixed in `d96fc2c1` (per-row bits tails),
  `2545dcda` (plan matrix entry order) and `acbf1f89` (plan table identity).
- Purpose: adopt the fixed public R19 declaration, localize the real-scale
  key-group shape failure along the F1-F4 checklist, and record every
  diagnostic ledger spend.
- Boundary: this trace is diagnostic. The clean end-to-end evidence is in the
  final trace of the round. The R18 refusal record stays; the diagnostic roots
  are spent and are not reset.

## Declaration change

- Fixed public declaration: lo=-262144, hi=262143, d=19, mu=5, offset=262144,
  bit_entries=2^24 (exactly the admitted limit), key_bits=20, table_log=19.
  No clamping, no dropped entries; the 48 R18 outliers stay on record.
- The verifier rebuilds the same declaration from the same public
  configuration; the diagnostic width override was removed from the driver.
- The range change changes the key plan/statement identity. The derived key
  plan under R19 has main code (18,16384,65), mask code (18,256,66), fold
  k=11, h=4.

## Diagnostic runs

Every diagnostic instance is committed fresh (PASS, about 3.8 s) and charges
its own 52-column durable budget once; no charge is refunded; a retry on a
spent root is refused.

| run | first rejected predicate |
|---|---|
| r653-a | `folded_tails`: bits member `tails.len()=1` against layout `main_rows=2048` (one flat 133120 = 2048x65 vector) |
| r653-b | fix 1 (per-row tails); the key group IOPP runs; the first verification rejects at the pending-claim/Eval-target comparison |
| r653-c | the first part proves; the first verification stops after the verified lookup |
| r653-d | verification stops between the lookup and the member point check |
| r653-e | pending row rebuilt: point=10, masks=2, psi_match=true, target_match=false |
| r653-f | prover and verifier lookup-claim fingerprints identical; the prover's own eval target already differs from its pending target |
| r653-g | fix 2 (plan matrix order); the first part and the first verification pass in full |
| r653-h | the shared stage stops with StatementMismatch before its first public-step probe |
| r653-i | shared-stage entry localization: the failure is in the shared pre-checks |
| r653-j | plan table identity differs from the committed reader identity (`identity_match=Ok(false)`); every other shared pre-check matches |
| r653-k | fix 3 (reader identity); the complete chain passes: first, shared, second and the independent verification |

## Findings

### F1: exact return point

The first real-scale rejection is in `folded_tails`, at the first
member-length check, before `MaskTree::commit`, `trees.tree_for`,
`loop_private_material` and `prove_multi_stage_iopp`. The earlier
"inside the multi-stage IOPP" attribution was wrong; once the FOLD handoff
receives one tail per real row the whole multi-stage IOPP passes.

### F2: member-to-fold layout

- Members and order: BAT oracles [0,1,2] = bits (source 21), lookup mask (22),
  index mask (60).
- Real rows: bits 2048 (R19: 2^24 entries at 2^14 per row), lookup mask 8,
  index mask 1. Per-row tail length is the fold code's `r` (65 main, 66 mask).
- Fresh mask: 16 rows (`2^h`, h=4) of 66; the fresh tree commits the original
  16-row mask.
- Defect: the bits member received one flat 2048x65 vector instead of one tail
  per row; the FOLD handoff contracts tails per real row. Fixed in `d96fc2c1`
  by splitting the committed flat tails at `r`; no virtual rows are added to
  any real member.

### F3: stage-zero handoff

- Handoff: k=11, h=4, main code (18,16384,65), mask code (18,256,66), main
  claim 16384, mask claims 256/256, gamma 11.
- `SingleRows`: main 16384, main tail 65, two masks 256 with tails 66, claims
  16384/256/256.
- `validate_stage_zero`, `check_stage_zero_ids`, the fresh tree root and the
  member tree plans all pass on both sides.

### F4: switches and last base

- Three stages: two switches plus one middle stage FOLD, as the plan requires.
- Stage 1: main code (14,1024,66), mask (13,128,66), k=4, h=3, positions 66,
  target main rows 16, mask rows 8; private tails 16x66 and 8x66.
- Stage 2: main code (15,1024,53), mask (15,512,53), k=0, h=0, positions 53,
  target 1x1; private tails 1x53 and 1x53.
- Every switch authentication, the middle stage FOLD and the last BASE pass on
  both the prover and the verifier.

### Key enumeration order

- The first failing predicate after the tails fix: the first verification
  rejects with InvalidProof because the rebuilt pending-row target differs
  from the sent Eval target (`psi_match=true`, `target_match=false`). The
  prover's lookup claim and the verifier's lookup claim are identical (same
  fingerprint), and the prover's own Eval target already differs from its own
  pending target, so this is not a verifier bug.
- Cause: the plan fixes the matrix order (matrix rows are the record places,
  matrix columns are the records; a key number's low coordinates are the
  matrix columns), and the fixture commits the table in that order. The real
  driver enumerated the asset file's flat order instead, i.e. the transpose,
  so the claim paired the row and column points with the wrong matrix axes.
- Fix `2545dcda`: the driver enumerates the matrix entries in the plan's order
  (the transpose of the file's flat order). The public plan, the verifier and
  the committed table instance are unchanged; the multiplicity counts and the
  bits message are permutation-invariant and unaffected.
- After the fix the prover's Eval target matches its pending target and the
  whole first part (prove and independent verify) passes.

### Plan table identity

- The hidden-send plan carries the table identity of the reader: a digest of
  the committed root, record, configuration and shape. The other production
  hidden-send plan and the shared-stage checks use the same convention, and
  the fixture set the plan's table identity from the committed reader.
- The key plan fed the model-level meta identity (a root-independent public
  fact). The shared stage refused the plan with StatementMismatch; the first
  two stages never observe the difference because only the shared stage
  compares against the reader identity. The probe shows every other shared
  pre-check matches and only the identity equality fails.
- Fix `acbf1f89`: the plan input is the committed reader identity. The
  registration check binds the same reader identity (root, record,
  configuration, shape) and keeps the model-level meta identity as a separate
  public assertion.

## Full-chain result on the diagnostic build

Run r653-k (probe build carrying the three fixes; the probes only print):
the complete chain passes in one run: first part, first verification, shared
part, shared verification, second part, the encoded byte bundle and the
independent `verify_key`, plus the byte negatives and the spent-budget
retry check.

- proof bytes 4,549,416 (first 3,551,120; shared 712,200; second 286,064)
- prove 189,230 ms; verify 59,403 ms
- VmHWM 35,430,220 KiB (process-wide)
- ledger 0 -> 52
- plan digest `e432b62e...`, reader identity `06f285b7...`, table root
  `b23748a2...`, table configuration `9017dd1a...`

These numbers are from the diagnostic build and are not the final evidence;
the clean run is recorded in the final trace of the round.

## Ledger

- Diagnostic roots r653-a .. r653-k: each charged 52 columns; no refund.
- The earlier roots (601d921c, r19-a, r19-b, r19-c) remain spent.

## Status propagation

The run wrappers record the true `cargo test` status and propagate it as the
command's exit status; a failed run is never recorded as a pass through an
outer zero.

## Deterministic diagnostic evidence (amended after review)

All diagnostic runs in this trace used the fixed public constant
`Digest([0x6B; 32])` as the run seed. They are deterministic diagnostic
evidence for the shape-handoff investigation, not hidden-key delivery
evidence. The private-seed delivery evidence is
`20261116-1147-table-key-8f9faab0-df994559`. The diagnostic roots stay spent
(52 each) and are not reset.
