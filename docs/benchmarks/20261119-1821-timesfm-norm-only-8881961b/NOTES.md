# timesfm norm-only segment acceptance

- Scheme: the timesfm L0 attention RMS pre-norm norm-only segment (TURN 772
  option A). The segment is the accepted norm chain minus the Dot/W rows:
  commitments, transcript, lookup, PSC instances, own BAT, FOLD, CS and BASE
  all run; the consumer projection stays on the generic path and the
  materialized `hd` rows bind the segment output to the trace's L0 activation
  rows.
- Code: `8881961b` (segment wiring, carrier, provider binding, tests),
  `111ae1c0` (shared mode-aware norm output rows; the RMS segment fix),
  `1b1d1f06` (sparse tail-unit representation; no semantic change).
- Run: one exclusive slot, timesfm-16patch, four rounds (1 cold + 3 warm) plus
  the negative matrix. The acceptance harness keeps `proof.bin`
  (`ZKIE_ACCEPT_KEEP`); stage working directories are avoided by passing
  absolute paths.

## Round numbers

| round | commit ms | prove ms | verify ms | artifact bytes |
|---|---|---|---|---|
| 1 (cold) | 155,365 | 373,424 | 132,350 | 5,259,831,827 |
| 2 | 151,912 | 373,019 | 132,034 | 5,259,830,069 |
| 3 | 153,230 | 373,333 | 132,403 | 5,259,834,355 |
| 4 | 151,096 | 373,486 | 132,705 | 5,259,834,135 |

Warm medians (rounds 2-4): commit 151,912 ms, prove 373,333 ms, verify
132,403 ms, artifact 5,259,834,245 B. Peak resident set size across all
stages: 58,581,632 kB (GNU time, includes the stage children). The stage
records agree: prove `vmhwm_kib` 58,571,468, verify 37,946,416 (round 1).

## Byte account (round 1)

| file | bytes |
|---|---|
| proof/statement.bin | 133,216 |
| proof/public-inputs.bin | 131,080 |
| proof/timesfm-norm-segment-carrier.bin | 1,644,035,480 |
| proof/proof.bin | 3,613,502,964 |
| proof/consumer_proof.bin | 2,024,320 |
| proof/manifest.json | 4,702 |
| proof/COMPLETE | 65 |
| total (the proof-bytes metric) | 5,259,831,827 |

The carrier is the segment's two canonical message stages; its size is
setup-dependent because every commit samples fresh table randomness (the two
component runs differed by 166,144 B on the same code). `proof.bin` is the
trace carrier, not a zero-knowledge proof; the artifact total above is the
honest byte figure for this segment build.

## Negatives (real verifier, round-1 artifact)

| case | outcome |
|---|---|
| control | PASS |
| version (carrier version 2) | rejected (FAIL) |
| messages_truncated (last byte removed) | rejected (FAIL) |
| statement_identity (declaration false, carrier present) | rejected (FAIL) |
| declaration_omitted (declaration false, carrier removed from manifest/complete) | rejected (FAIL) |
| reader_root (commit L0_lnw table root tampered) | rejected (FAIL) |
| record_digest (commit L0_lnw record tampered) | rejected (FAIL) |
| source_rows (first non-final RMS step output changed) | rejected (FAIL) |

The harness total is 460.05 s. Every proof-artifact tamper rewrites the
manifest entry and complete marker, so the artifact-integrity layer accepts
the file and the segment checks must reject it. The timesfm carrier has no
reader field (there is no output-head table), so `reader_root` and
`record_digest` tamper the commit bundle's L0_lnw asset-table metadata and
rewrite the commit manifest; this is the trusted gain identity source.

## Hard gate

The twenty tracked `proof-45080.bin` copies are 45,080 B each with sha256
`7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`; the docs
tree carries only this evidence directory. The diff touches no assembly or
serialization path: the sparse-unit and mode-aware changes are row
construction and branch changes.

## Four-column mapping (anchor -> landing point -> basis -> evidence)

| Anchor | Landing point | Basis | Evidence |
|---|---|---|---|
| 772-A: the norm-only entry is the accepted norm chain minus the Dot/W rows; commitments, transcript, lookup, PSC, own BAT, FOLD, CS and BASE all run | The plan is norm-only (empty weights, no dot/index-mask/closure); the own two-stage chain runs to BASE/LastBase | The norm-only validation arm; the switch relation and the BASE linear check pass on real rows | Rounds 1-4 PASS; verify warm median 132,403 ms |
| 772-A: the hd binding is the materialized identity; the root comes from a plan-fixed coin over the trace L0 activation rows | The first stage binds the activation root to the output-group entries; the verifier recomputes it and compares before any second-stage work | The root/shape/configuration comparison in the first-stage verify | Negative `source_rows` rejected |
| 772-A: the tail stage parameters are re-derived by the own two-stage planner from the real main/mask codes and k/h, not copied | Stage-0 positions/work 252/24 from the derived fold (k=12, h=7); BASE 259; the candidate report recomputes the bounds | Sweep table: cap 252 gives query 2^-104.00, carried 2^-175.49, joint 2^-128.00, BASE 2^-128.0037; cap 192 needs 2^49 PoW; 160/128 rejected | The candidate-report test; the ledger sweep table |
| 772-A: six segment negatives plus the control; source_rows must cover the materialized identity | Real-verifier matrix: control plus seven negative cases (the six planned cases and the declaration omission) | Every case tampers the artifact and rewrites the manifest/complete marker; the segment checks must reject | Control PASS, seven rejects (`negatives.json`) |
| 752/754/756/761/763 reuse: 4-byte widths with the two-form sum, B1-B3, no-op clamp in chain, fresh merge, 256^j base | RMS sum/mean digit widths sized for the sum-of-squares bound; 256^j unit powers; no-op clamp kept in-chain | round_widths; sparse tail-unit construction; LayerNorm path byte-identical | Rounds PASS; hard gate 20/20 at 45,080 B |
| 782: optimize the lifecycle first; only sweep parameters if the run still OOMs | Sparse tail units (representation only) and the single borrowed witness; the parameter sweep is evidence only and no parameter changed | Lifecycle finding; the sweep shows the next fixed point needs 2^49 PoW | Peak 58,581,632 kB (against 416.9 GB before the representation change) |
| 788(a): check the 761 pre-fold invariant first, then decompose per value | Row scan located 16 bad kind-1 rows; two mode-blind assumptions fixed (centered map and centered-end row) | Resolution of the switch/BASE failure; LayerNorm keeps its terms | Component logs: bad=0, delta [0,0,0], switch relation and LastBase pass |
| 772-2.1: the no-dot declaration enters the statement and the session seed; dot and norm-only cannot masquerade | `statement.norm_only` is required from the trusted policy patch count; the session seed absorbs the declaration; both sides take the segment positions from the patch count | Provider wiring corrections | Negative `declaration_omitted` rejected |
| Hard gate 45,080 B byte-identical | No assembly or serialization path changes | The twenty tracked frozen artifacts | 20/20 at 45,080 B, sha256 `7a5da6eb...` |
| Honest byte account | Carrier 1,644,035,480 B; trace 3,613,502,964 B; consumer proof 2,024,320 B; statement 133,216 B; inputs 131,080 B | The byte-account table above | Total 5,259,831,827 B |

## Boundaries

- Single-machine evidence; no push in this round.
- The carrier rides the proof artifact under its own file name and is bound by
  manifest length and digest; the declaration requirement comes from the
  trusted policy, never from the prover-supplied statement.
- The carrier bytes vary with the commit randomness; the numbers above are the
  round-1 artifact, and the other rounds differ within a few thousand bytes.
- No performance sign-off beyond the recorded stage numbers; the peak figure
  covers the whole acceptance process tree.

## Acceptance record

The timesfm L0 attention RMS pre-norm norm-only segment was approved and
closed on 2026-11-19. The review covered the segment shape (norm-only entry,
the two-stage switch repair, the materialized identity binding with the
source-row negative), the four-round evidence (one cold and three warm, all
passing), the seven-case negative matrix with the control, the hard gate, the
byte account and the stated boundaries. Code commits: `8881961b` (segment
wiring and tests), `111ae1c0` (shared mode-aware norm output rows), `1b1d1f06`
(sparse tail units); evidence commit `220deb06`.

Closed.

## Peak composition and the no-streaming justification

The acceptance peak is the prove stage high-water mark, 58,571,468 KiB; the
verify stage peaks at 37,946,416 KiB, and the whole run (including the test
and stage processes) stays at 58,581,632 kB. The composition is the norm-only
own chain itself: the single owned norm witness (it is never cloned into the
send witness; the second stage borrows it), the member group tables and their
BAT/FOLD/CS rows, and the two canonical message stages. The tail clamp and
padding rows already use the sparse unit representation - each row carries
only its bound-width or pre-clamp digits instead of a dense per-row
coefficient vector - which is what removed the earlier 416.9 GB peak, and no
dense per-row vectors remain on this path.

No streaming rework applies to this segment: it has no output-head table and
no 2^18-level read surface. The largest structures are the 2^17 code domains
and the 2^16-row member groups of this single L0 norm; the gain table has
2,048 columns and the frozen rstd table is read by point lookups. The measured
peak is therefore well below the 150 GiB reporting line, and the streaming
design used by the larger slices has nothing to stream here.
