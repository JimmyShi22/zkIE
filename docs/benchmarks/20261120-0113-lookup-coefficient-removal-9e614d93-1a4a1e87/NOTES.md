# Lookup coefficient removal (A17, gpt2 A04)

## Scheme and baseline

- Scheme: `lookup-coefficient-removal`.
- Code: `9e614d93`; baseline `1a4a1e87` (the shared read path streaming
  acceptance, carrier sha `3815f1af...`, recorded as pre-A17).
- Purpose: remove the derivable multiplicity payload from the lookup
  outcome. The coefficients on the multiplicity bits
  (`coefficients[plane + planes * row] = 1 << plane`) and the key total
  (the key count) are pure functions of the plan, so the message no longer
  transports them; the prover, the verifier and the decoder derive them
  with one shared rule (`multiplicity_from_plan`). The lookup message
  version advances to 3, the old format is rejected, and the transcript
  domain carries the new version.
- The change is one shared fix in the shared lookup codec; both named
  encoding entries (`LookupOutcome::encode_unchecked`,
  `NormLookupSecond::encode`/decode) and the first-stage digit lookup path
  funnel through it, so the three models inherit the same behavior.

## Method

- Single machine. Four rounds through the acceptance driver (one cold and
  three warm); every round commits fresh and runs the three stages as
  separate processes (the durable one-request column ledger is keyed by
  asset directory and table root with no refund).
- The negative matrix runs through the full public verify path with
  self-consistent manifests; the control must pass first.

## Results

| round | commit ms (VmHWM KiB) | prove ms (VmHWM KiB) | verify ms (VmHWM KiB) | proof bytes | carrier sha256 | carrier bytes |
|---|---|---|---|---|---|---|
| 1 (cold) | 237,611 (27,296,868) | 741,870 (212,584,360) | 188,722 (122,687,616) | 1,978,436,895 | `2c386fb9...` | 96,691,076 |
| 2 | 240,467 (27,297,052) | 743,402 (212,586,940) | 188,801 (122,621,128) | 1,978,431,969 | `2c386fb9...` | 96,691,076 |
| 3 | 234,389 (27,296,948) | 744,504 (212,587,064) | 189,000 (122,645,004) | 1,978,444,400 | `2c386fb9...` | 96,691,076 |
| 4 | 237,555 (27,297,008) | 742,410 (212,584,248) | 187,740 (122,697,756) | 1,978,433,597 | `2c386fb9...` | 96,691,076 |

- All rounds PASS. Warm medians: commit 237,555 ms, prove 743,402 ms
  (VmHWM 212,586,940 KiB), verify 188,801 ms (VmHWM 122,645,004 KiB),
  proof bytes 1,978,433,597 B.
- Carrier sha256 in every round:
  `2c386fb924fb9c3dde97752b0da09be7a980acf10f6f0bb0579d2ff2ef068f6b`,
  size 96,691,076 B - the post-A17 carrier invariant (commit-invariant,
  as its pre-A17 predecessor was).
- Semantic zero change (hard acceptance): `output_binding = expected` and
  the authenticated output digest
  `135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280`
  (16 elements) in every round, bit-identical to the pre-A17 round.
- Trace `proof.bin` 1,879,423,254 B (same serialized length; values differ
  per commit as before); `statement.bin` 463 B and `public-inputs.bin`
  72 B unchanged; the statement digest is identical across the four
  rounds.
- Negatives (real verifier): control PASS; `statement_identity`,
  `statement_source_identity`, `version`, `reader_root`, `record_digest`,
  `messages_truncated` and `source_rows` all rejected.

## Byte account

- Pre-A17 carrier: 1,707,601,596 B, sha256
  `3815f1af7ac5913195930449a8ecbf61175aa6df2e20595df549b21367d16819`
  (the invariant is retired by this change).
- Post-A17 carrier: 96,691,076 B. Delta: -1,610,910,520 B (-94.34%).
- Composition: the frozen rstd lookup instance removes 8 planes x 2^23
  rows x 24 B = 1,610,612,736 B; the byte table lookups remove about 48 KiB
  each; every lookup outcome also drops its 8-byte length and 24-byte total
  header; the version bump re-draws lookup challenges, so small
  presentation differences remain. The measured delta is the recorded one.
- Proof bytes per round drop from 3,589,344,213 to about 1.978 GB
  (-1,610,910,000 B level, consistent with the carrier delta).

## Gates

- Workspace: 56 suites, 1,319 passed, 0 failed, 32 ignored (release).
- Hard gate: 45,080 B sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (byte-identical; the gate path does not touch the lookup codec).

## Performance (same machine, pre-A17 single round as the reference)

- prove 755,339 -> 743,402 ms (-1.6%), VmHWM 212,621,908 -> 212,586,940
  KiB (-34,968 KiB, noise level); verify 191,993 -> 188,801 ms (-1.7%),
  VmHWM 125,784,612 -> 122,645,004 KiB (-3,139,608 KiB, -2.50%). The
  verify drop is the carrier payload the verifier no longer holds or
  decodes; no further reduction is claimed.

## Boundaries

- gpt2-seq16 at 16 positions. The gemma3 and timesfm consumers pick up the
  same codec by cherry-pick and are verified on their own retests.
- The version bump advances the lookup transcript domain, so all lookup
  proofs differ from the pre-A17 bytes; the semantic checks (output binding
  and output digest), the byte identity of the plan-derived payload and
  the component tests are the zero-change evidence.
