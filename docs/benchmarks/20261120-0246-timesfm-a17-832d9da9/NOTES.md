# timesfm A17 acceptance (lookup derivable multiplicity removal)

- Scheme: A17 - remove the derivable lookup multiplicity coefficient payload
  from the proof volume. The shared fix is one commit from the lane that owns
  the shared codecs; this branch only carries the cherry-pick, no separate
  implementation.
- Cut: the lookup outcome carries the key claim, theta, the row outputs and
  the slot lists; the multiplicity coefficients over the bits and the key
  total follow from the plan alone. The prover, the verifier and the message
  decoder now derive them with the one shared rule, the lookup message version
  advances, and the old format is rejected by the version tag and the
  trailing-byte check.
- Branch: base `5a51321e`; shared fix cherry-picked as `832d9da9`
  (`LOOKUP_VERSION` advances; single file, +51/-40); the timesfm carrier
  reaches it only through `NormPscSecond`'s lookup and rstd `NormLookupSecond`
  messages, so no timesfm-specific code changed.

## Build provenance

- Tree: the timesfm work tree at `/data/zkie-timesfm-first-asset`, HEAD
  `832d9da9496817b55bb8fc68672195f8ef06f67c`.
- Target directory: `/data/zkIE/target`; binary `/data/zkIE/target/release/zkie`,
  sha256 `ead9b56d7fabb5e60bb7be39182ff6eec72ceebffbb46d7bbb2016f3184dae16`
  (the acceptance harness ran exactly this path; the hash is recorded in
  `build-provenance.txt`).
- Built at 2026-11-19T18:46:41Z from the clean tree above.

## Round numbers

| round | commit ms | prove ms | verify ms | artifact bytes | prove vmhwm KiB | verify vmhwm KiB |
|---|---|---|---|---|---|---|
| 1 (cold) | 151,704 | 366,862 | 131,585 | 3,649,177,516 | 58,575,320 | 34,800,664 |
| 2 | 151,398 | 366,175 | 131,467 | 3,649,177,777 | 58,568,340 | 34,803,264 |
| 3 | 154,012 | 366,330 | 130,953 | 3,649,176,835 | 58,570,492 | 34,804,244 |
| 4 | 152,584 | 364,995 | 131,804 | 3,649,180,100 | 58,575,220 | 34,793,972 |

Warm medians (rounds 2-4): commit 152,584 ms, prove 366,175 ms, verify
131,467 ms. Peak resident set size across all stages: 58,575,320 kB (GNU
time, includes the stage children).

## Semantic zero change (hard item)

All four rounds report `output_binding=expected` and the same authenticated
output digest
`cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3` - bit for
bit the digest recorded before A17. The wrapper contains no behavior change;
the removed payload is regenerated before the same verify relation runs.

## Byte account (round 1) and the delta

| file | bytes |
|---|---|
| proof/statement.bin | 133,216 |
| proof/public-inputs.bin | 131,080 |
| proof/timesfm-norm-segment-carrier.bin | 33,378,520 |
| proof/proof.bin | 3,613,502,964 |
| proof/consumer_proof.bin | 2,026,976 |
| proof/manifest.json | 4,695 |
| proof/COMPLETE | 65 |
| total (the proof-bytes metric) | 3,649,177,516 |

The carrier is the whole change: pre-A17 it was 1,644,035,480 B with sha256
`35e9cb87acd94b7755ef82dac56e8afac6fcf59eb4bacf19cccf1ada099a538b`; post-A17 it
is 33,378,520 B with sha256
`c30f942bbaec6bac3ec2576569311217482ab1b8513e25078d6d6008519e94ef`. The measured
carrier delta is **1,610,656,960 B** (estimate 1,610,612,736 B; the difference
is commit randomness). The round-1 artifact total drops from 5,259,835,925 B
to 3,649,177,516 B (delta 1,610,658,409 B; the remainder beyond the carrier
delta is the randomly sampled consumer proof and manifest sizes).

## Gates

- Verifier PASS with `output_binding=expected` and the unchanged GT digest in
  every round; workspace: 61 suites, 1,337 passed, 0 failed.
- The hard gate was not touched: the twenty tracked `proof-45080.bin` copies
  are still 45,080 B each with sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` (the norm
  path carries no lookup message, so the A17 change does not reach it).
- Peak memory stays in the same band (58,575,320 kB); the verify stage drops
  from about 37.9 GB to about 34.8 GB, consistent with no longer holding the
  removed payload buffer. A17 is a proof-volume change, not a prove-peak
  optimization.

## Boundaries

- Single-machine evidence; no push in this round.
- Time and memory are recorded, not signed off; prove and verify walls sit
  slightly below the pre-A17 rounds because less payload is written and read.
- The cut is the shared commit only; no timesfm-specific encoding path was
  added or changed.
