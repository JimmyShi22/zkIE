# Request consumer chain over the committed table

- Scheme: `asset-consume` (the request-level consumer chain: executor witness,
  proof carrier and the production verify entry).
- Code: `9d6fc86b`; previous commit `8b66f51f` (the consumer matrix reads
  streamed one stored row at a time); branch `refactor/v2`.
- Purpose: one real gpt2 request's twelve `L0_k_w` head consumers are proved
  and verified by the request-level consumer chain over the committed encoded
  table, with the asset mapping streamed one stored row at a time. The
  request proof file travels beside the proof artifact inside the same
  manifest, and the production verify entry refuses a request without it.
- Boundary: local wiring of the consumer chain. The trace still carries the
  `L0_k_w` asset opening; this round does not delete it and does not claim a
  production-level total acceptance for the replaced asset. Qualification
  (`verify_key`/`AcceptedKey`), the pipeline sidecar version/length binding
  and the opening removal are the following increments.

## Change

- Executor: at the layer-0 key projections the executor captures the
  activation rows and, for every head, the raw accumulator and output of the
  production projection kernel. The request witness is converted from exactly
  those values (the exact-rounding remainders are re-derived and checked), so
  the consumer relation never reads a second, differently rounded copy.
- Witness assembly: one private request seed is split by domain into the raw
  and bit value commitments, the one-time mask message (PSC hiding entries,
  shown entry, eval entry and the raw relation slots), the W commitment, the
  binding salt, and the shared and own run seeds. The plan member and target
  are derived on a scratch chain with the stage's own prefix.
- Table handling: the encoded table is rebuilt from the commit's private
  randomness over the real asset mapping (one stored row at a time) and its
  root/record/config must equal the committed metadata; the keeper charges one
  fixed batch of 52 columns against the durable ledger. The PSC, W derivation
  and pending check stream the stored-record matrix view; no whole-matrix
  buffer is materialized.
- Carrier: the request proof file carries the prover-dependent plan values
  (member, target, group mask root, committed raw and bit roots) in a
  canonical section under the container digest (file version 2). The provider
  writes it beside `proof.bin` and registers its relative path, length and
  digest in the proof manifest; prove metrics record its bytes separately.
- Verifier: the request plan structure is rebuilt from the trusted model plan
  (the twelve contiguous head obligations of the registered steps), the
  committed table metadata, the validated trace outputs (activation, output
  rows and their digests, integer bound) and the fixed later plan; the file's
  plan values are only commitments checked by the first chain and the group.
  The verifier then runs first, binding, shared and second in the global order
  before the request can be accepted.
- Identity and order: the consumer stages bind the committed reader identity
  (root, configuration, shape, record), not the plan-level table identity. The
  isolated fold handoff lays main blocks in descending size, so the group's
  BAT order follows the fixed plan; the production plan builds the bit members
  first (raw members after), while component fixtures keep their own plan
  order. Both sides derive the order from the same plan.

## Real run

Production pipeline through the provider: commit, prove and verify on
`gpt2-seq16` from `/data/i1-cases/gpt2/case.toml` over `/data/zkie-assets`,
freshly committed for this run (new table instance).

| item | value |
| --- | --- |
| case | gpt2-seq16 |
| commit | 3,882 ms |
| prove | 34,605 ms |
| verify (accepted run) | 17,188 ms |
| proof.bin | 1,887,836,942 B |
| statement.bin | 214 B |
| consumer_proof.bin | 2,306,616 B |
| VmHWM (prove/verify process) | 5,922,084 KiB |
| model id | `c2971c0179463bf12aecaefc3488417228008c1314e6905c49a55078242c1268` |
| anchor id | `5570cea28761b8f37524bba72b336862cf451d2a1cce96e3d76cb3e5adb88a36` |
| table root | `8e839f5bf7b2a3752dd7669e32fa4ee978c5817d94a66d9000d7822561b5efba` |
| table record digest | `1b855d3142bd7d045f96804ca51b7bf01ea0132046d7fbcb2355fc42a8aa0048` |
| table configuration | `94dd0177b5a08d3e328a51222a4ab5aee7d6a982b91964b072eea7e75ae0ae925` |
| table shape | log_n=12, ell=2048, r=128, v=10, answers=1, salt=40, m_once=m_later=284, k_lo=0 |

These are single release-run measurements on this machine, not a stability
claim. The consumer proof bytes are new and are not netted against any removal
in this round; the hard gate (45,080 B) is untouched.

## Negatives

The production verify entry was driven six ways after the accepted run; every
one was refused:

| injection | result |
| --- | --- |
| request proof file removed | refused (fail closed) |
| plan member changed (one psi coordinate) | refused |
| plan target changed (plus one) | refused |
| group mask root changed | refused |
| one committed raw root changed | refused |
| one container byte changed | refused |

## Tests

- `cargo test --workspace`: 54 suites, 1,284 passed, 0 failed, EXIT 0 on this
  tree (`workspace-tests.log`). The hard-gate test is part of the suite and
  reports the unchanged 45,080 B artifact.
- Release evidence: `cargo test --release -p zkie --test
  consumer_request_evidence -- --ignored` — 1 passed (122.89 s wall for the
  test, including the accepted run and the six negatives); `run.log`.

## Open items

- Qualification: key bundle in the commit public bundle and manifest, a real
  `verify_key` per request into `AcceptedKey`, and the field-by-field match
  against the consumer table; separate key time/byte accounting.
- Pipeline: the sidecar version/length binding across the three CLI stages
  and the first→binding→shared→own→total carriage order on the request proof.
- Opening removal: the twelve registered projection steps must consume the
  consumer chain instead of the `L0_k_w` opening; only then may the opening be
  deleted, with the deletion negatives.
