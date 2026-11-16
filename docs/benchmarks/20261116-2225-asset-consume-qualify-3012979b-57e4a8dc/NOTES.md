# Table qualification and artifact-bound request proof

- Scheme: `asset-consume` (the one-time table key qualification and the
  artifact manifest binding of the request proof).
- Code: `3012979b`; previous commit `57e4a8dc` (the request consumer chain
  evidence); branch `refactor/v2`.
- Purpose: the registered encoded table's key qualification is proved once at
  commit and published in the commit's public bundle; every request re-verifies
  that same proof through the public entry into an in-memory accepted key and
  compares every accepted field with the committed table before consuming it.
  Both artifacts are read through their manifests, so the request proof file
  and every stage input are bound by artifact version, byte length and content
  digest. Key proof and key verification time and the key bundle bytes are
  recorded separately.
- Boundary: no `L0_k_w` opening is removed and no production-level total
  acceptance is claimed. The `first → binding → shared → own → total` order of
  the request chain is unchanged; the key qualification runs before the trace
  and the request stages.

## Change

- Key bundle: the commit stage proves the table's complete key chain over the
  same committed instance (the streaming table keeper rebuilt from the commit's
  private randomness, charged once against the durable column ledger) and
  publishes the canonical bundle as `public/asset-key.bin`, registered in the
  commit manifest with relative path, byte length and digest. The prover
  self-verifies the bundle before publishing. The witness reads the asset one
  stored row at a time; the full asset is never materialized.
- Per-request qualification: prove and verify both require the bundle and its
  manifest entry (missing, swapped or truncated bytes fail closed). Verify
  re-runs `verify_key` per request with a fresh plan, producing an in-memory
  accepted key whose plan digest, table identity, configuration, root, record
  digest and range digest are compared with this commit's table; the key chain
  itself checks the accepted reader against the plan. Re-verification spends
  only the plan's in-memory read ledger, never the durable column budget, and
  the qualification proof is never regenerated per request.
- Manifest binding: the commit and proof manifests must carry the supported
  artifact schema, protocol and backend versions and the request's model id.
  Every file the verifier reads (model commitment, provider config, table
  metadata, statement, public inputs, request proof file and key bundle) must
  match its manifest length and content digest; the large proof is hashed in
  bounded chunks. The request's auxiliary proof file is decoded only after its
  manifest check and then verified in the same global order.
- Accounting: `key_prove_ms`, `key_verify_ms` and `key_bytes` are separate
  counters; every stage writes its protocol counters to `metrics.json` beside
  the run report.

## Real run

Production pipeline through the provider: commit, prove and verify on
`gpt2-seq16` from `/data/i1-cases/gpt2/case.toml` over `/data/zkie-assets`,
freshly committed for this run (new table instance). The release evidence test
runs all stages in one process; the memory high-water mark below is cumulative
across those stages, not a per-stage figure.

| item | value |
| --- | --- |
| case | gpt2-seq16 |
| commit | 252,965 ms (key proof 190,178 ms + key self-verify 58,920 ms inside) |
| prove | 91,530 ms |
| verify (accepted run) | 77,180 ms (key re-verification 58,824 ms inside) |
| proof.bin | 1,887,836,942 B |
| consumer_proof.bin | 2,310,632 B |
| asset-key.bin | 4,549,664 B |
| VmHWM (process, commit+prove+verify) | 35,497,036 KiB |
| model id | `c2971c0179463bf12aecaefc3488417228008c1314e6905c49a55078242c1268` |
| anchor id | `eecb36fb4b8163ce99c962c379a03753e89443a9afe832c3672411c834e309d9` |
| table root | `e5eb25761773902f56460eeb7a36f780fc8dfdd6832c0cd311c993c945d5ff0d` |
| table record digest | `3494d7628807d72e008ad76e83626ec3091ece68015e6c3761df0106885f8bb0` |
| table configuration | `94dd0177b5a08d3e328a51222a4ab5aee7d6a982b91964b072eea7e75ae0e925` |
| table shape | log_n=12, ell=2048, r=128, v=10, answers=1, salt=40, m_once=m_later=284, k_lo=0 |

These are single release-run measurements on this machine, not a stability
claim. The key costs are real end-to-end costs: the one-time qualification is
paid at commit, and every request pays the re-verification. The hard gate
(45,080 B) is untouched.

## Negatives

The production verify entry was driven eight ways after the accepted run;
every one was refused:

| injection | result |
| --- | --- |
| request proof file removed (manifest lists it) | refused |
| plan member changed, proof manifest internally consistent | refused |
| plan target changed, proof manifest internally consistent | refused |
| group mask root changed, proof manifest internally consistent | refused |
| one committed raw root changed, proof manifest internally consistent | refused |
| one container byte changed, proof manifest internally consistent | refused |
| key bundle removed from the commit public bundle | refused |
| key bundle byte changed, commit manifest internally consistent | refused |

The changed-value negatives rewrite the manifest and its `COMPLETE` marker, so
the refusal comes from the protocol checks (plan member/target/roots, key
qualification), not from the container digest alone.

## Tests

- `cargo test --workspace`: 54 suites, 1,284 passed, 0 failed, EXIT 0 on this
  tree (`workspace-tests.log`); the hard-gate test reports the unchanged
  45,080 B artifact.
- Release evidence: `cargo test --release -p zkie --test
  consumer_request_evidence -- --ignored` — 1 passed (890.40 s wall: the
  accepted run plus eight negatives, each re-running the per-request key
  verification and the trace verification); `run.log`.

## Open items

- The request chain still verifies the `L0_k_w` asset opening; removing it,
  switching the twelve registered projection steps to the consumer chain and
  the deletion negatives remain the next deliverable.
- The per-request key re-verification dominates the verify wall time; any
  change to that cost is a protocol/parameter decision outside this round.
- seq512 with the qualification path has not been run.
