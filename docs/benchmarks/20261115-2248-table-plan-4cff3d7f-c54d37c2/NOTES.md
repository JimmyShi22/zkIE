# table-plan: first candidate instance (superseded)

## Scheme and provenance

- Scheme: table-plan (public-configuration-derived encoded table plan).
- Code: `4cff3d7f` on top of the accepted `c54d37c2` close. This directory is
  the first candidate instance of the round; it was superseded by
  `docs/benchmarks/20261116-0012-table-plan-fbf98f2f-f91f0c1d`.
- Purpose: commit and register a candidate instance whose encoded table plan is
  derived from one fixed public candidate, then run the existing `SharedTrace`
  prove/verify path with an explicit candidate verification policy. This
  directory records the instance and its full-model cross checks; it does not
  contain a local table check prefix.

## Instance

| Case | commit | anchor | table root | ledger |
|---|---|---|---|---|
| gpt2-seq16 | PASS 3,862 ms, VmHWM 357,680 KiB | `fe8a72cd…ffbed` | `46773cc3…f390` | 0 |
| gpt2-seq512 | PASS 3,864 ms, VmHWM 357,480 KiB | `9dc8d6c9…73080` | `6251981d…2e6278` | 0 |

- The first anchor attempt used the wrong asset root and failed; the recorded
  logs show the successful rerun against the model asset root.
- The candidate policy is `instance/case.policy`; both anchor fields of each
  case block carry that case's anchor.
- The derived table plan identity equals the committed `meta.identity`
  (asserted at the later local entry after the plan rebuild).

## Full-model cross checks

- Baseline policy over the new instance: prove PASS, verify FAIL
  `statement_mismatch` (the old configuration rejects the new instance).
- A first candidate-policy attempt still failed verify because the limited
  table identity anchors had not been replaced; the run summary records that
  failure and the corrected rerun is `run2`.
- `run2` with the candidate policy: prove PASS and verify PASS with the
  expected output binding.
  - seq16: prove result.json `proof_bytes` 1,887,840,915; 14,041 ms;
    VmHWM 3,897,632 KiB; verify
    `135c4d1c…a80` over 16 elements / 12,038 ms / 3,791,668 KiB.
  - seq512: prove result.json `proof_bytes` 6,798,965,242; 44,508 ms;
    VmHWM 18,922,904 KiB; verify
    `eac0949d…fdd` over 512 elements / 29,659 ms / 14,089,508 KiB.
- The `proof_bytes` values are `result.json` execution statistics; they are
  not the `proof.bin` file size. The protected 45,080 B object is untouched.

## Why superseded

The local first+shared prefix plan of this candidate was still assembled from
scaffold components (chain mask blocks and fixture index rows) and was not
derived from the same public configuration as the registered object; its local
prefix never closed. The candidate was therefore replaced by a fresh instance
under the public-configuration-derived plan. No production ledger was charged
for this instance: both table ledgers remain at 0, and the failed attempts used
temporary ledgers outside the asset root. Nothing was reset.

## Archive

All textual artifacts of this directory are committed. The gitignored binaries
(commit public/private bundles and proof files) remain on disk in this
directory. No textual trace was deleted.
