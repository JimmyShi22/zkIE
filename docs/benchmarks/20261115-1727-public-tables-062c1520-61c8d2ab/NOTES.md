# Trusted public tables outside the proof trace

- Scheme: `public-tables-b4` (the exp/rsqrt/gelu/silu migration to trusted
  public file sources; the plan source and the verifier-side plan landed in
  `948c60e7`/`21567071`/`7ed16d18`, this round carries the change through the
  production executors and the proof trace).
- Code: `062c1520`; previous commit `61c8d2ab`; branch `refactor/v2`.
- Purpose: the migrated tables are trusted public files. The trace must not
  carry their values or openings, and the verifier must take their values
  from the trusted public root after byte-checking the fixed identity.

## Change

- Executors (gpt2, gemma3, timesfm): a public trace built by
  `execute_shared_with_progress_public` opens only the non-public assets, and
  the trace pool index is the position in that filtered list. The
  exp/rsqrt/gelu ports (and the timesfm silu port) carry no external pool;
  every other asset keeps its own opening and pool. The component entries
  keep the unfiltered trace.
- The provider prove path uses the public executor entries; the verify path
  already uses the public plans and takes the asset root from the public case
  config. That root only locates the files; the expected digest and entry
  count are fixed constants in the model plan code, never read from the
  bundle, the proof, or a freshly opened file.
- Verifier: one public table is read from `root/locator` once per
  verification run. The byte length must equal `entries * 4`, the SHA-256
  must equal the digest fixed in the model plan code, and the bytes are
  decoded as signed i32. The
  cache key is the trusted identity `(digest, entries)`, never a proof value.
  The decoded values enter the relation check as materialized plan inputs,
  the same way the other non-external plan inputs do.
- The model commitment still lists the four files as committed assets; this
  round removes them from the trace, not from the anchor. The registered
  policy therefore stays valid.

## Real public files, byte-checked

| table | entries | bytes | sha256 |
| --- | ---: | ---: | --- |
| exp_table_i32.bin | 2,097,152 | 8,388,608 | `c6a3e6fae423e86bc2598392e4055c938381616c738414fa8bad859da4004f55` |
| rsqrt_table_i32.bin | 7,340,032 | 29,360,128 | `7cba6a7c88053b99112c1271a2d593e1147a4b3b4a92fe1091c148e2e33573ed` |
| gelu_table_i32.bin | 16,777,216 | 67,108,864 | `42cd0f218e5635bb2c1e94076f8f5da3d92456ce30960b1463774dca2dad3e5d` |
| silu_table_i32.bin | 1,048,576 | 4,194,304 | `107fba3ab21394f02ff1869ca8aaeda98f70f092a96e61bcf646cb53af19d6c7` |

The sizes are exactly `entries * 4`; `table-byte-check.txt` holds the raw
`sha256sum` and `stat` output.

## Negatives

`cargo test -p zkie-ops --test plan_check`: 21 passed. The public-table tests
serialize the trace, deserialize it, and run the full `verify_shared_trace`
path:

| Test | Injection | Rejection |
| --- | --- | --- |
| `public_table_source_verifies_from_the_trusted_file` | none; the table is read twice | accepted, no opening carried |
| `public_table_without_a_trusted_root_is_rejected` | verification without a root | refused |
| `public_table_file_swap_is_rejected` | same-size different bytes in the public file | refused |
| `public_table_truncated_or_padded_file_is_rejected` | one byte removed / one byte added | refused |
| `public_table_missing_file_is_rejected` | public file removed | refused |
| `public_table_wrong_fixed_identity_is_rejected` | fixed entries or fixed digest changed | refused |
| `public_table_value_mismatch_is_rejected` | file digest valid, one looked-up value differs from the proved input | `invalid_proof` |
| `prover_supplied_table_opening_is_rejected` | table opening attached and the port names a pool | `statement_mismatch` |
| `public_table_short_table_rejects_index_past_the_end` | three-entry trusted table, the proof reads entry 3 | refused (no clamp) |
| `public_table_wrong_output_is_rejected` | recorded argmax changed | refused |

## Measurement

These are single before/after measurements on this machine; the percentage
ratios describe these single runs and are not a claim of a stable
improvement rate. Same machine, release builds; both sides use the same
registered commit, the same case file, the same parameters
(`profile = "case"`), input and policy.
Only the binary differs: the baseline worktree at `61c8d2ab` versus the
`062c1520` build. The gpt2 registered commit from the registration trace is
reused by both sides; for gemma3 and timesfm (no random table instance) the
commit is deterministic and is created once by the after side, then reused by
the before side.

| case | stage | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| gpt2-seq16 | commit | reused | reused | no commit stage |
| | prove wall | 14,655 ms | 14,246 ms | -409 ms (-2.8%) |
| | prove VmHWM | 4,488,956 KiB | 4,088,916 KiB | -400,040 KiB (-8.9%) |
| | verify wall | 17,607 ms | 12,084 ms | -5,523 ms (-31.4%) |
| | verify VmHWM | 4,102,616 KiB | 3,895,816 KiB | -206,800 KiB (-5.0%) |
| | proof.bin | 2,190,155,518 B | 1,980,401,278 B | -209,754,240 B (-9.6%) |
| gemma3-seq16 | commit | reused | 9,558 ms / 1,085,692 KiB | created once, then reused |
| | prove wall | 35,334 ms | 34,982 ms | -352 ms (-1.0%) |
| | prove VmHWM | 12,718,900 KiB | 12,379,076 KiB | -339,824 KiB (-2.7%) |
| | verify wall | 30,132 ms | 26,891 ms | -3,241 ms (-10.8%) |
| | verify VmHWM | 10,949,088 KiB | 10,711,320 KiB | -237,768 KiB (-2.2%) |
| | proof.bin | 6,223,293,366 B | 6,013,555,254 B | -209,738,112 B (-3.4%) |
| timesfm-16patch | commit | reused | 5,539 ms / 175,604 KiB | |
| | prove wall | 23,724 ms | 23,686 ms | -38 ms (-0.2%) |
| | prove VmHWM | 8,207,512 KiB | 8,044,568 KiB | -162,944 KiB (-2.0%) |
| | verify wall | 22,200 ms | 21,266 ms | -934 ms (-4.2%) |
| | verify VmHWM | 4,209,436 KiB | 4,112,944 KiB | -96,492 KiB (-2.3%) |
| | proof.bin | 3,824,637,660 B | 3,740,667,388 B | -83,970,272 B (-2.2%) |

The gpt2 registered commit's own preparation cost is not part of these runs;
the registration trace records it at 3,815 ms / 355,520 KiB for this case.

Net bytes per migrated table (serialized values, eight bytes per entry):

| table | value bytes |
| --- | ---: |
| exp | 16,777,216 |
| rsqrt | 58,720,256 |
| gelu | 134,217,728 |
| silu | 8,388,608 |

- gpt2 removes exp+rsqrt+gelu: 209,715,200 value bytes of the 209,754,240 B
  delta; the remaining 39,040 B is the removed opening metadata (identity,
  link, roots, row bookkeeping) and the dropped external port entries.
- timesfm removes exp+rsqrt+silu: 83,886,080 value bytes of the 83,970,272 B
  delta; the remaining 84,192 B is the same kind of metadata.
- gemma3 removes exp+rsqrt+gelu: 209,715,200 value bytes of the 209,738,112 B
  delta; the remaining 22,912 B is the same kind of metadata. The metadata
  residual differs per model because the link and row bookkeeping depend on
  each model's anchor leaf set and asset rows, not on the table values.
- Carried trace assets read from the bincode header: gpt2 199 -> 196,
  gemma3 245 -> 242, timesfm 241 -> 238 (`proof-assets.txt`).
- First preparation cost: each table is read and SHA-256 checked once per
  verification run. A raw read+SHA-256 of the three gpt2 tables measured
  0.07 s with a warm page cache; that is only the raw read and hash, not the
  full first materialization. The full first materialization (open, digest
  check, signed decode to i64, optional layout) is not separately timed and
  is included in the single verify wall numbers above. Every later input
  reuses the checked values.

## Evidence

- `workspace.log`: `cargo test --workspace` 1151 passed / 0 failed /
  20 ignored over 51 suites.
- Hard gate: `/tmp/zkie-proof-assembly.bin`, 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  unchanged; the gate test is inside the green workspace run.
- `table-byte-check.txt`: the real files match the fixed sizes and digests.
- `before-*`/`after-*` run logs, per-stage `result.json`, `progress.jsonl`
  and `policy.toml` copies; `run-summary.txt` has the raw stage records. The
  `proof.bin` files are large binaries kept in the trace directories and
  gitignored. `run-summary.txt` keeps a failed gemma3 before line from an
  earlier attempt that referenced a commit directory removed during cleanup;
  the measured pair below is the complete rerun.

## Boundaries and open items

- The four tables stay in the committed asset list; the anchor and the
  registered policy are unchanged. This round stops the trace from carrying
  them and makes the trusted public file the value source.
- The rsqrt extension and the RoPE tables keep their original sources and
  verification.
- The full five-case matrix and the seq512 cases were not rerun; the routine
  self-check covers the three seq16 cases. The change is plan-level for the
  same four files.
- One fresh commit samples new private table randomness, so its anchor cannot
  match the registered policy; the registered instance is reused for the
  gpt2 measurement. This is the existing registration model, not a change of
  this round.
- No asset or ground-truth file was modified, regenerated or committed. Not
  pushed.
