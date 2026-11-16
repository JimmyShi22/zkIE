# table-key: production plan derivation and a new candidate instance

## Scheme and provenance

- Scheme: table-key (complete key plan from the public configuration and its
  candidate table instance). Code under test `b1cbac0b` (production plan
  derivation). Previous scheme close `9c8104fc` (real second stage over the
  verified shared state).
- Purpose: derive every key-plan shape, code, stage and identifier from the
  trusted public configuration, then explicitly register a new candidate table
  instance for the whole key chain (option (b) of the confirmed boundary).
- Boundary: this trace covers the plan derivation and the instance
  registration. The key proof, the independent verifier and the final
  evidence follow in later traces of the same round. This is not a complete
  one-time authentication and makes no later-batch or single-shot claim.

## Candidate instance

The commit uses the trusted table configuration of the gpt2 projection table.
The commit path samples fresh salts and tails, so the physical table root and
its column ledger are new; the registered identity and configuration are the
public configuration's values.

| item | value |
|---|---|
| commit outcome | PASS, stage 3,828 ms, VmHWM 357,140 KiB |
| new table root | `601d921c…` |
| table identity | `ad0803db…` |
| table configuration | `9017dd1a…` |
| record length | 1024 |
| previous instance root | `f439e4ba…` (kept, not overwritten) |

The registration test verifies the committed meta against the trusted model
identity, the asset spec and the table parameters, and checks the meta's
identity/configuration/shape against the derived plan. The old instance's
root differs; its policy and ledger are untouched. Private randomness and the
public/private commit artifacts stay on disk in this directory (gitignored by
the repository's binary-artifact rule).

## Derived plan

Public geometry: table shape `{log_n:12, ell:2048, r:128, v:10, answers:1,
salt:40}`, one-time distance `m_once=284`, `matrix_rows=1024`, `k_lo=0`,
`columns=1024`, `total_columns=52`, range `d=18`, `mu=5`, `key_bits=20`,
`bit_entries=2^23`, `offset=131072`.

Public object schedule: lookup bits/mask oracle `21/22`, lookup message trees
`101/203`; key group fresh `3` and its fresh mask tree `204`; index-mask
oracle `60`/tree `202`; W oracle `20`/tree `100`; Gu/Omega oracles `10/11`,
trees `300/301`; shared/own fresh masks `200/201`, fresh ids `2/8`; binding
leaves `1`, salt `40`.

Derived: shared and own main code `{18, 16384, 130}`, index-mask and both
group mask codes `{18, 256, 66}` (the index-mask message is one object shared
by the key group and the own group, so the key group takes that mask code by
role); shared/own stages `3/3`; BASE positions `66/53`; index-mask rows `1`;
column draws/budget `52/52`; plan digest `26c6375f…`.

The derivation test asserts the schedule, the codes, the stage counts, the
BASE positions and the table identity/configuration binding. The registration
test additionally checks the committed meta through
`verify_asset_table_meta`.

## Commands and logs

- `commit-gpt2-seq16.log`: the commit invocation and its result.
- `instance/gpt2-seq16/commit-run/result.json`, `progress.jsonl`: the commit
  run records.
- `registration.log`: the derivation and registration test output.
- `instance/gpt2-seq16/parameters.toml`, `anchor.json`: the instance's public
  input files.

## Open items

- The key proof over this instance (first/binding/shared/second), the
  independent `verify_key`/`AcceptedKey` entry and its negative matrix, and
  the release wall/VmHWM evidence are not in this trace.
