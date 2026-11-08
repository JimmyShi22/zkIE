# Project Development Conventions

## What this is

Three-stage verifiable ML inference (commit / prove / verify as independent
processes). Implementation details live in the code and docs/; this file does
not describe the implementation and does not change when the design evolves.

## Confidential local rules directory

Before starting any work item, read **every file** under `.dev-method/` at the
repository root and follow them: they carry the team collaboration protocol,
the working method for the current effort, and the channel rules between roles.
The directory is confidential, gitignored, never committed and never pushed; it
exists on the local machine only. If the directory or a file in it is absent
(for example in a clone that does not have it), ignore that part and follow this
file alone. Never copy its contents out, never reference its path in any
committed artifact, report, or external message.

## Confirmation before starting

No work item starts before the Tech Lead has confirmed its approach with the
technical expert through the local channel. Items without that confirmation stay
parked. PM coordinates the sequence and records the outcome; technical content of
the confirmation is not routed through other roles.

## Entry points

These two entries are the only ones in the project. New entries must be
registered here:

| Entry | Location | Purpose |
|---|---|---|
| zkie CLI | zkie/src/main.rs | three-stage commit/prove/verify + matrix/run |
| zkie-check CLI | tools/zkie-check/src/main.rs | acceptance report |

## Directory layout

```
crates/zkie-core/   foundations: field/fixed_point/codec, artifact/progress/error/types,
                     compute, storage, oracle, claim
crates/zkie-engine/ execution environment: primitives, storage::mmap, compute::cpu,
                     context, metrics
crates/zkie-ops/    operators and proofs: op/registry, pipeline, proof/*, live operator modules
models/{gpt2,gemma3,timesfm}/  config + executor (weights always external)
zkie/               the main three-stage CLI (commit/prove/verify + matrix/run)
tools/              zkie-check
scripts/            data pipeline (extraction / simulation / GT), not the proof path
docs/               benchmarks.md; spec / roadmap are TODO placeholders;
                     docs/benchmarks/ test trace archive (iteration tree)
```

## Build and test methods

- Build: `cargo build --workspace` / `cargo check --workspace`; for real runs use
  release (`cargo build --release -p zkie`).
- Tests: `cargo test --workspace` all green is the acceptance gate; unit tests go
  in-module under `#[cfg(test)]`, integration tests under `tests/`; negatives must
  be truly injected (serialize round-trip tamper, then the full verification path
  must reject) - adding a negative without verifying it is not allowed.
- Three-stage runs: `zkie matrix` (1 cold + 3 warm); numbers are measured on
  the same machine, warm medians go into the ledger, cold-run numbers do not.
- Profiling: `zkie profile-matrix --cases <case.toml>... --threads 1 8 64
  --runs 3 --out <dir>` - stage level (commit/prove/verify wall time and peak
  memory) plus operator-level and kernel-level cumulative wall time and call
  counts; progress.jsonl must carry real elapsed times and done/total units;
  output is a machine-readable record (thread tier x stage x operator) plus a
  human summary (operator share ranking); same measurement basis as matrix; all
  three thread tiers must produce numbers, to locate serial bottlenecks.
- Test trace archive (iteration tree, mandatory, committed): every test/bench run
  creates a directory under `docs/benchmarks/`:
  `<YYYYMMDD-HHMM>-<scheme>-<current-short-commit>-<previous-short-commit>/`
  The directory must contain, committed to git:
  - `NOTES.md`: scheme name, which change and which scheme/baseline (previous
    commit) the run is based on, purpose, conclusions and open items, key numbers
    (proof bytes / wall time / VmHWM / tests passed);
  - all textual result artifacts: result.json, progress.jsonl, profile.json,
    summaries, logs - nothing is deleted.
  Every run gets a new directory (multiple rounds of one scheme = multiple
  directories), forming a traceable scheme iteration tree; oversized binary
  artifacts (e.g. proof.bin, gigabytes) stay on disk in the same directory and
  are gitignored, everything else is committed with the scheme. When benching
  for a new scheme starts, the gitignored binary artifacts left by the previous
  scheme's rounds may be deleted to free space; textual traces are never
  deleted, and the deletion is recorded in the new round's NOTES.md. Numbers
  without provenance must not enter any document.

## Development rules (hard)

1. Single implementation: one concept, one implementation; no legacy
   implementations, duplicate paths, or leftover legacy modules.
2. No external source annotations in code: code, comments, docs, and commit
   messages never carry any external source clues (names, abbreviations, paths,
   file:line, numbering markers) - keep the implementation self-contained and let
   the AI focus on the code itself; self-check before committing.
3. Commits: English messages; one commit per small step; TL review before push.
4. Asset red line: model weights / GT / assets are read-only and external, read
   via mmap/ranges; committing them, modifying them, or materializing them whole
   is forbidden.
5. Measurement discipline: optimizations need same-machine before/after
   comparison (proof bytes, wall time, VmHWM); bytes must not grow, VmHWM must
   not rise, wall time within 3% noise; no gain is recorded honestly as
   "no gain observed"; never bend the measurement basis to hit numbers; archive
   every result and note which change and scheme it belongs to.
6. Docs in sync: deleting a feature deletes its documentation; docs/ matches the
   code.
7. Docs in English: every documentation file (README, docs/, notes, tables) is
   written in English; non-English text found in existing docs is translated
   before it is committed.

## Workflow red lines

- Any new mechanism must derive from some source, unless it is a scheme already
  explicitly approved by the boss or an expert.
- No self-invented equivalent mechanisms to pass gates.
- When technically unsure, report to the boss; never change the scheme on your
  own.
