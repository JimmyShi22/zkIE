# Independent commit, prove and verify processes

## Scheme

- Scheme: `lm-head-stages`.
- Current code: `04ae4cd9`.
- Previous code: `491b01c2`.
- Scope: the real GPT-2 LM-head local chain. The production three-stage
  command is unchanged; this is the local stage harness.

## Purpose

Run the three stages as separate processes with a public/private handoff,
measure each stage's wall time, peak memory, I/O and temporary bytes, and
reject the handoff, private-state and budget failure classes.

## Boundary

Public handoff:
- version, protocol identity and the canonical statement;
- the public reader (root, configuration, shape, record);
- plan identifier, case id and session random bytes;
- the base plan and position count;
- the old-mask plan and root;
- the fresh-mask plan and every switch plan (target, tree plan, stage FOLD
  shape/layout/mask tree and group/member/fresh ids).

Private prove state (owner-only directory and file):
- the table commitment state: public reader and name, column leaves, private
  salts and the commit peak;
- the table key randomness (row tails, column salts, answer pads);
- the G_u, Omega, fresh-mask and old-mask randomness;
- the switch master and the base salts/randomness;
- the draw budget.

The verifier reads only the public handoff and the artifact; it does not read
the private directory or the asset. The prove process reads the asset only
through the bounded read-only row source.

## Measurement basis

- Same machine, release build:
  `cargo test -p zkie --release --lib lm_head_stage_processes -- --ignored --nocapture`.
- The parent test spawns the same test binary with
  `--exact lm_head_key::stage_tests::lm_head_stage_worker --ignored`, once per
  stage. Each child reports its own wall time, `VmHWM`, `/proc/self/io`
  counters, temporary bytes and (for commit) the private-state size.
- The commit and prove children each run on a fresh process; the verify child
  receives no prover state.
- Values are one run per row; the run-1 and run-2 logs are included.

## Results

| stage | wall s | VmHWM KiB | temp bytes | notes |
|---|---:|---:|---:|---|
| commit | 32.067 | 461,168 | 1,073,774,656 | store write 1,073,741,824 B, read 2,147,483,648 B; private state 98,160,504 B |
| prove | 8.630 | 630,264 | 5,378,440 | setup 0.104 s, chain 8.340 s; artifact 5,378,440 B |
| verify | 4.254 | 69,784 | 0 | verdict accept; same seal |

- `/proc/self/io` disk counters read 0 for all stages on this run (the files
  were served from the page cache); the commit logical store counters above
  are the actual table write/read volumes.
- 16 sequence tokens per chain: prove 16 / 8.630 s = 1.854 tokens/s; the
  chain-only part is 16 / 8.340 s = 1.918 tokens/s. Commit is not a
  tokens/s stage.
- The artifact is 5,378,440 B (91.7% column symbols, as measured in the
  single-artifact trace). The archived `chain-artifact.bin` is the positive
  run's artifact, sha256
  `d4a2043763aadb770676a5eb5265b8d56dddea1fca1eebfb4e1107896f7f46e2`.

## Negative matrix

Every entry runs the same worker path and must fail closed.

| class | mutation | result |
|---|---|---|
| cross-process repeated consumption | second prove with the spent budget | rejected |
| missing private state | prove-state file removed | rejected |
| replaced private state | one private byte changed | rejected |
| interrupted recovery | stale budget reservation lock | rejected |
| truncated proof | artifact truncated | rejected |
| changed handoff | one public-handoff byte changed | rejected |
| wrong plan identity | handoff plan identifier replaced | rejected |
| artifact truncation/trailing/version | covered in the single-artifact trace | rejected |

The budget journal reserves the full draw count once with an exclusive lock
file, marks the mask material consumed, and never rolls back. A second claim
or a stale lock fails; a failed prove does not return the budget.

## Storage decision

The reviewed decision was to persist encoded codewords in the private prove
state. The measured implementation persists the table commitment state
(reader, name, leaves, salts, peak) plus the key randomness and uses the
bounded read-only asset source in prove. It recomputes no codeword, keeps the
private prove state at 98,160,504 B instead of a 1 GiB codeword store, and
brings the prove setup to 0.104 s; the chain proof remains 8.340 s and the
artifact verifies. Under the "if measurement shows recomputation is faster,
change it" clause this is the measured faster path; the trade-off awaits the
lead's confirmation. If the codeword store is required, a store-backed row
source remains to be built.

## Security and lifecycle

- Private directories are owner-only, private files are owner-readable only,
  files are created fresh and never overwrite, and a symlink at a private
  path is rejected.
- The commit temporary table store is cleaned after commit; no physical
  erasure is claimed. The public artifact and the private state are separate.
- The private state is bound to the public handoff digest; a mismatched or
  replaced state is rejected before any proof work.
- The verify process reads no prover memory, private file or asset.

## Open items

- The three-stage production command does not carry this chain; this harness
  is local.
- Full-model tokens/s, the three model groups and seq512 are outside this
  relation-chain trace.
- The lead's confirmation of the table-state storage path (versus persisted
  encoded codewords) is pending.
- Crash cleanup is best-effort and does not claim physical erasure.

## Logs

- `stage-processes.txt`: final positive run and all negative assertions.
- `stage-processes-r2.txt`: the preceding negative run.
- `stage-processes-r1.txt`: first successful three-process run.
- `first-attempt-io.txt`: first attempt, failed before the private store
  directory existed.
- `workspace-tests-pre.txt`, `workspace-tests.txt`: workspace gates.
- `chain-artifact.bin`: positive-run artifact (binary, gitignored).
