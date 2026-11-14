# The request-level W as its own oracle

## Scheme

- Scheme: `request-w-commit`.
- Current code: `a07c7c69`, following `a68fb4df`.
- Previous code: `a68fb4df`.

## Purpose

The request's W is a new column-domain vector (one entry per padded column),
committed by its own SEND oracle rather than relabelled from the per-position
rows. The session gains that commit entry: one logical vector through the
generic value-message committer, under its own oracle range, group and tree,
with the caller's layout, which may split one logical vector into several
encoded rows. The per-position rows of the send path remain a different object.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_request_w_commits_as_its_own_oracle`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused W-commit test | 1 passed |
| workspace | 51 suites, 1072 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| W is committed by its own SEND oracle, not relabelled from the per-position rows | object rule | `commit_request_w` through the generic value committer | focused test sees its own root |
| One logical vector follows the caller's layout and may take several encoded rows | layout rule | the value-message committer's layout | the entry takes the caller's shape and W |
| The per-position path stays untouched | no-relabel rule | separate entry and oracle range | focused test checks the per-position plan |
| Another vector has another root | binding rule | second commit | focused test |

## Conclusions

- The request-level W has its own commitment entry and oracle identity.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The first/shared/second chain wiring with the actual IOPP tests, then the
  frozen layout (W oracle count, encoded rows, slots, opening capacity).
