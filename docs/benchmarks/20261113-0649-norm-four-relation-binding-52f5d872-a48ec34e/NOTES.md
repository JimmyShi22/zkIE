# Norm four-relation row binding

## Scheme and baseline

- Scheme: `norm-four-relation-binding`.
- Code: `52f5d872`; previous commit `a48ec34e`.
- This is partial progress. It extends the active norm relation domain to sum,
  squares, mean and variance and binds their product-sumcheck endpoints to the
  committed messages. The remainder range/digit checks, the reciprocal-square-
  root lookup, the weight/shift/bias/clamp/padding tail and the output-head
  consumer remain outside this slice.

## Changes

- The active relation domain lists all four relations; the map construction of
  each relation is the fixed index construction.
- `norm_plain_rows` binds the mean relation to the committed sum and mean
  messages and the variance relation to the committed squares and variance
  messages. The witness defines `mean_remainder = sum - valid * mean` and
  `variance_remainder = squares - valid * variance`, so the remainder terms
  enter the rows as the same linear forms over the committed messages.
- The mask allocation covers every relation instance per position.
- The norm mask tree row budget and the plan validation cover the extended
  relation domain.

## Results

- `cargo test --workspace`: 48 suites, 829 passed, 0 failed, 17 ignored.
- Focused `norm_`: 14 passed, 0 failed.
- Norm-free byte gate: the toy output-head proof with the norm absent is
  45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.

## Positive coverage

- A hidden toy fixture with the norm present and all four relations active
  runs through `prove_hidden_lin_hidden_with_norm` and the public
  `verify_hidden_lin_hidden_with_witness` entry. The four relations share the
  committed x, sum, squares, mean and variance messages, the committed mask
  entries and the existing BAT/FOLD/BASE path.

## Negative matrix

Each case runs through the public hidden verifier after a serialized round
trip:

- changed endpoint value of every relation instance;
- changed product-sumcheck end value of every relation instance;
- swapped product-sumcheck transcripts (challenge order);
- zero mask root and changed mask sample answer;
- changed sample row target of every committed group;
- changed committed root together with its entry root;
- renumbered committed group oracle;
- changed coordinate domain (`valid`);
- changed active relation domain.

An inconsistent quotient remainder witness is rejected by the prover before
any proof is emitted; it is recorded as a prover-side rejection, not a
verifier-path negative.

## Limits

- The mean and variance maps bind the linear quotient identity only. Until the
  remainder range/digit checks of the next slice exist, a prover-chosen
  quotient and remainder that satisfy the identity are accepted; therefore
  this trace does not claim round-half-up.
- The four-relation row-binding core is complete at the committed-message
  level; the exact remainder range/digit, the reciprocal-square-root lookup,
  the weight/shift/bias/clamp/padding tail and the output-head consumer remain
  open.
- No soundness, privacy, complete-norm or batch-completion claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
