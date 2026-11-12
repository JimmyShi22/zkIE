# Committed product-sumcheck layer extraction

## Scheme and baseline

- Scheme: `committed-psc-layer` (T-003 batch 2b.1 step 1).
- Code: `ef077275`; previous commit `c16cefee`.
- Purpose: move the shared committed group declaration, the opening and
  position codecs, the masked product-sumcheck transcript codec and the
  position-domain challenge source out of the output closure into one module.
  The output closure delegates to the shared layer. No relation, challenge
  order, message order, encoding, plan, public/private boundary, security
  parameter or sampling basis changed; this is a code-placement refactor.

## Validation basis

- The seq16×3 rule is not applicable here: no seq512 logic and no all-case
  structural path changed. The workspace suite and the focused layer tests are
  the acceptance evidence.
- Proof-byte check: the toy output-head fixture proof of
  `proof::output_head_lin::tests::merged_request_accepts_and_round_trips` was
  dumped before and after the refactor with a temporary test-only write, which
  was reverted. Both dumps are 45,080 B and byte-identical; sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`.
  The two dumps stay in this directory; they are test artifacts, not proof
  format changes.

## Results

- `cargo test --workspace`: 48 suites, 815 passed, 0 failed, 17 ignored.
- Focused `committed_psc` tests: 7 passed, 0 failed.
- `cargo check -p zkie-ops`: clean; the same 8 pre-existing warnings as the
  baseline.
- Proof bytes: unchanged (45,080 B, sha256 above, `cmp` exit 0).
- Existing hidden-proof tests all pass in the workspace run.

## What moved

- `CommittedGroupMessage`: the group declaration and derived roots.
- `CommittedPscGroup` / `CommittedPscGroupRef`: group message, SEND sample,
  opening positions and runtime openings, with one encoder and one decoder.
- `put_psc_transcript` / `get_psc_transcript`: the masked product-sumcheck
  transcript batch and its masked point values.
- `PositionPscChallenges` and `PscTranscriptTags`: the position-domain
  challenge source and its absorb order.
- `absorb_point`: one position's masked point absorption.
- `PscPointBinding` / `PscRowBinding`: the checked binding shell for the next
  norm slice.
- The output closure keeps its public types and its exact field order; its
  first-stage groups, second-stage transcripts, challenge source and point
  absorption now call the shared layer.

## Open items

- 2b.1 step 2: norm x/sum/squares/mean/variance committed messages and the
  hidden verifier skeleton on top of this layer.
- 2b.1 step 3: positive and negative tests through the hidden verifier.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- `zkie-proof-before.bin` and `zkie-proof-after.bin` are the two 45 KB
  test-only proof dumps referenced above.

No textual trace was deleted. No proof binary of a previous scheme was
deleted in this round.
