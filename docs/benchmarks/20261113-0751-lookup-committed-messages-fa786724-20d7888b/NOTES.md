# Committed lookup messages

## Scheme and baseline

- Scheme: `lookup-committed-messages`.
- Code: `fa786724`; previous commit `20d7888b`.
- This is partial progress on the lookup module. It adds the committed key,
  multiplicity-bit and mask messages, the message shapes and identifiers, the
  public roots, the oracle-tagged root absorption before the first lookup
  challenge, and the migration of the mask slots to the committed mask entries.
  The roots are not yet part of a hidden public message or proof artifact; the
  openings, the ledger entries and the binding rows are the next steps.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Key message commitment | product-form encoding and a runtime tree over one message, as used by the hidden value messages | `LookupCommitPlan::key_shape`, `commit_lookup` through `commit_value_group`, root in `LookupRoots::key` | The committed vector is the key witness; no opening yet |
| Multiplicity-bit message commitment | the same value-message form over the plane-major bit vector | `bits_shape` and `bits_oracle`, root in `LookupRoots::bits` | No opening yet |
| Mask message commitment | mask-only runtime tree with one pad slot, as used by the mask-only closure message | `mask_rows`, tails, salts and mask tree root; `LookupMessages::mask_entry` | Entries feed the prover masks; the verifier holds only the root |
| Message identity and order | absorb the schema version, the lookup parameters, each message oracle and root before alpha; draw alpha afterwards | `LookupChallenges::absorb` with the key, bit and mask oracle tags on both sides | Hidden chain integration and the public encoding are next |
| Plan validation | distinct oracles and tree numbers, shape capacity, mask capacity | `LookupCommitPlan::validate` | The plan is a trusted input on both sides |

## Results

- `cargo test --workspace`: 48 suites, 842 passed, 0 failed, 17 ignored.
- Focused `lookup`: 4 passed, 0 failed; focused `psc_general`: 8 passed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: this is a module-level block with no end-to-end path change;
  the norm-free byte gate is the required byte gate for the step. No seq512 case
  was run because the block does not touch the seq512 path.

## Positive and negative coverage

- An honest four-key lookup with committed messages proves and verifies; the
  prover and verifier outcomes (key claim, multiplicity row and rows) agree.
- The message roots and oracles enter the challenge source before alpha.
  Changing the committed key root or the mask root changes the verifier's key
  claim, so the messages feed the transcript.
- Rejected structurally: a truncated tie round, a cleared table round, cleared
  bit end values and a missing key level; wrong version, trailing bytes and
  truncation are rejected by the decoder; a key outside the table cannot be
  proved.
- These are structural and transcript negatives. A changed root is not yet
  rejected as a false statement, because the opening and binding checks come
  with the next step; no soundness claim is made here.

## Scope and next

- No openings, ledger entries or mask opening checks.
- No BAT/FOLD/CS/Base binding; no norm digit or remainder rows; the committed
  roots are not yet carried by a hidden public message or proof artifact.
- No soundness, lookup-completeness or rounding claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
