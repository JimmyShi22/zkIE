# Norm lookup message carrier

## Scheme and baseline

- Scheme: `norm-lookup-message-carrier`.
- Code: `0ce9ab8e`; previous commit `c4bf7d76`.
- Purpose: carry the lookup commitments, SEND sample answers and openings in the
  norm first message, and the lookup transcript with its claims in the norm
  second message, so the real hidden session commits and verifies a byte lookup
  over the committed rounding digits.
- This is partial progress. The lookup claims are not registered in the
  second-stage BAT/FOLD yet, and the lookup plan's key sources are still
  fixture values read from the committed rounding rows.

## Validation basis

- The seq16 x3 rule is not applicable here: no production plan carries the
  lookup yet, no seq512 logic and no all-case structural path changes. The
  workspace suite, the focused lookup/norm tests and the byte-identical
  norm-free proof are the acceptance evidence.

## Changes

- `NormFirstMessages.lookup` carries the multiplicity-bit group (one committed
  message, one SEND answer, one opening) and the lookup mask message (tree,
  root, answer, positions, opening). The wire layout appends the field only
  when the plan carries a lookup, so the lookup-free bytes are unchanged.
- `NormPscSecond.lookup` carries the lookup proof and its claims (key claim,
  multiplicity row, general-sumcheck outputs). Encode/decode are plan-driven;
  the standalone skeleton path stays product-sumcheck only.
- The first stage absorbs the lookup roots and samples after the norm source
  commitments and the norm mask. The second stage derives the private lookup
  commitment seed from the run seed, draws the lookup challenge seed from the
  second chain, and runs the lookup product-sumcheck transcript.
- The lookup commit plan gains `bits_openings` and `mask_openings`; the plan
  digest binds them.
- Equality weights use the sumcheck index order, where the first point
  coordinate is the least significant map bit. The previous order only agreed
  when the two swapped coordinates carried equal values; duplicate
  multiplicities violated that and the table-side product check failed.

## Four-column mapping

| Requirement | Mechanism | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Lookup commitments before challenges | First message carries the multiplicity-bit and mask roots | `NormLookupMessage`; `prove_norm_first` commits through `commit_lookup` | Absorbed before the second-stage seed; roots are plan/transcript bound |
| SEND sample answers | One seed set and one answer per message | Bit sample uses the value-group form; mask sample uses the mask-only form | Answers absorbed under distinct tags; openings drawn after the answers |
| Openings | One opening per drawn position | Bit tree opening from the retained commit tree; mask opening from the mask tree | Verified against the message roots in `verify_norm_first` |
| Key source | Keys read from committed rounding rows | `LookupKeyPlan::read` over the round witness rows in `prove_norm_second` | Prover side only; verifier expansion and registration are the next step |
| Challenge order | Source commitments and samples first, then the lookup challenge and product-sumcheck | `draw_lookup_seed` after the first-message bound chain; `prove_lookup_with_messages` | Seed drawn from the second chain, which starts from the first-stage query |
| Claims carrier | Second message carries proof and claims | `NormLookupSecond { proof, claims }` | Verifier recomputes and compares; not yet registered on BAT/FOLD |
| Duplicate multiplicity | Binary counts over planes | Equality weights in sumcheck index order | Duplicate-key unit test; this failed before the order fix |
| Lookup-free layout | No bytes when the plan has no lookup | Conditional encode/decode driven by the plan | Hard-gate proof unchanged |

## Results

- `cargo test --workspace`: 48 suites, 856 passed, 0 failed, 17 ignored.
- Focused `proof::lookup`: 8 passed; `proof::hidden_norm_psc`: 16 passed;
  `norm_lookup_carrier_proves_and_verifies`: 1 passed.
- Hard gate: the norm-free toy output-head proof is 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  byte-identical to the baseline dump.
- New integrated positive test: a norm plan with a lookup over 2 positions and
  2 digit slots, 4 keys with one duplicated value, proves and verifies through
  the real hidden verifier; a changed claim and a changed transcript are both
  rejected.

## Explicitly deferred

- Registering the expanded key claim and the multiplicity/product-sumcheck
  claims in the hidden second BAT/FOLD, per physical member.
- Replacing the fixture key sources with the production round-row key map.
- Range and no-wrap relations and the rounding claim itself; no rounding claim
  is made here.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
