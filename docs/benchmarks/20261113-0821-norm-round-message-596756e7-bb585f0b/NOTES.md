# Norm round committed message

## Scheme and baseline

- Scheme: `norm-round-message`.
- Code: `596756e7`; previous commit `bb585f0b`.
- This is partial progress on the norm range representation. The rounding witness
  group now enters the norm first message: the prover commits it from the plan
  layout of the same integer core, absorbs the group root, draws its SEND sample,
  answers it and opens the drawn positions; the verifier rebuilds the same
  transcript and checks the root, sample, answers and openings. The source and
  reconstruction relations, the lookup forest and the BAT/FOLD binding are not
  part of this step.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Committed round group | one committed value message per position with sample and openings | `NormFirstMessages::round`, `NormWitness::round` | Committed and opened, not yet related to v/q |
| Same core as the plan layout | the committed slots equal the plan layout of the integer core | prover compares every message cell with `NormRoundLayout::values(&core)` | The proof-side relation is next |
| Root and challenge order | root before seeds; seed before answer; answer before positions | tags 84/85/86/87 around the existing norm tags | Theta/alpha and PSC challenges still follow |
| Fail closed | plan and message must agree on the group | `check_trusted` requires both or neither | Version stays 3 |

## Results

- `cargo test --workspace`: 48 suites, 847 passed, 0 failed, 17 ignored.
- Focused `norm`: 23 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Positive and negative coverage

- Every existing norm hidden-path positive test now runs with the round group in
  the plan and the first message; the honest fixture proves and verifies.
- The tamper test rejects a changed round root, a changed round sample answer
  and a changed round opening position, in addition to the existing mask, group
  answer and PSC tampering.
- Plan validation still rejects a missing round message, a mismatched group and
  a failing width set.

## Scope and next

- No source/reconstruction relation for the committed digits, no `u+s=d-1`,
  `r=lo+u` or `v=q*d+r` relation.
- No byte table lookup forest and no BAT/FOLD/CS/Base binding.
- No claim that integer rounding is proven.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
