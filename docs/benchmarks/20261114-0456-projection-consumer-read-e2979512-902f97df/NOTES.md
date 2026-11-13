# Projection consumer closure: production base and implementation map

## Scheme and baseline

- Scheme: `projection-consumer` (B2b reading step).
- Code: `e2979512` (B2 inventory); branch `refactor/v2`. Read-only: no code
  change, no prover/verifier run, no key material.

## Corrected production base (TURN 524)

- Production entries: `hidden_send_session::prove_hidden_lin`,
  `prove_hidden_lin_hidden`, `verify_hidden_lin`, `verify_hidden_lin_hidden`
  (and the norm/param variants). `lin_session.rs` is a demo and is not used.
- Claim/plan structures in the production session: `HiddenSendPlan` (table
  shape/identity, matrix rows, k_lo, columns, group, oracle base, W shape,
  `index_mask`, `dot`, expected roots) and `HiddenLinPlan` (send plan, output
  shape, public row weights for the diagnostic path, masks, proximity,
  output relation, closure, norm).
- Member/claim plumbing inside the session: claim slots `claim_range` /
  `claim_slot`; the `index_mask` plan/randomness/witness/message forms and
  `absorb_index_mask_root`; `fold_member_firsts` (Main and Mask parts) at the
  fold construction sites.
- Test plan builders exist for the output-head shape (`plan`, `hidden_plan`,
  `staged_plan` in the session tests). They are the closest templates for a
  Projection-shaped plan.
- Code fact: the file has **zero** `Projection` references. There is no
  general operator-reduction → pending claim → member/BATT glue for a
  Projection consumer; the existing plan builders cover the output-head
  matrix shape only.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Production base, not the demo | `hidden_send_session::prove/verify_hidden_lin*` | confirmed by reading; no demo reuse | this note |
| Claim slots and mask ownership | `claim_range`/`claim_slot`, `index_mask` plan/randomness/witness/message, `absorb_index_mask_root` | mapped | this note |
| Member/BATT reduction | `fold_member_firsts` (Main/Mask), production first/shared/second entries | mapped | this note |
| Projection reduction → pending | none present (`Projection` count 0) | **not implemented** | gap |
| One real consumer closed | — | **not delivered** | gap |

## Gap and next implementation steps

The general glue is the work: build a Projection claim from the trusted weight
table (B1 identity/layout), the consumer's actual activation and output, and
the fixed-point arithmetic; have the verifier rebuild the coefficient, target
and mask ownership from the plan rather than accept a prover-shaped equation;
pack it as a pending claim with member dimensions and the same authenticated
index-mask slot; reduce multiple claims with `member_first` (BATT); run
first/shared/second plus the proximity and output checks. Bias and Add stay on
the existing verified input relation until their reduction is closed; no
consumer is declared migrated before the closure passes.

This trace records the anchor reading and the exact production entry points
only. No operator wiring, no negative matrix, no plaintext removal and no
proof-byte claim is made. No code changed; the 45,080 B gate is untouched.

## Artifacts

`NOTES.md`.
