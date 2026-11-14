# At-start mask rules and witness read plan

## Scheme

- Scheme: `at-start-reads`.
- Current code: `cd3db22e`, following `2937404b`.
- Previous code: `2937404b`.

## Purpose

Apply the at-start rules to the consumer's mask message and add the read plan
of the reduced witness. When the chain carries the member relation, the
index-mask message is the at-start message: its mask code is the own group's
mask code, its leaf class is strictly above every other message and stage
tree, and its free entries cover the pending claims' slots besides the Eval
allocation, with every pending slot in the free range. The reduced-witness
reads gain a plan object that cuts one table into segments and checks coverage
against the witness plan.

## Measurement basis

- Same machine, release build for the real run; debug for the workspace.
- Focused: `cargo test -p zkie-ops --lib hidden_member` and the hidden
  consumer test over the registered seq16 artifacts.
- Hard gate: the norm-free toy output-head proof, dumped and removed.
- Workspace: `cargo test --workspace`.

## Key numbers

| figure | value |
|---|---:|
| hidden consumer run | 79.23 s, 1 passed (with the at-start negatives) |
| session package total | 2,191,269,699 B |
| session.bin bytes | 1,071,688 |
| session.bin sha256 | `f4241fd425ed34ef5753f58cc5e32be531df62651c7641d5fd1edda1e4950ed7` |
| focused hidden-member tests | 10 passed |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |
| workspace | 51 suites, 1024 passed, 0 failed, 19 ignored |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The at-start mask message shares no leaf with any other message or stage tree | request at-start shape rule (leaf class above every link message) | the plan check compares the index-mask leaf class with every send, proximity, closure and stage tree class and requires it strictly above | hidden run; a shared class 7 is rejected by the plan check |
| The mask message is the own group's mask message | member rows-fit rule (mask code equals the own group's) | the plan check requires the index mask code equal to the proximity mask code | a changed mask code is rejected |
| The message's entries cover the pending slots besides the Eval allocation | mask-ledger capacity rule | total allocation plus pending slots must fit the free entries; every pending slot lies in the free range | pending slots below the allocation and beyond the free range are rejected |
| The reduced-witness reads form a plan object with exact coverage | read coverage rule | `ReductionReadSet` cuts one table into segments and checks coverage against the witness plan | focused test: gap, outside read, zero weight and a bad segment are refused |

## Conclusions

- The at-start message's class, mask code and capacity rules are enforced in
  the plan and exercised on the real consumer run with their negatives.
- The read set is a plan object; the runtime slot ledger and the claim
  structure remain open.

## Open items

- The mask slots are fixed in the plan and checked against the allocation;
  a runtime ledger take is not yet wired.
- The own-group membership of the mask message is by the existing member
  construction; an explicit plan check of the LIN first/second coverage waits
  for the claim-structure ruling.
- The pending claim's consumption, the merge transcript and the table-read
  binding remain under the pending expert ruling.
