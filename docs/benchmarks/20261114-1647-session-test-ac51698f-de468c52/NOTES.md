# The session's chain section against its plan

## Scheme

- Scheme: `session-test`.
- Current code: `ac51698f`, following `de468c52`.
- Previous code: `de468c52`.

## Purpose

The session's chain entry is exercised on the session's own plan and message
fixture. A plan without the chain refuses the section; a plan with the chain
refuses a message without it; and a message whose proofs do not match the
plan's counts is refused. The dummy inputs never run on those paths, so the
test pins the presence and shape rules exactly at the session boundary.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_request_chain_section_requires`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused session test | 1 passed |
| workspace | 51 suites, 1063 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| A plan without the chain refuses the section | presence rule | `verify_request_chain_section` | focused session test |
| A plan with the chain refuses a message without it | presence rule | presence check | focused session test |
| Proofs of another shape are refused | shape rule | `check_shape` in the entry | focused session test |
| The old paths stay byte-identical | stability rule | plan-gated section | hard gate unchanged |

## Conclusions

- The presence and shape rules hold at the session boundary on the real
  fixture.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The positive session path waits for the source layer, which supplies the
  reads the chain ties; then the member chain and the batch report follow.
