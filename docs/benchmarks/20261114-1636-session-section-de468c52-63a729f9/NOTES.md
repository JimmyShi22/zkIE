# The session's request chain section entry

## Scheme

- Scheme: `session-section`.
- Current code: `de468c52`, following `63a729f9`.
- Previous code: `63a729f9`.

## Purpose

The session's plan and message already carry the request chain's shapes and
proofs; this block adds the entries that use them. Proving returns the section
the message carries, and verifying reads the section only when the plan carries
the shapes and the message the proofs, checks the proofs against the plan's
counts and then runs the chain's arithmetic entry. The shape check is one
implementation, shared with the encoder.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_chain`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused chain tests | 3 passed |
| workspace | 51 suites, 1062 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The session's prove entry returns the section the message carries | entry rule | `prove_request_chain_section` | compiles against the existing plan and message types |
| The verify entry runs only when the plan and the message both carry the chain | presence rule | `verify_request_chain_section` | presence checks in the entry |
| The proofs must match the plan's counts | shape rule | `RequestChainMessages::check_shape` | focused test; a wrong shape is refused |
| Old paths stay byte-identical | stability rule | plan-gated section | hard gate unchanged |

## Conclusions

- The session entry pair exists and the shape check is one implementation.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The session-level fixture with a chain-carrying plan, then the source layer
  and the member chain follow.
