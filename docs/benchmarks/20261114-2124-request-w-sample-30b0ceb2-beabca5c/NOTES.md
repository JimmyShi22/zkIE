# The request W's sample in its message

## Scheme

- Scheme: `request-w-sample`.
- Current code: `30b0ceb2`, following `beabca5c`.
- Previous code: `beabca5c`.

## Purpose

The request W's first-part message now carries its SEND sample: after the root
is absorbed, the prover draws the seeds on the chain and answers them from the
committed value message; the verifier draws the same seeds and refuses a message
whose sample differs. The sample travels only when the plan carries the request
W, so existing messages are unchanged.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_request_w_root_travels_in_the_message`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused message test | 1 passed |
| workspace | 51 suites, 1078 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The W's sample is drawn after its root's absorb | order rule | `draw_seed_set` after `absorb_request_w_root` | focused round-trip carries it |
| The sample's answers come from the committed value message | source rule | `oracle_sample_row` over the message | prover side; the own test checks it later |
| The verifier draws the same seeds and refuses a mismatch | check rule | verify side draws and compares | the message decode/verify path |
| Existing messages are unchanged | stability rule | plan-gated field | hard gate unchanged |

## Conclusions

- The request W's message carries its sample alongside its root.
- No existing path changes; the toy proof is byte-identical.

## Open items

- The own group's member and sample rows, then the request rows and the tests.
