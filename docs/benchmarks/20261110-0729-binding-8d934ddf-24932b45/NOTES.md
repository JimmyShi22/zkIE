# binding - 8d934ddf / 24932b45

Trace of the isolated candidate binding assembly and its tests.

## Change

Code commit `8d934ddf` ("Add candidate binding assembly over reused transcript
primitives"), based on `24932b45`. The production verify entry is unchanged; the
candidate is not called by production and no new protocol acceptance is
returned.

- New isolated module `crates/zkie-engine/src/pcs/request/binding.rs`.
- `BindingParameters` carries a version word, protocol identity, domain
  parameters and a minimal plan identifier. The domain parameters are the
  existing request-parameter digest.
- `BindingStatement` carries the case, request and table identity, plus the
  version word and protocol identity for a consistency check against the
  parameters.
- `assemble` encodes the parameters with the existing canonical writer and
  parameter domain, encodes the statement and the optional record, and starts a
  request chain with the existing chain-start primitive. The returned first
  query is the chain value produced by that primitive.
- `plan_identifier` is a generic digest over an ordered list of plan-item
  digests. The complete plan-item and trusted-source list is kept outside this
  repository; only the generic ordered list is encoded here.
- Missing required values and statements that contradict the parameters are
  rejected before any digest is produced.
- The field order and the encoding are candidate-internal and replaceable. They
  are not an external format, and the test examples are not format examples.

## Reuse note

The candidate reuses the existing canonical writer, parameter domain and
chain-start primitive. Being reused does not make the candidate a production
binding: production does not call this module, and no digest produced here has
derived a real challenge.

## Test coverage

| behavior | test |
| --- | --- |
| deterministic assembly and reuse of chain start | `assembly_is_deterministic_and_reuses_chain_start` |
| parameter version, protocol identity, domain parameters, plan identifier | `parameter_*_changes_the_digest` |
| statement case, request and table identity | `statement_case_changes_the_first_query`, `statement_request_and_table_identities_change_the_first_query` |
| statement field order | `statement_field_order_is_bound` |
| optional record presence, record bytes and length | `record_presence_bytes_and_length_are_bound` |
| case length and field framing | `case_id_length_and_field_framing_are_bound` |
| word, byte-string and digest width separation | `canonical_writer_separates_word_and_byte_string_types` |
| domain separation with the same encoded bytes | `domain_separation_isolates_the_domain_label` |
| plan identifier order, length and empty or zero parts | `plan_identifier_binds_order_length_and_presence` |
| missing parameter values | `missing_parameters_are_rejected` |
| missing statement values | `missing_statement_values_are_rejected` |
| contradictory statement version or protocol identity | `contradictory_statement_is_rejected` |

The binding unit log reports 16 passed tests.

## Injections

Each injection removes exactly one field absorption, runs its sensitivity test,
and restores the module. Every injected run failed and every restored run
passed.

| removed absorption | failing test | log |
| --- | --- | --- |
| parameter version word | `parameter_version_changes_the_digest` | `tests/inject-parameter-version-fail.txt` |
| parameter protocol identity | `parameter_protocol_identity_changes_the_digest` | `tests/inject-parameter-protocol-fail.txt` |
| parameter domain parameters | `parameter_domain_parameters_change_the_digest` | `tests/inject-parameter-domain-fail.txt` |
| parameter plan identifier | `parameter_plan_identifier_changes_the_digest` | `tests/inject-parameter-plan-fail.txt` |
| statement case id | `statement_case_changes_the_first_query` | `tests/inject-statement-case-fail.txt` |
| statement request identity | `statement_request_and_table_identities_change_the_first_query` | `tests/inject-statement-request-fail.txt` |
| statement table identity | `statement_request_and_table_identities_change_the_first_query` | `tests/inject-statement-table-fail.txt` |
| record bytes and length | `record_presence_bytes_and_length_are_bound` | `tests/inject-record-bytes-fail.txt` |

Restore logs use the same names with `-restore.txt`.

## Boundary

A candidate digest only describes the local behavior of the chosen encoding and
inputs. It does not show that a real challenge used it, that any preceding chain
value is authenticated, or that the full protocol is sound or hiding. Production
is not switched to this candidate and must not return a new protocol acceptance
before a later production closure makes its challenges derive from the prefix
that is actually absorbed. This trace does not claim that any proof passes.

## Validation scope

The production verify entry and the existing proof format are unchanged, so no
five-case three-stage run was made. Validation is the binding unit tests, the
eight single-field injections, and the workspace test gate.

## Workspace gate

`cargo test --workspace` on `8d934ddf`: 385 passed, 0 failed, 2 ignored (the two
multi-GiB byte-consumption tests from the earlier trace, unchanged). Logs:
`logs/cargo-test.txt` and `tests/binding-unit.txt`.

## Artifacts

- `tests/binding-unit.txt`: the sixteen binding tests.
- `tests/inject-*.txt`: eight fail logs and eight restore logs.
- `logs/cargo-test.txt`: full workspace log.
- No binary artifacts were produced or deleted.
