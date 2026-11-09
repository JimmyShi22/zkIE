# proof-header - 5b0d1bf1 / f18c9ad7

Trace of the isolated proof-header module and its fixture-level tests.

## Change

Code commit `5b0d1bf1` ("Add isolated proof-header parsing with exact reads"),
based on `f18c9ad7`. The production verify entry is not changed; the new
encoding is not enabled in production and accepted production proofs still use
the existing path.

- New isolated module `crates/zkie-core/src/proof_header.rs` with
  `ProofHeader { tag: [u8; 4], version_word: u32 }`.
- `read_header` reads exactly eight bytes: four tag bytes followed by one
  little-endian version word. The first tag byte is the low byte when the tag is
  viewed as a little-endian word.
- The version word is a single compile-time constant; a header word is only
  compared with it and can never extend the set of accepted words.
- Outcomes: zero to seven bytes are `invalid_encoding`; a wrong tag is
  `invalid_proof`; a correct tag with an unknown version is
  `unsupported_version`; any other read failure is `io`. The function returns on
  the first failure.
- The header parser consumes only the eight header bytes and does not check
  trailing bytes.

## Fixture marker

The complete-consumption example is fixture-only: a test helper reads the
header, a little-endian body length and that many body bytes, then requires end
of input. Comments mark it as fixture code, not production. The header parser
alone is not claimed to check trailing bytes or the whole proof; the fixture
parser only demonstrates the end-of-input gate for synthetic bytes.

The headerless old-format test uses a bincode-serialized synthetic old sample.
That concrete sample starts with a length word and is rejected with
`invalid_proof`. This does not assert that every old byte string starts with a
wrong tag: old data may accidentally begin with a valid header. Production
acceptance must fully decode and authenticate the proof and must not fall back
to the old implementation after a parse failure.

## Negative results

| input | result |
| --- | --- |
| legal eight-byte header | accepted; cursor position 8 |
| legal header plus one trailing byte, header parser only | accepted; cursor position 8 (no trailing check) |
| 0, 1, ..., 7 header bytes | `invalid_encoding` |
| wrong tag with correct version word | `invalid_proof` |
| correct tag with unknown version word | `unsupported_version` |
| reader fails before the first byte | `io` |
| reader fails after two bytes of the first word | `io` |
| reader fails after the tag | `io` |
| fixture: header + body + end of input | accepted; every byte consumed |
| fixture: header + body + trailing byte | `invalid_proof` (fixture parser only) |
| fixture: header + truncated body | `invalid_encoding` |
| bincode-serialized old headerless sample | `invalid_proof` (concrete sample; see caveat above) |

## Injections

Each injection removes exactly one check, runs the named test, and restores the
check.

| injection | observed | log |
| --- | --- | --- |
| version comparison removed | unknown version accepted; `unknown_version_is_unsupported` failed on an `Ok` value | `tests/injection-version-fail.txt` |
| tag comparison removed | wrong tag accepted; `wrong_tag_is_rejected_before_version` failed on an `Ok` value | `tests/injection-tag-fail.txt` |

Restore runs passed: `tests/injection-version-restore.txt`,
`tests/injection-tag-restore.txt`.

## Boundary

A successful header read only shows that the tag and the compile-time version
word match. It makes no soundness, hiding or single-shot claim. Trusted
transcript binding (protocol identity, configuration, plan, statement, output
and table identity) remains for the later production integration and must be
bound before the first challenge. This trace does not claim that any proof
passes.

## Validation scope

The production verify entry and the existing proof format are unchanged, so no
five-case three-stage run was made. Validation is the isolated module tests, the
old-sample integration test, the two injections above, and the workspace test
gate.

## Workspace gate

`cargo test --workspace` on `5b0d1bf1`: 369 passed, 0 failed, 2 ignored (the two
multi-GiB byte-consumption tests from the earlier trace, unchanged). Logs:
`logs/cargo-test.txt`; isolated logs in `tests/proof-header-unit.txt` and
`tests/proof-header-old-sample.txt`.

## Artifacts

- `tests/proof-header-unit.txt`: the eleven module tests.
- `tests/proof-header-old-sample.txt`: the headerless sample test.
- `tests/injection-*.txt`: four injection and restore logs.
- `logs/cargo-test.txt`: full workspace log.
- No binary artifacts were produced or deleted.
