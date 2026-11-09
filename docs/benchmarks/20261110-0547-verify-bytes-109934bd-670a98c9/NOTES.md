# verify-bytes - 109934bd / 670a98c9

Trace of the exact-byte-consumption hardening of the production verify entry,
with a full three-stage rerun on all five validation cases.

## Round 2 corrections

Expert review returned three items; this section supersedes the matching text
below.

- F1: the proof truncation now saves the removed tail byte and writes it back at
  the same offset, and the proof mutation block records a SHA-256 before and
  after and asserts equality. A separate non-ignored test covers a nonzero tail
  byte. Both multi-GiB integration tests were rerun and the workspace is green.
- F2: the baseline prove values for `gemma3-seq16`, `gemma3-seq512` and
  `timesfm-16patch` were corrected to the `b4af4698` result files (45,222,
  419,974, 35,380). The earlier 45,076 / 417,040 / 35,308 came from a different
  baseline and are removed. Every baseline cell now names the `b4af4698` round
  and the per-case `result.json` files; both sides are single measurements.
- F3: "proof bytes identical" is corrected to "proof byte counts identical".
  The previous proof files were deleted, so no proof content-digest comparison
  is claimed. Authenticated output digest equality is retained because those
  digests are recorded in the previous result files.

## Change

Code commit `109934bd` ("Require exact byte consumption in production
verification"), based on `670a98c9`, branch `refactor/v2`.

- `proof.bin` is deserialized from a `BufReader` over the file, then one more
  byte is read from the same reader. `Ok(0)` is the normal end of input,
  `Ok(_)` is a trailing-byte rejection, and `Err(_)` maps to the I/O error.
  The proof stays streamed; it is not buffered whole.
- `statement.bin` and `model-commitment.bin` are deserialized through a
  `Cursor` and accepted only when the cursor position equals the file length.
  A trailing statement maps to `invalid_proof`; a trailing commitment maps to
  `artifact_integrity_mismatch`. Truncation and other deserialization failures
  remain `invalid_encoding`.
- The public-input decoder is unchanged: it already checks the element count,
  performs checked length arithmetic, requires the exact byte length, and keeps
  the u32 Argmax and i32 IntOutput width semantics. Only verify-entry negatives
  and a width unit test were added.
- The policy path and the serialized formats are unchanged. No public field,
  error variant, or configuration field was added.

## Validation set and hit basis

Full five-case set: `gpt2-seq16`, `gpt2-seq512`, `gemma3-seq16`,
`gemma3-seq512`, `timesfm-16patch`.

Hit basis: the change sits in the shared production verify entry that reads the
commitment, statement, proof and public inputs for every model and every proof
size. It therefore touches both sequence shapes (`seq16` and `seq512`), both
output kinds (Argmax and IntOutput), and the four artifact reads for all cases.
The full five-case set is the required validation set for this change.

## Three-stage validation

All five cases ran commit, prove and verify once on the HEAD binary
(`sha256 58491fc2d11b08fb3bc232b4d5effeeaf84526675f35db188bb915a461303823`).
Every stage returned `PASS`; every verify ran with output binding `expected`.
The baseline for every row is the single run stored in
`docs/benchmarks/20261109-0908-oracle-ports-b4af4698-eaae60b0/prod/<case>/prove/result.json`
and the matching `verify/result.json` under the same case (commit `b4af4698`).
No warm median and no other round is used. Both this run and the baseline are
single measurements; this trace makes no claim about three warm-run variance.

| case | commit ms | prove ms | baseline prove ms | verify ms | baseline verify ms | proof bytes | baseline proof bytes | prove VmHWM KiB | baseline prove VmHWM KiB | verify VmHWM KiB | baseline verify VmHWM KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 3,449 | 21,835 | 21,710 | 25,145 | 25,202 | 4,161,786,553 | 4,161,786,553 | 8,316,292 | 8,317,232 | 5,391,180 | 5,405,436 |
| gpt2-seq512 | 3,436 | 167,596 | 166,840 | 234,055 | 232,603 | 38,992,550,944 | 38,992,550,944 | 78,571,136 | 78,570,220 | 47,036,836 | 47,018,280 |
| gemma3-seq16 | 9,828 | 45,430 | 45,222 | 45,336 | 45,736 | 8,766,306,050 | 8,766,306,050 | 17,523,544 | 17,523,512 | 13,351,300 | 13,350,856 |
| gemma3-seq512 | 9,713 | 419,204 | 419,974 | 821,930 | 831,845 | 101,812,159,615 | 101,812,159,615 | 207,848,880 | 207,849,064 | 120,673,912 | 120,691,540 |
| timesfm-16patch | 5,548 | 35,506 | 35,380 | 42,979 | 43,423 | 6,865,533,616 | 6,865,533,616 | 13,822,988 | 13,824,016 | 7,322,900 | 7,322,308 |

Proof byte counts are identical to the baseline for every case; the previous
proof files were deleted, so no proof content-digest comparison is claimed.
Verify wall time moves between -1.192% and +0.624%; prove wall time between
-0.183% and +0.576%. Prove VmHWM moves between -0.011% and +0.001%; verify
VmHWM between -0.264% and +0.040%. Both this run and the baseline are single
measurements; single samples are not sufficient to establish measurement noise.
Total three-stage wall time was 1,891 s across the five cases (`full3.txt`).

Authenticated output digests (all equal to the baseline):

| case | digest | elements |
| --- | --- | ---: |
| gpt2-seq16 | `135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280` | 16 |
| gpt2-seq512 | `eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd` | 512 |
| gemma3-seq16 | `8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562` | 16 |
| gemma3-seq512 | `6d1df6b50c20d02db715fd14b12b9134623739375c44a029dd15b6b30bd9344c` | 512 |
| timesfm-16patch | `cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3` | 32,768 |

## Byte-consumption tests

Two integration tests in `zkie/tests/verify_bytes.rs` use one legal
`gpt2-seq16` artifact set from this trace. The proof is multiple GiB and cannot
be committed, so the tests are `#[ignore]` by default and read the artifact
directory from `ZKIE_VERIFY_FIXTURE` (a directory with a `gpt2-seq16`
subdirectory holding `public/model-commitment.bin`, `statement.bin`,
`proof.bin`, `public-inputs.bin`, `policy.toml`, `model-id.txt`). Each test
copies the files to a private directory, mutates the copy, runs the full
production verify entry, and restores the file before the next mutation; the
source fixture is never modified. The legal artifact is accepted at the start
and at the end of each test. For the proof file the truncation saves the removed
tail byte and writes it back at the same offset; the mutation block records a
SHA-256 before and after and asserts equality. A separate non-ignored test,
`saved_len_restores_a_nonzero_tail_byte`, truncates a four-byte file ending in
`0xab` and checks the restored digest and content, so the nonzero-tail case does
not depend on the proof fixture's last byte.

Commands (release test build, one test thread):

```
ZKIE_VERIFY_FIXTURE=<fixture> cargo test --release -p zkie --test verify_bytes \
  -- --ignored --test-threads=1 production_verify_rejects_trailing_and_truncated_artifacts
ZKIE_VERIFY_FIXTURE=<fixture> cargo test --release -p zkie --test verify_bytes \
  -- --ignored --test-threads=1 production_verify_rejects_malformed_public_inputs
```

Both passed in round 2: `tests/r2-verify-bytes-artifacts.txt`,
`tests/r2-verify-bytes-public-inputs.txt`, and the final combined run
`tests/r2-verify-bytes-both.txt`. Round 1 logs remain in the directory; the
round 2 logs supersede them.

Observed categories through the production verify entry:

| file | tamper | before `670a98c9` | after `109934bd` |
| --- | --- | --- | --- |
| proof.bin | append one byte | accepted | `invalid_proof` |
| proof.bin | truncate one byte | `invalid_encoding` | `invalid_encoding` |
| statement.bin | append one byte | accepted | `invalid_proof` |
| statement.bin | truncate one byte | `invalid_encoding` | `invalid_encoding` |
| model-commitment.bin | append one byte | accepted | `artifact_integrity_mismatch` |
| model-commitment.bin | truncate one byte | `invalid_encoding` | `invalid_encoding` |
| public-inputs.bin | append one byte | `invalid_encoding` | `invalid_encoding` |
| public-inputs.bin | append four bytes | `invalid_encoding` | `invalid_encoding` |
| public-inputs.bin | truncate one byte | `invalid_encoding` | `invalid_encoding` |
| public-inputs.bin | element count + 1 | `shape_mismatch` | `shape_mismatch` |
| public-inputs.bin | flip one value byte | `statement_mismatch` | `statement_mismatch` |
| proof path | replace file with a directory | `invalid_encoding` | `invalid_encoding` |

The proof-directory case is rejected, but its category does not change: the
directory opens successfully on this platform and the read error happens inside
deserialization. The post-deserialization read-error branch (`Ok(0)`, `Ok(_)`,
`Err(_)`) is covered by the unit test with an injected reader.

The "before" column was measured directly on the previous commit
(`670a98c9`, binary sha256
`33af421c39cabb8cb9c5b7a39769fb859ab19c8aaa7714488a7e6a1d8dbf454c`): trailing
statement, commitment and proof all returned `PASS`, truncations returned
`invalid_encoding`, and the restored legal set returned `PASS`. Log:
`tests/previous-behavior.txt`.

The Argmax u32 and IntOutput i32 width semantics are covered by
`output::tests::argmax_and_int_output_keep_their_widths`: the same four bytes
`0x8000_0000` decode to `2147483648` for Argmax and `-2147483648` for
IntOutput. The timesfm case exercises the IntOutput path end to end in the
three-stage table above.

## Injections

The two byte gates and the proof tail write-back were removed in turn, the
named tests were run, and each change was restored; every restore run passed
again.

| injected omission | failing tests | log |
| --- | --- | --- |
| proof end-of-input check removed | `provider::tests::proof_bincode_requires_eof` failed with an `Ok` value; `production_verify_rejects_trailing_and_truncated_artifacts` failed with `expected rejection invalid_proof, got accepted=true` | `tests/injection-eof-unit-fail.txt`, `tests/injection-eof-integration-fail.txt` |
| cursor-position check removed from exact deserialization | `provider::tests::exact_bincode_rejects_trailing_and_truncation` failed; the integration test accepted the trailing statement | `tests/injection-exact-unit-and-integration-fail.txt` |
| proof tail write-back removed | `saved_len_restores_a_nonzero_tail_byte` failed with a SHA-256 mismatch | `tests/injection-restore-tail-fail.txt` |

Restore logs: `tests/injection-eof-restore.txt`,
`tests/injection-exact-restore.txt`, `tests/injection-restore-tail-restore.txt`.

## Workspace gate

`cargo test --workspace` on the round 2 code state: 357 passed, 0 failed, 2
ignored (the two multi-GiB artifact tests above). Log:
`logs/cargo-test-r2.txt`. The round 1 log `logs/cargo-test.txt` is retained.

## Artifacts

- `full3.txt`: copy of the harness log with the binary hash, per-case wall time
  and every stage result.
- `prod/<case>/`: commit, proof and verify run directories with `result.json`
  and `progress.jsonl`; commit and proof manifests; policy and parameters.
- `tests/`: both integration test logs and the injection logs. Round 1 logs are
  retained; the round 2 `r2-*` logs and `injection-restore-tail-*` logs are the
  current evidence.
- `logs/cargo-test.txt`: round 1 workspace test log.
- `logs/cargo-test-r2.txt`: round 2 workspace test log.
- The five `proof.bin` files (150 GiB total) remain on disk under
  `prod/<case>/proof/` and are gitignored. No binary artifact was deleted in
  this round and no previous binary cleanup is claimed here.
- The per-case `run.log` files are empty stdout/stderr captures and remain on
  disk; no textual result was deleted.

## Open items

- The two integration tests need an external multi-GiB fixture
  (`ZKIE_VERIFY_FIXTURE`) and are ignored in the default workspace run; their
  executed logs are committed here.
- A proof path that is a directory is rejected during deserialization with
  `invalid_encoding`; the post-deserialization I/O branch is unit-tested only.
- The multi-GiB proof fixture's last byte is zero; the nonzero-tail restore path
  is covered by the synthetic `saved_len_restores_a_nonzero_tail_byte` test, not
  by the proof fixture's byte value.
- Public-input verify-entry negatives use the Argmax case; IntOutput width
  semantics are unit-tested and the Int32 path is validated end to end by the
  timesfm case.
