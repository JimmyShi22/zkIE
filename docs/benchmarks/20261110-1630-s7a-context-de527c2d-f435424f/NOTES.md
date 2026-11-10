# s7a-context - de527c2d / f435424f

Interim trace of S7a (trusted source and lm_head asset identity adapter) inside the S7 batch. The batch is not frozen yet; this directory records the S7a test run only.

## Scope

- Code commit `de527c2d`, previous state `f435424f`, branch `refactor/v2`.
- New callable path: `zkie/src/s7_context.rs` (`load_gpt2_lm_head_context`, `find_lm_head_matmul`, `LmHeadContext::read_lm_head_row`); tests `zkie/tests/s7_context.rs`; `zkie/src/lib.rs` gains one module declaration.
- No shared module or production verification change: `crates/`, `ProductionProvider` and `verify_request` are untouched; the adapter only reads the production policy, public bundle and statement, rebuilds the existing anchor checks, and exposes a read-only mmap row path.
- The weight file is opened read-only and only the requested row is copied; no whole-table materialization, commit or modification.

## Real context read

- case: `models/gpt2/case.toml` (gpt2-seq16; asset root `/data/zkie-assets`).
- policy: `models/gpt2/case.policy` (anchor `dd08df8b…ae36`, model `c2971c01…1268`, sequence 16).
- public bundle: `docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16/commit/public/model-commitment.bin`.
- statement: the same artifact set, `proof/statement.bin`.
- lm_head table: `asset_id=192`, element_count `67,108,864`, row_width `65,536`, rows `1024`, encoding i32-le.

| item | value |
| --- | --- |
| lm_head table identity | `e1c5430264e587b23054e83fb36155e762bc818077ca5e7889dd6a17cfc6f4ba` |
| lm_head start | `5b4f58907ca5995fb726ae4f8ff7d87a7d3ca822ed5f480ec5ff40482f9b70b9` |
| lm_head root / leaf root | `9a8be4edbbc390fd6844710399441788506c2f1f3a1fc9c9c95e19e067971832` |
| anchor id | `dd08df8bcbba41adea6182709e00adb2f897fde4abc241d1ba09c552fd77ae36` |
| model id | `c2971c0179463bf12aecaefc3488417228008c1314e6905c49a55078242c1268` |
| MatMul step | `4444`, config `[16, 65536, 1024, 0, 0]` |
| activation input | port 0, `Prev{step 4443, output 0}` |
| weight input | port 1, `gpt2/weights/lm_head_w_i32.bin` |
| output | port 0 to step 4445 (vocab-mask Add) |

## Tests

- `cargo test -p zkie --test s7_context`: 3 passed, 0 failed, 0 ignored.
- `cargo test --workspace`: 494 passed, 0 failed, 11 ignored.
- Negative samples in the adapter test: wrong policy anchor id, wrong expected model id, tampered statement model commitment, shortened statement tokens, tampered public anchor, tampered public lm_head element count — all rejected by the load path.
- No proof bytes were produced and no performance number is claimed.

## Boundaries

- The adapter is not a proof verifier and creates no commitment; S7b adds the chain.
- The public bundle is produced by the commit side; the adapter rebuilds the anchor from the compiled specifications and policy, but does not add an independent committer one-time test or an external key-file link.
- S7b needs the per-item anchors and parameter decisions (LIN shape mapping, supported sizes, sampling/security parameters, public/private and mask source); this trace does not decide them.

## Artifacts

- `tests/s7-context.txt` (S7a test run with identity values)
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.
