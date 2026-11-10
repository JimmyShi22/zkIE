# Output-head LIN request in the stage manifest

## Scheme

- Scheme: `output-head-lin-manifest`.
- Previous code: `a923a5b9`.

## Purpose

Bind the real output-head LIN request into the versioned verification
manifest used by the stage entry. The manifest lists the trace proof, the
request proof and the public commitment with whole-file digests; the verifier
checks every file before accepting.

## Measurement basis

- Release build, production `gpt2-seq16` fixture.
- `ZKIE_SESSION_FIXTURE=/data/lin-fixture cargo test -p zkie --release --test
  lin_session selected_gpt2_output_head_request_session -- --ignored
  --nocapture`
- The fixture uses a hard link to the production trace proof and real copies
  of the public commitment and policy. Symlinks are deliberately rejected by
  the file reader, so the fixture holds no symlink at a referenced path.

## Results

- Manifest total: 5,416,471,995 B =
  4,161,782,574 (trace) + 1,254,646,928 (request) + 42,493 (public).
- Request proof: 1,254,646,928 B, reported separately from the full total.
- Prove 598.208 s; verify 399.475 s; whole test 1465.90 s.
- The manifest write/read/verify path accepts the complete set.
- A request byte flip is rejected both by the manifest and by the request
  verifier; a missing trace is rejected by the manifest.

## Boundary

- The request is a controlled diagnostic: the trace and request carry
  plaintext weights and activations; this is not a hidden production proof.
- Field equality is modulo the base field order; integer bounds are those of
  the traced MatMul, not re-proved here.
- The 16 claims are independent; no batching coefficient or combined error
  bound.
- The production CLI still does not carry this chain; this is the callable
  stage/test entry.

## Logs

- `real-trace-manifest.txt`: the full test output.
