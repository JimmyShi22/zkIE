use std::io::Cursor;

use zkie_core::proof_header::read_header;
use zkie_ops::proof::model_commitment::SharedTrace;

#[test]
fn a_headerless_old_sample_is_rejected() {
    let sample = SharedTrace {
        tokens: vec![1, 2, 3],
        assets: Vec::new(),
        steps: Vec::new(),
        argmax: vec![0; 3],
        output: Vec::new(),
    };
    let bytes = bincode::serialize(&sample).unwrap();
    let error = read_header(&mut Cursor::new(&bytes)).unwrap_err();
    assert_eq!(error.code(), "invalid_proof");
    // This concrete sample starts with a length word, not the tag, so the
    // header parser rejects it. This does not assert that every old byte string
    // is rejected by the tag: old data may accidentally begin with a valid
    // header. A production verifier must fully decode and authenticate the
    // proof and must not fall back to the old implementation when parsing
    // fails.
}
