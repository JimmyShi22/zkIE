//! Shared single-shot fixture used by several tests.

use zkie_core::session::{DecodeCoverage, EosPolicy, OutputPolicy, Statement};
use zkie_core::types::Digest;

use zkie_ops::proof::binding::OracleOpening;
use zkie_ops::single_shot::{
    build_single_shot_public, KvAddress, KvKind, SingleShotPublic, StepWitness,
};

pub const PROMPT: usize = 3;
pub const VOCAB: usize = 4;

pub fn statement(decode_len: usize) -> Statement {
    let tokens: Vec<u32> = (0..decode_len).map(|index| (index % VOCAB) as u32).collect();
    Statement {
        protocol_id: Digest([1u8; 32]),
        protocol_version: 1,
        parameter_digest: Digest([2u8; 32]),
        model_id: Digest([3u8; 32]),
        graph_digest: Digest([4u8; 32]),
        quantization_digest: Digest([5u8; 32]),
        tables_digest: Digest([6u8; 32]),
        prompt_tokens: (0..PROMPT).map(|index| index as u32).collect(),
        output_tokens: tokens,
        coverage: DecodeCoverage::full(decode_len as u32),
        eos_policy: EosPolicy::FixedLength,
        output_policy: OutputPolicy::ExactMatch,
    }
}

pub fn fixture(decode_len: usize) -> (SingleShotPublic, Vec<OracleOpening>, Vec<StepWitness>) {
    let statement = statement(decode_len);
    let mut logits_rows = Vec::with_capacity(decode_len);
    let mut witnesses = Vec::with_capacity(decode_len);
    let mut history: Vec<KvAddress> = Vec::new();
    for step in 0..decode_len {
        let token = statement.output_tokens[step];
        let mut logits = vec![0i64; VOCAB];
        logits[token as usize] = 10;
        logits_rows.push(logits);

        let position = (PROMPT - 1 + step) as u32;
        let mut appends = Vec::new();
        for kind in [KvKind::Key, KvKind::Value] {
            let address = KvAddress {
                layer: 0,
                head: 0,
                position,
                channel: 0,
                kind,
            };
            history.push(address);
            appends.push(address);
        }
        witnesses.push(StepWitness {
            token,
            kv_appends: appends,
            kv_reads: history.clone(),
            mask: vec![1u8; position as usize + 1],
            rope_offset: position,
        });
    }
    let (public, openings) =
        build_single_shot_public(statement, &logits_rows, 1, 1, 1).unwrap();
    (public, openings, witnesses)
}
