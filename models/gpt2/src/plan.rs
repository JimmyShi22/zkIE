//! Static execution plan for the GPT-2 shared trace.
//!
//! The plan mirrors the executor's construction order. It reads no proof and
//! performs no model execution.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_ops::op::OpCode;
use zkie_ops::proof::model_commitment::ModelAssetSpec;
use zkie_ops::proof::plan::{
    config, PlanBuilder, PlanConstant, PlanLayout, PlanSource, TracePlan,
};

use crate::config;

const LAYERS: usize = config::LAYERS;
const HEADS: usize = config::HEADS;
const HEAD_DIM: usize = config::HEAD_DIM;
const HIDDEN: usize = config::HIDDEN_PAD;
const REAL: usize = config::HIDDEN;
const FFN: usize = config::FFN;
const VOCAB: usize = config::VOCAB_PAD;
const VALID_VOCAB: usize = config::VOCAB;
const SHIFT: u32 = config::BASE_SHIFT;
const CAUSAL_MASK: i64 = config::CAUSAL_MASK;
const VOCAB_MASK: i64 = config::VOCAB_MASK;
const GELU_OFFSET: i64 = config::GELU_OFFSET;
const GELU_TABLE_LEN: usize = config::GELU_TABLE_LEN;
const NORM_OUTPUT_SHIFT: u32 = config::NORM_OUTPUT_SHIFT;
const QK_SCALE_FACTOR: i64 = config::QK_SCALE_FACTOR;

const POS_EMB: &str = "gpt2/weights/pos_emb_i32.bin";
const LM_HEAD: &str = "gpt2/weights/lm_head_w_i32.bin";
const RSQRT: &str = "gpt2/weights/rsqrt_table_i32.bin";
const EXP: &str = "gpt2/weights/exp_table_i32.bin";
const GELU: &str = "gpt2/weights/gelu_table_i32.bin";

fn layer_path(layer: usize, suffix: &str) -> String {
    format!("gpt2/weights/L{layer}_{suffix}_i32.bin")
}

fn project(
    builder: &mut PlanBuilder,
    x_step: usize,
    weight: PlanSource,
    bias: PlanSource,
    m: usize,
    n: usize,
    k: usize,
) -> usize {
    let proj = builder.step(
        OpCode::Projection,
        config(vec![m as i64, n as i64, k as i64, SHIFT as i64, 0]),
        vec![PlanSource::prev(x_step, 0), weight],
        1,
    );
    builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![PlanSource::prev(proj, 0), bias],
        1,
    )
}

fn ln(
    builder: &mut PlanBuilder,
    x_step: usize,
    weight: &str,
    bias: &str,
    rsqrt: &str,
    m: usize,
) -> usize {
    builder.step(
        OpCode::ProfileLayerNorm,
        config(vec![
            m as i64,
            HIDDEN as i64,
            REAL as i64,
            NORM_OUTPUT_SHIFT as i64,
            0,
            0,
            0,
        ]),
        vec![
            PlanSource::prev(x_step, 0),
            PlanSource::asset(weight).with_layout(PlanLayout::broadcast(m)),
            PlanSource::asset(bias).with_layout(PlanLayout::broadcast(m)),
            PlanSource::asset(rsqrt),
        ],
        1,
    )
}

fn affine(
    builder: &mut PlanBuilder,
    x_step: usize,
    multiplier: i64,
) -> usize {
    builder.step(
        OpCode::AffineScale,
        config(vec![multiplier, 0, SHIFT as i64, 0, 0, 0]),
        vec![PlanSource::prev(x_step, 0)],
        1,
    )
}

pub fn trace_plan(m: usize) -> Result<TracePlan> {
    if m == 0 || m > config::MAX_SEQUENCE {
        return Err(Error::InvalidConfig);
    }
    let mut builder = PlanBuilder::new();

    let embed_idx = builder.step(
        OpCode::EmbeddingIndex,
        config(vec![HIDDEN as i64, VOCAB as i64]),
        vec![PlanSource::Constant(PlanConstant::Tokens)],
        1,
    );
    let embedded = builder.step(
        OpCode::TableLookup,
        config(Vec::new()),
        vec![PlanSource::asset(LM_HEAD), PlanSource::prev(embed_idx, 0)],
        1,
    );
    let mut x_step = builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![
            PlanSource::prev(embedded, 0),
            PlanSource::asset_slice(POS_EMB, PlanLayout::range(0, m * HIDDEN)),
        ],
        1,
    );

    for layer in 0..LAYERS {
        let path = |suffix: &str| layer_path(layer, suffix);
        let h_step = ln(
            &mut builder,
            x_step,
            &path("ln1_w"),
            &path("ln1_b"),
            RSQRT,
            m,
        );
        let mut head_acc: Option<usize> = None;
        for head in 0..HEADS {
            let qw = PlanSource::asset_slice(
                path("q_w"),
                PlanLayout::columns(HIDDEN, head * HEAD_DIM, HEAD_DIM),
            );
            let qb = PlanSource::asset_slice(
                path("q_b"),
                PlanLayout::chain(vec![
                    PlanLayout::columns(HIDDEN, head * HEAD_DIM, HEAD_DIM),
                    PlanLayout::broadcast(m),
                ]),
            );
            let q_proj = project(&mut builder, h_step, qw, qb, m, HEAD_DIM, HIDDEN);
            let q = affine(&mut builder, q_proj, QK_SCALE_FACTOR);
            let kw = PlanSource::asset_slice(
                path("k_w"),
                PlanLayout::columns(HIDDEN, head * HEAD_DIM, HEAD_DIM),
            );
            let kb = PlanSource::asset_slice(
                path("k_b"),
                PlanLayout::chain(vec![
                    PlanLayout::columns(HIDDEN, head * HEAD_DIM, HEAD_DIM),
                    PlanLayout::broadcast(m),
                ]),
            );
            let k_proj = project(&mut builder, h_step, kw, kb, m, HEAD_DIM, HIDDEN);
            let k = affine(&mut builder, k_proj, QK_SCALE_FACTOR);
            let vw = PlanSource::asset_slice(
                path("v_w"),
                PlanLayout::columns(HIDDEN, head * HEAD_DIM, HEAD_DIM),
            );
            let vb = PlanSource::asset_slice(
                path("v_b"),
                PlanLayout::chain(vec![
                    PlanLayout::columns(HIDDEN, head * HEAD_DIM, HEAD_DIM),
                    PlanLayout::broadcast(m),
                ]),
            );
            let v = project(&mut builder, h_step, vw, vb, m, HEAD_DIM, HIDDEN);

            let scores = builder.step(
                OpCode::MatMul,
                config(vec![m as i64, m as i64, HEAD_DIM as i64, SHIFT as i64, 0]),
                vec![
                    PlanSource::prev(q, 0),
                    PlanSource::prev_slice(
                        k,
                        0,
                        PlanLayout::transpose(m, HEAD_DIM),
                    ),
                ],
                1,
            );
            let masked = builder.step(
                OpCode::Add,
                config(Vec::new()),
                vec![
                    PlanSource::prev(scores, 0),
                    PlanSource::Constant(PlanConstant::CausalMask {
                        rows: m,
                        cols: m,
                        masked: CAUSAL_MASK,
                    }),
                ],
                1,
            );
            let (probs, probs_edge): (PlanSource, Option<usize>) = if m > 16 {
                let softmax = builder.step(
                    OpCode::SoftmaxRows,
                    config(vec![m as i64, m as i64, SHIFT as i64]),
                    vec![PlanSource::prev(masked, 0), PlanSource::asset(EXP)],
                    1,
                );
                (PlanSource::prev(softmax, 0), Some(softmax))
            } else {
                let mut rows = Vec::with_capacity(m);
                for row in 0..m {
                    let softmax = builder.step(
                        OpCode::Softmax,
                        config(vec![SHIFT as i64]),
                        vec![
                            PlanSource::prev_slice(
                                masked,
                                0,
                                PlanLayout::rows(m, row, 1),
                            ),
                            PlanSource::asset(EXP),
                        ],
                        1,
                    );
                    rows.push(PlanSource::prev(softmax, 0));
                }
                (PlanSource::Concat(rows), None)
            };
            let _ = probs_edge;
            let attn = builder.step(
                OpCode::MatMul,
                config(vec![m as i64, HEAD_DIM as i64, m as i64, SHIFT as i64, 0]),
                vec![probs, PlanSource::prev(v, 0)],
                1,
            );
            let ow = PlanSource::asset_slice(
                path("o_proj_w"),
                PlanLayout::rows(HIDDEN, head * HEAD_DIM, HEAD_DIM),
            );
            let head_out = project(
                &mut builder,
                attn,
                ow,
                PlanSource::Constant(PlanConstant::Zeros { count: m * HIDDEN }),
                m,
                HIDDEN,
                HEAD_DIM,
            );
            head_acc = Some(match head_acc {
                None => head_out,
                Some(acc) => builder.step(
                    OpCode::Add,
                    config(Vec::new()),
                    vec![PlanSource::prev(acc, 0), PlanSource::prev(head_out, 0)],
                    1,
                ),
            });
        }
        let head_acc = head_acc.ok_or(Error::IncompleteArtifact)?;
        let attn_biased = builder.step(
            OpCode::Add,
            config(Vec::new()),
            vec![
                PlanSource::prev(head_acc, 0),
                PlanSource::asset(&path("o_proj_b")).with_layout(PlanLayout::broadcast(m)),
            ],
            1,
        );
        let x2 = builder.step(
            OpCode::Add,
            config(Vec::new()),
            vec![
                PlanSource::prev(x_step, 0),
                PlanSource::prev(attn_biased, 0),
            ],
            1,
        );
        let h2 = ln(
            &mut builder,
            x2,
            &path("ln2_w"),
            &path("ln2_b"),
            RSQRT,
            m,
        );
        let fc = project(
            &mut builder,
            h2,
            PlanSource::asset(path("fc_w")),
            PlanSource::asset(&path("fc_b")).with_layout(PlanLayout::broadcast(m)),
            m,
            FFN,
            HIDDEN,
        );
        let idx = builder.step(
            OpCode::TableIndex,
            config(vec![GELU_OFFSET, GELU_TABLE_LEN as i64]),
            vec![PlanSource::prev(fc, 0)],
            1,
        );
        let act = builder.step(
            OpCode::TableLookup,
            config(Vec::new()),
            vec![PlanSource::asset(GELU), PlanSource::prev(idx, 0)],
            1,
        );
        let proj = project(
            &mut builder,
            act,
            PlanSource::asset(path("proj_w")),
            PlanSource::asset(&path("proj_b")).with_layout(PlanLayout::broadcast(m)),
            m,
            HIDDEN,
            FFN,
        );
        x_step = builder.step(
            OpCode::Add,
            config(Vec::new()),
            vec![PlanSource::prev(x2, 0), PlanSource::prev(proj, 0)],
            1,
        );
    }

    let hf = ln(
        &mut builder,
        x_step,
        "gpt2/weights/ln_f_w_i32.bin",
        "gpt2/weights/ln_f_b_i32.bin",
        RSQRT,
        m,
    );
    let logits = builder.step(
        OpCode::MatMul,
        config(vec![m as i64, VOCAB as i64, HIDDEN as i64, 0, 0]),
        vec![PlanSource::prev(hf, 0), PlanSource::asset(LM_HEAD)],
        1,
    );
    let masked_logits = builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![
            PlanSource::prev(logits, 0),
            PlanSource::Constant(PlanConstant::VocabMask {
                rows: m,
                vocab: VOCAB,
                valid: VALID_VOCAB,
                masked: VOCAB_MASK,
            }),
        ],
        1,
    );
    for row in 0..m {
        let argmax = builder.step(
            OpCode::Argmax,
            config(Vec::new()),
            vec![PlanSource::prev_slice(
                masked_logits,
                0,
                PlanLayout::rows(VOCAB, row, 1),
            )],
            1,
        );
        builder.argmax_step(argmax);
    }
    let mut plan = builder.finish();
    plan.model_version = MODEL_VERSION;
    plan.declare_previous_object_for_activation(hf, m, HIDDEN)?;
    Ok(plan)
}

fn transcript_digest(tag: u64, data: &[u8]) -> Result<Digest> {
    zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            tag,
        ),
        0,
        None,
        data,
    )
}

pub const MODEL_VERSION: u32 = 1;

pub fn valid_vocab() -> usize {
    VALID_VOCAB
}

pub fn parameters_digest() -> Result<Digest> {
    transcript_digest(14, b"gpt2-seq16")
}

pub fn model_commitment(tokens: &[u32]) -> Result<Digest> {
    let bytes: Vec<u8> = tokens.iter().flat_map(|value| value.to_le_bytes()).collect();
    transcript_digest(15, &bytes)
}

pub fn asset_specs() -> Vec<ModelAssetSpec> {
    crate::executor::asset_specs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_step_counts_match_executor() {
        assert_eq!(trace_plan(16).unwrap().steps.len(), 4462);
        assert_eq!(trace_plan(512).unwrap().steps.len(), 2798);
    }

    #[test]
    fn output_head_declares_the_previous_object() {
        for m in [16, 512] {
            let plan = trace_plan(m).unwrap();
            let object = plan
                .previous_object
                .as_ref()
                .expect("the output-head plan declares a previous object");
            assert_eq!(object.shape.positions, m);
            assert_eq!(object.shape.hidden, HIDDEN);
            assert_eq!(object.layout.rows, m);
            assert_eq!(object.layout.columns, HIDDEN);
            assert_eq!(object.layout.row_start, 0);
            assert_eq!(object.layout.column_start, 0);
            assert_eq!(object.consumer_port, 0);
            assert!(object.owner.step < object.consumer_step);
            assert_eq!(
                plan.steps[object.consumer_step].opcode,
                OpCode::ProfileLayerNorm
            );
            let derived = zkie_ops::proof::plan::PreviousObjectPlan::derive_for_activation(
                &plan,
                object.consumer_step,
                m,
                HIDDEN,
            )
            .unwrap();
            assert_eq!(&derived, object);
        }
    }
}
