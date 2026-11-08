//! Static execution plan for the Gemma 3 shared trace.
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
const PAD: usize = config::HIDDEN_PAD;
const REAL: usize = config::HIDDEN;
const FFN: usize = config::FFN;
const VOCAB: usize = config::VOCAB;
const SHIFT: u32 = config::BASE_SHIFT;
const CAUSAL_MASK: i64 = config::CAUSAL_MASK;
const NORM_OUTPUT_SHIFT: u32 = config::RMS_NORM_OUTPUT_SHIFT;
const Q_SCALE_FACTOR: i64 = config::Q_SCALE_FACTOR;
const ROPE_SHIFT: u32 = config::ROPE_SHIFT;

const WEIGHT_DIR: &str = "models/gemma3/weights";
const SHARED_DIR: &str = "models/gpt2/weights";
const EXTENSION_PATH: &str = "cases/gemma3/rsqrt_ext_i32.bin";

fn layer_path(layer: usize, suffix: &str) -> String {
    format!("{WEIGHT_DIR}/L{layer}_{suffix}_i32.bin")
}

fn projection(
    builder: &mut PlanBuilder,
    x_step: usize,
    weight: PlanSource,
    m: usize,
    n: usize,
    k: usize,
) -> usize {
    builder.step(
        OpCode::Projection,
        config(vec![m as i64, n as i64, k as i64, SHIFT as i64, 0]),
        vec![PlanSource::prev(x_step, 0), weight],
        1,
    )
}

fn rms(
    builder: &mut PlanBuilder,
    x_step: usize,
    weight: &str,
    rows: usize,
    cols: usize,
    valid: usize,
    m: usize,
) -> usize {
    builder.step(
        OpCode::ProfileRmsNorm,
        config(vec![
            rows as i64,
            cols as i64,
            valid as i64,
            NORM_OUTPUT_SHIFT as i64,
            0,
            0,
            0,
        ]),
        vec![
            PlanSource::prev(x_step, 0),
            PlanSource::asset(weight).with_layout(PlanLayout::broadcast(m)),
            PlanSource::asset(format!("{SHARED_DIR}/rsqrt_table_i32.bin")),
            PlanSource::asset(EXTENSION_PATH),
        ],
        1,
    )
}

fn rope(
    builder: &mut PlanBuilder,
    x_step: usize,
    cos: &str,
    sin: &str,
    m: usize,
) -> usize {
    let half = HEAD_DIM / 2;
    builder.step(
        OpCode::ProfileRope,
        config(vec![HEAD_DIM as i64, ROPE_SHIFT as i64]),
        vec![
            PlanSource::prev(x_step, 0),
            PlanSource::asset_slice(cos, PlanLayout::range(0, m * half)),
            PlanSource::asset_slice(sin, PlanLayout::range(0, m * half)),
        ],
        1,
    )
}

pub fn trace_plan(m: usize) -> Result<TracePlan> {
    if m == 0 || m > config::MAX_SEQUENCE {
        return Err(Error::InvalidConfig);
    }
    let mut builder = PlanBuilder::new();
    let embed_idx = builder.step(
        OpCode::RowEmbeddingIndex,
        config(vec![PAD as i64, VOCAB as i64]),
        vec![PlanSource::Constant(PlanConstant::Tokens)],
        1,
    );
    let embed_path = format!("{WEIGHT_DIR}/embed_tokens_i32.bin");
    let mut x_step = builder.step(
        OpCode::TableLookup,
        config(Vec::new()),
        vec![
            PlanSource::asset(&embed_path),
            PlanSource::prev(embed_idx, 0),
        ],
        1,
    );

    for layer in 0..LAYERS {
        let path = |suffix: &str| layer_path(layer, suffix);
        let h_step = rms(
            &mut builder,
            x_step,
            &path("in_norm"),
            m,
            PAD,
            REAL,
            m,
        );
        let k = projection(
            &mut builder,
            h_step,
            PlanSource::asset(path("k_w")),
            m,
            HEAD_DIM,
            PAD,
        );
        let v = projection(
            &mut builder,
            h_step,
            PlanSource::asset(path("v_w")),
            m,
            HEAD_DIM,
            PAD,
        );
        let k = rms(
            &mut builder,
            k,
            &path("k_norm"),
            m,
            HEAD_DIM,
            HEAD_DIM,
            m,
        );
        let (cos, sin) = if config::FULL_LAYERS.contains(&layer) {
            (
                format!("{WEIGHT_DIR}/rope_cos_global_i32.bin"),
                format!("{WEIGHT_DIR}/rope_sin_global_i32.bin"),
            )
        } else {
            (
                format!("{WEIGHT_DIR}/rope_cos_local_i32.bin"),
                format!("{WEIGHT_DIR}/rope_sin_local_i32.bin"),
            )
        };
        let k_rot = rope(&mut builder, k, &cos, &sin, m);

        let mut head_acc: Option<usize> = None;
        for head in 0..HEADS {
            let q = projection(
                &mut builder,
                h_step,
                PlanSource::asset_slice(
                    path("q_w"),
                    PlanLayout::columns(PAD, head * HEAD_DIM, HEAD_DIM),
                ),
                m,
                HEAD_DIM,
                PAD,
            );
            let q = rms(
                &mut builder,
                q,
                &path("q_norm"),
                m,
                HEAD_DIM,
                HEAD_DIM,
                m,
            );
            let q_rot = rope(&mut builder, q, &cos, &sin, m);
            let q_scaled = builder.step(
                OpCode::AffineScale,
                config(vec![Q_SCALE_FACTOR, 0, SHIFT as i64, 0, 0, 0]),
                vec![PlanSource::prev(q_rot, 0)],
                1,
            );
            let scores = builder.step(
                OpCode::MatMul,
                config(vec![m as i64, m as i64, HEAD_DIM as i64, SHIFT as i64, 0]),
                vec![
                    PlanSource::prev(q_scaled, 0),
                    PlanSource::prev_slice(
                        k_rot,
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
            let probs = if m > 16 {
                PlanSource::prev(
                    builder.step(
                        OpCode::SoftmaxRows,
                        config(vec![m as i64, m as i64, SHIFT as i64]),
                        vec![
                            PlanSource::prev(masked, 0),
                            PlanSource::asset(format!("{SHARED_DIR}/exp_table_i32.bin")),
                        ],
                        1,
                    ),
                    0,
                )
            } else {
                let mut rows = Vec::with_capacity(m);
                for row in 0..m {
                    rows.push(PlanSource::prev(
                        builder.step(
                            OpCode::Softmax,
                            config(vec![SHIFT as i64]),
                            vec![
                                PlanSource::prev_slice(
                                    masked,
                                    0,
                                    PlanLayout::rows(m, row, 1),
                                ),
                                PlanSource::asset(format!(
                                    "{SHARED_DIR}/exp_table_i32.bin"
                                )),
                            ],
                            1,
                        ),
                        0,
                    ));
                }
                PlanSource::Concat(rows)
            };
            let attn = builder.step(
                OpCode::MatMul,
                config(vec![m as i64, HEAD_DIM as i64, m as i64, SHIFT as i64, 0]),
                vec![probs, PlanSource::prev(v, 0)],
                1,
            );
            let head_out = projection(
                &mut builder,
                attn,
                PlanSource::asset_slice(
                    path("o_w"),
                    PlanLayout::rows(PAD, head * HEAD_DIM, HEAD_DIM),
                ),
                m,
                PAD,
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
        let attn_norm = rms(
            &mut builder,
            head_acc,
            &path("post_attn_norm"),
            m,
            PAD,
            REAL,
            m,
        );
        let x2 = builder.step(
            OpCode::Add,
            config(Vec::new()),
            vec![
                PlanSource::prev(x_step, 0),
                PlanSource::prev(attn_norm, 0),
            ],
            1,
        );
        let h2 = rms(
            &mut builder,
            x2,
            &path("pre_ffn_norm"),
            m,
            PAD,
            REAL,
            m,
        );
        let gate = projection(
            &mut builder,
            h2,
            PlanSource::asset(path("gate_w")),
            m,
            FFN,
            PAD,
        );
        let up = projection(
            &mut builder,
            h2,
            PlanSource::asset(path("up_w")),
            m,
            FFN,
            PAD,
        );
        let idx = builder.step(
            OpCode::TableIndex,
            config(vec![config::GELU_OFFSET, config::GELU_TABLE_LEN as i64]),
            vec![PlanSource::prev(gate, 0)],
            1,
        );
        let gelu = builder.step(
            OpCode::TableLookup,
            config(Vec::new()),
            vec![
                PlanSource::asset(format!("{SHARED_DIR}/gelu_table_i32.bin")),
                PlanSource::prev(idx, 0),
            ],
            1,
        );
        let act = builder.step(
            OpCode::GatedFfn,
            config(vec![SHIFT as i64]),
            vec![PlanSource::prev(gelu, 0), PlanSource::prev(up, 0)],
            1,
        );
        let down = projection(
            &mut builder,
            act,
            PlanSource::asset(path("down_w")),
            m,
            PAD,
            FFN,
        );
        let down_norm = rms(
            &mut builder,
            down,
            &path("post_ffn_norm"),
            m,
            PAD,
            REAL,
            m,
        );
        x_step = builder.step(
            OpCode::Add,
            config(Vec::new()),
            vec![PlanSource::prev(x2, 0), PlanSource::prev(down_norm, 0)],
            1,
        );
    }

    let hf = rms(
        &mut builder,
        x_step,
        &format!("{WEIGHT_DIR}/final_norm_i32.bin"),
        m,
        PAD,
        REAL,
        m,
    );
    let logits = builder.step(
        OpCode::MatMul,
        config(vec![m as i64, VOCAB as i64, PAD as i64, 0, 0]),
        vec![
            PlanSource::prev(hf, 0),
            PlanSource::asset(format!("{WEIGHT_DIR}/lm_head_i32.bin")),
        ],
        1,
    );
    for row in 0..m {
        let argmax = builder.step(
            OpCode::Argmax,
            config(Vec::new()),
            vec![PlanSource::prev_slice(
                logits,
                0,
                PlanLayout::rows(VOCAB, row, 1),
            )],
            1,
        );
        builder.argmax_step(argmax);
    }
    let mut plan = builder.finish();
    plan.model_version = MODEL_VERSION;
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
    config::VOCAB
}

pub fn parameters_digest() -> Result<Digest> {
    transcript_digest(14, b"gemma3-seq16")
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
        assert_eq!(trace_plan(16).unwrap().steps.len(), 2090);
        assert_eq!(trace_plan(512).unwrap().steps.len(), 1506);
    }
}
