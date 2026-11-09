//! Static execution plan for the TimesFM shared trace.
//!
//! The plan mirrors the executor's construction order. It reads no proof and
//! performs no model execution.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_ops::op::OpCode;
use zkie_ops::proof::model_commitment::ModelAssetSpec;
use zkie_ops::proof::plan::{
    config, PlanBuilder, PlanConfigValue, PlanConstant, PlanLayout, PlanSource, TracePlan,
};

use crate::config;

const LAYERS: usize = config::LAYERS;
const HEADS: usize = config::HEADS;
const HEAD_DIM: usize = config::HEAD_DIM;
const HIDDEN: usize = config::HIDDEN;
const H_PAD: usize = config::HIDDEN_PAD;
const SEQ: usize = config::PATCH_TOKENS;
const PATCH: usize = config::RAW_POINTS_PER_PATCH;
const QKV_COLS: usize = 4096;
const QKV_BIAS_COLS: usize = 3 * HIDDEN;
const SHIFT: u32 = config::BASE_SHIFT;
const NORM_OUTPUT_SHIFT: u32 = config::NORM_OUTPUT_SHIFT;
const SILU_OFFSET: i64 = config::SILU_OFFSET;
const SILU_LEN: usize = config::SILU_TABLE_LEN;
const OUTPUT_SCALE_SHIFT: u32 = config::OUTPUT_SCALE_SHIFT;

const WEIGHT_DIR: &str = "models/timesfm/weights";
const INPUT_PATH: &str = "cases/timesfm/input_ts_i32.bin";
const RSQRT: &str = "models/gpt2/weights/rsqrt_table_i32.bin";
const EXP: &str = "models/gpt2/weights/exp_table_i32.bin";

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

fn add_bias(
    builder: &mut PlanBuilder,
    x_step: usize,
    bias: PlanSource,
) -> usize {
    builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![PlanSource::prev(x_step, 0), bias],
        1,
    )
}

fn silu(
    builder: &mut PlanBuilder,
    x_step: usize,
    silu_table: &str,
    silu_pool_external: bool,
) -> usize {
    let idx = builder.step(
        OpCode::TableIndex,
        config(vec![SILU_OFFSET, SILU_LEN as i64]),
        vec![PlanSource::prev(x_step, 0)],
        1,
    );
    let table = PlanSource::asset(silu_table);
    let _ = silu_pool_external;
    builder.step(
        OpCode::TableLookup,
        config(Vec::new()),
        vec![table, PlanSource::prev(idx, 0)],
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
            PlanSource::asset(weight).with_layout(PlanLayout::broadcast(rows)),
            PlanSource::asset(RSQRT),
        ],
        1,
    )
}

fn layer_norm(
    builder: &mut PlanBuilder,
    x_step: usize,
    weight: &str,
    bias: &str,
    rows: usize,
    cols: usize,
    valid: usize,
) -> usize {
    builder.step(
        OpCode::ProfileLayerNorm,
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
            PlanSource::asset(weight).with_layout(PlanLayout::broadcast(rows)),
            PlanSource::asset(bias).with_layout(PlanLayout::broadcast(rows)),
            PlanSource::asset(RSQRT),
        ],
        1,
    )
}

fn relu(builder: &mut PlanBuilder, x_step: usize) -> usize {
    builder.step(
        OpCode::Relu,
        config(Vec::new()),
        vec![PlanSource::prev(x_step, 0)],
        1,
    )
}

fn columns(x_step: usize, row_width: usize, start: usize, count: usize) -> PlanSource {
    PlanSource::prev_slice(x_step, 0, PlanLayout::columns(row_width, start, count))
}

pub fn trace_plan(m: usize) -> Result<TracePlan> {
    if m != config::PATCH_TOKENS {
        return Err(Error::InvalidConfig);
    }
    let mut builder = PlanBuilder::new();
    let mut x_step = builder.step(
        OpCode::ProfileStandardize,
        config(vec![
            SEQ as i64,
            PATCH as i64,
            (2 * PATCH) as i64,
            PATCH as i64,
            0,
            SHIFT as i64,
            0,
            0,
            0,
        ]),
        vec![PlanSource::asset(INPUT_PATH), PlanSource::asset(RSQRT)],
        1,
    );

    let pro_hid_path = format!("{WEIGHT_DIR}/pro_hid_w_i32.bin");
    let hid = projection(
        &mut builder,
        x_step,
        PlanSource::asset(&pro_hid_path),
        SEQ,
        H_PAD,
        2 * PATCH,
    );
    let hid = add_bias(
        &mut builder,
        hid,
        PlanSource::asset(format!("{WEIGHT_DIR}/pro_hid_b_i32.bin"))
            .with_layout(PlanLayout::broadcast(SEQ)),
    );
    let activation = silu(
        &mut builder,
        hid,
        &format!("{WEIGHT_DIR}/silu_table_i32.bin"),
        true,
    );
    let pro_out_path = format!("{WEIGHT_DIR}/pro_out_w_i32.bin");
    let out_proj = projection(
        &mut builder,
        activation,
        PlanSource::asset(&pro_out_path),
        SEQ,
        H_PAD,
        H_PAD,
    );
    let out_proj = add_bias(
        &mut builder,
        out_proj,
        PlanSource::asset(format!("{WEIGHT_DIR}/pro_out_b_i32.bin"))
            .with_layout(PlanLayout::broadcast(SEQ)),
    );
    let pro_res_path = format!("{WEIGHT_DIR}/pro_res_w_i32.bin");
    let res_proj = projection(
        &mut builder,
        x_step,
        PlanSource::asset(&pro_res_path),
        SEQ,
        H_PAD,
        2 * PATCH,
    );
    let res_proj = add_bias(
        &mut builder,
        res_proj,
        PlanSource::asset(format!("{WEIGHT_DIR}/pro_res_b_i32.bin"))
            .with_layout(PlanLayout::broadcast(SEQ)),
    );
    let sum = builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![
            PlanSource::prev(out_proj, 0),
            PlanSource::prev(res_proj, 0),
        ],
        1,
    );
    let gather_path = format!("{WEIGHT_DIR}/gather_i32.bin");
    let sum = builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![PlanSource::prev(sum, 0), PlanSource::asset(&gather_path)],
        1,
    );
    x_step = builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![
            PlanSource::prev(sum, 0),
            PlanSource::asset(format!("{WEIGHT_DIR}/embedding_i32.bin"))
                .with_layout(PlanLayout::broadcast(SEQ)),
        ],
        1,
    );

    let mask = format!("{WEIGHT_DIR}/mask_q_i32.bin");
    let q_scale = format!("{WEIGHT_DIR}/q_scale_i32.bin");

    for layer in 0..LAYERS {
        let path = |suffix: &str| layer_path(layer, suffix);
        let h = rms(&mut builder, x_step, &path("lnw"), SEQ, H_PAD, HIDDEN);
        let qkv = projection(
            &mut builder,
            h,
            PlanSource::asset(path("qkv_w")),
            SEQ,
            QKV_COLS,
            H_PAD,
        );
        let padded_bias = PlanSource::Concat(vec![
            PlanSource::asset(path("qkv_b")),
            PlanSource::Constant(PlanConstant::Zeros {
                count: QKV_COLS - QKV_BIAS_COLS,
            }),
        ])
        .with_layout(PlanLayout::broadcast(SEQ));
        let qkv = add_bias(&mut builder, qkv, padded_bias);

        let mut attention = Vec::with_capacity(HEADS);
        for head in 0..HEADS {
            let q = columns(qkv, QKV_COLS, head * HEAD_DIM, HEAD_DIM);
            let k = columns(
                qkv,
                QKV_COLS,
                HIDDEN + head * HEAD_DIM,
                HEAD_DIM,
            );
            let v = columns(
                qkv,
                QKV_COLS,
                2 * HIDDEN + head * HEAD_DIM,
                HEAD_DIM,
            );
            let q_scaled = builder.step(
                OpCode::Hadamard,
                config(vec![SHIFT as i64]),
                vec![
                    q,
                    PlanSource::asset_slice(
                        &q_scale,
                        PlanLayout::chain(vec![
                            PlanLayout::range(0, HEAD_DIM),
                            PlanLayout::broadcast(SEQ),
                        ]),
                    ),
                ],
                1,
            );
            let scores = builder.step(
                OpCode::MatMul,
                config(vec![SEQ as i64, SEQ as i64, HEAD_DIM as i64, SHIFT as i64, 0]),
                vec![
                    PlanSource::prev(q_scaled, 0),
                    k.with_layout(PlanLayout::transpose(SEQ, HEAD_DIM)),
                ],
                1,
            );
            let masked = builder.step(
                OpCode::Add,
                config(Vec::new()),
                vec![PlanSource::prev(scores, 0), PlanSource::asset(&mask)],
                1,
            );
            let mut probs = Vec::with_capacity(SEQ);
            for row in 0..SEQ {
                probs.push(PlanSource::prev(
                    builder.step(
                        OpCode::Softmax,
                        config(vec![SHIFT as i64]),
                        vec![
                            PlanSource::prev_slice(
                                masked,
                                0,
                                PlanLayout::rows(SEQ, row, 1),
                            ),
                            PlanSource::asset(EXP),
                        ],
                        1,
                    ),
                    0,
                ));
            }
            let attn = builder.step(
                OpCode::MatMul,
                config(vec![SEQ as i64, HEAD_DIM as i64, SEQ as i64, SHIFT as i64, 0]),
                vec![
                    PlanSource::Concat(probs),
                    v,
                ],
                1,
            );
            attention.push(PlanSource::prev(attn, 0));
        }

        let o_proj = builder.step(
            OpCode::Projection,
            config(vec![SEQ as i64, H_PAD as i64, HIDDEN as i64, SHIFT as i64, 0]),
            vec![
                PlanSource::HeadConcat {
                    parts: attention,
                    rows: SEQ,
                },
                PlanSource::asset_slice(
                    path("o_proj_w"),
                    PlanLayout::rows(H_PAD, 0, HIDDEN),
                ),
            ],
            1,
        );
        let o_proj = add_bias(
            &mut builder,
            o_proj,
            PlanSource::asset(path("o_proj_b")).with_layout(PlanLayout::broadcast(SEQ)),
        );
        let x2 = builder.step(
            OpCode::Add,
            config(Vec::new()),
            vec![PlanSource::prev(x_step, 0), PlanSource::prev(o_proj, 0)],
            1,
        );
        let mlp = layer_norm(
            &mut builder,
            x2,
            &path("mlp_w"),
            &path("mlp_b"),
            SEQ,
            H_PAD,
            HIDDEN,
        );
        let gate = projection(
            &mut builder,
            mlp,
            PlanSource::asset(path("gate_w")),
            SEQ,
            H_PAD,
            H_PAD,
        );
        let gate = add_bias(
            &mut builder,
            gate,
            PlanSource::asset(path("gate_b")).with_layout(PlanLayout::broadcast(SEQ)),
        );
        let gate = relu(&mut builder, gate);
        let down = projection(
            &mut builder,
            gate,
            PlanSource::asset(path("down_w")),
            SEQ,
            H_PAD,
            H_PAD,
        );
        let down = add_bias(
            &mut builder,
            down,
            PlanSource::asset(path("down_b")).with_layout(PlanLayout::broadcast(SEQ)),
        );
        x_step = builder.step(
            OpCode::Add,
            config(Vec::new()),
            vec![PlanSource::prev(x2, 0), PlanSource::prev(down, 0)],
            1,
        );
    }

    let head_hid_path = format!("{WEIGHT_DIR}/head_hid_w_i32.bin");
    let hh = projection(
        &mut builder,
        x_step,
        PlanSource::asset(&head_hid_path),
        SEQ,
        H_PAD,
        H_PAD,
    );
    let hh = add_bias(
        &mut builder,
        hh,
        PlanSource::asset(format!("{WEIGHT_DIR}/head_hid_b_i32.bin"))
            .with_layout(PlanLayout::broadcast(SEQ)),
    );
    let hs = silu(
        &mut builder,
        hh,
        &format!("{WEIGHT_DIR}/silu_table_i32.bin"),
        true,
    );
    let head_out_path = format!("{WEIGHT_DIR}/head_out_w_i32.bin");
    let ho = projection(
        &mut builder,
        hs,
        PlanSource::asset(&head_out_path),
        SEQ,
        H_PAD,
        H_PAD,
    );
    let ho = add_bias(
        &mut builder,
        ho,
        PlanSource::asset(format!("{WEIGHT_DIR}/head_out_b_i32.bin"))
            .with_layout(PlanLayout::broadcast(SEQ)),
    );
    let head_res_path = format!("{WEIGHT_DIR}/head_res_w_i32.bin");
    let hr = projection(
        &mut builder,
        x_step,
        PlanSource::asset(&head_res_path),
        SEQ,
        H_PAD,
        H_PAD,
    );
    let hr = add_bias(
        &mut builder,
        hr,
        PlanSource::asset(format!("{WEIGHT_DIR}/head_res_b_i32.bin"))
            .with_layout(PlanLayout::broadcast(SEQ)),
    );
    let add = builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![PlanSource::prev(ho, 0), PlanSource::prev(hr, 0)],
        1,
    );
    let scale_path = format!("{WEIGHT_DIR}/scale_i64.bin");
    let scaled = builder.step(
        OpCode::AffineScale,
        vec![
            PlanConfigValue::AssetI64 {
                path: scale_path.clone(),
                index: 0,
            },
            PlanConfigValue::Literal(0),
            PlanConfigValue::Literal(OUTPUT_SCALE_SHIFT as i64),
            PlanConfigValue::Literal(0),
            PlanConfigValue::Literal(0),
            PlanConfigValue::Literal(0),
        ],
        vec![PlanSource::prev(add, 0)],
        1,
    );
    let output = builder.step(
        OpCode::Add,
        config(Vec::new()),
        vec![
            PlanSource::prev(scaled, 0),
            PlanSource::asset_slice(
                &scale_path,
                PlanLayout::chain(vec![
                    PlanLayout::range(1, 1),
                    PlanLayout::broadcast(SEQ * H_PAD),
                ]),
            ),
        ],
        1,
    );
    builder.final_output(output, 0);
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

pub fn output_elements(patches: usize) -> Result<usize> {
    if patches != config::PATCH_TOKENS {
        return Err(Error::InvalidConfig);
    }
    Ok(SEQ * H_PAD)
}

pub fn parameters_digest() -> Result<Digest> {
    transcript_digest(14, b"timesfm-16patch")
}

pub fn model_commitment(tokens: &[u32]) -> Result<Digest> {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|value| (*value as i32 as i64).to_le_bytes())
        .collect();
    transcript_digest(15, &bytes)
}

pub fn asset_specs() -> Vec<ModelAssetSpec> {
    crate::executor::asset_specs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_step_count_matches_executor() {
        assert_eq!(trace_plan(16).unwrap().steps.len(), 6683);
    }

    #[test]
    fn signed_input_commitment_matches_i64_encoding() {
        let values = [-5i32, 7, -1];
        let tokens: Vec<u32> = values.iter().map(|value| *value as u32).collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| (*value as i64).to_le_bytes())
            .collect();
        let expected = transcript_digest(15, &bytes).unwrap();
        assert_eq!(model_commitment(&tokens).unwrap(), expected);
    }
}
