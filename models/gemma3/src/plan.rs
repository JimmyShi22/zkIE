//! Static execution plan for the Gemma 3 shared trace.
//!
//! The plan mirrors the executor's construction order. It reads no proof and
//! performs no model execution.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_ops::op::OpCode;
use zkie_ops::proof::model_commitment::ModelAssetSpec;
use zkie_engine::pcs::request::code::build_code_plan;
use zkie_engine::pcs::request::table::{IndexShape, TableDistances};
use zkie_ops::proof::asset_table::{
    asset_table_configuration, asset_table_identity, validate_asset_table_params,
    AssetTableLayout, AssetTableParams,
};
use zkie_ops::proof::hidden_send_session::SharedLinCheck;
use zkie_ops::proof::proximity_planner::{
    recorded_table_distance, shared_lin_columns, shared_lin_plan_fault, SharedLinPlanFault,
    SharedLinShape,
};
use zkie_ops::proof::plan::{
    config, PlanBuilder, PlanConstant, PlanLayout, PlanSource, TracePlan,
};
use zkie_ops::proof::table_range::{PRODUCTION_TABLE_RANGE_HI, PRODUCTION_TABLE_RANGE_LO};

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

pub const MODEL_VERSION: u32 = 2;

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

/// The trusted public table paths of this model.
pub fn public_table_paths() -> &'static [&'static str] {
    &[
        "models/gpt2/weights/exp_table_i32.bin",
        "models/gpt2/weights/rsqrt_table_i32.bin",
        "models/gpt2/weights/gelu_table_i32.bin",
        EXTENSION_PATH,
        "models/gemma3/weights/rope_cos_global_i32.bin",
        "models/gemma3/weights/rope_cos_local_i32.bin",
        "models/gemma3/weights/rope_sin_global_i32.bin",
        "models/gemma3/weights/rope_sin_local_i32.bin",
    ]
}

fn public_tables() -> Vec<(String, zkie_ops::proof::plan::PublicTable)> {
    use zkie_core::artifact::digest_from_hex;
    use zkie_ops::proof::plan::{PublicTable, PublicTableEncoding};
    let table = |path: &str, digest: &str, entries: u64| {
        (
            path.to_owned(),
            PublicTable {
                locator: path.to_owned(),
                digest: digest_from_hex(digest).expect("the fixed public table digest"),
                entries,
                encoding: PublicTableEncoding::I32LeSigned,
            },
        )
    };
    vec![
        table(
            "models/gpt2/weights/exp_table_i32.bin",
            "c6a3e6fae423e86bc2598392e4055c938381616c738414fa8bad859da4004f55",
            2_097_152,
        ),
        table(
            "models/gpt2/weights/rsqrt_table_i32.bin",
            "7cba6a7c88053b99112c1271a2d593e1147a4b3b4a92fe1091c148e2e33573ed",
            7_340_032,
        ),
        table(
            "models/gpt2/weights/gelu_table_i32.bin",
            "42cd0f218e5635bb2c1e94076f8f5da3d92456ce30960b1463774dca2dad3e5d",
            16_777_216,
        ),
        table(
            EXTENSION_PATH,
            "59fc2554a5d0f13abb265ed2eb5b56d5e75d82616561c60545013be7aa6dcf50",
            14_849_263,
        ),
        table(
            "models/gemma3/weights/rope_cos_global_i32.bin",
            "f1274259717ea29f4377d1b3a9510f44104007708de3f3a8bb15f35afe772bd1",
            65_536,
        ),
        table(
            "models/gemma3/weights/rope_cos_local_i32.bin",
            "59215ba7da66e69a9209ab0a39c068229aa974dd533752104e6075c4118b10c6",
            65_536,
        ),
        table(
            "models/gemma3/weights/rope_sin_global_i32.bin",
            "47c3d75d9b4a0f66476a20252e73cc3a0de33a68c422b36561755ce7f5dd5730",
            65_536,
        ),
        table(
            "models/gemma3/weights/rope_sin_local_i32.bin",
            "3096bad8797eed01cd733d508d726fff764630bce8fb3740b71746ad8e80f91d",
            65_536,
        ),
    ]
}

/// The first consumer-authenticated weight: the layer-0 query projection.
///
/// The layer-0 key and value projections are rectangular (1024 rows by 256
/// columns) and do not satisfy the existing key/range chain's square-matrix
/// geometry. The layer-0 query projection is the first registered square
/// weight (1024 by 1024) and is consumed by the four per-head projection
/// steps, the same head-slice form as the accepted chain.
pub const FIRST_PROJECTION: &str = "models/gemma3/weights/L0_q_w_i32.bin";

/// The consumer-authenticated assets of this model. The first key projection
/// is proved by the request consumer chain over the committed table, never by
/// a trace opening.
pub fn consumer_asset_paths() -> &'static [&'static str] {
    &[FIRST_PROJECTION]
}

/// The production plan: the shared tables, the rsqrt extension and the four
/// RoPE coefficient tables are trusted public files; the layer-0 key
/// projection is consumer-authenticated.
pub fn trace_plan_public(m: usize) -> Result<TracePlan> {
    let mut plan = trace_plan(m)?;
    zkie_ops::proof::plan::promote_public_tables(&mut plan, &public_tables());
    zkie_ops::proof::plan::promote_consumer_assets(&mut plan, consumer_asset_paths())?;
    Ok(plan)
}

/// The fixed public candidate of the production encoded table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TablePlanCandidate {
    pub log_n: u32,
    pub ell: u64,
    pub r: u64,
    pub v: u32,
    pub answers: u32,
    pub salt_bytes: usize,
    pub matrix_rows_log: u32,
    pub k_lo: u32,
    pub sharing: u64,
    pub later_batches: u64,
}

pub const TABLE_PLAN_CANDIDATE: TablePlanCandidate = TablePlanCandidate {
    log_n: 12,
    ell: 2048,
    r: 128,
    v: 10,
    answers: 1,
    salt_bytes: 40,
    matrix_rows_log: 10,
    k_lo: 0,
    sharing: 1,
    later_batches: 1,
};

/// The public model-level identity of the encoded tables.
pub fn table_model_identity() -> Result<Digest> {
    let parameters = parameters_digest()?;
    let mut data = Vec::new();
    data.extend_from_slice(b"gemma3-table-model-v1");
    data.extend_from_slice(&MODEL_VERSION.to_le_bytes());
    data.extend_from_slice(&parameters.0);
    for spec in asset_specs() {
        data.extend_from_slice(
            &zkie_ops::proof::model_commitment::asset_identity(&spec).0,
        );
    }
    transcript_digest(20, &data)
}

#[derive(Clone, Debug)]
pub struct DerivedTablePlan {
    pub params: AssetTableParams,
    pub table_shape: IndexShape,
    pub table_configuration: Digest,
    pub table_identity: Digest,
    pub table_model_identity: Digest,
    pub one_time_columns: u64,
    pub later_batch_columns: u64,
    pub declared_later_batches: u64,
    pub total_columns: u64,
    pub lin_check: SharedLinCheck,
}

/// Derives and checks the table plan of the fixed candidate over the real
/// layer-0 key projection shape: PAD rows and HEAD_DIM columns.
pub fn derive_table_plan(candidate: &TablePlanCandidate) -> Result<DerivedTablePlan> {
    if candidate.later_batches != 1 {
        return Err(Error::InvalidConfig);
    }
    let spec = asset_specs()
        .into_iter()
        .find(|spec| spec.path == FIRST_PROJECTION)
        .ok_or(Error::InvalidConfig)?;
    if spec.row_width != PAD
        || spec.element_count != (PAD as u64) * (PAD as u64)
    {
        return Err(Error::InvalidConfig);
    }
    let code = build_code_plan(candidate.log_n, candidate.ell, candidate.r)?;
    let rows_log = candidate.matrix_rows_log;
    let one_time_distance = recorded_table_distance(
        code,
        candidate.v,
        candidate.answers,
        candidate.v,
        candidate.sharing,
        0,
    )?;
    let shape = SharedLinShape {
        v: candidate.v,
        k_lo: candidate.k_lo,
        instances: 1,
        rows_log,
    };
    let one_time_columns = shared_lin_columns(shape, code, one_time_distance, 0, 0)?;
    let later = zkie_ops::proof::proximity_planner::derive_production_later_candidate(
        code,
        candidate.v,
        candidate.answers,
        candidate.v,
        rows_log,
        candidate.k_lo,
    )?;
    if later.m_once != one_time_distance || later.one_time_columns != one_time_columns {
        return Err(Error::InvalidConfig);
    }
    if later.capacity_batches < candidate.later_batches || later.later_columns == 0 {
        return Err(Error::InvalidConfig);
    }
    let later_batch_columns = later.later_columns;
    let total_columns = one_time_columns
        .checked_add(
            candidate
                .later_batches
                .checked_mul(later_batch_columns)
                .ok_or(Error::ResourceLimit)?,
        )
        .ok_or(Error::ResourceLimit)?;
    if total_columns > code.r {
        return Err(Error::InvalidConfig);
    }
    let fault = SharedLinPlanFault {
        v: candidate.v,
        k_lo: candidate.k_lo,
        rows_log,
        m: one_time_distance,
        column_draws: one_time_columns,
        point_work_bits: 0,
        column_work_bits: 0,
        zeta_work_bits: 0,
    };
    shared_lin_plan_fault(fault, code)?;
    let table_shape = IndexShape {
        code,
        v: candidate.v,
        answers: candidate.answers,
        salt_bytes: candidate.salt_bytes,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: later.m_once,
            m_later: later.m_later,
        },
    };
    let pads = 3u64
        .checked_mul(candidate.answers as u64)
        .ok_or(Error::ResourceLimit)?;
    let layout = AssetTableLayout {
        row_width: spec.row_width as u64,
        message_len: candidate.ell,
        tail_len: candidate.r,
        pads,
        valid_elements: spec.element_count,
        padding_zero: true,
    };
    let params = AssetTableParams {
        asset_id: spec.asset_id,
        path: spec.path.clone(),
        element_count: spec.element_count,
        row_width: spec.row_width as u64,
        encoding: spec.encoding.clone(),
        shape: table_shape,
        valid_min: PRODUCTION_TABLE_RANGE_LO,
        valid_max: PRODUCTION_TABLE_RANGE_HI,
        layout,
    };
    validate_asset_table_params(&spec, &params)?;
    let table_model_identity = table_model_identity()?;
    let table_configuration =
        asset_table_configuration(table_model_identity, &spec, &params)?;
    let table_identity = asset_table_identity(table_model_identity, &spec, &params)?;
    let lin_check = SharedLinCheck {
        point_work_bits: 0,
        zeta_work_bits: 0,
    };
    Ok(DerivedTablePlan {
        params,
        table_shape,
        table_configuration,
        table_identity,
        table_model_identity,
        one_time_columns,
        later_batch_columns,
        declared_later_batches: candidate.later_batches,
        total_columns,
        lin_check,
    })
}

/// The trusted table parameters of the registered first projection.
pub fn projection_table_params(path: &str) -> Option<AssetTableParams> {
    if path != FIRST_PROJECTION {
        return None;
    }
    derive_table_plan(&TABLE_PLAN_CANDIDATE)
        .ok()
        .map(|plan| plan.params)
}

/// The trusted encoded-table list of the model plan.
pub fn asset_table_params() -> Result<Vec<AssetTableParams>> {
    let plan = derive_table_plan(&TABLE_PLAN_CANDIDATE)?;
    let mut tables = vec![plan.params];
    tables.extend(norm_param_table_params()?);
    Ok(tables)
}

/// The trusted table parameters of the gain-only final norm asset: one
/// 1,024-entry row carried by one encoded table row.
pub fn norm_param_table_params() -> Result<Vec<AssetTableParams>> {
    let specs = asset_specs();
    let spec = specs
        .iter()
        .find(|spec| spec.path == "models/gemma3/weights/final_norm_i32.bin")
        .ok_or(Error::InvalidConfig)?
        .clone();
    let code = build_code_plan(12, 2049, 4)?;
    let shape = IndexShape {
        code,
        v: 0,
        answers: 1,
        salt_bytes: 40,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: 0,
            m_later: 0,
        },
    };
    let layout = AssetTableLayout {
        row_width: spec.row_width as u64,
        message_len: code.ell,
        tail_len: code.r,
        pads: shape.pads() as u64,
        valid_elements: spec.element_count,
        padding_zero: true,
    };
    let table = AssetTableParams {
        asset_id: spec.asset_id,
        path: spec.path.clone(),
        element_count: spec.element_count,
        row_width: spec.row_width as u64,
        encoding: spec.encoding.clone(),
        shape,
        valid_min: 0,
        valid_max: crate::norm_profile::GAIN_MAX,
        layout,
    };
    validate_asset_table_params(&spec, &table)?;
    Ok(vec![table])
}


pub fn asset_specs() -> Vec<ModelAssetSpec> {
    crate::executor::asset_specs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset_paths(source: &PlanSource, out: &mut Vec<String>) {
        match source {
            PlanSource::Asset { path, .. } | PlanSource::ConsumerAsset { path, .. } => {
                out.push(path.clone())
            }
            PlanSource::PublicTable { table, .. } => out.push(table.locator.clone()),
            PlanSource::Concat(parts) | PlanSource::HeadConcat { parts, .. } => {
                for part in parts {
                    asset_paths(part, out);
                }
            }
            PlanSource::Layout(inner, _) => asset_paths(inner, out),
            PlanSource::Prev { .. } | PlanSource::Constant(_) => {}
        }
    }

    #[test]
    fn the_public_plan_promotes_the_rsqrt_extension() {
        assert!(public_table_paths().contains(&EXTENSION_PATH));
        let plan = trace_plan_public(16).unwrap();
        let mut extension_public = false;
        for step in &plan.steps {
            for source in &step.inputs {
                match source {
                    PlanSource::Asset { path, .. } => {
                        assert_ne!(path, EXTENSION_PATH, "the extension stays an asset");
                    }
                    PlanSource::PublicTable { table, .. } => {
                        if table.locator == EXTENSION_PATH {
                            extension_public = true;
                            assert_eq!(table.entries, 14_849_263);
                            assert_eq!(
                                table.digest,
                                zkie_core::artifact::digest_from_hex(
                                    "59fc2554a5d0f13abb265ed2eb5b56d5e75d82616561c60545013be7aa6dcf50"
                                )
                                .unwrap()
                            );
                        }
                    }
                    _ => {}
                }
                let mut paths = Vec::new();
                asset_paths(source, &mut paths);
                assert!(!paths.iter().any(|path| path == EXTENSION_PATH) || extension_public);
            }
        }
        assert!(extension_public, "the extension source is public");
    }

    #[test]
    fn the_public_plan_promotes_the_rope_tables() {
        const ROPE: [(&str, &str); 4] = [
            (
                "models/gemma3/weights/rope_cos_global_i32.bin",
                "f1274259717ea29f4377d1b3a9510f44104007708de3f3a8bb15f35afe772bd1",
            ),
            (
                "models/gemma3/weights/rope_cos_local_i32.bin",
                "59215ba7da66e69a9209ab0a39c068229aa974dd533752104e6075c4118b10c6",
            ),
            (
                "models/gemma3/weights/rope_sin_global_i32.bin",
                "47c3d75d9b4a0f66476a20252e73cc3a0de33a68c422b36561755ce7f5dd5730",
            ),
            (
                "models/gemma3/weights/rope_sin_local_i32.bin",
                "3096bad8797eed01cd733d508d726fff764630bce8fb3740b71746ad8e80f91d",
            ),
        ];
        for (path, _) in ROPE {
            assert!(public_table_paths().contains(&path));
        }
        let m = 16;
        let half = HEAD_DIM / 2;
        let plan = trace_plan_public(m).unwrap();
        let mut global = false;
        let mut local = false;
        for step in &plan.steps {
            if step.opcode != OpCode::ProfileRope {
                continue;
            }
            let mut locators = Vec::new();
            for port in [1usize, 2] {
                match &step.inputs[port] {
                    PlanSource::PublicTable { table, layout } => {
                        assert_eq!(*layout, PlanLayout::range(0, m * half));
                        assert_eq!(table.entries, 65_536);
                        let (_, digest) = ROPE
                            .iter()
                            .find(|(path, _)| *path == table.locator.as_str())
                            .expect("the rope source is one of the fixed tables");
                        assert_eq!(
                            table.digest,
                            zkie_core::artifact::digest_from_hex(digest).unwrap()
                        );
                        locators.push(table.locator.clone());
                    }
                    other => panic!("the rope input is a public table: {other:?}"),
                }
            }
            let cos = &locators[0];
            let sin = &locators[1];
            assert_eq!(
                cos.replace("cos", "sin").as_str(),
                sin.as_str(),
                "cos and sin stay paired"
            );
            global |= cos.ends_with("cos_global_i32.bin");
            local |= cos.ends_with("cos_local_i32.bin");
        }
        assert!(global, "a global rope layer is promoted");
        assert!(local, "a local rope layer is promoted");
    }

    #[test]
    fn plan_step_counts_match_executor() {
        assert_eq!(trace_plan(16).unwrap().steps.len(), 2090);
        assert_eq!(trace_plan(512).unwrap().steps.len(), 1506);
    }

    #[test]
    fn projection_table_params_describe_the_first_k_weight() {
        let path = FIRST_PROJECTION;
        let params = projection_table_params(path).expect("table params");
        let spec = asset_specs()
            .into_iter()
            .find(|spec| spec.path == path)
            .expect("asset spec");
        validate_asset_table_params(&spec, &params).unwrap();
        assert_eq!(params.shape.rows(), 1024);
        assert_eq!(params.row_width, PAD as u64);
        assert_eq!(params.element_count, 1024 * 1024);
        assert_eq!(params.shape.code.ell, 2048);
        assert_eq!(params.shape.code.r, 128);
        assert!(params.shape.distances.m_once >= 3);
        assert_eq!(params.shape.distances.m_later, 284);
        assert_eq!(params.layout.tail_len, 128);
        assert_eq!(
            projection_table_params("models/gemma3/weights/L1_q_w_i32.bin"),
            None
        );
    }

    #[test]
    fn the_public_plan_defers_the_first_key_projection() {
        let plan = trace_plan_public(16).expect("public plan");
        let deferred = plan
            .steps
            .iter()
            .filter(|step| {
                step.inputs
                    .iter()
                    .any(|source| matches!(source, PlanSource::ConsumerAsset { .. }))
            })
            .count();
        assert_eq!(deferred, 4);
    }
}
