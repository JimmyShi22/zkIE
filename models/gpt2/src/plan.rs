//! Static execution plan for the GPT-2 shared trace.
//!
//! The plan mirrors the executor's construction order. It reads no proof and
//! performs no model execution.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_ops::op::OpCode;
use zkie_ops::proof::model_commitment::ModelAssetSpec;
use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
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

pub const MODEL_VERSION: u32 = 2;

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

/// The trusted public identities of the migrated tables. The bytes are the
/// fixed public files; the digests are SHA-256 of those bytes and the paths
/// only locate them under the case asset root.
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
            EXP,
            "c6a3e6fae423e86bc2598392e4055c938381616c738414fa8bad859da4004f55",
            2_097_152,
        ),
        table(
            RSQRT,
            "7cba6a7c88053b99112c1271a2d593e1147a4b3b4a92fe1091c148e2e33573ed",
            7_340_032,
        ),
        table(
            GELU,
            "42cd0f218e5635bb2c1e94076f8f5da3d92456ce30960b1463774dca2dad3e5d",
            16_777_216,
        ),
    ]
}

/// The migrated public table paths of this model.
pub fn public_table_paths() -> &'static [&'static str] {
    &[EXP, RSQRT, GELU]
}

/// The consumer-authenticated assets of this model: the first projection
/// weight is proved by the request consumer chain over the committed table,
/// never by a trace opening.
pub fn consumer_asset_paths() -> &'static [&'static str] {
    &[FIRST_PROJECTION]
}

/// The production plan: the migrated tables are trusted public files, not
/// traced assets, and the first projection weight is consumer-authenticated.
/// Component tests keep `trace_plan`.
pub fn trace_plan_public(m: usize) -> Result<TracePlan> {
    let mut plan = trace_plan(m)?;
    zkie_ops::proof::plan::promote_public_tables(&mut plan, &public_tables());
    zkie_ops::proof::plan::promote_consumer_assets(&mut plan, consumer_asset_paths())?;
    Ok(plan)
}

pub fn asset_specs() -> Vec<ModelAssetSpec> {
    crate::executor::asset_specs()
}

/// The fixed public candidate of the production encoded table. The candidate
/// is chosen offline inside the declared support domain and written here; the
/// derivation below only expands and checks it, it never searches.
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

/// The candidate declared for this round. The random tail carries the
/// derived one-time query count and one declared later batch; the later
/// distance and its columns are derived from the same public geometry and
/// security targets as the one-time rule, never chosen here.
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

const FIRST_PROJECTION: &str = "gpt2/weights/L0_k_w_i32.bin";

/// The public model-level identity of the encoded tables. It covers the model
/// name, version, parameters and asset identities, never the request tokens,
/// so one table configuration is shared by seq16 and seq512.
pub fn table_model_identity() -> Result<Digest> {
    let parameters = parameters_digest()?;
    let mut data = Vec::new();
    data.extend_from_slice(b"gpt2-table-model-v1");
    data.extend_from_slice(&MODEL_VERSION.to_le_bytes());
    data.extend_from_slice(&parameters.0);
    for spec in asset_specs() {
        data.extend_from_slice(
            &zkie_ops::proof::model_commitment::asset_identity(&spec).0,
        );
    }
    zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            20,
        ),
        0,
        None,
        &data,
    )
}

/// The deterministically derived table plan of one candidate. The table
/// configuration and plan identity are known before any randomness or tree;
/// the instance root and record are produced by the commit and checked
/// against the trusted meta afterwards.
#[derive(Clone, Debug)]
pub struct DerivedTablePlan {
    pub params: AssetTableParams,
    pub table_shape: IndexShape,
    pub table_configuration: Digest,
    pub table_identity: Digest,
    pub table_model_identity: Digest,
    pub one_time_columns: u64,
    pub later_batch_columns: u64,
    /// The later batches this candidate declares (one this round); the
    /// derivation's capacity is reported separately and never implies more.
    pub declared_later_batches: u64,
    pub total_columns: u64,
    pub lin_check: SharedLinCheck,
}

/// Derives and checks the table plan of one fixed candidate. The one-time
/// distance and columns come from the one-time rule; the later distance and
/// columns come from the same public geometry and security targets with the
/// record check removed, capped at the one-time distance. The declared later
/// batch must fit both the derived capacity and the code's random tail; the
/// total declared column use covers the one-time columns and the declared
/// batches and must fit the real hiding positions.
pub fn derive_table_plan(candidate: &TablePlanCandidate) -> Result<DerivedTablePlan> {
    if candidate.later_batches != 1 {
        return Err(Error::InvalidConfig);
    }
    let spec = asset_specs()
        .into_iter()
        .find(|spec| spec.path == FIRST_PROJECTION)
        .ok_or(Error::InvalidConfig)?;
    if spec.row_width != HIDDEN as usize
        || spec.element_count != (HIDDEN as u64) * (HIDDEN as u64)
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
    // The later candidate is derived from the same public geometry and
    // security target; it must agree with the one-time derivation, and the
    // declared batch must fit both the derived capacity and the code's tail.
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
        valid_min: i32::MIN as i64,
        valid_max: i32::MAX as i64,
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

/// The trusted table parameters of one square projection weight, derived from
/// the fixed public candidate. Only the registered first projection is
/// admitted; every other path keeps the raw row commitment.
pub fn projection_table_params(path: &str) -> Option<AssetTableParams> {
    if path != FIRST_PROJECTION {
        return None;
    }
    derive_table_plan(&TABLE_PLAN_CANDIDATE)
        .ok()
        .map(|plan| plan.params)
}

/// The trusted encoded-table list of the model plan. The layer-0 key
/// projection weight is the first migrated consumer; every other asset keeps
/// the raw row commitment until its own consumer relation closes.
pub fn asset_table_params() -> Result<Vec<AssetTableParams>> {
    let plan = derive_table_plan(&TABLE_PLAN_CANDIDATE)?;
    let mut tables = vec![plan.params];
    tables.extend(norm_param_table_params()?);
    Ok(tables)
}


/// The trusted table parameters of the two final-norm parameter assets. Each
/// asset is one 1,024-entry row; the encoded table has one row and the code
/// message carries the padded row. The parameters are plan values, never
/// proof data.
pub fn norm_param_table_params() -> Result<Vec<AssetTableParams>> {
    let specs = asset_specs();
    let mut params = Vec::new();
    for path in [
        "gpt2/weights/ln_f_w_i32.bin",
        "gpt2/weights/ln_f_b_i32.bin",
    ] {
        let spec = specs
            .iter()
            .find(|spec| spec.path == path)
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
            valid_min: i32::MIN as i64,
            valid_max: i32::MAX as i64,
            layout,
        };
        validate_asset_table_params(&spec, &table)?;
        params.push(table);
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_param_table_params_are_admitted() {
        let params = norm_param_table_params().expect("the two parameter tables");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].path, "gpt2/weights/ln_f_w_i32.bin");
        assert_eq!(params[1].path, "gpt2/weights/ln_f_b_i32.bin");
        for table in &params {
            assert_eq!(table.shape.rows(), 1);
            assert!(table.shape.plain() >= 1024);
            assert_eq!(table.row_width, 1024);
        }
    }



    #[test]
    fn projection_table_params_describe_the_first_k_weight() {
        let path = "gpt2/weights/L0_k_w_i32.bin";
        let params = projection_table_params(path).expect("table params");
        let spec = asset_specs()
            .into_iter()
            .find(|spec| spec.path == path)
            .expect("asset spec");
        validate_asset_table_params(&spec, &params).unwrap();
        assert_eq!(params.shape.rows(), 1024);
        assert_eq!(params.element_count, 1024 * 1024);
        assert_eq!(params.shape.code.ell, 2048);
        assert_eq!(params.shape.code.r, 128);
        assert!(params.shape.distances.m_once >= 3);
        assert_eq!(params.shape.distances.m_later, 284);
        assert_eq!(params.layout.tail_len, 128);
        assert_eq!(projection_table_params("gpt2/weights/L1_k_w_i32.bin"), None);
    }

    #[test]
    fn the_table_plan_is_derived_and_checked() {
        let plan = derive_table_plan(&TABLE_PLAN_CANDIDATE).expect("derived plan");
        let again = derive_table_plan(&TABLE_PLAN_CANDIDATE).expect("derived plan");
        assert_eq!(plan.params.shape, again.params.shape);
        assert_eq!(plan.params.layout, again.params.layout);
        assert_eq!(plan.params.path, again.params.path);
        assert_eq!(plan.table_identity, again.table_identity);
        assert_eq!(plan.table_configuration, again.table_configuration);
        assert_eq!(plan.one_time_columns, 52);
        assert_eq!(plan.later_batch_columns, 52);
        assert_eq!(plan.declared_later_batches, 1);
        assert_eq!(plan.total_columns, 104);
        assert_eq!(
            plan.total_columns,
            plan.one_time_columns + plan.declared_later_batches * plan.later_batch_columns
        );
        assert!(plan.total_columns <= plan.params.shape.code.r);
        assert_eq!(plan.params.shape.distances.m_once, {
            let code = build_code_plan(
                TABLE_PLAN_CANDIDATE.log_n,
                TABLE_PLAN_CANDIDATE.ell,
                TABLE_PLAN_CANDIDATE.r,
            )
            .unwrap();
            recorded_table_distance(
                code,
                TABLE_PLAN_CANDIDATE.v,
                TABLE_PLAN_CANDIDATE.answers,
                TABLE_PLAN_CANDIDATE.v,
                TABLE_PLAN_CANDIDATE.sharing,
                0,
            )
            .unwrap()
        });
        assert_eq!(plan.params.shape.distances.m_later, plan.params.shape.distances.m_once);
    }

    #[test]
    fn a_short_tail_or_declared_later_batch_is_refused() {
        let mut short = TABLE_PLAN_CANDIDATE;
        short.r = 8;
        assert!(derive_table_plan(&short).is_err());
        let mut more_batches = TABLE_PLAN_CANDIDATE;
        more_batches.later_batches = 2;
        assert!(derive_table_plan(&more_batches).is_err());
        let mut no_batches = TABLE_PLAN_CANDIDATE;
        no_batches.later_batches = 0;
        assert!(derive_table_plan(&no_batches).is_err());
        let mut wider = TABLE_PLAN_CANDIDATE;
        wider.r = 256;
        let plan = derive_table_plan(&wider).unwrap();
        assert_ne!(plan.table_identity, derive_table_plan(&TABLE_PLAN_CANDIDATE).unwrap().table_identity);
    }

    #[test]
    fn the_plan_declares_the_first_projection_table() {
        let tables = asset_table_params().unwrap();
        assert_eq!(tables.len(), 3);
        assert_eq!(tables[0].path, "gpt2/weights/L0_k_w_i32.bin");
        assert_eq!(tables[1].path, "gpt2/weights/ln_f_w_i32.bin");
        assert_eq!(tables[2].path, "gpt2/weights/ln_f_b_i32.bin");
        for table in &tables {
            let spec = asset_specs()
                .into_iter()
                .find(|spec| spec.path == table.path)
                .expect("asset spec");
            zkie_ops::proof::asset_table::validate_asset_table_params(&spec, table).unwrap();
        }
    }

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
