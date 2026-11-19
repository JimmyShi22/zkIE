//! Static execution plan for the TimesFM shared trace.
//!
//! The plan mirrors the executor's construction order. It reads no proof and
//! performs no model execution.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::build_code_plan;
use zkie_engine::pcs::request::table::{IndexShape, TableDistances};
use zkie_ops::op::OpCode;
use zkie_ops::proof::asset_table::{
    asset_table_configuration, asset_table_identity, validate_asset_table_params,
    AssetTableLayout, AssetTableParams,
};
use zkie_ops::proof::hidden_send_session::SharedLinCheck;
use zkie_ops::proof::model_commitment::ModelAssetSpec;
use zkie_ops::proof::plan::{
    config, PlanBuilder, PlanConfigValue, PlanConstant, PlanLayout, PlanSource, TracePlan,
};
use zkie_ops::proof::proximity_planner::{
    recorded_table_distance, shared_lin_columns, shared_lin_plan_fault, SharedLinPlanFault,
    SharedLinShape,
};
use zkie_ops::proof::table_range::{
    derive_entry_range, EntryRange, EntryRangeLimits, EntryRangePlan,
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
        PlanSource::asset_slice(&pro_out_path, PlanLayout::columns(H_PAD, 0, H_PAD)),
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

/// The registered first projection weight compatible with the square encoded
/// table geometry. The input projection `pro_hid_w` is 64x2048 and the
/// registered key witness requires a square stored matrix, so the first
/// non-alias square projection of the production plan is `pro_out_w`.
pub const FIRST_PROJECTION: &str = "models/timesfm/weights/pro_out_w_i32.bin";

/// The consumer-authenticated assets of this model. Only the registered first
/// projection is admitted; every other asset keeps the raw row commitment.
pub fn consumer_asset_paths() -> &'static [&'static str] {
    &[FIRST_PROJECTION]
}

/// The declared integer interval and matrix geometry of the first encodable
/// weight table. The values of the registered projection fit this width; the
/// declaration is public and does not depend on a run's private extremes.
pub const FIRST_TABLE_RANGE_LO: i64 = -131_072;
pub const FIRST_TABLE_RANGE_HI: i64 = 131_071;
pub const FIRST_TABLE_COLUMN_BITS: u32 = 11;
pub const FIRST_TABLE_MAX_TABLE_LOG: u32 = 19;
pub const FIRST_TABLE_MAX_BIT_ENTRIES: u64 = 1 << 24;

/// The public range plan of the first weight table, derived from the fixed
/// public configuration above.
pub fn production_table_range() -> Result<EntryRangePlan> {
    derive_entry_range(
        H_PAD as u64,
        FIRST_TABLE_COLUMN_BITS,
        EntryRange::new(FIRST_TABLE_RANGE_LO, FIRST_TABLE_RANGE_HI)?,
        EntryRangeLimits {
            max_table_log: FIRST_TABLE_MAX_TABLE_LOG,
            max_bit_entries: FIRST_TABLE_MAX_BIT_ENTRIES,
        },
    )
}

/// The public model-level identity of the encoded table. It covers the model
/// name, version, parameters and asset identities, never the request input, so
/// the same table configuration is shared by every timesfm request.
pub fn table_model_identity() -> Result<Digest> {
    let parameters = parameters_digest()?;
    let mut data = Vec::new();
    data.extend_from_slice(b"timesfm-table-model-v1");
    data.extend_from_slice(&MODEL_VERSION.to_le_bytes());
    data.extend_from_slice(&parameters.0);
    for spec in asset_specs() {
        data.extend_from_slice(&zkie_ops::proof::model_commitment::asset_identity(&spec).0);
    }
    transcript_digest(20, &data)
}

/// The fixed public candidate of the production encoded table. The candidate
/// is chosen offline inside the declared support domain and written here; the
/// derivation below only expands and checks it, never searches.
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

/// The candidate for the first projection's encoded table. The one-time rule
/// derives the distance and columns; one later batch is declared and must fit
/// the same public geometry and the code's random tail.
pub const TABLE_PLAN_CANDIDATE: TablePlanCandidate = TablePlanCandidate {
    log_n: 13,
    ell: 4096,
    r: 128,
    v: 11,
    answers: 1,
    salt_bytes: 40,
    matrix_rows_log: 11,
    k_lo: 0,
    sharing: 1,
    later_batches: 1,
};

/// The deterministically derived table plan of one candidate. The table
/// configuration and plan identity are known before any randomness or tree;
/// the instance root and record are produced by the commit and checked against
/// the trusted meta afterwards.
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

/// Derives and checks the table plan of the fixed first-projection candidate.
/// The one-time distance and columns come from the one-time rule; the later
/// distance and columns come from the same public geometry with the record
/// check removed, capped at the one-time distance. The declared later batch
/// must fit both the derived capacity and the code's random tail; the total
/// declared column use must fit the real hiding positions.
pub fn derive_table_plan(candidate: &TablePlanCandidate) -> Result<DerivedTablePlan> {
    if candidate.later_batches != 1 {
        return Err(Error::InvalidConfig);
    }
    let spec = asset_specs()
        .into_iter()
        .find(|spec| spec.path == FIRST_PROJECTION)
        .ok_or(Error::InvalidConfig)?;
    if spec.row_width != H_PAD as usize
        || spec.element_count != (H_PAD as u64) * (H_PAD as u64)
    {
        return Err(Error::InvalidConfig);
    }
    if candidate.v != 11 || candidate.matrix_rows_log != 11 {
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

/// The trusted table parameters of the registered first projection. Only that
/// path is admitted; every other path keeps the raw row commitment.
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

/// The trusted table parameters of the L0 attention norm gain. The asset is one
/// 2,048-entry row; the encoded table has one row and the code message carries
/// the padded row.
pub fn norm_param_table_params() -> Result<Vec<AssetTableParams>> {
    let path = "models/timesfm/weights/L0_lnw_i32.bin";
    let spec = asset_specs()
        .into_iter()
        .find(|spec| spec.path == path)
        .ok_or(Error::InvalidConfig)?;
    let code = build_code_plan(12, 2051, 4)?;
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
    Ok(vec![table])
}

fn transcript_digest(tag: u64, data: &[u8]) -> Result<Digest> {    zkie_engine::primitives::transcript::hash_labeled(
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

/// The migrated public table paths of this model.
pub fn public_table_paths() -> &'static [&'static str] {
    &[EXP, RSQRT, "models/timesfm/weights/silu_table_i32.bin"]
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
            "models/timesfm/weights/silu_table_i32.bin",
            "107fba3ab21394f02ff1869ca8aaeda98f70f092a96e61bcf646cb53af19d6c7",
            1_048_576,
        ),
    ]
}

/// The production plan: the shared exp/rsqrt and the silu table are trusted
/// public files.
pub fn trace_plan_public(m: usize) -> Result<TracePlan> {
    let mut plan = trace_plan(m)?;
    zkie_ops::proof::plan::promote_public_tables(&mut plan, &public_tables());
    zkie_ops::proof::plan::promote_consumer_assets(&mut plan, consumer_asset_paths())?;
    Ok(plan)
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

    #[test]
    fn first_projection_table_params_are_valid() {
        let range = production_table_range().expect("range plan");
        assert_eq!(range.matrix_rows, H_PAD as u64);
        assert_eq!(range.column_bits, FIRST_TABLE_COLUMN_BITS);
        assert_eq!(range.table_log, 18);
        assert!(range.bit_entries <= FIRST_TABLE_MAX_BIT_ENTRIES);
        let spec = asset_specs()
            .into_iter()
            .find(|spec| spec.path == FIRST_PROJECTION)
            .expect("first projection spec");
        let params = projection_table_params(FIRST_PROJECTION).expect("table params");
        validate_asset_table_params(&spec, &params).unwrap();
        assert_eq!(params.row_width, H_PAD as u64);
        assert_eq!(params.element_count, (H_PAD as u64) * (H_PAD as u64));
        assert_eq!(params.shape.rows(), 1 << 11);
        assert_eq!(params.shape.code.log_n, 13);
        assert_eq!(params.shape.code.ell, 4096);
        assert_eq!(params.shape.code.r, 128);
        assert_eq!(params.shape.answers, 1);
        let derived = derive_table_plan(&TABLE_PLAN_CANDIDATE).unwrap();
        assert_ne!(derived.table_identity, Digest::ZERO);
        assert_ne!(derived.table_configuration, Digest::ZERO);
        assert_eq!(derived.total_columns, derived.one_time_columns + derived.later_batch_columns);
        assert!(derived.total_columns <= params.shape.code.r);
    }

    #[test]
    fn the_production_plan_promotes_the_first_projection() {
        let plan = trace_plan_public(16).unwrap();
        let count = plan
            .steps
            .iter()
            .flat_map(|step| &step.inputs)
            .filter(|source| {
                matches!(source, PlanSource::ConsumerAsset { path, .. } if path == FIRST_PROJECTION)
            })
            .count();
        assert_eq!(count, 1, "the first projection is consumer-authenticated once");
    }

    #[test]
    fn the_first_projection_derives_the_consumer_obligation() {
        use zkie_ops::proof::consumer_psc::{
            ConsumerMaskSource, ConsumerObligation, ConsumerTable,
        };
        let plan = trace_plan_public(16).unwrap();
        let (index, step) = plan
            .steps
            .iter()
            .enumerate()
            .find(|(_, step)| {
                step.inputs.iter().any(|source| {
                    matches!(source, PlanSource::ConsumerAsset { path, .. } if path == FIRST_PROJECTION)
                })
            })
            .expect("consumer step");
        let activation_step = match &step.inputs[0] {
            PlanSource::Prev { step, output, layout }
                if *output == 0 && *layout == PlanLayout::Flat =>
            {
                *step as u32
            }
            other => panic!("unexpected activation source: {other:?}"),
        };
        let table = ConsumerTable {
            path: FIRST_PROJECTION.to_string(),
            oracle: 30,
            root: Digest([1; 32]),
            identity: Digest([2; 32]),
            configuration: Digest([3; 32]),
        };
        let mask = ConsumerMaskSource {
            oracle: 3000,
            hiding_base: 0,
            shown_slot: 1 << 40,
        };
        let obligation =
            ConsumerObligation::from_projection_step(1, index as u32, activation_step, 0, step, table, mask)
                .expect("consumer obligation");
        assert_eq!(obligation.shape.positions, 16);
        assert_eq!(obligation.shape.places, H_PAD as u64);
        assert_eq!(obligation.shape.records, H_PAD as u64);
        assert_eq!(obligation.shape.slice, (0, H_PAD as u64));
        assert_eq!(obligation.shift, SHIFT as i64);
    }
}
