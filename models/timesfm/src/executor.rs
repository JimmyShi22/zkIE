//! TimesFM 16-patch executor over a shared model commitment.
//!
//! The raw 512-point input is standardized with statistics from the first
//! 32-point patch, padded to 64 columns, then the 20-layer encoder and the
//! regression head run as shared relation steps. Weights and LUTs are opened
//! once in `SharedTrace::assets`; relation steps reference the pool.

use std::path::Path;

use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink};
use zkie_ops::add::AddOp;
use zkie_ops::affine_scale::AffineScaleOp;
use zkie_ops::hadamard::HadamardOp;
use zkie_ops::index_gen::TableIndexOp;
use zkie_ops::matmul::MatMulOp;
use zkie_ops::profile_norm::ProfileLayerNormOp;
use zkie_ops::profile_rms::ProfileRmsNormOp;
use zkie_ops::profile_standardize::ProfileStandardizeOp;
use zkie_ops::projection::ProjectionOp;
use zkie_ops::proof::model_commitment::{
    commit_assets, AssetOpening, AssetStore, ModelAssetSpec, ModelCommitment, SharedTrace,
    SharedTraceStep,
};
use zkie_ops::proof::relations::{
    proof_output_values, prove_op_relation_shared, RelationConfig, RelationReceiptMessage,
};
use zkie_ops::registry::PureForward;
use zkie_ops::relu::ReluOp;
use zkie_ops::softmax::SoftmaxOp;
use zkie_ops::table_lookup::TableLookupOp;

const SEQ: usize = 16;
const PATCH: usize = 32;
const HIDDEN: usize = 1280;
const H_PAD: usize = 2048;
const HEADS: usize = 16;
const HEAD_DIM: usize = 80;
const LAYERS: usize = 20;
const QKV_COLS: usize = 4096;
const QKV_BIAS_COLS: usize = 3 * HIDDEN;
const SILU_OFFSET: i64 = 1 << 19;
const SILU_LEN: usize = 1 << 20;
const RAW_POINTS: usize = 512;

const WEIGHT_DIR: &str = "models/timesfm/weights";
const SHARED_DIR: &str = "models/gpt2/weights";
const INPUT_PATH: &str = "cases/timesfm/input_ts_i32.bin";

pub fn asset_specs() -> Vec<ModelAssetSpec> {
    let mut specs = Vec::new();
    let mut push = |id: u64, path: String, count: usize, width: usize, encoding: &str| {
        specs.push(ModelAssetSpec {
            asset_id: id,
            path,
            element_count: count as u64,
            row_width: width,
            encoding: encoding.into(),
        });
    };
    let mut id = 0u64;
    for layer in 0..LAYERS {
        for (suffix, count, width) in [
            ("qkv_w", H_PAD * QKV_COLS, QKV_COLS),
            ("qkv_b", QKV_BIAS_COLS, QKV_BIAS_COLS),
            ("o_proj_w", H_PAD * H_PAD, H_PAD),
            ("o_proj_b", H_PAD, H_PAD),
            ("gate_w", H_PAD * H_PAD, H_PAD),
            ("gate_b", H_PAD, H_PAD),
            ("down_w", H_PAD * H_PAD, H_PAD),
            ("down_b", H_PAD, H_PAD),
            ("lnw", H_PAD, H_PAD),
            ("mlp_w", H_PAD, H_PAD),
            ("mlp_b", H_PAD, H_PAD),
        ] {
            push(
                id,
                format!("{WEIGHT_DIR}/L{layer}_{suffix}_i32.bin"),
                count,
                width,
                "i32-le",
            );
            id += 1;
        }
    }
    for (name, count, width, encoding) in [
        ("embedding_i32.bin", H_PAD, H_PAD, "i32-le"),
        ("gather_i32.bin", SEQ * H_PAD, H_PAD, "i32-le"),
        ("q_scale_i32.bin", 128, 128, "i32-le"),
        ("mask_q_i32.bin", SEQ * SEQ, SEQ, "i32-le"),
        ("scale_i64.bin", 2, 2, "i64-le"),
        ("pro_hid_w_i32.bin", 64 * H_PAD, H_PAD, "i32-le"),
        ("pro_hid_b_i32.bin", H_PAD, H_PAD, "i32-le"),
        ("pro_out_w_i32.bin", H_PAD * H_PAD, H_PAD, "i32-le"),
        ("pro_out_b_i32.bin", H_PAD, H_PAD, "i32-le"),
        ("pro_res_w_i32.bin", 64 * H_PAD, H_PAD, "i32-le"),
        ("pro_res_b_i32.bin", H_PAD, H_PAD, "i32-le"),
        ("head_hid_w_i32.bin", H_PAD * H_PAD, H_PAD, "i32-le"),
        ("head_hid_b_i32.bin", H_PAD, H_PAD, "i32-le"),
        ("head_out_w_i32.bin", H_PAD * H_PAD, H_PAD, "i32-le"),
        ("head_out_b_i32.bin", H_PAD, H_PAD, "i32-le"),
        ("head_res_w_i32.bin", H_PAD * H_PAD, H_PAD, "i32-le"),
        ("head_res_b_i32.bin", H_PAD, H_PAD, "i32-le"),
        ("silu_table_i32.bin", SILU_LEN, SILU_LEN, "i32-le"),
    ] {
        push(id, format!("{WEIGHT_DIR}/{name}"), count, width, encoding);
        id += 1;
    }
    push(
        id,
        format!("{SHARED_DIR}/exp_table_i32.bin"),
        1 << 21,
        1 << 21,
        "i32-le",
    );
    id += 1;
    push(
        id,
        format!("{SHARED_DIR}/rsqrt_table_i32.bin"),
        7_340_032,
        7_340_032,
        "i32-le",
    );
    id += 1;
    push(id, INPUT_PATH.into(), RAW_POINTS, RAW_POINTS, "i32-le");
    specs
}

struct Assets<'a> {
    store: AssetStore<'a>,
    trace_assets: Vec<AssetOpening>,
}

impl<'a> Assets<'a> {
    fn load(commitment: &'a ModelCommitment) -> Result<Self> {
        let store = AssetStore::open(commitment)?;
        let trace_assets = store.open_all()?;
        Ok(Self { store, trace_assets })
    }

    fn value(&self, path: &str) -> Result<Vec<i64>> {
        self.store.value(path)
    }

    fn value_range(&self, path: &str, start: u64, count: usize) -> Result<Vec<i64>> {
        self.store.value_range(path, start, count)
    }

    fn rows(&self, path: &str, cols: usize, r0: usize, r1: usize) -> Result<Vec<i64>> {
        self.store.rows(path, cols, r0, r1)
    }

    fn pool(&self, path: &str) -> Result<usize> {
        self.store.pool(path)
    }
}

fn broadcast(values: &[i64], rows: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(values.len() * rows);
    for _ in 0..rows {
        out.extend_from_slice(values);
    }
    out
}

fn columns(matrix: &[i64], cols: usize, c0: usize, c1: usize) -> Vec<i64> {
    let rows = matrix.len() / cols;
    let mut out = Vec::with_capacity(rows * (c1 - c0));
    for row in 0..rows {
        out.extend_from_slice(&matrix[row * cols + c0..row * cols + c1]);
    }
    out
}

fn transpose(matrix: &[i64], rows: usize, cols: usize) -> Vec<i64> {
    let mut out = vec![0i64; rows * cols];
    for row in 0..rows {
        for column in 0..cols {
            out[column * rows + row] = matrix[row * cols + column];
        }
    }
    out
}

struct Builder<'a> {
    steps: Vec<SharedTraceStep>,
    progress: Option<&'a dyn ProgressSink>,
    planned_total: u64,
    completed: u64,
}

impl<'a> Builder<'a> {
    fn step<T: PureForward + RelationConfig>(
        &mut self,
        op: &T,
        inputs: Vec<(Vec<i64>, Option<usize>)>,
        sources: Vec<Option<(usize, usize)>>,
    ) -> Result<(Vec<Vec<i64>>, usize)> {
        let mut values = Vec::with_capacity(inputs.len());
        let mut external = Vec::with_capacity(inputs.len());
        for (value, pool) in inputs {
            values.push(value);
            external.push(pool);
        }
        let external_indices: Vec<usize> = external
            .iter()
            .enumerate()
            .filter_map(|(index, pool)| pool.map(|_| index))
            .collect();
        let config = op.relation_config(&values)?;
        let (public, proof) = prove_op_relation_shared(op, &values, &config, &external_indices)?;
        let mut roots = Vec::new();
        for id in public.registry.ids() {
            let root = public.registry.get(id).ok_or(Error::ProverStateMissing)?;
            roots.push((root.root, root.start, root.count));
        }
        let outputs = proof_output_values(&proof)?;
        self.steps.push(SharedTraceStep {
            message: RelationReceiptMessage {
                opcode: op.opcode() as u16,
                config,
                roots,
                external_inputs: external_indices,
                proof,
            },
            external,
            sources,
        });
        self.completed = self.completed.saturating_add(1);
        if let Some(progress) = self.progress {
            if self.completed % 64 == 0 || self.completed == self.planned_total {
                let mut event =
                    ProgressEvent::new("prove", "prove", ProgressKind::WorkAdvanced);
                event.completed_units = self.completed;
                event.total_units = Some(self.planned_total);
                event.unit = Some("relation_steps".into());
                let _ = progress.emit(event);
            }
        }
        Ok((outputs, self.steps.len() - 1))
    }
}

fn project(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    weight: Vec<i64>,
    weight_pool: Option<usize>,
    m: usize,
    n: usize,
    k: usize,
    shift: u32,
) -> Result<(Vec<i64>, usize)> {
    let op = ProjectionOp { m, n, k, shift, bias: None };
    let (out, step) = builder.step(
        &op,
        vec![(x, None), (weight, weight_pool)],
        vec![Some((x_step, 0)), None],
    )?;
    Ok((out[0].clone(), step))
}

fn project_inline(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    weight: Vec<i64>,
    weight_pool: Option<usize>,
    m: usize,
    n: usize,
    k: usize,
    shift: u32,
) -> Result<(Vec<i64>, usize)> {
    let op = ProjectionOp { m, n, k, shift, bias: None };
    let (out, step) = builder.step(&op, vec![(x, None), (weight, weight_pool)], vec![None, None])?;
    Ok((out[0].clone(), step))
}

fn add_bias(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    bias: &[i64],
    rows: usize,
) -> Result<(Vec<i64>, usize)> {
    let (out, step) = builder.step(
        &AddOp,
        vec![(x, None), (broadcast(bias, rows), None)],
        vec![Some((x_step, 0)), None],
    )?;
    Ok((out[0].clone(), step))
}

fn rms(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    weight: Vec<i64>,
    base: Vec<i64>,
    base_pool: usize,
    rows: usize,
    cols: usize,
    valid: usize,
) -> Result<(Vec<i64>, usize)> {
    let op = ProfileRmsNormOp {
        rows,
        cols,
        valid_cols: valid,
        output_shift: 32,
        clamp: None,
    };
    let inputs = vec![
        (x, None),
        (broadcast(&weight, rows), None),
        (base, Some(base_pool)),
    ];
    let (outputs, step) = builder.step(&op, inputs, vec![Some((x_step, 0)), None, None])?;
    Ok((outputs[0].clone(), step))
}

fn layer_norm(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    weight: Vec<i64>,
    bias: Vec<i64>,
    base: Vec<i64>,
    base_pool: usize,
    rows: usize,
    cols: usize,
    valid: usize,
) -> Result<(Vec<i64>, usize)> {
    let op = ProfileLayerNormOp {
        rows,
        cols,
        valid_cols: valid,
        output_shift: 32,
        clamp: None,
    };
    let inputs = vec![
        (x, None),
        (broadcast(&weight, rows), None),
        (broadcast(&bias, rows), None),
        (base, Some(base_pool)),
    ];
    let (outputs, step) = builder.step(&op, inputs, vec![Some((x_step, 0)), None, None, None])?;
    Ok((outputs[0].clone(), step))
}

fn silu(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    silu_table: Vec<i64>,
    silu_pool: usize,
) -> Result<(Vec<i64>, usize)> {
    let (idx, idx_step) = builder.step(
        &TableIndexOp { offset: SILU_OFFSET, table_len: SILU_LEN },
        vec![(x, None)],
        vec![Some((x_step, 0))],
    )?;
    let (out, step) = builder.step(
        &TableLookupOp,
        vec![(silu_table, Some(silu_pool)), (idx[0].clone(), None)],
        vec![None, Some((idx_step, 0))],
    )?;
    Ok((out[0].clone(), step))
}

pub fn execute_shared(commitment: &ModelCommitment, input: &[i64]) -> Result<SharedTrace> {
    execute_shared_with_progress(commitment, input, None)
}

pub fn execute_shared_with_progress(
    commitment: &ModelCommitment,
    input: &[i64],
    progress: Option<&dyn ProgressSink>,
) -> Result<SharedTrace> {
    if input.len() != RAW_POINTS {
        return Err(Error::InvalidConfig);
    }
    let planned_total = 6683;
    let assets = Assets::load(commitment)?;
    let mut builder = Builder {
        steps: Vec::new(),
        progress,
        planned_total,
        completed: 0,
    };
    let weight = |name: &str| assets.value(name);
    let pool = |name: &str| assets.pool(name);

    let base_path = format!("{SHARED_DIR}/rsqrt_table_i32.bin");
    let exp_path = format!("{SHARED_DIR}/exp_table_i32.bin");
    let base = weight(&base_path)?;
    let base_pool = pool(&base_path)?;
    let exp_table = weight(&exp_path)?;
    let exp_pool = pool(&exp_path)?;
    let silu_path = format!("{WEIGHT_DIR}/silu_table_i32.bin");
    let silu_table = weight(&silu_path)?;
    let silu_pool = pool(&silu_path)?;

    let (cat, mut x_step) = builder.step(
        &ProfileStandardizeOp {
            rows: SEQ,
            cols: PATCH,
            output_cols: 2 * PATCH,
            valid_cols: PATCH,
            source_row: 0,
            output_shift: 16,
            clamp: None,
        },
        vec![
            (input.to_vec(), Some(pool(INPUT_PATH)?)),
            (base.clone(), Some(base_pool)),
        ],
        vec![None, None],
    )?;
    let mut current = cat[0].clone();

    let pro_hid_path = format!("{WEIGHT_DIR}/pro_hid_w_i32.bin");
    let (hid, hid_step) = project(
        &mut builder, current.clone(), x_step, weight(&pro_hid_path)?, Some(pool(&pro_hid_path)?),
        SEQ, H_PAD, 2 * PATCH, 16,
    )?;
    let (hid, hid_step) = add_bias(
        &mut builder, hid, hid_step, &weight(&format!("{WEIGHT_DIR}/pro_hid_b_i32.bin"))?, SEQ,
    )?;
    let (activation, activation_step) = silu(&mut builder, hid, hid_step, silu_table.clone(), silu_pool)?;
    let pro_out_path = format!("{WEIGHT_DIR}/pro_out_w_i32.bin");
    let (out_proj, out_step) = project(
        &mut builder, activation, activation_step, weight(&pro_out_path)?, Some(pool(&pro_out_path)?),
        SEQ, H_PAD, H_PAD, 16,
    )?;
    let (out_proj, out_step) = add_bias(
        &mut builder, out_proj, out_step, &weight(&format!("{WEIGHT_DIR}/pro_out_b_i32.bin"))?, SEQ,
    )?;
    let pro_res_path = format!("{WEIGHT_DIR}/pro_res_w_i32.bin");
    let (res_proj, res_step) = project(
        &mut builder, current, x_step, weight(&pro_res_path)?, Some(pool(&pro_res_path)?),
        SEQ, H_PAD, 2 * PATCH, 16,
    )?;
    let (res_proj, res_step) = add_bias(
        &mut builder, res_proj, res_step, &weight(&format!("{WEIGHT_DIR}/pro_res_b_i32.bin"))?, SEQ,
    )?;
    let (sum, sum_step) = builder.step(
        &AddOp,
        vec![(out_proj, None), (res_proj, None)],
        vec![Some((out_step, 0)), Some((res_step, 0))],
    )?;
    let gather_path = format!("{WEIGHT_DIR}/gather_i32.bin");
    let (sum, sum_step) = builder.step(
        &AddOp,
        vec![(sum[0].clone(), None), (weight(&gather_path)?, Some(pool(&gather_path)?))],
        vec![Some((sum_step, 0)), None],
    )?;
    let embedding = weight(&format!("{WEIGHT_DIR}/embedding_i32.bin"))?;
    let (x0, x0_step) = builder.step(
        &AddOp,
        vec![(sum[0].clone(), None), (broadcast(&embedding, SEQ), None)],
        vec![Some((sum_step, 0)), None],
    )?;
    current = x0[0].clone();
    x_step = x0_step;

    let mask_path = format!("{WEIGHT_DIR}/mask_q_i32.bin");
    let mask = weight(&mask_path)?;
    let mask_pool = pool(&mask_path)?;
    let q_scale = assets.value_range(&format!("{WEIGHT_DIR}/q_scale_i32.bin"), 0, HEAD_DIM)?;

    for layer in 0..LAYERS {
        let path = |suffix: &str| format!("{WEIGHT_DIR}/L{layer}_{suffix}_i32.bin");
        let (h, h_step) = rms(
            &mut builder, current.clone(), x_step, weight(&path("lnw"))?, base.clone(), base_pool,
            SEQ, H_PAD, HIDDEN,
        )?;
        let (qkv, qkv_step) = project(
            &mut builder, h, h_step, weight(&path("qkv_w"))?, Some(pool(&path("qkv_w"))?),
            SEQ, QKV_COLS, H_PAD, 16,
        )?;
        let mut padded_bias = weight(&path("qkv_b"))?;
        padded_bias.resize(QKV_COLS, 0);
        let (qkv, _qkv_step) = add_bias(&mut builder, qkv, qkv_step, &padded_bias, SEQ)?;

        let mut attention = vec![0i64; SEQ * HIDDEN];
        for head in 0..HEADS {
            let q = columns(&qkv, QKV_COLS, head * HEAD_DIM, (head + 1) * HEAD_DIM);
            let k = columns(
                &qkv,
                QKV_COLS,
                HIDDEN + head * HEAD_DIM,
                HIDDEN + (head + 1) * HEAD_DIM,
            );
            let v = columns(
                &qkv,
                QKV_COLS,
                2 * HIDDEN + head * HEAD_DIM,
                2 * HIDDEN + (head + 1) * HEAD_DIM,
            );
            let (q_scaled, q_scaled_step) = builder.step(
                &HadamardOp { shift: 16 },
                vec![(q, None), (broadcast(&q_scale, SEQ), None)],
                vec![None, None],
            )?;
            let kt = transpose(&k, SEQ, HEAD_DIM);
            let (scores, scores_step) = builder.step(
                &MatMulOp { m: SEQ, n: SEQ, k: HEAD_DIM, shift: 16 },
                vec![(q_scaled[0].clone(), None), (kt, None)],
                vec![Some((q_scaled_step, 0)), None],
            )?;
            let (masked, _masked_step) = builder.step(
                &AddOp,
                vec![(scores[0].clone(), None), (mask.clone(), Some(mask_pool))],
                vec![Some((scores_step, 0)), None],
            )?;
            let mut probs = Vec::with_capacity(SEQ * SEQ);
            for row in 0..SEQ {
                let row_values = masked[0][row * SEQ..(row + 1) * SEQ].to_vec();
                let (row_probs, _) = builder.step(
                    &SoftmaxOp { shift: 16 },
                    vec![(row_values, None), (exp_table.clone(), Some(exp_pool))],
                    vec![None, None],
                )?;
                probs.extend_from_slice(&row_probs[0]);
            }
            let (attn, _attn_step) = builder.step(
                &MatMulOp { m: SEQ, n: HEAD_DIM, k: SEQ, shift: 16 },
                vec![(probs, None), (v, None)],
                vec![None, None],
            )?;
            for row in 0..SEQ {
                for dimension in 0..HEAD_DIM {
                    attention[row * HIDDEN + head * HEAD_DIM + dimension] =
                        attn[0][row * HEAD_DIM + dimension];
                }
            }
        }
        let ow = assets.rows(&path("o_proj_w"), H_PAD, 0, HIDDEN)?;
        let (o, o_step) = project_inline(
            &mut builder, attention, ow, None, SEQ, H_PAD, HIDDEN, 16,
        )?;
        let (o, o_step) = add_bias(
            &mut builder, o, o_step, &weight(&path("o_proj_b"))?, SEQ,
        )?;
        let (x2, x2_step) = builder.step(
            &AddOp,
            vec![(current, None), (o, None)],
            vec![Some((x_step, 0)), Some((o_step, 0))],
        )?;
        let x2 = x2[0].clone();
        let (mlp, mlp_step) = layer_norm(
            &mut builder, x2.clone(), x2_step, weight(&path("mlp_w"))?, weight(&path("mlp_b"))?,
            base.clone(), base_pool, SEQ, H_PAD, HIDDEN,
        )?;
        let (gate, gate_step) = project(
            &mut builder, mlp, mlp_step, weight(&path("gate_w"))?, Some(pool(&path("gate_w"))?),
            SEQ, H_PAD, H_PAD, 16,
        )?;
        let (gate, gate_step) = add_bias(
            &mut builder, gate, gate_step, &weight(&path("gate_b"))?, SEQ,
        )?;
        let (gate, gate_step) = builder.step(
            &ReluOp,
            vec![(gate, None)],
            vec![Some((gate_step, 0))],
        )?;
        let (down, down_step) = project(
            &mut builder, gate[0].clone(), gate_step, weight(&path("down_w"))?, Some(pool(&path("down_w"))?),
            SEQ, H_PAD, H_PAD, 16,
        )?;
        let (down, down_step) = add_bias(
            &mut builder, down, down_step, &weight(&path("down_b"))?, SEQ,
        )?;
        let (next, next_step) = builder.step(
            &AddOp,
            vec![(x2, None), (down, None)],
            vec![Some((x2_step, 0)), Some((down_step, 0))],
        )?;
        current = next[0].clone();
        x_step = next_step;
    }

    let head_hid_path = format!("{WEIGHT_DIR}/head_hid_w_i32.bin");
    let (hh, hh_step) = project(
        &mut builder, current.clone(), x_step, weight(&head_hid_path)?, Some(pool(&head_hid_path)?),
        SEQ, H_PAD, H_PAD, 16,
    )?;
    let (hh, hh_step) = add_bias(
        &mut builder, hh, hh_step, &weight(&format!("{WEIGHT_DIR}/head_hid_b_i32.bin"))?, SEQ,
    )?;
    let (hs, hs_step) = silu(&mut builder, hh, hh_step, silu_table.clone(), silu_pool)?;
    let head_out_path = format!("{WEIGHT_DIR}/head_out_w_i32.bin");
    let (ho, ho_step) = project(
        &mut builder, hs, hs_step, weight(&head_out_path)?, Some(pool(&head_out_path)?),
        SEQ, H_PAD, H_PAD, 16,
    )?;
    let (ho, ho_step) = add_bias(
        &mut builder, ho, ho_step, &weight(&format!("{WEIGHT_DIR}/head_out_b_i32.bin"))?, SEQ,
    )?;
    let head_res_path = format!("{WEIGHT_DIR}/head_res_w_i32.bin");
    let (hr, hr_step) = project(
        &mut builder, current, x_step, weight(&head_res_path)?, Some(pool(&head_res_path)?),
        SEQ, H_PAD, H_PAD, 16,
    )?;
    let (hr, hr_step) = add_bias(
        &mut builder, hr, hr_step, &weight(&format!("{WEIGHT_DIR}/head_res_b_i32.bin"))?, SEQ,
    )?;
    let (add, add_step) = builder.step(
        &AddOp,
        vec![(ho, None), (hr, None)],
        vec![Some((ho_step, 0)), Some((hr_step, 0))],
    )?;
    let scale = weight(&format!("{WEIGHT_DIR}/scale_i64.bin"))?;
    let (scaled, scaled_step) = builder.step(
        &AffineScaleOp { multiplier: scale[0] as i128, addend: 0, shift: 16, clamp: None },
        vec![(add[0].clone(), None)],
        vec![Some((add_step, 0))],
    )?;
    let (out, _out_step) = builder.step(
        &AddOp,
        vec![(scaled[0].clone(), None), (broadcast(&scale[1..2], SEQ * H_PAD), None)],
        vec![Some((scaled_step, 0)), None],
    )?;
    let output = out[0].clone();
    let Builder { steps, .. } = builder;
    if let Some(progress) = progress {
        let actual = steps.len() as u64;
        let mut event = ProgressEvent::new("prove", "prove", ProgressKind::WorkAdvanced);
        event.completed_units = actual;
        event.total_units = Some(actual);
        event.unit = Some("relation_steps".into());
        let _ = progress.emit(event);
    }
    Ok(SharedTrace {
        tokens: input.iter().map(|value| *value as u32).collect(),
        assets: assets.trace_assets,
        steps,
        argmax: Vec::new(),
        output,
    })
}

pub fn build_commitment(asset_root: &Path, input: &[i64]) -> Result<ModelCommitment> {
    let parameters = zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            14,
        ),
        0,
        None,
        b"timesfm-16patch",
    )?;
    let bytes: Vec<u8> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
    let model = zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            15,
        ),
        0,
        None,
        &bytes,
    )?;
    commit_assets(asset_root, model, parameters, &asset_specs())
}
