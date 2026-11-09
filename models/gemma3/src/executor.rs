//! Gemma3 seq16/seq512 executor over a shared model commitment.
//!
//! Weights, RoPE tables, the shared rsqrt/exp/gelu tables and the rsqrt
//! extension are committed once in the model commitment; every relation step
//! marks the assets it reads as external and references the pool instead of
//! committing them again inside the proof.

use std::path::Path;

use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink};
use zkie_ops::add::AddOp;
use zkie_ops::affine_scale::AffineScaleOp;
use zkie_ops::argmax::ArgmaxOp;
use zkie_ops::gated_ffn::GatedFfnOp;
use zkie_ops::index_gen::{RowEmbeddingIndexOp, TableIndexOp};
use zkie_ops::matmul::MatMulOp;
use zkie_ops::profile_rms::ProfileRmsNormOp;
use zkie_ops::profile_rope::ProfileRopeOp;
use zkie_ops::projection::ProjectionOp;
use zkie_ops::proof::model_commitment::{
    commit_assets, AssetOpening, AssetStore, ModelAssetSpec, ModelCommitment, SharedTrace,
    SharedTraceStep,
};
use zkie_ops::proof::relations::{
    proof_output_values, prove_op_relation_shared, RelationConfig, RelationReceiptMessage,
};
use zkie_ops::registry::PureForward;
use zkie_ops::softmax::SoftmaxOp;
use zkie_ops::softmax_rows::SoftmaxRowsOp;
use zkie_ops::table_lookup::TableLookupOp;

const PAD: usize = 1024;
const REAL: usize = 640;
const HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const FFN: usize = 2048;
const VOCAB: usize = 262144;
const LAYERS: usize = 18;
const MAX_SEQUENCE: usize = 512;
const FULL_LAYERS: [usize; 3] = [5, 11, 17];

const WEIGHT_DIR: &str = "models/gemma3/weights";
const SHARED_DIR: &str = "models/gpt2/weights";
const EXTENSION_PATH: &str = "cases/gemma3/rsqrt_ext_i32.bin";
const EXTENSION_ENTRIES: usize = 14_849_263;
const RSQRT_ENTRIES: usize = 7_340_032;

pub fn asset_specs() -> Vec<ModelAssetSpec> {
    let mut specs = Vec::new();
    let mut push = |id: u64, path: String, count: usize, width: usize| {
        specs.push(ModelAssetSpec {
            asset_id: id,
            path,
            element_count: count as u64,
            row_width: width,
            encoding: "i32-le".into(),
        });
    };
    let mut id = 0u64;
    for layer in 0..LAYERS {
        for (suffix, count, width) in [
            ("q_w", PAD * PAD, PAD),
            ("k_w", PAD * HEAD_DIM, HEAD_DIM),
            ("v_w", PAD * HEAD_DIM, HEAD_DIM),
            ("o_w", PAD * PAD, PAD),
            ("gate_w", PAD * FFN, FFN),
            ("up_w", PAD * FFN, FFN),
            ("down_w", FFN * PAD, PAD),
            ("in_norm", PAD, PAD),
            ("post_attn_norm", PAD, PAD),
            ("pre_ffn_norm", PAD, PAD),
            ("post_ffn_norm", PAD, PAD),
            ("q_norm", HEAD_DIM, HEAD_DIM),
            ("k_norm", HEAD_DIM, HEAD_DIM),
        ] {
            push(
                id,
                format!("{WEIGHT_DIR}/L{layer}_{suffix}_i32.bin"),
                count,
                width,
            );
            id += 1;
        }
    }
    for (name, count, width) in [
        ("embed_tokens_i32.bin", VOCAB * PAD, PAD),
        ("lm_head_i32.bin", PAD * VOCAB, VOCAB),
        ("final_norm_i32.bin", PAD, PAD),
        ("rope_cos_global_i32.bin", MAX_SEQUENCE * (HEAD_DIM / 2), HEAD_DIM / 2),
        ("rope_sin_global_i32.bin", MAX_SEQUENCE * (HEAD_DIM / 2), HEAD_DIM / 2),
        ("rope_cos_local_i32.bin", MAX_SEQUENCE * (HEAD_DIM / 2), HEAD_DIM / 2),
        ("rope_sin_local_i32.bin", MAX_SEQUENCE * (HEAD_DIM / 2), HEAD_DIM / 2),
    ] {
        push(id, format!("{WEIGHT_DIR}/{name}"), count, width);
        id += 1;
    }
    for (path, count) in [
        (format!("{SHARED_DIR}/exp_table_i32.bin"), 1usize << 21),
        (format!("{SHARED_DIR}/rsqrt_table_i32.bin"), RSQRT_ENTRIES),
        (format!("{SHARED_DIR}/gelu_table_i32.bin"), 1usize << 24),
        (EXTENSION_PATH.into(), EXTENSION_ENTRIES),
    ] {
        push(id, path, count, count);
        id += 1;
    }
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

    fn columns(&self, path: &str, cols: usize, c0: usize, c1: usize) -> Result<Vec<i64>> {
        self.store.columns(path, cols, c0, c1)
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

#[allow(clippy::too_many_arguments)]
fn rms(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    weight: Vec<i64>,
    base: Vec<i64>,
    base_pool: usize,
    extension: Vec<i64>,
    extension_pool: usize,
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
        (extension, Some(extension_pool)),
    ];
    let (outputs, step) = builder.step(&op, inputs, vec![Some((x_step, 0)), None, None, None])?;
    Ok((outputs[0].clone(), step))
}

fn rope(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    cos: Vec<i64>,
    sin: Vec<i64>,
    m: usize,
) -> Result<(Vec<i64>, usize)> {
    let op = ProfileRopeOp { head_dim: HEAD_DIM, shift: 16 };
    let (out, step) = builder.step(
        &op,
        vec![(x, None), (cos, None), (sin, None)],
        vec![Some((x_step, 0)), None, None],
    )?;
    Ok((out[0].clone(), step))
}

pub fn execute_shared(commitment: &ModelCommitment, tokens: &[i64]) -> Result<SharedTrace> {
    execute_shared_with_progress(commitment, tokens, None)
}

pub fn execute_shared_with_progress(
    commitment: &ModelCommitment,
    tokens: &[i64],
    progress: Option<&dyn ProgressSink>,
) -> Result<SharedTrace> {
    if tokens.is_empty() || tokens.len() > 512 {
        return Err(Error::InvalidConfig);
    }
    let m = tokens.len();
    let planned_total = if m > 16 { 1506 } else { 2090 };
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
    let gelu_path = format!("{SHARED_DIR}/gelu_table_i32.bin");
    let base = weight(&base_path)?;
    let base_pool = pool(&base_path)?;
    let extension = weight(EXTENSION_PATH)?;
    let extension_pool = pool(EXTENSION_PATH)?;
    let exp_table = weight(&exp_path)?;
    let exp_pool = pool(&exp_path)?;

    let embed_path = format!("{WEIGHT_DIR}/embed_tokens_i32.bin");
    let (embed_idx, embed_idx_step) = builder.step(
        &RowEmbeddingIndexOp { hidden: PAD, vocab: VOCAB },
        vec![(tokens.to_vec(), None)],
        vec![None],
    )?;
    let (embedded, embed_step) = builder.step(
        &TableLookupOp,
        vec![
            (weight(&embed_path)?, Some(pool(&embed_path)?)),
            (embed_idx[0].clone(), None),
        ],
        vec![None, Some((embed_idx_step, 0))],
    )?;
    let mut current = embedded[0].clone();
    let mut x_step = embed_step;
    let mask = {
        let mut values = vec![0i64; m * m];
        for row in 0..m {
            for column in row + 1..m {
                values[row * m + column] = -(1i64 << 30);
            }
        }
        values
    };

    for layer in 0..LAYERS {
        let path = |suffix: &str| format!("{WEIGHT_DIR}/L{layer}_{suffix}_i32.bin");
        let k_w = weight(&path("k_w"))?;
        let v_w = weight(&path("v_w"))?;
        let gate_w = weight(&path("gate_w"))?;
        let up_w = weight(&path("up_w"))?;
        let down_w = weight(&path("down_w"))?;
        let in_norm = weight(&path("in_norm"))?;
        let post_attn_norm = weight(&path("post_attn_norm"))?;
        let pre_ffn_norm = weight(&path("pre_ffn_norm"))?;
        let post_ffn_norm = weight(&path("post_ffn_norm"))?;
        let q_norm = weight(&path("q_norm"))?;
        let k_norm = weight(&path("k_norm"))?;
        let cos_name = if FULL_LAYERS.contains(&layer) {
            format!("{WEIGHT_DIR}/rope_cos_global_i32.bin")
        } else {
            format!("{WEIGHT_DIR}/rope_cos_local_i32.bin")
        };
        let sin_name = if FULL_LAYERS.contains(&layer) {
            format!("{WEIGHT_DIR}/rope_sin_global_i32.bin")
        } else {
            format!("{WEIGHT_DIR}/rope_sin_local_i32.bin")
        };
        let cos_prefix = assets.value_range(&cos_name, 0, m * (HEAD_DIM / 2))?;
        let sin_prefix = assets.value_range(&sin_name, 0, m * (HEAD_DIM / 2))?;

        let (h, h_step) = rms(
            &mut builder, current.clone(), x_step, in_norm,
            base.clone(), base_pool, extension.clone(), extension_pool,
            m, PAD, REAL,
        )?;
        let (k, k_step) = project(
            &mut builder, h.clone(), h_step, k_w, Some(pool(&path("k_w"))?), m, HEAD_DIM, PAD, 16,
        )?;
        let (v, v_step) = project(
            &mut builder, h.clone(), h_step, v_w, Some(pool(&path("v_w"))?), m, HEAD_DIM, PAD, 16,
        )?;
        let (k, k_step) = rms(
            &mut builder, k, k_step, k_norm,
            base.clone(), base_pool, extension.clone(), extension_pool,
            m, HEAD_DIM, HEAD_DIM,
        )?;
        let (k_rot, k_rot_step) = rope(&mut builder, k, k_step, cos_prefix.clone(), sin_prefix.clone(), m)?;

        let mut head_acc: Option<(Vec<i64>, usize)> = None;
        for head in 0..HEADS {
            let qw = assets.columns(&path("q_w"), PAD, head * HEAD_DIM, (head + 1) * HEAD_DIM)?;
            let (q, q_step) = project(&mut builder, h.clone(), h_step, qw, None, m, HEAD_DIM, PAD, 16)?;
            let (q, q_step) = rms(
                &mut builder, q, q_step, q_norm.clone(),
                base.clone(), base_pool, extension.clone(), extension_pool,
                m, HEAD_DIM, HEAD_DIM,
            )?;
            let (q_rot, q_rot_step) =
                rope(&mut builder, q, q_step, cos_prefix.clone(), sin_prefix.clone(), m)?;
            let (q_scaled, q_scaled_step) = builder.step(
                &AffineScaleOp { multiplier: 4096, addend: 0, shift: 16, clamp: None },
                vec![(q_rot, None)],
                vec![Some((q_rot_step, 0))],
            )?;
            let kt = transpose(&k_rot, m, HEAD_DIM);
            let (scores, scores_step) = builder.step(
                &MatMulOp { m, n: m, k: HEAD_DIM, shift: 16 },
                vec![(q_scaled[0].clone(), None), (kt, None)],
                vec![Some((q_scaled_step, 0)), None],
            )?;
            let (masked, masked_step) = builder.step(
                &AddOp,
                vec![(scores[0].clone(), None), (mask.clone(), None)],
                vec![Some((scores_step, 0)), None],
            )?;
            let (probs, probs_step) = if m > 16 {
                let (out, step) = builder.step(
                    &SoftmaxRowsOp { rows: m, cols: m, shift: 16 },
                    vec![
                        (masked[0].clone(), None),
                        (exp_table.clone(), Some(exp_pool)),
                    ],
                    vec![Some((masked_step, 0)), None],
                )?;
                (out[0].clone(), Some((step, 0)))
            } else {
                let mut probs = Vec::with_capacity(m * m);
                for row in 0..m {
                    let row_values = masked[0][row * m..(row + 1) * m].to_vec();
                    let (row_probs, _) = builder.step(
                        &SoftmaxOp { shift: 16 },
                        vec![(row_values, None), (exp_table.clone(), Some(exp_pool))],
                        vec![None, None],
                    )?;
                    probs.extend_from_slice(&row_probs[0]);
                }
                (probs, None)
            };
            let (attn, attn_step) = builder.step(
                &MatMulOp { m, n: HEAD_DIM, k: m, shift: 16 },
                vec![(probs, None), (v.clone(), None)],
                vec![probs_step, Some((v_step, 0))],
            )?;
            let ow = assets.rows(&path("o_w"), PAD, head * HEAD_DIM, (head + 1) * HEAD_DIM)?;
            let (head_out, head_step) = project(
                &mut builder, attn[0].clone(), attn_step, ow, None, m, PAD, HEAD_DIM, 16,
            )?;
            head_acc = Some(match head_acc {
                None => (head_out, head_step),
                Some((acc, acc_step)) => {
                    let (sum, step) = builder.step(
                        &AddOp,
                        vec![(acc, None), (head_out, None)],
                        vec![Some((acc_step, 0)), Some((head_step, 0))],
                    )?;
                    (sum[0].clone(), step)
                }
            });
        }
        let (head_acc, head_step) = head_acc.ok_or(Error::IncompleteArtifact)?;
        let (attn_norm, attn_norm_step) = rms(
            &mut builder, head_acc, head_step, post_attn_norm,
            base.clone(), base_pool, extension.clone(), extension_pool,
            m, PAD, REAL,
        )?;
        let (x2, x2_step) = builder.step(
            &AddOp,
            vec![(current, None), (attn_norm, None)],
            vec![Some((x_step, 0)), Some((attn_norm_step, 0))],
        )?;
        let x2 = x2[0].clone();
        let (h2, h2_step) = rms(
            &mut builder, x2.clone(), x2_step, pre_ffn_norm,
            base.clone(), base_pool, extension.clone(), extension_pool,
            m, PAD, REAL,
        )?;
        let (gate, gate_step) = project(
            &mut builder, h2.clone(), h2_step, gate_w, Some(pool(&path("gate_w"))?), m, FFN, PAD, 16,
        )?;
        let (up, up_step) = project(
            &mut builder, h2, h2_step, up_w, Some(pool(&path("up_w"))?), m, FFN, PAD, 16,
        )?;
        let (idx, idx_step) = builder.step(
            &TableIndexOp { offset: 1 << 23, table_len: 1 << 24 },
            vec![(gate, None)],
            vec![Some((gate_step, 0))],
        )?;
        let (gelu, gelu_step) = builder.step(
            &TableLookupOp,
            vec![
                (weight(&gelu_path)?, Some(pool(&gelu_path)?)),
                (idx[0].clone(), None),
            ],
            vec![None, Some((idx_step, 0))],
        )?;
        let (act, act_step) = builder.step(
            &GatedFfnOp { shift: 16 },
            vec![(gelu[0].clone(), None), (up, None)],
            vec![Some((gelu_step, 0)), Some((up_step, 0))],
        )?;
        let (down, down_step) = project(
            &mut builder, act[0].clone(), act_step, down_w, Some(pool(&path("down_w"))?), m, PAD, FFN, 16,
        )?;
        let (down_norm, down_norm_step) = rms(
            &mut builder, down, down_step, post_ffn_norm,
            base.clone(), base_pool, extension.clone(), extension_pool,
            m, PAD, REAL,
        )?;
        let (next, next_step) = builder.step(
            &AddOp,
            vec![(x2, None), (down_norm, None)],
            vec![Some((x2_step, 0)), Some((down_norm_step, 0))],
        )?;
        current = next[0].clone();
        x_step = next_step;
    }

    let (hf, hf_step) = rms(
        &mut builder, current, x_step, weight(&format!("{WEIGHT_DIR}/final_norm_i32.bin"))?,
        base, base_pool, extension, extension_pool,
        m, PAD, REAL,
    )?;
    let lm_path = format!("{WEIGHT_DIR}/lm_head_i32.bin");
    let (logits, _logits_step) = builder.step(
        &MatMulOp { m, n: VOCAB, k: PAD, shift: 0 },
        vec![(hf, None), (weight(&lm_path)?, Some(pool(&lm_path)?))],
        vec![Some((hf_step, 0)), None],
    )?;
    let logits = logits[0].clone();
    let mut argmax = Vec::with_capacity(m);
    for row in 0..m {
        let values = logits[row * VOCAB..(row + 1) * VOCAB].to_vec();
        let (out, _step) = builder.step(&ArgmaxOp, vec![(values, None)], vec![None])?;
        argmax.push(out[0][0] as u32);
    }
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
        tokens: tokens.iter().map(|value| *value as u32).collect(),
        assets: assets.trace_assets,
        steps,
        argmax,
        output: Vec::new(),
    })
}

pub fn build_commitment(asset_root: &Path, tokens: &[u32]) -> Result<ModelCommitment> {
    let parameters = zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            14,
        ),
        0,
        None,
        b"gemma3-seq16",
    )?;
    let model = zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            15,
        ),
        0,
        None,
        &tokens.iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    commit_assets(asset_root, model, parameters, &asset_specs())
}
