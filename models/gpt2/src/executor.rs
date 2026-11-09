//! GPT-2 seq16/seq512 executor over a shared model commitment.
//!
//! Weights and LUTs are opened once in `SharedTrace::assets`; every relation
//! step marks those inputs as external and references the pool instead of
//! committing the asset again.

use std::path::Path;

use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink};
use zkie_ops::add::AddOp;
use zkie_ops::affine_scale::AffineScaleOp;
use zkie_ops::argmax::ArgmaxOp;
use zkie_ops::index_gen::{EmbeddingIndexOp, TableIndexOp};
use zkie_ops::matmul::MatMulOp;
use zkie_ops::profile_norm::ProfileLayerNormOp;
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

const HIDDEN: usize = 1024;
const REAL: usize = 768;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const FFN: usize = 4096;
const VOCAB: usize = 65536;
const LAYERS: usize = 12;

pub fn asset_specs() -> Vec<ModelAssetSpec> {
    let mut specs = Vec::new();
    let mut push = |id: u64, path: String, count: u64, width: usize| {
        specs.push(ModelAssetSpec {
            asset_id: id,
            path,
            element_count: count,
            row_width: width,
            encoding: "i32-le".into(),
        });
    };
    let mut id = 0u64;
    for layer in 0..LAYERS {
        for (suffix, count, width) in [
            ("q_w", HIDDEN * HIDDEN, HIDDEN),
            ("q_b", HIDDEN, HIDDEN),
            ("k_w", HIDDEN * HIDDEN, HIDDEN),
            ("k_b", HIDDEN, HIDDEN),
            ("v_w", HIDDEN * HIDDEN, HIDDEN),
            ("v_b", HIDDEN, HIDDEN),
            ("o_proj_w", HIDDEN * HIDDEN, HIDDEN),
            ("o_proj_b", HIDDEN, HIDDEN),
            ("ln1_w", HIDDEN, HIDDEN),
            ("ln1_b", HIDDEN, HIDDEN),
            ("ln2_w", HIDDEN, HIDDEN),
            ("ln2_b", HIDDEN, HIDDEN),
            ("fc_w", HIDDEN * FFN, FFN),
            ("fc_b", FFN, FFN),
            ("proj_w", FFN * HIDDEN, HIDDEN),
            ("proj_b", HIDDEN, HIDDEN),
        ] {
            push(
                id,
                format!("gpt2/weights/L{layer}_{suffix}_i32.bin"),
                count as u64,
                width,
            );
            id += 1;
        }
    }
    for (name, count, width) in [
        ("lm_head_w_i32.bin", HIDDEN * VOCAB, VOCAB),
        ("pos_emb_i32.bin", 512 * HIDDEN, HIDDEN),
        ("ln_f_w_i32.bin", HIDDEN, HIDDEN),
        ("ln_f_b_i32.bin", HIDDEN, HIDDEN),
        ("exp_table_i32.bin", 1 << 21, 1 << 21),
        ("rsqrt_table_i32.bin", 7_340_032, 7_340_032),
        ("gelu_table_i32.bin", 1 << 24, 1 << 24),
    ] {
        push(id, format!("gpt2/weights/{name}"), count as u64, width);
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
    bias: Vec<i64>,
    m: usize,
    n: usize,
    k: usize,
    shift: u32,
) -> Result<(Vec<i64>, usize)> {
    let op = ProjectionOp { m, n, k, shift, bias: None };
    let (out, step) = builder.step(
        &op,
        vec![(x, None), (weight, None)],
        vec![Some((x_step, 0)), None],
    )?;
    let (out, step) = builder.step(
        &AddOp,
        vec![(out[0].clone(), None), (bias, None)],
        vec![Some((step, 0)), None],
    )?;
    Ok((out[0].clone(), step))
}

#[allow(clippy::too_many_arguments)]
fn ln(
    builder: &mut Builder<'_>,
    x: Vec<i64>,
    x_step: usize,
    weight: Vec<i64>,
    bias: Vec<i64>,
    rsqrt: Vec<i64>,
    rsqrt_pool: Option<usize>,
    m: usize,
) -> Result<(Vec<i64>, usize)> {
    let op = ProfileLayerNormOp {
        rows: m,
        cols: HIDDEN,
        valid_cols: REAL,
        output_shift: 32,
        clamp: None,
    };
    let inputs = vec![
        (x, None),
        (broadcast(&weight, m), None),
        (broadcast(&bias, m), None),
        (rsqrt, rsqrt_pool),
    ];
    let (outputs, step) = builder.step(&op, inputs, vec![Some((x_step, 0)), None, None, None])?;
    Ok((outputs[0].clone(), step))
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
    let planned_total = if m > 16 { 2798 } else { 4462 };
    let assets = Assets::load(commitment)?;
    let mut builder = Builder {
        steps: Vec::new(),
        progress,
        planned_total,
        completed: 0,
    };
    let weight = |name: &str| assets.value(name);
    let pool = |name: &str| assets.pool(name);

    let (embed_idx, embed_idx_step) = builder.step(
        &EmbeddingIndexOp { hidden: HIDDEN, vocab: VOCAB },
        vec![(tokens.to_vec(), None)],
        vec![None],
    )?;
    let embed_idx = embed_idx[0].clone();
    let (embedded, embed_step) = builder.step(
        &TableLookupOp,
        vec![
            (assets.value("gpt2/weights/lm_head_w_i32.bin")?, Some(pool("gpt2/weights/lm_head_w_i32.bin")?)),
            (embed_idx, None),
        ],
        vec![None, Some((embed_idx_step, 0))],
    )?;
    let embedded = embedded[0].clone();
    let pos = assets.value_range("gpt2/weights/pos_emb_i32.bin", 0, m * HIDDEN)?;
    let (x, mut x_step) = builder.step(
        &AddOp,
        vec![(embedded, None), (pos, None)],
        vec![Some((embed_step, 0)), None],
    )?;
    let mut current = x[0].clone();
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
        let path = |suffix: &str| format!("gpt2/weights/L{layer}_{suffix}_i32.bin");
        let q_b = weight(&path("q_b"))?;
        let k_b = weight(&path("k_b"))?;
        let v_b = weight(&path("v_b"))?;
        let o_b = weight(&path("o_proj_b"))?;
        let ln1_w = weight(&path("ln1_w"))?;
        let ln1_b = weight(&path("ln1_b"))?;
        let ln2_w = weight(&path("ln2_w"))?;
        let ln2_b = weight(&path("ln2_b"))?;
        let fc_w = weight(&path("fc_w"))?;
        let fc_b = weight(&path("fc_b"))?;
        let proj_w = weight(&path("proj_w"))?;
        let proj_b = weight(&path("proj_b"))?;
        let rsqrt = weight("gpt2/weights/rsqrt_table_i32.bin")?;
        let exp_table = weight("gpt2/weights/exp_table_i32.bin")?;
        let gelu_table = weight("gpt2/weights/gelu_table_i32.bin")?;

        let (h, h_step) = ln(&mut builder, current.clone(), x_step, ln1_w, ln1_b, rsqrt.clone(), Some(pool("gpt2/weights/rsqrt_table_i32.bin")?), m)?;
        let mut head_acc: Option<(Vec<i64>, usize)> = None;
        for head in 0..HEADS {
            let qw = assets.columns(&path("q_w"), HIDDEN, head * HEAD_DIM, (head + 1) * HEAD_DIM)?;
            let kw = assets.columns(&path("k_w"), HIDDEN, head * HEAD_DIM, (head + 1) * HEAD_DIM)?;
            let vw = assets.columns(&path("v_w"), HIDDEN, head * HEAD_DIM, (head + 1) * HEAD_DIM)?;
            let qb = broadcast(&q_b[head * HEAD_DIM..(head + 1) * HEAD_DIM], m);
            let kb = broadcast(&k_b[head * HEAD_DIM..(head + 1) * HEAD_DIM], m);
            let vb = broadcast(&v_b[head * HEAD_DIM..(head + 1) * HEAD_DIM], m);
            let (q, q_step) = project(&mut builder, h.clone(), h_step, qw, qb, m, HEAD_DIM, HIDDEN, 16)?;
            let (q, q_step) = {
                let op = AffineScaleOp { multiplier: 23170, addend: 0, shift: 16, clamp: None };
                let (out, step) = builder.step(&op, vec![(q, None)], vec![Some((q_step, 0))])?;
                (out[0].clone(), step)
            };
            let (k, k_step) = project(&mut builder, h.clone(), h_step, kw, kb, m, HEAD_DIM, HIDDEN, 16)?;
            let (k, k_step) = {
                let op = AffineScaleOp { multiplier: 23170, addend: 0, shift: 16, clamp: None };
                let (out, step) = builder.step(&op, vec![(k, None)], vec![Some((k_step, 0))])?;
                (out[0].clone(), step)
            };
            let (v, v_step) = project(&mut builder, h.clone(), h_step, vw, vb, m, HEAD_DIM, HIDDEN, 16)?;
            let kt = transpose(&k, m, HEAD_DIM);
            let (scores, scores_step) = builder.step(
                &MatMulOp { m, n: m, k: HEAD_DIM, shift: 16 },
                vec![(q, None), (kt, None)],
                vec![Some((q_step, 0)), None],
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
                        (exp_table.clone(), Some(pool("gpt2/weights/exp_table_i32.bin")?)),
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
                        vec![(row_values, None), (exp_table.clone(), Some(pool("gpt2/weights/exp_table_i32.bin")?))],
                        vec![None, None],
                    )?;
                    probs.extend_from_slice(&row_probs[0]);
                }
                (probs, None)
            };
            let (attn, attn_step) = builder.step(
                &MatMulOp { m, n: HEAD_DIM, k: m, shift: 16 },
                vec![(probs, None), (v, None)],
                vec![probs_step, Some((v_step, 0))],
            )?;
            let ow = assets.rows(&path("o_proj_w"), HIDDEN, head * HEAD_DIM, (head + 1) * HEAD_DIM)?;
            let (head_out, head_step) = project(
                &mut builder, attn[0].clone(), attn_step, ow, vec![0; m * HIDDEN], m, HIDDEN, HEAD_DIM, 16,
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
        let (attn_biased, attn_step) = builder.step(
            &AddOp,
            vec![(head_acc, None), (broadcast(&o_b, m), None)],
            vec![Some((head_step, 0)), None],
        )?;
        let (x2, x2_step) = builder.step(
            &AddOp,
            vec![(current, None), (attn_biased[0].clone(), None)],
            vec![Some((x_step, 0)), Some((attn_step, 0))],
        )?;
        let x2 = x2[0].clone();
        let (h2, h2_step) = ln(&mut builder, x2.clone(), x2_step, ln2_w, ln2_b, rsqrt.clone(), Some(pool("gpt2/weights/rsqrt_table_i32.bin")?), m)?;
        let (fc, fc_step) = project(&mut builder, h2, h2_step, fc_w, broadcast(&fc_b, m), m, FFN, HIDDEN, 16)?;
        let (idx, idx_step) = builder.step(
            &TableIndexOp { offset: 1 << 23, table_len: 1 << 24 },
            vec![(fc.clone(), None)],
            vec![Some((fc_step, 0))],
        )?;
        let act = {
            let (out, step) = builder.step(
                &TableLookupOp,
                vec![(gelu_table, Some(pool("gpt2/weights/gelu_table_i32.bin")?)), (idx[0].clone(), None)],
                vec![None, Some((idx_step, 0))],
            )?;
            (out[0].clone(), step)
        };
        let (proj, proj_step) = project(&mut builder, act.0, act.1, proj_w, broadcast(&proj_b, m), m, HIDDEN, FFN, 16)?;
        let (next, next_step) = builder.step(
            &AddOp,
            vec![(x2, None), (proj.clone(), None)],
            vec![Some((x2_step, 0)), Some((proj_step, 0))],
        )?;
        current = next[0].clone();
        x_step = next_step;
    }

    let (hf, hf_step) = ln(
        &mut builder,
        current,
        x_step,
        weight("gpt2/weights/ln_f_w_i32.bin")?,
        weight("gpt2/weights/ln_f_b_i32.bin")?,
        weight("gpt2/weights/rsqrt_table_i32.bin")?,
        Some(pool("gpt2/weights/rsqrt_table_i32.bin")?),
        m,
    )?;
    let (logits, logits_step) = builder.step(
        &MatMulOp { m, n: VOCAB, k: HIDDEN, shift: 0 },
        vec![(hf, None), (weight("gpt2/weights/lm_head_w_i32.bin")?, Some(pool("gpt2/weights/lm_head_w_i32.bin")?))],
        vec![Some((hf_step, 0)), None],
    )?;
    let logits = logits[0].clone();
    let vocab_mask: Vec<i64> = (0..m)
        .flat_map(|_| (0..VOCAB).map(|column| if column < 50257 { 0 } else { -(1i64 << 41) }))
        .collect();
    let (masked_logits, _masked_logits_step) = builder.step(
        &AddOp,
        vec![(logits, None), (vocab_mask, None)],
        vec![Some((logits_step, 0)), None],
    )?;
    let logits = masked_logits[0].clone();
    let mut argmax = Vec::with_capacity(m);
    for row in 0..m {
        let values = logits[row * VOCAB..(row + 1) * VOCAB].to_vec();
        let (out, step) = builder.step(&ArgmaxOp, vec![(values, None)], vec![None])?;
        argmax.push(out[0][0] as u32);
        let _ = step;
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
        b"gpt2-seq16",
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
