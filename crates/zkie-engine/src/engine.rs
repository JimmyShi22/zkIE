//! IE (Inference Engine): compile a model into a shard DAG at a tunable
//! granularity, assign a *per-stage* CPU/GPU backend schedule (mixed within a
//! shard), and autotune the `granularity x schedule x layout` space. One model
//! is tuned once and the result is reusable.

use std::fmt;

/// Where a single proof-pipeline stage runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Backend {
    Cpu,
    Gpu,
}

/// The proof pipeline inside one shard. Each stage is independently dispatchable
/// to CPU or GPU — the "mixed backend" knob.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Stage {
    /// Forward matmul (witness generation).
    Forward,
    /// WHIR/FRI commitment of the codeword.
    Commit,
    /// GKR reduction + affine + logUp sumchecks.
    Sumcheck,
    /// FRI opening.
    Open,
}

pub const ALL_STAGES: [Stage; 4] = [Stage::Forward, Stage::Commit, Stage::Sumcheck, Stage::Open];

/// A per-stage backend assignment (CPU/GPU may be mixed within one shard).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StageSchedule {
    pub forward: Backend,
    pub commit: Backend,
    pub sumcheck: Backend,
    pub open: Backend,
}

impl StageSchedule {
    pub fn uniform(b: Backend) -> Self {
        StageSchedule { forward: b, commit: b, sumcheck: b, open: b }
    }

    pub fn get(&self, s: Stage) -> Backend {
        match s {
            Stage::Forward => self.forward,
            Stage::Commit => self.commit,
            Stage::Sumcheck => self.sumcheck,
            Stage::Open => self.open,
        }
    }

    pub fn set(&mut self, s: Stage, b: Backend) {
        match s {
            Stage::Forward => self.forward = b,
            Stage::Commit => self.commit = b,
            Stage::Sumcheck => self.sumcheck = b,
            Stage::Open => self.open = b,
        }
    }
}

/// Shard granularity — a public parameter, *not* hardcoded to "layer".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Granularity {
    /// `n` ops per shard.
    Ops(usize),
    /// `n` layers per shard.
    Layers(usize),
    /// The whole model is one shard.
    WholeModel,
}

impl Granularity {
    /// Map this granularity to a concrete `ops_per_shard`. `layers` is the model
    /// depth (used to turn `Layers(n)` into an op count).
    pub fn ops_per_shard(self, op_count: usize, layers: usize) -> usize {
        match self {
            Granularity::Ops(n) => n.max(1),
            Granularity::Layers(n) => {
                let ops_per_layer = op_count / layers.max(1);
                (n.saturating_mul(ops_per_layer)).max(1)
            }
            Granularity::WholeModel => op_count.max(1),
        }
    }
}

/// A shard: a contiguous op range + its schedule.
#[derive(Clone, Debug)]
pub struct Shard {
    pub op_start: usize,
    pub op_end: usize,
    pub schedule: StageSchedule,
}

/// The shard DAG: shards plus cross-shard boundary bindings. Boundary `i` means
/// the output commitment of `shards[i]` must equal the input commitment of
/// `shards[i+1]` (enforced via `same_poly` + a shared boundary commitment).
#[derive(Clone, Debug)]
pub struct ShardDag {
    pub shards: Vec<Shard>,
    pub boundaries: Vec<(usize, usize)>,
}

/// Compile a flat op list (`op_count` ops, `layers` layers) into a shard DAG at
/// the requested granularity, with a uniform schedule (the autotuner then varies
/// the schedule per stage).
pub fn compile_shard_dag(
    op_count: usize,
    layers: usize,
    granularity: Granularity,
    schedule: StageSchedule,
) -> ShardDag {
    let ops_per_shard = granularity.ops_per_shard(op_count, layers);

    let mut shards = Vec::new();
    let mut s = 0;
    while s < op_count {
        let e = (s + ops_per_shard).min(op_count);
        shards.push(Shard { op_start: s, op_end: e, schedule });
        s = e;
    }
    let boundaries: Vec<(usize, usize)> =
        (0..shards.len().saturating_sub(1)).map(|i| (i, i + 1)).collect();
    ShardDag { shards, boundaries }
}

/// Prove a whole model (flat op list) at the given granularity, driving the real
/// shard-DAG composer. This is the bridge between the autotuner's `Granularity`
/// knob and the actual proof — the granularity is no longer just op counting.
pub fn prove_model(
    store: &mut zkie_ops::compose::Store,
    ops: &[zkie_ops::compose::Op],
    granularity: Granularity,
    layers: usize,
    rng: &mut zkie_core::common::field::XorShift64,
) -> zkie_ops::compose::ShardDagProof {
    let ops_per_shard = granularity.ops_per_shard(ops.len(), layers);
    zkie_ops::compose::prove_shard_dag(store, ops, ops_per_shard, rng)
}

/// Verify a whole-model proof produced by [`prove_model`].
pub fn verify_model(
    store: &zkie_ops::compose::Store,
    ops: &[zkie_ops::compose::Op],
    granularity: Granularity,
    layers: usize,
    proof: &zkie_ops::compose::ShardDagProof,
) -> bool {
    let ops_per_shard = granularity.ops_per_shard(ops.len(), layers);
    zkie_ops::compose::verify_shard_dag(store, ops, ops_per_shard, proof)
}

/// Per-stage CPU costs. `Forward`/`Sumcheck` are per op; `Commit`/`Open` are per
/// boundary.
#[derive(Clone, Copy, Debug)]
pub struct StageCosts {
    pub forward_s_per_op: f64,
    pub commit_s_per_boundary: f64,
    pub sumcheck_s_per_op: f64,
    pub open_s_per_boundary: f64,
}

/// Per-stage GPU speedup. A value > 1 means GPU is faster; a value < 1 means the
/// stage is launch/transfer-bound and the CPU wins (the "mixed" insight).
#[derive(Clone, Copy, Debug)]
pub struct GpuSpeedup {
    pub forward: f64,
    pub commit: f64,
    pub sumcheck: f64,
    pub open: f64,
}

impl GpuSpeedup {
    pub fn factor(&self, s: Stage) -> f64 {
        match s {
            Stage::Forward => self.forward,
            Stage::Commit => self.commit,
            Stage::Sumcheck => self.sumcheck,
            Stage::Open => self.open,
        }
    }
}

/// A concrete model + hardware profile to tune against.
#[derive(Clone, Copy, Debug)]
pub struct Model {
    pub op_count: usize,
    pub layers: usize,
    pub costs: StageCosts,
    pub gpu: GpuSpeedup,
    /// Hardware parallelism limit (shards that can run concurrently).
    pub max_parallel_shards: usize,
}

/// The reusable tuning result: one config's cost breakdown.
#[derive(Clone, Debug)]
pub struct TuningResult {
    pub granularity: Granularity,
    pub schedule: StageSchedule,
    pub shards: usize,
    pub boundaries: usize,
    pub forward_s: f64,
    pub commit_s: f64,
    pub sumcheck_s: f64,
    pub open_s: f64,
    pub total_s: f64,
}

impl fmt::Display for TuningResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} shards={} sched=[F:{:?} C:{:?} S:{:?} O:{:?}] forward={:.1}s commit={:.1}s sumcheck={:.1}s open={:.1}s total={:.1}s",
            self.granularity,
            self.shards,
            self.schedule.forward,
            self.schedule.commit,
            self.schedule.sumcheck,
            self.schedule.open,
            self.forward_s,
            self.commit_s,
            self.sumcheck_s,
            self.open_s,
            self.total_s
        )
    }
}

fn backend_time(work_s: f64, backend: Backend, gpu_speedup: f64, par: f64) -> f64 {
    let factor = match backend {
        Backend::Cpu => 1.0,
        Backend::Gpu => 1.0 / gpu_speedup,
    };
    work_s * factor / par
}

/// Estimate a (granularity, schedule) config. `layout` (parallelism) is captured
/// by `max_parallel_shards` on the model.
pub fn estimate(model: &Model, granularity: Granularity, schedule: StageSchedule) -> TuningResult {
    let dag = compile_shard_dag(model.op_count, model.layers, granularity, schedule);
    let shards = dag.shards.len();
    let boundaries = dag.boundaries.len();
    let par = shards.min(model.max_parallel_shards).max(1) as f64;

    let forward_s = backend_time(
        model.op_count as f64 * model.costs.forward_s_per_op,
        schedule.forward,
        model.gpu.forward,
        par,
    );
    let commit_s = backend_time(
        boundaries as f64 * model.costs.commit_s_per_boundary,
        schedule.commit,
        model.gpu.commit,
        par,
    );
    let sumcheck_s = backend_time(
        model.op_count as f64 * model.costs.sumcheck_s_per_op,
        schedule.sumcheck,
        model.gpu.sumcheck,
        par,
    );
    let open_s = backend_time(
        boundaries as f64 * model.costs.open_s_per_boundary,
        schedule.open,
        model.gpu.open,
        par,
    );

    TuningResult {
        granularity,
        schedule,
        shards,
        boundaries,
        forward_s,
        commit_s,
        sumcheck_s,
        open_s,
        total_s: forward_s + commit_s + sumcheck_s + open_s,
    }
}

/// Enumerate all 16 stage schedules (2^4) for the search.
pub fn all_schedules() -> Vec<StageSchedule> {
    let mut out = Vec::with_capacity(16);
    for f in [Backend::Cpu, Backend::Gpu] {
        for c in [Backend::Cpu, Backend::Gpu] {
            for s in [Backend::Cpu, Backend::Gpu] {
                for o in [Backend::Cpu, Backend::Gpu] {
                    out.push(StageSchedule { forward: f, commit: c, sumcheck: s, open: o });
                }
            }
        }
    }
    out
}

/// Autotune over `granularity x schedule`; returns the lowest-total config.
pub fn autotune(model: &Model, granularities: &[Granularity]) -> TuningResult {
    let schedules = all_schedules();
    let mut best: Option<TuningResult> = None;
    for &g in granularities {
        for &sched in &schedules {
            let r = estimate(model, g, sched);
            if best.as_ref().map_or(true, |b| r.total_s < b.total_s) {
                best = Some(r);
            }
        }
    }
    best.expect("non-empty granularity set")
}

/// Autotune with a caller-supplied measurement function instead of the cost
/// model. This is the "real" loop: the caller runs the actual proof at each
/// (granularity, schedule) config and returns the measured `TuningResult`; the
/// engine only searches and returns the lowest-total one. Tune once per model
/// and cache the returned result.
pub fn autotune_with<F>(
    granularities: &[Granularity],
    schedules: &[StageSchedule],
    mut measure: F,
) -> TuningResult
where
    F: FnMut(Granularity, StageSchedule) -> TuningResult,
{
    let mut best: Option<TuningResult> = None;
    for &g in granularities {
        for &sched in schedules {
            let r = measure(g, sched);
            if best.as_ref().map_or(true, |b| r.total_s < b.total_s) {
                best = Some(r);
            }
        }
    }
    best.expect("non-empty config set")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpt2() -> Model {
        Model {
            op_count: 12 * 14,
            layers: 12,
            costs: StageCosts {
                forward_s_per_op: 1.7,
                commit_s_per_boundary: 0.15,
                sumcheck_s_per_op: 2.0,
                open_s_per_boundary: 0.15,
            },
            gpu: GpuSpeedup {
                forward: 5.0,
                commit: 1.2,
                sumcheck: 0.8,
                open: 1.2,
            },
            max_parallel_shards: 64,
        }
    }

    #[test]
    fn granularity_is_a_parameter_not_a_layer() {
        let m = gpt2();
        let op = compile_shard_dag(m.op_count, m.layers, Granularity::Ops(1), StageSchedule::uniform(Backend::Cpu));
        let layer = compile_shard_dag(m.op_count, m.layers, Granularity::Layers(1), StageSchedule::uniform(Backend::Cpu));
        let model = compile_shard_dag(m.op_count, m.layers, Granularity::WholeModel, StageSchedule::uniform(Backend::Cpu));
        assert_eq!(op.shards.len(), m.op_count);
        assert_eq!(layer.shards.len(), m.layers);
        assert_eq!(model.shards.len(), 1);
        // boundaries = shards - 1
        assert_eq!(layer.boundaries.len(), m.layers - 1);
    }

    #[test]
    fn mixed_schedule_beats_uniform() {
        let m = gpt2();
        let uniform_gpu = estimate(&m, Granularity::Layers(1), StageSchedule::uniform(Backend::Gpu));
        let uniform_cpu = estimate(&m, Granularity::Layers(1), StageSchedule::uniform(Backend::Cpu));
        // sumcheck is CPU-favored (0.8x), so all-GPU is worse than all-CPU on it.
        assert!(uniform_gpu.sumcheck_s > uniform_cpu.sumcheck_s);
        // autotune finds something at least as good as both uniforms.
        let best = autotune(&m, &[Granularity::Layers(1)]);
        assert!(best.total_s <= uniform_gpu.total_s && best.total_s <= uniform_cpu.total_s);
    }

    #[test]
    fn granularity_drives_real_shard_dag() {
        use zkie_ops::compose::{Op, Store};
        use zkie_core::common::field::XorShift64;
        use zkie_core::common::fixed_point::from_i64;

        let mut rng = XorShift64::new(0x1313);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..4 {
            ws.push(store.push((0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect()));
            bs.push(store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect()));
        }
        let mut outs = Vec::new();
        let mut rems = Vec::new();
        for _ in 0..4 {
            outs.push(store.push(vec![]));
            rems.push(store.push(vec![]));
        }
        let ops = vec![
            Op::Projection { x, w: ws[0], bias: bs[0], out: outs[0], rem: rems[0], m, k: d, n: d, shift },
            Op::Projection { x: outs[0], w: ws[1], bias: bs[1], out: outs[1], rem: rems[1], m, k: d, n: d, shift },
            Op::Projection { x: outs[1], w: ws[2], bias: bs[2], out: outs[2], rem: rems[2], m, k: d, n: d, shift },
            Op::Projection { x: outs[2], w: ws[3], bias: bs[3], out: outs[3], rem: rems[3], m, k: d, n: d, shift },
        ];

        let p_op = prove_model(&mut store, &ops, Granularity::Ops(1), 1, &mut rng);
        assert_eq!(p_op.shards.len(), 4);
        assert!(verify_model(&store, &ops, Granularity::Ops(1), 1, &p_op));

        let p_whole = prove_model(&mut store, &ops, Granularity::WholeModel, 1, &mut rng);
        assert_eq!(p_whole.shards.len(), 1);
        assert!(verify_model(&store, &ops, Granularity::WholeModel, 1, &p_whole));
    }
}
