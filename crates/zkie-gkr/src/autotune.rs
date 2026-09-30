//! Autotuning loop: search `shard granularity x backend` for the lowest total
//! proof time, and return a reusable result (tune once per model).
//!
//! The cost model is seeded from measured constants (proof ~27.6s/layer,
//! WHIR commit+open ~0.30s/boundary) and captures the real tradeoff the
//! autotuner searches: finer granularity = more shards = more parallelism but
//! more boundary commits; coarser granularity = fewer commits but less
//! parallelism. GPU is a per-config speedup factor (a placeholder for the real
//! per-shard CUDA dispatch).

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Granularity {
    Op,
    Layer,
    Model,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug)]
pub struct TuningConfig {
    pub granularity: Granularity,
    pub backend: Backend,
}

#[derive(Clone, Debug)]
pub struct TuningResult {
    pub config: TuningConfig,
    pub shards: usize,
    pub boundaries: usize,
    pub proof_s: f64,
    pub commit_s: f64,
    pub total_s: f64,
}

impl fmt::Display for TuningResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}/{:?}: shards={} boundaries={} proof={:.1}s commit={:.1}s total={:.1}s",
            self.config.granularity, self.config.backend, self.shards, self.boundaries,
            self.proof_s, self.commit_s, self.total_s
        )
    }
}

#[derive(Clone, Debug)]
pub struct CostModel {
    pub layers: usize,
    pub ops_per_layer: usize,
    pub proof_s_per_layer: f64,
    pub commit_s_per_boundary: f64,
    pub gpu_speedup: f64,
    pub max_parallel_shards: usize,
}

impl CostModel {
    pub fn shards(&self, g: Granularity) -> usize {
        match g {
            Granularity::Op => self.ops_per_layer * self.layers,
            Granularity::Layer => self.layers,
            Granularity::Model => 1,
        }
    }

    pub fn estimate(&self, c: &TuningConfig) -> TuningResult {
        let shards = self.shards(c.granularity);
        let boundaries = shards.saturating_sub(1);
        // Single-thread proof work is ~proportional to layers; it is parallelized
        // across shards up to the hardware limit.
        let single_thread = self.proof_s_per_layer * self.layers as f64;
        let par = shards.min(self.max_parallel_shards).max(1) as f64;
        let proof_s = single_thread / par;
        let commit_s = boundaries as f64 * self.commit_s_per_boundary;
        let backend = match c.backend {
            Backend::Cpu => 1.0,
            Backend::Gpu => self.gpu_speedup,
        };
        let total_s = (proof_s + commit_s) / backend;
        TuningResult {
            config: *c,
            shards,
            boundaries,
            proof_s,
            commit_s,
            total_s,
        }
    }
}

/// Search the given configs and return the lowest-total one.
pub fn autotune(model: &CostModel, configs: &[TuningConfig]) -> TuningResult {
    configs
        .iter()
        .map(|c| model.estimate(c))
        .min_by(|a, b| a.total_s.partial_cmp(&b.total_s).unwrap())
        .expect("non-empty config set")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpt2_model() -> CostModel {
        CostModel {
            layers: 12,
            ops_per_layer: 14,
            proof_s_per_layer: 27.6,
            commit_s_per_boundary: 0.30,
            gpu_speedup: 5.0,
            max_parallel_shards: 64,
        }
    }

    #[test]
    fn finer_granularity_means_more_parallelism() {
        let m = gpt2_model();
        let op = m.estimate(&TuningConfig { granularity: Granularity::Op, backend: Backend::Cpu });
        let layer = m.estimate(&TuningConfig { granularity: Granularity::Layer, backend: Backend::Cpu });
        let model = m.estimate(&TuningConfig { granularity: Granularity::Model, backend: Backend::Cpu });
        // finer granularity -> more shards -> lower single-thread proof (more parallel)
        assert!(op.shards > layer.shards && layer.shards > model.shards);
        assert!(op.proof_s < layer.proof_s && layer.proof_s < model.proof_s);
        // but finer granularity -> more boundaries -> more commit cost
        assert!(op.commit_s > layer.commit_s && layer.commit_s > model.commit_s);
    }

    #[test]
    fn autotune_picks_lowest_total() {
        let m = gpt2_model();
        let configs = vec![
            TuningConfig { granularity: Granularity::Op, backend: Backend::Cpu },
            TuningConfig { granularity: Granularity::Layer, backend: Backend::Cpu },
            TuningConfig { granularity: Granularity::Model, backend: Backend::Cpu },
            TuningConfig { granularity: Granularity::Layer, backend: Backend::Gpu },
        ];
        let best = autotune(&m, &configs);
        let min_total = configs
            .iter()
            .map(|c| m.estimate(c).total_s)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(best.total_s, min_total);
    }
}
