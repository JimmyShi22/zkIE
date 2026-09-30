use zkie_gkr::autotune::{autotune, Backend, CostModel, Granularity, TuningConfig};

fn main() {
    // GPT-2 512 cost model, seeded from measured constants:
    // proof ~27.6s/layer, WHIR commit+open ~0.30s/boundary.
    let model = CostModel {
        layers: 12,
        ops_per_layer: 14,
        proof_s_per_layer: 27.6,
        commit_s_per_boundary: 0.30,
        gpu_speedup: 5.0, // placeholder for the real per-shard CUDA dispatch
        max_parallel_shards: 64, // ~CPU cores; GPU partition would be higher
    };

    let configs = vec![
        TuningConfig { granularity: Granularity::Op, backend: Backend::Cpu },
        TuningConfig { granularity: Granularity::Layer, backend: Backend::Cpu },
        TuningConfig { granularity: Granularity::Model, backend: Backend::Cpu },
        TuningConfig { granularity: Granularity::Op, backend: Backend::Gpu },
        TuningConfig { granularity: Granularity::Layer, backend: Backend::Gpu },
        TuningConfig { granularity: Granularity::Model, backend: Backend::Gpu },
    ];

    println!("GPT-2 512 autotuning search (granularity x backend):");
    for c in &configs {
        println!("  {}", model.estimate(c));
    }
    let best = autotune(&model, &configs);
    println!("best: {}", best);
}
