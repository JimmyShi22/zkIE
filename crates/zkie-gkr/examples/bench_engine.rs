use zkie_gkr::engine::{autotune, Backend, Granularity, Model, StageCosts, StageSchedule, GpuSpeedup, estimate};

fn main() {
    let gpt2 = Model {
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
    };

    let granularities = vec![
        Granularity::Ops(1),
        Granularity::Layers(1),
        Granularity::Layers(2),
        Granularity::WholeModel,
    ];

    println!("GPT-2 512 autotuning (granularity x per-stage schedule):");
    let uniform_cpu = estimate(&gpt2, Granularity::Layers(1), StageSchedule::uniform(Backend::Cpu));
    let uniform_gpu = estimate(&gpt2, Granularity::Layers(1), StageSchedule::uniform(Backend::Gpu));
    println!("  uniform cpu: {}", uniform_cpu);
    println!("  uniform gpu: {}", uniform_gpu);

    let best = autotune(&gpt2, &granularities);
    println!("best: {}", best);
}
