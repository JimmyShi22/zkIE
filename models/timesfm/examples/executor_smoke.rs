use std::path::Path;

use zkie_core::types::Digest;

use zkie_models_timesfm::executor::{build_commitment, execute_shared};
use zkie_ops::proof::model_commitment::{trace_plan_from_trace, verify_shared_trace_output, ModelCommitmentPublic};

fn main() {
    let input: Vec<i32> = std::fs::read("/data/zkie-assets/cases/timesfm/input_ts_i32.bin")
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let values: Vec<i64> = input.iter().map(|value| *value as i64).collect();
    let expected: Vec<u32> = input.iter().map(|value| *value as u32).collect();
    let t0 = std::time::Instant::now();
    let commitment = build_commitment(Path::new("/data/zkie-assets"), &values).unwrap();
    eprintln!("commitment {:?} assets={}", t0.elapsed(), commitment.assets.len());
    let t1 = std::time::Instant::now();
    let trace = execute_shared(&commitment, &values).unwrap();
    eprintln!("trace {:?} steps={} assets={}", t1.elapsed(), trace.steps.len(), trace.assets.len());
    let t2 = std::time::Instant::now();
    let mut public = ModelCommitmentPublic::from_commitment(&commitment);
    public.trace_plan = Some(trace_plan_from_trace(&trace));
    let output =
        verify_shared_trace_output(&public, &trace, &expected, &Digest::ZERO).unwrap();
    eprintln!("verify {:?}", t2.elapsed());
    let gt: Vec<i32> = std::fs::read("/data/zkie-assets/cases/timesfm/gt_int_output_i32.bin")
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let bytes: Vec<u8> = output
        .iter()
        .flat_map(|value| (*value as i32).to_le_bytes())
        .collect();
    std::fs::write("/tmp/timesfm_out.bin", bytes).unwrap();
    let mut matches = 0usize;
    let mut maxdiff = 0i64;
    for (value, expected) in output.iter().zip(&gt) {
        if *value as i32 == *expected {
            matches += 1;
        }
        let diff = (*value - *expected as i64).abs();
        if diff > maxdiff {
            maxdiff = diff;
        }
    }
    println!("matches={matches}/{} maxdiff={maxdiff}", gt.len());
}
