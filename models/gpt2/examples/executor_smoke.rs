use std::path::Path;

use zkie_core::types::Digest;

use zkie_models_gpt2::executor::{build_commitment, execute_shared};
use zkie_ops::proof::model_commitment::{verify_shared_trace, ModelCommitmentPublic};

fn main() {
    let seq: usize = std::env::var("ZKIE_SEQ")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    let tokens: Vec<u32> = std::fs::read(format!(
        "/data/zkie-assets/cases/gpt2/tokens_{seq}_i32.bin"
    ))
    .unwrap()
    .chunks_exact(4)
    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as u32)
    .collect();
    let t0 = std::time::Instant::now();
    let commitment = build_commitment(Path::new("/data/zkie-assets/models"), &tokens).unwrap();
    eprintln!("commitment {:?} assets={}", t0.elapsed(), commitment.assets.len());
    let t1 = std::time::Instant::now();
    let trace = execute_shared(
        &commitment,
        &tokens.iter().map(|value| *value as i64).collect::<Vec<_>>(),
    )
    .unwrap();
    eprintln!("trace {:?} steps={} assets={}", t1.elapsed(), trace.steps.len(), trace.assets.len());
    let t2 = std::time::Instant::now();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let argmax =
        verify_shared_trace(&public, &trace, &tokens, &Digest::ZERO).unwrap();
    eprintln!("verify {:?}", t2.elapsed());
    let gt: Vec<i32> = std::fs::read(format!(
        "/data/zkie-assets/cases/gpt2/gt_int_argmax_{seq}_i32.bin"
    ))
    .unwrap()
    .chunks_exact(4)
    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
    .collect();
    let matches = argmax.iter().zip(&gt).filter(|(a, b)| **a as i32 == **b).count();
    let mismatches: Vec<usize> = argmax
        .iter()
        .zip(&gt)
        .enumerate()
        .filter(|(_, (a, b))| **a as i32 != **b)
        .map(|(index, _)| index)
        .collect();
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    if let Some(line) = status.lines().find(|line| line.starts_with("VmHWM:")) {
        eprintln!("{line}");
    }
    if seq <= 16 {
        println!(
            "seq={seq} argmax={argmax:?} matches={matches}/{} mismatches={mismatches:?}",
            gt.len()
        );
    } else {
        println!(
            "seq={seq} matches={matches}/{} mismatches={mismatches:?}",
            gt.len()
        );
    }
}
