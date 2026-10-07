//! Benchmark: KZG-style multilinear PC over BN254 (ark-poly-commit 0.5).
//!
//! Run: cargo run --release --bin kzg
//!
//! Env controls: PCS_N (comma-separated table arities, default "6,12"),
//! PCS_ROUNDS (default 5).
//!
//! The SRS is generated locally for this benchmark with OS randomness; the
//! returned parameters carry GROUP ENCODINGS of trapdoor-derived values (not
//! the raw trapdoor) and are benchmark-only — NOT production parameters. Its
//! contents are never printed.

use pcs_comparison::kzg;

fn main() {
    let ns: Vec<usize> = std::env::var("PCS_N")
        .unwrap_or_else(|_| "6,12".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let rounds: usize = std::env::var("PCS_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    println!("=== KZG multilinear PC (ark-poly-commit 0.5, BN254) ===");
    println!("2 integer-valued tables each, {rounds} rounds of fresh random challenge points;");
    println!("SRS: locally generated (OS randomness), group-encoded trapdoor-derived values, benchmark-only.");
    println!("rayon threads: {}", rayon::current_num_threads());
    println!();
    for n in &ns {
        let r = kzg::run(*n, 2, rounds);
        println!("n={n} (tables of 2^{n} values):");
        println!("  setup          : {:>10.3} ms", r.setup.as_secs_f64() * 1e3);
        println!("  commit (2 tbls): {:>10.3} ms", r.commit.as_secs_f64() * 1e3);
        println!("  open  (avg/rd) : {:>10.3} ms", r.open.as_secs_f64() * 1e3);
        println!("  verify(avg/rd) : {:>10.3} ms (PCS-only; caller-side MLE eval excluded)",
            r.verify.as_secs_f64() * 1e3);
        println!("  proof bytes    : {} (sum over both tables)", r.proof_bytes);
        println!("  commitment b   : {} (sum over both tables)", r.commitment_bytes);
        println!("  SRS bytes      : {}", r.srs_bytes);
        println!("  wrong value rejected     : {}", r.wrong_value_rejected);
        println!("  wrong commitment rejected: {}", r.wrong_commitment_rejected);
        println!();
    }
}
