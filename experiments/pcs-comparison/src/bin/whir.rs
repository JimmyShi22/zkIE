//! Benchmark: upstream p3-whir 0.7 over Goldilocks with a quadratic extension
//! challenge field.
//!
//! Run: cargo run --release --bin whir
//!
//! Env controls:
//! - PCS_N (comma-separated table arities, default "6,12")
//! - PCS_ROUNDS (default 5, must be >= 1)
//! - PCS_POW_BUDGET (default 0) — grinding budget for the FIXED 90-bit target;
//!   no automatic downgrade or fallback if the construction is rejected.
//! - PCS_TESTING_BASELINE (default 1; set 0 to skip) — the explicitly insecure
//!   32-bit/10-PoW-bit functional baseline. Skip it to measure peak RSS for a
//!   single configuration.
//!
//! If p3-whir rejects the 90-bit/budget combination, the rejection reason is
//! printed as-is and security is NOT weakened.

use pcs_comparison::whir;

fn main() {
    let ns: Vec<usize> = std::env::var("PCS_N")
        .unwrap_or_else(|_| "6,12".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&n| n > 0)
        .collect();
    let rounds: usize = std::env::var("PCS_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    if rounds == 0 {
        eprintln!("PCS_ROUNDS must be >= 1");
        std::process::exit(2);
    }
    let pow_budget: usize = std::env::var("PCS_POW_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let testing_baseline: bool = std::env::var("PCS_TESTING_BASELINE")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);

    println!("=== WHIR (p3-whir 0.7, Goldilocks base + BinomialExtensionField<F,2>) ===");
    println!("security target: 90-bit (FIXED) | pow budget: {pow_budget} (PCS_POW_BUDGET) | testing baseline: {testing_baseline}");
    println!("2 tables batched in ONE witness, ONE FRI proof, {rounds} rounds of fresh random challenge points.");
    println!("rayon threads: {}", rayon::current_num_threads());
    println!();
    for n in &ns {
        println!("n={n} (tables of 2^{n} values):");
        let r90 = whir::run(*n, 90, pow_budget, rounds);
        match &r90.config_error {
            Some(e) => {
                println!("  [90-bit, pow_budget={pow_budget}] REJECTED by p3-whir (recorded, not weakened):");
                println!("      {e}");
            }
            None => {
                println!("  [90-bit, pow_budget={pow_budget}] constructed:");
                println!("      final_queries={} max_pow_bits={}",
                    r90.final_queries.unwrap(), r90.max_pow_bits.unwrap());
                println!("      setup {:>10.3} ms | dft_init {:>10.3} ms | commit {:>10.3} ms | open {:>10.3} ms | verify {:>10.3} ms",
                    r90.setup.unwrap().as_secs_f64() * 1e3,
                    r90.dft_init.unwrap().as_secs_f64() * 1e3,
                    r90.commit.unwrap().as_secs_f64() * 1e3,
                    r90.open.unwrap().as_secs_f64() * 1e3,
                    r90.verify.unwrap().as_secs_f64() * 1e3);
                println!("      proof {} B (batched, both tables) | commitment {} B",
                    r90.proof_bytes.unwrap(), r90.commitment_bytes.unwrap());
                println!("      wrong point rejected: {} | wrong commitment rejected: {} | mutated value rejected: {}",
                    r90.wrong_point_rejected.unwrap(),
                    r90.wrong_commitment_rejected.unwrap(),
                    r90.mutated_value_rejected.unwrap());
            }
        }
        if testing_baseline {
            let r32 = whir::run(*n, 32, 10, rounds);
            println!("  [32-bit, pow_budget=10, EXPLICITLY INSECURE functional baseline]:");
            match &r32.config_error {
                Some(e) => println!("      REJECTED: {e}"),
                None => {
                    println!("      final_queries={} max_pow_bits={}",
                        r32.final_queries.unwrap(), r32.max_pow_bits.unwrap());
                    println!("      setup {:>10.3} ms | dft_init {:>10.3} ms | commit {:>10.3} ms | open {:>10.3} ms | verify {:>10.3} ms",
                        r32.setup.unwrap().as_secs_f64() * 1e3,
                        r32.dft_init.unwrap().as_secs_f64() * 1e3,
                        r32.commit.unwrap().as_secs_f64() * 1e3,
                        r32.open.unwrap().as_secs_f64() * 1e3,
                        r32.verify.unwrap().as_secs_f64() * 1e3);
                    println!("      proof {} B (batched, both tables) | commitment {} B",
                        r32.proof_bytes.unwrap(), r32.commitment_bytes.unwrap());
                    println!("      wrong point rejected: {} | wrong commitment rejected: {} | mutated value rejected: {}",
                        r32.wrong_point_rejected.unwrap(),
                        r32.wrong_commitment_rejected.unwrap(),
                        r32.mutated_value_rejected.unwrap());
                }
            }
        }
        println!();
    }
}
