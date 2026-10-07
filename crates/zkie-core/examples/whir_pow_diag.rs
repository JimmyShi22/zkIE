//! Config-only WHIR PoW/query diagnostic.
//!
//! Prints the CONFIGURED security parameters and the DERIVED per-round
//! PoW/query schedule for n = 6 and n = 19 variables, for both the production
//! (`Whir::new`: 90-bit, 32 PoW-bit budget) and testing (`Whir::new_testing`:
//! 32-bit, 10 PoW-bit budget) instances.
//!
//! This constructs `WhirConfig` only — no commitments, no openings, no
//! proof-of-work grinding.
//!
//! Run: cargo run --release -p zkie-core --example whir_pow_diag

use zkie_core::pcs::whir::Whir;

fn report(label: &str, whir: &Whir) {
    let cfg = whir.config();
    println!("=== {label}: n={} variables ===", cfg.num_variables);
    println!("configured: security_level={} bits, pow_budget={} bits (grinding cap), soundness_type={:?}",
        cfg.params.security_level, cfg.params.pow_bits, cfg.params.soundness_type);
    println!("derived:    folding_schedule={:?} final_sumcheck_rounds={}",
        cfg.folding_schedule, cfg.final_sumcheck_rounds);
    println!("derived:    starting_folding_pow_bits={} final_folding_pow_bits={}",
        cfg.starting_folding_pow_bits, cfg.final_folding_pow_bits);
    for (i, r) in cfg.round_parameters.iter().enumerate() {
        println!("  round {i}: log_inv_rate={} num_queries={} pow_bits={} folding_pow_bits={} domain_size={}",
            r.log_inv_rate, r.num_queries, r.pow_bits, r.folding_pow_bits, r.domain_size);
    }
    println!("  final phase: num_queries={} pow_bits={}", cfg.final_queries, cfg.final_pow_bits);
    println!("derived:    max_pow_bits={} (<= budget: {})", cfg.max_pow_bits(), cfg.check_pow_bits());
    println!();
}

fn main() {
    for n in [6usize, 19] {
        report(&format!("prod Whir::new (90-bit, 32 PoW budget)"), &Whir::new(n));
        report(&format!("testing Whir::new_testing (32-bit, 10 PoW budget)"), &Whir::new_testing(n));
    }
    println!("(config-only: no commitments, no openings, no grinding performed)");
}
