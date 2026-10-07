//! Reproducible benchmark for the generic root-bound LogUp lookup
//! (`extension_lookup_logup`), end-to-end from the actual IR (Store +
//! Op::Lookup) to the root-only verifier.
//!
//! Sizes: `ROWS_LOG` / `TABLE_LOG` env vars (usize, power-of-two log2 of
//! the logical row count / table length). Modest defaults; invalid or
//! unreasonable values fall back to the defaults with a note. The witness
//! is deterministic (no RNG); the WHIR instances target 90-bit security
//! with a `POW_BUDGET`-capped grinding budget (default 32 — at large
//! arities p3-whir derives per-round PoW and zero budget is infeasible),
//! and an infeasible configuration exits with an error instead of hanging.
//!
//! The Store and the Op are DROPPED before verification — the verifier
//! holds statement + proof only.
//!
//! Run: ROWS_LOG=15 TABLE_LOG=20 cargo run --release -p zkie-ops \
//!      --example bench_lookup_logup

use std::time::Instant;
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_ops::compose::{Op, Store};
use zkie_ops::extension_lookup_logup::{prove_op_lookup, verify, LookupWhir};

const DEFAULT_ROWS_LOG: usize = 10;
const DEFAULT_TABLE_LOG: usize = 12;
const MAX_LOG: usize = 20;
const SECURITY_LEVEL: usize = 90;
const DEFAULT_POW_BUDGET: usize = 32;

fn env_log(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Err(_) => default,
        Ok(v) => match v.parse::<usize>() {
            Ok(x) if x >= 1 && x <= MAX_LOG => x,
            Ok(_) => {
                eprintln!(
                    "{} out of range 1..={}; using default {}",
                    name, MAX_LOG, default
                );
                default
            }
            Err(_) => {
                eprintln!("{} is not a valid integer; using default {}", name, default);
                default
            }
        },
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Err(_) => default,
        Ok(v) => v.parse::<usize>().unwrap_or(default),
    }
}

fn main() {
    let rows_log = env_log("ROWS_LOG", DEFAULT_ROWS_LOG);
    let table_log = env_log("TABLE_LOG", DEFAULT_TABLE_LOG);
    let pow_budget = env_usize("POW_BUDGET", DEFAULT_POW_BUDGET);
    let p = 1usize << rows_log;
    let t = 1usize << table_log;
    println!(
        "lookup_logup bench: rows=2^{}={} table=2^{}={} security={} pow_budget={}",
        rows_log, p, table_log, t, SECURITY_LEVEL, pow_budget
    );

    // Deterministic witness built through the actual IR.
    let mut store = Store::new();
    let table: Vec<Goldilocks> =
        (0..t).map(|j| Goldilocks::from_u64((j % 1000) as u64)).collect();
    let idx: Vec<u32> = (0..p).map(|i| (((i * 7) + 3) % t) as u32).collect();
    let out: Vec<Goldilocks> = idx.iter().map(|&i| table[i as usize]).collect();
    let idx_id = store.push_idx(idx);
    let tbl_id = store.push(table);
    let out_id = store.push(out);
    let op = Op::Lookup {
        idx: idx_id,
        out: out_id,
        table: tbl_id,
    };

    let Some(lw) = LookupWhir::new(p, t, SECURITY_LEVEL, pow_budget) else {
        eprintln!("LookupWhir::new rejected the configuration (security/budget infeasible)");
        std::process::exit(1);
    };
    let prove_start = Instant::now();
    let Some((stmt, proof)) = prove_op_lookup(&lw, &op, &store) else {
        eprintln!("prove_op_lookup rejected the configuration");
        std::process::exit(1);
    };
    let prove_secs = prove_start.elapsed().as_secs_f64();
    drop(store);
    drop(op);

    let verify_start = Instant::now();
    let accepted = verify(&lw, &stmt, &proof);
    let verify_secs = verify_start.elapsed().as_secs_f64();

    let (rows_opens, rows_open_s) = lw.rows.open_stats();
    let (rows_ver, rows_ver_s) = lw.rows.verify_stats();
    let (ent_opens, ent_open_s) = lw.entries.open_stats();
    let (ent_ver, ent_ver_s) = lw.entries.verify_stats();
    println!("prove_elapsed_secs: {:.3}", prove_secs);
    println!(
        "phase_initial_commit_secs: {:.3}",
        proof.timings.initial_commit_secs
    );
    println!(
        "phase_derived_inverse_commit_secs: {:.3}",
        proof.timings.derived_inverse_commit_secs
    );
    println!(
        "phase_relation_prove_secs: {:.3}",
        proof.timings.relation_prove_secs
    );
    println!(
        "phase_terminal_open_secs: {:.3}",
        proof.timings.terminal_open_secs
    );
    println!("verify_elapsed_secs: {:.3}", verify_secs);
    println!(
        "rows_whir open_count={} open_secs={:.3} verify_count={} verify_secs={:.3}",
        rows_opens, rows_open_s, rows_ver, rows_ver_s
    );
    println!(
        "entries_whir open_count={} open_secs={:.3} verify_count={} verify_secs={:.3}",
        ent_opens, ent_open_s, ent_ver, ent_ver_s
    );
    println!("accepted: {}", accepted);
    if !accepted {
        println!("result: REJECTED");
        std::process::exit(1);
    }
    println!("result: accepted");
}
