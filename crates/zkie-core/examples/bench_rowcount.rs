//! THROWAWAY BENCHMARK TOOL -- not part of the normal build/test surface.
//!
//! Written for the resource-requirement estimate in
//! `docs/superpowers/specs/2026-07-26-zkie-full-model-resource-estimate.md`.
//! Not committed by the authoring session; left in place for a supervisor to
//! review/discard. This is an `examples/` target specifically so it is NOT
//! picked up by `cargo test --workspace` (which only builds lib/bins/tests/
//! doctests by default, not examples) -- it must never affect the normal
//! test suite.
//!
//! Recreates the spirit of the prior (deleted, uncommitted)
//! `crates/zkie-core/tests/scratch_bench_rowcount.rs` mentioned in
//! `docs/superpowers/specs/2026-07-26-zkie-smaller-timesfm-attempt.md`, but
//! measures setup/keygen/prove *separately* (not just combined), at more `k`
//! values, and reports real peak memory (via the caller wrapping this binary
//! in `/usr/bin/time -l`) and real on-disk SRS size.
//!
//! Circuit shape: `n_instances` independent `DotProductChip` regions, each a
//! real length-`DOT_LEN` dot product (`DOT_LEN = 264` = FinText 8M's real
//! `hidden_size`, so the benchmark's row-cost-per-instance ties directly to
//! Part 2's real row-count math), all sharing one configured
//! `DotProductConfig` -- mirroring exactly how `AssemblerChip::assign_dot_general`
//! instantiates one region per (i,j) output element of a real matmul. This
//! gives a realistic column count and gate-degree profile (the same
//! accumulation gate, final-rescale gate, and three range-check
//! sub-chips -- q/r/slack -- as every real DotGeneral dispatch), not a
//! degenerate trivial circuit.
//!
//! Usage: `cargo run --release --example bench_rowcount -- <k> [fill_fraction] [dump-srs]`

use std::env;
use std::fs::File;
use std::time::Instant;

use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::dev::MockProver;
use halo2_proofs::halo2curves::bn256::{Bn256, G1Affine};
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ConstraintSystem, ErrorFront,
};
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use halo2_proofs::SerdeFormat;
use rand_core::OsRng;

use zkie_core::chips::dot_general::{DotProductChip, DotProductConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

/// FinText 8M's real `hidden_size` (see the smaller-TimesFM-attempt report) --
/// used here as the dot-product contraction length so the measured
/// rows-per-instance number is directly reusable as a real (not guessed)
/// rows-per-scalar-mult multiplier in Part 2 of the resource estimate.
const DOT_LEN: usize = 264;

/// `DotProductChip::configure`'s fixed range-check overhead: 64 bits for `q`,
/// `REMAINDER_BITS = 60` for `r`, and 60 for `slack` (see
/// `crates/zkie-core/src/chips/dot_general.rs`). Independent of `DOT_LEN`.
const RANGE_CHECK_OVERHEAD_ROWS: usize = 64 + 60 + 60;

#[derive(Clone)]
struct BenchConfig {
    dot: DotProductConfig,
}

struct BenchCircuit {
    a: Vec<I18>,
    b: Vec<I18>,
    n_instances: usize,
}

impl Circuit<Fr> for BenchCircuit {
    type Config = BenchConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        BenchCircuit {
            a: self.a.clone(),
            b: self.b.clone(),
            n_instances: self.n_instances,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let accumulator = meta.advice_column();
        let q = meta.advice_column();
        let r = meta.advice_column();
        let slack = meta.advice_column();
        let bits = meta.advice_column();
        BenchConfig {
            dot: DotProductChip::configure(meta, a, b, accumulator, q, r, slack, bits, DOT_LEN),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = DotProductChip::construct(config.dot);
        for i in 0..self.n_instances {
            chip.assign(
                layouter.namespace(|| format!("bench dot {i}")),
                self.a.clone(),
                self.b.clone(),
            )
            .map(|_| ())
            .unwrap_or_else(|e| panic!("bench dot assign failed at instance {i}: {e}"));
        }
        Ok(())
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bench_rowcount <k> [fill_fraction] [dump-srs]");
        std::process::exit(1);
    }
    let k: u32 = args[1].parse().expect("k must be a u32");
    let fill_fraction: f64 = args
        .get(2)
        .map(|s| s.parse().expect("fill_fraction must be a f64"))
        .unwrap_or(0.9);
    let dump_srs = args.get(3).map(|s| s == "dump-srs").unwrap_or(false);

    let max_rows = 1u64 << k;
    let rows_per_instance = (DOT_LEN + RANGE_CHECK_OVERHEAD_ROWS) as u64;
    let n_instances =
        (((max_rows as f64) * fill_fraction) / (rows_per_instance as f64)).floor() as usize;
    let n_instances = n_instances.max(1);
    let total_rows = n_instances as u64 * rows_per_instance;
    let total_scalar_mults = n_instances as u64 * DOT_LEN as u64;

    println!("=== bench_rowcount k={k} ===");
    println!(
        "dot_len={DOT_LEN} rows_per_instance={rows_per_instance} n_instances={n_instances} \
         total_rows={total_rows} max_rows={max_rows} fill_pct={:.1} total_scalar_mults={total_scalar_mults} \
         rows_per_mult={:.4}",
        100.0 * total_rows as f64 / max_rows as f64,
        total_rows as f64 / total_scalar_mults as f64,
    );

    // Small, fixed values -- keep host-side raw_sum comfortably within i128
    // and the requantized I18 range regardless of DOT_LEN or n_instances
    // (each instance's dot product is independent; there is no
    // cross-instance accumulation to overflow).
    let a: Vec<I18> = (0..DOT_LEN)
        .map(|_| I18::from_f64(0.0001).unwrap())
        .collect();
    let b: Vec<I18> = (0..DOT_LEN)
        .map(|_| I18::from_f64(0.0002).unwrap())
        .collect();

    let circuit = BenchCircuit { a, b, n_instances };

    // Correctness check first (cheap relative to the real KZG phases) so a
    // shape/overflow bug surfaces immediately instead of silently corrupting
    // proving-time numbers.
    let mock_start = Instant::now();
    MockProver::run(k, &circuit, vec![])
        .unwrap()
        .assert_satisfied();
    println!(
        "mock_prover_check_secs={:.3}",
        mock_start.elapsed().as_secs_f64()
    );

    let mut rng = OsRng;

    let t0 = Instant::now();
    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let setup_secs = t0.elapsed().as_secs_f64();
    println!("setup_secs={setup_secs:.3}");

    let t1 = Instant::now();
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");
    let keygen_secs = t1.elapsed().as_secs_f64();
    println!("keygen_secs={keygen_secs:.3}");

    let t2 = Instant::now();
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[vec![]],
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let proof = transcript.finalize();
    let prove_secs = t2.elapsed().as_secs_f64();
    println!("prove_secs={prove_secs:.3}");
    println!("proof_bytes={}", proof.len());

    let t3 = Instant::now();
    let mut verifier_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let verifier_params = params.verifier_params();
    let strategy = SingleStrategy::new(&verifier_params);
    let result = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
        &verifier_params,
        &vk,
        strategy,
        &[vec![]],
        &mut verifier_transcript,
    );
    let verify_secs = t3.elapsed().as_secs_f64();
    println!("verify_secs={verify_secs:.3} verify_ok={}", result.is_ok());

    let total_secs = setup_secs + keygen_secs + prove_secs;
    println!("total_setup_keygen_prove_secs={total_secs:.3}");

    if dump_srs {
        let dir = ".spike-test/bench";
        std::fs::create_dir_all(dir).ok();
        let raw_path = format!("{dir}/srs_k{k}_raw.bin");
        let mut f = File::create(&raw_path).unwrap();
        params.write(&mut f).unwrap();
        drop(f);
        let raw_size = std::fs::metadata(&raw_path).unwrap().len();
        println!("srs_raw_bytes={raw_size} path={raw_path}");

        let proc_path = format!("{dir}/srs_k{k}_processed.bin");
        let mut f2 = File::create(&proc_path).unwrap();
        params
            .write_custom(&mut f2, SerdeFormat::Processed)
            .unwrap();
        drop(f2);
        let proc_size = std::fs::metadata(&proc_path).unwrap().len();
        println!("srs_processed_bytes={proc_size} path={proc_path}");
    }
}
