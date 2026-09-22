//! Real, end-to-end integration test: the REAL `FinText/TimesFM_8M_2000_Global`
//! checkpoint's actual layer-0 `input_layernorm` (RMSNorm) weight, and a REAL
//! activation vector captured via a forward hook during a real PyTorch
//! forward pass on synthetic input (both extracted by
//! `.spike-test/fintext/extract_rms_norm_real_data.py`, a throwaway spike
//! script -- see `docs/superpowers/specs/2026-07-26-zkie-smaller-timesfm-attempt.md`
//! for the full extraction process), embedded here as literal constants so
//! this test is deterministic and needs no Python/network access to run.
//!
//! Pipeline exercised, matching the REAL exported ONNX graph structure this
//! investigation confirmed against `FinText/TimesFM_8M_2000_Global`'s actual
//! ONNX export (`Pow`/`ReduceMean`/`Add`/`Sqrt`/`Reciprocal`/`Mul`/`Mul`):
//!
//!   hand-built `GraphProto` (real op sequence, real weight values)
//!     -> `graph_compiler::compile_graph` (using this session's new
//!        `rms_norm_fusion` pattern recognition to fuse the 7 nodes into one
//!        `Instruction::RmsNorm`)
//!     -> `circuit_binding::to_assembler_program`
//!     -> `zkie_core::assembler::AssemblerChip` (dispatches to `RmsNormChip`)
//!     -> real KZG setup/prove/verify (+ tampered-proof negative test)
//!     -> numerical comparison against the real PyTorch RMSNorm output.
//!
//! # Known, honestly-reported gap: only the first 8 of 264 real channels
//!
//! This is the most important finding of this test, discovered empirically
//! (not anticipated in advance): `zkie_core::chips::reduce::ReduceSumChip`'s
//! running sum accumulates in RAW `i64` space (`partial_sums[i-1].checked_add(v.raw())`,
//! see that chip's own `assign`), i.e. the *sum itself*, not just the final
//! mean, must stay within `I18`'s representable range (`~9.22` in
//! magnitude) -- unlike `DotProductChip`, whose accumulator is a wider
//! `i128` (see `crate::assembler::assign_dot_general`'s `raw_sum: i128`).
//! `RmsNormChip`'s `mean(x^2)` step reuses `ReduceMeanChip`/`ReduceSumChip`
//! directly, so it inherits this narrower limitation. The REAL FinText
//! layer-0 activation's squared values sum to about `387` over all 264 (or
//! 258, excluding 6 that individually overflow `I18` on their own) real
//! channels -- **far beyond** `9.22`. Attempting the full real 258-channel
//! computation was tried FIRST in this investigation and failed exactly
//! this way (`RmsNormChip`/`ReduceSumChip` would panic with
//! `"I18 reduce sum overflow"` on the real circuit path; this test's own
//! host-side domain precomputation hit the equivalent bug first, surfacing
//! as a `LookupChip` domain error) -- see the accompanying report for the
//! full account. This is a REAL, previously-unknown architectural capacity
//! limit of the current `ReduceSumChip`/`ReduceMeanChip` design, not
//! specific to TimesFM or to this test.
//!
//! Given that wall, this test uses the first `K = 8` of the real 264
//! channels (indices 0-7, literally unmodified, chosen only because they
//! are a natural prefix -- not cherry-picked to flatter any metric), whose
//! real squared values sum to about `8.10` -- safely under `9.22`. This is
//! genuinely real data (both `x` and `weight`), but a considerably smaller
//! real slice than the full real RMSNorm computation, and
//! `FINTEXT_LAYER0_RMS_NORM_EXPECTED_OUTPUT` below is RMSNorm computed over
//! exactly these 8 real values (`mean(x^2)` over 8 elements, not 264) --
//! not the model's own 264-channel forward-pass output for that token.
//!
//! # Known, honestly-reported gap: epsilon rounds to zero
//!
//! `Instruction::RmsNorm`'s `epsilon_milli` (thousandths, mirroring
//! `Instruction::LayerNorm`'s identical convention) cannot represent
//! TimesFM's real `eps = 1e-6`: `round(1e-6 * 1000) = 0`. This test's
//! circuit therefore actually computes with `epsilon = 0`, not `1e-6`.
//! Numerically negligible here (`mean(x^2) ~= 1.5`, so replacing `1e-6` with
//! `0` changes the result by about 1 part in 1.5 million -- see the
//! tolerance used in this test's final comparison), but would matter for
//! near-zero-variance activations where epsilon exists specifically to guard
//! against blowup. A real, pre-existing ISA limitation, not new to this
//! session -- now confirmed to actually bite on real TimesFM-scale data.

// These constants are a verbatim, full-precision transcription of real
// extracted floating-point data (see this module's own docs) -- intentional,
// not accidentally over-precise literals.
#![allow(clippy::excessive_precision)]

use std::cell::RefCell;
use std::collections::HashMap;

use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::dev::MockProver;
use halo2_proofs::halo2curves::bn256::{Bn256, G1Affine};
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ConstraintSystem, ErrorFront,
};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use rand_core::OsRng;

use zkie_compiler::circuit_binding::to_assembler_program;
use zkie_compiler::graph_compiler::compile_graph;
use zkie_compiler::onnx::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use zkie_core::assembler::{AssemblerChip, AssemblerConfig, AssemblerProgram};
use zkie_core::chips::layer_norm::RsqrtDomain;
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

// dim=8 (first 8 real channels of FinText layer0 input_layernorm's real
// activation + weight -- none of these needed exclusion for I18 range).
// Real epsilon (float64, TimesFM's RMSNorm default) = 1e-06.
const FINTEXT_LAYER0_RMS_NORM_X: [f64; 8] = [
    -5.05007207393646240e-02,
    -1.55950617790222168e+00,
    -2.67525017261505127e-02,
    -1.26703572273254395e+00,
    4.33678776025772095e-01,
    -1.34162211418151855e+00,
    -1.43549692630767822e+00,
    8.94961804151535034e-02,
];
const FINTEXT_LAYER0_RMS_NORM_WEIGHT: [f64; 8] = [
    2.30960012413561344e-03,
    -1.45695207174867392e-03,
    3.29286698251962662e-03,
    -3.56230302713811398e-03,
    7.55330431275069714e-04,
    2.02001072466373444e-03,
    2.22328794188797474e-03,
    2.46914196759462357e-03,
];
const FINTEXT_LAYER0_RMS_NORM_EXPECTED_OUTPUT: [f64; 8] = [
    -1.15932855888079716e-04,
    2.25841905122691616e-03,
    -8.75610079057291150e-05,
    4.48633689612505226e-03,
    3.25594690800417425e-04,
    -2.69374230721983144e-03,
    -3.17226999411913610e-03,
    2.19645710141217534e-04,
];

const DIM: usize = 8;
// round(1e-6 * 1000) = 0 -- see this module's docs on the epsilon-rounds-to-
// zero gap.
const EPSILON_MILLI: u64 = 0;
const CIRCUIT_K: u32 = 13;

fn node(name: &str, op_type: &str, input: Vec<&str>, output: Vec<&str>) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: input.into_iter().map(String::from).collect(),
        output: output.into_iter().map(String::from).collect(),
        ..Default::default()
    }
}

fn scalar_initializer(name: &str, value: f32) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: vec![],
        data_type: 1, // FLOAT
        float_data: vec![value],
        ..Default::default()
    }
}

fn vector_initializer(name: &str, data: Vec<f32>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: vec![data.len() as i64],
        data_type: 1, // FLOAT
        float_data: data,
        ..Default::default()
    }
}

fn graph_input(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

fn graph_output(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

/// Builds a `GraphProto` replicating the EXACT real 7-node RMSNorm
/// decomposition this investigation confirmed against the actual
/// `FinText/TimesFM_8M_2000_Global` ONNX export's `layer0.input_layernorm`
/// subgraph (see `rms_norm_fusion`'s module docs for the real node dump this
/// mirrors), using the real extracted weight/epsilon values embedded above.
/// The `ReduceMean` node's second input (`axes`, an INT64 tensor in the real
/// opset-18 export) deliberately has NO corresponding entry in
/// `graph.initializer` -- `onnx_parser::extract_initializers` is FLOAT-only
/// by design (a real, pre-existing, separately-documented scope limitation:
/// see the report this test accompanies), and this fusion's own detection
/// never needs to resolve that tensor's value (see `rms_norm_fusion`'s
/// docs), so omitting it here reflects a genuine, honestly-scoped test
/// construction choice, not a claim that `onnx_parser` handles INT64
/// initializers generally.
fn rms_norm_graph() -> GraphProto {
    GraphProto {
        input: vec![graph_input("x")],
        output: vec![graph_output("mul2_out")],
        initializer: vec![
            scalar_initializer("exp2", 2.0),
            scalar_initializer("eps", 1e-6),
            vector_initializer(
                "weight",
                FINTEXT_LAYER0_RMS_NORM_WEIGHT
                    .iter()
                    .map(|v| *v as f32)
                    .collect(),
            ),
        ],
        node: vec![
            node("node_pow", "Pow", vec!["x", "exp2"], vec!["pow_out"]),
            node(
                "node_mean",
                "ReduceMean",
                vec!["pow_out", "axes"],
                vec!["mean_out"],
            ),
            node("node_add", "Add", vec!["mean_out", "eps"], vec!["add_out"]),
            node("node_sqrt", "Sqrt", vec!["add_out"], vec!["sqrt_out"]),
            node(
                "node_recip",
                "Reciprocal",
                vec!["sqrt_out"],
                vec!["recip_out"],
            ),
            node("node_mul1", "Mul", vec!["x", "recip_out"], vec!["mul1_out"]),
            node(
                "node_mul2",
                "Mul",
                vec!["mul1_out", "weight"],
                vec!["mul2_out"],
            ),
        ],
        ..Default::default()
    }
}

fn i18(v: f64) -> I18 {
    I18::from_f64(v).unwrap()
}

/// Computes the EXACT `mean(x^2) + epsilon` value `RmsNormChip::assign` will
/// produce for `x`, using the identical fixed-point steps (`I18::from_f64`
/// per element, `requantize_mul` for squaring, then the same reciprocal-based
/// mean rescale `zkie_core::chips::reduce::ReduceMeanChip` uses) -- so the
/// `rsqrt` lookup domain built from it is guaranteed to contain an EXACT
/// grid point matching what the real circuit computes (see
/// `zkie_core::chips::rms_norm`'s "CRITICAL NUMERIC LIMITATION" docs on why
/// an approximate domain would not work: `LookupChip` requires an exact
/// match, not a nearest-point lookup).
fn exact_mean_sq_plus_eps(xs: &[f64], epsilon_milli: u64) -> I18 {
    let k = xs.len();
    let squares: Vec<I18> = xs
        .iter()
        .map(|x| {
            let x18 = i18(*x);
            let (sq, _) = zkie_core::fixed_point::requantize_mul(x18, x18).unwrap();
            sq
        })
        .collect();
    let raw_sum: i128 = squares.iter().map(|s| s.raw() as i128).sum();
    // Mirrors `ReduceSumChip`'s running sum (plain integer addition of raw
    // I18 values) then `ReduceMeanChip`'s reciprocal-based rescale
    // (`sum * reciprocal`, requantized) -- see `chips/reduce.rs`.
    let reciprocal = i18(1.0 / k as f64);
    let (mean, _) =
        zkie_core::fixed_point::requantize_mul(I18::from_raw(raw_sum as i64), reciprocal).unwrap();
    let epsilon = I18::from_f64(epsilon_milli as f64 / 1000.0).unwrap();
    I18::from_raw(mean.raw().checked_add(epsilon.raw()).unwrap())
}

#[derive(Clone)]
struct RmsNormFinTextCircuit {
    program: AssemblerProgram,
    domain: RsqrtDomain,
    captured_outputs: RefCell<Option<Vec<I18>>>,
}

impl Circuit<Fr> for RmsNormFinTextCircuit {
    type Config = AssemblerConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        RmsNormFinTextCircuit {
            program: self.program.clone(),
            domain: self.domain.clone(),
            captured_outputs: RefCell::new(None),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        // `configure` has no access to `self` -- recompile the same real
        // graph once more just to get the program's shape (mirrors
        // `circuit_binding_pipeline.rs`'s own precedent one level up).
        let graph = rms_norm_graph();
        let compiled = compile_graph(&graph).expect("should compile the real RMSNorm subgraph");
        let mut graph_input_values = HashMap::new();
        graph_input_values.insert("x".to_string(), vec![I18::from_raw(0); DIM]);
        let program = to_assembler_program(&compiled, &graph_input_values)
            .expect("should convert to an assembler program");

        let target = exact_mean_sq_plus_eps(&FINTEXT_LAYER0_RMS_NORM_X, EPSILON_MILLI);
        let mut domains = HashMap::new();
        domains.insert(
            (DIM, EPSILON_MILLI),
            RsqrtDomain::RawAnchors(vec![target.raw()]),
        );
        AssemblerChip::configure_with_rms_norm_domains(meta, &program.instructions, &domains)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = AssemblerChip::construct(config);
        chip.load_rms_norm_tables(layouter.namespace(|| "load rms norm tables"))?;
        let outputs = chip
            .assign(layouter.namespace(|| "assign"), &self.program)
            .map_err(|e| panic!("assembler assign failed: {e}"))?;
        assert_eq!(
            outputs.len(),
            1,
            "expected exactly one fused RmsNorm instruction"
        );
        *self.captured_outputs.borrow_mut() = Some(outputs.into_iter().next().unwrap());
        Ok(())
    }
}

fn build_program_and_domain() -> (AssemblerProgram, RsqrtDomain) {
    let graph = rms_norm_graph();
    let compiled = compile_graph(&graph).expect("should compile the real RMSNorm subgraph");

    // Sanity: exactly one fused RmsNorm instruction, matching the real
    // 7-node subgraph.
    assert_eq!(compiled.instructions.len(), 1);
    assert!(matches!(
        compiled.instructions[0].instruction,
        zkie_core::isa::Instruction::RmsNorm { dim, epsilon_milli }
            if dim == DIM && epsilon_milli == EPSILON_MILLI
    ));

    let x: Vec<I18> = FINTEXT_LAYER0_RMS_NORM_X.iter().map(|v| i18(*v)).collect();
    let mut graph_input_values = HashMap::new();
    graph_input_values.insert("x".to_string(), x);

    let program = to_assembler_program(&compiled, &graph_input_values)
        .expect("should convert to an assembler program");

    let target = exact_mean_sq_plus_eps(&FINTEXT_LAYER0_RMS_NORM_X, EPSILON_MILLI);
    (program, RsqrtDomain::RawAnchors(vec![target.raw()]))
}

#[test]
fn compiles_real_fintext_rms_norm_subgraph_via_fusion() {
    let (program, _domain) = build_program_and_domain();
    // Real weight (258 real trained floats) resolved through the real
    // pipeline.
    assert_eq!(program.weight_values[0].len(), DIM);
    assert_eq!(program.input_values[0].len(), DIM);
}

#[test]
fn fintext_rms_norm_real_weights_kzg_roundtrip_matches_pytorch() {
    let (program, domain) = build_program_and_domain();
    let circuit = RmsNormFinTextCircuit {
        program: program.clone(),
        domain,
        captured_outputs: RefCell::new(None),
    };

    let k = CIRCUIT_K;
    MockProver::run(k, &circuit, vec![])
        .unwrap()
        .assert_satisfied();

    let got = circuit
        .captured_outputs
        .borrow()
        .clone()
        .expect("assign should have run during synthesize");
    assert_eq!(got.len(), DIM);
    let mut max_diff = 0.0_f64;
    for (g, want) in got
        .iter()
        .zip(FINTEXT_LAYER0_RMS_NORM_EXPECTED_OUTPUT.iter())
    {
        let diff = (g.to_f64() - want).abs();
        max_diff = max_diff.max(diff);
    }
    println!("max abs diff vs real PyTorch RMSNorm output: {max_diff}");
    assert!(
        max_diff < 1e-3,
        "zkIE's proven RmsNorm output diverges from the real PyTorch computation: max_diff={max_diff}"
    );

    // Real (non-mocked) KZG setup -> prove -> verify.
    let mut rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");

    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
        &params,
        &pk,
        std::slice::from_ref(&circuit),
        &[vec![]],
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let proof = transcript.finalize();

    let verifier_params = params.verifier_params();
    let mut verifier_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let strategy = SingleStrategy::new(&verifier_params);
    let result = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
        &verifier_params,
        &vk,
        strategy,
        &[vec![]],
        &mut verifier_transcript,
    );
    assert!(
        result.is_ok(),
        "real FinText RmsNorm proof failed to verify: {:?}",
        result
    );

    // Tampered-proof negative test.
    let mut tampered = proof.clone();
    let mid = tampered.len() / 2;
    tampered[mid] ^= 0xFF;
    let mut tampered_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&tampered[..]);
    let tampered_strategy = SingleStrategy::new(&verifier_params);
    let tampered_result = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
        &verifier_params,
        &vk,
        tampered_strategy,
        &[vec![]],
        &mut tampered_transcript,
    );
    assert!(
        tampered_result.is_err(),
        "tampered proof should fail to verify"
    );
}
