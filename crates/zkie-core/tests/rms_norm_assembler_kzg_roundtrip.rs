//! Real (non-mocked) KZG setup -> prove -> verify roundtrip for
//! `AssemblerChip`'s `Instruction::RmsNorm` dispatch -- the new instruction
//! this session's smaller-TimesFM investigation added so a real TimesFM-
//! architecture RMSNorm subgraph (`Pow`/`ReduceMean`/`Add`/`Sqrt`/
//! `Reciprocal`/`Mul`/`Mul`, fused by `zkie_compiler::graph_compiler`'s new
//! RMSNorm-pattern recognition) can be compiled and proven end-to-end. See
//! `docs/superpowers/specs/2026-07-26-zkie-smaller-timesfm-attempt.md`.
//!
//! This test uses small hand-picked numbers (not yet the real FinText
//! checkpoint's weights -- that's covered by the separate
//! `fintext_rms_norm_real_weights` integration test) chosen so the exact
//! `mean(x^2) + epsilon` value lands on the `rsqrt` lookup domain's exact
//! quantization grid -- see `zkie_core::chips::rms_norm`'s "CRITICAL NUMERIC
//! LIMITATION" docs on why this is necessary.

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
use zkie_core::assembler::{
    AssemblerChip, AssemblerConfig, AssemblerInstruction, AssemblerProgram, RegisterRef,
};
use zkie_core::chips::layer_norm::RsqrtDomain;
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;
use zkie_core::isa::Instruction;

const CIRCUIT_K: u32 = 14;
const DIM: usize = 4;
const EPSILON_MILLI: u64 = 10;
// mean(x^2) for xs() below is exactly 1.25, so mean(x^2)+eps = 1.26 exactly
// -- the same domain-hit construction `chips::rms_norm`'s own tests use.
const RSQRT_DOMAIN_MIN: f64 = 0.1;
const RSQRT_DOMAIN_MAX: f64 = 3.0;
const RSQRT_DOMAIN_N: usize = 101;

fn i18(v: f64) -> I18 {
    I18::from_f64(v).unwrap()
}

fn xs() -> Vec<I18> {
    vec![i18(-1.5), i18(-0.5), i18(0.5), i18(1.5)]
}

fn weight() -> Vec<I18> {
    vec![i18(1.0), i18(2.0), i18(0.5), i18(-1.0)]
}

fn rms_norm_instructions() -> Vec<AssemblerInstruction> {
    vec![AssemblerInstruction {
        instruction: Instruction::RmsNorm {
            dim: DIM,
            epsilon_milli: EPSILON_MILLI,
        },
        inputs: vec![RegisterRef::Input(0), RegisterRef::Weight(0)],
    }]
}

fn rms_norm_program() -> AssemblerProgram {
    AssemblerProgram {
        instructions: rms_norm_instructions(),
        input_values: vec![xs()],
        weight_values: vec![weight()],
    }
}

fn domains() -> HashMap<(usize, u64), RsqrtDomain> {
    let mut m = HashMap::new();
    m.insert(
        (DIM, EPSILON_MILLI),
        RsqrtDomain::Range {
            min: RSQRT_DOMAIN_MIN,
            max: RSQRT_DOMAIN_MAX,
            n: RSQRT_DOMAIN_N,
        },
    );
    m
}

#[derive(Clone)]
struct RmsNormCircuit {
    program: AssemblerProgram,
}

impl Circuit<Fr> for RmsNormCircuit {
    type Config = AssemblerConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        RmsNormCircuit {
            program: self.program.clone(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        AssemblerChip::configure_with_rms_norm_domains(meta, &rms_norm_instructions(), &domains())
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = AssemblerChip::construct(config);
        chip.load_rms_norm_tables(layouter.namespace(|| "load rms norm tables"))?;
        chip.assign(layouter.namespace(|| "assign"), &self.program)
            .map(|_| ())
            .map_err(|e| panic!("assembler assign failed: {e}"))
    }
}

#[test]
fn rms_norm_real_kzg_roundtrip() {
    let k = CIRCUIT_K;
    let mut rng = OsRng;
    let circuit = RmsNormCircuit {
        program: rms_norm_program(),
    };

    MockProver::run(k, &circuit, vec![])
        .unwrap()
        .assert_satisfied();

    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");

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
    assert!(
        result.is_ok(),
        "rms norm assembler proof failed to verify: {:?}",
        result
    );
}

#[test]
fn rms_norm_tampered_proof_fails_verification() {
    let k = CIRCUIT_K;
    let mut rng = OsRng;
    let circuit = RmsNormCircuit {
        program: rms_norm_program(),
    };

    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");

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
    let mut proof = transcript.finalize();

    // Flip a byte in the middle of the proof.
    let mid = proof.len() / 2;
    proof[mid] ^= 0xFF;

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
    assert!(
        result.is_err(),
        "tampered rms norm proof should fail to verify"
    );
}
