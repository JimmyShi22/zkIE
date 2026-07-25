//! Real (non-mocked) KZG setup -> prove -> verify roundtrip for
//! `AssemblerChip`, on the project's primary end-to-end target: the
//! linear-layer program `y = MatMul(x, W); z = Add(y, b)` (structurally
//! identical to `zkie_compiler::graph_compiler`'s
//! `compiles_matmul_plus_bias_linear_layer_end_to_end` test, and to a
//! transformer projection layer). This is the "does the whole
//! ONNX -> compile -> circuit -> real proof -> verify pipeline work"
//! milestone `docs/superpowers/specs/2026-07-26-zkie-subproject4-scope-decision.md`
//! names as this sub-project's circuit-assembler deliverable.

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
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;
use zkie_core::isa::{EltwiseOp, Instruction};

const CIRCUIT_K: u32 = 12;

fn i18(v: f64) -> I18 {
    I18::from_f64(v).unwrap()
}

fn linear_layer_instructions() -> Vec<AssemblerInstruction> {
    vec![
        AssemblerInstruction {
            instruction: Instruction::DotGeneral {
                m: 2,
                n: 2,
                k: 2,
                batch_dims: vec![],
                trans_a: false,
                trans_b: false,
            },
            inputs: vec![RegisterRef::Input(0), RegisterRef::Weight(0)],
        },
        AssemblerInstruction {
            instruction: Instruction::Eltwise { op: EltwiseOp::Add },
            inputs: vec![RegisterRef::Virtual(0), RegisterRef::Weight(1)],
        },
    ]
}

/// Small values (see `zkie_core::assembler`'s own test module docs) so every
/// dot-product accumulation and the final biased sum stay comfortably within
/// I18's representable range.
fn linear_layer_program() -> AssemblerProgram {
    let x = vec![i18(0.5), i18(-0.25), i18(0.25), i18(0.5)];
    let w = vec![i18(1.0), i18(0.5), i18(-0.5), i18(1.0)];
    let b = vec![i18(0.1), i18(-0.2)];

    AssemblerProgram {
        instructions: linear_layer_instructions(),
        input_values: vec![x],
        weight_values: vec![w, b],
    }
}

#[derive(Clone)]
struct LinearLayerCircuit {
    program: AssemblerProgram,
}

impl Circuit<Fr> for LinearLayerCircuit {
    type Config = AssemblerConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        LinearLayerCircuit {
            program: self.program.clone(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        AssemblerChip::configure(meta, &linear_layer_instructions())
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        AssemblerChip::construct(config)
            .assign(layouter, &self.program)
            .map(|_| ())
            .map_err(|e| panic!("assembler assign failed: {e}"))
    }
}

#[test]
fn linear_layer_real_kzg_roundtrip() {
    let k = CIRCUIT_K;
    let mut rng = OsRng;
    let circuit = LinearLayerCircuit {
        program: linear_layer_program(),
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
        "linear layer assembler proof failed to verify: {:?}",
        result
    );
}

#[test]
fn linear_layer_tampered_proof_fails_verification() {
    let k = CIRCUIT_K;
    let mut rng = OsRng;
    let circuit = LinearLayerCircuit {
        program: linear_layer_program(),
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
        "tampered linear layer assembler proof should fail verification"
    );
}
