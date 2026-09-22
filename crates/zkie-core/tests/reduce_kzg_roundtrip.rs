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
use zkie_core::chips::reduce::{ReduceMeanChip, ReduceMeanConfig, ReduceSumChip, ReduceSumConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

const K_INPUTS: usize = 4;

#[derive(Clone)]
struct SumCircuitConfig {
    reduce: ReduceSumConfig,
}

struct SumCircuit {
    inputs: Vec<I18>,
}

impl Circuit<Fr> for SumCircuit {
    type Config = SumCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        SumCircuit {
            inputs: vec![I18::from_raw(0); self.inputs.len()],
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let values = meta.advice_column();
        let sum = meta.advice_column();
        let sum_shift = meta.advice_column();
        let bits = meta.advice_column();
        SumCircuitConfig {
            reduce: ReduceSumChip::configure(meta, values, sum, sum_shift, bits, K_INPUTS),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        ReduceSumChip::construct(config.reduce).assign(layouter, &self.inputs)?;
        Ok(())
    }
}

#[derive(Clone)]
struct MeanCircuitConfig {
    reduce: ReduceMeanConfig,
}

struct MeanCircuit {
    inputs: Vec<I18>,
}

impl Circuit<Fr> for MeanCircuit {
    type Config = MeanCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        MeanCircuit {
            inputs: vec![I18::from_raw(0); self.inputs.len()],
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let values = meta.advice_column();
        let sum = meta.advice_column();
        let sum_shift = meta.advice_column();
        let q = meta.advice_column();
        let r = meta.advice_column();
        let slack = meta.advice_column();
        let bits = meta.advice_column();
        MeanCircuitConfig {
            reduce: ReduceMeanChip::configure(
                meta, values, sum, sum_shift, q, r, slack, bits, K_INPUTS,
            ),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        ReduceMeanChip::construct(config.reduce).assign(layouter, &self.inputs)?;
        Ok(())
    }
}

fn sample_inputs() -> Vec<I18> {
    vec![
        I18::from_f64(2.0).unwrap(),
        I18::from_f64(-1.5).unwrap(),
        I18::from_f64(3.25).unwrap(),
        I18::from_f64(-0.75).unwrap(),
    ]
}

#[test]
fn reduce_sum_real_kzg_roundtrip() {
    let k = 12;
    let mut rng = OsRng;
    let circuit = SumCircuit {
        inputs: sample_inputs(),
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
        "reduce sum proof failed to verify: {:?}",
        result
    );
}

#[test]
fn reduce_sum_tampered_proof_fails_verification() {
    let k = 12;
    let mut rng = OsRng;
    let circuit = SumCircuit {
        inputs: sample_inputs(),
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
        "tampered reduce sum proof should fail verification"
    );
}

#[test]
fn reduce_mean_real_kzg_roundtrip() {
    let k = 12;
    let mut rng = OsRng;
    let circuit = MeanCircuit {
        inputs: sample_inputs(),
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
        "reduce mean proof failed to verify: {:?}",
        result
    );
}

#[test]
fn reduce_mean_tampered_proof_fails_verification() {
    let k = 12;
    let mut rng = OsRng;
    let circuit = MeanCircuit {
        inputs: sample_inputs(),
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
        "tampered reduce mean proof should fail verification"
    );
}
