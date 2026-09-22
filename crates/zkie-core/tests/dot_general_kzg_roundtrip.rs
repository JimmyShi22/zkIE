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
use zkie_core::chips::dot_general::{DotProductChip, DotProductConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

const K: usize = 3;

#[derive(Clone)]
struct DotCircuitConfig {
    dot: DotProductConfig,
}

struct DotCircuit {
    a: Vec<I18>,
    b: Vec<I18>,
}

impl Circuit<Fr> for DotCircuit {
    type Config = DotCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        DotCircuit {
            a: vec![I18::from_raw(0); K],
            b: vec![I18::from_raw(0); K],
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
        DotCircuitConfig {
            dot: DotProductChip::configure(meta, a, b, accumulator, q, r, slack, bits, K),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        DotProductChip::construct(config.dot)
            .assign(layouter, self.a.clone(), self.b.clone())
            .map(|_| ())
            .map_err(|e| panic!("dot product assign failed: {e}"))
    }
}

#[test]
fn dot_general_real_kzg_roundtrip() {
    let k = 10;
    let mut rng = OsRng;
    let circuit = DotCircuit {
        a: vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-3.0).unwrap(),
            I18::from_f64(1.5).unwrap(),
        ],
        b: vec![
            I18::from_f64(3.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-4.0).unwrap(),
        ],
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
        "dot general proof failed to verify: {:?}",
        result
    );
}

#[test]
fn dot_general_tampered_proof_fails_verification() {
    let k = 10;
    let mut rng = OsRng;
    let circuit = DotCircuit {
        a: vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-3.0).unwrap(),
            I18::from_f64(1.5).unwrap(),
        ],
        b: vec![
            I18::from_f64(3.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-4.0).unwrap(),
        ],
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
        "tampered dot general proof should fail verification"
    );
}
