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
use zkie_core::chips::div::{DivChip, DivConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

#[derive(Clone)]
struct DivCircuitConfig {
    div: DivConfig,
}

struct DivCircuit {
    numerator: I18,
    divisor: I18,
}

impl Circuit<Fr> for DivCircuit {
    type Config = DivCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        DivCircuit {
            numerator: I18::from_raw(0),
            divisor: I18::from_raw(1),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let numerator = meta.advice_column();
        let divisor = meta.advice_column();
        let q_shift = meta.advice_column();
        let r = meta.advice_column();
        let slack = meta.advice_column();
        let dm1 = meta.advice_column();
        let bits = meta.advice_column();
        DivCircuitConfig {
            div: DivChip::configure(meta, numerator, divisor, q_shift, r, slack, dm1, bits),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        DivChip::construct(config.div)
            .assign(layouter, self.numerator, self.divisor)
            .map(|_| ())
            .map_err(|_| ErrorFront::Synthesis)
    }
}

#[test]
fn div_real_kzg_roundtrip() {
    let k = 11;
    let mut rng = OsRng;
    let circuit = DivCircuit {
        numerator: I18::from_f64(5.0).unwrap(),
        divisor: I18::from_f64(2.0).unwrap(),
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
    assert!(result.is_ok(), "div proof failed to verify: {:?}", result);
}

#[test]
fn div_tampered_proof_fails_verification() {
    let k = 11;
    let mut rng = OsRng;
    let circuit = DivCircuit {
        numerator: I18::from_f64(5.0).unwrap(),
        divisor: I18::from_f64(2.0).unwrap(),
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
        "tampered div proof should fail verification"
    );
}
