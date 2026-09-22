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
use zkie_core::chips::lookup::{build_domain, LookupChip, LookupConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

// Deliberately small domain (16 points over [0.0, 2.0], f(x) = x * x) to keep
// `k` reasonable for this roundtrip test.
fn small_square_domain() -> (Vec<I18>, Vec<I18>) {
    build_domain(|x| x * x, 0.0, 2.0, 16)
}

#[derive(Clone)]
struct LookupCircuitConfig {
    lookup: LookupConfig,
}

struct LookupCircuit {
    domain: Vec<I18>,
    values: Vec<I18>,
    input: I18,
}

impl Circuit<Fr> for LookupCircuit {
    type Config = LookupCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        LookupCircuit {
            domain: self.domain.clone(),
            values: self.values.clone(),
            input: I18::from_raw(0),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let input = meta.advice_column();
        let output = meta.advice_column();
        LookupCircuitConfig {
            lookup: LookupChip::configure(meta, input, output),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = LookupChip::construct(config.lookup, self.domain.clone(), self.values.clone());
        chip.load_table(layouter.namespace(|| "table"))?;
        chip.assign(layouter.namespace(|| "assign"), self.input)
            .expect("input should be an exact domain point in this test");
        Ok(())
    }
}

#[test]
fn lookup_real_kzg_roundtrip() {
    let k = 8;
    let mut rng = OsRng;
    let (domain, values) = small_square_domain();
    let circuit = LookupCircuit {
        input: domain[5],
        domain,
        values,
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
        "lookup proof failed to verify: {:?}",
        result
    );
}

#[test]
fn lookup_tampered_proof_fails_verification() {
    let k = 8;
    let mut rng = OsRng;
    let (domain, values) = small_square_domain();
    let circuit = LookupCircuit {
        input: domain[5],
        domain,
        values,
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
        "tampered lookup proof should fail verification"
    );
}
