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
use zkie_core::chips::gelu::GeluChip;
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

// Deliberately small domain (33 points over [-6.0, 6.0]) to keep `k`
// reasonable for this roundtrip test. 33 (odd) guarantees x = 0.0 is an
// exact grid point, which matters because `LookupChip`'s lookup argument
// constrains every row of the input/output columns, including rows
// MockProver/the real prover default to (0, 0); since gelu(0.0) == 0.0
// exactly, (0, 0) is a valid table row and those default rows are free.
// See `crates/zkie-core/src/chips/gelu.rs`'s test module for the full
// explanation.
const DOMAIN_MIN: f64 = -6.0;
const DOMAIN_MAX: f64 = 6.0;
const DOMAIN_N: usize = 33;

#[derive(Clone)]
struct GeluCircuitConfig {
    gelu: zkie_core::chips::gelu::GeluConfig,
}

struct GeluCircuit {
    input: I18,
}

impl Circuit<Fr> for GeluCircuit {
    type Config = GeluCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        GeluCircuit {
            input: I18::from_raw(0),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let input = meta.advice_column();
        let output = meta.advice_column();
        GeluCircuitConfig {
            gelu: GeluChip::configure(meta, input, output),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = GeluChip::construct(config.gelu, DOMAIN_MIN, DOMAIN_MAX, DOMAIN_N);
        chip.load_table(layouter.namespace(|| "table"))?;
        chip.assign(layouter.namespace(|| "assign"), self.input)
            .expect("input should be an exact domain point in this test");
        Ok(())
    }
}

fn sample_input() -> I18 {
    // Same function `GeluChip::construct` uses internally to build its
    // table, so this is guaranteed to be an exact domain point.
    let (domain, _values) = zkie_core::chips::lookup::build_domain(
        zkie_core::chips::gelu::gelu_f64,
        DOMAIN_MIN,
        DOMAIN_MAX,
        DOMAIN_N,
    );
    domain[20]
}

#[test]
fn gelu_real_kzg_roundtrip() {
    let k = 7;
    let mut rng = OsRng;
    let circuit = GeluCircuit {
        input: sample_input(),
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
    assert!(result.is_ok(), "gelu proof failed to verify: {:?}", result);
}

#[test]
fn gelu_tampered_proof_fails_verification() {
    let k = 7;
    let mut rng = OsRng;
    let circuit = GeluCircuit {
        input: sample_input(),
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
        "tampered gelu proof should fail verification"
    );
}
