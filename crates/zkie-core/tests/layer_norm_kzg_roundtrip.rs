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
use zkie_core::chips::layer_norm::{LayerNormChip, LayerNormConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

const K_INPUTS: usize = 4;
// epsilon = 10 / 1000 = 0.01.
const EPSILON_MILLI: u64 = 10;
// See `crates/zkie-core/src/chips/layer_norm.rs`'s module docs and its own
// test module for the full derivation: this domain keeps `rsqrt`'s output
// comfortably within I18's representable range, and its evenly-spaced grid
// includes x = 1.26 exactly (index 40) -- the exact `variance + epsilon`
// this test's chosen inputs produce.
const RSQRT_DOMAIN_MIN: f64 = 0.1;
const RSQRT_DOMAIN_MAX: f64 = 3.0;
const RSQRT_DOMAIN_N: usize = 101;

#[derive(Clone)]
struct LayerNormCircuitConfig {
    layer_norm: LayerNormConfig,
}

struct LayerNormCircuit {
    inputs: Vec<I18>,
}

impl Circuit<Fr> for LayerNormCircuit {
    type Config = LayerNormCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        LayerNormCircuit {
            inputs: vec![I18::from_raw(0); self.inputs.len()],
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let values = meta.advice_column();
        let sum = meta.advice_column();
        let sum_shift = meta.advice_column();
        let mean_q = meta.advice_column();
        let mean_r = meta.advice_column();
        let mean_slack = meta.advice_column();
        let add_a = meta.advice_column();
        let add_b = meta.advice_column();
        let add_c = meta.advice_column();
        let mul_a = meta.advice_column();
        let mul_b = meta.advice_column();
        let mul_q = meta.advice_column();
        let mul_r = meta.advice_column();
        let mul_slack = meta.advice_column();
        let bits = meta.advice_column();
        let rsqrt_input = meta.advice_column();
        let rsqrt_output = meta.advice_column();
        let mean_link = meta.advice_column();
        let neg_mean = meta.advice_column();
        let unshift_in = meta.advice_column();
        let unshift_out = meta.advice_column();

        LayerNormCircuitConfig {
            layer_norm: LayerNormChip::configure(
                meta,
                values,
                sum,
                sum_shift,
                mean_q,
                mean_r,
                mean_slack,
                add_a,
                add_b,
                add_c,
                mul_a,
                mul_b,
                mul_q,
                mul_r,
                mul_slack,
                bits,
                rsqrt_input,
                rsqrt_output,
                mean_link,
                neg_mean,
                unshift_in,
                unshift_out,
                K_INPUTS,
                EPSILON_MILLI,
                RSQRT_DOMAIN_MIN,
                RSQRT_DOMAIN_MAX,
                RSQRT_DOMAIN_N,
            ),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = LayerNormChip::construct(config.layer_norm);
        chip.load_table(layouter.namespace(|| "table"))?;
        chip.assign(layouter.namespace(|| "assign"), &self.inputs)
            .expect("layer norm assign should not fail in this test");
        Ok(())
    }
}

fn sample_inputs() -> Vec<I18> {
    vec![
        I18::from_f64(-1.5).unwrap(),
        I18::from_f64(-0.5).unwrap(),
        I18::from_f64(0.5).unwrap(),
        I18::from_f64(1.5).unwrap(),
    ]
}

#[test]
fn layer_norm_real_kzg_roundtrip() {
    let k = 12;
    let mut rng = OsRng;
    let circuit = LayerNormCircuit {
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
        "layer norm proof failed to verify: {:?}",
        result
    );
}

#[test]
fn layer_norm_tampered_proof_fails_verification() {
    let k = 12;
    let mut rng = OsRng;
    let circuit = LayerNormCircuit {
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
        "tampered layer norm proof should fail verification"
    );
}
