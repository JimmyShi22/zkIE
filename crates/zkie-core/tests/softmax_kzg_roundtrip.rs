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
use zkie_core::chips::softmax::{SoftmaxChip, SoftmaxConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

// See the module-level numeric-limitation doc on `SoftmaxChip`: exp over
// [-4.0, 0.0] is bounded by [~0.0183, 1.0], so summing even K=8 such values
// stays well under I18's ~+/-9.22 representable range. K=4 here tops out at
// a worst-case sum of ~4.0.
const EXP_DOMAIN_MIN: f64 = -4.0;
const EXP_DOMAIN_MAX: f64 = 0.0;
const EXP_DOMAIN_N: usize = 33;
const K: usize = 4;

#[derive(Clone)]
struct SoftmaxCircuitConfig {
    softmax: SoftmaxConfig,
}

struct SoftmaxCircuit {
    inputs: Vec<I18>,
}

impl Circuit<Fr> for SoftmaxCircuit {
    type Config = SoftmaxCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        // 0.0 is the domain's upper bound (an exact grid point), so this
        // remains a valid input during keygen's `without_witnesses` pass.
        SoftmaxCircuit {
            inputs: vec![I18::from_raw(0); self.inputs.len()],
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let exp_input = meta.advice_column();
        let exp_output = meta.advice_column();
        let sum_values = meta.advice_column();
        let sum = meta.advice_column();
        let sum_shift = meta.advice_column();
        let div_numerator = meta.advice_column();
        let div_divisor = meta.advice_column();
        let div_q_shift = meta.advice_column();
        let div_r = meta.advice_column();
        let div_slack = meta.advice_column();
        let div_dm1 = meta.advice_column();
        let bits = meta.advice_column();
        SoftmaxCircuitConfig {
            softmax: SoftmaxChip::configure(
                meta,
                exp_input,
                exp_output,
                sum_values,
                sum,
                sum_shift,
                div_numerator,
                div_divisor,
                div_q_shift,
                div_r,
                div_slack,
                div_dm1,
                bits,
                K,
            ),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip =
            SoftmaxChip::construct(config.softmax, EXP_DOMAIN_MIN, EXP_DOMAIN_MAX, EXP_DOMAIN_N);
        chip.load_table(layouter.namespace(|| "table"))?;
        chip.assign(layouter.namespace(|| "assign"), &self.inputs)
            .expect("inputs should be exact domain points in this test");
        Ok(())
    }
}

fn sample_inputs() -> Vec<I18> {
    // Same function `SoftmaxChip::construct` uses internally to build its
    // exp table, so these are guaranteed to be exact domain points.
    let (domain, _values) = zkie_core::chips::lookup::build_domain(
        f64::exp,
        EXP_DOMAIN_MIN,
        EXP_DOMAIN_MAX,
        EXP_DOMAIN_N,
    );
    let indices = [0usize, 12, 20, 32];
    indices.iter().map(|&i| domain[i]).collect()
}

#[test]
fn softmax_real_kzg_roundtrip() {
    let k = 14;
    let mut rng = OsRng;
    let circuit = SoftmaxCircuit {
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
        "softmax proof failed to verify: {:?}",
        result
    );
}

#[test]
fn softmax_tampered_proof_fails_verification() {
    let k = 14;
    let mut rng = OsRng;
    let circuit = SoftmaxCircuit {
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
        "tampered softmax proof should fail verification"
    );
}
