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
use zkie_core::chips::patch_embed::{PatchEmbedChip, PatchEmbedConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

const PATCH_LEN: usize = 3;
const EMBED_DIM: usize = 2;

#[derive(Clone)]
struct PatchEmbedCircuitConfig {
    embed: PatchEmbedConfig,
}

struct PatchEmbedCircuit {
    patch: Vec<I18>,
    weights: Vec<Vec<I18>>,
}

impl Circuit<Fr> for PatchEmbedCircuit {
    type Config = PatchEmbedCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        PatchEmbedCircuit {
            patch: vec![I18::from_raw(0); PATCH_LEN],
            weights: vec![vec![I18::from_raw(0); PATCH_LEN]; EMBED_DIM],
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
        PatchEmbedCircuitConfig {
            embed: PatchEmbedChip::configure(
                meta,
                a,
                b,
                accumulator,
                q,
                r,
                slack,
                bits,
                PATCH_LEN,
                EMBED_DIM,
            ),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        PatchEmbedChip::construct(config.embed)
            .assign(layouter, &self.patch, &self.weights)
            .map(|_| ())
            .map_err(|e| panic!("patch embed assign failed: {e}"))
    }
}

fn sample_circuit() -> PatchEmbedCircuit {
    PatchEmbedCircuit {
        patch: vec![
            I18::from_f64(1.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
        ],
        weights: vec![
            vec![
                I18::from_f64(1.0).unwrap(),
                I18::from_f64(0.0).unwrap(),
                I18::from_f64(0.0).unwrap(),
            ],
            vec![
                I18::from_f64(0.5).unwrap(),
                I18::from_f64(0.5).unwrap(),
                I18::from_f64(1.0).unwrap(),
            ],
        ],
    }
}

#[test]
fn patch_embed_real_kzg_roundtrip() {
    let k = 10;
    let mut rng = OsRng;
    let circuit = sample_circuit();

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
        "patch embed proof failed to verify: {:?}",
        result
    );
}

#[test]
fn patch_embed_tampered_proof_fails_verification() {
    let k = 10;
    let mut rng = OsRng;
    let circuit = sample_circuit();

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
        "tampered patch embed proof should fail verification"
    );
}
