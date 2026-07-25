use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::dev::MockProver;
use halo2_proofs::halo2curves::bn256::{Bn256, G1Affine};
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column, ConstraintSystem,
    ErrorFront,
};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use rand_core::OsRng;
use zkie_core::chips::embed_lookup::{EmbedLookupChip, EmbedLookupConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

const TABLE_SIZE: usize = 4;
const EMBED_DIM: usize = 3;

/// A small embedding table with a deliberately non-zero row 0, so this
/// roundtrip actually exercises the `lookup.rs` selector-gating fix (rows
/// outside the one `assign` touches default to (0, 0), which must not be
/// mistaken for a valid table entry for a real, non-zero row 0).
fn sample_table() -> Vec<Vec<I18>> {
    (0..TABLE_SIZE)
        .map(|i| {
            (0..EMBED_DIM)
                .map(|j| I18::from_f64(((i * EMBED_DIM + j) as f64 + 1.0) * 0.01).unwrap())
                .collect()
        })
        .collect()
}

#[derive(Clone)]
struct EmbedLookupCircuitConfig {
    embed: EmbedLookupConfig,
}

struct EmbedLookupCircuit {
    table: Vec<Vec<I18>>,
    index: usize,
}

impl Circuit<Fr> for EmbedLookupCircuit {
    type Config = EmbedLookupCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        EmbedLookupCircuit {
            table: self.table.clone(),
            index: 0,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let index_col = meta.advice_column();
        let output_cols: Vec<Column<Advice>> =
            (0..EMBED_DIM).map(|_| meta.advice_column()).collect();
        EmbedLookupCircuitConfig {
            embed: EmbedLookupChip::configure(meta, index_col, &output_cols, TABLE_SIZE, EMBED_DIM),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = EmbedLookupChip::construct(config.embed);
        chip.load_table(layouter.namespace(|| "table"), &self.table)
            .map_err(|e| panic!("embed lookup load_table failed: {e}"))
            .unwrap();
        chip.assign(layouter.namespace(|| "assign"), self.index, &self.table)
            .map(|_| ())
            .map_err(|e| panic!("embed lookup assign failed: {e}"))
    }
}

fn sample_circuit() -> EmbedLookupCircuit {
    EmbedLookupCircuit {
        table: sample_table(),
        index: 2,
    }
}

#[test]
fn embed_lookup_real_kzg_roundtrip() {
    let k = 8;
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
        "embed lookup proof failed to verify: {:?}",
        result
    );
}

#[test]
fn embed_lookup_tampered_proof_fails_verification() {
    let k = 8;
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
        "tampered embed lookup proof should fail verification"
    );
}
