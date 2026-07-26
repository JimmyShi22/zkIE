//! zkie-evm-verifier-poc
//!
//! Standalone, decoupled proof-of-concept for Approach 2 of the zkIE EVM-verifier
//! attempt (see docs/superpowers/specs/2026-07-26-zkie-evm-verifier-attempt.md).
//!
//! This crate is pinned to `halo2_proofs` git tag `v0.3.0` (NOT the `v0.4.0` used
//! by zkie-core/zkie-compiler) because that is the only version
//! `halo2-solidity-verifier`'s main branch can build against. It reimplements a
//! tiny toy circuit (a single multiplication gate, ported from halo2's own
//! `halo2_proofs/examples/simple-example.rs`) from scratch -- it does NOT reuse
//! any of zkie-core's real chips, which remain pinned to v0.4.0 and untouched.
//!
//! What this program does:
//! 1. Builds the toy circuit over the BN256 scalar field, runs MockProver as a
//!    sanity check.
//! 2. Runs a real KZG trusted setup + keygen + `create_proof` using a Keccak256
//!    transcript (matching what the EVM verifier expects for Fiat-Shamir).
//! 3. Verifies the proof off-chain with halo2's own `verify_proof` as a sanity
//!    check before touching Solidity/EVM.
//! 4. Uses halo2-solidity-verifier's `SolidityGenerator` to render
//!    `Halo2Verifier.sol` with the verifying key embedded.
//! 5. Uses halo2-solidity-verifier's `encode_calldata` helper to build the raw
//!    calldata for `verifyProof(bytes,uint256[])` for both a valid proof and a
//!    tampered (one byte flipped) proof, and writes both to disk as 0x-prefixed
//!    hex so a shell script can `cast send`/`cast call` them directly against
//!    Anvil.
//!
//! Outputs are written under `./out/` (relative to this crate's directory):
//!   - out/Halo2Verifier.sol
//!   - out/valid_calldata.hex
//!   - out/tampered_calldata.hex
//!   - out/instances.json (human-readable, for the report)

use std::fs;
use std::marker::PhantomData;
use std::path::Path;

use halo2_proofs::arithmetic::Field;
use halo2_proofs::circuit::{AssignedCell, Chip, Layouter, Region, SimpleFloorPlanner, Value};
use halo2_proofs::halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column, ConstraintSystem,
    Error, Fixed, Instance, Selector,
};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::poly::Rotation;
use halo2_proofs::transcript::{TranscriptReadBuffer, TranscriptWriterBuffer};
use halo2_solidity_verifier::{encode_calldata, BatchOpenScheme::Bdfg21, Keccak256Transcript, SolidityGenerator};
use rand_core::OsRng;

// ---------------------------------------------------------------------------
// Toy circuit: a single multiplication gate, ported from halo2's own
// `halo2_proofs/examples/simple-example.rs` (tag v0.3.0), generalized over the
// BN256 scalar field instead of pasta::Fp.
// ---------------------------------------------------------------------------

trait NumericInstructions<F: Field>: Chip<F> {
    type Num;
    fn load_private(&self, layouter: impl Layouter<F>, a: Value<F>) -> Result<Self::Num, Error>;
    fn load_constant(&self, layouter: impl Layouter<F>, constant: F) -> Result<Self::Num, Error>;
    fn mul(&self, layouter: impl Layouter<F>, a: Self::Num, b: Self::Num) -> Result<Self::Num, Error>;
    fn expose_public(&self, layouter: impl Layouter<F>, num: Self::Num, row: usize) -> Result<(), Error>;
}

struct FieldChip<F: Field> {
    config: FieldConfig,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
struct FieldConfig {
    advice: [Column<Advice>; 2],
    instance: Column<Instance>,
    s_mul: Selector,
}

impl<F: Field> FieldChip<F> {
    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 2],
        instance: Column<Instance>,
        constant: Column<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        meta.enable_equality(instance);
        meta.enable_constant(constant);
        for column in &advice {
            meta.enable_equality(*column);
        }
        let s_mul = meta.selector();

        meta.create_gate("mul", |meta| {
            let lhs = meta.query_advice(advice[0], Rotation::cur());
            let rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[0], Rotation::next());
            let s_mul = meta.query_selector(s_mul);
            vec![s_mul * (lhs * rhs - out)]
        });

        FieldConfig {
            advice,
            instance,
            s_mul,
        }
    }
}

impl<F: Field> Chip<F> for FieldChip<F> {
    type Config = FieldConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

#[derive(Clone)]
struct Number<F: Field>(AssignedCell<F, F>);

impl<F: Field> NumericInstructions<F> for FieldChip<F> {
    type Num = Number<F>;

    fn load_private(&self, mut layouter: impl Layouter<F>, value: Value<F>) -> Result<Self::Num, Error> {
        let config = self.config();
        layouter.assign_region(
            || "load private",
            |mut region| {
                region
                    .assign_advice(|| "private input", config.advice[0], 0, || value)
                    .map(Number)
            },
        )
    }

    fn load_constant(&self, mut layouter: impl Layouter<F>, constant: F) -> Result<Self::Num, Error> {
        let config = self.config();
        layouter.assign_region(
            || "load constant",
            |mut region| {
                region
                    .assign_advice_from_constant(|| "constant value", config.advice[0], 0, constant)
                    .map(Number)
            },
        )
    }

    fn mul(&self, mut layouter: impl Layouter<F>, a: Self::Num, b: Self::Num) -> Result<Self::Num, Error> {
        let config = self.config();
        layouter.assign_region(
            || "mul",
            |mut region: Region<'_, F>| {
                config.s_mul.enable(&mut region, 0)?;
                a.0.copy_advice(|| "lhs", &mut region, config.advice[0], 0)?;
                b.0.copy_advice(|| "rhs", &mut region, config.advice[1], 0)?;
                let value = a.0.value().copied() * b.0.value();
                region
                    .assign_advice(|| "lhs * rhs", config.advice[0], 1, || value)
                    .map(Number)
            },
        )
    }

    fn expose_public(&self, mut layouter: impl Layouter<F>, num: Self::Num, row: usize) -> Result<(), Error> {
        let config = self.config();
        layouter.constrain_instance(num.0.cell(), config.instance, row)
    }
}

#[derive(Default)]
struct MyCircuit<F: Field> {
    constant: F,
    a: Value<F>,
    b: Value<F>,
}

impl<F: Field> Circuit<F> for MyCircuit<F> {
    type Config = FieldConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let advice = [meta.advice_column(), meta.advice_column()];
        let instance = meta.instance_column();
        let constant = meta.fixed_column();
        FieldChip::configure(meta, advice, instance, constant)
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let field_chip = FieldChip::<F>::construct(config);
        let a = field_chip.load_private(layouter.namespace(|| "load a"), self.a)?;
        let b = field_chip.load_private(layouter.namespace(|| "load b"), self.b)?;
        let constant = field_chip.load_constant(layouter.namespace(|| "load constant"), self.constant)?;
        let ab = field_chip.mul(layouter.namespace(|| "a * b"), a, b)?;
        let absq = field_chip.mul(layouter.namespace(|| "ab * ab"), ab.clone(), ab)?;
        let c = field_chip.mul(layouter.namespace(|| "constant * absq"), constant, absq)?;
        field_chip.expose_public(layouter.namespace(|| "expose c"), c, 0)
    }
}

fn main() {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    fs::create_dir_all(&out_dir).expect("failed to create out dir");

    // Same numbers as halo2's own simple-example.rs, over the BN256 scalar field.
    let k = 4;
    let constant = Fr::from(7u64);
    let a = Fr::from(2u64);
    let b = Fr::from(3u64);
    let c = constant * a.square() * b.square();

    println!("=== zkie-evm-verifier-poc: Approach 2 (standalone v0.3.0 toy circuit) ===");
    println!("k = {k}, constant = 7, a = 2, b = 3, expected c = constant*a^2*b^2 = {c:?}");

    let circuit = MyCircuit {
        constant,
        a: Value::known(a),
        b: Value::known(b),
    };
    let instances = vec![c];

    // 1. MockProver sanity check.
    {
        use halo2_proofs::dev::MockProver;
        let prover = MockProver::run(k, &circuit, vec![instances.clone()]).unwrap();
        prover.verify().expect("MockProver: circuit should be satisfied");
        println!("[ok] MockProver::verify() passed");
    }

    // 2. Real KZG setup + keygen + create_proof (Keccak256 transcript, matching
    //    what the Solidity verifier expects).
    let mut rng = OsRng;
    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");
    println!("[ok] keygen_vk / keygen_pk succeeded");

    let proof = {
        let mut transcript =
            <Keccak256Transcript<G1Affine, Vec<u8>> as TranscriptWriterBuffer<_, _, _>>::init(Vec::new());
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[&instances]],
            &mut rng,
            &mut transcript,
        )
        .expect("create_proof failed");
        transcript.finalize()
    };
    println!("[ok] create_proof succeeded, proof length = {} bytes", proof.len());

    // 3. Off-chain sanity check with halo2's own verifier before touching Solidity.
    {
        let mut transcript =
            <Keccak256Transcript<G1Affine, _> as TranscriptReadBuffer<_, _, _>>::init(proof.as_slice());
        let strategy = SingleStrategy::new(&params);
        let result = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
            &params,
            pk.get_vk(),
            strategy,
            &[&[&instances]],
            &mut transcript,
        );
        assert!(result.is_ok(), "off-chain verify_proof failed: {result:?}");
        println!("[ok] off-chain verify_proof (halo2 native) passed");
    }

    // 4. Generate the Solidity verifier (VK embedded).
    let generator = SolidityGenerator::new(&params, &vk, Bdfg21, instances.len());
    let verifier_solidity = generator.render().expect("solidity render failed");
    let sol_path = out_dir.join("Halo2Verifier.sol");
    fs::write(&sol_path, &verifier_solidity).expect("failed to write Halo2Verifier.sol");
    println!(
        "[ok] Solidity verifier rendered ({} bytes) -> {}",
        verifier_solidity.len(),
        sol_path.display()
    );

    // 5. Encode calldata for a valid proof and for a tampered proof.
    let valid_calldata = encode_calldata(None, &proof, &instances);
    let valid_hex = format!("0x{}", hex::encode(&valid_calldata));
    fs::write(out_dir.join("valid_calldata.hex"), &valid_hex).unwrap();
    println!(
        "[ok] valid calldata encoded ({} bytes) -> out/valid_calldata.hex",
        valid_calldata.len()
    );

    let mut tampered_proof = proof.clone();
    let mid = tampered_proof.len() / 2;
    tampered_proof[mid] ^= 0xFF;
    let tampered_calldata = encode_calldata(None, &tampered_proof, &instances);
    let tampered_hex = format!("0x{}", hex::encode(&tampered_calldata));
    fs::write(out_dir.join("tampered_calldata.hex"), &tampered_hex).unwrap();
    println!(
        "[ok] tampered calldata encoded ({} bytes) -> out/tampered_calldata.hex",
        tampered_calldata.len()
    );

    // Human-readable summary for the report / manual cast calls.
    let instances_json = format!(
        "{{\n  \"k\": {k},\n  \"constant\": 7,\n  \"a\": 2,\n  \"b\": 3,\n  \"c_instance\": \"{c:?}\",\n  \"proof_len_bytes\": {},\n  \"num_instances\": {}\n}}\n",
        proof.len(),
        instances.len()
    );
    fs::write(out_dir.join("instances.json"), instances_json).unwrap();

    println!("=== done. Artifacts written under {} ===", out_dir.display());
}
