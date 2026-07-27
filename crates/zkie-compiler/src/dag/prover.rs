//! A pluggable prover for one `Shard`, plus the only implementation used in
//! this sub-project: `MockProver`, which computes real BLAKE3 commitments
//! over witness values but never generates an actual proof (`valid` is
//! always `true`). See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`
//! section 0 for why a real STARK-backed `Prover` is out of scope here --
//! this exists so `Linker`'s commitment-matching logic (the part of this
//! sub-project that must be genuinely correct) can be tested end-to-end
//! today, without waiting on a real STARK backend. A future real `Prover`
//! implementation is a drop-in replacement: `Dag`/`Linker` never change.

use std::collections::HashMap;

use crate::graph_compiler::Register;
use zkie_core::fixed_point::I18;

use super::model::Shard;

/// A binding commitment to a register's concrete witness value(s) -- a
/// BLAKE3 hash of its `I18` raw `i64` values, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Commitment(pub [u8; 32]);

impl Commitment {
    pub fn of_i18_values(values: &[I18]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for value in values {
            hasher.update(&value.raw().to_le_bytes());
        }
        Commitment(*hasher.finalize().as_bytes())
    }
}

/// The concrete `I18` values for every register a shard's `prove` call
/// needs -- in this sub-project, supplied directly by the caller (there is
/// no plaintext ONNX interpreter yet; see the design doc for why building
/// one is out of scope here). A real value must be present for every
/// register in `shard.inputs` and `shard.outputs`.
pub type Witness = HashMap<Register, Vec<I18>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardProof {
    pub shard_id: usize,
    pub input_commitments: HashMap<Register, Commitment>,
    pub output_commitments: HashMap<Register, Commitment>,
    /// Whether the shard's own proof is valid. Always `true` for
    /// `MockProver`; a real STARK-backed `Prover` would set this from
    /// actual proof verification.
    pub valid: bool,
}

/// Proves (or, for now, mocks proving) one `Shard`. Pluggable so a real
/// STARK-backed implementation can later replace `MockProver` without any
/// change to `Dag`/`Linker`.
pub trait Prover {
    fn prove(&self, shard: &Shard, witness: &Witness) -> ShardProof;
}

/// Computes real commitments over the supplied witness values, but never
/// generates an actual proof -- `valid` is always `true`. See module docs.
pub struct MockProver;

impl Prover for MockProver {
    fn prove(&self, shard: &Shard, witness: &Witness) -> ShardProof {
        let commit_all = |registers: &[Register]| -> HashMap<Register, Commitment> {
            registers
                .iter()
                .map(|register| {
                    let values = witness
                        .get(register)
                        .unwrap_or_else(|| panic!("missing witness value for {register:?}"));
                    (register.clone(), Commitment::of_i18_values(values))
                })
                .collect()
        };

        ShardProof {
            shard_id: shard.id,
            input_commitments: commit_all(&shard.inputs),
            output_commitments: commit_all(&shard.outputs),
            valid: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_is_deterministic_and_content_sensitive() {
        let a =
            Commitment::of_i18_values(&[I18::from_f64(1.0).unwrap(), I18::from_f64(2.0).unwrap()]);
        let a_again =
            Commitment::of_i18_values(&[I18::from_f64(1.0).unwrap(), I18::from_f64(2.0).unwrap()]);
        let b =
            Commitment::of_i18_values(&[I18::from_f64(1.0).unwrap(), I18::from_f64(3.0).unwrap()]);

        assert_eq!(a, a_again);
        assert_ne!(a, b);
    }

    #[test]
    fn mock_prover_commits_declared_inputs_and_outputs() {
        let shard = Shard {
            id: 0,
            name: "s0".into(),
            range: 0..1,
            inputs: vec![Register::GraphInput("x".into())],
            outputs: vec![Register::Virtual(0)],
        };
        let mut witness: Witness = HashMap::new();
        witness.insert(
            Register::GraphInput("x".into()),
            vec![I18::from_f64(1.0).unwrap()],
        );
        witness.insert(Register::Virtual(0), vec![I18::from_f64(2.0).unwrap()]);

        let proof = MockProver.prove(&shard, &witness);

        assert!(proof.valid);
        assert_eq!(proof.shard_id, 0);
        assert!(proof
            .input_commitments
            .contains_key(&Register::GraphInput("x".into())));
        assert!(proof.output_commitments.contains_key(&Register::Virtual(0)));
    }

    #[test]
    #[should_panic(expected = "missing witness value")]
    fn missing_witness_value_panics_with_a_clear_message() {
        let shard = Shard {
            id: 0,
            name: "s0".into(),
            range: 0..1,
            inputs: vec![],
            outputs: vec![Register::Virtual(0)],
        };
        let witness: Witness = HashMap::new();

        MockProver.prove(&shard, &witness);
    }
}
