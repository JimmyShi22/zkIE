//! Verifies a set of per-shard `ShardProof`s against a `Dag`: every
//! shard's own validity, and every cross-shard edge's commitment
//! consistency. This is a plain consistency check, not a succinct proof --
//! see the design doc section 7 for turning this into a real
//! recursive/aggregation SNARK as follow-up work.

use std::fmt;

use super::model::Dag;
use super::prover::{Commitment, ShardProof};
use crate::graph_compiler::Register;

#[derive(Debug, PartialEq, Eq)]
pub enum LinkError {
    /// A shard's own proof was not valid (see `ShardProof::valid`).
    InvalidShard { shard_id: usize },
    /// The producer's committed output for `register` does not match the
    /// consumer's committed input for the same register.
    CommitmentMismatch {
        producer: usize,
        consumer: usize,
        register: Register,
    },
    /// A `ShardProof` for some shard referenced by `dag` was not supplied
    /// to `link` (caller error: `proofs` must contain exactly one entry
    /// per `dag.shards`, indexed by `shard.id`).
    MissingProof { shard_id: usize },
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkError::InvalidShard { shard_id } => {
                write!(f, "shard {shard_id}'s proof is not valid")
            }
            LinkError::CommitmentMismatch {
                producer,
                consumer,
                register,
            } => write!(
                f,
                "commitment mismatch on {register:?}: producer shard {producer}'s output commitment does not match consumer shard {consumer}'s input commitment"
            ),
            LinkError::MissingProof { shard_id } => {
                write!(f, "no proof supplied for shard {shard_id}")
            }
        }
    }
}

impl std::error::Error for LinkError {}

/// Verifies `proofs` against `dag`: every shard's own `valid` flag, and
/// every cross-shard edge's producer-output vs. consumer-input commitment
/// consistency (identical check for `Sequential` and `Broadcast` edges --
/// see `EdgeKind`'s doc comment).
///
/// `proofs[i]` must be the `ShardProof` for `dag.shards[i]`.
pub fn link(dag: &Dag, proofs: &[ShardProof]) -> Result<(), LinkError> {
    let get_proof = |shard_id: usize| -> Result<&ShardProof, LinkError> {
        proofs
            .get(shard_id)
            .ok_or(LinkError::MissingProof { shard_id })
    };

    for shard in &dag.shards {
        let proof = get_proof(shard.id)?;
        if !proof.valid {
            return Err(LinkError::InvalidShard { shard_id: shard.id });
        }
    }

    for edge in &dag.edges {
        let producer_proof = get_proof(edge.producer)?;
        let consumer_proof = get_proof(edge.consumer)?;

        let produced: &Commitment = producer_proof
            .output_commitments
            .get(&edge.register)
            .unwrap_or_else(|| {
                panic!(
                    "producer shard {} has no output commitment for {:?} (Prover implementation bug)",
                    edge.producer, edge.register
                )
            });
        let consumed: &Commitment = consumer_proof
            .input_commitments
            .get(&edge.register)
            .unwrap_or_else(|| {
                panic!(
                    "consumer shard {} has no input commitment for {:?} (Prover implementation bug)",
                    edge.consumer, edge.register
                )
            });

        if produced != consumed {
            return Err(LinkError::CommitmentMismatch {
                producer: edge.producer,
                consumer: edge.consumer,
                register: edge.register.clone(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::model::{build_dag, ShardSpec};
    use crate::dag::prover::{MockProver, Prover, Witness};
    use crate::graph_compiler::{CompiledInstruction, CompiledProgram};
    use std::collections::HashMap;
    use zkie_core::fixed_point::I18;
    use zkie_core::isa::{EltwiseOp, Instruction};

    fn add_instr(inputs: Vec<Register>, output_name: &str) -> CompiledInstruction {
        CompiledInstruction {
            instruction: Instruction::Eltwise { op: EltwiseOp::Add },
            inputs,
            output_name: output_name.to_string(),
        }
    }

    /// instr0 (shard0) -> instr1 (shard1), a plain hand-off, plus a
    /// matching witness map with correct values.
    fn linear_chain_fixture() -> (Dag, Witness) {
        let program = CompiledProgram {
            instructions: vec![
                add_instr(vec![Register::GraphInput("x".into())], "a"),
                add_instr(vec![Register::Virtual(0)], "b"),
            ],
            weights: Default::default(),
            graph_inputs: vec![],
            graph_outputs: vec![],
        };
        let specs = vec![
            ShardSpec {
                name: "s0".into(),
                range: 0..1,
            },
            ShardSpec {
                name: "s1".into(),
                range: 1..2,
            },
        ];
        let dag = build_dag(&program, &specs).expect("valid specs");

        let mut witness: Witness = HashMap::new();
        witness.insert(
            Register::GraphInput("x".into()),
            vec![I18::from_f64(1.0).unwrap()],
        );
        witness.insert(Register::Virtual(0), vec![I18::from_f64(2.0).unwrap()]);

        (dag, witness)
    }

    #[test]
    fn consistent_proofs_link_successfully() {
        let (dag, witness) = linear_chain_fixture();
        let proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();

        assert_eq!(link(&dag, &proofs), Ok(()));
    }

    #[test]
    fn corrupted_output_commitment_is_caught_precisely() {
        let (dag, witness) = linear_chain_fixture();
        let mut proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();

        proofs[0]
            .output_commitments
            .insert(Register::Virtual(0), Commitment([0xAA; 32]));

        let result = link(&dag, &proofs);
        assert_eq!(
            result,
            Err(LinkError::CommitmentMismatch {
                producer: 0,
                consumer: 1,
                register: Register::Virtual(0),
            })
        );
    }

    #[test]
    fn invalid_shard_proof_is_rejected() {
        let (dag, witness) = linear_chain_fixture();
        let mut proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();
        proofs[1].valid = false;

        assert_eq!(
            link(&dag, &proofs),
            Err(LinkError::InvalidShard { shard_id: 1 })
        );
    }

    #[test]
    fn broadcast_edge_corruption_flags_only_the_broken_consumer() {
        let program = CompiledProgram {
            instructions: vec![
                add_instr(vec![], "mask"),
                add_instr(vec![Register::Virtual(0)], "layer0_out"),
                add_instr(vec![Register::Virtual(0)], "layer1_out"),
            ],
            weights: Default::default(),
            graph_inputs: vec![],
            graph_outputs: vec![],
        };
        let specs = vec![
            ShardSpec {
                name: "prologue".into(),
                range: 0..1,
            },
            ShardSpec {
                name: "layer0".into(),
                range: 1..2,
            },
            ShardSpec {
                name: "layer1".into(),
                range: 2..3,
            },
        ];
        let dag = build_dag(&program, &specs).expect("valid specs");

        let mut witness: Witness = HashMap::new();
        witness.insert(Register::Virtual(0), vec![I18::from_f64(9.0).unwrap()]);

        let proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();
        assert_eq!(link(&dag, &proofs), Ok(()));

        let mut broken_proofs = proofs;
        broken_proofs[2]
            .input_commitments
            .insert(Register::Virtual(0), Commitment([0xBB; 32]));

        assert_eq!(
            link(&dag, &broken_proofs),
            Err(LinkError::CommitmentMismatch {
                producer: 0,
                consumer: 2,
                register: Register::Virtual(0),
            })
        );
    }
}
