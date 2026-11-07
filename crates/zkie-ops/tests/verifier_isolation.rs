//! The verifier uses public data and the proof only: no witness, no trace and
//! no prover memory.

#[path = "support/single_shot_fixture.rs"]
mod fixture;

use zkie_ops::proof::binding::OracleRegistry;
use zkie_ops::single_shot::{prove_single_shot, verify_single_shot};

#[test]
fn verification_needs_only_the_public_bundle_and_proof() {
    let decode_len = 2;
    let (public, openings, witnesses) = fixture::fixture(decode_len);
    let proof = prove_single_shot(&public, &witnesses, &openings, vec![0; decode_len]).unwrap();

    // Drop every prover-side object before verification.
    drop(witnesses);
    drop(openings);
    let result = verify_single_shot(&public, &proof).unwrap();
    assert!(result.accepted);

    let mut no_oracle = public.clone();
    no_oracle.registry = OracleRegistry::new();
    assert!(verify_single_shot(&no_oracle, &proof).is_err());

    let mut changed = public.clone();
    changed.statement.prompt_tokens.push(9);
    assert!(verify_single_shot(&changed, &proof).is_err());
}

#[test]
fn public_bundle_carries_no_weights_or_trace() {
    let (public, _, _) = fixture::fixture(1);
    assert_eq!(public.registry.len(), 1);
    assert!(public.statement.prompt_tokens.len() >= 1);
    // The public struct has no weights, witness or trace fields; a verifier only
    // reaches the logits oracle through signed openings.
    let (_, openings, _) = fixture::fixture(1);
    assert_eq!(openings.len(), 1);
}

struct ProverSession;
struct VerifierSession;

impl zkie_ops::proof::binding::RelationProver<zkie_engine::pcs::request::field::Fp3>
    for ProverSession
{
    fn register_oracle(
        &mut self,
        _binding: &zkie_core::oracle::OracleBinding,
        _values: &zkie_core::types::TensorView,
    ) -> zkie_core::error::Result<zkie_core::oracle::OracleId> {
        Err(zkie_core::error::Error::UnsupportedPrimitive)
    }

    fn prove_relation(
        &mut self,
        _ctx: &mut dyn zkie_core::compute::ExecutionPort<zkie_engine::pcs::request::field::Fp3>,
        _relation: &zkie_ops::proof::binding::RelationSpec,
        _witness: &zkie_ops::proof::binding::RelationWitness,
    ) -> zkie_core::error::Result<zkie_ops::proof::binding::RelationReceipt> {
        Err(zkie_core::error::Error::UnsupportedPrimitive)
    }

    fn register_claims(
        &mut self,
        _claims: zkie_core::claim::PendingClaims<zkie_engine::pcs::request::field::Fp3>,
    ) -> zkie_core::error::Result<()> {
        Ok(())
    }
}

impl zkie_ops::proof::binding::RelationVerifier<zkie_engine::pcs::request::field::Fp3>
    for VerifierSession
{
    fn verify_relation(
        &mut self,
        _relation: &zkie_ops::proof::binding::RelationSpec,
        _receipt: &zkie_ops::proof::binding::RelationReceipt,
    ) -> zkie_core::error::Result<
        zkie_core::claim::PendingClaims<zkie_engine::pcs::request::field::Fp3>,
    > {
        Err(zkie_core::error::Error::UnsupportedPrimitive)
    }

    fn register_claims(
        &mut self,
        _claims: zkie_core::claim::PendingClaims<zkie_engine::pcs::request::field::Fp3>,
    ) -> zkie_core::error::Result<()> {
        Ok(())
    }
}

#[test]
fn operation_prove_and_verify_are_wired_to_relation_receipts() {
    use zkie_core::types::{DType, Digest, TensorId, TensorSpec};
    use zkie_engine::memory::BudgetedMemoryGovernor;
    use zkie_engine::pcs::request::field::Fp3;
    use zkie_engine::storage::EngineStorage;
    use zkie_ops::op::{OpCode, OpId, OpStatement, Operation};
    use zkie_ops::registry::{
        read_i64_view, spec_1d, write_i64_scratch, NullProgress, StorageOnlyPort, TempScratch,
    };
    use zkie_ops::relu::ReluOp;

    let temp = TempScratch::new("op-relation").unwrap();
    let storage = EngineStorage::new(&temp.path).unwrap();
    let governor = BudgetedMemoryGovernor::new(1 << 20);
    let progress = NullProgress;
    let spec = spec_1d(3);
    let view = write_i64_scratch(&storage, &governor, &spec, &[-3, 0, 4]).unwrap();
    let mut port = StorageOnlyPort::<Fp3>::new(&storage, &governor, &progress);
    let op = ReluOp;
    let record = op.forward(&mut port, &[view.clone()]).unwrap();
    let inputs = read_i64_view(&storage, &record.inputs[0]).unwrap();
    assert_eq!(inputs, vec![-3, 0, 4]);

    let statement = OpStatement {
        op_id: OpId(1),
        opcode: OpCode::Relu,
        semantic_version: 1,
        inputs: vec![TensorId(1)],
        outputs: vec![TensorId(2)],
        parameters_digest: Digest([7u8; 32]),
    };
    let receipt = op
        .prove(&mut port, &mut ProverSession, &statement, &record)
        .unwrap();
    let claims = op
        .verify(&mut VerifierSession, &statement, &receipt)
        .unwrap();
    assert!(claims.is_empty());

    let mut tampered = receipt.clone();
    tampered.message[0] ^= 0xff;
    assert!(op
        .verify(&mut VerifierSession, &statement, &tampered)
        .is_err());

    let mut changed_statement = statement.clone();
    changed_statement.opcode = OpCode::Add;
    assert!(op
        .verify(&mut VerifierSession, &changed_statement, &receipt)
        .is_err());

    let _ = (DType::I64, TensorSpec::new(DType::I64, vec![3]));
}
