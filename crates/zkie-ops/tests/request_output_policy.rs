//! Output policy: seven-word acceptance policy, exact match and clipped gap.

use zkie_core::error::Error;
use zkie_core::session::{
    ClippedGapPolicy, DecodeCoverage, EosPolicy, OutputPolicy, Statement,
};
use zkie_core::types::Digest;

use zkie_ops::proof::request::output_policy::{
    policy_digest, prove_output, statement_digest, validate_statement_policy, verify_output,
    OutputProof, PolicyParams, POLICY_BYTES,
};
use zkie_ops::proof::request::request::ThreeChainCoordinator;

fn make_statement(policy: OutputPolicy) -> Statement {
    Statement {
        protocol_id: Digest([1u8; 32]),
        protocol_version: 1,
        parameter_digest: Digest([2u8; 32]),
        model_id: Digest([3u8; 32]),
        graph_digest: Digest([4u8; 32]),
        quantization_digest: Digest([5u8; 32]),
        tables_digest: Digest([6u8; 32]),
        prompt_tokens: vec![1, 2],
        output_tokens: vec![7, 8],
        coverage: DecodeCoverage::full(2),
        eos_policy: EosPolicy::FixedLength,
        output_policy: policy,
    }
}

fn policy(params: PolicyParams) -> OutputPolicy {
    params.validate().unwrap();
    let parameters = params.encode().to_vec();
    let mut policy = OutputPolicy::ClippedGap(ClippedGapPolicy {
        parameters: parameters.clone(),
        parameters_digest: Digest::ZERO,
    });
    let digest = policy_digest(&policy).unwrap();
    if let OutputPolicy::ClippedGap(inner) = &mut policy {
        inner.parameters_digest = digest;
    }
    policy
}

fn clipped(c: u64, theta: u64, cap: u64) -> OutputPolicy {
    policy(
        PolicyParams::clipped(c, theta, cap, 0, 7, 3).unwrap(),
    )
}

#[test]
fn policy_words_are_validated() {
    assert!(PolicyParams::clipped(100, 101, 0, 0, 7, 3).is_err());
    assert!(PolicyParams::clipped(300, 200, 0, 0, 6, 2).is_err());
    assert!(PolicyParams::clipped(300, 200, 0, 0, 7, 3).is_ok());
    assert!(PolicyParams::clipped(300, 200, 0, 64, 7, 3).is_err());
    let mut bad_f = PolicyParams::clipped(300, 200, 0, 0, 7, 3).unwrap();
    bad_f.f = 23;
    assert!(bad_f.validate().is_err());
    assert!(PolicyParams::clipped(300, 200, 0, 0, 7, 11).is_err());
    assert!(PolicyParams::clipped(300, 200, 0, 0, 7, 4).is_err());
    assert!(PolicyParams::exact_match(7).is_ok());
    assert!(PolicyParams::exact_match(8).is_ok());

    // Encoding is exactly seven little-endian words.
    let params = PolicyParams::clipped(300, 250, 1, 0, 7, 3).unwrap();
    assert_eq!(params.encode().len(), POLICY_BYTES);
    assert_eq!(PolicyParams::decode(&params.encode()).unwrap(), params);
    assert!(PolicyParams::decode(&params.encode()[..POLICY_BYTES - 1]).is_err());
}

#[test]
fn exact_match_accepts_only_zero_gaps() {
    let statement = make_statement(OutputPolicy::ExactMatch);
    let proof = OutputProof {
        tokens: statement.output_tokens.clone(),
        gaps: vec![0, 0],
    };
    verify_output(&statement, &proof).unwrap();
    let nonzero = OutputProof {
        tokens: statement.output_tokens.clone(),
        gaps: vec![0, 1],
    };
    assert!(verify_output(&statement, &nonzero).is_err());
    let wrong = OutputProof {
        tokens: vec![7, 9],
        gaps: vec![0, 0],
    };
    assert!(verify_output(&statement, &wrong).is_err());
}

#[test]
fn clipped_gap_checks_cost_sum_and_count() {
    let statement = make_statement(clipped(300, 250, 1));
    validate_statement_policy(&statement).unwrap();
    assert!(verify_output(
        &statement,
        &OutputProof {
            tokens: statement.output_tokens.clone(),
            gaps: vec![0, 3],
        }
    )
    .is_ok());
    assert!(verify_output(
        &statement,
        &OutputProof {
            tokens: statement.output_tokens.clone(),
            gaps: vec![0, 200],
        }
    )
    .is_ok());
    // Two positions above C, but cap = 1.
    assert!(verify_output(
        &statement,
        &OutputProof {
            tokens: statement.output_tokens.clone(),
            gaps: vec![200, 200],
        }
    )
    .is_err());
    // Negative gaps and wrong lengths are rejected.
    assert!(verify_output(
        &statement,
        &OutputProof {
            tokens: statement.output_tokens.clone(),
            gaps: vec![-1, 0],
        }
    )
    .is_err());
    assert!(verify_output(
        &statement,
        &OutputProof {
            tokens: statement.output_tokens.clone(),
            gaps: vec![0],
        }
    )
    .is_err());

    // A tight theta rejects even when the count is within cap.
    let tight = make_statement(clipped(300, 100, 1));
    assert!(verify_output(
        &tight,
        &OutputProof {
            tokens: tight.output_tokens.clone(),
            gaps: vec![0, 200],
        }
    )
    .is_err());
}

#[test]
fn policy_digest_and_statement_binding_reject_relaxation() {
    let mut broken = match clipped(300, 250, 1) {
        OutputPolicy::ClippedGap(clipped) => clipped,
        _ => unreachable!(),
    };
    broken.parameters = PolicyParams::clipped(300, 50, 1, 0, 7, 3).unwrap().encode().to_vec();
    broken.parameters_digest = Digest([9u8; 32]);
    let statement = make_statement(OutputPolicy::ClippedGap(broken));
    assert!(matches!(
        verify_output(
            &statement,
            &OutputProof {
                tokens: statement.output_tokens.clone(),
                gaps: vec![0, 0],
            }
        ),
        Err(Error::StatementMismatch)
    ));

    let exact = make_statement(OutputPolicy::ExactMatch);
    let bounded = make_statement(clipped(300, 250, 1));
    assert_ne!(
        statement_digest(&exact).unwrap(),
        statement_digest(&bounded).unwrap()
    );

    let parameters = Digest([2u8; 32]);
    let mut exact_coordinator = ThreeChainCoordinator::new(&exact, &parameters).unwrap();
    let mut bounded_coordinator = ThreeChainCoordinator::new(&bounded, &parameters).unwrap();
    exact_coordinator.start_request().unwrap();
    bounded_coordinator.start_request().unwrap();
    exact_coordinator
        .submit(
            zkie_ops::proof::request::request::ChainRole::Request,
            1,
            &[1, 2, 3],
        )
        .unwrap();
    bounded_coordinator
        .submit(
            zkie_ops::proof::request::request::ChainRole::Request,
            1,
            &[1, 2, 3],
        )
        .unwrap();
    assert_ne!(
        exact_coordinator.request_value(),
        bounded_coordinator.request_value()
    );
}

#[test]
fn prove_output_refuses_inconsistent_proofs() {
    let statement = make_statement(clipped(300, 250, 1));
    assert!(prove_output(
        &statement,
        OutputProof {
            tokens: statement.output_tokens.clone(),
            gaps: vec![200, 200],
        }
    )
    .is_err());
    assert!(prove_output(
        &make_statement(OutputPolicy::ExactMatch),
        OutputProof {
            tokens: vec![7, 8],
            gaps: vec![0, 0],
        }
    )
    .is_ok());
}
