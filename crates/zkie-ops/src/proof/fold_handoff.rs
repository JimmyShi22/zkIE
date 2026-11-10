//! Candidate FOLD output handoff into the local CS/BASE algebraic component.
//!
//! This module maps a FOLD session's final output into the claim data used by
//! the local CS/BASE component. The handoff runs on the verifier side: the
//! verifier supplies the trusted anchor (shape, group, layout and start
//! identity) and checks the output against it before any claim is built. The
//! output itself is a candidate from the FOLD session whose input rows are
//! still a self-reported fixture; this module does not claim a real FOLD
//! witness, a complete FOLD result or a complete IOPP/CS/BASE chain.
//!
//! Field mapping into [`SingleRows`]:
//!
//! - `FoldOutput.code` and `mask_code` enter the row codes;
//! - `FoldOutput.coefficients` enters the main claim `c`;
//! - `FoldOutput.mask_coefficients` enters the single old-mask claim `d`;
//! - `FoldOutput.target` enters the claim target;
//! - the plaintext main/mask rows do not exist in `FoldOutput`; they stay a
//!   prover-side fixture and are supplied to [`FoldHandoff::single_rows`];
//! - `fresh_coefficients`, `kept`, `kept_factor`, `gamma`, `eps`, `sum`,
//!   `fresh`, `fresh_root`, `tree_number`, `group`, `k`, `h`, `members` and the
//!   layout do not enter `SingleRows`; they are carried by the handoff for
//!   identity checks and are explicitly outside the local claim check.
//!
//! The implemented subset accepts only no-kept outputs (`kept` empty and
//! `kept_rows` zero). A kept output is rejected as unsupported rather than
//! silently dropped.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::fold::{
    fold_mask_weights, FoldGroupLayout, FoldOutput, FoldShape,
};
use zkie_engine::pcs::request::proximity::{ProximityStatement, SingleRows};

/// The verifier's trusted anchor for one FOLD output. It comes from the
/// verifier's independent FOLD plan and the start message the session accepted,
/// not from the output under test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldHandoffAnchor {
    pub shape: FoldShape,
    pub group: u64,
    pub members: Vec<u64>,
    pub fresh: u64,
    pub fresh_root: Digest,
    pub tree_number: u64,
    pub layout: FoldGroupLayout,
    /// The verifier session's FOLD round challenges.
    pub fold_gamma: Vec<Fp3>,
    /// The verifier session's FOLD epsilon.
    pub fold_epsilon: Fp3,
    /// The accepted FOLD start sum.
    pub fold_sum: Fp3,
}

/// The candidate FOLD output after verifier-side checking. Fields that enter
/// the local claim are separated from identity fields that do not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldHandoff {
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub main_claim: Vec<Fp3>,
    pub mask_claim: Vec<Fp3>,
    pub target: Fp3,
    /// Fresh-mask coefficients. They do not enter `SingleRows`; the local
    /// component does not check their contribution, and no fresh witness is
    /// connected here.
    pub fresh_claim: Vec<Fp3>,
    pub group: u64,
    pub k: u32,
    pub h: u32,
    pub members: Vec<u64>,
    pub gamma: Vec<Fp3>,
    pub eps: Fp3,
    pub sum: Fp3,
    pub fresh: u64,
    pub fresh_root: Digest,
    pub tree_number: u64,
    pub kept_factor: Fp3,
    pub kept_rows: u64,
    pub layout: FoldGroupLayout,
}

fn kept_factor(eps: Fp3, gamma: &[Fp3]) -> Fp3 {
    let mut factor = eps;
    for challenge in gamma {
        factor = factor.mul(&Fp3::ONE.sub(challenge));
    }
    factor
}

impl FoldHandoff {
    /// Checks the candidate output against the verifier's trusted anchor and
    /// maps the claim fields. Identity, shape, kept and gamma/kept-factor
    /// mismatches are rejected before any claim is handed to the component.
    pub fn from_output(output: &FoldOutput, anchor: &FoldHandoffAnchor) -> Result<Self> {
        build_code_plan(
            anchor.shape.code.log_n,
            anchor.shape.code.ell,
            anchor.shape.code.r,
        )?;
        build_code_plan(
            anchor.shape.mask_code.log_n,
            anchor.shape.mask_code.ell,
            anchor.shape.mask_code.r,
        )?;
        if anchor.layout.fault(&anchor.shape).is_some()
            || anchor.fold_gamma.len() != anchor.shape.k as usize
            || anchor.members
                != anchor
                    .layout
                    .members
                    .iter()
                    .map(|member| member.oracle)
                    .collect::<Vec<_>>()
        {
            return Err(Error::InvalidConfig);
        }
        if output.code != anchor.shape.code
            || output.mask_code != anchor.shape.mask_code
            || output.group != anchor.group
            || output.k != anchor.shape.k
            || output.h != anchor.shape.h
            || output.members != anchor.members
            || output.fresh != anchor.fresh
            || output.fresh_root != anchor.fresh_root
            || output.tree_number != anchor.tree_number
            || output.layout != anchor.layout
            || output.gamma != anchor.fold_gamma
            || output.eps != anchor.fold_epsilon
            || output.sum != anchor.fold_sum
        {
            return Err(Error::StatementMismatch);
        }
        if output.gamma.len() != anchor.shape.k as usize {
            return Err(Error::ShapeMismatch);
        }
        if output.kept_rows != anchor.layout.kept_rows_checked()? {
            return Err(Error::StatementMismatch);
        }
        if output.kept_rows != 0 || !output.kept.is_empty() {
            return Err(Error::UnsupportedPrimitive);
        }
        if output.kept_factor != kept_factor(output.eps, &output.gamma) {
            return Err(Error::StatementMismatch);
        }
        if output.coefficients.len() as u64 != anchor.shape.code.ell
            || output.mask_coefficients.len() as u64 != anchor.shape.mask_code.ell
            || output.fresh_coefficients.len() as u64 != anchor.shape.mask_code.ell
        {
            return Err(Error::ShapeMismatch);
        }
        let expected_fresh = fold_mask_weights(
            anchor.shape.k,
            anchor.shape.h,
            anchor.shape.mask_code.ell as usize,
            &output.gamma,
        )?;
        if output.fresh_coefficients != expected_fresh {
            return Err(Error::StatementMismatch);
        }
        Ok(Self {
            code: output.code,
            mask_code: output.mask_code,
            main_claim: output.coefficients.clone(),
            mask_claim: output.mask_coefficients.clone(),
            target: output.target,
            fresh_claim: output.fresh_coefficients.clone(),
            group: output.group,
            k: output.k,
            h: output.h,
            members: output.members.clone(),
            gamma: output.gamma.clone(),
            eps: output.eps,
            sum: output.sum,
            fresh: output.fresh,
            fresh_root: output.fresh_root,
            tree_number: output.tree_number,
            kept_factor: output.kept_factor,
            kept_rows: output.kept_rows,
            layout: output.layout.clone(),
        })
    }

    /// The verifier-side statement built from the mapped claim data. This is
    /// the product the local CS/BASE component consumes.
    pub fn statement(&self) -> Result<ProximityStatement> {
        let statement = ProximityStatement {
            code: self.code,
            mask_code: self.mask_code,
            main_len: self.code.ell,
            main_tail_len: self.code.r,
            mask_len: self.mask_code.ell,
            mask_tail_len: self.mask_code.r,
            claim: self.main_claim.clone(),
            mask_claims: vec![self.mask_claim.clone()],
            target: self.target,
        };
        statement.check()?;
        Ok(statement)
    }

    /// Builds the prover-side row fixture with the mapped claim/target. The
    /// plaintext rows are supplied by the caller and are not part of the FOLD
    /// output; they remain a fixture. The claim and target are the mapped
    /// values, not a re-report from these rows.
    pub fn single_rows(
        &self,
        main: Vec<Fp3>,
        main_tail: Vec<Fp3>,
        mask: Vec<Fp3>,
        mask_tail: Vec<Fp3>,
    ) -> Result<SingleRows> {
        if main.len() as u64 != self.code.ell
            || main_tail.len() as u64 != self.code.r
            || mask.len() as u64 != self.mask_code.ell
            || mask_tail.len() as u64 != self.mask_code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let rows = SingleRows {
            code: self.code,
            mask_code: self.mask_code,
            main,
            main_tail,
            masks: vec![mask],
            mask_tails: vec![mask_tail],
            claim: self.main_claim.clone(),
            mask_claims: vec![self.mask_claim.clone()],
            target: self.target,
        };
        rows.base_ready()?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::fold::{FoldKeptForm, FoldMemberLayout};

    fn code(log_n: u32, ell: u64, r: u64) -> CodeParams {
        build_code_plan(log_n, ell, r).unwrap()
    }

    fn shape() -> FoldShape {
        FoldShape {
            code: code(3, 4, 1),
            mask_code: code(3, 6, 1),
            k: 2,
            h: 0,
            mask_leaf_class: 0,
        }
    }

    fn layout() -> FoldGroupLayout {
        FoldGroupLayout {
            group: 5,
            k: 2,
            h: 0,
            members: vec![FoldMemberLayout {
                oracle: 11,
                main_rows: 1,
                mask_rows: 1,
                kept: false,
            }],
        }
    }

    fn anchor() -> FoldHandoffAnchor {
        FoldHandoffAnchor {
            shape: shape(),
            group: 5,
            members: vec![11],
            fresh: 99,
            fresh_root: Digest([0x31; 32]),
            tree_number: 3,
            layout: layout(),
            fold_gamma: vec![Fp3::from_u64(2), Fp3::from_u64(3)],
            fold_epsilon: Fp3::from_u64(5),
            fold_sum: Fp3::from_u64(7),
        }
    }

    fn output() -> FoldOutput {
        let eps = Fp3::from_u64(5);
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        FoldOutput {
            code: shape().code,
            mask_code: shape().mask_code,
            group: 5,
            k: 2,
            h: 0,
            members: vec![11],
            gamma: gamma.clone(),
            eps,
            sum: Fp3::from_u64(7),
            fresh: 99,
            fresh_root: Digest([0x31; 32]),
            tree_number: 3,
            coefficients: vec![
                Fp3::from_u64(1),
                Fp3::from_u64(2),
                Fp3::from_u64(3),
                Fp3::from_u64(4),
            ],
            mask_coefficients: (6..12).map(Fp3::from_u64).collect(),
            fresh_coefficients: fold_mask_weights(2, 0, 6, &gamma).unwrap(),
            kept: Vec::new(),
            kept_factor: kept_factor(eps, &gamma),
            layout: layout(),
            kept_rows: 0,
            target: Fp3::from_u64(0x2e),
        }
    }

    #[test]
    fn mapping_enters_claim_and_target_and_keeps_identity_separate() {
        let output = output();
        let handoff = FoldHandoff::from_output(&output, &anchor()).unwrap();
        assert_eq!(handoff.main_claim, output.coefficients);
        assert_eq!(handoff.mask_claim, output.mask_coefficients);
        assert_eq!(handoff.target, output.target);
        assert_eq!(handoff.fresh_claim, output.fresh_coefficients);
        assert_eq!(handoff.group, output.group);
        assert_eq!(handoff.members, output.members);
        assert_eq!(handoff.layout, output.layout);
        let statement = handoff.statement().unwrap();
        assert_eq!(statement.claim, output.coefficients);
        assert_eq!(statement.mask_claims, vec![output.mask_coefficients.clone()]);
        assert_eq!(statement.target, output.target);
        let rows = handoff
            .single_rows(
                vec![
                    Fp3::from_u64(1),
                    Fp3::from_u64(2),
                    Fp3::from_u64(3),
                    Fp3::from_u64(4),
                ],
                vec![Fp3::from_u64(5)],
                (6..12).map(Fp3::from_u64).collect(),
                vec![Fp3::from_u64(12)],
            )
            .unwrap();
        assert_eq!(rows.claim, output.coefficients);
        assert_eq!(rows.mask_claims, vec![output.mask_coefficients.clone()]);
        assert_eq!(rows.target, output.target);
        assert!(rows.base_ready().is_ok());
    }

    #[test]
    fn identity_or_kept_mismatch_is_rejected_at_the_handoff() {
        let mut bad = output();
        bad.group += 1;
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.fresh_root = Digest([0x32; 32]);
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.tree_number += 1;
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.members = vec![12];
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.layout.members[0].main_rows = 2;
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut kept_anchor = anchor();
        kept_anchor.layout.members[0].kept = true;
        let mut bad = output();
        bad.layout = kept_anchor.layout.clone();
        bad.kept = vec![FoldKeptForm {
            oracle: 11,
            coefficients: vec![Fp3::ZERO; 4],
        }];
        bad.kept_rows = 1;
        assert_eq!(
            FoldHandoff::from_output(&bad, &kept_anchor).unwrap_err(),
            Error::UnsupportedPrimitive
        );
        let mut bad = output();
        bad.kept_factor = bad.kept_factor.add(&Fp3::ONE);
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.gamma[0] = bad.gamma[0].add(&Fp3::ONE);
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.eps = bad.eps.add(&Fp3::ONE);
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.sum = bad.sum.add(&Fp3::ONE);
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.fresh_coefficients[0] = bad.fresh_coefficients[0].add(&Fp3::ONE);
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
    }

    #[test]
    fn shape_mismatch_or_truncation_is_rejected() {
        let mut bad = output();
        bad.coefficients.pop();
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::ShapeMismatch
        );
        let mut bad = output();
        bad.mask_coefficients.pop();
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::ShapeMismatch
        );
        let mut bad = output();
        bad.gamma.pop();
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::StatementMismatch
        );
        let mut bad = output();
        bad.fresh_coefficients.pop();
        assert_eq!(
            FoldHandoff::from_output(&bad, &anchor()).unwrap_err(),
            Error::ShapeMismatch
        );
        let handoff = FoldHandoff::from_output(&output(), &anchor()).unwrap();
        assert!(handoff
            .single_rows(vec![Fp3::ZERO; 3], vec![Fp3::ZERO], vec![Fp3::ZERO; 6], vec![Fp3::ZERO])
            .is_err());
    }
}
