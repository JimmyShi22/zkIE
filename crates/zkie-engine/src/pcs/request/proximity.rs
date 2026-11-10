//! CS/BASE local algebraic component: plan, statement view and row handoff.
//!
//! This is candidate development code. It defines only the verifier-side plan
//! constraints, the verifier-side statement view and the prover-side row
//! handoff used by the local CS/BASE algebraic component. It does not implement
//! a complete proximity test, a production statement or a proof-byte format.
//!
//! The checks are the implemented subset, not an iopp_fault-equivalent check.
//! The k = 0, ell = 6 fixture is an algebraic fixture and is not claimed to
//! satisfy iopp_fault. Real IOPP needs a real FOLD output and witness source,
//! every applicable plan constraint, and a real SEND sample declaration with
//! its follow-up checks; those remain on the project unclosed list.

use zkie_core::error::{Error, Result};

use crate::pcs::request::code::{build_code_plan, CodeParams};
use crate::pcs::request::field::Fp3;

/// The role of one plan stage in the isolated two-stage chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageRole {
    CodeSwitch,
    Base,
}

/// Where the choices used by a stage come from. The isolated demo admits only
/// the verifier's independent plan; a proof cannot select them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceReference {
    VerifierPlan,
}

/// One independent verifier stage constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProximityStage {
    pub role: StageRole,
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub k: u32,
    pub h: u32,
    pub answers: u32,
    pub samples: u32,
    pub folds: bool,
    pub choice: ChoiceReference,
}

/// The independent verifier plan for the local CS -> BASE algebraic component.
///
/// The plan is supplied by the verifier, not by the proof. It fixes the input
/// codes and the two stage shapes. The implemented subset requires exactly one
/// code-switch stage followed by one BASE stage; adjacent code lengths and
/// mask capacity are checked, and the last stage has no folded rows. These are
/// implemented-subset constraints, not an iopp_fault-equivalent check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProximityPlan {
    pub input_code: CodeParams,
    pub input_mask_code: CodeParams,
    pub stages: Vec<ProximityStage>,
}

impl ProximityPlan {
    pub fn fault(&self) -> Option<&'static str> {
        if self.stages.len() != 2 {
            return Some("the isolated plan needs exactly a switch stage and a base stage");
        }
        let switch = self.stages[0];
        let base = self.stages[1];
        if switch.role != StageRole::CodeSwitch || base.role != StageRole::Base {
            return Some("the isolated plan stages are not switch then base");
        }
        for code in [self.input_code, switch.code, base.code] {
            if build_code_plan(code.log_n, code.ell, code.r).is_err() {
                return Some("a plan code is not a code");
            }
        }
        for code in [self.input_mask_code, switch.mask_code, base.mask_code] {
            if build_code_plan(code.log_n, code.ell, code.r).is_err() {
                return Some("a plan mask code is not a code");
            }
        }
        if self.input_mask_code.log_n > self.input_code.log_n
            || switch.mask_code.log_n > switch.code.log_n
            || base.mask_code.log_n > base.code.log_n
        {
            return Some("a mask code is longer than its main code");
        }
        if switch.code.log_n > self.input_code.log_n {
            return Some("the switch code is longer than the input code");
        }
        if base.code.log_n > switch.code.log_n {
            return Some("the base code is longer than the switch code");
        }
        if switch.choice != ChoiceReference::VerifierPlan
            || base.choice != ChoiceReference::VerifierPlan
        {
            return Some("a stage choice is not from the verifier plan");
        }
        if switch.folds && switch.k < 1 {
            return Some("a folding stage has no row variable");
        }
        if !switch.folds && switch.k != 0 {
            return Some("a non-folding stage names rows");
        }
        if switch.folds {
            return Some("the isolated subset does not implement row folding");
        }
        if base.folds {
            return Some("the final stage cannot fold rows");
        }
        if switch.k >= 64
            || switch.h > switch.k
            || base.k != 0
            || base.h != 0
        {
            return Some("the stage row shapes do not fit the isolated handoff");
        }
        if switch.answers != 1 || base.answers != 1 || switch.samples == 0 || base.samples == 0 {
            return Some("the isolated subset needs one answer and at least one sample per stage");
        }
        if switch.mask_code.ell < switch.answers as u64
            || base.mask_code.ell < base.answers as u64
        {
            return Some("a mask row has no room for its sample answer");
        }
        if switch.h as u64 > switch.mask_code.ell {
            return Some("a stage has more mask rows than mask capacity");
        }
        let switch_main_needed = self.input_code.ell.saturating_add(self.input_code.r);
        let switch_mask_needed = self
            .input_mask_code
            .ell
            .saturating_add(self.input_mask_code.r);
        if switch.code.ell < switch_main_needed || switch.mask_code.ell < switch_mask_needed {
            return Some("the switch message does not hold the old message and tail");
        }
        if switch.code.ell < 1 || base.code.ell < 1 {
            return Some("a stage has no message entry");
        }
        if base.code.ell != switch.code.ell
            || base.code.r != switch.code.r
            || base.mask_code.ell != switch.mask_code.ell
            || base.mask_code.r != switch.mask_code.r
        {
            return Some("the base message shape does not match the switched message");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        if self.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

/// The verifier-side statement view: codes, claim data and row lengths only.
/// The old plaintext rows are prover material and never enter this view; the
/// verifier authenticates old symbols by opening the committed rows at the
/// drawn positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProximityStatement {
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub main_len: u64,
    pub main_tail_len: u64,
    pub mask_len: u64,
    pub mask_tail_len: u64,
    pub claim: Vec<Fp3>,
    pub mask_claims: Vec<Vec<Fp3>>,
    pub target: Fp3,
}

impl ProximityStatement {
    pub fn from_rows(rows: &SingleRows) -> Result<Self> {
        rows.check()?;
        Ok(Self {
            code: rows.code,
            mask_code: rows.mask_code,
            main_len: rows.main.len() as u64,
            main_tail_len: rows.main_tail.len() as u64,
            mask_len: rows.masks[0].len() as u64,
            mask_tail_len: rows.mask_tails[0].len() as u64,
            claim: rows.claim.clone(),
            mask_claims: rows.mask_claims.clone(),
            target: rows.target,
        })
    }

    pub fn check(&self) -> Result<()> {
        build_code_plan(self.code.log_n, self.code.ell, self.code.r)?;
        build_code_plan(self.mask_code.log_n, self.mask_code.ell, self.mask_code.r)?;
        if self.main_len != self.code.ell
            || self.main_tail_len != self.code.r
            || self.mask_len != self.mask_code.ell
            || self.mask_tail_len != self.mask_code.r
            || self.claim.len() as u64 != self.code.ell
            || self.mask_claims.len() != 1
            || self.mask_claims[0].len() as u64 != self.mask_code.ell
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    /// A position on a mask row: the main position modulo the mask code's
    /// column count.
    pub fn mask_position(&self, position: u64) -> u64 {
        position % self.mask_code.n()
    }
}

/// The prover-side last-stage handoff: one main row and one mask row with one
/// combined claim. Kept rows are deliberately absent; this is a BASE
/// precondition, not a general group representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleRows {
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub main: Vec<Fp3>,
    pub main_tail: Vec<Fp3>,
    pub masks: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub claim: Vec<Fp3>,
    pub mask_claims: Vec<Vec<Fp3>>,
    pub target: Fp3,
}

impl SingleRows {
    pub fn check(&self) -> Result<()> {
        build_code_plan(self.code.log_n, self.code.ell, self.code.r)?;
        build_code_plan(self.mask_code.log_n, self.mask_code.ell, self.mask_code.r)?;
        if self.main.len() as u64 != self.code.ell
            || self.main_tail.len() as u64 != self.code.r
            || self.claim.len() as u64 != self.code.ell
            || self.masks.len() != self.mask_tails.len()
            || self.masks.len() != self.mask_claims.len()
            || self.masks.len() > 2
        {
            return Err(Error::ShapeMismatch);
        }
        for (index, mask) in self.masks.iter().enumerate() {
            if mask.len() as u64 != self.mask_code.ell
                || self.mask_tails[index].len() as u64 != self.mask_code.r
                || self.mask_claims[index].len() as u64 != self.mask_code.ell
            {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(())
    }

    /// The BASE precondition: no kept rows. The representation has no kept
    /// field, so this is true by construction; the method makes the boundary
    /// explicit at the handoff.
    pub fn base_ready(&self) -> Result<()> {
        self.check()
    }

    /// A position on a mask row: the main position modulo the mask code's
    /// column count.
    pub fn mask_position(&self, position: u64) -> u64 {
        position % self.mask_code.n()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(log_n: u32, ell: u64, r: u64) -> CodeParams {
        build_code_plan(log_n, ell, r).unwrap()
    }

    fn plan() -> ProximityPlan {
        ProximityPlan {
            input_code: code(3, 4, 1),
            input_mask_code: code(3, 4, 1),
            stages: vec![
                ProximityStage {
                    role: StageRole::CodeSwitch,
                    code: code(3, 6, 1),
                    mask_code: code(3, 6, 1),
                    k: 0,
                    h: 0,
                    answers: 1,
                    samples: 2,
                    folds: false,
                    choice: ChoiceReference::VerifierPlan,
                },
                ProximityStage {
                    role: StageRole::Base,
                    code: code(3, 6, 1),
                    mask_code: code(3, 6, 1),
                    k: 0,
                    h: 0,
                    answers: 1,
                    samples: 4,
                    folds: false,
                    choice: ChoiceReference::VerifierPlan,
                },
            ],
        }
    }

    #[test]
    fn plan_checks_adjacent_codes_and_last_stage_shape() {
        let plan = plan();
        assert!(plan.check().is_ok());
        let mut bad = plan.clone();
        bad.stages[1].k = 1;
        assert!(bad.check().is_err());
        let mut bad = plan.clone();
        bad.stages[1].code = code(4, 4, 1);
        assert!(bad.check().is_err());
        let mut bad = plan.clone();
        bad.stages[0].mask_code = code(4, 4, 1);
        assert!(bad.check().is_err());
        let mut bad = plan.clone();
        bad.stages[0].folds = true;
        assert!(bad.check().is_err());
        let mut bad = plan.clone();
        bad.stages[0].folds = true;
        bad.stages[0].k = 1;
        assert_eq!(
            bad.fault(),
            Some("the isolated subset does not implement row folding")
        );
        let mut bad = plan.clone();
        bad.stages[0].h = 7;
        assert!(bad.check().is_err());
        let mut bad = plan.clone();
        bad.stages[0].role = StageRole::Base;
        assert!(bad.check().is_err());
        let mut bad = plan.clone();
        bad.stages[1].role = StageRole::CodeSwitch;
        assert!(bad.check().is_err());
        let mut bad = plan;
        bad.stages.truncate(1);
        assert!(bad.check().is_err());
    }

    #[test]
    fn single_rows_check_shapes_and_base_precondition() {
        let code = code(3, 4, 1);
        let rows = SingleRows {
            code,
            mask_code: code,
            main: vec![Fp3::ZERO; 4],
            main_tail: vec![Fp3::ZERO; 1],
            masks: vec![vec![Fp3::ZERO; 4]],
            mask_tails: vec![vec![Fp3::ZERO; 1]],
            claim: vec![Fp3::ZERO; 4],
            mask_claims: vec![vec![Fp3::ZERO; 4]],
            target: Fp3::ZERO,
        };
        assert!(rows.check().is_ok());
        assert!(rows.base_ready().is_ok());
        assert_eq!(rows.mask_position(9), 1);
        let statement = ProximityStatement::from_rows(&rows).unwrap();
        assert!(statement.check().is_ok());
        assert_eq!(statement.mask_position(9), 1);
        let mut bad = rows;
        bad.main.pop();
        assert!(bad.check().is_err());
    }
}
