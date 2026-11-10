//! Isolated CS -> BASE component session.
//!
//! The session commits the old single rows, binds the switched message, draws
//! the CS positions and batching challenge, registers the pending switch
//! claims, then prepares and checks the BASE final rows. It is a component
//! demonstration: the G_u/FOLD fixtures and fixed masks are not a real oracle
//! group, and the CS claims are registered for a later proximity test rather
//! than proved correct here.
//!
//! Boundary update: the earlier 2e/2f `FoldOutput::next_stage` remains
//! fail-closed; this module adds a separate component outcome whose
//! `next_stage()` returns `BaseComplete` only after the CS registration,
//! handoff, BASE linear check and symbol checks pass. That is a component
//! result, not a production proof acceptance.

use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::base_stage;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::code::{encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::cs_stage;
use zkie_engine::pcs::request::field::{Fp, Fp3};
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;
use zkie_engine::pcs::request::proximity::{ProximityPlan, SingleRows};
use zkie_engine::pcs::request::read::draw_positions;

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleOpening, OracleRegistry,
};

/// The concrete component returned after the isolated CS -> BASE checks. It
/// is not a production stage and does not accept a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProximityNext {
    BaseComplete,
}

/// The result of the isolated two-stage component chain. This is not an
/// accepted proof and does not claim a complete proximity test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProximityOutcome {
    pub old_main_root: Digest,
    pub old_mask_root: Digest,
    pub switched_root: Digest,
    pub star_root: Digest,
    pub cs_positions: Vec<u64>,
    pub base_positions: Vec<u64>,
    pub theta: Fp3,
    pub lambda: Fp3,
    pub registration_count: usize,
    pub stage_handoff: bool,
    pub base_linear: bool,
    pub symbol_checks: bool,
    pub next_stage: bool,
}

impl ProximityOutcome {
    /// Returns the concrete component only when every recorded check passed.
    /// A missing stage or a failed check is rejected; `FoldOutput` remains
    /// independently fail-closed.
    pub fn next_stage(&self) -> Result<ProximityNext> {
        if self.stage_handoff && self.base_linear && self.symbol_checks && self.next_stage {
            Ok(ProximityNext::BaseComplete)
        } else {
            Err(Error::InvalidSessionState)
        }
    }
}

struct CommittedRow {
    registry: OracleRegistry,
    openings: Vec<OracleOpening>,
    root: Digest,
}

fn fp3_words(value: &Fp3) -> Vec<u64> {
    vec![
        value.c[0].as_canonical_u64(),
        value.c[1].as_canonical_u64(),
        value.c[2].as_canonical_u64(),
    ]
}

fn fp3_from_words(values: &[u64]) -> Result<Fp3> {
    if values.len() != 3 {
        return Err(Error::ShapeMismatch);
    }
    Ok(Fp3::new(
        Fp::from_u64(values[0]),
        Fp::from_u64(values[1]),
        Fp::from_u64(values[2]),
    ))
}

fn commit_row(label: Digest, code: CodeParams, message: &[Fp3], tail: &[Fp3]) -> Result<CommittedRow> {
    let symbols = encode_row_fp3(&code, message, tail)?;
    let rows: Vec<Vec<u64>> = symbols.iter().map(fp3_words).collect();
    let mut registry = OracleRegistry::new();
    let (oracle, openings) = commit_rows_with_openings(&mut registry, &label, &rows)?;
    let root = registry.get(oracle).ok_or(Error::ProverStateMissing)?.root;
    Ok(CommittedRow {
        registry,
        openings,
        root,
    })
}

fn open_symbol(row: &CommittedRow, position: u64, count: u64) -> Result<Fp3> {
    let opening = row
        .openings
        .get(position as usize)
        .ok_or(Error::InvalidIndex)?;
    verify_opening(&row.registry, opening, count)?;
    fp3_from_words(&opening.values)
}

fn absorb(chain: &mut Chain, stage: u64, words: &[u64]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(stage);
    message.push_u64(words.len() as u64);
    for word in words {
        message.push_u64(*word);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn push_digest(words: &mut Vec<u64>, digest: &Digest) {
    for chunk in digest.0.chunks_exact(8) {
        words.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
}

/// Runs the component chain. The caller supplies the independent plan and the
/// fixture single rows; commitments and openings are local to this session.
pub fn verify_proximity_session(
    plan: &ProximityPlan,
    rows: &SingleRows,
    star_main: &[Fp3],
    star_tail: &[Fp3],
    star_mask: &[Fp3],
    star_mask_tail: &[Fp3],
) -> Result<ProximityOutcome> {
    plan.check()?;
    rows.base_ready()?;
    if rows.code != plan.input_code
        || rows.mask_code != plan.input_mask_code
        || rows.masks.len() != 1
        || plan.stages[0].code != plan.stages[1].code
        || plan.stages[0].mask_code != plan.stages[1].mask_code
    {
        return Err(Error::StatementMismatch);
    }
    let switch = plan.stages[0];
    let base = plan.stages[1];
    if switch.role != zkie_engine::pcs::request::proximity::StageRole::CodeSwitch
        || base.role != zkie_engine::pcs::request::proximity::StageRole::Base
    {
        return Err(Error::StatementMismatch);
    }

    let mut chain = Chain::begin(
        REQUEST_CHAIN_LABEL,
        &Digest([0x71; 32]),
        b"proximity-demo",
        &[],
        None,
    )?;

    let old_main = commit_row(Digest([0x72; 32]), rows.code, &rows.main, &rows.main_tail)?;
    let old_mask = commit_row(
        Digest([0x73; 32]),
        rows.mask_code,
        &rows.masks[0],
        &rows.mask_tails[0],
    )?;

    // CS SEND: the switched message root is fixed before its sample/positions.
    let mut g_main = vec![Fp3::ZERO; switch.code.ell as usize];
    let g_main_tail = vec![Fp3::ZERO; switch.code.r as usize];
    let mut g_mask = vec![Fp3::ZERO; switch.mask_code.ell as usize];
    let g_mask_tail = vec![Fp3::ZERO; switch.mask_code.r as usize];
    g_main[..rows.main.len()].copy_from_slice(&rows.main);
    g_main[rows.main.len()..rows.main.len() + rows.main_tail.len()]
        .copy_from_slice(&rows.main_tail);
    g_mask[..rows.masks[0].len()].copy_from_slice(&rows.masks[0]);
    g_mask[rows.masks[0].len()..rows.masks[0].len() + rows.mask_tails[0].len()]
        .copy_from_slice(&rows.mask_tails[0]);
    let switched_main = commit_row(
        Digest([0x74; 32]),
        switch.code,
        &g_main,
        &g_main_tail,
    )?;
    let switched_mask = commit_row(
        Digest([0x75; 32]),
        switch.mask_code,
        &g_mask,
        &g_mask_tail,
    )?;
    let mut send_words = Vec::new();
    push_digest(&mut send_words, &switched_main.root);
    push_digest(&mut send_words, &switched_mask.root);
    absorb(&mut chain, 1, &send_words)?;
    let _sample_seed = chain.draw_e()?;
    let sample_answer = Fp3::from_u64(0x31);
    absorb(&mut chain, 2, &fp3_words(&sample_answer))?;

    let cs_positions = draw_positions(&mut chain, switch.samples as usize, rows.code.log_n)?;
    let theta = chain.draw_e()?;
    for position in &cs_positions {
        let _main_symbol = open_symbol(&old_main, *position, rows.code.n())?;
        let _mask_symbol = open_symbol(&old_mask, rows.mask_position(*position), rows.mask_code.n())?;
    }
    let switched = cs_stage::register_switch(
        rows,
        switch.code,
        switch.mask_code,
        &cs_positions,
        theta,
    )?;
    let final_rows = cs_stage::switched_to_single_rows(&switched, base.code, base.mask_code)?;
    // BASE: fix star root and target* before drawing lambda.
    let target_star = {
        let main = base_stage::prepare(
            &final_rows,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
            Fp3::ZERO,
        )?;
        main.target_star
    };
    let star = commit_row(
        Digest([0x76; 32]),
        base.code,
        star_main,
        star_tail,
    )?;
    let star_mask_row = commit_row(
        Digest([0x77; 32]),
        base.mask_code,
        star_mask,
        star_mask_tail,
    )?;
    let mut star_words = Vec::new();
    push_digest(&mut star_words, &star.root);
    push_digest(&mut star_words, &star_mask_row.root);
    star_words.extend(fp3_words(&target_star));
    absorb(&mut chain, 3, &star_words)?;
    let lambda = chain.draw_e()?;
    let prepared = base_stage::prepare(
        &final_rows,
        star_main,
        star_tail,
        star_mask,
        star_mask_tail,
        lambda,
    )?;
    let mut plain = Vec::new();
    for value in &prepared.main_prime {
        plain.extend(fp3_words(value));
    }
    for value in &prepared.main_tail_prime {
        plain.extend(fp3_words(value));
    }
    for value in &prepared.mask_prime {
        plain.extend(fp3_words(value));
    }
    for value in &prepared.mask_tail_prime {
        plain.extend(fp3_words(value));
    }
    absorb(&mut chain, 4, &plain)?;
    let base_positions = draw_positions(&mut chain, base.samples as usize, base.code.log_n)?;

    let base_linear = base_stage::linear_holds(&final_rows, &prepared)?;
    let encoded_main = encode_row_fp3(&base.code, &prepared.main_prime, &prepared.main_tail_prime)?;
    let encoded_mask = encode_row_fp3(
        &base.mask_code,
        &prepared.mask_prime,
        &prepared.mask_tail_prime,
    )?;
    let mut symbol_checks = true;
    for position in &base_positions {
        let old_main_symbol = open_symbol(&switched_main, *position, switch.code.n())?;
        let old_mask_symbol = open_symbol(
            &switched_mask,
            final_rows.mask_position(*position),
            switch.mask_code.n(),
        )?;
        let star_main_symbol = open_symbol(&star, *position, base.code.n())?;
        let star_mask_symbol =
            open_symbol(&star_mask_row, final_rows.mask_position(*position), base.mask_code.n())?;
        if !base_stage::symbol_holds(
            encoded_main
                .get(*position as usize)
                .copied()
                .ok_or(Error::InvalidIndex)?,
            star_main_symbol,
            lambda,
            old_main_symbol,
        ) || !base_stage::symbol_holds(
            encoded_mask
                .get(final_rows.mask_position(*position) as usize)
                .copied()
                .ok_or(Error::InvalidIndex)?,
            star_mask_symbol,
            lambda,
            old_mask_symbol,
        ) {
            symbol_checks = false;
        }
    }
    if !base_linear || !symbol_checks {
        return Err(Error::InvalidProof);
    }

    Ok(ProximityOutcome {
        old_main_root: old_main.root,
        old_mask_root: old_mask.root,
        switched_root: switched_main.root,
        star_root: star.root,
        cs_positions,
        base_positions,
        theta,
        lambda,
        registration_count: switched.claims.len(),
        stage_handoff: final_rows.base_ready().is_ok(),
        base_linear,
        symbol_checks,
        next_stage: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::proximity::{ProximityStage, StageRole};

    fn fixture_rows() -> SingleRows {
        let old = build_code_plan(3, 4, 1).unwrap();
        let mask = vec![Fp3::from_u64(6), Fp3::from_u64(7), Fp3::from_u64(8), Fp3::from_u64(9)];
        SingleRows {
            code: old,
            mask_code: old,
            main: vec![
                Fp3::from_u64(1),
                Fp3::from_u64(2),
                Fp3::from_u64(3),
                Fp3::from_u64(4),
            ],
            main_tail: vec![Fp3::from_u64(5)],
            masks: vec![mask.clone()],
            mask_tails: vec![vec![Fp3::from_u64(10)]],
            claim: vec![Fp3::from_u64(1), Fp3::ZERO, Fp3::ZERO, Fp3::ZERO],
            mask_claims: vec![vec![Fp3::ZERO, Fp3::from_u64(1), Fp3::ZERO, Fp3::ZERO]],
            target: Fp3::from_u64(8),
        }
    }

    fn plan() -> ProximityPlan {
        let old = build_code_plan(3, 4, 1).unwrap();
        let new_code = build_code_plan(3, 6, 1).unwrap();
        ProximityPlan {
            input_code: old,
            input_mask_code: old,
            stages: vec![
                ProximityStage {
                    role: StageRole::CodeSwitch,
                    code: new_code,
                    mask_code: new_code,
                    k: 0,
                    h: 0,
                    answers: 1,
                    samples: 2,
                    choice: zkie_engine::pcs::request::proximity::ChoiceReference::VerifierPlan,
                },
                ProximityStage {
                    role: StageRole::Base,
                    code: new_code,
                    mask_code: new_code,
                    k: 0,
                    h: 0,
                    answers: 1,
                    samples: 4,
                    choice: zkie_engine::pcs::request::proximity::ChoiceReference::VerifierPlan,
                },
            ],
        }
    }

    #[test]
    fn component_chain_is_deterministic_and_verified() {
        let rows = fixture_rows();
        let plan = plan();
        let star_main = vec![Fp3::from_u64(11); 6];
        let star_tail = vec![Fp3::from_u64(12)];
        let star_mask = vec![Fp3::from_u64(13); 6];
        let star_mask_tail = vec![Fp3::from_u64(14)];
        let first = verify_proximity_session(
            &plan,
            &rows,
            &star_main,
            &star_tail,
            &star_mask,
            &star_mask_tail,
        )
        .unwrap();
        let second = verify_proximity_session(
            &plan,
            &rows,
            &star_main,
            &star_tail,
            &star_mask,
            &star_mask_tail,
        )
        .unwrap();
        assert_eq!(first, second);
        assert!(first.stage_handoff && first.base_linear && first.symbol_checks && first.next_stage);
        assert_eq!(first.next_stage().unwrap(), ProximityNext::BaseComplete);
        assert_eq!(first.registration_count, 3);
    }

    #[test]
    fn plan_or_shape_mismatch_fails_closed() {
        let mut plan = plan();
        plan.stages[1].k = 1;
        let rows = fixture_rows();
        let star = vec![Fp3::from_u64(11); 6];
        let tail = vec![Fp3::from_u64(12)];
        assert!(verify_proximity_session(&plan, &rows, &star, &tail, &star, &tail).is_err());
    }
}
