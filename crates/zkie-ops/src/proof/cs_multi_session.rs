//! Multi-row code switch session: one stage of the folded group's proximity
//! test. The switched oracle message is committed with `2^k` main rows and
//! `2^h` mask rows; the verifier authenticates the old rows at drawn positions
//! through the original member and fresh commitments, registers the switch
//! claims and combines them with theta. BASE is a later stage.

use std::slice;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::cs_stage::{
    commit_switch, register_switch_multi, switched_rows_multi, AuthenticatedRows, SwitchTarget,
    SwitchedClaims, SwitchedRows,
};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::gu::GuOracleShape;
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_engine::pcs::request::read::{draw_positions, unique_sorted};
use zkie_engine::pcs::request::relation::{oracle_sample_row_index, sample_cube_of, GuWitness};
use zkie_engine::pcs::request::runtime_tree::{
    verify_mask_opening, MaskTreePlan, RuntimeTree, RuntimeTreeOpening, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::SampleSeed;

use crate::proof::base_final_session::{
    fold_fresh_columns, fold_identity, fold_original_columns, FoldBasis,
};
use crate::proof::fold_handoff::FoldHandoff;

const STAGE_DECLARED: u64 = 20;
const STAGE_SEED: u64 = 21;
const STAGE_ANSWER: u64 = 22;
const STAGE_POSITIONS: u64 = 23;

fn absorb(chain: &mut Chain, stage: u64, words: &[u64]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(stage);
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
        words.push(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
}

fn witness(message: &SwitchedRows) -> GuWitness {
    GuWitness {
        main: message.main.clone(),
        mask: message.mask.clone(),
    }
}

/// The prover's side of one switch: the committed switched message and its
/// sample answer.
pub struct CsMultiProver {
    rows: SingleRows,
    target: SwitchTarget,
    message: SwitchedRows,
    plan: RuntimeTreePlan,
    tree: RuntimeTree,
    u: GuOracleShape,
}

impl CsMultiProver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: SingleRows,
        target: SwitchTarget,
        main_tails: Vec<Vec<Fp3>>,
        mask_tails: Vec<Vec<Fp3>>,
        pad_value: Fp3,
        plan: RuntimeTreePlan,
        salts: &[Vec<u8>],
    ) -> Result<Self> {
        let message = switched_rows_multi(&rows, &target, main_tails, mask_tails, pad_value)?;
        let tree = commit_switch(&target, &message, &plan, salts)?;
        let u = GuOracleShape {
            code: target.code,
            main_rows: target.main_rows,
            main_deg: 3,
            mask_rows: target.mask_rows,
            mask_code: Some(target.mask_code),
            answers: target.answers,
        };
        if u.fault().is_some() || plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            rows,
            target,
            message,
            plan,
            tree,
            u,
        })
    }

    pub fn root(&self) -> Digest {
        *self.tree.root()
    }

    pub fn plan(&self) -> &RuntimeTreePlan {
        &self.plan
    }

    pub fn shape(&self) -> &GuOracleShape {
        &self.u
    }

    pub fn target(&self) -> &SwitchTarget {
        &self.target
    }

    pub fn message(&self) -> &SwitchedRows {
        &self.message
    }

    pub fn rows(&self) -> &SingleRows {
        &self.rows
    }

    pub fn sample_answer(&self, seed: &SampleSeed) -> Result<Fp3> {
        let row = oracle_sample_row_index(
            self.u.code.log_n as u64,
            0,
            &self.u,
            slice::from_ref(seed),
            0,
            Fp3::ZERO,
        )?;
        row.evaluate(&witness(&self.message))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Declared,
    Seeded,
    Answered,
    Positions,
    Claims,
}

/// The verifier's side of one switch stage.
pub struct CsMultiSession {
    chain: Chain,
    handoff: FoldHandoff,
    basis: FoldBasis,
    target: SwitchTarget,
    u: GuOracleShape,
    plan: RuntimeTreePlan,
    old_mask_tree: MaskTreePlan,
    old_mask_root: Digest,
    stage: Stage,
    identity: Digest,
    root: Option<Digest>,
    seed: Option<SampleSeed>,
    answer: Option<Fp3>,
    positions: Vec<u64>,
    theta: Fp3,
    claims: Option<SwitchedClaims>,
}

impl CsMultiSession {
    pub fn new(
        chain: Chain,
        handoff: FoldHandoff,
        basis: FoldBasis,
        target: SwitchTarget,
        u: GuOracleShape,
        plan: RuntimeTreePlan,
        old_mask_tree: MaskTreePlan,
        old_mask_root: Digest,
    ) -> Result<Self> {
        if u.fault().is_some()
            || plan.fault().is_some()
            || old_mask_tree.fault().is_some()
            || old_mask_root == Digest::ZERO
            || basis.layout != handoff.layout
            || handoff.kept_rows != 0
        {
            return Err(Error::InvalidConfig);
        }
        let identity = fold_identity(&handoff);
        Ok(Self {
            chain,
            handoff,
            basis,
            target,
            u,
            plan,
            old_mask_tree,
            old_mask_root,
            stage: Stage::Fresh,
            identity,
            root: None,
            seed: None,
            answer: None,
            positions: Vec::new(),
            theta: Fp3::ZERO,
            claims: None,
        })
    }

    pub fn chain_value(&self) -> &Digest {
        self.chain.value()
    }

    pub fn receive_declarations(&mut self, identity: Digest, root: Digest) -> Result<()> {
        if self.stage != Stage::Fresh || identity != self.identity || root == Digest::ZERO {
            return Err(Error::StatementMismatch);
        }
        let mut words = Vec::new();
        push_digest(&mut words, &identity);
        push_digest(&mut words, &root);
        absorb(&mut self.chain, STAGE_DECLARED, &words)?;
        self.root = Some(root);
        self.stage = Stage::Declared;
        Ok(())
    }

    pub fn draw_sample_seed(&mut self) -> Result<SampleSeed> {
        if self.stage != Stage::Declared {
            return Err(Error::InvalidSessionState);
        }
        absorb(&mut self.chain, STAGE_SEED, &[])?;
        let bits = sample_cube_of(&self.u)?.bits as usize;
        let mut z = Vec::with_capacity(bits);
        for _ in 0..bits {
            z.push(self.chain.draw_e()?);
        }
        let beta = self.chain.draw_e_nonzero()?;
        let seed = SampleSeed { z, beta };
        self.seed = Some(seed.clone());
        self.stage = Stage::Seeded;
        Ok(seed)
    }

    pub fn receive_answer(&mut self, answer: Fp3) -> Result<()> {
        if self.stage != Stage::Seeded {
            return Err(Error::InvalidSessionState);
        }
        absorb(&mut self.chain, STAGE_ANSWER, &[])?;
        self.answer = Some(answer);
        self.stage = Stage::Answered;
        Ok(())
    }

    pub fn draw_positions_theta(&mut self, count: usize) -> Result<(Vec<u64>, Fp3)> {
        if self.stage != Stage::Answered || count == 0 {
            return Err(Error::InvalidSessionState);
        }
        absorb(&mut self.chain, STAGE_POSITIONS, &[count as u64])?;
        let positions = draw_positions(&mut self.chain, count, self.handoff.code.log_n)?;
        let theta = self.chain.draw_e()?;
        self.positions = positions.clone();
        self.theta = theta;
        self.stage = Stage::Positions;
        Ok((positions, theta))
    }

    /// Authenticates the old main and mask symbols through the original member
    /// and fresh commitments, registers the multi-row switch claims and appends
    /// the switched oracle's own sample claim.
    pub fn authenticate_and_register(
        &mut self,
        member_openings: &[RuntimeTreeOpening],
        fresh_opening: &RuntimeTreeOpening,
        old_mask_opening: &RuntimeTreeOpening,
    ) -> Result<&SwitchedClaims> {
        if self.stage != Stage::Positions {
            return Err(Error::InvalidSessionState);
        }
        let seed = self.seed.clone().ok_or(Error::InvalidSessionState)?;
        let answer = self.answer.ok_or(Error::InvalidSessionState)?;
        let unique = unique_sorted(&self.positions);
        let (fold_main, _fold_mask) =
            fold_original_columns(&self.handoff, &self.basis, member_openings, &unique)?;
        let mask_positions: Vec<u64> = self
            .positions
            .iter()
            .map(|position| *position % self.handoff.mask_code.n())
            .collect();
        let mask_unique = unique_sorted(&mask_positions);
        let fold_fresh =
            fold_fresh_columns(&self.handoff, &self.basis, fresh_opening, &mask_unique)?;
        let old_symbols = verify_mask_opening(
            &self.old_mask_tree,
            &self.old_mask_root,
            &mask_unique,
            old_mask_opening,
        )?;
        let old_width = usize::try_from(
            self.old_mask_tree
                .mask_rows
                .checked_mul(3)
                .ok_or(Error::ResourceLimit)?,
        )
        .map_err(|_| Error::ResourceLimit)?;
        if old_symbols.len() != mask_unique.len() * old_width || old_width < 3 {
            return Err(Error::ShapeMismatch);
        }
        let old_mask_symbols: Vec<Fp3> = old_symbols
            .chunks_exact(old_width)
            .map(|symbols| Fp3::new(symbols[0], symbols[1], symbols[2]))
            .collect();
        let mut authenticated = Vec::with_capacity(self.positions.len());
        for position in &self.positions {
            let at = unique
                .binary_search(position)
                .map_err(|_| Error::InvalidIndex)?;
            let mask_position = *position % self.handoff.mask_code.n();
            let mask_at = mask_unique
                .binary_search(&mask_position)
                .map_err(|_| Error::InvalidIndex)?;
            authenticated.push(AuthenticatedRows {
                position: *position,
                main: fold_main[at],
                masks: [old_mask_symbols[mask_at], fold_fresh[mask_at]],
            });
        }
        let old_rows = SingleRows {
            code: self.handoff.code,
            mask_code: self.handoff.mask_code,
            main: vec![Fp3::ZERO; self.handoff.code.ell as usize],
            main_tail: vec![Fp3::ZERO; self.handoff.code.r as usize],
            masks: vec![
                vec![Fp3::ZERO; self.handoff.mask_code.ell as usize],
                vec![Fp3::ZERO; self.handoff.mask_code.ell as usize],
            ],
            mask_tails: vec![
                vec![Fp3::ZERO; self.handoff.mask_code.r as usize],
                vec![Fp3::ZERO; self.handoff.mask_code.r as usize],
            ],
            claim: self.handoff.main_claim.clone(),
            mask_claims: vec![
                self.handoff.mask_claim.clone(),
                self.handoff.fresh_claim.clone(),
            ],
            target: self.handoff.target,
        };
        let mut claims =
            register_switch_multi(&old_rows, &self.target, &authenticated, self.theta)?;
        let sample = oracle_sample_row_index(
            self.u.code.log_n as u64,
            0,
            &self.u,
            slice::from_ref(&seed),
            0,
            answer,
        )?;
        append_claim(
            &mut claims,
            sample.main,
            sample.mask,
            sample.target,
            self.theta,
        )?;
        self.claims = Some(claims);
        self.stage = Stage::Claims;
        Ok(&self.claims.as_ref().expect("just stored"))
    }

    pub fn claims(&self) -> Option<&SwitchedClaims> {
        self.claims.as_ref()
    }

    pub fn theta(&self) -> Fp3 {
        self.theta
    }

    pub fn positions(&self) -> &[u64] {
        &self.positions
    }
}

fn append_claim(
    claims: &mut SwitchedClaims,
    main: Vec<Fp3>,
    mask: Vec<Fp3>,
    target: Fp3,
    theta: Fp3,
) -> Result<()> {
    if main.len() != claims.combined.main.len() || mask.len() != claims.combined.mask.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut power = Fp3::ONE;
    for _ in 0..claims.claims.len() {
        power = power.mul(&theta);
    }
    for (index, value) in main.iter().enumerate() {
        claims.combined.main[index] = claims.combined.main[index].add(&power.mul(value));
    }
    for (index, value) in mask.iter().enumerate() {
        claims.combined.mask[index] = claims.combined.mask[index].add(&power.mul(value));
    }
    claims.combined.target = claims.combined.target.add(&power.mul(&target));
    claims.claims.push(zkie_engine::pcs::request::cs_stage::SwitchClaim {
        main,
        mask,
        target,
    });
    Ok(())
}
