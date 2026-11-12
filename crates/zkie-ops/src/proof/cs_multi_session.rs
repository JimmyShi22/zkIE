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
use zkie_engine::pcs::request::pow::{search_nonce_for_plan, verify_nonce, MAX_WORK_BITS};
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
    main_tails: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
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
        let message = switched_rows_multi(
            &rows,
            &target,
            main_tails.clone(),
            mask_tails.clone(),
            pad_value,
        )?;
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
            main_tails,
            mask_tails,
        })
    }

    pub fn root(&self) -> Digest {
        *self.tree.root()
    }

    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.tree.open(positions)
    }

    pub fn main_tails(&self) -> &[Vec<Fp3>] {
        &self.main_tails
    }

    pub fn mask_tails(&self) -> &[Vec<Fp3>] {
        &self.mask_tails
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

/// Where the switch stage's old mask rows come from: a dedicated mask-only
/// commitment (the first stage's mask), the mask part of the switched member's
/// general commitment (every following stage), or the trusted layout's zero
/// mask when no member carries mask rows and the fold hides nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OldMaskSource {
    Tree { plan: MaskTreePlan, root: Digest },
    Member,
    /// The old mask is the zero vector by the trusted layout: no member has
    /// mask rows, the fold hides no mask variable, and the FOLD output's mask
    /// claim is zero. It is derived from the layout, not claimed by the proof,
    /// and carries no tree or opening.
    Zero,
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
    old_mask: OldMaskSource,
    stage: Stage,
    identity: Digest,
    root: Option<Digest>,
    seed: Option<SampleSeed>,
    answer: Option<Fp3>,
    positions: Vec<u64>,
    theta: Fp3,
    work_bits: u8,
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
        old_mask: OldMaskSource,
    ) -> Result<Self> {
        let old_mask_ok = match old_mask {
            OldMaskSource::Tree { plan, root } => {
                plan.fault().is_none() && root != Digest::ZERO
            }
            OldMaskSource::Member => basis
                .layout
                .members
                .iter()
                .any(|member| member.mask_rows > 0),
            OldMaskSource::Zero => {
                handoff.h == 0
                    && !basis
                        .layout
                        .members
                        .iter()
                        .any(|member| member.mask_rows > 0)
                    && handoff.mask_claim.iter().all(|value| *value == Fp3::ZERO)
            }
        };
        if u.fault().is_some()
            || plan.fault().is_some()
            || !old_mask_ok
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
            old_mask,
            stage: Stage::Fresh,
            identity,
            root: None,
            seed: None,
            answer: None,
            positions: Vec::new(),
            theta: Fp3::ZERO,
            work_bits: 0,
            claims: None,
        })
    }

    pub fn chain_value(&self) -> &Digest {
        self.chain.value()
    }

    /// The switched oracle root declared and absorbed by this session.
    pub fn declared_root(&self) -> Option<Digest> {
        self.root
    }

    pub fn into_chain(self) -> Chain {
        self.chain
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
        let (positions, theta, _nonce) = self.draw_positions_theta_with_nonce(count, Some(0))?;
        Ok((positions, theta))
    }

    /// Sets the work bits of the following position draw. The value comes from
    /// the trusted request plan; zero keeps the plain draw.
    pub fn set_work_bits(&mut self, bits: u8) -> Result<()> {
        if self.stage != Stage::Answered || bits > MAX_WORK_BITS {
            return Err(Error::InvalidSessionState);
        }
        self.work_bits = bits;
        Ok(())
    }

    /// Draws the positions with proof of work before the verifier's message.
    /// The prover passes `None` and searches; the verifier passes the nonce the
    /// prover carried. A failed test leaves the chain unchanged. Returns the
    /// positions, theta and the nonce.
    pub fn draw_positions_theta_with_nonce(
        &mut self,
        count: usize,
        nonce: Option<u64>,
    ) -> Result<(Vec<u64>, Fp3, u64)> {
        if self.stage != Stage::Answered || count == 0 {
            return Err(Error::InvalidSessionState);
        }
        let mut message = self.chain.message();
        message.push_u64(STAGE_POSITIONS);
        message.push_u64(count as u64);
        let prefix = message.prefix(KIND_ALGEBRAIC, self.work_bits)?;
        let nonce = match nonce {
            Some(value) => {
                if !verify_nonce(&prefix, value, self.work_bits)? {
                    return Err(Error::InvalidProof);
                }
                value
            }
            None => search_nonce_for_plan(&prefix)?,
        };
        if !self.chain.step(&prefix, nonce)? {
            return Err(Error::InvalidProof);
        }
        let positions = draw_positions(&mut self.chain, count, self.handoff.code.log_n)?;
        let theta = self.chain.draw_e()?;
        self.positions = positions.clone();
        self.theta = theta;
        self.stage = Stage::Positions;
        Ok((positions, theta, nonce))
    }

    /// Authenticates the old main and mask symbols through the original member
    /// and fresh commitments, registers the multi-row switch claims and appends
    /// the switched oracle's own sample claim.
    pub fn authenticate_and_register(
        &mut self,
        member_openings: &[RuntimeTreeOpening],
        fresh_opening: &RuntimeTreeOpening,
        old_mask_opening: Option<&RuntimeTreeOpening>,
    ) -> Result<&SwitchedClaims> {
        if self.stage != Stage::Positions {
            return Err(Error::InvalidSessionState);
        }
        let seed = self.seed.clone().ok_or(Error::InvalidSessionState)?;
        let answer = self.answer.ok_or(Error::InvalidSessionState)?;
        let unique = unique_sorted(&self.positions);
        let (fold_main, fold_mask) =
            fold_original_columns(&self.handoff, &self.basis, member_openings, &unique)?;
        let mask_positions: Vec<u64> = self
            .positions
            .iter()
            .map(|position| *position % self.handoff.mask_code.n())
            .collect();
        let mask_unique = unique_sorted(&mask_positions);
        let fold_fresh =
            fold_fresh_columns(&self.handoff, &self.basis, fresh_opening, &mask_unique)?;
        let old_mask_symbols: Vec<Fp3> = match self.old_mask {
            OldMaskSource::Tree { plan, root } => {
                let opening = old_mask_opening.ok_or(Error::StatementMismatch)?;
                let old_symbols = verify_mask_opening(&plan, &root, &mask_unique, opening)?;
                let old_width = usize::try_from(
                    plan.mask_rows
                        .checked_mul(3)
                        .ok_or(Error::ResourceLimit)?,
                )
                .map_err(|_| Error::ResourceLimit)?;
                if old_symbols.len() != mask_unique.len() * old_width || old_width < 3 {
                    return Err(Error::ShapeMismatch);
                }
                old_symbols
                    .chunks_exact(old_width)
                    .map(|symbols| Fp3::new(symbols[0], symbols[1], symbols[2]))
                    .collect()
            }
            OldMaskSource::Zero => {
                if old_mask_opening.is_some() {
                    return Err(Error::StatementMismatch);
                }
                vec![Fp3::ZERO; mask_unique.len()]
            }
            OldMaskSource::Member => {
                if old_mask_opening.is_some() {
                    return Err(Error::StatementMismatch);
                }
                let mut folded = vec![None; mask_unique.len()];
                for (at, position) in unique.iter().enumerate() {
                    let mask_position = *position % self.handoff.mask_code.n();
                    let mask_at = mask_unique
                        .binary_search(&mask_position)
                        .map_err(|_| Error::InvalidIndex)?;
                    let value = *fold_mask.get(at).ok_or(Error::InvalidIndex)?;
                    match folded[mask_at] {
                        Some(seen) if seen != value => return Err(Error::InvalidProof),
                        _ => folded[mask_at] = Some(value),
                    }
                }
                folded
                    .into_iter()
                    .map(|value| value.ok_or(Error::ShapeMismatch))
                    .collect::<Result<Vec<_>>>()?
            }
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::base_final_session::{FoldBasis, FoldMember};
    use crate::proof::fold_handoff::FoldHandoff;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::field::{Fp, E_DEGREE};
    use zkie_engine::pcs::request::cs_stage::switch_tree_plan_multi;
    use zkie_engine::pcs::request::fold::{FoldGroupLayout, FoldMemberLayout, FoldShape};
    use zkie_engine::pcs::request::runtime_tree::{
        MaskTree, RuntimePartRows, RuntimeTree, RuntimeTreePart,
    };

    const CONFIGURATION: Digest = Digest([0xb0; 32]);

    struct Fixture {
        rows: SingleRows,
        target: SwitchTarget,
        plan: RuntimeTreePlan,
        salts: Vec<Vec<u8>>,
        main_tails: Vec<Vec<Fp3>>,
        mask_tails: Vec<Vec<Fp3>>,
        handoff: FoldHandoff,
        basis: FoldBasis,
        member_tree: RuntimeTree,
        fresh_tree: MaskTree,
    }

    fn salts(columns: usize, seed: u8) -> Vec<Vec<u8>> {
        (0..columns)
            .map(|position| vec![(position as u8) ^ seed; 40])
            .collect()
    }

    fn fixture() -> Fixture {
        let target_code = build_code_plan(4, 4, 1).expect("target code");
        let target_mask = build_code_plan(4, 12, 1).expect("target mask code");
        let target =
            SwitchTarget::from_stage(target_code, target_mask, 2, 2, 1).expect("target");
        let code = build_code_plan(3, 4, 1).expect("rows code");
        let mask_code = target_mask;
        let rows = SingleRows {
            code,
            mask_code,
            main: (0..4).map(|i| Fp3::from_u64(0x10 + i)).collect(),
            main_tail: vec![Fp3::from_u64(0x20)],
            masks: vec![
                (0..12).map(|i| Fp3::from_u64(0x30 + i)).collect(),
                (0..12).map(|i| Fp3::from_u64(0x40 + i)).collect(),
            ],
            mask_tails: vec![vec![Fp3::from_u64(0x50)], vec![Fp3::from_u64(0x51)]],
            claim: (0..4).map(|i| Fp3::from_u64(0x60 + i)).collect(),
            mask_claims: vec![
                (0..12).map(|i| Fp3::from_u64(0x70 + i)).collect(),
                (0..12).map(|i| Fp3::from_u64(0x80 + i)).collect(),
            ],
            target: Fp3::from_u64(7),
        };
        rows.check().expect("rows");
        let plan = switch_tree_plan_multi(
            9,
            &target,
            40,
            Digest([0x91; 32]),
            CONFIGURATION,
        )
        .expect("switch plan");
        let plan_salts = salts(plan.columns() as usize, 0x55);

        let member_plan = RuntimeTreePlan {
            tree_number: 20,
            leaf_class: 0,
            log_n: 4,
            log_n_zk: 4,
            parts: vec![RuntimeTreePart {
                main_symbols: 12,
                mask_symbols: 12,
            }],
            salt_bytes: 40,
            run: Digest([0xa0; 32]),
            configuration: CONFIGURATION,
        };
        let member_salts = salts(16, 0x66);
        let member_main: Vec<Vec<Fp3>> = (0..4)
            .map(|row| (0..16).map(|col| Fp3::from_u64((row * 16 + col + 1) as u64)).collect())
            .collect();
        let member_mask: Vec<Vec<Fp3>> = (0..4)
            .map(|row| {
                (0..16)
                    .map(|col| Fp3::from_u64((0x100 + row * 16 + col) as u64))
                    .collect()
            })
            .collect();
        let member_tree = RuntimeTree::commit(
            member_plan.clone(),
            &[RuntimePartRows {
                main: member_main,
                mask: member_mask,
                main_deg: E_DEGREE as u32,
            }],
            &member_salts,
        )
        .expect("member tree");

        let shape = FoldShape {
            code: target.code,
            mask_code: target.mask_code,
            k: 2,
            h: 2,
            mask_leaf_class: 0,
        };
        let fresh_plan = MaskTreePlan {
            tree_number: 21,
            leaf_class: 0,
            log_n: 4,
            mask_rows: 4,
            salt_bytes: 40,
            run: Digest([0xa1; 32]),
            configuration: CONFIGURATION,
        };
        let fresh_rows: Vec<Vec<Fp3>> = (0..4)
            .map(|row| (0..12).map(|i| Fp3::from_u64((0x200 + row * 12 + i) as u64)).collect())
            .collect();
        let fresh_tails: Vec<Vec<Fp3>> = (0..4).map(|row| vec![Fp3::from_u64(0x300 + row as u64)]).collect();
        let fresh_tree = MaskTree::commit(
            fresh_plan,
            &shape,
            &fresh_rows,
            &fresh_tails,
            &salts(16, 0x77),
        )
        .expect("fresh tree");

        let layout = FoldGroupLayout {
            group: 1,
            k: 2,
            h: 2,
            members: vec![FoldMemberLayout {
                oracle: 1,
                main_rows: 4,
                mask_rows: 4,
                kept: false,
            }],
        };
        let handoff = FoldHandoff {
            code: target.code,
            mask_code: target.mask_code,
            main_claim: (0..4).map(|i| Fp3::from_u64(0x400 + i)).collect(),
            mask_claim: (0..12).map(|i| Fp3::from_u64(0x500 + i)).collect(),
            target: Fp3::from_u64(7),
            fresh_claim: (0..12).map(|i| Fp3::from_u64(0x600 + i)).collect(),
            group: 1,
            k: 2,
            h: 2,
            members: vec![1],
            gamma: vec![Fp3::from_u64(0x700), Fp3::from_u64(0x701)],
            eps: Fp3::from_u64(0x702),
            sum: Fp3::from_u64(0x703),
            fresh: 3,
            fresh_root: fresh_tree.root,
            tree_number: fresh_plan.tree_number,
            kept_factor: Fp3::ONE,
            kept_rows: 0,
            layout: layout.clone(),
        };
        let basis = FoldBasis {
            layout,
            fresh_root: fresh_tree.root,
            fresh_tree: fresh_plan,
            members: vec![FoldMember {
                oracle: 1,
                main_code: code,
                mask_code,
                root: *member_tree.root(),
                tree: member_plan,
            }],
        };
        Fixture {
            rows,
            target,
            plan,
            salts: plan_salts,
            main_tails: (0..4)
                .map(|row| vec![Fp3::from_u64(0x800 + row as u64)])
                .collect(),
            mask_tails: (0..4)
                .map(|row| vec![Fp3::from_u64(0x900 + row as u64)])
                .collect(),
            handoff,
            basis,
            member_tree,
            fresh_tree,
        }
    }

    fn run(
        fixture: &Fixture,
        tamper_member: bool,
        tamper_fresh: bool,
        wrong_member_root: bool,
        wrong_fresh_root: bool,
        tree_old_mask: bool,
    ) -> Result<()> {
        let prover = CsMultiProver::new(
            fixture.rows.clone(),
            fixture.target,
            fixture.main_tails.clone(),
            fixture.mask_tails.clone(),
            Fp3::from_u64(0x99),
            fixture.plan.clone(),
            &fixture.salts,
        )?;
        let mut basis = fixture.basis.clone();
        if wrong_member_root {
            basis.members[0].root = Digest([0xee; 32]);
        }
        if wrong_fresh_root {
            basis.fresh_root = Digest([0xef; 32]);
        }
        let chain = Chain::begin(77, &Digest::ZERO, b"cs multi session test", &[], None)?;
        let mut session = CsMultiSession::new(
            chain,
            fixture.handoff.clone(),
            basis,
            fixture.target,
            *prover.shape(),
            fixture.plan.clone(),
            OldMaskSource::Member,
        )?;
        session.receive_declarations(fold_identity(&fixture.handoff), prover.root())?;
        let seed = session.draw_sample_seed()?;
        session.receive_answer(prover.sample_answer(&seed)?)?;
        let (positions, _theta) = session.draw_positions_theta(2)?;
        let unique = unique_sorted(&positions);
        let mask_positions: Vec<u64> = positions
            .iter()
            .map(|position| *position % fixture.handoff.mask_code.n())
            .collect();
        let mask_unique = unique_sorted(&mask_positions);
        let mut member_opening = fixture.member_tree.open(&unique)?;
        let mut fresh_opening = fixture.fresh_tree.open(&mask_unique)?;
        if tamper_member {
            member_opening.columns[0].symbols[0] =
                member_opening.columns[0].symbols[0] + Fp::new(1);
        }
        if tamper_fresh {
            fresh_opening.columns[0].salt[0] ^= 1;
        }
        let old_mask_opening = if tree_old_mask {
            Some(&fresh_opening)
        } else {
            None
        };
        session.authenticate_and_register(&[member_opening], &fresh_opening, old_mask_opening)?;
        Ok(())
    }

    #[test]
    fn the_switch_session_accepts_honest_openings() {
        run(&fixture(), false, false, false, false, false).expect("honest session");
    }

    #[test]
    fn the_switch_session_rejects_a_wrong_member_root() {
        assert!(matches!(
            run(&fixture(), false, false, true, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_switch_session_rejects_a_wrong_fresh_root() {
        assert!(matches!(
            run(&fixture(), false, false, false, true, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_switch_session_rejects_a_tampered_member_opening() {
        assert!(matches!(
            run(&fixture(), true, false, false, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_switch_session_rejects_a_tampered_fresh_opening() {
        assert!(matches!(
            run(&fixture(), false, true, false, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_switch_session_rejects_a_tree_old_mask_under_a_member_source() {
        assert!(matches!(
            run(&fixture(), false, false, false, false, true),
            Err(Error::StatementMismatch)
        ));
    }
}
