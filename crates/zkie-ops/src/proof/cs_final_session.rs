//! Isolated fixture code-switch continuation from the FOLD output into BASE.
//!
//! The session starts on the same chain and from the same FOLD declaration,
//! layout, original member identities and transcript challenges that produced
//! the FOLD output; it does not create a second plan or rebuild unrelated rows.
//! The old rows are authenticated at the code-switch positions by opening the
//! original member commitments and the FOLD fresh mask tree and folding the
//! opened columns exactly as the BASE fold relation does. That relation remains
//! sampled at the drawn positions (`sampled_fold_relation_only`); the original
//! commitments are not described as completely bound.
//!
//! The prover sends the two new code-switch oracles (the switched main and mask
//! rows) as commitments before the SEND seed. The verifier derives the new
//! group declaration from its own authenticated old symbols and the transcript
//! theta: `register_switch` registers the original claim and one coefficient
//! claim per authenticated position and combines them into a group row. The
//! registration is pending; the code-switch stage does not prove the new group
//! declaration true on the spot, and no proximity test runs here. The final
//! single-row view is derived from that combined group row and the switch
//! layout, never from the old FOLD objects relabelled as the switch output.
//! BASE then opens the new code-switch commitments at its own positions; the
//! old FOLD row commitments are not substituted for them.
//!
//! Ordering: the two new roots are fixed, then the SEND seed is drawn, then the
//! answer is received, then one challenge round draws the old positions and
//! theta, then the old openings are authenticated and the new group is
//! registered and combined. BASE commitments/target* come before lambda, its
//! rows before its position draw, and its openings after. No new message is
//! inserted into an already frozen challenge window. The old-row fold relation
//! is checked only at the drawn positions; it is sampled, not complete.
//!
//! This is a conditional local experiment. It is not the reference proof, not
//! a complete proximity test, not a zero-knowledge result, not a production
//! pass and not an equivalence proof. The fixture is public, the fresh rows are
//! plaintext and the star row is fixed; there is no hiding. Unsupported switch
//! shapes and code lengths fail closed.

use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::base_stage::{self, BaseRows};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::cs_stage::{self, AuthenticatedSymbols, SwitchedClaims, SwitchedMessage};
use zkie_engine::pcs::request::field::{Fp, Fp3};
use zkie_engine::pcs::request::fold::{validate_fold_mask_shape, FoldShape};
use zkie_engine::pcs::request::proximity::{ProximityStatement, SingleRows};
use zkie_engine::pcs::request::read::{draw_positions, unique_sorted};
use zkie_engine::pcs::request::runtime_tree::RuntimeTreeOpening;

use crate::proof::base_final_session::{
    fold_fresh_columns, fold_identity, fold_original_columns, BaseOpening, FoldBasis,
};
use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};
use crate::proof::fold_handoff::FoldHandoff;

/// One new code-switch oracle commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsCommitment {
    pub root: Digest,
    pub start: Digest,
    pub count: u64,
}

/// The code-switch declaration fixed before the SEND seed: the FOLD identity
/// and the two new oracle commitments. There is no target field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsDeclarations {
    pub fold_identity: Digest,
    pub switched_main: CsCommitment,
    pub switched_mask: CsCommitment,
}

/// The BASE commitment message for the code-switch output: the two star row
/// commitments and target*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsBaseMessages {
    pub star: CsCommitment,
    pub star_mask: CsCommitment,
    pub target_star: Fp3,
}

/// The BASE openings: star rows and the new code-switch rows, in drawn order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsBaseOpenings {
    pub star: Vec<BaseOpening>,
    pub star_mask: Vec<BaseOpening>,
    pub switched_main: Vec<BaseOpening>,
    pub switched_mask: Vec<BaseOpening>,
}

/// The local code-switch continuation result. It stays pending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsOutcome {
    pub fold_identity: Digest,
    pub switched_main_root: Digest,
    pub switched_mask_root: Digest,
    pub star_root: Digest,
    pub star_mask_root: Digest,
    pub theta: Fp3,
    pub lambda: Fp3,
    pub target_star: Fp3,
    pub cs_positions: Vec<u64>,
    pub base_positions: Vec<u64>,
    pub registered_claims: usize,
    pub old_columns_authenticated: bool,
    pub new_group_registered: bool,
    pub cs_claims_pending: bool,
    pub cs_proves_new_group: bool,
    pub fold_relation_sampled_at_cs_positions: bool,
    pub sampled_fold_relation_only: bool,
    pub complete_fold_binding: bool,
    pub base_linear: bool,
    pub symbol_checks: bool,
    pub openings_verified: bool,
    pub pending: bool,
    pub complete_cs: bool,
    pub cs_connected_in_fixture: bool,
    pub reference_iopp_path: bool,
    pub conditional_local_experiment: bool,
    pub chain_handoff: bool,
    pub declaration_rounds: u64,
    pub answer_rounds: u64,
    pub positions_rounds: u64,
    pub base_commit_rounds: u64,
    pub base_rows_rounds: u64,
    pub final_rounds: u64,
    pub final_value: Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Declared,
    SeedDrawn,
    AnswerReceived,
    PositionsTheta,
    OldOpened,
    BaseCommitments,
    Lambda,
    BaseRows,
    BasePositions,
    BaseOpenings,
}

struct ProverCommitment {
    registry: OracleRegistry,
    openings: Vec<OracleOpening>,
    oracle: OracleId,
}

impl ProverCommitment {
    fn public(&self) -> Result<CsCommitment> {
        let root = self
            .registry
            .get(self.oracle)
            .ok_or(Error::ProverStateMissing)?;
        Ok(CsCommitment {
            root: root.root,
            start: root.start,
            count: root.count,
        })
    }

    fn open(&self, position: u64) -> Result<BaseOpening> {
        let opening = self
            .openings
            .get(position as usize)
            .ok_or(Error::InvalidIndex)?;
        Ok(BaseOpening {
            index: opening.index,
            values: opening.values.clone(),
            path: opening.path.clone(),
        })
    }
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

fn push_digest(words: &mut Vec<u64>, digest: &Digest) {
    for chunk in digest.0.chunks_exact(8) {
        words.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
}

fn push_commitment(words: &mut Vec<u64>, commitment: &CsCommitment) {
    push_digest(words, &commitment.root);
    push_digest(words, &commitment.start);
    words.push(commitment.count);
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

fn commit_row(code: CodeParams, message: &[Fp3], tail: &[Fp3]) -> Result<ProverCommitment> {
    let symbols = encode_row_fp3(&code, message, tail)?;
    let rows: Vec<Vec<u64>> = symbols.iter().map(fp3_words).collect();
    let mut registry = OracleRegistry::new();
    let (oracle, openings) = commit_rows_with_openings(&mut registry, &Digest([0x74; 32]), &rows)?;
    Ok(ProverCommitment {
        registry,
        openings,
        oracle,
    })
}

/// Places the FOLD fresh coefficients in the combined mask claim at the
/// position where the switched mask message carries the FOLD fresh row. The
/// original claim's target already includes the fresh term, so the target is
/// not changed; the basis changes so the inner product covers that term.
fn extend_combined_with_fresh(
    claims: &mut SwitchedClaims,
    handoff: &FoldHandoff,
    switch_mask_code: CodeParams,
) -> Result<()> {
    let offset = usize::try_from(
        handoff
            .mask_code
            .ell
            .checked_add(handoff.mask_code.r)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let end = offset
        .checked_add(handoff.fresh_claim.len())
        .ok_or(Error::ResourceLimit)?;
    if end > switch_mask_code.ell as usize {
        return Err(Error::UnsupportedPrimitive);
    }
    claims.combined.mask[offset..end].copy_from_slice(&handoff.fresh_claim);
    Ok(())
}

fn switch_statement(handoff: &FoldHandoff) -> Result<ProximityStatement> {
    let statement = ProximityStatement {
        code: handoff.code,
        mask_code: handoff.mask_code,
        main_len: handoff.code.ell,
        main_tail_len: handoff.code.r,
        mask_len: handoff.mask_code.ell,
        mask_tail_len: handoff.mask_code.r,
        claim: handoff.main_claim.clone(),
        mask_claims: vec![handoff.mask_claim.clone()],
        target: handoff.target,
    };
    statement.check()?;
    Ok(statement)
}

/// The prover side of the isolated code-switch continuation.
pub struct CsProver {
    statement: ProximityStatement,
    handoff: FoldHandoff,
    switch_code: CodeParams,
    switch_mask_code: CodeParams,
    switched: SwitchedMessage,
    old_rows: SingleRows,
    switched_main: ProverCommitment,
    switched_mask: ProverCommitment,
    star_main: Vec<Fp3>,
    star_tail: Vec<Fp3>,
    star_mask: Vec<Fp3>,
    star_mask_tail: Vec<Fp3>,
    star: ProverCommitment,
    star_mask_row: ProverCommitment,
}

impl CsProver {
    pub fn new(
        rows: &SingleRows,
        handoff: &FoldHandoff,
        switch_code: CodeParams,
        switch_mask_code: CodeParams,
        fresh_row: Vec<Fp3>,
        star_main: Vec<Fp3>,
        star_tail: Vec<Fp3>,
        star_mask: Vec<Fp3>,
        star_mask_tail: Vec<Fp3>,
    ) -> Result<Self> {
        rows.base_ready()?;
        if star_main.len() as u64 != switch_code.ell
            || star_tail.len() as u64 != switch_code.r
            || star_mask.len() as u64 != switch_mask_code.ell
            || star_mask_tail.len() as u64 != switch_mask_code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let statement = switch_statement(handoff)?;
        let mut switched = cs_stage::switched_message(rows, switch_code, switch_mask_code)?;
        let fresh_offset = usize::try_from(
            handoff
                .mask_code
                .ell
                .checked_add(handoff.mask_code.r)
                .ok_or(Error::ResourceLimit)?,
        )
        .map_err(|_| Error::ResourceLimit)?;
        let fresh_end = fresh_offset
            .checked_add(fresh_row.len())
            .ok_or(Error::ResourceLimit)?;
        if fresh_row.len() as u64 != handoff.mask_code.ell
            || fresh_end > switched.mask.len()
        {
            return Err(Error::UnsupportedPrimitive);
        }
        switched.mask[fresh_offset..fresh_end].copy_from_slice(&fresh_row);
        let switched_main = commit_row(switch_code, &switched.main, &switched.main_tail)?;
        let switched_mask = commit_row(switch_mask_code, &switched.mask, &switched.mask_tail)?;
        let star = commit_row(switch_code, &star_main, &star_tail)?;
        let star_mask_row = commit_row(switch_mask_code, &star_mask, &star_mask_tail)?;
        Ok(Self {
            statement,
            handoff: handoff.clone(),
            switch_code,
            switch_mask_code,
            switched,
            old_rows: rows.clone(),
            switched_main,
            switched_mask,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
            star,
            star_mask_row,
        })
    }

    pub fn declarations(&self, fold_identity: Digest) -> Result<CsDeclarations> {
        Ok(CsDeclarations {
            fold_identity,
            switched_main: self.switched_main.public()?,
            switched_mask: self.switched_mask.public()?,
        })
    }

    pub fn answer(&self, seed: Fp3) -> Result<Fp3> {
        zkie_engine::pcs::request::code::evaluate_ood_fp3(
            &self.switch_code,
            &self.switched.main,
            &self.switched.main_tail,
            seed,
        )
    }

    fn rows(&self, claims: &SwitchedClaims) -> Result<SingleRows> {
        let rows = SingleRows {
            code: self.switch_code,
            mask_code: self.switch_mask_code,
            main: self.switched.main.clone(),
            main_tail: self.switched.main_tail.clone(),
            masks: vec![self.switched.mask.clone()],
            mask_tails: vec![self.switched.mask_tail.clone()],
            claim: claims.combined.main.clone(),
            mask_claims: vec![claims.combined.mask.clone()],
            target: claims.combined.target,
        };
        rows.base_ready()?;
        Ok(rows)
    }

    /// The prover-side registration from the plaintext old rows. The harness
    /// compares it with the verifier's registration; it is not a verifier
    /// source.
    pub fn claims(&self, positions: &[u64], theta: Fp3) -> Result<SwitchedClaims> {
        let main = encode_row_fp3(
            &self.old_rows.code,
            &self.old_rows.main,
            &self.old_rows.main_tail,
        )?;
        let mask = encode_row_fp3(
            &self.old_rows.mask_code,
            &self.old_rows.masks[0],
            &self.old_rows.mask_tails[0],
        )?;
        let mut authenticated = Vec::with_capacity(positions.len());
        for position in positions {
            let mask_position = *position % self.old_rows.mask_code.n();
            authenticated.push(AuthenticatedSymbols {
                position: *position,
                main: *main
                    .get(*position as usize)
                    .ok_or(Error::InvalidIndex)?,
                mask: *mask
                    .get(mask_position as usize)
                    .ok_or(Error::InvalidIndex)?,
            });
        }
        let mut claims = cs_stage::register_switch(
            &self.statement,
            self.switch_code,
            self.switch_mask_code,
            &authenticated,
            theta,
        )?;
        extend_combined_with_fresh(&mut claims, &self.handoff, self.switch_mask_code)?;
        Ok(claims)
    }

    pub fn base_messages(&self, claims: &SwitchedClaims) -> Result<CsBaseMessages> {
        let rows = self.rows(claims)?;
        let prepared = base_stage::prepare(
            &rows,
            &self.star_main,
            &self.star_tail,
            &self.star_mask,
            &self.star_mask_tail,
            Fp3::ZERO,
        )?;
        Ok(CsBaseMessages {
            star: self.star.public()?,
            star_mask: self.star_mask_row.public()?,
            target_star: prepared.target_star,
        })
    }

    pub fn base_rows(&self, claims: &SwitchedClaims, lambda: Fp3) -> Result<BaseRows> {
        let rows = self.rows(claims)?;
        base_stage::prepare(
            &rows,
            &self.star_main,
            &self.star_tail,
            &self.star_mask,
            &self.star_mask_tail,
            lambda,
        )
    }

    pub fn switched_openings(&self, positions: &[u64]) -> Result<(Vec<BaseOpening>, Vec<BaseOpening>)> {
        let mask_n = self.switched_mask.registry
            .get(self.switched_mask.oracle)
            .ok_or(Error::ProverStateMissing)?
            .count;
        let mut main = Vec::with_capacity(positions.len());
        let mut mask = Vec::with_capacity(positions.len());
        for position in positions {
            main.push(self.switched_main.open(*position)?);
            mask.push(self.switched_mask.open(*position % mask_n)?);
        }
        Ok((main, mask))
    }

    pub fn star_openings(&self, positions: &[u64]) -> Result<(Vec<BaseOpening>, Vec<BaseOpening>)> {
        let mask_n = self.star_mask_row.registry
            .get(self.star_mask_row.oracle)
            .ok_or(Error::ProverStateMissing)?
            .count;
        let mut star = Vec::with_capacity(positions.len());
        let mut star_mask = Vec::with_capacity(positions.len());
        for position in positions {
            star.push(self.star.open(*position)?);
            star_mask.push(self.star_mask_row.open(*position % mask_n)?);
        }
        Ok((star, star_mask))
    }
}

/// The verifier session for the isolated FOLD-to-code-switch-to-BASE
/// continuation. It owns the chain moved from the FOLD session and never reads
/// prover rows except through declarations, received rows and openings.
pub struct CsSession {
    chain: Chain,
    handoff: FoldHandoff,
    basis: FoldBasis,
    switch_code: CodeParams,
    switch_mask_code: CodeParams,
    registry: OracleRegistry,
    main_id: Option<OracleId>,
    mask_id: Option<OracleId>,
    star_id: Option<OracleId>,
    star_mask_id: Option<OracleId>,
    stage: Stage,
    seed: Option<Fp3>,
    answer: Option<Fp3>,
    positions: Vec<u64>,
    theta: Fp3,
    claims: Option<SwitchedClaims>,
    fold_main: Vec<Fp3>,
    fold_mask: Vec<Fp3>,
    fold_fresh: Vec<Fp3>,
    star_target: Option<Fp3>,
    lambda: Option<Fp3>,
    base_rows: Option<BaseRows>,
    base_positions: Vec<u64>,
    base_linear: bool,
    symbol_checks: bool,
    declaration_rounds: u64,
    answer_rounds: u64,
    positions_rounds: u64,
    base_commit_rounds: u64,
    base_rows_rounds: u64,
}

impl CsSession {
    pub fn new(
        chain: Chain,
        handoff: FoldHandoff,
        basis: FoldBasis,
        switch_code: CodeParams,
        switch_mask_code: CodeParams,
    ) -> Result<Self> {
        build_code_plan(switch_code.log_n, switch_code.ell, switch_code.r)?;
        build_code_plan(
            switch_mask_code.log_n,
            switch_mask_code.ell,
            switch_mask_code.r,
        )?;
        let shape = FoldShape {
            code: handoff.code,
            mask_code: handoff.mask_code,
            k: handoff.k,
            h: handoff.h,
            mask_leaf_class: basis.fresh_tree.leaf_class,
        };
        validate_fold_mask_shape(&shape)?;
        if handoff.gamma.len() != handoff.k as usize
            || handoff.main_claim.len() as u64 != handoff.code.ell
            || handoff.mask_claim.len() as u64 != handoff.mask_code.ell
            || handoff.fresh_claim.len() as u64 != handoff.mask_code.ell
            || handoff.kept_rows != 0
            || basis.layout != handoff.layout
            || basis.layout.fault(&shape).is_some()
            || basis.layout.members.iter().any(|member| member.kept)
            || basis.members.len() != basis.layout.members.len()
        {
            return Err(Error::InvalidConfig);
        }
        if switch_code.ell < handoff.code.ell + handoff.code.r
            || switch_mask_code.ell < handoff.mask_code.ell + handoff.mask_code.r
        {
            return Err(Error::UnsupportedPrimitive);
        }
        let declaration_rounds = chain.rounds();
        Ok(Self {
            chain,
            handoff,
            basis,
            switch_code,
            switch_mask_code,
            registry: OracleRegistry::new(),
            main_id: None,
            mask_id: None,
            star_id: None,
            star_mask_id: None,
            stage: Stage::Fresh,
            seed: None,
            answer: None,
            positions: Vec::new(),
            theta: Fp3::ZERO,
            claims: None,
            fold_main: Vec::new(),
            fold_mask: Vec::new(),
            fold_fresh: Vec::new(),
            star_target: None,
            lambda: None,
            base_rows: None,
            base_positions: Vec::new(),
            base_linear: false,
            symbol_checks: false,
            declaration_rounds,
            answer_rounds: declaration_rounds,
            positions_rounds: declaration_rounds,
            base_commit_rounds: declaration_rounds,
            base_rows_rounds: declaration_rounds,
        })
    }

    pub fn receive_declarations(&mut self, declarations: CsDeclarations) -> Result<()> {
        if self.stage != Stage::Fresh || declarations.fold_identity != fold_identity(&self.handoff)
        {
            return Err(Error::StatementMismatch);
        }
        if declarations.switched_main.count != self.switch_code.n()
            || declarations.switched_mask.count != self.switch_mask_code.n()
            || declarations.switched_main.root == Digest::ZERO
            || declarations.switched_mask.root == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        self.main_id = Some(self.registry.insert(
            declarations.switched_main.root,
            declarations.switched_main.start,
            declarations.switched_main.count,
        )?);
        self.mask_id = Some(self.registry.insert(
            declarations.switched_mask.root,
            declarations.switched_mask.start,
            declarations.switched_mask.count,
        )?);
        let mut words = Vec::new();
        push_digest(&mut words, &declarations.fold_identity);
        push_commitment(&mut words, &declarations.switched_main);
        push_commitment(&mut words, &declarations.switched_mask);
        absorb(&mut self.chain, 6, &words)?;
        self.declaration_rounds = self.chain.rounds();
        self.stage = Stage::Declared;
        Ok(())
    }

    pub fn draw_seed(&mut self) -> Result<Fp3> {
        if self.stage != Stage::Declared {
            return Err(Error::InvalidSessionState);
        }
        let seed = self.chain.draw_e()?;
        self.seed = Some(seed);
        self.stage = Stage::SeedDrawn;
        Ok(seed)
    }

    pub fn receive_answer(&mut self, answer: Fp3) -> Result<()> {
        if self.stage != Stage::SeedDrawn {
            return Err(Error::InvalidSessionState);
        }
        self.answer = Some(answer);
        absorb(&mut self.chain, 7, &fp3_words(&answer))?;
        self.answer_rounds = self.chain.rounds();
        self.stage = Stage::AnswerReceived;
        Ok(())
    }

    pub fn draw_positions_theta(&mut self, count: usize) -> Result<(Vec<u64>, Fp3)> {
        if self.stage != Stage::AnswerReceived {
            return Err(Error::InvalidSessionState);
        }
        if count == 0 {
            return Err(Error::InvalidConfig);
        }
        absorb(&mut self.chain, 8, &[count as u64])?;
        let positions = draw_positions(&mut self.chain, count, self.handoff.code.log_n)?;
        let theta = self.chain.draw_e()?;
        self.positions = positions.clone();
        self.theta = theta;
        self.positions_rounds = self.chain.rounds();
        self.stage = Stage::PositionsTheta;
        Ok((positions, theta))
    }

    /// Authenticates the old (FOLD output) columns at the drawn positions from
    /// the original member and fresh commitments, then registers the pending
    /// switch claims and combines them with theta. The fold relation remains
    /// sampled at these positions.
    pub fn authenticate_and_register(
        &mut self,
        member_openings: &[RuntimeTreeOpening],
        fresh_opening: &RuntimeTreeOpening,
    ) -> Result<&SwitchedClaims> {
        if self.stage != Stage::PositionsTheta {
            return Err(Error::InvalidSessionState);
        }
        let unique = unique_sorted(&self.positions);
        let (fold_main, fold_mask) =
            fold_original_columns(&self.handoff, &self.basis, member_openings, &unique)?;
        let mask_positions: Vec<u64> = self
            .positions
            .iter()
            .map(|position| *position % self.handoff.mask_code.n())
            .collect();
        let mask_unique = unique_sorted(&mask_positions);
        let fold_fresh = fold_fresh_columns(
            &self.handoff,
            &self.basis,
            fresh_opening,
            &mask_unique,
        )?;
        let statement = switch_statement(&self.handoff)?;
        let mut authenticated = Vec::with_capacity(self.positions.len());
        for position in &self.positions {
            let at = unique.binary_search(position).map_err(|_| Error::InvalidIndex)?;
            authenticated.push(AuthenticatedSymbols {
                position: *position,
                main: fold_main[at],
                mask: fold_mask[at],
            });
        }
        let mut claims = cs_stage::register_switch(
            &statement,
            self.switch_code,
            self.switch_mask_code,
            &authenticated,
            self.theta,
        )?;
        extend_combined_with_fresh(&mut claims, &self.handoff, self.switch_mask_code)?;
        claims.base_ready(self.switch_code, self.switch_mask_code)?;
        self.fold_main = fold_main;
        self.fold_mask = fold_mask;
        self.fold_fresh = fold_fresh;
        self.claims = Some(claims);
        self.stage = Stage::OldOpened;
        Ok(self.claims.as_ref().expect("just stored"))
    }

    pub fn claims(&self) -> Option<&SwitchedClaims> {
        self.claims.as_ref()
    }

    pub fn receive_base_commitments(&mut self, messages: CsBaseMessages) -> Result<()> {
        if self.stage != Stage::OldOpened {
            return Err(Error::InvalidSessionState);
        }
        if messages.star.count != self.switch_code.n()
            || messages.star_mask.count != self.switch_mask_code.n()
            || messages.star.root == Digest::ZERO
            || messages.star_mask.root == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        if self.registry.get(self.main_id.ok_or(Error::InvalidSessionState)?).is_none()
            || self.registry.get(self.mask_id.ok_or(Error::InvalidSessionState)?).is_none()
        {
            return Err(Error::StatementMismatch);
        }
        let mut words = Vec::new();
        push_commitment(&mut words, &messages.star);
        push_commitment(&mut words, &messages.star_mask);
        words.extend(fp3_words(&messages.target_star));
        absorb(&mut self.chain, 9, &words)?;
        // Insert the star roots after binding them in the transcript.
        self.star_id = Some(self.registry.insert(
            messages.star.root,
            messages.star.start,
            messages.star.count,
        )?);
        self.star_mask_id = Some(self.registry.insert(
            messages.star_mask.root,
            messages.star_mask.start,
            messages.star_mask.count,
        )?);
        self.star_target = Some(messages.target_star);
        self.base_commit_rounds = self.chain.rounds();
        self.stage = Stage::BaseCommitments;
        Ok(())
    }

    pub fn draw_lambda(&mut self) -> Result<Fp3> {
        if self.stage != Stage::BaseCommitments {
            return Err(Error::InvalidSessionState);
        }
        let lambda = self.chain.draw_e()?;
        self.lambda = Some(lambda);
        self.stage = Stage::Lambda;
        Ok(lambda)
    }

    pub fn receive_base_rows(&mut self, rows: BaseRows) -> Result<()> {
        if self.stage != Stage::Lambda {
            return Err(Error::InvalidSessionState);
        }
        let lambda = self.lambda.ok_or(Error::InvalidSessionState)?;
        let target_star = self.star_target.ok_or(Error::InvalidSessionState)?;
        if rows.lambda != lambda || rows.target_star != target_star {
            return Err(Error::StatementMismatch);
        }
        if rows.main_prime.len() as u64 != self.switch_code.ell
            || rows.main_tail_prime.len() as u64 != self.switch_code.r
            || rows.mask_prime.len() as u64 != self.switch_mask_code.ell
            || rows.mask_tail_prime.len() as u64 != self.switch_mask_code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let mut m_words = Vec::new();
        for value in &rows.main_prime {
            m_words.extend(fp3_words(value));
        }
        for value in &rows.mask_prime {
            m_words.extend(fp3_words(value));
        }
        absorb(&mut self.chain, 10, &m_words)?;
        let mut r_words = Vec::new();
        for value in &rows.main_tail_prime {
            r_words.extend(fp3_words(value));
        }
        for value in &rows.mask_tail_prime {
            r_words.extend(fp3_words(value));
        }
        absorb(&mut self.chain, 11, &r_words)?;
        self.base_rows_rounds = self.chain.rounds();
        self.base_rows = Some(rows);
        self.stage = Stage::BaseRows;
        Ok(())
    }

    pub fn draw_base_positions(&mut self, count: usize) -> Result<Vec<u64>> {
        if self.stage != Stage::BaseRows {
            return Err(Error::InvalidSessionState);
        }
        if count == 0 {
            return Err(Error::InvalidConfig);
        }
        let positions = draw_positions(&mut self.chain, count, self.switch_code.log_n)?;
        self.base_positions = positions.clone();
        self.stage = Stage::BasePositions;
        Ok(positions)
    }

    fn open_committed(
        &self,
        oracle: OracleId,
        position: u64,
        count: u64,
        opening: &BaseOpening,
    ) -> Result<Fp3> {
        if opening.index != position {
            return Err(Error::InvalidIndex);
        }
        let opening = OracleOpening {
            oracle,
            index: opening.index,
            values: opening.values.clone(),
            path: opening.path.clone(),
        };
        verify_opening(&self.registry, &opening, count)?;
        fp3_from_words(&opening.values)
    }

    pub fn verify_base_openings(&mut self, openings: &CsBaseOpenings) -> Result<()> {
        if self.stage != Stage::BasePositions {
            return Err(Error::InvalidSessionState);
        }
        let claims = self.claims.as_ref().ok_or(Error::InvalidSessionState)?;
        let rows = self.base_rows.as_ref().ok_or(Error::InvalidSessionState)?;
        let lambda = self.lambda.ok_or(Error::InvalidSessionState)?;
        let main_id = self.main_id.ok_or(Error::InvalidSessionState)?;
        let mask_id = self.mask_id.ok_or(Error::InvalidSessionState)?;
        let star_id = self.star_id.ok_or(Error::InvalidSessionState)?;
        let star_mask_id = self.star_mask_id.ok_or(Error::InvalidSessionState)?;
        if openings.star.len() != self.base_positions.len()
            || openings.star_mask.len() != self.base_positions.len()
            || openings.switched_main.len() != self.base_positions.len()
            || openings.switched_mask.len() != self.base_positions.len()
        {
            return Err(Error::ShapeMismatch);
        }
        let mask_n = self.switch_mask_code.n();
        let mut star = Vec::with_capacity(self.base_positions.len());
        let mut star_mask = Vec::with_capacity(self.base_positions.len());
        let mut switched_main = Vec::with_capacity(self.base_positions.len());
        let mut switched_mask = Vec::with_capacity(self.base_positions.len());
        for (index, position) in self.base_positions.iter().enumerate() {
            let mask_position = *position % mask_n;
            star.push(self.open_committed(
                star_id,
                *position,
                self.switch_code.n(),
                &openings.star[index],
            )?);
            star_mask.push(self.open_committed(
                star_mask_id,
                mask_position,
                self.switch_mask_code.n(),
                &openings.star_mask[index],
            )?);
            switched_main.push(self.open_committed(
                main_id,
                *position,
                self.switch_code.n(),
                &openings.switched_main[index],
            )?);
            switched_mask.push(self.open_committed(
                mask_id,
                mask_position,
                self.switch_mask_code.n(),
                &openings.switched_mask[index],
            )?);
        }
        self.base_linear = base_stage::linear_holds(
            &claims.combined.main,
            &claims.combined.mask,
            claims.combined.target,
            rows,
        )?;
        let encoded_main =
            encode_row_fp3(&self.switch_code, &rows.main_prime, &rows.main_tail_prime)?;
        let encoded_mask = encode_row_fp3(
            &self.switch_mask_code,
            &rows.mask_prime,
            &rows.mask_tail_prime,
        )?;
        let mut symbol_checks = true;
        for (index, position) in self.base_positions.iter().enumerate() {
            let mask_position = *position % mask_n;
            symbol_checks &= base_stage::symbol_holds(
                *encoded_main
                    .get(*position as usize)
                    .ok_or(Error::InvalidIndex)?,
                star[index],
                lambda,
                switched_main[index],
            );
            symbol_checks &= base_stage::symbol_holds(
                *encoded_mask
                    .get(mask_position as usize)
                    .ok_or(Error::InvalidIndex)?,
                star_mask[index],
                lambda,
                switched_mask[index],
            );
        }
        self.symbol_checks = symbol_checks;
        self.stage = Stage::BaseOpenings;
        Ok(())
    }

    pub fn finish(self) -> Result<CsOutcome> {
        if self.stage != Stage::BaseOpenings {
            return Err(Error::InvalidSessionState);
        }
        let claims = self.claims.as_ref().ok_or(Error::InvalidSessionState)?;
        if !self.base_linear || !self.symbol_checks {
            return Err(Error::InvalidProof);
        }
        Ok(CsOutcome {
            fold_identity: fold_identity(&self.handoff),
            switched_main_root: self
                .registry
                .get(self.main_id.ok_or(Error::InvalidSessionState)?)
                .ok_or(Error::ProverStateMissing)?
                .root,
            switched_mask_root: self
                .registry
                .get(self.mask_id.ok_or(Error::InvalidSessionState)?)
                .ok_or(Error::ProverStateMissing)?
                .root,
            star_root: self
                .registry
                .get(self.star_id.ok_or(Error::InvalidSessionState)?)
                .ok_or(Error::ProverStateMissing)?
                .root,
            star_mask_root: self
                .registry
                .get(self.star_mask_id.ok_or(Error::InvalidSessionState)?)
                .ok_or(Error::ProverStateMissing)?
                .root,
            theta: self.theta,
            lambda: self.lambda.ok_or(Error::InvalidSessionState)?,
            target_star: self.star_target.ok_or(Error::InvalidSessionState)?,
            cs_positions: self.positions,
            base_positions: self.base_positions,
            registered_claims: claims.claims.len(),
            old_columns_authenticated: true,
            new_group_registered: true,
            cs_claims_pending: true,
            cs_proves_new_group: false,
            fold_relation_sampled_at_cs_positions: true,
            sampled_fold_relation_only: true,
            complete_fold_binding: false,
            base_linear: self.base_linear,
            symbol_checks: self.symbol_checks,
            openings_verified: true,
            pending: true,
            complete_cs: false,
            cs_connected_in_fixture: true,
            reference_iopp_path: false,
            conditional_local_experiment: true,
            chain_handoff: true,
            declaration_rounds: self.declaration_rounds,
            answer_rounds: self.answer_rounds,
            positions_rounds: self.positions_rounds,
            base_commit_rounds: self.base_commit_rounds,
            base_rows_rounds: self.base_rows_rounds,
            final_rounds: self.chain.rounds(),
            final_value: *self.chain.value(),
        })
    }
}
