//! Isolated fixture code-switch continuation from the FOLD output into BASE,
//! with fixture SEND sample registration and fresh-column correspondence.
//!
//! The session starts on the same chain and from the same FOLD declaration,
//! layout, original member identities and transcript challenges that produced
//! the FOLD output; it does not create a second plan or rebuild unrelated rows.
//! The old rows are authenticated at the switch positions by opening the
//! original member commitments and the FOLD fresh mask tree and folding the
//! opened columns exactly as the BASE fold relation does. That relation remains
//! sampled at the drawn positions; the original commitments are not described
//! as completely bound.
//!
//! The prover commits one new oracle carrying the switched main row and the
//! switched mask row in one runtime tree; its root and shape are fixed before
//! the seed. The switched mask row carries the old mask row, its tail, the
//! FOLD fresh mask row and the FOLD fresh tail, followed by the sample pad. A
//! fixture SEND phase draws a real sample seed (the cube's z values and a
//! nonzero beta), receives one answer, and registers the sample claim on the
//! same oracle shape and commitment. Unsupported answer counts and shapes fail
//! closed. This is not a production SEND receipt and not the reference
//! standard SEND-to-code-switch wiring; the sample claim is pending.
//!
//! In the same challenge round that draws the old positions and theta, the
//! verifier registers pending claims and combines them by powers of theta into
//! one group row: the original claim and one coefficient claim per
//! authenticated old position; one correspondence claim per drawn fresh
//! column, mapping the authenticated FOLD fresh column to the switched mask
//! row's fresh block and tail; and the SEND sample claim. The fresh
//! correspondence is derived from the verified fresh openings, never from the
//! prover's plaintext fresh row. The registration does not prove any claim true
//! on the spot and no proximity test runs here.
//!
//! BASE opens the new switch commitment at its own positions, derives the old
//! symbols from those openings, and checks the linear and symbol relations on
//! the combined claims. The old FOLD row commitments are not substituted for
//! the switch output. The result stays pending; the switch claims, the fresh
//! correspondence and the sample claim are pending.
//!
//! Ordering: the new root is fixed, then the seed, then the answer, then one
//! challenge round draws the old positions and theta, then the old openings are
//! authenticated and the claims registered and combined. BASE commitments and
//! target* come before lambda, its rows before its position draw, and its
//! openings after. No new message is inserted into an already frozen challenge
//! window. This is a conditional local experiment, not a proof, not complete
//! proximity, not zero knowledge and not a production pass; the fixture is
//! public, the fresh rows are plaintext and the star row is fixed.

use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::base_stage::{self, BaseRows};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::cs_stage::{
    self, AuthenticatedSymbols, SwitchClaim, SwitchedClaims,
};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{validate_fold_mask_shape, FoldShape};
use zkie_engine::pcs::request::gu::{GuNotice, GuOracleShape, GuRegistry, GuShape};
use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
use zkie_engine::pcs::request::proximity::{ProximityStatement, SingleRows};
use zkie_engine::pcs::request::read::{draw_positions, unique_sorted};
use zkie_engine::pcs::request::relation::{
    gu_layout_fault, sample_cube, sample_row, validate_shared_sample_seeds, GuWitness,
};
use zkie_engine::pcs::request::runtime_tree::{
    verify_runtime_opening, RuntimePartRows, RuntimeTree, RuntimeTreeOpening, RuntimeTreePart,
    RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::{IndexShape, SampleSeed, TableDistances};

use crate::proof::base_final_session::{
    fold_fresh_columns, fold_identity, fold_original_columns, BaseOpening, FoldBasis,
};
use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};
use crate::proof::fold_handoff::FoldHandoff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsCommitment {
    pub root: Digest,
    pub start: Digest,
    pub count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsDeclarations {
    pub fold_identity: Digest,
    pub switched: CsCommitment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsBaseMessages {
    pub star: CsCommitment,
    pub star_mask: CsCommitment,
    pub target_star: Fp3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsBaseOpenings {
    pub star: Vec<BaseOpening>,
    pub star_mask: Vec<BaseOpening>,
    pub switched: RuntimeTreeOpening,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsOutcome {
    pub fold_identity: Digest,
    pub switched_root: Digest,
    pub star_root: Digest,
    pub star_mask_root: Digest,
    pub sample_answer: Fp3,
    pub sample_beta_nonzero: bool,
    pub sample_claim_registered: bool,
    pub sample_claim_pending: bool,
    pub send_standard_wiring: bool,
    pub fresh_correspondence_registered: bool,
    pub fresh_correspondence_sampled: bool,
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

fn fp3_from_symbols(symbols: &[Fp], offset: usize) -> Result<Fp3> {
    let values = symbols
        .get(offset..offset.checked_add(3).ok_or(Error::ResourceLimit)?)
        .ok_or(Error::ShapeMismatch)?;
    Ok(Fp3::new(values[0], values[1], values[2]))
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

fn fixture_index_shape() -> Result<IndexShape> {
    Ok(IndexShape {
        code: build_code_plan(3, 4, 1)?,
        v: 2,
        answers: 1,
        salt_bytes: LEAF_SALT_MIN_BYTES,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: 0,
            m_later: 0,
        },
    })
}

pub(crate) fn switch_tree_plan(
    tag: u64,
    main_code: CodeParams,
    mask_code: CodeParams,
) -> Result<RuntimeTreePlan> {
    let plan = RuntimeTreePlan {
        tree_number: 0x200u64.wrapping_add(tag & 0x3ff),
        leaf_class: 6,
        log_n: main_code.log_n,
        log_n_zk: mask_code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: E_DEGREE as u64,
            mask_symbols: E_DEGREE as u64,
        }],
        salt_bytes: LEAF_SALT_MIN_BYTES,
        run: Digest([0xc1; 32]),
        configuration: Digest([0xc2; 32]),
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(plan)
}

fn switch_shape(
    plan: &RuntimeTreePlan,
    main_code: CodeParams,
    mask_code: CodeParams,
) -> Result<GuShape> {
    let shape = GuShape {
        oracle: 0,
        group: 0,
        tree: plan.clone(),
        part: 0,
        rows_high: false,
        u: GuOracleShape {
            code: main_code,
            main_rows: 1,
            main_deg: E_DEGREE as u32,
            mask_rows: 1,
            mask_code: Some(mask_code),
            answers: 1,
        },
        index: fixture_index_shape()?,
    };
    if shape.fault().is_some() || gu_layout_fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

fn symbol_coefficients(
    code: CodeParams,
    position: u64,
    len: usize,
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    let zeros = vec![Fp3::ZERO; code.r as usize];
    let mut message = vec![Fp3::ZERO; len];
    let mut message_out = vec![Fp3::ZERO; len];
    for index in 0..len {
        message[index] = Fp3::ONE;
        let encoded = encode_row_fp3(&code, &message, &zeros)?;
        message_out[index] = *encoded.get(position as usize).ok_or(Error::InvalidIndex)?;
        message[index] = Fp3::ZERO;
    }
    let empty = vec![Fp3::ZERO; len];
    let mut tail = vec![Fp3::ZERO; code.r as usize];
    let mut tail_out = vec![Fp3::ZERO; code.r as usize];
    for index in 0..code.r as usize {
        tail[index] = Fp3::ONE;
        let encoded = encode_row_fp3(&code, &empty, &tail)?;
        tail_out[index] = *encoded.get(position as usize).ok_or(Error::InvalidIndex)?;
        tail[index] = Fp3::ZERO;
    }
    Ok((message_out, tail_out))
}

fn add_scaled(target: &mut [Fp3], source: &[Fp3], scale: Fp3) -> Result<()> {
    if target.len() != source.len() {
        return Err(Error::ShapeMismatch);
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target = target.add(&scale.mul(source));
    }
    Ok(())
}

fn append_claim(claims: &mut SwitchedClaims, claim: SwitchClaim, theta: Fp3) -> Result<()> {
    if claim.main.len() != claims.combined.main.len()
        || claim.mask.len() != claims.combined.mask.len()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut power = Fp3::ONE;
    for _ in 0..claims.claims.len() {
        power = power.mul(&theta);
    }
    add_scaled(&mut claims.combined.main, &claim.main, power)?;
    add_scaled(&mut claims.combined.mask, &claim.mask, power)?;
    claims.combined.target = claims.combined.target.add(&power.mul(&claim.target));
    claims.claims.push(claim);
    Ok(())
}

fn extend_combined_with_fresh(
    claims: &mut SwitchedClaims,
    handoff: &FoldHandoff,
    switch_mask_code: CodeParams,
) -> Result<usize> {
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
    Ok(offset)
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

pub struct CsProver {
    shape: GuShape,
    plan: RuntimeTreePlan,
    statement: ProximityStatement,
    handoff: FoldHandoff,
    switch_code: CodeParams,
    switch_mask_code: CodeParams,
    old_rows: SingleRows,
    main_message: Vec<Fp3>,
    mask_message: Vec<Fp3>,
    main_tail: Vec<Fp3>,
    mask_tail: Vec<Fp3>,
    fresh_row: Vec<Fp3>,
    fresh_tail: Vec<Fp3>,
    tree: RuntimeTree,
    gu: GuRegistry,
    star_main: Vec<Fp3>,
    star_tail: Vec<Fp3>,
    star_mask: Vec<Fp3>,
    star_mask_tail: Vec<Fp3>,
    star: ProverCommitment,
    star_mask_row: ProverCommitment,
}

impl CsProver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: &SingleRows,
        handoff: &FoldHandoff,
        switch_code: CodeParams,
        switch_mask_code: CodeParams,
        fresh_row: Vec<Fp3>,
        fresh_tail: Vec<Fp3>,
        tag: u64,
        star_main: Vec<Fp3>,
        star_tail: Vec<Fp3>,
        star_mask: Vec<Fp3>,
        star_mask_tail: Vec<Fp3>,
    ) -> Result<Self> {
        Self::new_with_fresh_tamper(
            rows,
            handoff,
            switch_code,
            switch_mask_code,
            fresh_row,
            fresh_tail,
            tag,
            Fp3::ZERO,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
        )
    }

    /// The committed fresh block may carry a fixed shift while the
    /// correspondence targets still use the true fresh row. This is the
    /// "replace the new group fresh row" negative hook.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_fresh_tamper(
        rows: &SingleRows,
        handoff: &FoldHandoff,
        switch_code: CodeParams,
        switch_mask_code: CodeParams,
        fresh_row: Vec<Fp3>,
        fresh_tail: Vec<Fp3>,
        tag: u64,
        committed_fresh_shift: Fp3,
        star_main: Vec<Fp3>,
        star_tail: Vec<Fp3>,
        star_mask: Vec<Fp3>,
        star_mask_tail: Vec<Fp3>,
    ) -> Result<Self> {
        rows.base_ready()?;
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
            .checked_add(fresh_row.len() + fresh_tail.len())
            .ok_or(Error::ResourceLimit)?;
        if fresh_row.len() as u64 != handoff.mask_code.ell
            || fresh_tail.len() as u64 != handoff.mask_code.r
            || fresh_end >= switched.mask.len()
        {
            return Err(Error::UnsupportedPrimitive);
        }
        for (index, value) in fresh_row.iter().enumerate() {
            switched.mask[fresh_offset + index] = value.add(&committed_fresh_shift);
        }
        for (index, value) in fresh_tail.iter().enumerate() {
            switched.mask[fresh_offset + fresh_row.len() + index] = *value;
        }
        // One public deterministic pad per answer, reserved before the seed.
        let last = switched
            .mask
            .len()
            .checked_sub(1)
            .ok_or(Error::ResourceLimit)?;
        switched.mask[last] = Fp3::from_u64(0xd000u64.wrapping_add(tag.wrapping_mul(0x10)));
        let plan = switch_tree_plan(tag, switch_code, switch_mask_code)?;
        let shape = switch_shape(&plan, switch_code, switch_mask_code)?;
        let main_codeword = encode_row_fp3(&switch_code, &switched.main, &switched.main_tail)?;
        let mask_codeword = encode_row_fp3(&switch_mask_code, &switched.mask, &switched.mask_tail)?;
        let salts: Vec<Vec<u8>> = (0..plan.columns())
            .map(|position| {
                let byte = ((position as u8).wrapping_mul(0x1b)) ^ (tag as u8) ^ 0x5d;
                vec![byte; plan.salt_bytes]
            })
            .collect();
        let tree = RuntimeTree::commit(
            plan.clone(),
            &[RuntimePartRows {
                main: vec![main_codeword],
                mask: vec![mask_codeword],
                main_deg: E_DEGREE as u32,
            }],
            &salts,
        )?;
        let mut gu = GuRegistry::new();
        gu.receive(
            &shape,
            &GuNotice {
                root: *tree.root(),
                oracle: shape.oracle,
                tree_number: plan.tree_number,
                part: 0,
            },
        )?;
        if star_main.len() as u64 != switch_code.ell
            || star_tail.len() as u64 != switch_code.r
            || star_mask.len() as u64 != switch_mask_code.ell
            || star_mask_tail.len() as u64 != switch_mask_code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let star = commit_row(switch_code, &star_main, &star_tail)?;
        let star_mask_row = commit_row(switch_mask_code, &star_mask, &star_mask_tail)?;
        Ok(Self {
            shape,
            plan,
            statement,
            handoff: handoff.clone(),
            switch_code,
            switch_mask_code,
            old_rows: rows.clone(),
            main_message: switched.main,
            mask_message: switched.mask,
            main_tail: switched.main_tail,
            mask_tail: switched.mask_tail,
            fresh_row,
            fresh_tail,
            tree,
            gu,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
            star,
            star_mask_row,
        })
    }

    pub fn plan(&self) -> RuntimeTreePlan {
        self.plan.clone()
    }

    pub fn declarations(&self, fold_identity: Digest) -> Result<CsDeclarations> {
        Ok(CsDeclarations {
            fold_identity,
            switched: CsCommitment {
                root: *self.tree.root(),
                start: self.plan.run,
                count: self.plan.columns(),
            },
        })
    }

    pub fn sample_answer(&self, seeds: &[SampleSeed]) -> Result<Fp3> {
        let witness = GuWitness {
            main: self.main_message.clone(),
            mask: self.mask_message.clone(),
        };
        sample_row(&self.shape, &self.gu, seeds, 0, Fp3::ZERO)?.evaluate(&witness)
    }

    fn rows(&self, claims: &SwitchedClaims) -> Result<SingleRows> {
        let rows = SingleRows {
            code: self.switch_code,
            mask_code: self.switch_mask_code,
            main: self.main_message.clone(),
            main_tail: self.main_tail.clone(),
            masks: vec![self.mask_message.clone()],
            mask_tails: vec![self.mask_tail.clone()],
            claim: claims.combined.main.clone(),
            mask_claims: vec![claims.combined.mask.clone()],
            target: claims.combined.target,
        };
        rows.base_ready()?;
        Ok(rows)
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

    pub fn openings(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.tree.open(positions)
    }

    pub fn star_openings(
        &self,
        positions: &[u64],
    ) -> Result<(Vec<BaseOpening>, Vec<BaseOpening>)> {
        let mask_n = self
            .star_mask_row
            .registry
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

    /// The prover-side registration over the plaintext old rows. The harness
    /// compares it with the verifier's registration; it is not a verifier
    /// source. The fresh correspondence targets use the true fresh row and
    /// tail; only the committed block may carry the tamper hook.
    pub fn claims(
        &self,
        positions: &[u64],
        theta: Fp3,
        seeds: &[SampleSeed],
    ) -> Result<SwitchedClaims> {
        let main_rows = encode_row_fp3(
            &self.old_rows.code,
            &self.old_rows.main,
            &self.old_rows.main_tail,
        )?;
        let mask_rows = encode_row_fp3(
            &self.old_rows.mask_code,
            &self.old_rows.masks[0],
            &self.old_rows.mask_tails[0],
        )?;
        let mut authenticated = Vec::with_capacity(positions.len());
        for position in positions {
            let mask_position = *position % self.old_rows.mask_code.n();
            authenticated.push(AuthenticatedSymbols {
                position: *position,
                main: *main_rows
                    .get(*position as usize)
                    .ok_or(Error::InvalidIndex)?,
                mask: *mask_rows
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
        let fresh_offset =
            extend_combined_with_fresh(&mut claims, &self.handoff, self.switch_mask_code)?;
        let fresh_column = encode_row_fp3(
            &self.statement.mask_code,
            &self.fresh_row,
            &self.fresh_tail,
        )?;
        for position in positions {
            let mask_position = *position % self.statement.mask_code.n();
            let (message, tail) = symbol_coefficients(
                self.statement.mask_code,
                mask_position,
                self.handoff.mask_code.ell as usize,
            )?;
            let mut mask = vec![
                Fp3::ZERO;
                usize::try_from(self.switch_mask_code.ell).map_err(|_| Error::ResourceLimit)?
            ];
            mask[fresh_offset..fresh_offset + message.len()].copy_from_slice(&message);
            mask[fresh_offset + message.len()..fresh_offset + message.len() + tail.len()]
                .copy_from_slice(&tail);
            let target = *fresh_column
                .get(mask_position as usize)
                .ok_or(Error::InvalidIndex)?;
            let zero_main = vec![Fp3::ZERO; claims.combined.main.len()];
            append_claim(
                &mut claims,
                SwitchClaim {
                    main: zero_main,
                    mask,
                    target,
                },
                theta,
            )?;
        }
        let answer = self.sample_answer(seeds)?;
        let sample = sample_row(&self.shape, &self.gu, seeds, 0, answer)?;
        append_claim(
            &mut claims,
            SwitchClaim {
                main: sample.main,
                mask: sample.mask,
                target: sample.target,
            },
            theta,
        )?;
        Ok(claims)
    }
}

pub struct CsSession {
    chain: Chain,
    handoff: FoldHandoff,
    basis: FoldBasis,
    plan: RuntimeTreePlan,
    shape: GuShape,
    switch_code: CodeParams,
    switch_mask_code: CodeParams,
    root: Option<Digest>,
    stage: Stage,
    seed: Option<SampleSeed>,
    answer: Option<Fp3>,
    positions: Vec<u64>,
    theta: Fp3,
    claims: Option<SwitchedClaims>,
    fold_fresh: Vec<Fp3>,
    star_target: Option<Fp3>,
    lambda: Option<Fp3>,
    base_rows: Option<BaseRows>,
    base_positions: Vec<u64>,
    base_linear: bool,
    symbol_checks: bool,
    star_id: Option<OracleId>,
    star_mask_id: Option<OracleId>,
    registry: OracleRegistry,
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
        plan: RuntimeTreePlan,
        switch_code: CodeParams,
        switch_mask_code: CodeParams,
    ) -> Result<Self> {
        build_code_plan(switch_code.log_n, switch_code.ell, switch_code.r)?;
        build_code_plan(
            switch_mask_code.log_n,
            switch_mask_code.ell,
            switch_mask_code.r,
        )?;
        let fold_shape = FoldShape {
            code: handoff.code,
            mask_code: handoff.mask_code,
            k: handoff.k,
            h: handoff.h,
            mask_leaf_class: basis.fresh_tree.leaf_class,
        };
        validate_fold_mask_shape(&fold_shape)?;
        if handoff.gamma.len() != handoff.k as usize
            || handoff.main_claim.len() as u64 != handoff.code.ell
            || handoff.mask_claim.len() as u64 != handoff.mask_code.ell
            || handoff.fresh_claim.len() as u64 != handoff.mask_code.ell
            || handoff.kept_rows != 0
            || basis.layout != handoff.layout
            || basis.layout.members.iter().any(|member| member.kept)
            || basis.members.len() != basis.layout.members.len()
        {
            return Err(Error::InvalidConfig);
        }
        let fresh_offset = handoff
            .mask_code
            .ell
            .checked_add(handoff.mask_code.r)
            .ok_or(Error::ResourceLimit)?;
        let fresh_end = fresh_offset
            .checked_add(handoff.mask_code.ell + handoff.mask_code.r)
            .ok_or(Error::ResourceLimit)?;
        if switch_code.ell < handoff.code.ell + handoff.code.r
            || switch_mask_code.ell < fresh_end
        {
            return Err(Error::UnsupportedPrimitive);
        }
        let shape = switch_shape(&plan, switch_code, switch_mask_code)?;
        let declaration_rounds = chain.rounds();
        Ok(Self {
            chain,
            handoff,
            basis,
            plan,
            shape,
            switch_code,
            switch_mask_code,
            root: None,
            stage: Stage::Fresh,
            seed: None,
            answer: None,
            positions: Vec::new(),
            theta: Fp3::ZERO,
            claims: None,
            fold_fresh: Vec::new(),
            star_target: None,
            lambda: None,
            base_rows: None,
            base_positions: Vec::new(),
            base_linear: false,
            symbol_checks: false,
            star_id: None,
            star_mask_id: None,
            registry: OracleRegistry::new(),
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
        if declarations.switched.count != self.plan.columns()
            || declarations.switched.root == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        self.root = Some(declarations.switched.root);
        let mut words = Vec::new();
        push_digest(&mut words, &declarations.fold_identity);
        push_commitment(&mut words, &declarations.switched);
        absorb(&mut self.chain, 6, &words)?;
        self.declaration_rounds = self.chain.rounds();
        self.stage = Stage::Declared;
        Ok(())
    }

    pub fn draw_sample_seed(&mut self) -> Result<SampleSeed> {
        if self.stage != Stage::Declared {
            return Err(Error::InvalidSessionState);
        }
        let cube = sample_cube(&self.shape)?;
        let mut z = Vec::with_capacity(cube.bits as usize);
        for _ in 0..cube.bits {
            z.push(self.chain.draw_e()?);
        }
        let beta = self.chain.draw_e_nonzero()?;
        if beta.is_zero() || z.len() != cube.bits as usize {
            return Err(Error::InvalidConfig);
        }
        let seed = SampleSeed { z, beta };
        validate_shared_sample_seeds(&[&self.shape], std::slice::from_ref(&seed))?;
        self.seed = Some(seed.clone());
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

    pub fn authenticate_and_register(
        &mut self,
        member_openings: &[RuntimeTreeOpening],
        fresh_opening: &RuntimeTreeOpening,
    ) -> Result<&SwitchedClaims> {
        if self.stage != Stage::PositionsTheta {
            return Err(Error::InvalidSessionState);
        }
        let seed = self.seed.clone().ok_or(Error::InvalidSessionState)?;
        let answer = self.answer.ok_or(Error::InvalidSessionState)?;
        let root = self.root.ok_or(Error::InvalidSessionState)?;
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
        let fresh_offset =
            extend_combined_with_fresh(&mut claims, &self.handoff, self.switch_mask_code)?;
        for position in &self.positions {
            let mask_position = *position % self.handoff.mask_code.n();
            let (message, tail) = symbol_coefficients(
                self.handoff.mask_code,
                mask_position,
                self.handoff.mask_code.ell as usize,
            )?;
            let mut mask = vec![
                Fp3::ZERO;
                usize::try_from(self.switch_mask_code.ell).map_err(|_| Error::ResourceLimit)?
            ];
            mask[fresh_offset..fresh_offset + message.len()].copy_from_slice(&message);
            mask[fresh_offset + message.len()..fresh_offset + message.len() + tail.len()]
                .copy_from_slice(&tail);
            let at = mask_unique
                .binary_search(&mask_position)
                .map_err(|_| Error::InvalidIndex)?;
            let zero_main = vec![Fp3::ZERO; claims.combined.main.len()];
            append_claim(
                &mut claims,
                SwitchClaim {
                    main: zero_main,
                    mask,
                    target: fold_fresh[at],
                },
                self.theta,
            )?;
        }
        let mut gu = GuRegistry::new();
        gu.receive(
            &self.shape,
            &GuNotice {
                root,
                oracle: self.shape.oracle,
                tree_number: self.plan.tree_number,
                part: 0,
            },
        )?;
        let sample = sample_row(&self.shape, &gu, std::slice::from_ref(&seed), 0, answer)?;
        append_claim(
            &mut claims,
            SwitchClaim {
                main: sample.main,
                mask: sample.mask,
                target: sample.target,
            },
            self.theta,
        )?;
        claims.base_ready(self.switch_code, self.switch_mask_code)?;
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
        let mut words = Vec::new();
        push_commitment(&mut words, &messages.star);
        push_commitment(&mut words, &messages.star_mask);
        words.extend(fp3_words(&messages.target_star));
        absorb(&mut self.chain, 9, &words)?;
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
        let star_id = self.star_id.ok_or(Error::InvalidSessionState)?;
        let star_mask_id = self.star_mask_id.ok_or(Error::InvalidSessionState)?;
        let root = self.root.ok_or(Error::InvalidSessionState)?;
        if openings.star.len() != self.base_positions.len()
            || openings.star_mask.len() != self.base_positions.len()
        {
            return Err(Error::ShapeMismatch);
        }
        let unique = unique_sorted(&self.base_positions);
        verify_runtime_opening(&self.plan, &root, &unique, &openings.switched)?;
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
            let at = unique
                .binary_search(position)
                .map_err(|_| Error::InvalidIndex)?;
            let column = openings
                .switched
                .columns
                .get(at)
                .ok_or(Error::ShapeMismatch)?;
            switched_main.push(fp3_from_symbols(&column.symbols, 0)?);
            switched_mask.push(fp3_from_symbols(&column.symbols, E_DEGREE)?);
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
        let seed = self.seed.as_ref().ok_or(Error::InvalidSessionState)?;
        Ok(CsOutcome {
            fold_identity: fold_identity(&self.handoff),
            switched_root: self.root.ok_or(Error::InvalidSessionState)?,
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
            sample_answer: self.answer.ok_or(Error::InvalidSessionState)?,
            sample_beta_nonzero: !seed.beta.is_zero(),
            sample_claim_registered: true,
            sample_claim_pending: true,
            send_standard_wiring: false,
            fresh_correspondence_registered: true,
            fresh_correspondence_sampled: true,
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
