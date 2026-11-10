//! CS/BASE local algebraic component session.
//!
//! This component checks only the local algebraic relations between the
//! authenticated old openings, the pending CS claims and the local BASE
//! equations. It is not a complete BASE, IOPP or proof acceptance and does not
//! claim that the pending CS claims are proved.
//!
//! The sample comparison is an auxiliary test-side check: the expected value
//! comes from an independent `SampleOracle` built from fixed test data, not
//! from the prover channel, and is not a verifier trust source.
//!
//! The protocol is split into a prover channel and a verifier session. The
//! prover owns the plaintext rows, commitments and openings; the verifier owns
//! the independent plan, the statement view and its own registry built from the
//! public commitments. The verifier draws every challenge, verifies every
//! opening it receives against its own registry, and never reads prover fields.
//! Boundary update: the earlier 2e/2f `FoldOutput::next_stage` remains
//! fail-closed; this module adds a separate component outcome whose
//! `component_complete()` returns `ComponentComplete` only after the
//! verifier-side CS registration, handoff, local BASE linear check and symbol
//! checks pass. That is a local algebraic component result, not a production
//! proof acceptance, not a complete BASE/IOPP result, and it does not prove
//! the pending CS claims.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::base_stage::{self, BaseRows};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::code::{encode_row_fp3, evaluate_ood_fp3, CodeParams};
use zkie_engine::pcs::request::cs_stage::{self, AuthenticatedSymbols};
use zkie_engine::pcs::request::field::{Fp, Fp3};
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;
use zkie_engine::pcs::request::proximity::{ProximityPlan, ProximityStatement, SingleRows};
use zkie_engine::pcs::request::read::draw_positions;
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};

/// The local algebraic component marker returned after the CS -> BASE checks.
/// It is not a production stage, not a complete BASE/IOPP result and does not
/// accept a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProximityComponent {
    ComponentComplete,
}

/// The result of the local CS/BASE algebraic component. This is not an
/// accepted proof, not a complete proximity test and not a complete BASE/IOPP
/// result. The private `verified` field prevents a caller from forming the
/// value from the public booleans alone.
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
    pub sample_answer: Fp3,
    pub registration_count: usize,
    pub stage_handoff: bool,
    pub base_linear: bool,
    pub symbol_checks: bool,
    /// The sample answer matched the independent test-side oracle. This is an
    /// auxiliary component check; it is not a verifier trust source and not a
    /// protocol verification of an OOD claim.
    pub sample_oracle_match: bool,
    pub component_checks_passed: bool,
    verified: bool,
}

impl ProximityOutcome {
    /// Returns the local algebraic component marker only when every recorded
    /// check passed. This does not accept a proof, does not prove the pending
    /// CS claims, does not claim a complete BASE/IOPP result and does not
    /// establish a FOLD-to-CS/BASE handoff. A caller that uses projected FOLD
    /// data must not present this marker as a valid handoff. `FoldOutput`
    /// remains independently fail-closed.
    pub fn component_complete(&self) -> Result<ProximityComponent> {
        if self.verified
            && self.stage_handoff
            && self.sample_oracle_match
            && self.base_linear
            && self.symbol_checks
            && self.component_checks_passed
        {
            Ok(ProximityComponent::ComponentComplete)
        } else {
            Err(Error::InvalidSessionState)
        }
    }
}

/// A public commitment message: the verifier inserts exactly these values into
/// its own registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub root: Digest,
    pub start: Digest,
    pub count: u64,
}

/// A prover opening response. It contains no oracle id; the verifier assigns
/// the id from the slot it asked about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverOpening {
    pub index: u64,
    pub values: Vec<u64>,
    pub path: Vec<Digest>,
}

/// The six commitments the prover sends before any challenge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProximityMessages {
    pub old_main: Commitment,
    pub old_mask: Commitment,
    pub switched_main: Commitment,
    pub switched_mask: Commitment,
    pub star: Commitment,
    pub star_mask: Commitment,
}

/// An independent test-side oracle for the declared OOD sample answer. It is
/// built from fixed test data separately from the prover channel, so the
/// comparison does not trust a value supplied by the channel. It is an
/// auxiliary component check only and is not a verifier trust source.
pub trait SampleOracle {
    fn expected(&self, seed: Fp3) -> Result<Fp3>;
}

/// The prover message channel. The verifier knows only this interface; it
/// cannot read the prover's rows or commitments directly.
pub trait ProximityProverChannel {
    /// The prover's declared OOD sample answer.
    fn sample_answer(&self, seed: Fp3) -> Result<Fp3>;
    fn open_old_main(&self, position: u64) -> Result<ProverOpening>;
    fn open_old_mask(&self, position: u64) -> Result<ProverOpening>;
    fn open_switched_main(&self, position: u64) -> Result<ProverOpening>;
    fn open_switched_mask(&self, position: u64) -> Result<ProverOpening>;
    fn open_star(&self, position: u64) -> Result<ProverOpening>;
    fn open_star_mask(&self, position: u64) -> Result<ProverOpening>;
    /// target* computed from the prover's own rows and the verifier's CS
    /// decisions. The verifier absorbs it before drawing lambda.
    fn base_target(&self, positions: &[u64], theta: Fp3) -> Result<Fp3>;
    /// M'/R' computed after lambda. The verifier checks the linear and symbol
    /// relations; it does not recompute them as the protocol.
    fn base_rows(&self, positions: &[u64], theta: Fp3, lambda: Fp3) -> Result<BaseRows>;
}

struct ProverCommitment {
    registry: OracleRegistry,
    openings: Vec<OracleOpening>,
    oracle: OracleId,
}

impl ProverCommitment {
    fn public(&self) -> Result<Commitment> {
        let root = self.registry.get(self.oracle).ok_or(Error::ProverStateMissing)?;
        Ok(Commitment {
            root: root.root,
            start: root.start,
            count: root.count,
        })
    }

    fn open(&self, position: u64) -> Result<ProverOpening> {
        let opening = self
            .openings
            .get(position as usize)
            .ok_or(Error::InvalidIndex)?;
        Ok(ProverOpening {
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

fn commit_row(
    label: Digest,
    code: CodeParams,
    message: &[Fp3],
    tail: &[Fp3],
) -> Result<ProverCommitment> {
    let symbols = encode_row_fp3(&code, message, tail)?;
    let rows: Vec<Vec<u64>> = symbols.iter().map(fp3_words).collect();
    let mut registry = OracleRegistry::new();
    let (oracle, openings) = commit_rows_with_openings(&mut registry, &label, &rows)?;
    Ok(ProverCommitment {
        registry,
        openings,
        oracle,
    })
}

fn open_symbol(row: &ProverCommitment, position: u64) -> Result<Fp3> {
    let opening = row.open(position)?;
    fp3_from_words(&opening.values)
}

/// The prover-side fixture. The plaintext rows and all openings stay here; the
/// verifier only receives `ProximityMessages` and opening responses.
pub struct ProximityProver {
    plan: ProximityPlan,
    statement: ProximityStatement,
    star_main: Vec<Fp3>,
    star_tail: Vec<Fp3>,
    star_mask: Vec<Fp3>,
    star_mask_tail: Vec<Fp3>,
    switched: cs_stage::SwitchedMessage,
    old_main: ProverCommitment,
    old_mask: ProverCommitment,
    switched_main: ProverCommitment,
    switched_mask: ProverCommitment,
    star: ProverCommitment,
    star_mask_row: ProverCommitment,
}

impl ProximityProver {
    pub fn new(
        plan: &ProximityPlan,
        rows: SingleRows,
        star_main: Vec<Fp3>,
        star_tail: Vec<Fp3>,
        star_mask: Vec<Fp3>,
        star_mask_tail: Vec<Fp3>,
    ) -> Result<Self> {
        plan.check()?;
        rows.base_ready()?;
        if rows.code != plan.input_code
            || rows.mask_code != plan.input_mask_code
            || rows.masks.len() != 1
        {
            return Err(Error::StatementMismatch);
        }
        let statement = ProximityStatement::from_rows(&rows)?;
        let switch = plan.stages[0];
        let switched = cs_stage::switched_message(&rows, switch.code, switch.mask_code)?;
        let old_main = commit_row(Digest([0x72; 32]), rows.code, &rows.main, &rows.main_tail)?;
        let old_mask = commit_row(
            Digest([0x73; 32]),
            rows.mask_code,
            &rows.masks[0],
            &rows.mask_tails[0],
        )?;
        let switched_main = commit_row(
            Digest([0x74; 32]),
            switch.code,
            &switched.main,
            &switched.main_tail,
        )?;
        let switched_mask = commit_row(
            Digest([0x75; 32]),
            switch.mask_code,
            &switched.mask,
            &switched.mask_tail,
        )?;
        let base = plan.stages[1];
        let star = commit_row(Digest([0x76; 32]), base.code, &star_main, &star_tail)?;
        let star_mask_row = commit_row(
            Digest([0x77; 32]),
            base.mask_code,
            &star_mask,
            &star_mask_tail,
        )?;
        Ok(Self {
            plan: plan.clone(),
            statement,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
            switched,
            old_main,
            old_mask,
            switched_main,
            switched_mask,
            star,
            star_mask_row,
        })
    }

    pub fn messages(&self) -> Result<ProximityMessages> {
        Ok(ProximityMessages {
            old_main: self.old_main.public()?,
            old_mask: self.old_mask.public()?,
            switched_main: self.switched_main.public()?,
            switched_mask: self.switched_mask.public()?,
            star: self.star.public()?,
            star_mask: self.star_mask_row.public()?,
        })
    }

    fn final_rows(&self, claims: &cs_stage::SwitchedClaims) -> Result<SingleRows> {
        let base = self.plan.stages[1];
        let rows = SingleRows {
            code: base.code,
            mask_code: base.mask_code,
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

    fn own_claims(
        &self,
        positions: &[u64],
        theta: Fp3,
    ) -> Result<(cs_stage::SwitchedClaims, SingleRows)> {
        let switch = self.plan.stages[0];
        let mut authenticated = Vec::with_capacity(positions.len());
        for position in positions {
            let main = open_symbol(&self.old_main, *position)?;
            let mask = open_symbol(&self.old_mask, self.statement.mask_position(*position))?;
            authenticated.push(AuthenticatedSymbols {
                position: *position,
                main,
                mask,
            });
        }
        let claims = cs_stage::register_switch(
            &self.statement,
            switch.code,
            switch.mask_code,
            &authenticated,
            theta,
        )?;
        let final_rows = self.final_rows(&claims)?;
        Ok((claims, final_rows))
    }

    fn prepared(&self, positions: &[u64], theta: Fp3, lambda: Fp3) -> Result<BaseRows> {
        let (_, final_rows) = self.own_claims(positions, theta)?;
        base_stage::prepare(
            &final_rows,
            &self.star_main,
            &self.star_tail,
            &self.star_mask,
            &self.star_mask_tail,
            lambda,
        )
    }
}

impl ProximityProverChannel for ProximityProver {
    fn sample_answer(&self, seed: Fp3) -> Result<Fp3> {
        let switch = self.plan.stages[0];
        evaluate_ood_fp3(
            &switch.code,
            &self.switched.main,
            &self.switched.main_tail,
            seed,
        )
    }

    fn open_old_main(&self, position: u64) -> Result<ProverOpening> {
        self.old_main.open(position)
    }

    fn open_old_mask(&self, position: u64) -> Result<ProverOpening> {
        self.old_mask.open(position)
    }

    fn open_switched_main(&self, position: u64) -> Result<ProverOpening> {
        self.switched_main.open(position)
    }

    fn open_switched_mask(&self, position: u64) -> Result<ProverOpening> {
        self.switched_mask.open(position)
    }

    fn open_star(&self, position: u64) -> Result<ProverOpening> {
        self.star.open(position)
    }

    fn open_star_mask(&self, position: u64) -> Result<ProverOpening> {
        self.star_mask_row.open(position)
    }

    fn base_target(&self, positions: &[u64], theta: Fp3) -> Result<Fp3> {
        Ok(self.prepared(positions, theta, Fp3::ZERO)?.target_star)
    }

    fn base_rows(&self, positions: &[u64], theta: Fp3, lambda: Fp3) -> Result<BaseRows> {
        self.prepared(positions, theta, lambda)
    }
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

fn put_code(writer: &mut CanonicalWriter, code: CodeParams) {
    writer.put_u64(code.log_n as u64);
    writer.put_u64(code.ell);
    writer.put_u64(code.r);
}

fn put_commitment(writer: &mut CanonicalWriter, commitment: &Commitment) {
    writer.put_digest(&commitment.root);
    writer.put_digest(&commitment.start);
    writer.put_u64(commitment.count);
}

fn plan_prefix(
    plan: &ProximityPlan,
    statement: &ProximityStatement,
    messages: &ProximityMessages,
) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(2);
    put_code(&mut writer, plan.input_code);
    put_code(&mut writer, plan.input_mask_code);
    writer.put_u64(plan.stages.len() as u64);
    for stage in &plan.stages {
        writer.put_u64(match stage.role {
            zkie_engine::pcs::request::proximity::StageRole::CodeSwitch => 1,
            zkie_engine::pcs::request::proximity::StageRole::Base => 2,
        });
        put_code(&mut writer, stage.code);
        put_code(&mut writer, stage.mask_code);
        writer.put_u64(stage.k as u64);
        writer.put_u64(stage.h as u64);
        writer.put_u64(stage.answers as u64);
        writer.put_u64(stage.samples as u64);
        writer.put_u64(u64::from(stage.folds));
        writer.put_u64(1);
    }
    put_code(&mut writer, statement.code);
    put_code(&mut writer, statement.mask_code);
    writer.put_u64(statement.main_len);
    writer.put_u64(statement.main_tail_len);
    writer.put_u64(statement.mask_len);
    writer.put_u64(statement.mask_tail_len);
    writer.put_u64(statement.claim.len() as u64);
    for value in &statement.claim {
        writer.put_fp3(value);
    }
    writer.put_u64(statement.mask_claims.len() as u64);
    for value in &statement.mask_claims {
        writer.put_u64(value.len() as u64);
        for value in value {
            writer.put_fp3(value);
        }
    }
    writer.put_fp3(&statement.target);
    for commitment in [
        messages.old_main,
        messages.old_mask,
        messages.switched_main,
        messages.switched_mask,
        messages.star,
        messages.star_mask,
    ] {
        put_commitment(&mut writer, &commitment);
    }
    digest_bytes(&writer.into_bytes())
}

fn push_digest(words: &mut Vec<u64>, digest: &Digest) {
    for chunk in digest.0.chunks_exact(8) {
        words.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
}

#[derive(Clone, Copy)]
enum Slot {
    OldMain,
    OldMask,
    SwitchedMain,
    SwitchedMask,
    Star,
    StarMask,
}

/// The verifier session for the local algebraic component. It owns the
/// independent plan, the statement and its own registry; the prover is an
/// external channel.
pub struct ProximityVerifier<'a> {
    plan: &'a ProximityPlan,
    statement: &'a ProximityStatement,
    messages: ProximityMessages,
    registry: OracleRegistry,
    ids: [OracleId; 6],
    chain: Chain,
    sample_answer: Option<Fp3>,
    sample_oracle_match: bool,
    cs_positions: Vec<u64>,
    theta: Fp3,
    claims: Option<cs_stage::SwitchedClaims>,
    stage_handoff: bool,
    lambda: Option<Fp3>,
    base_positions: Vec<u64>,
    base_linear: bool,
    symbol_checks: bool,
}

impl<'a> ProximityVerifier<'a> {
    pub fn new(
        plan: &'a ProximityPlan,
        statement: &'a ProximityStatement,
        messages: ProximityMessages,
    ) -> Result<Self> {
        plan.check()?;
        statement.check()?;
        if statement.code != plan.input_code || statement.mask_code != plan.input_mask_code {
            return Err(Error::StatementMismatch);
        }
        let expected = [
            (messages.old_main, statement.code.n()),
            (messages.old_mask, statement.mask_code.n()),
            (messages.switched_main, plan.stages[0].code.n()),
            (messages.switched_mask, plan.stages[0].mask_code.n()),
            (messages.star, plan.stages[1].code.n()),
            (messages.star_mask, plan.stages[1].mask_code.n()),
        ];
        let mut registry = OracleRegistry::new();
        let mut ids = Vec::with_capacity(6);
        for (commitment, count) in expected {
            if commitment.count != count {
                return Err(Error::StatementMismatch);
            }
            ids.push(registry.insert(commitment.root, commitment.start, commitment.count)?);
        }
        let mut ids_array = [ids[0]; 6];
        ids_array.copy_from_slice(&ids);
        let prefix = plan_prefix(plan, statement, &messages);
        let chain = Chain::begin(REQUEST_CHAIN_LABEL, &prefix, b"proximity-plan", &[], None)?;
        Ok(Self {
            plan,
            statement,
            messages,
            registry,
            ids: ids_array,
            chain,
            sample_answer: None,
            sample_oracle_match: false,
            cs_positions: Vec::new(),
            theta: Fp3::ZERO,
            claims: None,
            stage_handoff: false,
            lambda: None,
            base_positions: Vec::new(),
            base_linear: false,
            symbol_checks: false,
        })
    }

    fn expected_count(&self, slot: Slot) -> u64 {
        match slot {
            Slot::OldMain => self.statement.code.n(),
            Slot::OldMask => self.statement.mask_code.n(),
            Slot::SwitchedMain => self.plan.stages[0].code.n(),
            Slot::SwitchedMask => self.plan.stages[0].mask_code.n(),
            Slot::Star => self.plan.stages[1].code.n(),
            Slot::StarMask => self.plan.stages[1].mask_code.n(),
        }
    }

    fn open_slot(&self, slot: Slot, position: u64, opening: ProverOpening) -> Result<Fp3> {
        let index = slot as usize;
        let oracle = self.ids[index];
        if opening.index != position {
            return Err(Error::InvalidIndex);
        }
        let opening = OracleOpening {
            oracle,
            index: opening.index,
            values: opening.values,
            path: opening.path,
        };
        verify_opening(&self.registry, &opening, self.expected_count(slot))?;
        fp3_from_words(&opening.values)
    }

    /// CS: sends the switched commitments, draws the sample seed, the sample
    /// positions and theta, authenticates the old openings and registers the
    /// pending claims.
    pub fn run_code_switch(
        &mut self,
        prover: &dyn ProximityProverChannel,
        oracle: &dyn SampleOracle,
    ) -> Result<()> {
        let switch = self.plan.stages[0];
        let mut send_words = Vec::new();
        push_digest(&mut send_words, &self.messages.switched_main.root);
        push_digest(&mut send_words, &self.messages.switched_mask.root);
        absorb(&mut self.chain, 1, &send_words)?;
        let sample_seed = self.chain.draw_e()?;
        let answer = prover.sample_answer(sample_seed)?;
        self.sample_answer = Some(answer);
        absorb(&mut self.chain, 2, &fp3_words(&answer))?;
        let expected = oracle.expected(sample_seed)?;
        self.sample_oracle_match = answer == expected;

        self.cs_positions = draw_positions(
            &mut self.chain,
            switch.samples as usize,
            self.statement.code.log_n,
        )?;
        self.theta = self.chain.draw_e()?;
        let mut authenticated = Vec::with_capacity(self.cs_positions.len());
        for position in &self.cs_positions {
            let main = self.open_slot(Slot::OldMain, *position, prover.open_old_main(*position)?)?;
            let mask_position = self.statement.mask_position(*position);
            let mask =
                self.open_slot(Slot::OldMask, mask_position, prover.open_old_mask(mask_position)?)?;
            authenticated.push(AuthenticatedSymbols {
                position: *position,
                main,
                mask,
            });
        }
        let claims = cs_stage::register_switch(
            self.statement,
            switch.code,
            switch.mask_code,
            &authenticated,
            self.theta,
        )?;
        self.stage_handoff = claims
            .base_ready(self.plan.stages[1].code, self.plan.stages[1].mask_code)
            .is_ok();
        self.claims = Some(claims);
        Ok(())
    }

    /// Local BASE algebra: sends the star commitments and target*, draws lambda, absorbs all
    /// M' then all R', draws the positions, authenticates the final and star
    /// openings, and checks the linear and symbol relations.
    pub fn run_base(&mut self, prover: &dyn ProximityProverChannel) -> Result<()> {
        let claims = self.claims.as_ref().ok_or(Error::InvalidSessionState)?;
        let base = self.plan.stages[1];
        let target_star = prover.base_target(&self.cs_positions, self.theta)?;
        let mut star_words = Vec::new();
        push_digest(&mut star_words, &self.messages.star.root);
        push_digest(&mut star_words, &self.messages.star_mask.root);
        star_words.extend(fp3_words(&target_star));
        absorb(&mut self.chain, 3, &star_words)?;
        let lambda = self.chain.draw_e()?;
        self.lambda = Some(lambda);

        let prepared = prover.base_rows(&self.cs_positions, self.theta, lambda)?;
        if prepared.lambda != lambda || prepared.target_star != target_star {
            return Err(Error::StatementMismatch);
        }
        // All M' values are sent first, then all R' values.
        let mut m_words = Vec::new();
        for value in &prepared.main_prime {
            m_words.extend(fp3_words(value));
        }
        for value in &prepared.mask_prime {
            m_words.extend(fp3_words(value));
        }
        absorb(&mut self.chain, 4, &m_words)?;
        let mut r_words = Vec::new();
        for value in &prepared.main_tail_prime {
            r_words.extend(fp3_words(value));
        }
        for value in &prepared.mask_tail_prime {
            r_words.extend(fp3_words(value));
        }
        absorb(&mut self.chain, 5, &r_words)?;
        self.base_positions =
            draw_positions(&mut self.chain, base.samples as usize, base.code.log_n)?;

        self.base_linear = base_stage::linear_holds(
            &claims.combined.main,
            &claims.combined.mask,
            claims.combined.target,
            &prepared,
        )?;
        let encoded_main =
            encode_row_fp3(&base.code, &prepared.main_prime, &prepared.main_tail_prime)?;
        let encoded_mask = encode_row_fp3(
            &base.mask_code,
            &prepared.mask_prime,
            &prepared.mask_tail_prime,
        )?;
        let base_mask_position = |position: u64| position % base.mask_code.n();
        let mut symbol_checks = true;
        for position in &self.base_positions {
            let old_main =
                self.open_slot(Slot::SwitchedMain, *position, prover.open_switched_main(*position)?)?;
            let old_mask = self.open_slot(
                Slot::SwitchedMask,
                base_mask_position(*position),
                prover.open_switched_mask(base_mask_position(*position))?,
            )?;
            let star = self.open_slot(Slot::Star, *position, prover.open_star(*position)?)?;
            let star_mask = self.open_slot(
                Slot::StarMask,
                base_mask_position(*position),
                prover.open_star_mask(base_mask_position(*position))?,
            )?;
            if !base_stage::symbol_holds(
                encoded_main
                    .get(*position as usize)
                    .copied()
                    .ok_or(Error::InvalidIndex)?,
                star,
                lambda,
                old_main,
            ) || !base_stage::symbol_holds(
                encoded_mask
                    .get(base_mask_position(*position) as usize)
                    .copied()
                    .ok_or(Error::InvalidIndex)?,
                star_mask,
                lambda,
                old_mask,
            ) {
                symbol_checks = false;
            }
        }
        self.symbol_checks = symbol_checks;
        Ok(())
    }

    pub fn finish(self) -> Result<ProximityOutcome> {
        let claims = self.claims.as_ref().ok_or(Error::InvalidSessionState)?;
        if !self.sample_oracle_match
            || !self.stage_handoff
            || !self.base_linear
            || !self.symbol_checks
        {
            return Err(Error::InvalidProof);
        }
        Ok(ProximityOutcome {
            old_main_root: self.messages.old_main.root,
            old_mask_root: self.messages.old_mask.root,
            switched_root: self.messages.switched_main.root,
            star_root: self.messages.star.root,
            cs_positions: self.cs_positions,
            base_positions: self.base_positions,
            theta: self.theta,
            lambda: self.lambda.ok_or(Error::InvalidSessionState)?,
            sample_answer: self.sample_answer.ok_or(Error::InvalidSessionState)?,
            registration_count: claims.claims.len(),
            stage_handoff: self.stage_handoff,
            base_linear: self.base_linear,
            symbol_checks: self.symbol_checks,
            sample_oracle_match: self.sample_oracle_match,
            component_checks_passed: true,
            verified: true,
        })
    }
}

/// Runs the local CS/BASE algebraic component. The caller supplies the plan,
/// the statement, the public commitments, a prover channel and a test-side
/// sample oracle.
///
/// The statement is caller-supplied. A statement built from a field projection
/// fixture (for example a FOLD output whose fresh term is uncovered) exercises
/// only the local algebraic behaviour of the projected data. `ComponentComplete`
/// does not establish a FOLD-to-CS/BASE handoff, does not prove the pending CS
/// claims and does not accept a proof.
pub fn verify_proximity_session(
    plan: &ProximityPlan,
    statement: &ProximityStatement,
    messages: ProximityMessages,
    prover: &dyn ProximityProverChannel,
    oracle: &dyn SampleOracle,
) -> Result<ProximityOutcome> {
    let mut verifier = ProximityVerifier::new(plan, statement, messages)?;
    verifier.run_code_switch(prover, oracle)?;
    verifier.run_base(prover)?;
    verifier.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::proximity::{ChoiceReference, ProximityStage, StageRole};

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
                    folds: false,
                    choice: ChoiceReference::VerifierPlan,
                },
                ProximityStage {
                    role: StageRole::Base,
                    code: new_code,
                    mask_code: new_code,
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

    /// Test-side oracle built from the fixed fixture rows and the switch code.
    /// It never reads or calls the prover channel; the values are fixed before
    /// the prover is constructed.
    struct FixedOracle {
        code: CodeParams,
        main: Vec<Fp3>,
        tail: Vec<Fp3>,
    }

    impl SampleOracle for FixedOracle {
        fn expected(&self, seed: Fp3) -> Result<Fp3> {
            evaluate_ood_fp3(&self.code, &self.main, &self.tail, seed)
        }
    }

    fn fixture() -> (
        ProximityPlan,
        ProximityStatement,
        ProximityMessages,
        ProximityProver,
        FixedOracle,
    ) {
        let rows = fixture_rows();
        let statement = ProximityStatement::from_rows(&rows).unwrap();
        let plan = plan();
        let switch = plan.stages[0];
        let switched = cs_stage::switched_message(&rows, switch.code, switch.mask_code).unwrap();
        let oracle = FixedOracle {
            code: switch.code,
            main: switched.main.clone(),
            tail: switched.main_tail.clone(),
        };
        let prover = ProximityProver::new(
            &plan,
            rows,
            vec![Fp3::from_u64(11); 6],
            vec![Fp3::from_u64(12)],
            vec![Fp3::from_u64(13); 6],
            vec![Fp3::from_u64(14)],
        )
        .unwrap();
        let messages = prover.messages().unwrap();
        (plan, statement, messages, prover, oracle)
    }

    fn session(
        plan: &ProximityPlan,
        statement: &ProximityStatement,
        messages: ProximityMessages,
        prover: &dyn ProximityProverChannel,
        oracle: &dyn SampleOracle,
    ) -> Result<ProximityOutcome> {
        verify_proximity_session(plan, statement, messages, prover, oracle)
    }

    #[derive(Clone, Copy)]
    enum Tamper {
        SampleAnswer,
        BaseTarget,
        BaseRows,
        OldMainValue,
        OldMainPath,
        OldMainIndex,
    }

    struct Tampered<'a> {
        inner: &'a dyn ProximityProverChannel,
        mode: Tamper,
    }

    impl ProximityProverChannel for Tampered<'_> {
        fn sample_answer(&self, seed: Fp3) -> Result<Fp3> {
            let value = self.inner.sample_answer(seed)?;
            Ok(match self.mode {
                Tamper::SampleAnswer => value.add(&Fp3::ONE),
                _ => value,
            })
        }
        fn open_old_main(&self, position: u64) -> Result<ProverOpening> {
            let mut opening = self.inner.open_old_main(position)?;
            match self.mode {
                Tamper::OldMainValue => {
                    opening.values[0] = opening.values[0].wrapping_add(1);
                }
                Tamper::OldMainPath => {
                    opening.path[0] = Digest::ZERO;
                }
                Tamper::OldMainIndex => {
                    opening.index = opening.index.wrapping_add(1);
                }
                _ => {}
            }
            Ok(opening)
        }
        fn open_old_mask(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_old_mask(position)
        }
        fn open_switched_main(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_switched_main(position)
        }
        fn open_switched_mask(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_switched_mask(position)
        }
        fn open_star(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_star(position)
        }
        fn open_star_mask(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_star_mask(position)
        }
        fn base_target(&self, positions: &[u64], theta: Fp3) -> Result<Fp3> {
            let value = self.inner.base_target(positions, theta)?;
            Ok(match self.mode {
                Tamper::BaseTarget => value.add(&Fp3::ONE),
                _ => value,
            })
        }
        fn base_rows(&self, positions: &[u64], theta: Fp3, lambda: Fp3) -> Result<BaseRows> {
            let mut rows = self.inner.base_rows(positions, theta, lambda)?;
            if let Tamper::BaseRows = self.mode {
                rows.main_prime[0] = rows.main_prime[0].add(&Fp3::ONE);
            }
            Ok(rows)
        }
    }

    struct Replaced<'a> {
        inner: &'a dyn ProximityProverChannel,
        switched: &'a ProverCommitment,
    }

    impl ProximityProverChannel for Replaced<'_> {
        fn sample_answer(&self, seed: Fp3) -> Result<Fp3> {
            self.inner.sample_answer(seed)
        }
        fn open_old_main(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_old_main(position)
        }
        fn open_old_mask(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_old_mask(position)
        }
        fn open_switched_main(&self, position: u64) -> Result<ProverOpening> {
            self.switched.open(position)
        }
        fn open_switched_mask(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_switched_mask(position)
        }
        fn open_star(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_star(position)
        }
        fn open_star_mask(&self, position: u64) -> Result<ProverOpening> {
            self.inner.open_star_mask(position)
        }
        fn base_target(&self, positions: &[u64], theta: Fp3) -> Result<Fp3> {
            self.inner.base_target(positions, theta)
        }
        fn base_rows(&self, positions: &[u64], theta: Fp3, lambda: Fp3) -> Result<BaseRows> {
            self.inner.base_rows(positions, theta, lambda)
        }
    }

    struct Tracked<'a> {
        inner: &'a dyn ProximityProverChannel,
        order: RefCell<Vec<&'static str>>,
    }

    impl ProximityProverChannel for Tracked<'_> {
        fn sample_answer(&self, seed: Fp3) -> Result<Fp3> {
            self.order.borrow_mut().push("sample_answer");
            self.inner.sample_answer(seed)
        }
        fn open_old_main(&self, position: u64) -> Result<ProverOpening> {
            self.order.borrow_mut().push("old_main");
            self.inner.open_old_main(position)
        }
        fn open_old_mask(&self, position: u64) -> Result<ProverOpening> {
            self.order.borrow_mut().push("old_mask");
            self.inner.open_old_mask(position)
        }
        fn open_switched_main(&self, position: u64) -> Result<ProverOpening> {
            self.order.borrow_mut().push("switched_main");
            self.inner.open_switched_main(position)
        }
        fn open_switched_mask(&self, position: u64) -> Result<ProverOpening> {
            self.order.borrow_mut().push("switched_mask");
            self.inner.open_switched_mask(position)
        }
        fn open_star(&self, position: u64) -> Result<ProverOpening> {
            self.order.borrow_mut().push("star");
            self.inner.open_star(position)
        }
        fn open_star_mask(&self, position: u64) -> Result<ProverOpening> {
            self.order.borrow_mut().push("star_mask");
            self.inner.open_star_mask(position)
        }
        fn base_target(&self, positions: &[u64], theta: Fp3) -> Result<Fp3> {
            self.order.borrow_mut().push("base_target");
            self.inner.base_target(positions, theta)
        }
        fn base_rows(&self, positions: &[u64], theta: Fp3, lambda: Fp3) -> Result<BaseRows> {
            self.order.borrow_mut().push("base_rows");
            self.inner.base_rows(positions, theta, lambda)
        }
    }

    #[test]
    fn component_chain_is_deterministic_and_verified() {
        let (plan, statement, messages, prover, oracle) = fixture();
        let first = session(&plan, &statement, messages, &prover, &oracle).unwrap();
        let second = session(&plan, &statement, messages, &prover, &oracle).unwrap();
        assert_eq!(first, second);
        assert!(first.stage_handoff && first.sample_oracle_match && first.base_linear && first.symbol_checks && first.component_checks_passed);
        assert_eq!(first.component_complete().unwrap(), ProximityComponent::ComponentComplete);
        assert_eq!(first.registration_count, 5);
    }

    #[test]
    fn plan_or_statement_shape_mismatch_fails_closed() {
        let (plan, statement, messages, _prover, _oracle) = fixture();
        let mut bad_plan = plan.clone();
        bad_plan.stages.truncate(1);
        assert!(matches!(
            ProximityVerifier::new(&bad_plan, &statement, messages),
            Err(Error::InvalidConfig)
        ));
        let mut bad_statement = statement.clone();
        bad_statement.code = build_code_plan(3, 6, 1).unwrap();
        assert!(matches!(
            ProximityVerifier::new(&plan, &bad_statement, messages),
            Err(Error::ShapeMismatch)
        ));
    }

    #[test]
    fn wrong_commitment_is_rejected_at_the_commitment_check() {
        // Class 1: the commitment object itself is wrong, so the failure is
        // reached before any claim relation is built.
        let (plan, statement, messages, prover, oracle) = fixture();
        let mut extra_registry = OracleRegistry::new();
        let extra_rows = vec![vec![0u64]; 16];
        let (extra, _) = commit_rows_with_openings(
            &mut extra_registry,
            &Digest([0x91; 32]),
            &extra_rows,
        )
        .unwrap();
        let extra_root = extra_registry.get(extra).unwrap();
        let mut wrong_count = messages;
        wrong_count.switched_main = Commitment {
            root: extra_root.root,
            start: extra_root.start,
            count: extra_root.count,
        };
        assert!(matches!(
            ProximityVerifier::new(&plan, &statement, wrong_count),
            Err(Error::StatementMismatch)
        ));
        // A valid same-shape commitment cannot authenticate openings produced
        // for the original switched commitment.
        let mut swapped = messages;
        swapped.switched_main = messages.old_main;
        assert!(matches!(
            session(&plan, &statement, swapped, &prover, &oracle),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn valid_commitment_with_a_wrong_statement_is_rejected_at_the_relation_check() {
        // Class 2: all commitments are valid and authenticate; the statement
        // is wrong, so the failure is reached at the local algebraic relation.
        let (plan, statement, messages, prover, oracle) = fixture();
        let mut wrong_statement = statement.clone();
        wrong_statement.target = wrong_statement.target.add(&Fp3::ONE);
        assert!(matches!(
            session(&plan, &wrong_statement, messages, &prover, &oracle),
            Err(Error::InvalidProof)
        ));
        let mut wrong_statement = statement.clone();
        wrong_statement.claim[0] = wrong_statement.claim[0].add(&Fp3::ONE);
        assert!(matches!(
            session(&plan, &wrong_statement, messages, &prover, &oracle),
            Err(Error::InvalidProof)
        ));
        assert!(session(&plan, &statement, messages, &prover, &oracle).is_ok());
    }

    #[test]
    fn valid_commitment_with_a_wrong_prover_message_is_rejected() {
        let (plan, statement, messages, prover, oracle) = fixture();
        let target = Tampered {
            inner: &prover,
            mode: Tamper::BaseTarget,
        };
        assert!(matches!(
            session(&plan, &statement, messages, &target, &oracle),
            Err(Error::StatementMismatch)
        ));
        let rows = Tampered {
            inner: &prover,
            mode: Tamper::BaseRows,
        };
        assert!(matches!(
            session(&plan, &statement, messages, &rows, &oracle),
            Err(Error::InvalidProof)
        ));
        let sample = Tampered {
            inner: &prover,
            mode: Tamper::SampleAnswer,
        };
        assert!(matches!(
            session(&plan, &statement, messages, &sample, &oracle),
            Err(Error::InvalidProof)
        ));
        assert!(session(&plan, &statement, messages, &prover, &oracle).is_ok());
    }

    #[test]
    fn opening_tampering_is_rejected_before_any_relation_check() {
        let (plan, statement, messages, prover, oracle) = fixture();
        for mode in [Tamper::OldMainValue, Tamper::OldMainPath, Tamper::OldMainIndex] {
            let channel = Tampered { inner: &prover, mode };
            assert!(session(&plan, &statement, messages, &channel, &oracle).is_err());
        }
        assert!(session(&plan, &statement, messages, &prover, &oracle).is_ok());
    }

    #[test]
    fn cs_output_replacement_with_a_valid_commitment_is_rejected() {
        let (plan, statement, messages, prover, oracle) = fixture();
        // A valid commitment with the same shape cannot authenticate the old
        // switched openings; the opening check rejects it.
        let mut swapped = messages;
        swapped.switched_main = messages.old_main;
        assert!(session(&plan, &statement, swapped, &prover, &oracle).is_err());
        assert!(session(&plan, &statement, messages, &prover, &oracle).is_ok());
    }

    #[test]
    fn a_consistent_replaced_switched_message_is_caught_by_base_symbols() {
        let (plan, statement, messages, prover, oracle) = fixture();
        let mut changed = prover.switched.main.clone();
        for value in changed.iter_mut() {
            *value = value.add(&Fp3::ONE);
        }
        let replacement = commit_row(
            Digest([0x9a; 32]),
            plan.stages[0].code,
            &changed,
            &prover.switched.main_tail,
        )
        .unwrap();
        let mut replaced_messages = messages;
        replaced_messages.switched_main = replacement.public().unwrap();
        let channel = Replaced {
            inner: &prover,
            switched: &replacement,
        };
        assert!(session(&plan, &statement, replaced_messages, &channel, &oracle).is_err());
        assert!(session(&plan, &statement, messages, &prover, &oracle).is_ok());
    }

    #[test]
    fn challenge_order_is_fixed() {
        let (plan, statement, messages, prover, oracle) = fixture();
        let tracked = Tracked {
            inner: &prover,
            order: RefCell::new(Vec::new()),
        };
        assert!(session(&plan, &statement, messages, &tracked, &oracle).is_ok());
        let order = tracked.order.borrow();
        let first_base_target = order.iter().position(|entry| *entry == "base_target").unwrap();
        let first_switched = order.iter().position(|entry| *entry == "switched_main").unwrap();
        let first_sample = order.iter().position(|entry| *entry == "sample_answer").unwrap();
        let first_old = order.iter().position(|entry| *entry == "old_main").unwrap();
        assert!(first_sample < first_old);
        assert!(first_old < first_base_target);
        assert!(first_base_target < first_switched);
        assert_eq!(order.iter().filter(|entry| **entry == "base_rows").count(), 1);
        assert_eq!(order.iter().filter(|entry| **entry == "base_target").count(), 1);
    }

    struct Refusing;

    impl ProximityProverChannel for Refusing {
        fn sample_answer(&self, _seed: Fp3) -> Result<Fp3> {
            Err(Error::ProverStateMissing)
        }
        fn open_old_main(&self, _position: u64) -> Result<ProverOpening> {
            Err(Error::ProverStateMissing)
        }
        fn open_old_mask(&self, _position: u64) -> Result<ProverOpening> {
            Err(Error::ProverStateMissing)
        }
        fn open_switched_main(&self, _position: u64) -> Result<ProverOpening> {
            Err(Error::ProverStateMissing)
        }
        fn open_switched_mask(&self, _position: u64) -> Result<ProverOpening> {
            Err(Error::ProverStateMissing)
        }
        fn open_star(&self, _position: u64) -> Result<ProverOpening> {
            Err(Error::ProverStateMissing)
        }
        fn open_star_mask(&self, _position: u64) -> Result<ProverOpening> {
            Err(Error::ProverStateMissing)
        }
        fn base_target(&self, _positions: &[u64], _theta: Fp3) -> Result<Fp3> {
            Err(Error::ProverStateMissing)
        }
        fn base_rows(
            &self,
            _positions: &[u64],
            _theta: Fp3,
            _lambda: Fp3,
        ) -> Result<BaseRows> {
            Err(Error::ProverStateMissing)
        }
    }

    #[test]
    fn a_missing_message_never_completes_the_component() {
        let (plan, statement, messages, _prover, oracle) = fixture();
        assert!(session(&plan, &statement, messages, &Refusing, &oracle).is_err());
    }

    #[test]
    fn public_booleans_alone_do_not_form_the_component() {
        let (plan, statement, messages, prover, oracle) = fixture();
        let mut outcome = session(&plan, &statement, messages, &prover, &oracle).unwrap();
        assert!(outcome.stage_handoff && outcome.sample_oracle_match && outcome.base_linear);
        outcome.verified = false;
        assert!(outcome.component_complete().is_err());
    }

    #[test]
    fn decision_inputs_are_bound_in_the_prefix() {
        let (plan, statement, messages, _prover, _oracle) = fixture();
        let base = plan_prefix(&plan, &statement, &messages);
        let mut changed_plan = plan.clone();
        changed_plan.stages[0].samples += 1;
        assert_ne!(base, plan_prefix(&changed_plan, &statement, &messages));
        let mut changed_statement = statement.clone();
        changed_statement.claim[0] = changed_statement.claim[0].add(&Fp3::ONE);
        assert_ne!(base, plan_prefix(&plan, &changed_statement, &messages));
        let mut changed_messages = messages;
        changed_messages.old_main.count += 1;
        assert_ne!(base, plan_prefix(&plan, &statement, &changed_messages));
    }
}
