//! Isolated fixture BASE final-state check wired to a FOLD output continuation.
//!
//! The verifier's claim, target and FOLD identity come from the same verified
//! FOLD session that produced the fresh-mask output; the old main/mask rows come
//! from the FOLD prover's end rows, not from unrelated fixture rows. The BASE
//! star rows are the BASE prover's own committed fixture rows. The chain is
//! moved from the FOLD session into this session, so the commitment/target
//! message, lambda, M'/R' message and position draws all use one continuous
//! transcript.
//!
//! The supported range is the local no-kept BASE algebra with the FOLD fresh
//! row: the five row commitments (old main, old mask, star main, star mask and
//! the folded FOLD fresh row) are sent with the FOLD identity and target*
//! before lambda; M'/R' and the fresh combined row are fixed before the
//! position draw; the openings are authenticated against the committed roots;
//! the extended linear check and the encoded-symbol checks against those
//! openings are recorded. Missing any precondition fails closed.
//!
//! Root source and binding status: the BASE builds a new fixture fresh-row
//! root, rebuilds the old main/mask roots from the same prover copies and uses
//! a same-prover expectation with a second independent commit. These roots are
//! approved only for this local fixture experiment. They are not a production
//! mechanism, and root binding is not completed: the fresh-row root is not
//! authenticated by the FOLD fresh root. The result stays pending on that
//! boundary and must not be read as a complete root binding or reference
//! equivalence.
//!
//! The reference path is FOLD -> CS -> final SingleRows -> BASE. The extra
//! commitment and the zero-star third row implemented here are not the
//! reference protocol and are not an equivalence proof. Later wiring must use a
//! reference-verifiable commitment source and the reference folding/conversion
//! relation, not this fixture extension.
//!
//! A single equality at one fixed lambda does not show that every wrong
//! declaration is rejected and does not give an IOPP security bound; wrong
//! declarations can cancel through aggregation. The fixed inconsistent sample
//! in the tests is crafted to fail at the target layer; it is not a general
//! rejection guarantee. The verification entry consumes a failed check and
//! returns a typed rejection; a passed local check remains pending.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::base_stage::{self, BaseRows};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::code::{encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3};
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_engine::pcs::request::read::draw_positions;
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};
use crate::proof::fold_handoff::FoldHandoff;

/// One commitment message for a BASE row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseCommitment {
    pub root: Digest,
    pub start: Digest,
    pub count: u64,
}

/// The verifier's independently committed fixture expectations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseExpectation {
    pub fold_identity: Digest,
    pub old_main: BaseCommitment,
    pub old_mask: BaseCommitment,
    pub star: BaseCommitment,
    pub star_mask: BaseCommitment,
    /// The folded FOLD fresh row. It is committed by this fixture; the FOLD
    /// fresh root does not authenticate this folded row.
    pub fresh: BaseCommitment,
}

/// The prover's BASE commitment message: the FOLD identity, the four row
/// commitments and target*. It is fixed before lambda.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseMessages {
    pub fold_identity: Digest,
    pub old_main: BaseCommitment,
    pub old_mask: BaseCommitment,
    pub star: BaseCommitment,
    pub star_mask: BaseCommitment,
    pub fresh: BaseCommitment,
    pub target_star: Fp3,
}

/// One opening response for a BASE row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseOpening {
    pub index: u64,
    pub values: Vec<u64>,
    pub path: Vec<Digest>,
}

/// The four opening lists, in position order for each row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseOpenings {
    pub old_main: Vec<BaseOpening>,
    pub old_mask: Vec<BaseOpening>,
    pub star: Vec<BaseOpening>,
    pub star_mask: Vec<BaseOpening>,
    pub fresh: Vec<BaseOpening>,
}

/// The BASE message rows including the FOLD fresh row. The fresh row has a
/// zero star row, so its combined message is lambda times the fresh message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseFinalRows {
    pub base: BaseRows,
    pub fresh_prime: Vec<Fp3>,
    pub fresh_tail_prime: Vec<Fp3>,
}

/// The layer at which a well-formed but inconsistent BASE sample is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseRejection {
    TargetLinear,
    Symbol,
}

/// The result of the local fixture BASE wiring. It is pending, not a complete
/// BASE and not an accepted proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseFinalOutcome {
    pub fold_identity: Digest,
    pub old_main_root: Digest,
    pub old_mask_root: Digest,
    pub star_root: Digest,
    pub star_mask_root: Digest,
    pub fresh_root: Digest,
    pub lambda: Fp3,
    pub target_star: Fp3,
    pub base_positions: Vec<u64>,
    pub base_linear: bool,
    pub symbol_checks: bool,
    pub openings_verified: bool,
    pub rejection: Option<BaseRejection>,
    pub pending: bool,
    pub complete_base: bool,
    pub fresh_row_committed: bool,
    pub fresh_term_checked: bool,
    pub fresh_term_authenticated_by_fold_root: bool,
    pub projection_only: bool,
    pub chain_handoff: bool,
    pub entry_rounds: u64,
    pub entry_value: Digest,
    pub commit_rounds: u64,
    pub commit_value: Digest,
    pub rows_rounds: u64,
    pub rows_value: Digest,
}

/// The typed rejection of a well-formed but inconsistent local check. It
/// carries the fixed challenge/conditions and the rejection layer. It is a
/// local fixture rejection, not a general wrong-declaration guarantee.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseFinalRejection {
    pub layer: BaseRejection,
    pub fold_identity: Digest,
    pub old_main_root: Digest,
    pub old_mask_root: Digest,
    pub star_root: Digest,
    pub star_mask_root: Digest,
    pub fresh_root: Digest,
    pub lambda: Fp3,
    pub target_star: Fp3,
    pub base_positions: Vec<u64>,
    pub base_linear: bool,
    pub symbol_checks: bool,
}

/// The explicit result of the local check: a passed local result (still
/// pending) or a typed rejection consumed by the verification entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaseFinalCheck {
    Passed(BaseFinalOutcome),
    Rejected(BaseFinalRejection),
}

fn inner(left: &[Fp3], right: &[Fp3]) -> Result<Fp3> {
    if left.len() != right.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (a, b) in left.iter().zip(right) {
        sum = sum.add(&a.mul(b));
    }
    Ok(sum)
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

fn put_code(writer: &mut CanonicalWriter, code: CodeParams) {
    writer.put_u64(code.log_n as u64);
    writer.put_u64(code.ell);
    writer.put_u64(code.r);
}

fn push_digest(words: &mut Vec<u64>, digest: &Digest) {
    for chunk in digest.0.chunks_exact(8) {
        words.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
}

fn push_commitment(words: &mut Vec<u64>, commitment: &BaseCommitment) {
    push_digest(words, &commitment.root);
    push_digest(words, &commitment.start);
    words.push(commitment.count);
}

/// The FOLD identity absorbed by the BASE commitment message. It covers the
/// handoff shape, identity fields, challenges, plaintext claim and target.
pub fn fold_identity(handoff: &FoldHandoff) -> Digest {
    let mut writer = CanonicalWriter::new();
    put_code(&mut writer, handoff.code);
    put_code(&mut writer, handoff.mask_code);
    writer.put_u64(handoff.group);
    writer.put_u64(handoff.members.len() as u64);
    for member in &handoff.members {
        writer.put_u64(*member);
    }
    writer.put_u64(handoff.k as u64);
    writer.put_u64(handoff.h as u64);
    writer.put_u64(handoff.fresh);
    writer.put_digest(&handoff.fresh_root);
    writer.put_u64(handoff.tree_number);
    writer.put_u64(handoff.gamma.len() as u64);
    for gamma in &handoff.gamma {
        writer.put_fp3(gamma);
    }
    writer.put_fp3(&handoff.eps);
    writer.put_fp3(&handoff.sum);
    writer.put_fp3(&handoff.kept_factor);
    writer.put_u64(handoff.kept_rows);
    writer.put_u64(handoff.main_claim.len() as u64);
    for value in &handoff.main_claim {
        writer.put_fp3(value);
    }
    writer.put_u64(handoff.mask_claim.len() as u64);
    for value in &handoff.mask_claim {
        writer.put_fp3(value);
    }
    writer.put_fp3(&handoff.target);
    writer.put_u64(handoff.fresh_claim.len() as u64);
    for value in &handoff.fresh_claim {
        writer.put_fp3(value);
    }
    digest_bytes(&writer.into_bytes())
}

struct ProverCommitment {
    registry: OracleRegistry,
    openings: Vec<OracleOpening>,
    oracle: OracleId,
}

impl ProverCommitment {
    fn public(&self) -> Result<BaseCommitment> {
        let root = self
            .registry
            .get(self.oracle)
            .ok_or(Error::ProverStateMissing)?;
        Ok(BaseCommitment {
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

/// The fixture BASE prover. It holds the plaintext old rows, star rows,
/// commitments and openings; only the messages and opening responses cross to
/// the verifier.
pub struct BaseProver {
    rows: SingleRows,
    fresh_row: Vec<Fp3>,
    fresh_tail: Vec<Fp3>,
    star_main: Vec<Fp3>,
    star_tail: Vec<Fp3>,
    star_mask: Vec<Fp3>,
    star_mask_tail: Vec<Fp3>,
    old_main: ProverCommitment,
    old_mask: ProverCommitment,
    star: ProverCommitment,
    star_mask_row: ProverCommitment,
    fresh: ProverCommitment,
    expectation: BaseExpectation,
    target_shift: Fp3,
}

impl BaseProver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: SingleRows,
        fold_identity: Digest,
        fresh_claim: Vec<Fp3>,
        fresh_row: Vec<Fp3>,
        fresh_tail: Vec<Fp3>,
        star_main: Vec<Fp3>,
        star_tail: Vec<Fp3>,
        star_mask: Vec<Fp3>,
        star_mask_tail: Vec<Fp3>,
        target_shift: Fp3,
    ) -> Result<Self> {
        rows.base_ready()?;
        if rows.masks.len() != 1 {
            return Err(Error::UnsupportedPrimitive);
        }
        if star_main.len() as u64 != rows.code.ell
            || star_tail.len() as u64 != rows.code.r
            || star_mask.len() as u64 != rows.mask_code.ell
            || star_mask_tail.len() as u64 != rows.mask_code.r
            || fresh_claim.len() as u64 != rows.mask_code.ell
            || fresh_row.len() as u64 != rows.mask_code.ell
            || fresh_tail.len() as u64 != rows.mask_code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let old_main = commit_row(Digest([0x72; 32]), rows.code, &rows.main, &rows.main_tail)?;
        let old_mask = commit_row(
            Digest([0x73; 32]),
            rows.mask_code,
            &rows.masks[0],
            &rows.mask_tails[0],
        )?;
        let star = commit_row(Digest([0x76; 32]), rows.code, &star_main, &star_tail)?;
        let star_mask_row = commit_row(
            Digest([0x77; 32]),
            rows.mask_code,
            &star_mask,
            &star_mask_tail,
        )?;
        let fresh = commit_row(
            Digest([0x78; 32]),
            rows.mask_code,
            &fresh_row,
            &fresh_tail,
        )?;
        let expected_old_main =
            commit_row(Digest([0x72; 32]), rows.code, &rows.main, &rows.main_tail)?;
        let expected_old_mask = commit_row(
            Digest([0x73; 32]),
            rows.mask_code,
            &rows.masks[0],
            &rows.mask_tails[0],
        )?;
        let expected_star = commit_row(Digest([0x76; 32]), rows.code, &star_main, &star_tail)?;
        let expected_star_mask = commit_row(
            Digest([0x77; 32]),
            rows.mask_code,
            &star_mask,
            &star_mask_tail,
        )?;
        let expected_fresh = commit_row(
            Digest([0x78; 32]),
            rows.mask_code,
            &fresh_row,
            &fresh_tail,
        )?;
        let expectation = BaseExpectation {
            fold_identity,
            old_main: expected_old_main.public()?,
            old_mask: expected_old_mask.public()?,
            star: expected_star.public()?,
            star_mask: expected_star_mask.public()?,
            fresh: expected_fresh.public()?,
        };
        Ok(Self {
            rows,
            fresh_row,
            fresh_tail,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
            old_main,
            old_mask,
            star,
            star_mask_row,
            fresh,
            expectation,
            target_shift,
        })
    }

    pub fn expectation(&self) -> BaseExpectation {
        self.expectation
    }

    pub fn messages(&self) -> Result<BaseMessages> {
        let prepared = base_stage::prepare(
            &self.rows,
            &self.star_main,
            &self.star_tail,
            &self.star_mask,
            &self.star_mask_tail,
            Fp3::ZERO,
        )?;
        Ok(BaseMessages {
            fold_identity: self.expectation.fold_identity,
            old_main: self.old_main.public()?,
            old_mask: self.old_mask.public()?,
            star: self.star.public()?,
            star_mask: self.star_mask_row.public()?,
            fresh: self.fresh.public()?,
            target_star: prepared.target_star.add(&self.target_shift),
        })
    }

    pub fn base_rows(&self, lambda: Fp3) -> Result<BaseFinalRows> {
        let mut prepared = base_stage::prepare(
            &self.rows,
            &self.star_main,
            &self.star_tail,
            &self.star_mask,
            &self.star_mask_tail,
            lambda,
        )?;
        prepared.target_star = prepared.target_star.add(&self.target_shift);
        let fresh_prime = self
            .fresh_row
            .iter()
            .map(|value| lambda.mul(value))
            .collect();
        let fresh_tail_prime = self
            .fresh_tail
            .iter()
            .map(|value| lambda.mul(value))
            .collect();
        Ok(BaseFinalRows {
            base: prepared,
            fresh_prime,
            fresh_tail_prime,
        })
    }

    pub fn openings(&self, positions: &[u64]) -> Result<BaseOpenings> {
        if positions.is_empty() {
            return Err(Error::InvalidIndex);
        }
        let mask_n = self.rows.mask_code.n();
        let mut old_main = Vec::with_capacity(positions.len());
        let mut old_mask = Vec::with_capacity(positions.len());
        let mut star = Vec::with_capacity(positions.len());
        let mut star_mask = Vec::with_capacity(positions.len());
        let mut fresh = Vec::with_capacity(positions.len());
        for position in positions {
            if *position >= self.rows.code.n() {
                return Err(Error::InvalidIndex);
            }
            let mask_position = *position % mask_n;
            old_main.push(self.old_main.open(*position)?);
            old_mask.push(self.old_mask.open(mask_position)?);
            star.push(self.star.open(*position)?);
            star_mask.push(self.star_mask_row.open(mask_position)?);
            fresh.push(self.fresh.open(mask_position)?);
        }
        Ok(BaseOpenings {
            old_main,
            old_mask,
            star,
            star_mask,
            fresh,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Commitments,
    Lambda,
    Rows,
    Positions,
    Openings,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OpenedSymbols {
    old_main: Vec<Fp3>,
    old_mask: Vec<Fp3>,
    star: Vec<Fp3>,
    star_mask: Vec<Fp3>,
    fresh: Vec<Fp3>,
}

/// The verifier-side fixture BASE continuation. It owns the chain moved from
/// the FOLD session and the fixture expectations; it never reads prover rows
/// except through the declared messages and opening responses.
pub struct BaseFinalSession {
    chain: Chain,
    handoff: FoldHandoff,
    fresh_claim: Vec<Fp3>,
    expectation: BaseExpectation,
    registry: OracleRegistry,
    ids: [OracleId; 5],
    stage: Stage,
    target_star: Option<Fp3>,
    lambda: Option<Fp3>,
    prepared: Option<BaseFinalRows>,
    positions: Vec<u64>,
    opened: Option<OpenedSymbols>,
    entry_rounds: u64,
    entry_value: Digest,
    commit_rounds: u64,
    commit_value: Digest,
    rows_rounds: u64,
    rows_value: Digest,
}

impl BaseFinalSession {
    pub fn new(
        chain: Chain,
        handoff: FoldHandoff,
        expectation: BaseExpectation,
    ) -> Result<Self> {
        if handoff.main_claim.len() as u64 != handoff.code.ell
            || handoff.mask_claim.len() as u64 != handoff.mask_code.ell
            || handoff.fresh_claim.len() as u64 != handoff.mask_code.ell
        {
            return Err(Error::InvalidConfig);
        }
        let expected = [
            (expectation.old_main, handoff.code.n()),
            (expectation.old_mask, handoff.mask_code.n()),
            (expectation.star, handoff.code.n()),
            (expectation.star_mask, handoff.mask_code.n()),
            (expectation.fresh, handoff.mask_code.n()),
        ];
        let mut registry = OracleRegistry::new();
        let mut ids = Vec::with_capacity(5);
        for (commitment, count) in expected {
            if commitment.count != count || commitment.root == Digest::ZERO {
                return Err(Error::StatementMismatch);
            }
            ids.push(registry.insert(commitment.root, commitment.start, commitment.count)?);
        }
        let mut ids_array = [ids[0]; 5];
        ids_array.copy_from_slice(&ids);
        let entry_rounds = chain.rounds();
        let entry_value = *chain.value();
        let fresh_claim = handoff.fresh_claim.clone();
        Ok(Self {
            chain,
            handoff,
            fresh_claim,
            expectation,
            registry,
            ids: ids_array,
            stage: Stage::Fresh,
            target_star: None,
            lambda: None,
            prepared: None,
            positions: Vec::new(),
            opened: None,
            entry_rounds,
            entry_value,
            commit_rounds: entry_rounds,
            commit_value: entry_value,
            rows_rounds: entry_rounds,
            rows_value: entry_value,
        })
    }

    pub fn receive_commitments(&mut self, messages: BaseMessages) -> Result<()> {
        if self.stage != Stage::Fresh {
            return Err(Error::InvalidSessionState);
        }
        if messages.fold_identity != self.expectation.fold_identity
            || messages.old_main != self.expectation.old_main
            || messages.old_mask != self.expectation.old_mask
            || messages.star != self.expectation.star
            || messages.star_mask != self.expectation.star_mask
            || messages.fresh != self.expectation.fresh
        {
            return Err(Error::StatementMismatch);
        }
        let mut words = Vec::new();
        push_digest(&mut words, &messages.fold_identity);
        push_commitment(&mut words, &messages.old_main);
        push_commitment(&mut words, &messages.old_mask);
        push_commitment(&mut words, &messages.star);
        push_commitment(&mut words, &messages.star_mask);
        push_commitment(&mut words, &messages.fresh);
        words.extend(fp3_words(&messages.target_star));
        absorb(&mut self.chain, 3, &words)?;
        self.target_star = Some(messages.target_star);
        self.commit_rounds = self.chain.rounds();
        self.commit_value = *self.chain.value();
        self.stage = Stage::Commitments;
        Ok(())
    }

    pub fn draw_lambda(&mut self) -> Result<Fp3> {
        if self.stage != Stage::Commitments {
            return Err(Error::InvalidSessionState);
        }
        let lambda = self.chain.draw_e()?;
        self.lambda = Some(lambda);
        self.stage = Stage::Lambda;
        Ok(lambda)
    }

    pub fn receive_base_rows(&mut self, prepared: BaseFinalRows) -> Result<()> {
        if self.stage != Stage::Lambda {
            return Err(Error::InvalidSessionState);
        }
        let lambda = self.lambda.ok_or(Error::InvalidSessionState)?;
        let target_star = self.target_star.ok_or(Error::InvalidSessionState)?;
        if prepared.base.lambda != lambda || prepared.base.target_star != target_star {
            return Err(Error::StatementMismatch);
        }
        if prepared.base.main_prime.len() as u64 != self.handoff.code.ell
            || prepared.base.main_tail_prime.len() as u64 != self.handoff.code.r
            || prepared.base.mask_prime.len() as u64 != self.handoff.mask_code.ell
            || prepared.base.mask_tail_prime.len() as u64 != self.handoff.mask_code.r
            || prepared.fresh_prime.len() as u64 != self.handoff.mask_code.ell
            || prepared.fresh_tail_prime.len() as u64 != self.handoff.mask_code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let mut m_words = Vec::new();
        for value in &prepared.base.main_prime {
            m_words.extend(fp3_words(value));
        }
        for value in &prepared.base.mask_prime {
            m_words.extend(fp3_words(value));
        }
        for value in &prepared.fresh_prime {
            m_words.extend(fp3_words(value));
        }
        absorb(&mut self.chain, 4, &m_words)?;
        let mut r_words = Vec::new();
        for value in &prepared.base.main_tail_prime {
            r_words.extend(fp3_words(value));
        }
        for value in &prepared.base.mask_tail_prime {
            r_words.extend(fp3_words(value));
        }
        for value in &prepared.fresh_tail_prime {
            r_words.extend(fp3_words(value));
        }
        absorb(&mut self.chain, 5, &r_words)?;
        self.rows_rounds = self.chain.rounds();
        self.rows_value = *self.chain.value();
        self.prepared = Some(prepared);
        self.stage = Stage::Rows;
        Ok(())
    }

    pub fn draw_base_positions(&mut self, count: usize) -> Result<Vec<u64>> {
        if self.stage != Stage::Rows {
            return Err(Error::InvalidSessionState);
        }
        if count == 0 {
            return Err(Error::InvalidConfig);
        }
        let positions = draw_positions(&mut self.chain, count, self.handoff.code.log_n)?;
        self.positions = positions.clone();
        self.stage = Stage::Positions;
        Ok(positions)
    }

    fn open_symbol(
        &self,
        slot: usize,
        expected_position: u64,
        expected_count: u64,
        opening: &BaseOpening,
    ) -> Result<Fp3> {
        if opening.index != expected_position {
            return Err(Error::InvalidIndex);
        }
        let oracle = self.ids[slot];
        if self.registry.get(oracle).is_none() {
            return Err(Error::StatementMismatch);
        }
        let opening = OracleOpening {
            oracle,
            index: opening.index,
            values: opening.values.clone(),
            path: opening.path.clone(),
        };
        verify_opening(&self.registry, &opening, expected_count)?;
        fp3_from_words(&opening.values)
    }

    pub fn verify_openings(&mut self, openings: &BaseOpenings) -> Result<()> {
        if self.stage != Stage::Positions {
            return Err(Error::InvalidSessionState);
        }
        let mask_n = self.handoff.mask_code.n();
        if openings.old_main.len() != self.positions.len()
            || openings.old_mask.len() != self.positions.len()
            || openings.star.len() != self.positions.len()
            || openings.star_mask.len() != self.positions.len()
            || openings.fresh.len() != self.positions.len()
        {
            return Err(Error::ShapeMismatch);
        }
        let mut opened = OpenedSymbols {
            old_main: Vec::with_capacity(self.positions.len()),
            old_mask: Vec::with_capacity(self.positions.len()),
            star: Vec::with_capacity(self.positions.len()),
            star_mask: Vec::with_capacity(self.positions.len()),
            fresh: Vec::with_capacity(self.positions.len()),
        };
        for (index, position) in self.positions.iter().enumerate() {
            let mask_position = *position % mask_n;
            opened.old_main.push(self.open_symbol(
                0,
                *position,
                self.handoff.code.n(),
                &openings.old_main[index],
            )?);
            opened.old_mask.push(self.open_symbol(
                1,
                mask_position,
                self.handoff.mask_code.n(),
                &openings.old_mask[index],
            )?);
            opened.star.push(self.open_symbol(
                2,
                *position,
                self.handoff.code.n(),
                &openings.star[index],
            )?);
            opened.star_mask.push(self.open_symbol(
                3,
                mask_position,
                self.handoff.mask_code.n(),
                &openings.star_mask[index],
            )?);
            opened.fresh.push(self.open_symbol(
                4,
                mask_position,
                self.handoff.mask_code.n(),
                &openings.fresh[index],
            )?);
        }
        self.opened = Some(opened);
        self.stage = Stage::Openings;
        Ok(())
    }

    pub fn finish(self) -> Result<BaseFinalCheck> {
        if self.stage != Stage::Openings {
            return Err(Error::InvalidSessionState);
        }
        let prepared = self.prepared.as_ref().ok_or(Error::InvalidSessionState)?;
        let opened = self.opened.as_ref().ok_or(Error::InvalidSessionState)?;
        let lambda = self.lambda.ok_or(Error::InvalidSessionState)?;
        let target_star = self.target_star.ok_or(Error::InvalidSessionState)?;
        // The two-row BASE linear check is extended by the FOLD fresh row. The
        // fresh row has a zero star row, so its combined message is lambda
        // times the fresh message. This is the same FOLD target identity split
        // into old and fresh components; it is not a rewritten target.
        let main = inner(&self.handoff.main_claim, &prepared.base.main_prime)?;
        let mask = inner(&self.handoff.mask_claim, &prepared.base.mask_prime)?;
        let fresh = inner(&self.fresh_claim, &prepared.fresh_prime)?;
        let lhs = main.add(&mask).add(&fresh);
        let rhs = prepared.base.target_star.add(&lambda.mul(&self.handoff.target));
        let base_linear = lhs == rhs;
        let encoded_main = encode_row_fp3(
            &self.handoff.code,
            &prepared.base.main_prime,
            &prepared.base.main_tail_prime,
        )?;
        let encoded_mask = encode_row_fp3(
            &self.handoff.mask_code,
            &prepared.base.mask_prime,
            &prepared.base.mask_tail_prime,
        )?;
        let encoded_fresh = encode_row_fp3(
            &self.handoff.mask_code,
            &prepared.fresh_prime,
            &prepared.fresh_tail_prime,
        )?;
        let mask_n = self.handoff.mask_code.n();
        let mut symbol_checks = true;
        for (index, position) in self.positions.iter().enumerate() {
            let mask_position = *position % mask_n;
            if !base_stage::symbol_holds(
                *encoded_main
                    .get(*position as usize)
                    .ok_or(Error::InvalidIndex)?,
                opened.star[index],
                lambda,
                opened.old_main[index],
            ) || !base_stage::symbol_holds(
                *encoded_mask
                    .get(mask_position as usize)
                    .ok_or(Error::InvalidIndex)?,
                opened.star_mask[index],
                lambda,
                opened.old_mask[index],
            ) || !base_stage::symbol_holds(
                *encoded_fresh
                    .get(mask_position as usize)
                    .ok_or(Error::InvalidIndex)?,
                Fp3::ZERO,
                lambda,
                opened.fresh[index],
            ) {
                symbol_checks = false;
            }
        }
        let rejection = if !base_linear {
            Some(BaseRejection::TargetLinear)
        } else if !symbol_checks {
            Some(BaseRejection::Symbol)
        } else {
            None
        };
        let fold_identity = self.expectation.fold_identity;
        let old_main_root = self.expectation.old_main.root;
        let old_mask_root = self.expectation.old_mask.root;
        let star_root = self.expectation.star.root;
        let star_mask_root = self.expectation.star_mask.root;
        let fresh_root = self.expectation.fresh.root;
        if let Some(layer) = rejection {
            return Ok(BaseFinalCheck::Rejected(BaseFinalRejection {
                layer,
                fold_identity,
                old_main_root,
                old_mask_root,
                star_root,
                star_mask_root,
                fresh_root,
                lambda,
                target_star,
                base_positions: self.positions,
                base_linear,
                symbol_checks,
            }));
        }
        Ok(BaseFinalCheck::Passed(BaseFinalOutcome {
            fold_identity,
            old_main_root,
            old_mask_root,
            star_root,
            star_mask_root,
            fresh_root,
            lambda,
            target_star,
            base_positions: self.positions,
            base_linear,
            symbol_checks,
            openings_verified: true,
            rejection: None,
            pending: true,
            complete_base: false,
            fresh_row_committed: true,
            fresh_term_checked: true,
            fresh_term_authenticated_by_fold_root: false,
            projection_only: true,
            chain_handoff: true,
            entry_rounds: self.entry_rounds,
            entry_value: self.entry_value,
            commit_rounds: self.commit_rounds,
            commit_value: self.commit_value,
            rows_rounds: self.rows_rounds,
            rows_value: self.rows_value,
        }))
    }

    pub fn chain_value(&self) -> &Digest {
        self.chain.value()
    }

    pub fn chain_rounds(&self) -> u64 {
        self.chain.rounds()
    }

    pub fn expectation(&self) -> &BaseExpectation {
        &self.expectation
    }

    pub fn handoff(&self) -> &FoldHandoff {
        &self.handoff
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

#[cfg(test)]
fn fixture_rows(code: CodeParams, mask_code: CodeParams, target: Fp3) -> SingleRows {
    SingleRows {
        code,
        mask_code,
        main: vec![
            Fp3::from_u64(11),
            Fp3::from_u64(12),
            Fp3::from_u64(13),
            Fp3::from_u64(14),
        ],
        main_tail: vec![Fp3::from_u64(15)],
        masks: vec![vec![
            Fp3::from_u64(21),
            Fp3::from_u64(22),
            Fp3::from_u64(23),
            Fp3::from_u64(24),
        ]],
        mask_tails: vec![vec![Fp3::from_u64(25)]],
        claim: vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::ZERO, Fp3::ZERO],
        mask_claims: vec![vec![Fp3::from_u64(3), Fp3::ZERO, Fp3::ZERO, Fp3::ZERO]],
        target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::chain::Chain;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

    fn code() -> CodeParams {
        build_code_plan(3, 4, 1).unwrap()
    }

    fn chain(seed: &[u8]) -> Chain {
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest::ZERO, seed, &[], None).unwrap()
    }

    fn handoff(rows: &SingleRows) -> FoldHandoff {
        FoldHandoff {
            code: rows.code,
            mask_code: rows.mask_code,
            main_claim: rows.claim.clone(),
            mask_claim: rows.mask_claims[0].clone(),
            target: rows.target,
            fresh_claim: vec![Fp3::ZERO; rows.mask_code.ell as usize],
            group: 0,
            k: 2,
            h: 1,
            members: vec![0, 1],
            gamma: vec![Fp3::from_u64(2), Fp3::from_u64(3)],
            eps: Fp3::from_u64(5),
            sum: Fp3::from_u64(7),
            fresh: 3,
            fresh_root: Digest([0x31; 32]),
            tree_number: 5,
            kept_factor: Fp3::ONE,
            kept_rows: 0,
            layout: zkie_engine::pcs::request::fold::FoldGroupLayout {
                group: 0,
                k: 2,
                h: 1,
                members: vec![
                    zkie_engine::pcs::request::fold::FoldMemberLayout {
                        oracle: 0,
                        main_rows: 1,
                        mask_rows: 1,
                        kept: false,
                    },
                    zkie_engine::pcs::request::fold::FoldMemberLayout {
                        oracle: 1,
                        main_rows: 1,
                        mask_rows: 1,
                        kept: false,
                    },
                ],
            },
        }
    }

    fn star() -> (Vec<Fp3>, Vec<Fp3>, Vec<Fp3>, Vec<Fp3>) {
        (
            vec![Fp3::from_u64(31); 4],
            vec![Fp3::from_u64(32)],
            vec![Fp3::from_u64(33); 4],
            vec![Fp3::from_u64(34)],
        )
    }

    fn setup(
        seed: &[u8],
        target_shift: Fp3,
    ) -> (BaseProver, BaseFinalSession, BaseMessages) {
        let mut rows = fixture_rows(code(), code(), Fp3::from_u64(8));
        let fresh_claim = vec![
            Fp3::from_u64(41),
            Fp3::from_u64(42),
            Fp3::ZERO,
            Fp3::ZERO,
        ];
        let fresh_row = vec![
            Fp3::from_u64(51),
            Fp3::from_u64(52),
            Fp3::from_u64(53),
            Fp3::from_u64(54),
        ];
        let fresh_tail = vec![Fp3::from_u64(55)];
        // The FOLD target is the sum of the old main, old mask and fresh row
        // dot products; a zero target shift then gives a passing local check.
        rows.target = inner(&rows.claim, &rows.main)
            .unwrap()
            .add(&inner(&rows.mask_claims[0], &rows.masks[0]).unwrap())
            .add(&inner(&fresh_claim, &fresh_row).unwrap());
        let mut handoff = handoff(&rows);
        handoff.fresh_claim = fresh_claim.clone();
        let identity = fold_identity(&handoff);
        let (star_main, star_tail, star_mask, star_mask_tail) = star();
        let prover = BaseProver::new(
            rows,
            identity,
            fresh_claim,
            fresh_row,
            fresh_tail,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
            target_shift,
        )
        .unwrap();
        let session =
            BaseFinalSession::new(chain(seed), handoff, prover.expectation()).unwrap();
        let messages = prover.messages().unwrap();
        (prover, session, messages)
    }

    fn finish(prover: &BaseProver, mut session: BaseFinalSession) -> BaseFinalCheck {
        let messages = prover.messages().unwrap();
        session.receive_commitments(messages).unwrap();
        let lambda = session.draw_lambda().unwrap();
        let prepared = prover.base_rows(lambda).unwrap();
        session.receive_base_rows(prepared).unwrap();
        let positions = session.draw_base_positions(2).unwrap();
        let openings = prover.openings(&positions).unwrap();
        session.verify_openings(&openings).unwrap();
        session.finish().unwrap()
    }

    #[test]
    fn the_local_base_wiring_passes_but_stays_pending() {
        let (prover, session, _messages) = setup(b"base final", Fp3::ZERO);
        let outcome = match finish(&prover, session) {
            BaseFinalCheck::Passed(outcome) => outcome,
            BaseFinalCheck::Rejected(rejection) => {
                panic!("the local check rejected a consistent fixture: {:?}", rejection)
            }
        };
        assert!(outcome.base_linear);
        assert!(outcome.symbol_checks);
        assert!(outcome.openings_verified);
        assert!(outcome.rejection.is_none());
        assert!(outcome.pending);
        assert!(!outcome.complete_base);
        assert!(outcome.fresh_row_committed);
        assert!(outcome.fresh_term_checked);
        assert!(!outcome.fresh_term_authenticated_by_fold_root);
        assert!(outcome.projection_only);
        assert!(outcome.chain_handoff);
        assert!(!outcome.base_positions.is_empty());
        assert!(outcome.commit_rounds > outcome.entry_rounds);
        assert!(outcome.rows_rounds > outcome.commit_rounds);
        let _ = _messages;
    }

    #[test]
    fn a_fixed_target_shift_is_rejected_at_the_target_layer() {
        let (prover, session, _messages) = setup(b"base target", Fp3::ONE);
        let rejection = match finish(&prover, session) {
            BaseFinalCheck::Rejected(rejection) => rejection,
            BaseFinalCheck::Passed(_) => panic!("the fixed target shift was not rejected"),
        };
        assert_eq!(rejection.layer, BaseRejection::TargetLinear);
        assert!(!rejection.base_linear);
        assert!(!rejection.base_positions.is_empty());
        assert!(!rejection.lambda.is_zero());
    }

    #[test]
    fn a_changed_encoding_tail_fails_the_symbol_layer() {
        let (prover, mut session, messages) = setup(b"base symbol", Fp3::ZERO);
        session.receive_commitments(messages).unwrap();
        let lambda = session.draw_lambda().unwrap();
        let mut prepared = prover.base_rows(lambda).unwrap();
        prepared.base.main_tail_prime[0] =
            prepared.base.main_tail_prime[0].add(&Fp3::ONE);
        session.receive_base_rows(prepared).unwrap();
        let positions = session.draw_base_positions(2).unwrap();
        let openings = prover.openings(&positions).unwrap();
        session.verify_openings(&openings).unwrap();
        let rejection = match session.finish().unwrap() {
            BaseFinalCheck::Rejected(rejection) => rejection,
            BaseFinalCheck::Passed(_) => panic!("the changed tail was not rejected"),
        };
        // The linear check does not use the tails, so it still holds; the
        // encoded-symbol check is the layer that rejects the changed tail.
        assert!(rejection.base_linear);
        assert!(!rejection.symbol_checks);
        assert_eq!(rejection.layer, BaseRejection::Symbol);
    }

    #[test]
    fn changed_openings_and_swapped_bindings_are_rejected() {
        let (prover, mut session, messages) = setup(b"base openings", Fp3::ZERO);
        session.receive_commitments(messages).unwrap();
        let lambda = session.draw_lambda().unwrap();
        session
            .receive_base_rows(prover.base_rows(lambda).unwrap())
            .unwrap();
        let positions = session.draw_base_positions(2).unwrap();
        let mut openings = prover.openings(&positions).unwrap();
        openings.old_main[0].values[0] ^= 1;
        assert!(session.verify_openings(&openings).is_err());

        let (prover, mut session, messages) = setup(b"base identity", Fp3::ZERO);
        let mut wrong = messages;
        wrong.fold_identity.0[0] ^= 1;
        assert!(matches!(
            session.receive_commitments(wrong),
            Err(Error::StatementMismatch)
        ));
        let mut wrong = messages;
        wrong.star.root.0[0] ^= 1;
        assert!(matches!(
            session.receive_commitments(wrong),
            Err(Error::StatementMismatch)
        ));
        let (prover, session, messages) = setup(b"base handoff", Fp3::ZERO);
        let mut wrong_handoff = session.handoff().clone();
        wrong_handoff.target = wrong_handoff.target.add(&Fp3::ONE);
        let mut wrong_expectation = prover.expectation();
        wrong_expectation.fold_identity = fold_identity(&wrong_handoff);
        let mut wrong_session =
            BaseFinalSession::new(chain(b"base handoff"), wrong_handoff, wrong_expectation)
                .unwrap();
        assert!(matches!(
            wrong_session.receive_commitments(messages),
            Err(Error::StatementMismatch)
        ));
    }

    #[test]
    fn missing_preconditions_fail_closed() {
        let (prover, mut session, messages) = setup(b"base order", Fp3::ZERO);
        assert!(session.draw_lambda().is_err());
        assert!(session.draw_base_positions(1).is_err());
        assert!(session
            .receive_base_rows(prover.base_rows(Fp3::ONE).unwrap())
            .is_err());
        assert!(session.verify_openings(&BaseOpenings {
            old_main: Vec::new(),
            old_mask: Vec::new(),
            star: Vec::new(),
            star_mask: Vec::new(),
            fresh: Vec::new(),
        }).is_err());
        session.receive_commitments(messages).unwrap();
        assert!(session.receive_commitments(messages).is_err());
        let lambda = session.draw_lambda().unwrap();
        assert!(session.draw_base_positions(1).is_err());
        session
            .receive_base_rows(prover.base_rows(lambda).unwrap())
            .unwrap();
        assert!(session.draw_base_positions(0).is_err());
        let positions = session.draw_base_positions(1).unwrap();
        assert!(session.draw_base_positions(1).is_err());
        let openings = prover.openings(&positions).unwrap();
        session.verify_openings(&openings).unwrap();
        assert!(matches!(
            session.finish(),
            Ok(BaseFinalCheck::Passed(_))
        ));
    }
}
