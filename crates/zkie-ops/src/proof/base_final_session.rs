//! Isolated fixture BASE final-state check over the FOLD output rows.
//!
//! The verifier's claim, target and FOLD identity come from the same verified
//! FOLD session that produced the output rows; the chain is moved from the FOLD
//! session into this session, so the commitment/target message, lambda, M'/R'
//! message and position draws all use one continuous transcript. The challenge
//! order stays fixed: commitments and target* before lambda, M'/R' and the
//! fresh rows before the position draw, then the openings and the local checks.
//! The added original-column openings are checked locally after the position
//! draw; they are not inserted into the challenge window and add no transcript
//! round.
//!
//! This is a conditional local experiment, not the reference standard IOPP
//! path. The code-switch stage is not connected, so the reference path
//! FOLD -> CS -> final SingleRows -> BASE is not completed and this module does
//! not claim that skipping CS is equivalent to that path. The result is not a
//! complete BASE, not a complete proximity test, not a production pass, and it
//! cannot be used to claim either. A zero-fold degenerate input is not used to
//! prove general conclusions.
//!
//! Folded-column relation. The FOLD output rows are authenticated at the drawn
//! BASE positions against the original member commitments that were bound
//! before the BAT challenge and against the FOLD fresh mask commitment. The
//! verifier opens every original member column at the drawn positions, derives
//! each member row's group-row placement from the bound fold layout, folds the
//! opened columns with the transcript's fold challenges, and compares the
//! result with the received rows:
//!
//! - the combined main row equals the star row plus lambda times the folded
//!   original main columns;
//! - the combined mask row equals the star mask row plus lambda times the
//!   folded original mask columns;
//! - the fresh row equals the folded fresh mask columns.
//!
//! The first two are the BASE symbol checks with the old row opening replaced
//! by the folded original columns; the third binds the fresh term to the fresh
//! root. The verifier does not use a rebuilt same-prover root, a same-root
//! assertion or only a fresh opening as a substitute for this relation. The
//! relation is checked only at the drawn positions, so it is a sampled local
//! check, not a complete binding of every column and not a full proximity
//! result. The original member roots and tree plans are fixture inputs that
//! were bound through the group session; they are not SEND receipts, asset
//! roots or a production trust anchor. The fresh mask tree plan is a fixture
//! input as well.
//!
//! The star rows are the BASE prover's own fixture rows. Their commitments and
//! target* are sent before lambda and bound in the transcript; their openings
//! are checked against the declared roots. No independent trusted star source
//! exists here.
//!
//! A single equality at one lambda and two sampled positions does not show that
//! every wrong declaration is rejected and does not give an IOPP security
//! bound; wrong declarations can cancel through aggregation. The fixed
//! inconsistent samples in the tests are crafted to fail at a named layer; they
//! are not a general rejection guarantee. A passed local check remains pending.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::base_stage::{self, BaseRows};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::code::{encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3};
use zkie_engine::pcs::request::fold::{
    fold_mask_weights, fold_member_firsts, validate_fold_mask_shape, FoldGroupLayout, FoldPart,
    FoldShape,
};
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_engine::pcs::request::read::{draw_positions, unique_sorted};
use zkie_engine::pcs::request::runtime_tree::{
    verify_mask_opening, verify_runtime_opening, MaskTreePlan, RuntimeTreeOpening, RuntimeTreePlan,
};
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

/// One original member commitment: its oracle, codes, tree plan and bound
/// root. The root is the value the group session received and bound before the
/// BAT challenge; the tree plan is the fixture's public layout for that member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldMember {
    pub oracle: u64,
    pub main_code: CodeParams,
    pub mask_code: CodeParams,
    pub root: Digest,
    pub tree: RuntimeTreePlan,
}

/// The verifier-side inputs used to relate the FOLD output rows to the
/// original commitments. The member roots and tree plans are fixture inputs
/// bound through the group session; the fresh tree plan is a fixture input.
/// This is not an independent trust anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldBasis {
    pub layout: FoldGroupLayout,
    pub fresh_root: Digest,
    pub fresh_tree: MaskTreePlan,
    pub members: Vec<FoldMember>,
}

/// The prover's BASE commitment message: the FOLD identity, the two star row
/// commitments and target*. It is fixed before lambda.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseMessages {
    pub fold_identity: Digest,
    pub star: BaseCommitment,
    pub star_mask: BaseCommitment,
    pub target_star: Fp3,
}

/// One opening response for a BASE star row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseOpening {
    pub index: u64,
    pub values: Vec<u64>,
    pub path: Vec<Digest>,
}

/// The star row openings in drawn-position order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseStarOpenings {
    pub star: Vec<BaseOpening>,
    pub star_mask: Vec<BaseOpening>,
}

/// The complete opening response set consumed by the session: the star rows,
/// one original-member opening per bound member, and the FOLD fresh mask tree
/// opening. The original and fresh openings are at the unique sorted drawn
/// positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseOpenings {
    pub star: Vec<BaseOpening>,
    pub star_mask: Vec<BaseOpening>,
    pub members: Vec<RuntimeTreeOpening>,
    pub fresh: RuntimeTreeOpening,
}

/// The BASE message rows. M' and R' include the star row and lambda; the fresh
/// row and tail are the FOLD fresh row sent before the fresh comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseFinalRows {
    pub base: BaseRows,
    pub fresh_row: Vec<Fp3>,
    pub fresh_tail: Vec<Fp3>,
}

/// The layer at which a well-formed but inconsistent local sample is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseRejection {
    TargetLinear,
    FoldMain,
    FoldMask,
    FoldFresh,
}

/// The result of the local fixture BASE wiring. It is pending, not a complete
/// BASE and not an accepted proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseFinalOutcome {
    pub fold_identity: Digest,
    pub star_root: Digest,
    pub star_mask_root: Digest,
    pub member_roots: Vec<Digest>,
    pub fresh_root: Digest,
    pub lambda: Fp3,
    pub target_star: Fp3,
    pub base_positions: Vec<u64>,
    pub base_linear: bool,
    pub fold_main: bool,
    pub fold_mask: bool,
    pub fold_fresh: bool,
    pub openings_verified: bool,
    pub member_openings_verified: bool,
    pub fresh_opening_verified: bool,
    pub rejection: Option<BaseRejection>,
    pub pending: bool,
    pub complete_base: bool,
    pub sampled_fold_relation_only: bool,
    pub complete_fold_binding: bool,
    pub fresh_term_checked_against_fold_root: bool,
    pub projection_only: bool,
    pub conditional_local_experiment: bool,
    pub cs_connected: bool,
    pub reference_iopp_path: bool,
    pub chain_handoff: bool,
    pub check_rounds: u64,
    pub check_value: Digest,
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
    pub star_root: Digest,
    pub star_mask_root: Digest,
    pub member_roots: Vec<Digest>,
    pub fresh_root: Digest,
    pub lambda: Fp3,
    pub target_star: Fp3,
    pub base_positions: Vec<u64>,
    pub base_linear: bool,
    pub fold_main: bool,
    pub fold_mask: bool,
    pub fold_fresh: bool,
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

fn fp3_from_symbols(symbols: &[Fp], offset: usize) -> Result<Fp3> {
    let values = symbols
        .get(offset..offset.checked_add(3).ok_or(Error::ResourceLimit)?)
        .ok_or(Error::ShapeMismatch)?;
    Ok(Fp3::new(values[0], values[1], values[2]))
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

/// The fold weights of the row variables: the coefficient of original group
/// row `g` after one low-fold per variable. The order is the layout's own
/// aligned-block order: variable 0 pairs adjacent group rows.
pub(crate) fn fold_coefficients(variables: u32, gamma: &[Fp3]) -> Result<Vec<Fp3>> {
    if variables >= 64 || gamma.len() < variables as usize {
        return Err(Error::InvalidConfig);
    }
    let rows = 1usize
        .checked_shl(variables)
        .ok_or(Error::ResourceLimit)?;
    let mut out = Vec::with_capacity(rows);
    for group in 0..rows {
        let mut coefficient = Fp3::ONE;
        for variable in 0..variables as usize {
            let weight = if (group >> variable) & 1 == 1 {
                gamma[variable]
            } else {
                Fp3::ONE.sub(&gamma[variable])
            };
            coefficient = coefficient.mul(&weight);
        }
        out.push(coefficient);
    }
    Ok(out)
}

pub(crate) fn fold_original_columns(
    handoff: &FoldHandoff,
    basis: &FoldBasis,
        openings: &[RuntimeTreeOpening],
        unique: &[u64],
    ) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
        let main_coeff = fold_coefficients(handoff.k, &handoff.gamma)?;
        let mask_coeff = fold_coefficients(handoff.h, &handoff.gamma[..handoff.h as usize])?;
        let main_firsts = fold_member_firsts(&basis.layout, FoldPart::Main)?;
        let mask_firsts = fold_member_firsts(&basis.layout, FoldPart::Mask)?;
        let mut fold_main = vec![Fp3::ZERO; unique.len()];
        let mut fold_mask = vec![Fp3::ZERO; unique.len()];
        if openings.len() != basis.members.len() {
            return Err(Error::ShapeMismatch);
        }
        for (index, (member, opening)) in basis.members.iter().zip(openings).enumerate() {
            let layout_member = &basis.layout.members[index];
            if opening.positions != unique {
                return Err(Error::InvalidIndex);
            }
            verify_runtime_opening(&member.tree, &member.root, unique, opening)?;
            let part = &member.tree.parts[0];
            let main_symbols =
                usize::try_from(part.main_symbols).map_err(|_| Error::ResourceLimit)?;
            let expected_main = usize::try_from(
                layout_member
                    .main_rows
                    .checked_mul(3)
                    .ok_or(Error::ResourceLimit)?,
            )
            .map_err(|_| Error::ResourceLimit)?;
            let expected_mask = usize::try_from(
                layout_member
                    .mask_rows
                    .checked_mul(3)
                    .ok_or(Error::ResourceLimit)?,
            )
            .map_err(|_| Error::ResourceLimit)?;
            if main_symbols != expected_main {
                return Err(Error::ShapeMismatch);
            }
            let main_first = main_firsts[index];
            let mask_first = mask_firsts[index];
            for (column_index, column) in opening.columns.iter().enumerate() {
                if column.symbols.len() != main_symbols + expected_mask {
                    return Err(Error::ShapeMismatch);
                }
                for row in 0..usize::try_from(layout_member.main_rows)
                    .map_err(|_| Error::ResourceLimit)?
                {
                    let value = fp3_from_symbols(&column.symbols, row * 3)?;
                    let group = main_first
                        .ok_or(Error::StatementMismatch)?
                        .checked_add(row as u64)
                        .ok_or(Error::ResourceLimit)?;
                    let coefficient = *main_coeff
                        .get(group as usize)
                        .ok_or(Error::InvalidIndex)?;
                    fold_main[column_index] =
                        fold_main[column_index].add(&coefficient.mul(&value));
                }
                for row in 0..usize::try_from(layout_member.mask_rows)
                    .map_err(|_| Error::ResourceLimit)?
                {
                    let value = fp3_from_symbols(&column.symbols, main_symbols + row * 3)?;
                    let group = mask_first
                        .ok_or(Error::StatementMismatch)?
                        .checked_add(row as u64)
                        .ok_or(Error::ResourceLimit)?;
                    let coefficient = *mask_coeff
                        .get(group as usize)
                        .ok_or(Error::InvalidIndex)?;
                    fold_mask[column_index] =
                        fold_mask[column_index].add(&coefficient.mul(&value));
                }
            }
        }
        Ok((fold_main, fold_mask))
    }


pub(crate) fn fold_fresh_columns(
    handoff: &FoldHandoff,
    basis: &FoldBasis,
        opening: &RuntimeTreeOpening,
        unique: &[u64],
    ) -> Result<Vec<Fp3>> {
        if opening.positions != unique {
            return Err(Error::InvalidIndex);
        }
        verify_mask_opening(
            &basis.fresh_tree,
            &basis.fresh_root,
            unique,
            opening,
        )?;
        let mask_coeff = fold_coefficients(handoff.h, &handoff.gamma[..handoff.h as usize])?;
        let width = usize::try_from(basis.fresh_tree.width()?).map_err(|_| Error::ResourceLimit)?;
        let rows = usize::try_from(basis.fresh_tree.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let mut fold = vec![Fp3::ZERO; unique.len()];
        for (column_index, column) in opening.columns.iter().enumerate() {
            if column.symbols.len() != width || width != rows * 3 {
                return Err(Error::ShapeMismatch);
            }
            for row in 0..rows {
                let value = fp3_from_symbols(&column.symbols, row * 3)?;
                let coefficient = *mask_coeff.get(row).ok_or(Error::InvalidIndex)?;
                fold[column_index] = fold[column_index].add(&coefficient.mul(&value));
            }
        }
        Ok(fold)
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

/// The fixture BASE prover. It holds the plaintext folded rows, the star rows
/// and their openings; only the messages and opening responses cross to the
/// verifier. The old rows are the FOLD output rows, not a new commitment.
pub struct BaseProver {
    rows: SingleRows,
    fresh_row: Vec<Fp3>,
    fresh_tail: Vec<Fp3>,
    star_main: Vec<Fp3>,
    star_tail: Vec<Fp3>,
    star_mask: Vec<Fp3>,
    star_mask_tail: Vec<Fp3>,
    star: ProverCommitment,
    star_mask_row: ProverCommitment,
    identity: Digest,
    target_shift: Fp3,
}

impl BaseProver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: SingleRows,
        fold_identity: Digest,
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
            || fresh_row.len() as u64 != rows.mask_code.ell
            || fresh_tail.len() as u64 != rows.mask_code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let star = commit_row(Digest([0x76; 32]), rows.code, &star_main, &star_tail)?;
        let star_mask_row = commit_row(
            Digest([0x77; 32]),
            rows.mask_code,
            &star_mask,
            &star_mask_tail,
        )?;
        Ok(Self {
            rows,
            fresh_row,
            fresh_tail,
            star_main,
            star_tail,
            star_mask,
            star_mask_tail,
            star,
            star_mask_row,
            identity: fold_identity,
            target_shift,
        })
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
            fold_identity: self.identity,
            star: self.star.public()?,
            star_mask: self.star_mask_row.public()?,
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
        Ok(BaseFinalRows {
            base: prepared,
            fresh_row: self.fresh_row.clone(),
            fresh_tail: self.fresh_tail.clone(),
        })
    }

    pub fn openings(&self, positions: &[u64]) -> Result<BaseStarOpenings> {
        if positions.is_empty() {
            return Err(Error::InvalidIndex);
        }
        let mask_n = self.rows.mask_code.n();
        let mut star = Vec::with_capacity(positions.len());
        let mut star_mask = Vec::with_capacity(positions.len());
        for position in positions {
            if *position >= self.rows.code.n() {
                return Err(Error::InvalidIndex);
            }
            let mask_position = *position % mask_n;
            star.push(self.star.open(*position)?);
            star_mask.push(self.star_mask_row.open(mask_position)?);
        }
        Ok(BaseStarOpenings { star, star_mask })
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
    star_main: Vec<Fp3>,
    star_mask: Vec<Fp3>,
    fold_main: Vec<Fp3>,
    fold_mask: Vec<Fp3>,
    fold_fresh: Vec<Fp3>,
}

/// The verifier-side fixture BASE continuation. It owns the chain moved from
/// the FOLD session, the FOLD output view and the original commitment basis;
/// it never reads prover rows except through the declared messages, the
/// received rows and the opening responses.
pub struct BaseFinalSession {
    chain: Chain,
    handoff: FoldHandoff,
    basis: FoldBasis,
    expected_identity: Digest,
    registry: OracleRegistry,
    star_id: Option<OracleId>,
    star_mask_id: Option<OracleId>,
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
    pub fn new(chain: Chain, handoff: FoldHandoff, basis: FoldBasis) -> Result<Self> {
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
        {
            return Err(Error::InvalidConfig);
        }
        // No kept rows are supported by this local BASE subset.
        if handoff.kept_rows != 0
            || basis.layout != handoff.layout
            || basis.layout.fault(&shape).is_some()
            || basis.layout.members.iter().any(|member| member.kept)
            || basis.layout.kept_rows_checked()? != 0
        {
            return Err(Error::InvalidConfig);
        }
        // The fresh coefficients are fixed by the transcript's fold
        // challenges, not by a prover-supplied vector.
        let expected_fresh = fold_mask_weights(
            handoff.k,
            handoff.h,
            usize::try_from(handoff.mask_code.ell).map_err(|_| Error::ResourceLimit)?,
            &handoff.gamma,
        )?;
        if handoff.fresh_claim != expected_fresh {
            return Err(Error::StatementMismatch);
        }
        // The fresh commitment is the FOLD fresh mask tree; its plan must match
        // the transcript identity and the shape.
        let fresh_rows = 1u64
            .checked_shl(handoff.h)
            .ok_or(Error::ResourceLimit)?;
        if basis.fresh_tree.fault().is_some()
            || basis.fresh_tree.tree_number != handoff.tree_number
            || basis.fresh_tree.log_n != handoff.mask_code.log_n
            || basis.fresh_tree.mask_rows != fresh_rows
            || basis.fresh_tree.leaf_class != shape.mask_leaf_class
            || basis.fresh_root != handoff.fresh_root
            || basis.fresh_root == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        if basis.members.len() != basis.layout.members.len() {
            return Err(Error::ShapeMismatch);
        }
        for (member, layout_member) in basis.members.iter().zip(&basis.layout.members) {
            if member.oracle != layout_member.oracle
                || member.main_code != handoff.code
                || member.mask_code != handoff.mask_code
                || member.root == Digest::ZERO
                || member.tree.fault().is_some()
                || member.tree.log_n != handoff.code.log_n
                || member.tree.log_n_zk != handoff.mask_code.log_n
                || member.tree.parts.len() != 1
            {
                return Err(Error::InvalidConfig);
            }
            let main_symbols = layout_member
                .main_rows
                .checked_mul(3)
                .ok_or(Error::ResourceLimit)?;
            let mask_symbols = layout_member
                .mask_rows
                .checked_mul(3)
                .ok_or(Error::ResourceLimit)?;
            if member.tree.parts[0].main_symbols != main_symbols
                || member.tree.parts[0].mask_symbols != mask_symbols
            {
                return Err(Error::InvalidConfig);
            }
        }
        let expected_identity = fold_identity(&handoff);
        let entry_rounds = chain.rounds();
        let entry_value = *chain.value();
        Ok(Self {
            chain,
            handoff,
            basis,
            expected_identity,
            registry: OracleRegistry::new(),
            star_id: None,
            star_mask_id: None,
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
        if messages.fold_identity != self.expected_identity
            || messages.star.count != self.handoff.code.n()
            || messages.star_mask.count != self.handoff.mask_code.n()
            || messages.star.root == Digest::ZERO
            || messages.star_mask.root == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        self.star_id = Some(
            self.registry
                .insert(messages.star.root, messages.star.start, messages.star.count)?,
        );
        self.star_mask_id = Some(
            self.registry
                .insert(messages.star_mask.root, messages.star_mask.start, messages.star_mask.count)?,
        );
        let mut words = Vec::new();
        push_digest(&mut words, &messages.fold_identity);
        push_commitment(&mut words, &messages.star);
        push_commitment(&mut words, &messages.star_mask);
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
            || prepared.fresh_row.len() as u64 != self.handoff.mask_code.ell
            || prepared.fresh_tail.len() as u64 != self.handoff.mask_code.r
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
        for value in &prepared.fresh_row {
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
        for value in &prepared.fresh_tail {
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

    fn open_star(
        &self,
        slot: bool,
        expected_position: u64,
        expected_count: u64,
        opening: &BaseOpening,
    ) -> Result<Fp3> {
        if opening.index != expected_position {
            return Err(Error::InvalidIndex);
        }
        let oracle = if slot { self.star_mask_id } else { self.star_id }
            .ok_or(Error::InvalidSessionState)?;
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

    fn fold_original_columns(
        &self,
        openings: &[RuntimeTreeOpening],
        unique: &[u64],
    ) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
        fold_original_columns(&self.handoff, &self.basis, openings, unique)
    }

    fn fold_fresh_columns(
        &self,
        opening: &RuntimeTreeOpening,
        unique: &[u64],
    ) -> Result<Vec<Fp3>> {
        fold_fresh_columns(&self.handoff, &self.basis, opening, unique)
    }

    pub fn verify_openings(&mut self, openings: &BaseOpenings) -> Result<()> {
        if self.stage != Stage::Positions {
            return Err(Error::InvalidSessionState);
        }
        let mask_n = self.handoff.mask_code.n();
        if openings.star.len() != self.positions.len()
            || openings.star_mask.len() != self.positions.len()
        {
            return Err(Error::ShapeMismatch);
        }
        let mut star_main = Vec::with_capacity(self.positions.len());
        let mut star_mask = Vec::with_capacity(self.positions.len());
        for (index, position) in self.positions.iter().enumerate() {
            let mask_position = *position % mask_n;
            star_main.push(self.open_star(
                false,
                *position,
                self.handoff.code.n(),
                &openings.star[index],
            )?);
            star_mask.push(self.open_star(
                true,
                mask_position,
                self.handoff.mask_code.n(),
                &openings.star_mask[index],
            )?);
        }
        let unique = unique_sorted(&self.positions);
        let (fold_main_unique, fold_mask_unique) =
            self.fold_original_columns(&openings.members, &unique)?;
        let mask_positions: Vec<u64> = self
            .positions
            .iter()
            .map(|position| *position % mask_n)
            .collect();
        let mask_unique = unique_sorted(&mask_positions);
        let fold_fresh_unique = self.fold_fresh_columns(&openings.fresh, &mask_unique)?;
        let mut fold_main = Vec::with_capacity(self.positions.len());
        let mut fold_mask = Vec::with_capacity(self.positions.len());
        let mut fold_fresh = Vec::with_capacity(self.positions.len());
        for position in &self.positions {
            let at = unique
                .binary_search(position)
                .map_err(|_| Error::InvalidIndex)?;
            let mask_position = *position % mask_n;
            let mask_at = mask_unique
                .binary_search(&mask_position)
                .map_err(|_| Error::InvalidIndex)?;
            fold_main.push(fold_main_unique[at]);
            fold_mask.push(fold_mask_unique[at]);
            fold_fresh.push(fold_fresh_unique[mask_at]);
        }
        self.opened = Some(OpenedSymbols {
            star_main,
            star_mask,
            fold_main,
            fold_mask,
            fold_fresh,
        });
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
        let check_rounds = self.chain.rounds();
        let check_value = *self.chain.value();
        // The two-row BASE linear check is extended by the FOLD fresh row. The
        // fresh row's combined message is lambda times the fresh message. This
        // is the same FOLD target identity split into old and fresh components;
        // it is not a rewritten target.
        let main = inner(&self.handoff.main_claim, &prepared.base.main_prime)?;
        let mask = inner(&self.handoff.mask_claim, &prepared.base.mask_prime)?;
        let fresh = inner(&self.handoff.fresh_claim, &prepared.fresh_row)?;
        let lhs = main.add(&mask).add(&lambda.mul(&fresh));
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
            &prepared.fresh_row,
            &prepared.fresh_tail,
        )?;
        let mask_n = self.handoff.mask_code.n();
        let mut fold_main = true;
        let mut fold_mask = true;
        let mut fold_fresh = true;
        for (index, position) in self.positions.iter().enumerate() {
            let mask_position = *position % mask_n;
            let main_symbol = *encoded_main
                .get(*position as usize)
                .ok_or(Error::InvalidIndex)?;
            let expected_main = opened.star_main[index].add(&lambda.mul(&opened.fold_main[index]));
            let mask_symbol = *encoded_mask
                .get(mask_position as usize)
                .ok_or(Error::InvalidIndex)?;
            let expected_mask =
                opened.star_mask[index].add(&lambda.mul(&opened.fold_mask[index]));
            let fresh_symbol = *encoded_fresh
                .get(mask_position as usize)
                .ok_or(Error::InvalidIndex)?;
            fold_main &= main_symbol == expected_main;
            fold_mask &= mask_symbol == expected_mask;
            fold_fresh &= fresh_symbol == opened.fold_fresh[index];
        }
        let rejection = if !base_linear {
            Some(BaseRejection::TargetLinear)
        } else if !fold_main {
            Some(BaseRejection::FoldMain)
        } else if !fold_mask {
            Some(BaseRejection::FoldMask)
        } else if !fold_fresh {
            Some(BaseRejection::FoldFresh)
        } else {
            None
        };
        let member_roots: Vec<Digest> = self.basis.members.iter().map(|member| member.root).collect();
        let star_root = self
            .registry
            .get(self.star_id.ok_or(Error::InvalidSessionState)?)
            .ok_or(Error::ProverStateMissing)?
            .root;
        let star_mask_root = self
            .registry
            .get(self.star_mask_id.ok_or(Error::InvalidSessionState)?)
            .ok_or(Error::ProverStateMissing)?
            .root;
        if let Some(layer) = rejection {
            return Ok(BaseFinalCheck::Rejected(BaseFinalRejection {
                layer,
                fold_identity: self.expected_identity,
                star_root,
                star_mask_root,
                member_roots,
                fresh_root: self.basis.fresh_root,
                lambda,
                target_star,
                base_positions: self.positions,
                base_linear,
                fold_main,
                fold_mask,
                fold_fresh,
            }));
        }
        Ok(BaseFinalCheck::Passed(BaseFinalOutcome {
            fold_identity: self.expected_identity,
            star_root,
            star_mask_root,
            member_roots,
            fresh_root: self.basis.fresh_root,
            lambda,
            target_star,
            base_positions: self.positions,
            base_linear,
            fold_main,
            fold_mask,
            fold_fresh,
            openings_verified: true,
            member_openings_verified: true,
            fresh_opening_verified: true,
            rejection: None,
            pending: true,
            complete_base: false,
            sampled_fold_relation_only: true,
            complete_fold_binding: false,
            fresh_term_checked_against_fold_root: true,
            projection_only: true,
            conditional_local_experiment: true,
            cs_connected: false,
            reference_iopp_path: false,
            chain_handoff: true,
            check_rounds,
            check_value,
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

    pub fn handoff(&self) -> &FoldHandoff {
        &self.handoff
    }

    pub fn basis(&self) -> &FoldBasis {
        &self.basis
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
