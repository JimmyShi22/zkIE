//! Isolated fixture group commitment, SEND sample registration and BAT-to-FOLD
//! component handoff.
//!
//! This is not a production verification and not a complete soundness result.
//! Each member oracle of the fixture group has its own runtime tree, code,
//! message rows, encoding tail, salts, reserved sample pads and root. The
//! ordered root message is absorbed into the session chain as the SEND root/seed
//! round, the shared seed is drawn from that round, the complete ordered sample
//! answers are received and absorbed, the sampled registry and sample-form
//! claims are registered, the member openings are checked, and only then is the
//! same chain moved to BAT. The extra column draws are after the answers, never
//! between the seed and the answers. The ordered list of roots is not called one
//! group root, and the chain transcript value is not a group commitment root.
//!
//! The roots and plans are fixture values checked against the fixture's own
//! independently re-encoded expectation and the member openings. They are not
//! SEND receipts, not G_u or asset roots, and not a production trust anchor.
//! The opening check covers the drawn positions, not every row. The sample
//! claims are pending: the verifier registers them from the received answers
//! and the drawn seed; it does not compare them with the provider's plaintext
//! answers, and SEND/BAT do not decide truth. A caller that replaces both the
//! witness and the fixture expectation has no independent trusted binding here;
//! that remains unclosed.
//!
//! Group source table:
//!
//! | item | source and layout |
//! | --- | --- |
//! | member messages | deterministic fixture values per tag/oracle, `main_rows` rows of the main code's `ell` entries and `mask_rows` rows of the mask code's `ell` entries |
//! | member tails | deterministic fixture values, one tail of the code's `r` entries per row; encoded with the row and not part of the FOLD relation view |
//! | member tree | one part, `main_symbols = main_rows * 3` and `mask_symbols = mask_rows * 3`, `log_n` from the main code, `log_n_zk` from the mask code, `2^log_n` columns; each leaf holds every main symbol, then every mask symbol, then a 40-byte salt |
//! | pad | one public deterministic pad per answer, reserved in the member mask row (or the main row when no mask exists) at `ell - answers + answer`; encoded into the commitment before the seed; the seed's `beta` only supplies the linear coefficient; not hiding |
//! | roots | one root per member, committed by the runtime tree; the ordered list of roots is the declaration, not a concatenated group root |
//!
//! Sample source table:
//!
//! | item | source |
//! | --- | --- |
//! | shared seed | drawn from the root/seed round after the fixed roots; one seed per answer number, `z` of the largest member cube and a nonzero `beta` |
//! | answer order | by answer number, then member order; each member contributes its own answers |
//! | sample form | the existing sample cube and pad form over the member's own prefix of the shared seed |
//! | answers | received from the provider in the fixed order; the verifier records the target |
//! | registration | a pending claim per answer in the sampled registry; the claim is added to the corresponding BAT sample rows |
//! | plaintext comparison | the provider's evaluation over the same fixed member witness is an auxiliary test oracle only |
//!
//! The BAT group's base and mask codes are the FOLD shape's codes; the member
//! mask width is a power of two; the BAT layout is the FOLD fixture layout. No
//! field is copied from a width adapter. Plain and generator targets are
//! computed by BAT's own combination against the group tables reconstructed
//! from the committed member messages. Sample targets are the received answers,
//! not the provider's plaintext values.
//!
//! FOLD row source table:
//!
//! | item | source |
//! | --- | --- |
//! | main coefficients | BAT combined main coefficients (plain, generator and received-answer sample claims) |
//! | old-mask coefficients | BAT combined mask coefficients |
//! | kept forms | none in this fixture |
//! | main messages | group table reconstructed from the committed member main messages at the BAT layout offsets |
//! | old-mask messages | group table reconstructed from the committed member mask messages at the BAT layout offsets |
//! | fresh mask | deterministic fresh-mask witness committed under its own mask tree; the fresh-mask root and sum are fixed at the FOLD start after BAT |
//! | tails | committed member tails; not part of the FOLD relation view |
//!
//! The FOLD verifier consumes the fixed row and the prover messages; it does
//! not re-open the member roots, which the group session checks before `x`.
//!
//! Order and ownership: the ordered group roots are bound as the seed round,
//! the shared seed is drawn, every answer is received in the fixed order and the
//! complete answer list is absorbed, the sampled/sample claims are registered,
//! the member opening positions are drawn and checked, the chain is moved to
//! BAT, BAT freezes the group and draws `x`, the advanced chain moves to the
//! FOLD-only entry, the fixed row is bound, the start message fixes the fresh
//! root and sum before epsilon, each round message is absorbed before its gamma,
//! and the final output stays pending. Round counts and values are recorded
//! across every move.
//!
//! FOLD does not decide truth: a wrong BAT target or a well-formed wrong sample
//! answer still completes the rounds and remains pending. Wrong shape, code,
//! group, oracle, layout, root, opening, answer order and challenge order
//! violations fail closed. The wrong-target case is not evidence that FOLD
//! judges an arbitrary declaration.
//!
//! The code-switch continuation runs on the same chain and the same FOLD
//! declaration after `fold_finish`: it fixes two new switched-row commitments,
//! draws a SEND seed and answer, draws the old positions and theta in one
//! challenge round, authenticates the old columns from the original
//! commitments, registers the pending switch claims over the new group
//! declaration and combines them with theta. The switched mask message carries
//! the FOLD fresh row after the old mask and tail, and the combined mask claim
//! carries the FOLD fresh coefficients at the same offset so the FOLD target
//! term is preserved. BASE then consumes the switched rows and opens their
//! commitments; the old FOLD objects are not relabelled as the switch output.
//! The code-switch claim stays pending and the stage does not prove the new
//! group true. This path is an isolated fixture, not the reference proof.
//!
//! The BASE continuation runs on the same chain after `fold_finish`: the claim,
//! target and FOLD identity are taken from the same FOLD output; the old main
//! and old mask rows are the FOLD prover's end rows; the fresh row is the FOLD
//! prover's folded fresh message. The star rows are the BASE prover's own
//! fixture rows; their two commitments and target* are fixed before lambda, and
//! M'/R' and the fresh rows are fixed before the position draw. No extra
//! commitment is created for the old or fresh rows. At the drawn positions the
//! verifier opens every original member commitment and the FOLD fresh mask
//! tree, folds the opened columns with the transcript's fold challenges, and
//! compares them with the received rows: the combined main and mask rows equal
//! the star openings plus lambda times the folded original columns, and the
//! fresh row equals the folded fresh columns. The extended linear check and
//! these folded-column checks are recorded. This is a conditional local
//! experiment, not the reference standard IOPP path: the code-switch stage is
//! not connected, so the reference path FOLD -> CS -> final SingleRows -> BASE
//! is not completed and skipping CS is not claimed to be equivalent to it. The
//! result stays pending: it is not a complete BASE, IOPP bound, complete
//! proximity result, production pass or valid FOLD-to-BASE handoff. The
//! folded-column relation is checked only at the drawn positions, so the fold
//! binding is sampled, not complete, and one equality at one lambda is not a
//! general wrong-declaration rejection.
//!
//! The single G_u witness from the SEND fixture is not applied to this group:
//! each member has its own shape-checked fixture commitment and a per-member
//! G_u-shaped view for the shared sample form. Unsupported plans, shapes, answer
//! counts and seed lengths are rejected. Unclosed: real SEND/G_u and send
//! receipts, authenticated assets, fresh-mask openings and proximity, folded
//! tails, a relation/proximity connection, random hiding, single-shot semantics,
//! production and performance acceptance, a complete BASE/IOPP result, a
//! complete (not sampled) fold binding, the CS stage, a reference-path
//! equivalence claim, a deterministic wrong-declaration guarantee, and a trust
//! anchor independent of the fixture expectation. A zero-fold degenerate input
//! is not used to prove general conclusions. The plan's reader and binding are a
//! valid but inert scaffold required by the existing plan type; the FOLD-only
//! entry never reads them and they are not a witness or an asset binding.
//! `verify_request` is unchanged.

use std::sync::{Arc, Mutex};

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::proof_header::PROOF_VERSION_WORD;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::bat::{
    combine, BatGroupLayout, BatGroupRow, BatOracleShape, BatPart, BatRow, BatRowKind, BatSession,
    BatStatement,
};
use zkie_engine::pcs::request::binding::{BindingParameters, BindingStatement};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE, MODULUS};
use zkie_engine::pcs::request::fold::{
    fold_member_firsts, fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_mask_sum,
    fold_start_prove, validate_fold_mask_shape, FoldGroupLayout, FoldInputRow, FoldMask,
    FoldOutput, FoldPart, FoldProverSession, FoldShape, FoldStartMessage, MaskCoins, MaskOracle,
    OracleMetadata, RuntimeOracles, VerifiedGroup, VerifiedGroups, VerifiedStatement,
};
use zkie_engine::pcs::request::foldeq::{
    reader_table_identity, FoldeqBinding, FoldeqEdgeBinding, FoldeqPlan, FoldeqSession, GammaPath,
};
use zkie_engine::pcs::request::gu::{GuNotice, GuRegistry};
use zkie_engine::pcs::request::params::{RequestParams, LEAF_SALT_MIN_BYTES};
use zkie_engine::pcs::request::read::{unique_sorted, ReadLedger};
use zkie_engine::pcs::request::relation::{sample_row, GuRelationKind, GuRelationRow};
use zkie_engine::pcs::request::runtime_tree::{MaskTree, MaskTreePlan, RuntimeTreeOpening};
use zkie_engine::pcs::request::table::{
    commit_table_streaming, IndexReader, IndexShape, SampleSeed, TableDistances, VecTableSource,
};
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, TEST_COINS};

use crate::proof::base_final_session::{
    fold_coefficients, fold_identity, BaseFinalCheck, BaseFinalOutcome, BaseFinalRejection,
    BaseFinalSession, BaseOpenings, BaseProver, FoldBasis, FoldMember,
};
use crate::proof::cs_final_session::{CsBaseOpenings, CsOutcome, CsProver, CsSession};
use crate::proof::fold_handoff::{FoldHandoff, FoldHandoffAnchor};
use crate::proof::group_commitment_session::{
    member_gu_shapes, shared_answer_order, CommittedGroupMember, GroupCommitmentSession,
    GroupMemberPlan, GroupRootMessage, MemberFixture,
};

const FIXTURE_GROUP: u64 = 0;
const FIXTURE_FRESH: u64 = 3;
const FIXTURE_TREE: u64 = 5;
const FIXTURE_LEAF: u32 = 3;
const FIXTURE_MEMBERS: u64 = 3;
const FIXTURE_MAIN_ROWS: u64 = 2;
const FIXTURE_MASK_ROWS: u64 = 2;
const FIXTURE_K: u32 = 3;
const FIXTURE_H: u32 = 3;
const FIXTURE_PLANNED_ROWS: u64 = 6;
const FIXTURE_OPENINGS_PER_ORACLE: usize = 2;
const FIXTURE_MEMBER_TREE_BASE: u64 = 10;
const FIXTURE_MEMBER_LEAF: u32 = 4;
const FIXTURE_BASE_POSITIONS: usize = 2;

/// A deterministic numbered source for the fixture's fresh mask. It is public,
/// reproducible and provides no hiding; it must not be used outside a test or
/// fixture context.
#[derive(Clone, Debug)]
pub struct FixtureMaskCoins {
    seed: Digest,
    draw: u32,
}

impl FixtureMaskCoins {
    pub fn new(tag: u64) -> Self {
        let byte = (tag as u8).wrapping_mul(31).wrapping_add(0x5a);
        Self {
            seed: Digest([byte; 32]),
            draw: 0,
        }
    }

    fn value_at(&self, number: u32) -> Result<Fp3> {
        let mut coefficients = [Fp::ZERO; E_DEGREE];
        for (coordinate, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient = zkie_engine::pcs::request::field::uniform_fp(|attempt| {
                let digest = hash_labeled(
                    tag_word(TEST_COINS, 1 + coordinate as u64),
                    number as u64,
                    Some(&self.seed),
                    &attempt.to_le_bytes(),
                )?;
                Ok(u64::from_le_bytes(
                    digest.0[0..8].try_into().map_err(|_| Error::InvalidEncoding)?,
                ))
            })?;
        }
        Ok(Fp3::new(coefficients[0], coefficients[1], coefficients[2]))
    }
}

impl MaskCoins for FixtureMaskCoins {
    fn draw(&mut self) -> Result<(Fp3, u32)> {
        let number = self.draw;
        self.draw = self.draw.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok((self.value_at(number)?, number))
    }

    fn redraw(&self, number: u32) -> Result<Fp3> {
        self.value_at(number)
    }
}

fn fixture_codes() -> Result<(CodeParams, CodeParams)> {
    let code = build_code_plan(4, 8, 1)?;
    Ok((code, code))
}

fn fixture_shape() -> Result<FoldShape> {
    let (code, mask_code) = fixture_codes()?;
    Ok(FoldShape {
        code,
        mask_code,
        k: FIXTURE_K,
        h: FIXTURE_H,
        mask_leaf_class: FIXTURE_LEAF,
    })
}

fn row_bits(rows: u64) -> u32 {
    if rows <= 1 {
        0
    } else {
        64 - (rows - 1).leading_zeros()
    }
}

fn table_len(rows: u64, ell: u64) -> Result<usize> {
    if ell == 0 || !ell.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let bits = row_bits(rows)
        .checked_add(ell.trailing_zeros())
        .ok_or(Error::ResourceLimit)?;
    1usize.checked_shl(bits).ok_or(Error::ResourceLimit)
}

/// Reorders a row-major member table (`row * ell + place`) into BAT's virtual
/// order (`row + place * 2^row_bits`), zero-padding virtual rows past the real
/// rows. The sample-form builder is row-major; BAT terms are column-major.
fn to_bat_table(values: &[Fp3], rows: u64, ell: u64) -> Result<Vec<Fp3>> {
    let bits = row_bits(rows);
    let table_len = table_len(rows, ell)?;
    let expected = usize::try_from(
        rows.checked_mul(ell).ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    if values.len() != expected {
        return Err(Error::ShapeMismatch);
    }
    let mut out = vec![Fp3::ZERO; table_len];
    let real_rows = usize::try_from(rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(ell).map_err(|_| Error::ResourceLimit)?;
    let row_stride = 1usize << bits;
    for row in 0..real_rows {
        for place in 0..ell {
            let at = row
                .checked_add(
                    place
                        .checked_mul(row_stride)
                        .ok_or(Error::ResourceLimit)?,
                )
                .ok_or(Error::ResourceLimit)?;
            let slot = out.get_mut(at).ok_or(Error::InvalidIndex)?;
            *slot = values[row * ell + place];
        }
    }
    Ok(out)
}

fn coefficient_table(len: usize, seed: u64, tag: u64) -> Vec<Fp3> {
    (0..len)
        .map(|index| {
            Fp3::from_u64(
                seed.wrapping_add(tag.wrapping_mul(0x40))
                    .wrapping_add(index as u64),
            )
        })
        .collect()
}

fn dot(left: &[Fp3], right: &[Fp3]) -> Result<Fp3> {
    if left.len() != right.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (a, b) in left.iter().zip(right) {
        sum = sum.add(&a.mul(b));
    }
    Ok(sum)
}

/// Builds the group message table from the committed member messages. The
/// index mapping is the layout's own: member message entries are row-major and
/// placed at the member's first group row, exactly where BAT carries the
/// coefficients.
fn build_group_table(
    layout: &FoldGroupLayout,
    bat_layout: &BatGroupLayout,
    part: BatPart,
    members: &[CommittedGroupMember],
) -> Result<Vec<Fp3>> {
    let group_bits = bat_layout.group_row_bits(part);
    let group_len = bat_layout.group_len(part)?;
    let mut values = vec![Fp3::ZERO; group_len];
    if layout.members.len() != bat_layout.members.len() || layout.members.len() != members.len() {
        return Err(Error::StatementMismatch);
    }
    for (index, (fold_member, bat_member)) in
        layout.members.iter().zip(&bat_layout.members).enumerate()
    {
        if fold_member.oracle != bat_member.oracle || fold_member.kept {
            return Err(Error::UnsupportedPrimitive);
        }
        let member = members.get(index).ok_or(Error::StatementMismatch)?;
        if member.plan().oracle != fold_member.oracle {
            return Err(Error::StatementMismatch);
        }
        let rows = match part {
            BatPart::Main => fold_member.main_rows,
            BatPart::Mask => fold_member.mask_rows,
        };
        if rows == 0 {
            continue;
        }
        let ell = match part {
            BatPart::Main => member.plan().main_code.ell,
            BatPart::Mask => member.plan().mask_code.ell,
        };
        let messages = match part {
            BatPart::Main => member.main_messages(),
            BatPart::Mask => member.mask_messages(),
        };
        if messages.len() as u64 != rows || messages.iter().any(|message| message.len() as u64 != ell)
        {
            return Err(Error::ShapeMismatch);
        }
        let first = match part {
            BatPart::Main => bat_member.main_first,
            BatPart::Mask => bat_member.mask_first,
        }
        .ok_or(Error::StatementMismatch)?;
        for (row, message) in messages.iter().enumerate() {
            for (place, value) in message.iter().enumerate() {
                let at = first
                    .checked_add(row as u64)
                    .and_then(|value| value.checked_add((place as u64).checked_shl(group_bits)?))
                    .ok_or(Error::ResourceLimit)?;
                let slot = values.get_mut(at as usize).ok_or(Error::InvalidIndex)?;
                *slot = *value;
            }
        }
    }
    Ok(values)
}

fn claim_target(
    statement: &BatStatement,
    layout: &BatGroupLayout,
    row: &BatRow,
    group_main: &[Fp3],
    group_mask: &[Fp3],
) -> Result<Fp3> {
    let mut probe = row.clone();
    probe.target = Fp3::ZERO;
    let mapped = combine(statement, layout, &[probe], Fp3::ONE)?;
    if !mapped.kept.is_empty() {
        return Err(Error::UnsupportedPrimitive);
    }
    Ok(dot(&mapped.main, group_main)?.add(&dot(&mapped.mask, group_mask)?))
}

/// Builds a probe statement with the same shapes, zero sample-class rows and
/// joined members, solely to obtain BAT's own group layout before the real
/// sample targets are known. The layout depends only on the shapes and join
/// marks, so the real statement must reproduce it; the fixture checks that.
fn probe_bat_statement(
    shapes: &[BatOracleShape],
    code: CodeParams,
    mask_code: CodeParams,
) -> Result<(BatGroupLayout, BatStatement)> {
    let mut probe = BatStatement::new();
    for shape in shapes {
        probe.add_oracle(*shape)?;
    }
    for shape in shapes {
        probe.sample(shape.oracle, vec![BatRow::new(BatRowKind::Sample)])?;
    }
    let group = probe.new_group(code, mask_code)?;
    for shape in shapes {
        probe.join(group, shape.oracle, false)?;
    }
    let layout = probe.layout(group)?;
    Ok((layout, probe))
}

/// One deterministic fixture group shared natively by BAT and FOLD. Each member
/// oracle has its own committed runtime tree and root; the ordered roots are
/// bound into the session before the BAT challenge.
pub struct BatFoldFixture {
    tag: u64,
    target_shift: Fp3,
    shape: FoldShape,
    layout: FoldGroupLayout,
    shapes: Vec<BatOracleShape>,
    bat_layout: BatGroupLayout,
    plain_rows: Vec<BatRow>,
    generator_rows: Vec<BatRow>,
    members: Vec<CommittedGroupMember>,
    group_main: Vec<Fp3>,
    group_mask: Vec<Fp3>,
    tree_plan: MaskTreePlan,
    start: FoldStartMessage,
    witness_root: Digest,
    mask: FoldMask,
    witness_rows: Vec<Vec<Fp3>>,
    witness_tails: Vec<Vec<Fp3>>,
    salts: Vec<Vec<u8>>,
    reader: IndexReader,
    binding: FoldeqBinding,
}

impl BatFoldFixture {
    /// Builds the positive fixture.
    pub fn new(tag: u64) -> Result<Self> {
        Self::new_with_target_shift(tag, Fp3::ZERO)
    }

    /// Builds the fixture with a value added to the first BAT claim's target.
    /// The shift is a fixture-only negative hook for the pending-declaration
    /// test; the positive path uses zero.
    pub fn new_with_target_shift(tag: u64, target_shift: Fp3) -> Result<Self> {
        let shape = fixture_shape()?;
        validate_fold_mask_shape(&shape)?;
        let shapes = (0..FIXTURE_MEMBERS)
            .map(|oracle| BatOracleShape {
                oracle,
                main_rows: FIXTURE_MAIN_ROWS,
                mask_rows: FIXTURE_MASK_ROWS,
                main_code: Some(shape.code),
                mask_code: Some(shape.mask_code),
                answers: 1,
            })
            .collect::<Vec<_>>();
        let (bat_layout, probe) = probe_bat_statement(&shapes, shape.code, shape.mask_code)?;
        let layout = probe.to_fold_layout_checked(FIXTURE_GROUP, &shape)?;
        if layout.group != FIXTURE_GROUP
            || layout.members.iter().any(|member| member.kept)
            || bat_layout.k != shape.k
            || bat_layout.h != shape.h
        {
            return Err(Error::UnsupportedPrimitive);
        }
        let mut members = Vec::with_capacity(shapes.len());
        for entry in &shapes {
            let member = CommittedGroupMember::build(MemberFixture {
                tag,
                oracle: entry.oracle,
                group: FIXTURE_GROUP,
                tree_number: FIXTURE_MEMBER_TREE_BASE + entry.oracle,
                leaf_class: FIXTURE_MEMBER_LEAF,
                main_code: shape.code,
                mask_code: shape.mask_code,
                main_rows: FIXTURE_MAIN_ROWS,
                mask_rows: FIXTURE_MASK_ROWS,
                answers: entry.answers,
                pad_shift: Fp3::ZERO,
            })?;
            if member.plan().main_code != shape.code
                || member.plan().mask_code != shape.mask_code
                || member.plan().main_rows != FIXTURE_MAIN_ROWS
                || member.plan().mask_rows != FIXTURE_MASK_ROWS
                || member.plan().answers != entry.answers
            {
                return Err(Error::StatementMismatch);
            }
            members.push(member);
        }
        let main_len = table_len(FIXTURE_MAIN_ROWS, shape.code.ell)?;
        let mask_len = table_len(FIXTURE_MASK_ROWS, shape.mask_code.ell)?;
        let group_main = build_group_table(&layout, &bat_layout, BatPart::Main, &members)?;
        let group_mask = build_group_table(&layout, &bat_layout, BatPart::Mask, &members)?;

        let mut first = BatRow::new(BatRowKind::Plain);
        first.add_term(
            0,
            BatPart::Main,
            coefficient_table(main_len, 0x100, tag),
        );
        first.add_term(
            1,
            BatPart::Mask,
            coefficient_table(mask_len, 0x200, tag),
        );
        first.target = claim_target(
            &probe,
            &bat_layout,
            &first,
            &group_main,
            &group_mask,
        )?
        .add(&target_shift);

        let mut second = BatRow::new(BatRowKind::Plain);
        second.add_term(
            2,
            BatPart::Main,
            coefficient_table(main_len, 0x300, tag),
        );
        second.add_term(
            0,
            BatPart::Mask,
            coefficient_table(mask_len, 0x400, tag),
        );
        second.add_unit(1, BatPart::Main, 3, Fp3::from_u64(7 + tag % 97));
        second.add_unit(2, BatPart::Mask, 5, Fp3::from_u64(9 + tag % 89));
        second.target = claim_target(
            &probe,
            &bat_layout,
            &second,
            &group_main,
            &group_mask,
        )?;

        let mut third = BatRow::new(BatRowKind::Generator);
        third.add_run(
            1,
            BatPart::Main,
            1,
            3,
            vec![
                Fp3::from_u64(2 + tag % 83),
                Fp3::from_u64(3 + tag % 79),
            ],
        );
        third.add_term(
            2,
            BatPart::Mask,
            coefficient_table(mask_len, 0x500, tag),
        );
        third.target = claim_target(
            &probe,
            &bat_layout,
            &third,
            &group_main,
            &group_mask,
        )?;
        let plain_rows = vec![first, second];
        let generator_rows = vec![third];

        let tree_plan = MaskTreePlan {
            tree_number: FIXTURE_TREE,
            leaf_class: FIXTURE_LEAF,
            log_n: shape.mask_code.log_n,
            mask_rows: shape.mask_rows()?,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([0x41; 32]),
            configuration: Digest([0x42; 32]),
        };
        let groups = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: FIXTURE_GROUP,
                members: layout.members.iter().map(|member| member.oracle).collect(),
                fresh: Some(FIXTURE_FRESH),
                code: Some(shape.code),
                mask_code: Some(shape.mask_code),
            }],
        };
        let mut coins = FixtureMaskCoins::new(tag);
        let mut mask = fold_mask_make(&shape, &mut coins)?;
        fold_mask_make_tail(&shape, &mut mask, &mut coins)?;
        mask.sent_as(FIXTURE_FRESH)?;
        let notice = fold_mask_notice(&shape)?;
        let oracle = MaskOracle::prover_view(FIXTURE_FRESH, FIXTURE_GROUP, &notice);
        let salts: Vec<Vec<u8>> = (0..tree_plan.columns())
            .map(|position| {
                let byte = ((position as u8).wrapping_mul(3) ^ 0x5a) ^ (tag as u8);
                vec![byte; tree_plan.salt_bytes]
            })
            .collect();
        let start_prover = fold_start_prove(
            &shape,
            &mut mask,
            &coins,
            &oracle,
            &groups,
            tree_plan,
            &salts,
        )?;
        let start = start_prover.message;

        let mask_rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
        let mask_columns =
            usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let mut witness_rows = Vec::with_capacity(mask_rows);
        for row in 0..mask_rows {
            let mut entries = Vec::with_capacity(mask_columns);
            for column in 0..mask_columns {
                entries.push(start_prover.draft.table.at(row, column)?);
            }
            witness_rows.push(entries);
        }
        let tail_columns =
            usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?;
        let mut witness_tails = Vec::with_capacity(mask_rows);
        for row in 0..mask_rows {
            let mut entries = Vec::with_capacity(tail_columns);
            for column in 0..tail_columns {
                entries.push(start_prover.draft.tail.at(row, column)?);
            }
            witness_tails.push(entries);
        }
        let witness_root =
            MaskTree::commit(tree_plan, &shape, &witness_rows, &witness_tails, &salts)?.root;
        if witness_root != start.root {
            return Err(Error::StatementMismatch);
        }

        let (reader, binding) = fixture_plan_scaffold()?;
        Ok(Self {
            tag,
            target_shift,
            shape,
            layout,
            shapes,
            bat_layout,
            plain_rows,
            generator_rows,
            members,
            group_main,
            group_mask,
            tree_plan,
            start,
            witness_root,
            mask,
            witness_rows,
            witness_tails,
            salts,
            reader,
            binding,
        })
    }

    pub fn tag(&self) -> u64 {
        self.tag
    }

    pub fn shape(&self) -> &FoldShape {
        &self.shape
    }

    pub fn layout(&self) -> &FoldGroupLayout {
        &self.layout
    }

    pub fn start_message(&self) -> &FoldStartMessage {
        &self.start
    }

    pub fn witness_root(&self) -> Digest {
        self.witness_root
    }

    pub fn group(&self) -> u64 {
        FIXTURE_GROUP
    }

    pub fn planned_rows(&self) -> u64 {
        FIXTURE_PLANNED_ROWS
    }

    pub fn member_plans(&self) -> Vec<GroupMemberPlan> {
        self.members
            .iter()
            .map(|member| member.plan().clone())
            .collect()
    }

    pub fn group_root_message(&self) -> GroupRootMessage {
        GroupRootMessage::from_plans(&self.member_plans())
    }

    pub fn open_members(&self, positions: &[Vec<u64>]) -> Result<Vec<RuntimeTreeOpening>> {
        if positions.len() != self.members.len() {
            return Err(Error::ShapeMismatch);
        }
        self.members
            .iter()
            .zip(positions)
            .map(|(member, list)| member.open(list))
            .collect()
    }

    /// The fresh mask tree's public plan.
    pub fn fresh_plan(&self) -> MaskTreePlan {
        self.tree_plan
    }

    /// Re-commits the stored fresh mask rows and tails and opens the given
    /// positions. The root is checked against the started mask root by the
    /// caller.
    pub fn open_fresh(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        MaskTree::commit(
            self.tree_plan,
            &self.shape,
            &self.witness_rows,
            &self.witness_tails,
            &self.salts,
        )?
        .open(positions)
    }

    /// The committed fresh mask encoding tails, one row per fresh mask row.
    pub fn fresh_tails(&self) -> &[Vec<Fp3>] {
        &self.witness_tails
    }

    /// Re-encodes every member message and tail and re-commits its tree. The
    /// caller compares the results with the member plans' roots.
    pub fn recompute_member_roots(&self) -> Result<Vec<Digest>> {
        self.members
            .iter()
            .map(CommittedGroupMember::recompute_root)
            .collect()
    }

    /// The prover fixture's plaintext sample answers over the fixed member
    /// witness for the given shared seed set. This is an auxiliary test
    /// oracle, not a verifier comparison and not a protocol source.
    pub fn provider_answers(&self, seeds: &[SampleSeed]) -> Result<Vec<Fp3>> {
        let plans = self.member_plans();
        let shapes = member_gu_shapes(&plans)?;
        let mut registry = GuRegistry::new();
        for (shape, plan) in shapes.iter().zip(&plans) {
            registry.receive(
                shape,
                &GuNotice {
                    root: plan.root,
                    oracle: shape.oracle,
                    tree_number: shape.tree.tree_number,
                    part: shape.part,
                },
            )?;
        }
        let order = shared_answer_order(&shapes)?;
        let mut answers = Vec::with_capacity(order.len());
        for (answer, oracle) in order {
            let shape = shapes
                .iter()
                .find(|shape| shape.oracle == oracle)
                .ok_or(Error::StatementMismatch)?;
            let member = self
                .members
                .iter()
                .find(|member| member.plan().oracle == oracle)
                .ok_or(Error::StatementMismatch)?;
            answers.push(
                sample_row(shape, &registry, seeds, answer, Fp3::ZERO)?
                    .evaluate(&member.gu_witness())?,
            );
        }
        Ok(answers)
    }

    /// Builds the BAT statement after the SEND answers are registered. The
    /// sample rows come from the verifier's session; they are not the prover's
    /// plaintext answers.
    pub fn bat_statement_with_samples(
        &self,
        sample_rows: &[GuRelationRow],
    ) -> Result<BatStatement> {
        let mut statement = BatStatement::new();
        for shape in &self.shapes {
            statement.add_oracle(*shape)?;
        }
        let mut by_oracle: Vec<Vec<BatRow>> = vec![Vec::new(); self.shapes.len()];
        for row in sample_rows {
            if row.kind != GuRelationKind::Sample {
                return Err(Error::UnsupportedPrimitive);
            }
            let index = self
                .members
                .iter()
                .position(|member| member.plan().oracle == row.oracle)
                .ok_or(Error::StatementMismatch)?;
            let plan = self.members[index].plan();
            if row.group != FIXTURE_GROUP
                || row.main.len() as u64
                    != plan
                        .main_rows
                        .checked_mul(plan.main_code.ell)
                        .ok_or(Error::ResourceLimit)?
                || row.mask.len() as u64
                    != plan
                        .mask_rows
                        .checked_mul(plan.mask_code.ell)
                        .ok_or(Error::ResourceLimit)?
            {
                return Err(Error::ShapeMismatch);
            }
            let mut bat = BatRow::new(BatRowKind::Sample);
            if !row.main.is_empty() {
                bat.add_term(
                    row.oracle,
                    BatPart::Main,
                    to_bat_table(&row.main, plan.main_rows, plan.main_code.ell)?,
                );
            }
            if !row.mask.is_empty() {
                bat.add_term(
                    row.oracle,
                    BatPart::Mask,
                    to_bat_table(&row.mask, plan.mask_rows, plan.mask_code.ell)?,
                );
            }
            bat.target = row.target;
            by_oracle[index].push(bat);
        }
        for (index, shape) in self.shapes.iter().enumerate() {
            if by_oracle[index].len() != shape.answers as usize {
                return Err(Error::ShapeMismatch);
            }
            statement.sample(shape.oracle, by_oracle[index].clone())?;
        }
        let group = statement.new_group(self.shape.code, self.shape.mask_code)?;
        for shape in &self.shapes {
            statement.join(group, shape.oracle, false)?;
        }
        if statement.layout(group)? != self.bat_layout {
            return Err(Error::StatementMismatch);
        }
        for row in self.plain_rows.iter().chain(&self.generator_rows) {
            statement.add_row(group, row.clone())?;
        }
        Ok(statement)
    }

    /// Re-commits the stored mask rows and tails independently of the start
    /// message. This is the fixture's encoding-to-root consistency check.
    pub fn recompute_root(&self) -> Result<Digest> {
        Ok(
            MaskTree::commit(
                self.tree_plan,
                &self.shape,
                &self.witness_rows,
                &self.witness_tails,
                &self.salts,
            )?
            .root,
        )
    }

    fn metadata_for(&self, shape: &FoldShape) -> Result<OracleMetadata> {
        Ok(OracleMetadata {
            id: FIXTURE_FRESH,
            group: FIXTURE_GROUP,
            tree_number: FIXTURE_TREE,
            leaf_class: shape.mask_leaf_class,
            mask_rows: shape.mask_rows()?,
            mask_code: shape.mask_code,
        })
    }

    fn statement_for(
        &self,
        shape: &FoldShape,
        layout: &FoldGroupLayout,
    ) -> Result<VerifiedStatement> {
        if layout.members.iter().any(|member| member.kept) {
            return Err(Error::UnsupportedPrimitive);
        }
        VerifiedStatement::new(
            VerifiedGroups {
                groups: vec![VerifiedGroup {
                    group: layout.group,
                    members: layout.members.iter().map(|member| member.oracle).collect(),
                    fresh: Some(FIXTURE_FRESH),
                    code: Some(shape.code),
                    mask_code: Some(shape.mask_code),
                }],
            },
            vec![self.tree_plan],
            vec![self.metadata_for(shape)?],
        )
    }

    fn plan_for(
        &self,
        shape: &FoldShape,
        layout: &FoldGroupLayout,
        root: Digest,
    ) -> Result<FoldeqPlan> {
        let statement = self.statement_for(shape, layout)?;
        let mut runtime = RuntimeOracles::default();
        runtime.receive_mask(
            &statement,
            &self.tree_plan,
            FIXTURE_FRESH,
            &fold_mask_notice(shape)?,
            root,
        )?;
        Ok(FoldeqPlan {
            reader: self.reader.clone(),
            binding: self.binding.clone(),
            edge: FoldeqEdgeBinding::NotAnEdge,
            fold: *shape,
            statement: Arc::new(statement),
            oracles: Arc::new(Mutex::new(runtime)),
            mask_tree_number: FIXTURE_TREE,
            group_id: layout.group,
            leaf_class: FIXTURE_LEAF,
            seed_budget: self.reader.shape.answers as usize,
            ledger: Arc::new(Mutex::new(ReadLedger::new(1))),
            column_draws: 1,
            mask_query_count: 0,
            gamma_path: GammaPath::Direct {
                columns: self.reader.shape.v,
            },
            gu: None,
            omega: None,
            fold_layout: Some(layout.clone()),
        })
    }
}

fn fixture_plan_scaffold() -> Result<(IndexReader, FoldeqBinding)> {
    let shape = IndexShape {
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
    };
    let request = RequestParams::default_with_code(shape.code.log_n, shape.code.ell, shape.code.r)?;
    let parameters = BindingParameters::new(
        PROOF_VERSION_WORD,
        Digest([0x61; 32]),
        &request,
        Digest([0x62; 32]),
    )?;
    let configuration = parameters.digest()?;
    let rows = shape.rows() as usize;
    let columns = shape.code.n() as usize;
    let messages: Vec<Vec<Fp>> = (0..rows)
        .map(|row| {
            (0..shape.code.ell)
                .map(|column| Fp::from_u64((row * 7 + column as usize + 1) as u64))
                .collect()
        })
        .collect();
    let tails: Vec<Vec<Fp>> = (0..rows)
        .map(|row| {
            (0..shape.code.r)
                .map(|column| Fp::from_u64((row + column as usize + 1) as u64))
                .collect()
        })
        .collect();
    let salts: Vec<Vec<u8>> = (0..columns)
        .map(|position| vec![(position as u8).wrapping_add(1); shape.salt_bytes])
        .collect();
    let mut source = VecTableSource::new(messages, tails)?;
    let table =
        commit_table_streaming(&mut source, shape, configuration, &salts, 64, u64::MAX)?;
    let reader = table.reader;
    let statement = BindingStatement::new(
        parameters.version_word,
        parameters.protocol_identity,
        b"bat-fold-fixture".to_vec(),
        Digest([0x63; 32]),
        reader_table_identity(&reader)?,
    )?;
    let binding = FoldeqBinding::new(&parameters, statement, &reader)?;
    Ok((reader, binding))
}

/// The result of one fixture BAT-to-FOLD run. The group roots are fixture
/// commitments checked against fixture expectations and openings; the sample
/// rows remain plaintext fixture claims. The FOLD output is a pending
/// candidate, not an acceptance.
pub struct BatFoldOutcome {
    pub tag: u64,
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub bat_layout: BatGroupLayout,
    pub group_roots: Vec<Digest>,
    pub group_positions: Vec<Vec<u64>>,
    pub group_rounds: u64,
    pub group_value: Digest,
    pub seeds: Vec<SampleSeed>,
    pub answer_order: Vec<(u32, u64)>,
    pub provider_answers: Vec<Fp3>,
    pub received_answers: Vec<Fp3>,
    pub sample_rows: Vec<GuRelationRow>,
    pub sample_claims_in_bat: usize,
    pub bat_order: Vec<BatRowKind>,
    pub bat_planned_rows: u64,
    pub bat_input_targets: Vec<Fp3>,
    pub combined: BatGroupRow,
    pub x: Fp3,
    pub fold_row: FoldInputRow,
    pub start: FoldStartMessage,
    pub epsilon: Fp3,
    pub gamma: Vec<Fp3>,
    pub output: FoldOutput,
    pub auxiliary_value: Fp3,
    pub combined_matches_witness: bool,
    pub fold_matches_witness: bool,
    pub target_shifted: bool,
    pub pending: bool,
    pub fold_decides_truth: bool,
    pub native_shared_codes: bool,
    pub adapter_used: bool,
    pub group_roots_bound: bool,
    pub single_group_root_claimed: bool,
    pub openings_verified: bool,
    pub member_roots_from_fixture: bool,
    pub real_send_receipts: bool,
    pub group_roots_authenticated: bool,
    pub fold_messages_from_committed_members: bool,
    pub sample_claims_registered: bool,
    pub sample_claims_pending: bool,
    pub sample_claims_from_received_answers: bool,
    pub sample_answers_provider_constructed: bool,
    pub verifier_standard_answer_comparison: bool,
    pub send_judges_truth: bool,
    pub sample_pads_committed: bool,
    pub sample_pad_hiding: bool,
    pub real_assets_bound: bool,
    pub mask_root_authenticated: bool,
    pub production_wired: bool,
    pub single_shot: bool,
    pub proximity_connected: bool,
    pub folded_tails_derived: bool,
    pub hiding: bool,
    pub chain_handoff: bool,
    pub bat_chain_rounds: u64,
    pub bat_chain_value: Digest,
    pub fold_entry_rounds: u64,
    pub fold_entry_value: Digest,
    pub start_chain_rounds: u64,
    pub start_chain_value: Digest,
    pub final_rounds: u64,
    pub final_value: Digest,
    pub witness_root: Digest,
}

/// The fixture run result together with the FOLD continuation needed by the
/// BASE stage.
struct BatFoldCore {
    outcome: BatFoldOutcome,
    fold: FoldeqSession,
    handoff: FoldHandoff,
    old_main: Vec<Fp3>,
    old_mask: Vec<Fp3>,
    old_fresh: Vec<Fp3>,
    member_plans: Vec<GroupMemberPlan>,
    group_roots: Vec<Digest>,
}

/// Runs the isolated fixture with the prover's plaintext sample answers as the
/// received answers.
pub fn run_bat_fold(fixture: &BatFoldFixture, chain: Chain) -> Result<BatFoldOutcome> {
    run_bat_fold_with_provider(fixture, chain, |_, _, true_answers| {
        Ok(true_answers.to_vec())
    })
}

/// Runs the isolated fixture with a caller-supplied answer provider. The
/// provider receives the shared answer order, the drawn seeds and the prover's
/// auxiliary plaintext answers, and returns the answers to send. The verifier
/// registers the returned values as pending sample claims; it never compares
/// them with the auxiliary answers.
pub fn run_bat_fold_with_provider<F>(
    fixture: &BatFoldFixture,
    chain: Chain,
    provider: F,
) -> Result<BatFoldOutcome>
where
    F: FnOnce(&[(u32, u64)], &[SampleSeed], &[Fp3]) -> Result<Vec<Fp3>>,
{
    run_bat_fold_core(fixture, chain, provider).map(|core| core.outcome)
}

fn run_bat_fold_core<F>(
    fixture: &BatFoldFixture,
    chain: Chain,
    provider: F,
) -> Result<BatFoldCore>
where
    F: FnOnce(&[(u32, u64)], &[SampleSeed], &[Fp3]) -> Result<Vec<Fp3>>,
{
    if fixture.recompute_root()? != fixture.start.root {
        return Err(Error::StatementMismatch);
    }
    if fold_mask_sum(fixture.shape.k, fixture.mask.coefficients())? != fixture.start.sum {
        return Err(Error::StatementMismatch);
    }
    let member_plans = fixture.member_plans();
    let member_roots = fixture.recompute_member_roots()?;
    if member_roots.len() != member_plans.len()
        || member_roots
            .iter()
            .zip(&member_plans)
            .any(|(root, plan)| *root != plan.root)
    {
        return Err(Error::StatementMismatch);
    }

    let mut group = GroupCommitmentSession::new(chain, member_plans.clone())?;
    group.bind_roots(&fixture.group_root_message())?;
    group.draw_shared_seeds()?;
    let seeds = group.seeds().to_vec();
    let order = group.answer_order().to_vec();
    let true_answers = fixture.provider_answers(&seeds)?;
    if true_answers.len() != order.len() {
        return Err(Error::ShapeMismatch);
    }
    let received_answers = provider(&order, &seeds, &true_answers)?;
    if received_answers.len() != order.len() {
        return Err(Error::ShapeMismatch);
    }
    for ((_, oracle), answer) in order.iter().zip(&received_answers) {
        group.receive_answer(*oracle, *answer)?;
    }
    let group_positions = group.draw_positions(FIXTURE_OPENINGS_PER_ORACLE)?;
    let group_openings = fixture.open_members(&group_positions)?;
    group.verify_openings(&group_openings)?;
    let group_roots = group.roots().to_vec();
    let group_rounds = group.chain_rounds();
    let group_value = *group.chain_value();
    let sample_gu_rows = group.sample_rows().to_vec();
    let sample_claims_in_bat = sample_gu_rows.len();
    let group_chain = group.into_chain()?;
    if group_chain.rounds() != group_rounds || *group_chain.value() != group_value {
        return Err(Error::StatementMismatch);
    }

    let bat_statement = fixture.bat_statement_with_samples(&sample_gu_rows)?;
    let mut bat = BatSession::new(bat_statement, group_chain, fixture.planned_rows())?;
    let input = bat.begin_run(fixture.group())?;
    let bat_layout = input.layout.clone();
    let bat_order = input.rows.iter().map(|row| row.kind).collect::<Vec<_>>();
    let bat_input_targets = input.rows.iter().map(|row| row.target).collect::<Vec<_>>();
    let combined = bat.finish_run(input.clone())?;
    let x = bat.last_x().ok_or(Error::InvalidSessionState)?;
    let combined_value = dot(&combined.main, &fixture.group_main)?
        .add(&dot(&combined.mask, &fixture.group_mask)?);
    let combined_matches_witness = combined.target == combined_value;
    let combined_target = combined.target;
    let fold_row = bat
        .statement()
        .to_fold_input_row(combined.clone(), &bat_layout)?;
    if fold_row.target != combined_target {
        return Err(Error::StatementMismatch);
    }
    let bat_chain = bat.into_chain()?;
    let bat_chain_rounds = bat_chain.rounds();
    let bat_chain_value = *bat_chain.value();

    let plan = fixture.plan_for(&fixture.shape, &fixture.layout, fixture.start.root)?;
    let mut fold = FoldeqSession::open_fold_fixture(plan, bat_chain)?;
    let fold_entry_rounds = fold.chain().rounds();
    let fold_entry_value = *fold.transcript_value();
    if fold_entry_rounds != bat_chain_rounds || fold_entry_value != bat_chain_value {
        return Err(Error::StatementMismatch);
    }
    fold.fold_bind_row(fold_row.clone())?;
    fold.start_fold(&fixture.start)?;
    let epsilon = fold.epsilon().ok_or(Error::InvalidSessionState)?;
    let start_chain_rounds = fold.chain().rounds();
    let start_chain_value = *fold.transcript_value();
    fold.fold_rounds_start()?;

    let coins = FixtureMaskCoins::new(fixture.tag);
    let mut prover = FoldProverSession::new(
        &fixture.shape,
        &fixture.layout,
        &fixture.mask,
        fold_row.clone(),
        fixture.group_main.clone(),
        fixture.group_mask.clone(),
        Vec::new(),
    )?;
    if prover.sum() != fixture.start.sum {
        return Err(Error::StatementMismatch);
    }
    prover.start(epsilon, &coins)?;
    for _ in 0..fixture.shape.k {
        let q = prover.round()?;
        fold.receive_fold_round(q[0], q[2])?;
        let gamma = *fold
            .fold_gamma()
            .last()
            .ok_or(Error::InvalidSessionState)?;
        prover.bind(gamma)?;
    }
    let output = fold.fold_finish()?;
    if output.next_stage().is_ok() {
        return Err(Error::InvalidSessionState);
    }
    let end = prover.end()?;
    if !output.kept.is_empty() {
        return Err(Error::UnsupportedPrimitive);
    }
    let mut auxiliary = dot(&output.coefficients, prover.main_messages())?;
    auxiliary = auxiliary.add(&dot(&output.mask_coefficients, &end.mu)?);
    auxiliary = auxiliary.add(&dot(&output.fresh_coefficients, &end.fresh)?);
    let fold_matches_witness = output.target == auxiliary;

    let final_rounds = fold.chain().rounds();
    let final_value = *fold.transcript_value();
    let anchor = FoldHandoffAnchor {
        shape: fixture.shape,
        group: FIXTURE_GROUP,
        members: fixture
            .layout
            .members
            .iter()
            .map(|member| member.oracle)
            .collect(),
        fresh: fixture.start.id,
        fresh_root: fixture.start.root,
        tree_number: fixture.start.tree_number,
        layout: fixture.layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(&output, &anchor)?;
    let old_main = prover.main_messages().to_vec();
    let old_mask = end.mu.clone();
    let old_fresh = end.fresh.clone();
    let outcome = BatFoldOutcome {
        tag: fixture.tag,
        shape: fixture.shape,
        layout: fixture.layout.clone(),
        bat_layout,
        group_roots: group_roots.clone(),
        group_positions,
        group_rounds,
        group_value,
        seeds,
        answer_order: order,
        provider_answers: true_answers,
        received_answers,
        sample_rows: sample_gu_rows,
        sample_claims_in_bat,
        bat_order,
        bat_planned_rows: fixture.planned_rows(),
        bat_input_targets,
        combined,
        x,
        fold_row,
        start: fixture.start.clone(),
        epsilon,
        gamma: fold.fold_gamma().to_vec(),
        output,
        auxiliary_value: auxiliary,
        combined_matches_witness,
        fold_matches_witness,
        target_shifted: fixture.target_shift != Fp3::ZERO,
        pending: true,
        fold_decides_truth: false,
        native_shared_codes: true,
        adapter_used: false,
        group_roots_bound: true,
        single_group_root_claimed: false,
        openings_verified: true,
        member_roots_from_fixture: true,
        real_send_receipts: false,
        group_roots_authenticated: false,
        fold_messages_from_committed_members: true,
        sample_claims_registered: true,
        sample_claims_pending: true,
        sample_claims_from_received_answers: true,
        sample_answers_provider_constructed: true,
        verifier_standard_answer_comparison: false,
        send_judges_truth: false,
        sample_pads_committed: true,
        sample_pad_hiding: false,
        real_assets_bound: false,
        mask_root_authenticated: false,
        production_wired: false,
        single_shot: false,
        proximity_connected: false,
        folded_tails_derived: false,
        hiding: false,
        chain_handoff: true,
        bat_chain_rounds,
        bat_chain_value,
        fold_entry_rounds,
        fold_entry_value,
        start_chain_rounds,
        start_chain_value,
        final_rounds,
        final_value,
        witness_root: fixture.witness_root,
    };
    Ok(BatFoldCore {
        outcome,
        fold,
        handoff,
        old_main,
        old_mask,
        old_fresh,
        member_plans,
        group_roots,
    })
}

/// The fixture run plus the local BASE final-state wiring result.
pub struct BatFoldBaseOutcome {
    pub fold: BatFoldOutcome,
    pub base: BaseFinalOutcome,
    pub fold_exit_rounds: u64,
    pub fold_exit_value: Digest,
}

/// The explicit result of the BASE verification entry: a missing precondition
/// is a stage error, and a completed local check either passes (still pending)
/// or returns a typed rejection. The rejection is consumed by the entry and is
/// never returned as a passing outcome.
#[derive(Debug)]
pub enum BaseFinalRunError {
    Stage(Error),
    Rejected(BaseFinalRejection),
}

impl From<Error> for BaseFinalRunError {
    fn from(error: Error) -> Self {
        Self::Stage(error)
    }
}

/// Runs the same fixture chain and then wires the FOLD output into the local
/// BASE final-state check. A failed local check returns a typed rejection; a
/// passed local result remains pending. It is not a complete BASE or a valid
/// FOLD-to-BASE handoff.
pub fn run_bat_fold_with_base(
    fixture: &BatFoldFixture,
    chain: Chain,
) -> std::result::Result<BatFoldBaseOutcome, BaseFinalRunError> {
    run_bat_fold_with_base_and_target_shift(fixture, chain, Fp3::ZERO)
}

/// Runs the same wiring with a fixture target* shift for the fixed
/// inconsistent sample. A nonzero shift must fail the target/linear layer; it
/// is not a general wrong-declaration claim.
pub fn run_bat_fold_with_base_and_target_shift(
    fixture: &BatFoldFixture,
    chain: Chain,
    target_shift: Fp3,
) -> std::result::Result<BatFoldBaseOutcome, BaseFinalRunError> {
    let core = run_bat_fold_core(fixture, chain, |_, _, true_answers| {
        Ok(true_answers.to_vec())
    })?;
    run_base_from_core(fixture, core, target_shift)
}

/// Runs the same wiring with a caller-supplied SEND answer provider. This is
/// the wrong-upstream-declaration path: a well-formed wrong answer is carried
/// through BAT and FOLD on the same chain and then rejected by the local BASE
/// check.
pub fn run_bat_fold_with_base_and_provider<F>(
    fixture: &BatFoldFixture,
    chain: Chain,
    provider: F,
) -> std::result::Result<BatFoldBaseOutcome, BaseFinalRunError>
where
    F: FnOnce(&[(u32, u64)], &[SampleSeed], &[Fp3]) -> Result<Vec<Fp3>>,
{
    let core = run_bat_fold_core(fixture, chain, provider)?;
    run_base_from_core(fixture, core, Fp3::ZERO)
}

fn run_base_from_core(
    fixture: &BatFoldFixture,
    core: BatFoldCore,
    target_shift: Fp3,
) -> std::result::Result<BatFoldBaseOutcome, BaseFinalRunError> {
    let BatFoldCore {
        outcome,
        fold,
        handoff,
        old_main,
        old_mask,
        old_fresh,
        member_plans,
        group_roots,
    } = core;
    let code = outcome.shape.code;
    let mask_code = outcome.shape.mask_code;
    let fold_exit_rounds = fold.chain().rounds();
    let fold_exit_value = *fold.transcript_value();
    let base_chain = fold.take_chain_for_base()?;
    if base_chain.rounds() != fold_exit_rounds || *base_chain.value() != fold_exit_value {
        return Err(BaseFinalRunError::Stage(Error::StatementMismatch));
    }
    let star = fixture_base_star(outcome.tag, code, mask_code);
    let (old_main_tail, old_mask_tail, fresh_tail) = folded_tails(fixture, &handoff)?;
    let rows = handoff.single_rows(old_main, old_main_tail, old_mask, old_mask_tail)?;
    let identity = fold_identity(&handoff);
    let prover = BaseProver::new(
        rows,
        identity,
        old_fresh,
        fresh_tail,
        star.0,
        star.1,
        star.2,
        star.3,
        target_shift,
    )?;
    let basis = base_basis(fixture, &handoff, &member_plans, &group_roots)?;
    let mut session = BaseFinalSession::new(base_chain, handoff, basis)?;
    session.receive_commitments(prover.messages()?)?;
    let lambda = session.draw_lambda()?;
    session.receive_base_rows(prover.base_rows(lambda)?)?;
    let positions = session.draw_base_positions(FIXTURE_BASE_POSITIONS)?;
    let unique = unique_sorted(&positions);
    let member_openings = fixture.open_members(&vec![unique.clone(); member_plans.len()])?;
    let mask_positions: Vec<u64> = positions
        .iter()
        .map(|position| *position % mask_code.n())
        .collect();
    let fresh_opening = fixture.open_fresh(&unique_sorted(&mask_positions))?;
    let star_openings = prover.openings(&positions)?;
    let openings = BaseOpenings {
        star: star_openings.star,
        star_mask: star_openings.star_mask,
        members: member_openings,
        fresh: fresh_opening,
    };
    session.verify_openings(&openings)?;
    let base = match session.finish()? {
        BaseFinalCheck::Passed(base) => base,
        BaseFinalCheck::Rejected(rejection) => {
            return Err(BaseFinalRunError::Rejected(rejection));
        }
    };
    Ok(BatFoldBaseOutcome {
        fold: outcome,
        base,
        fold_exit_rounds,
        fold_exit_value,
    })
}

fn base_basis(
    fixture: &BatFoldFixture,
    handoff: &FoldHandoff,
    member_plans: &[GroupMemberPlan],
    roots: &[Digest],
) -> Result<FoldBasis> {
    if member_plans.len() != roots.len() {
        return Err(Error::ShapeMismatch);
    }
    let members = member_plans
        .iter()
        .zip(roots)
        .map(|(plan, root)| FoldMember {
            oracle: plan.oracle,
            main_code: plan.main_code,
            mask_code: plan.mask_code,
            root: *root,
            tree: plan.tree.clone(),
        })
        .collect();
    Ok(FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: fixture.fresh_plan(),
        members,
    })
}

/// The encoding-tail rows of the folded output, derived by folding the
/// committed member tails with the transcript's fold challenges. The result is
/// the tail that the encoded folded row must have for the folded-column
/// relation to hold.
fn folded_tails(
    fixture: &BatFoldFixture,
    handoff: &FoldHandoff,
) -> Result<(Vec<Fp3>, Vec<Fp3>, Vec<Fp3>)> {
    let layout = &handoff.layout;
    let main_coeff = fold_coefficients(handoff.k, &handoff.gamma)?;
    let mask_gamma = &handoff.gamma[..handoff.h as usize];
    let mask_coeff = fold_coefficients(handoff.h, mask_gamma)?;
    let main_firsts = fold_member_firsts(layout, FoldPart::Main)?;
    let mask_firsts = fold_member_firsts(layout, FoldPart::Mask)?;
    let main_len = usize::try_from(handoff.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_len = usize::try_from(handoff.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let mut main_tail = vec![Fp3::ZERO; main_len];
    let mut mask_tail = vec![Fp3::ZERO; mask_len];
    if fixture.members.len() != layout.members.len() {
        return Err(Error::ShapeMismatch);
    }
    for (index, (member, layout_member)) in fixture.members.iter().zip(&layout.members).enumerate() {
        let main_first = main_firsts[index].ok_or(Error::StatementMismatch)?;
        for row in 0..usize::try_from(layout_member.main_rows).map_err(|_| Error::ResourceLimit)? {
            let tail = member.main_tails().get(row).ok_or(Error::ShapeMismatch)?;
            if tail.len() != main_len {
                return Err(Error::ShapeMismatch);
            }
            let group = main_first
                .checked_add(row as u64)
                .ok_or(Error::ResourceLimit)?;
            let coefficient = *main_coeff
                .get(group as usize)
                .ok_or(Error::InvalidIndex)?;
            for (value, tail) in main_tail.iter_mut().zip(tail) {
                *value = value.add(&coefficient.mul(tail));
            }
        }
        let mask_first = mask_firsts[index].ok_or(Error::StatementMismatch)?;
        for row in 0..usize::try_from(layout_member.mask_rows).map_err(|_| Error::ResourceLimit)? {
            let tail = member.mask_tails().get(row).ok_or(Error::ShapeMismatch)?;
            if tail.len() != mask_len {
                return Err(Error::ShapeMismatch);
            }
            let group = mask_first
                .checked_add(row as u64)
                .ok_or(Error::ResourceLimit)?;
            let coefficient = *mask_coeff
                .get(group as usize)
                .ok_or(Error::InvalidIndex)?;
            for (value, tail) in mask_tail.iter_mut().zip(tail) {
                *value = value.add(&coefficient.mul(tail));
            }
        }
    }
    let fresh_rows = 1usize
        .checked_shl(handoff.h)
        .ok_or(Error::ResourceLimit)?;
    let fresh_tails = fixture.fresh_tails();
    if fresh_tails.len() != fresh_rows {
        return Err(Error::ShapeMismatch);
    }
    let mut fresh_tail = vec![Fp3::ZERO; mask_len];
    for (row, tail) in fresh_tails.iter().enumerate() {
        if tail.len() != mask_len {
            return Err(Error::ShapeMismatch);
        }
        let coefficient = *mask_coeff.get(row).ok_or(Error::InvalidIndex)?;
        for (value, tail) in fresh_tail.iter_mut().zip(tail) {
            *value = value.add(&coefficient.mul(tail));
        }
    }
    Ok((main_tail, mask_tail, fresh_tail))
}

/// A fixture run of the FOLD-to-code-switch-to-BASE continuation on the same
/// chain, plus the FOLD outcome.
pub struct BatFoldCsOutcome {
    pub fold: BatFoldOutcome,
    pub cs: CsOutcome,
}

const FIXTURE_CS_POSITIONS: usize = 2;
const FIXTURE_SWITCH_ELL: u64 = 20;

fn fixture_cs_star(
    tag: u64,
    code: CodeParams,
    mask_code: CodeParams,
) -> (Vec<Fp3>, Vec<Fp3>, Vec<Fp3>, Vec<Fp3>) {
    let main: Vec<Fp3> = (0..code.ell)
        .map(|index| Fp3::from_u64(0xa100 + tag * 0x10 + index))
        .collect();
    let main_tail: Vec<Fp3> = (0..code.r)
        .map(|index| Fp3::from_u64(0xa400 + tag * 0x10 + index))
        .collect();
    let mask: Vec<Fp3> = (0..mask_code.ell)
        .map(|index| Fp3::from_u64(0xa500 + tag * 0x10 + index))
        .collect();
    let mask_tail: Vec<Fp3> = (0..mask_code.r)
        .map(|index| Fp3::from_u64(0xa600 + tag * 0x10 + index))
        .collect();
    (main, main_tail, mask, mask_tail)
}

/// Runs the isolated fixture through FOLD and then the code-switch
/// continuation on the same chain. The old columns are authenticated at the
/// code-switch positions from the original commitments; the new group
/// declaration is registered and combined with theta; BASE consumes the switch
/// output commitments. The result rests pending.
pub fn run_bat_fold_cs(
    fixture: &BatFoldFixture,
    chain: Chain,
) -> Result<BatFoldCsOutcome> {
    let core = run_bat_fold_core(fixture, chain, |_, _, answers| Ok(answers.to_vec()))?;
    let BatFoldCore {
        outcome,
        fold,
        handoff,
        old_main,
        old_mask,
        old_fresh,
        member_plans,
        group_roots,
    } = core;
    let code = outcome.shape.code;
    let mask_code = outcome.shape.mask_code;
    let fold_exit_rounds = fold.chain().rounds();
    let fold_exit_value = *fold.transcript_value();
    let base_chain = fold.take_chain_for_base()?;
    if base_chain.rounds() != fold_exit_rounds || *base_chain.value() != fold_exit_value {
        return Err(Error::StatementMismatch);
    }
    let (main_tail, mask_tail, fresh_tail) = folded_tails(fixture, &handoff)?;
    let rows = handoff.single_rows(old_main, main_tail, old_mask, mask_tail)?;
    let switch_code = build_code_plan(code.log_n + 1, FIXTURE_SWITCH_ELL, 1)?;
    let switch_mask_code = build_code_plan(mask_code.log_n + 1, FIXTURE_SWITCH_ELL, 1)?;
    let star = fixture_cs_star(outcome.tag, switch_code, switch_mask_code);
    let prover = CsProver::new(
        &rows,
        &handoff,
        switch_code,
        switch_mask_code,
        old_fresh,
        fresh_tail,
        outcome.tag,
        star.0,
        star.1,
        star.2,
        star.3,
    )?;
    let plan = prover.plan();
    let identity = fold_identity(&handoff);
    let declarations = prover.declarations(identity)?;
    let basis = base_basis(fixture, &handoff, &member_plans, &group_roots)?;
    let mut session =
        CsSession::new(base_chain, handoff, basis, plan, switch_code, switch_mask_code)?;
    session.receive_declarations(declarations)?;
    let seed = session.draw_sample_seed()?;
    let answer = prover.sample_answer(std::slice::from_ref(&seed))?;
    session.receive_answer(answer)?;
    let (positions, theta) = session.draw_positions_theta(FIXTURE_CS_POSITIONS)?;
    let unique = unique_sorted(&positions);
    let member_openings = fixture.open_members(&vec![unique.clone(); member_plans.len()])?;
    let mask_positions: Vec<u64> = positions
        .iter()
        .map(|position| *position % mask_code.n())
        .collect();
    let fresh_opening = fixture.open_fresh(&unique_sorted(&mask_positions))?;
    let session_claims = session
        .authenticate_and_register(&member_openings, &fresh_opening)?
        .clone();
    let prover_claims = prover.claims(&positions, theta, std::slice::from_ref(&seed))?;
    if session_claims != prover_claims {
        return Err(Error::StatementMismatch);
    }
    let messages = prover.base_messages(&session_claims)?;
    session.receive_base_commitments(messages)?;
    let lambda = session.draw_lambda()?;
    let base_rows = prover.base_rows(&session_claims, lambda)?;
    session.receive_base_rows(base_rows)?;
    let base_positions = session.draw_base_positions(FIXTURE_BASE_POSITIONS)?;
    let (star_openings, star_mask_openings) = prover.star_openings(&base_positions)?;
    let switched = prover.openings(&unique_sorted(&base_positions))?;
    session.verify_base_openings(&CsBaseOpenings {
        star: star_openings,
        star_mask: star_mask_openings,
        switched,
    })?;
    Ok(BatFoldCsOutcome {
        fold: outcome,
        cs: session.finish()?,
    })
}

fn fixture_base_star(
    tag: u64,
    code: CodeParams,
    mask_code: CodeParams,
) -> (Vec<Fp3>, Vec<Fp3>, Vec<Fp3>, Vec<Fp3>) {
    let main: Vec<Fp3> = (0..code.ell)
        .map(|index| Fp3::from_u64(0x9100 + tag * 0x10 + index))
        .collect();
    let main_tail: Vec<Fp3> = (0..code.r)
        .map(|index| Fp3::from_u64(0x9400 + tag * 0x10 + index))
        .collect();
    let mask: Vec<Fp3> = (0..mask_code.ell)
        .map(|index| Fp3::from_u64(0x9500 + tag * 0x10 + index))
        .collect();
    let mask_tail: Vec<Fp3> = (0..mask_code.r)
        .map(|index| Fp3::from_u64(0x9600 + tag * 0x10 + index))
        .collect();
    (main, main_tail, mask, mask_tail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::base_final_session::BaseRejection;
    use crate::proof::cs_final_session::switch_tree_plan;
    use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

    fn chain(seed: &[u8]) -> Chain {
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest::ZERO, seed, &[], None).unwrap()
    }

    fn bat_stage(fixture: &BatFoldFixture, seed: &[u8]) -> Result<(Chain, FoldInputRow)> {
        let mut group =
            GroupCommitmentSession::new(chain(seed), fixture.member_plans())?;
        group.bind_roots(&fixture.group_root_message())?;
        group.draw_shared_seeds()?;
        let seeds = group.seeds().to_vec();
        let order = group.answer_order().to_vec();
        let answers = fixture.provider_answers(&seeds)?;
        for ((_, oracle), answer) in order.iter().zip(&answers) {
            group.receive_answer(*oracle, *answer)?;
        }
        let positions = group.draw_positions(FIXTURE_OPENINGS_PER_ORACLE)?;
        let openings = fixture.open_members(&positions)?;
        group.verify_openings(&openings)?;
        let statement = fixture.bat_statement_with_samples(group.sample_rows())?;
        let handoff = group.into_chain()?;
        let mut bat = BatSession::new(statement, handoff, fixture.planned_rows())?;
        let input = bat.begin_run(fixture.group())?;
        let combined = bat.finish_run(input.clone())?;
        let row = bat.statement().to_fold_input_row(combined, &input.layout)?;
        Ok((bat.into_chain()?, row))
    }

    fn fixture_bat_statement(
        fixture: &BatFoldFixture,
        seed: &[u8],
    ) -> Result<BatStatement> {
        let mut group =
            GroupCommitmentSession::new(chain(seed), fixture.member_plans())?;
        group.bind_roots(&fixture.group_root_message())?;
        group.draw_shared_seeds()?;
        let seeds = group.seeds().to_vec();
        let order = group.answer_order().to_vec();
        let answers = fixture.provider_answers(&seeds)?;
        for ((_, oracle), answer) in order.iter().zip(&answers) {
            group.receive_answer(*oracle, *answer)?;
        }
        fixture.bat_statement_with_samples(group.sample_rows())
    }

    fn open_session(
        fixture: &BatFoldFixture,
        shape: &FoldShape,
        layout: &FoldGroupLayout,
        handoff: Chain,
    ) -> Result<FoldeqSession> {
        let plan = fixture.plan_for(shape, layout, fixture.start.root)?;
        FoldeqSession::open_fold_fixture(plan, handoff)
    }

    #[test]
    fn the_fixture_group_is_native_and_the_witness_is_committed() {
        let fixture = BatFoldFixture::new(21).unwrap();
        assert_eq!(fixture.shape.code, fixture.shape.mask_code);
        let (bat_layout, probe) = probe_bat_statement(
            &fixture.shapes,
            fixture.shape.code,
            fixture.shape.mask_code,
        )
        .unwrap();
        assert_eq!(
            probe
                .to_fold_layout_checked(FIXTURE_GROUP, &fixture.shape)
                .unwrap(),
            fixture.layout
        );
        assert_eq!(bat_layout, fixture.bat_layout);
        assert_eq!(bat_layout.k, fixture.shape.k);
        assert_eq!(bat_layout.h, fixture.shape.h);
        assert!(fixture
            .layout
            .members
            .iter()
            .all(|member| !member.kept));
        assert_eq!(fixture.recompute_root().unwrap(), fixture.start.root);
        assert_eq!(fixture.witness_root, fixture.start.root);
        assert_eq!(
            fold_mask_sum(fixture.shape.k, fixture.mask.coefficients()).unwrap(),
            fixture.start.sum
        );
        let plans = fixture.member_plans();
        assert_eq!(plans.len(), 3);
        assert_eq!(
            fixture.recompute_member_roots().unwrap(),
            plans.iter().map(|plan| plan.root).collect::<Vec<_>>()
        );
        assert_eq!(
            fixture.group_root_message().entries.len(),
            plans.len()
        );
        assert_eq!(
            build_group_table(&fixture.layout, &bat_layout, BatPart::Main, &fixture.members)
                .unwrap(),
            fixture.group_main
        );
        assert_eq!(
            build_group_table(&fixture.layout, &bat_layout, BatPart::Mask, &fixture.members)
                .unwrap(),
            fixture.group_mask
        );
        // The first member's first message row sits at its BAT first offset;
        // later places move by the group row space.
        let first = &bat_layout.members[0];
        assert_eq!(first.oracle, 0);
        assert_eq!(first.main_first, Some(0));
        assert_eq!(
            fixture.group_main[0],
            fixture.members[0].main_messages()[0][0]
        );
        assert_eq!(
            fixture.group_main[1],
            fixture.members[0].main_messages()[1][0]
        );
        assert_eq!(
            fixture.group_main[1 << bat_layout.k],
            fixture.members[0].main_messages()[0][1]
        );
        assert_eq!(
            fixture.group_main.len(),
            1 << (bat_layout.k + bat_layout.log_ell)
        );
        assert_eq!(
            fixture.group_mask.len(),
            1 << (bat_layout.h + bat_layout.log_ell_zk)
        );
    }

    #[test]
    fn the_bat_row_enters_fold_unchanged_and_the_run_is_pending() {
        let fixture = BatFoldFixture::new(22).unwrap();
        let run = run_bat_fold(&fixture, chain(b"native handoff")).unwrap();
        assert_eq!(
            run.bat_order,
            vec![
                BatRowKind::Plain,
                BatRowKind::Plain,
                BatRowKind::Sample,
                BatRowKind::Sample,
                BatRowKind::Sample,
                BatRowKind::Generator,
            ]
        );
        assert_eq!(run.combined.group, FIXTURE_GROUP);
        assert_eq!(run.fold_row.group, run.combined.group);
        assert_eq!(run.fold_row.members, vec![0, 1, 2]);
        assert_eq!(run.fold_row.coefficients, run.combined.main);
        assert_eq!(run.fold_row.mask_coefficients, run.combined.mask);
        assert_eq!(run.fold_row.target, run.combined.target);
        assert!(run.combined_matches_witness);
        assert!(run.fold_matches_witness);
        assert!(run.pending);
        assert!(!run.fold_decides_truth);
        assert!(!run.adapter_used);
        assert!(run.native_shared_codes);
        assert!(run.group_roots_bound);
        assert!(!run.single_group_root_claimed);
        assert!(run.openings_verified);
        assert!(run.member_roots_from_fixture);
        assert!(!run.real_send_receipts);
        assert!(!run.group_roots_authenticated);
        assert!(run.fold_messages_from_committed_members);
        assert!(run.sample_claims_registered);
        assert!(run.sample_claims_pending);
        assert!(run.sample_claims_from_received_answers);
        assert!(run.sample_answers_provider_constructed);
        assert!(!run.verifier_standard_answer_comparison);
        assert!(!run.send_judges_truth);
        assert!(run.sample_pads_committed);
        assert!(!run.sample_pad_hiding);
        assert!(!run.real_assets_bound);
        assert!(!run.mask_root_authenticated);
        assert!(!run.production_wired);
        assert_eq!(run.group_roots.len(), 3);
        assert_eq!(run.group_positions.len(), 3);
        assert_eq!(run.group_roots, fixture.recompute_member_roots().unwrap());
        assert_eq!(run.group_rounds, 3);
        assert_eq!(run.bat_chain_rounds, run.group_rounds + 1);
        assert_eq!(run.sample_claims_in_bat, 3);
        assert_eq!(run.sample_rows.len(), 3);
        assert_eq!(run.seeds.len(), 1);
        assert_eq!(run.answer_order.len(), 3);
        assert_eq!(run.provider_answers, run.received_answers);
        assert_eq!(&run.bat_input_targets[2..5], run.received_answers.as_slice());
        assert_eq!(run.fold_entry_rounds, run.bat_chain_rounds);
        assert_eq!(run.fold_entry_value, run.bat_chain_value);
        assert_ne!(run.group_value, run.bat_chain_value);
        assert_eq!(run.gamma.len(), fixture.shape.k as usize);
        assert_eq!(run.start_chain_rounds, run.fold_entry_rounds + 1);
        assert_eq!(
            run.final_rounds,
            run.start_chain_rounds + fixture.shape.k as u64
        );
        assert!(run.output.next_stage().is_err());
        assert_eq!(run.output.target, run.auxiliary_value);
    }

    #[test]
    fn the_same_fixture_replays_with_another_chain() {
        let first_fixture = BatFoldFixture::new(23).unwrap();
        let second_fixture = BatFoldFixture::new(23).unwrap();
        let first = run_bat_fold(&first_fixture, chain(b"first chain")).unwrap();
        let second = run_bat_fold(&second_fixture, chain(b"second chain")).unwrap();
        assert_eq!(first.start, second.start);
        assert_eq!(first.witness_root, second.witness_root);
        assert_eq!(first.start.root, second.start.root);
        assert_eq!(first.start.sum, second.start.sum);
        assert_eq!(first.group_roots, second.group_roots);
        assert_ne!(first.x, second.x);
        assert_ne!(first.epsilon, second.epsilon);
        assert_ne!(first.combined.main, second.combined.main);
        assert!(first.combined_matches_witness);
        assert!(second.combined_matches_witness);
        assert!(first.fold_matches_witness);
        assert!(second.fold_matches_witness);
    }

    #[test]
    fn a_frozen_group_rejects_replacement() {
        let fixture = BatFoldFixture::new(24).unwrap();
        let mut bat = BatSession::new(
            fixture_bat_statement(&fixture, b"frozen group").unwrap(),
            chain(b"frozen group"),
            fixture.planned_rows(),
        )
        .unwrap();
        let input = bat.begin_run(fixture.group()).unwrap();
        assert!(matches!(
            bat.add_row(fixture.group(), BatRow::new(BatRowKind::Plain)),
            Err(Error::InvalidSessionState)
        ));
        assert!(matches!(
            bat.join(fixture.group(), 0, false),
            Err(Error::InvalidSessionState)
        ));
        let mut changed = input.clone();
        std::sync::Arc::make_mut(&mut changed.rows)[0].target = Fp3::from_u64(9999);
        assert!(matches!(
            bat.finish_run(changed),
            Err(Error::StatementMismatch)
        ));
        assert!(bat.finish_run(input).is_err());

        let mut bat = BatSession::new(
            fixture_bat_statement(&fixture, b"frozen layout").unwrap(),
            chain(b"frozen layout"),
            fixture.planned_rows(),
        )
        .unwrap();
        let input = bat.begin_run(fixture.group()).unwrap();
        let mut changed_layout = input.clone();
        changed_layout.layout.k += 1;
        assert!(bat.finish_run(changed_layout).is_err());

        let mut bat = BatSession::new(
            fixture_bat_statement(&fixture, b"frozen completed").unwrap(),
            chain(b"frozen completed"),
            fixture.planned_rows(),
        )
        .unwrap();
        let input = bat.begin_run(fixture.group()).unwrap();
        bat.finish_run(input).unwrap();
        assert!(bat.begin_run(fixture.group()).is_err());
    }

    #[test]
    fn wrong_shape_code_and_row_are_rejected() {
        let fixture = BatFoldFixture::new(25).unwrap();
        let (handoff, row) = bat_stage(&fixture, b"wrong shapes").unwrap();

        let mut wrong_k = fixture.shape;
        wrong_k.k = FIXTURE_K - 1;
        wrong_k.h = FIXTURE_H - 1;
        assert!(open_session(&fixture, &wrong_k, &fixture.layout, handoff.clone()).is_err());

        let mut wrong_code = fixture.shape;
        wrong_code.code = build_code_plan(4, 4, 1).unwrap();
        let mut session =
            open_session(&fixture, &wrong_code, &fixture.layout, handoff.clone()).unwrap();
        assert!(matches!(
            session.fold_bind_row(row.clone()),
            Err(Error::ShapeMismatch) | Err(Error::StatementMismatch)
        ));

        let mut wrong_layout = fixture.layout.clone();
        wrong_layout.k += 1;
        assert!(open_session(&fixture, &fixture.shape, &wrong_layout, handoff.clone()).is_err());

        let mut session =
            open_session(&fixture, &fixture.shape, &fixture.layout, handoff).unwrap();
        let mut truncated = row;
        truncated.coefficients.pop();
        assert!(matches!(
            session.fold_bind_row(truncated),
            Err(Error::ShapeMismatch)
        ));
    }

    #[test]
    fn wrong_oracle_and_root_are_rejected_before_epsilon() {
        let fixture = BatFoldFixture::new(26).unwrap();
        let (handoff, row) = bat_stage(&fixture, b"wrong oracle").unwrap();
        let mut session =
            open_session(&fixture, &fixture.shape, &fixture.layout, handoff).unwrap();
        session.fold_bind_row(row).unwrap();

        let mut wrong_oracle = fixture.start.clone();
        wrong_oracle.id = 0;
        assert!(matches!(
            session.start_fold(&wrong_oracle),
            Err(Error::StatementMismatch)
        ));

        let mut wrong_group = fixture.start.clone();
        wrong_group.claimed_group = FIXTURE_GROUP + 1;
        assert!(matches!(
            session.start_fold(&wrong_group),
            Err(Error::StatementMismatch)
        ));

        let mut wrong_tree = fixture.start.clone();
        wrong_tree.tree_number += 1;
        assert!(matches!(
            session.start_fold(&wrong_tree),
            Err(Error::StatementMismatch)
        ));

        let mut wrong_root = fixture.start.clone();
        wrong_root.root.0[0] ^= 1;
        assert!(matches!(
            session.start_fold(&wrong_root),
            Err(Error::StatementMismatch)
        ));

        assert!(session.epsilon().is_none());
        session.start_fold(&fixture.start).unwrap();
        assert!(session.epsilon().is_some());

        let fixture = BatFoldFixture::new(27).unwrap();
        let (handoff, row) = bat_stage(&fixture, b"wrong registry").unwrap();
        let mut forged_root = fixture.start.root;
        forged_root.0[1] ^= 1;
        let plan = fixture
            .plan_for(&fixture.shape, &fixture.layout, forged_root)
            .unwrap();
        let mut session = FoldeqSession::open_fold_fixture(plan, handoff).unwrap();
        session.fold_bind_row(row).unwrap();
        assert!(matches!(
            session.start_fold(&fixture.start),
            Err(Error::StatementMismatch)
        ));
    }

    #[test]
    fn order_violations_are_rejected() {
        let fixture = BatFoldFixture::new(28).unwrap();
        assert!(matches!(
            open_session(
                &fixture,
                &fixture.shape,
                &fixture.layout,
                chain(b"unadvanced")
            ),
            Err(Error::InvalidSessionState)
        ));

        let (handoff, row) = bat_stage(&fixture, b"round order").unwrap();
        let mut session =
            open_session(&fixture, &fixture.shape, &fixture.layout, handoff).unwrap();
        assert!(session.fold_finish().is_err());
        assert!(session
            .receive_fold_round(Fp3::ONE, Fp3::ONE)
            .is_err());
        assert!(session.fold_rounds_start().is_err());
        assert!(session.fold_bind_row(row.clone()).is_ok());
        assert!(session.fold_bind_row(row.clone()).is_err());
        session.start_fold(&fixture.start).unwrap();
        assert!(session.start_fold(&fixture.start).is_err());
        assert!(session.fold_bind_row(row).is_err());
        session.fold_rounds_start().unwrap();
        assert!(session.fold_finish().is_err());
        session
            .receive_fold_round(Fp3::ONE, Fp3::ONE)
            .unwrap();
        assert!(session.fold_finish().is_err());
        for _ in 1..fixture.shape.k {
            session
                .receive_fold_round(Fp3::from_u64(2), Fp3::from_u64(3))
                .unwrap();
        }
        assert!(session.fold_finish().is_ok());
        assert!(session
            .receive_fold_round(Fp3::ONE, Fp3::ONE)
            .is_err());
    }

    #[test]
    fn group_commitment_is_per_oracle_and_fixture_only() {
        let fixture = BatFoldFixture::new(30).unwrap();
        let run = run_bat_fold(&fixture, chain(b"fixture group commitment")).unwrap();
        assert!(run.group_roots_bound);
        assert!(!run.single_group_root_claimed);
        assert!(run.openings_verified);
        assert!(run.member_roots_from_fixture);
        assert!(!run.real_send_receipts);
        assert!(!run.group_roots_authenticated);
        assert!(run.fold_messages_from_committed_members);
        assert!(run.sample_claims_registered);
        assert!(run.sample_claims_pending);
        assert!(run.sample_claims_from_received_answers);
        assert!(run.sample_answers_provider_constructed);
        assert!(!run.verifier_standard_answer_comparison);
        assert!(!run.send_judges_truth);
        assert!(run.sample_pads_committed);
        assert!(!run.sample_pad_hiding);
        assert_eq!(run.group_roots.len(), 3);
        assert_eq!(run.group_positions.len(), 3);
        assert!(run
            .group_roots
            .windows(2)
            .all(|pair| pair[0] != pair[1]));
        assert_eq!(run.group_roots, fixture.recompute_member_roots().unwrap());
        assert_eq!(run.group_rounds, 3);
        assert_eq!(run.bat_chain_rounds, run.group_rounds + 1);
        assert_eq!(run.sample_claims_in_bat, 3);
        assert_eq!(run.received_answers.len(), 3);
        assert_eq!(&run.bat_input_targets[2..5], run.received_answers.as_slice());
        assert_eq!(fixture.recompute_root().unwrap(), run.start.root);
        assert_eq!(run.witness_root, run.start.root);
        assert!(run.pending);
        assert!(!run.fold_decides_truth);
        assert!(run.combined_matches_witness);
        assert!(run.fold_matches_witness);
    }

    #[test]
    fn a_well_formed_wrong_answer_stays_pending_after_fold() {
        let fixture = BatFoldFixture::new(31).unwrap();
        let run = run_bat_fold_with_provider(
            &fixture,
            chain(b"wrong answer"),
            |_order, _seeds, true_answers| {
                let mut answers = true_answers.to_vec();
                answers[0] = answers[0].add(&Fp3::ONE);
                Ok(answers)
            },
        )
        .unwrap();
        assert!(run.pending);
        assert!(!run.combined_matches_witness);
        assert!(!run.fold_matches_witness);
        assert_eq!(run.sample_claims_in_bat, 3);
        assert_eq!(
            run.received_answers[0],
            run.provider_answers[0].add(&Fp3::ONE)
        );
        assert_ne!(run.provider_answers, run.received_answers);
        assert_ne!(run.output.target, run.auxiliary_value);
    }

    #[test]
    fn truncated_or_overlong_answer_lists_are_rejected() {
        let fixture = BatFoldFixture::new(32).unwrap();
        assert!(matches!(
            run_bat_fold_with_provider(
                &fixture,
                chain(b"short answers"),
                |_order, _seeds, answers| Ok(answers[..2].to_vec()),
            ),
            Err(Error::ShapeMismatch)
        ));
        assert!(matches!(
            run_bat_fold_with_provider(
                &fixture,
                chain(b"long answers"),
                |_order, _seeds, answers| {
                    let mut values = answers.to_vec();
                    values.push(Fp3::ONE);
                    Ok(values)
                },
            ),
            Err(Error::ShapeMismatch)
        ));
    }

    #[test]
    fn the_base_final_wiring_passes_but_stays_pending() {
        let fixture = BatFoldFixture::new(33).unwrap();
        let run = run_bat_fold_with_base(&fixture, chain(b"base final")).unwrap();
        assert!(run.base.base_linear);
        assert!(run.base.fold_main);
        assert!(run.base.fold_mask);
        assert!(run.base.fold_fresh);
        assert!(run.base.openings_verified);
        assert!(run.base.member_openings_verified);
        assert!(run.base.fresh_opening_verified);
        assert!(run.base.rejection.is_none());
        assert!(run.base.pending);
        assert!(!run.base.complete_base);
        assert!(run.base.sampled_fold_relation_only);
        assert!(!run.base.complete_fold_binding);
        assert!(run.base.fresh_term_checked_against_fold_root);
        assert!(run.base.projection_only);
        assert!(run.base.conditional_local_experiment);
        assert!(!run.base.cs_connected);
        assert!(!run.base.reference_iopp_path);
        assert!(run.base.chain_handoff);
        assert_eq!(run.base.entry_rounds, run.fold_exit_rounds);
        assert_eq!(run.base.entry_value, run.fold_exit_value);
        assert!(run.base.commit_rounds > run.base.entry_rounds);
        assert!(run.base.rows_rounds > run.base.commit_rounds);
        assert_eq!(run.base.check_rounds, run.base.rows_rounds);
        assert_eq!(run.base.check_value, run.base.rows_value);
        assert_eq!(run.base.base_positions.len(), FIXTURE_BASE_POSITIONS);
        assert_eq!(run.base.member_roots.len(), 3);
        assert_eq!(run.base.fresh_root, run.fold.start.root);
        assert_eq!(run.base.fold_identity, fold_identity_of(&run.fold));
        assert_eq!(run.fold.output.next_stage() , Err(Error::InvalidSessionState));
    }

    fn fold_identity_of(outcome: &BatFoldOutcome) -> Digest {
        // Rebuild the same handoff identity from the outcome's FOLD output and
        // fixture identity fields. This is an independent check of the
        // identity absorbed by the BASE stage, not a second protocol source.
        let anchor = FoldHandoffAnchor {
            shape: outcome.shape,
            group: FIXTURE_GROUP,
            members: outcome
                .layout
                .members
                .iter()
                .map(|member| member.oracle)
                .collect(),
            fresh: outcome.start.id,
            fresh_root: outcome.start.root,
            tree_number: outcome.start.tree_number,
            layout: outcome.layout.clone(),
            fold_gamma: outcome.output.gamma.clone(),
            fold_epsilon: outcome.output.eps,
            fold_sum: outcome.output.sum,
        };
        let handoff = FoldHandoff::from_output(&outcome.output, &anchor).unwrap();
        crate::proof::base_final_session::fold_identity(&handoff)
    }

    enum BaseTamper {
        FoldGamma,
        MainTail,
        MaskTail,
        FreshTail,
        MemberOpening,
        FreshOpening,
        MemberRoot,
        FreshRoot,
        Layout,
        Identity,
        StarRoot,
    }

    fn dot(left: &[Fp3], right: &[Fp3]) -> Result<Fp3> {
        if left.len() != right.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut sum = Fp3::ZERO;
        for (a, b) in left.iter().zip(right) {
            sum = sum.add(&a.mul(b));
        }
        Ok(sum)
    }

    /// Runs the same BASE flow as the fixture entry with one fixed tamper.
    /// Each tamper names the layer that must reject it; this is a local
    /// negative sample, not a general wrong-declaration guarantee.
    fn run_base_tampered(
        fixture: &BatFoldFixture,
        seed: &[u8],
        tamper: BaseTamper,
    ) -> std::result::Result<BatFoldBaseOutcome, BaseFinalRunError> {
        let core = run_bat_fold_core(fixture, chain(seed), |_, _, answers| {
            Ok(answers.to_vec())
        })?;
        let BatFoldCore {
            outcome,
            fold,
            handoff,
            old_main,
            old_mask,
            old_fresh,
            member_plans,
            group_roots,
        } = core;
        let code = outcome.shape.code;
        let mask_code = outcome.shape.mask_code;
        let fold_exit_rounds = fold.chain().rounds();
        let fold_exit_value = *fold.transcript_value();
        let base_chain = fold.take_chain_for_base()?;
        if base_chain.rounds() != fold_exit_rounds || *base_chain.value() != fold_exit_value {
            return Err(BaseFinalRunError::Stage(Error::StatementMismatch));
        }
        let star = fixture_base_star(outcome.tag, code, mask_code);
        let (old_main_tail, old_mask_tail, fresh_tail) = folded_tails(fixture, &handoff)?;
        let true_fresh_claim = handoff.fresh_claim.clone();
        let mut handoff = handoff;
        if matches!(tamper, BaseTamper::FoldGamma) {
            handoff.gamma[0] = handoff.gamma[0].add(&Fp3::ONE);
            handoff.fresh_claim = zkie_engine::pcs::request::fold::fold_mask_weights(
                handoff.k,
                handoff.h,
                usize::try_from(mask_code.ell)
                    .map_err(|_| BaseFinalRunError::Stage(Error::ResourceLimit))?,
                &handoff.gamma,
            )
            .map_err(BaseFinalRunError::Stage)?;
            let delta = dot(&handoff.fresh_claim, &old_fresh)?
                .sub(&dot(&true_fresh_claim, &old_fresh)?);
            handoff.target = handoff.target.add(&delta);
        }
        let rows = handoff.single_rows(old_main, old_main_tail, old_mask, old_mask_tail)?;
        let identity = fold_identity(&handoff);
        let prover = BaseProver::new(
            rows,
            identity,
            old_fresh,
            fresh_tail,
            star.0,
            star.1,
            star.2,
            star.3,
            Fp3::ZERO,
        )?;
        let mut basis = base_basis(fixture, &handoff, &member_plans, &group_roots)?;
        match tamper {
            BaseTamper::MemberRoot => basis.members[0].root = Digest([0xEE; 32]),
            BaseTamper::FreshRoot => basis.fresh_root = Digest([0xEE; 32]),
            BaseTamper::Layout => basis.layout.k += 1,
            _ => {}
        }
        let mut session = BaseFinalSession::new(base_chain, handoff, basis)?;
        let mut messages = prover.messages()?;
        match tamper {
            BaseTamper::Identity => messages.fold_identity.0[0] ^= 1,
            BaseTamper::StarRoot => messages.star.root.0[0] ^= 1,
            _ => {}
        }
        session.receive_commitments(messages)?;
        let lambda = session.draw_lambda()?;
        let mut prepared = prover.base_rows(lambda)?;
        match tamper {
            BaseTamper::MainTail => {
                prepared.base.main_tail_prime[0] =
                    prepared.base.main_tail_prime[0].add(&Fp3::ONE)
            }
            BaseTamper::MaskTail => {
                prepared.base.mask_tail_prime[0] =
                    prepared.base.mask_tail_prime[0].add(&Fp3::ONE)
            }
            BaseTamper::FreshTail => {
                prepared.fresh_tail[0] = prepared.fresh_tail[0].add(&Fp3::ONE)
            }
            _ => {}
        }
        session.receive_base_rows(prepared)?;
        let positions = session.draw_base_positions(FIXTURE_BASE_POSITIONS)?;
        let unique = unique_sorted(&positions);
        let mut member_openings =
            fixture.open_members(&vec![unique.clone(); member_plans.len()])?;
        let mask_positions: Vec<u64> = positions
            .iter()
            .map(|position| *position % mask_code.n())
            .collect();
        let mut fresh_opening = fixture.open_fresh(&unique_sorted(&mask_positions))?;
        match tamper {
            BaseTamper::MemberOpening => {
                member_openings[0].columns[0].symbols[0] =
                    member_openings[0].columns[0].symbols[0] + Fp::from_u64(1)
            }
            BaseTamper::FreshOpening => {
                fresh_opening.columns[0].symbols[0] =
                    fresh_opening.columns[0].symbols[0] + Fp::from_u64(1)
            }
            _ => {}
        }
        let star_openings = prover.openings(&positions)?;
        let openings = BaseOpenings {
            star: star_openings.star,
            star_mask: star_openings.star_mask,
            members: member_openings,
            fresh: fresh_opening,
        };
        session.verify_openings(&openings)?;
        let base = match session.finish()? {
            BaseFinalCheck::Passed(base) => base,
            BaseFinalCheck::Rejected(rejection) => {
                return Err(BaseFinalRunError::Rejected(rejection));
            }
        };
        Ok(BatFoldBaseOutcome {
            fold: outcome,
            base,
            fold_exit_rounds,
            fold_exit_value,
        })
    }

    fn rejection_layer(
        result: std::result::Result<BatFoldBaseOutcome, BaseFinalRunError>,
    ) -> BaseRejection {
        match result {
            Err(BaseFinalRunError::Rejected(rejection)) => rejection.layer,
            Err(BaseFinalRunError::Stage(error)) => {
                panic!("expected a typed rejection, got a stage error: {error:?}")
            }
            Ok(_) => panic!("expected a typed rejection, got a passing outcome"),
        }
    }

    #[test]
    fn replaced_folded_rows_are_rejected_at_the_named_layer() {
        let fixture = BatFoldFixture::new(40).unwrap();
        assert_eq!(
            rejection_layer(run_base_tampered(
                &fixture,
                b"tamper main tail",
                BaseTamper::MainTail
            )),
            BaseRejection::FoldMain
        );
        assert_eq!(
            rejection_layer(run_base_tampered(
                &fixture,
                b"tamper mask tail",
                BaseTamper::MaskTail
            )),
            BaseRejection::FoldMask
        );
        assert_eq!(
            rejection_layer(run_base_tampered(
                &fixture,
                b"tamper fresh tail",
                BaseTamper::FreshTail
            )),
            BaseRejection::FoldFresh
        );
        assert_eq!(
            rejection_layer(run_base_tampered(
                &fixture,
                b"tamper fold challenge",
                BaseTamper::FoldGamma
            )),
            BaseRejection::FoldMain
        );
    }

    #[test]
    fn replaced_identity_layout_and_commitments_fail_closed() {
        let fixture = BatFoldFixture::new(41).unwrap();
        for tamper in [
            BaseTamper::Identity,
            BaseTamper::StarRoot,
            BaseTamper::MemberRoot,
            BaseTamper::FreshRoot,
            BaseTamper::Layout,
            BaseTamper::MemberOpening,
            BaseTamper::FreshOpening,
        ] {
            let label = match tamper {
                BaseTamper::Identity => "identity",
                BaseTamper::StarRoot => "star root",
                BaseTamper::MemberRoot => "member root",
                BaseTamper::FreshRoot => "fresh root",
                BaseTamper::Layout => "layout",
                BaseTamper::MemberOpening => "member opening",
                BaseTamper::FreshOpening => "fresh opening",
                _ => "unexpected",
            };
            assert!(
                matches!(
                    run_base_tampered(&fixture, label.as_bytes(), tamper),
                    Err(BaseFinalRunError::Stage(_))
                ),
                "the replaced {label} was not rejected"
            );
        }
    }

    #[test]
    fn a_fixed_base_target_shift_is_rejected_at_the_target_layer() {
        let fixture = BatFoldFixture::new(34).unwrap();
        let rejection = match run_bat_fold_with_base_and_target_shift(
            &fixture,
            chain(b"base target shift"),
            Fp3::ONE,
        ) {
            Err(BaseFinalRunError::Rejected(rejection)) => rejection,
            Err(BaseFinalRunError::Stage(error)) => panic!("stage error: {error:?}"),
            Ok(_) => panic!("the fixed target shift was not rejected"),
        };
        assert_eq!(rejection.layer, BaseRejection::TargetLinear);
        assert!(!rejection.base_linear);
        assert_eq!(rejection.base_positions.len(), FIXTURE_BASE_POSITIONS);
    }

    #[test]
    fn a_wrong_upstream_answer_is_rejected_at_base() {
        let fixture = BatFoldFixture::new(35).unwrap();
        let rejection = match run_bat_fold_with_base_and_provider(
            &fixture,
            chain(b"base wrong upstream"),
            |_order, _seeds, true_answers| {
                let mut answers = true_answers.to_vec();
                answers[0] = answers[0].add(&Fp3::ONE);
                Ok(answers)
            },
        ) {
            Err(BaseFinalRunError::Rejected(rejection)) => rejection,
            Err(BaseFinalRunError::Stage(error)) => panic!("stage error: {error:?}"),
            Ok(_) => panic!("the wrong upstream answer was not rejected"),
        };
        assert_eq!(rejection.layer, BaseRejection::TargetLinear);
        assert!(!rejection.base_linear);
        assert_eq!(rejection.base_positions.len(), FIXTURE_BASE_POSITIONS);
    }

    #[test]
    fn a_wrong_declaration_remains_pending_after_fold() {
        let fixture = BatFoldFixture::new_with_target_shift(29, Fp3::from_u64(5)).unwrap();
        let run = run_bat_fold(&fixture, chain(b"pending declaration")).unwrap();
        assert!(run.target_shifted);
        assert!(!run.combined_matches_witness);
        assert!(!run.fold_matches_witness);
        assert!(run.pending);
        assert_eq!(run.gamma.len(), fixture.shape.k as usize);
        assert_ne!(run.output.target, run.auxiliary_value);
        assert!(run.output.next_stage().is_err());
    }

    #[test]
    fn the_fold_cs_base_chain_passes_but_stays_pending() {
        let fixture = BatFoldFixture::new(50).unwrap();
        let run = run_bat_fold_cs(&fixture, chain(b"fold cs base")).unwrap();
        assert!(run.cs.old_columns_authenticated);
        assert!(run.cs.new_group_registered);
        assert!(run.cs.cs_claims_pending);
        assert!(!run.cs.cs_proves_new_group);
        assert!(run.cs.sample_claim_registered);
        assert!(run.cs.sample_claim_pending);
        assert!(run.cs.sample_beta_nonzero);
        assert!(!run.cs.send_standard_wiring);
        assert!(run.cs.fresh_correspondence_registered);
        assert!(run.cs.fresh_correspondence_sampled);
        assert!(run.cs.sampled_fold_relation_only);
        assert!(!run.cs.complete_fold_binding);
        assert!(run.cs.base_linear);
        assert!(run.cs.symbol_checks);
        assert!(run.cs.openings_verified);
        assert!(run.cs.pending);
        assert!(!run.cs.complete_cs);
        assert!(run.cs.cs_connected_in_fixture);
        assert!(!run.cs.reference_iopp_path);
        assert!(run.cs.conditional_local_experiment);
        assert_eq!(run.cs.declaration_rounds, run.fold.final_rounds + 1);
        assert_eq!(run.cs.positions_rounds, run.cs.answer_rounds + 1);
        assert_eq!(run.cs.base_commit_rounds, run.cs.positions_rounds + 1);
        assert_eq!(run.cs.base_rows_rounds, run.cs.base_commit_rounds + 2);
        assert_eq!(run.cs.final_rounds, run.cs.base_rows_rounds);
        assert_eq!(
            run.cs.registered_claims,
            2 + 3 * run.cs.cs_positions.len()
        );
        println!(
            "cs root={:?} fold_identity={:?} theta={:?} lambda={:?} target_star={:?} cs_positions={:?} base_positions={:?} claims={} sample_answer={:?}",
            run.cs.switched_root,
            run.cs.fold_identity,
            run.cs.theta,
            run.cs.lambda,
            run.cs.target_star,
            run.cs.cs_positions,
            run.cs.base_positions,
            run.cs.registered_claims,
            run.cs.sample_answer
        );
    }

    enum CsTamper {
        DeclarationRoot,
        MemberOpening,
        SwitchedOpening,
        Theta,
        WrongAnswer,
        TamperedFresh,
    }

    /// Runs the same flow with one fixed tamper. The returned flag says whether
    /// the registration stage completed; the error may come from registration
    /// or from the later BASE check.
    fn run_cs_tampered(
        fixture: &BatFoldFixture,
        seed: &[u8],
        tamper: CsTamper,
    ) -> (bool, Result<CsOutcome>) {
        let core = match run_bat_fold_core(fixture, chain(seed), |_, _, answers| {
            Ok(answers.to_vec())
        }) {
            Ok(core) => core,
            Err(error) => return (false, Err(error)),
        };
        let BatFoldCore {
            outcome,
            fold,
            handoff,
            old_main,
            old_mask,
            old_fresh,
            member_plans,
            group_roots,
        } = core;
        let code = outcome.shape.code;
        let mask_code = outcome.shape.mask_code;
        let base_chain = match fold.take_chain_for_base() {
            Ok(chain) => chain,
            Err(error) => return (false, Err(error)),
        };
        let (main_tail, mask_tail, fresh_tail) = match folded_tails(fixture, &handoff) {
            Ok(tails) => tails,
            Err(error) => return (false, Err(error)),
        };
        let rows = match handoff.single_rows(old_main, main_tail, old_mask, mask_tail) {
            Ok(rows) => rows,
            Err(error) => return (false, Err(error)),
        };
        let switch_code = match build_code_plan(code.log_n + 1, FIXTURE_SWITCH_ELL, 1) {
            Ok(code) => code,
            Err(error) => return (false, Err(error)),
        };
        let switch_mask_code = match build_code_plan(mask_code.log_n + 1, FIXTURE_SWITCH_ELL, 1) {
            Ok(code) => code,
            Err(error) => return (false, Err(error)),
        };
        let star = fixture_cs_star(outcome.tag, switch_code, switch_mask_code);
        let prover = if matches!(tamper, CsTamper::TamperedFresh) {
            CsProver::new_with_fresh_tamper(
                &rows,
                &handoff,
                switch_code,
                switch_mask_code,
                old_fresh,
                fresh_tail,
                outcome.tag,
                Fp3::ONE,
                star.0,
                star.1,
                star.2,
                star.3,
            )
        } else {
            CsProver::new(
                &rows,
                &handoff,
                switch_code,
                switch_mask_code,
                old_fresh,
                fresh_tail,
                outcome.tag,
                star.0,
                star.1,
                star.2,
                star.3,
            )
        };
        let prover = match prover {
            Ok(prover) => prover,
            Err(error) => return (false, Err(error)),
        };
        let plan = prover.plan();
        let identity = fold_identity(&handoff);
        let mut declarations = match prover.declarations(identity) {
            Ok(declarations) => declarations,
            Err(error) => return (false, Err(error)),
        };
        if matches!(tamper, CsTamper::DeclarationRoot) {
            declarations.switched.root.0[0] ^= 1;
        }
        let basis = match base_basis(fixture, &handoff, &member_plans, &group_roots) {
            Ok(basis) => basis,
            Err(error) => return (false, Err(error)),
        };
        let mut session = match CsSession::new(
            base_chain,
            handoff,
            basis,
            plan,
            switch_code,
            switch_mask_code,
        ) {
            Ok(session) => session,
            Err(error) => return (false, Err(error)),
        };
        if let Err(error) = session.receive_declarations(declarations) {
            return (false, Err(error));
        }
        let seed = match session.draw_sample_seed() {
            Ok(seed) => seed,
            Err(error) => return (false, Err(error)),
        };
        let true_answer = match prover.sample_answer(std::slice::from_ref(&seed)) {
            Ok(answer) => answer,
            Err(error) => return (false, Err(error)),
        };
        let received_answer = if matches!(tamper, CsTamper::WrongAnswer) {
            true_answer.add(&Fp3::ONE)
        } else {
            true_answer
        };
        if let Err(error) = session.receive_answer(received_answer) {
            return (false, Err(error));
        }
        let (positions, theta) = match session.draw_positions_theta(FIXTURE_CS_POSITIONS) {
            Ok(values) => values,
            Err(error) => return (false, Err(error)),
        };
        let unique = unique_sorted(&positions);
        let mut member_openings = match fixture.open_members(
            &vec![unique.clone(); member_plans.len()],
        ) {
            Ok(openings) => openings,
            Err(error) => return (false, Err(error)),
        };
        if matches!(tamper, CsTamper::MemberOpening) {
            member_openings[0].columns[0].symbols[0] =
                member_openings[0].columns[0].symbols[0] + Fp::from_u64(1);
        }
        let mask_positions: Vec<u64> = positions
            .iter()
            .map(|position| *position % mask_code.n())
            .collect();
        let fresh_opening = match fixture.open_fresh(&unique_sorted(&mask_positions)) {
            Ok(opening) => opening,
            Err(error) => return (false, Err(error)),
        };
        let claims = match session.authenticate_and_register(&member_openings, &fresh_opening) {
            Ok(claims) => claims.clone(),
            Err(error) => return (false, Err(error)),
        };
        let prover_theta = if matches!(tamper, CsTamper::Theta) {
            theta.add(&Fp3::ONE)
        } else {
            theta
        };
        let prover_claims = match prover.claims(&positions, prover_theta, std::slice::from_ref(&seed)) {
            Ok(claims) => claims,
            Err(error) => return (true, Err(error)),
        };
        if !matches!(tamper, CsTamper::Theta | CsTamper::WrongAnswer)
            && claims != prover_claims
        {
            return (true, Err(Error::StatementMismatch));
        }
        let _ = &claims;
        let auth_claims = prover_claims;
        let messages = match prover.base_messages(&auth_claims) {
            Ok(messages) => messages,
            Err(error) => return (true, Err(error)),
        };
        if let Err(error) = session.receive_base_commitments(messages) {
            return (true, Err(error));
        }
        let lambda = match session.draw_lambda() {
            Ok(lambda) => lambda,
            Err(error) => return (true, Err(error)),
        };
        let base_rows = match prover.base_rows(&auth_claims, lambda) {
            Ok(rows) => rows,
            Err(error) => return (true, Err(error)),
        };
        if let Err(error) = session.receive_base_rows(base_rows) {
            return (true, Err(error));
        }
        let base_positions = match session.draw_base_positions(FIXTURE_BASE_POSITIONS) {
            Ok(positions) => positions,
            Err(error) => return (true, Err(error)),
        };
        let (star_openings, star_mask_openings) = match prover.star_openings(&base_positions) {
            Ok(openings) => openings,
            Err(error) => return (true, Err(error)),
        };
        let mut switched = match prover.openings(&unique_sorted(&base_positions)) {
            Ok(opening) => opening,
            Err(error) => return (true, Err(error)),
        };
        if matches!(tamper, CsTamper::SwitchedOpening) {
            switched.columns[0].symbols[0] =
                switched.columns[0].symbols[0] + Fp::from_u64(1);
        }
        if let Err(error) = session.verify_base_openings(&CsBaseOpenings {
            star: star_openings,
            star_mask: star_mask_openings,
            switched,
        }) {
            return (true, Err(error));
        }
        (true, session.finish())
    }

    #[test]
    fn replaced_cs_inputs_fail_closed() {
        let fixture = BatFoldFixture::new(51).unwrap();
        for (seed, tamper) in [
            (b"cs declaration".as_slice(), CsTamper::DeclarationRoot),
            (b"cs member opening".as_slice(), CsTamper::MemberOpening),
            (b"cs switched opening".as_slice(), CsTamper::SwitchedOpening),
            (b"cs theta".as_slice(), CsTamper::Theta),
        ] {
            let (_, result) = run_cs_tampered(&fixture, seed, tamper);
            assert!(result.is_err(), "the replaced CS input was not rejected");
        }
    }

    #[test]
    fn altered_send_answer_and_fresh_block_are_rejected_after_registration() {
        let fixture = BatFoldFixture::new(54).unwrap();
        for (seed, tamper) in [
            (b"cs wrong answer".as_slice(), CsTamper::WrongAnswer),
            (b"cs fresh block".as_slice(), CsTamper::TamperedFresh),
        ] {
            let (registered, result) = run_cs_tampered(&fixture, seed, tamper);
            assert!(
                registered,
                "the registration stage should complete before the later check"
            );
            assert!(
                result.is_err(),
                "the altered relation was not rejected at the later stage"
            );
        }
    }

    #[test]
    fn unsupported_switch_shapes_fail_closed() {
        let fixture = BatFoldFixture::new(52).unwrap();
        let core = run_bat_fold_core(&fixture, chain(b"cs shapes"), |_, _, answers| {
            Ok(answers.to_vec())
        })
        .unwrap();
        let BatFoldCore {
            outcome,
            fold,
            handoff,
            old_main,
            old_mask,
            old_fresh,
            member_plans,
            group_roots,
        } = core;
        let _ = (&outcome, old_main, old_mask, old_fresh);
        let code = fixture.shape.code;
        let mask_code = fixture.shape.mask_code;
        let switch_code = build_code_plan(code.log_n + 1, FIXTURE_SWITCH_ELL, 1).unwrap();
        let switch_mask_code = build_code_plan(mask_code.log_n + 1, FIXTURE_SWITCH_ELL, 1).unwrap();
        let plan = switch_tree_plan(outcome.tag, switch_code, switch_mask_code).unwrap();
        let basis = base_basis(&fixture, &handoff, &member_plans, &group_roots).unwrap();
        let mut order_session = CsSession::new(
            fold_chain(&fixture, b"cs stage order"),
            handoff.clone(),
            basis.clone(),
            plan.clone(),
            switch_code,
            switch_mask_code,
        )
        .unwrap();
        assert!(order_session.draw_sample_seed().is_err());
        assert!(order_session.receive_answer(Fp3::ONE).is_err());
        assert!(order_session.draw_positions_theta(1).is_err());
        let unsupported = build_code_plan(code.log_n, code.ell, 1).unwrap();
        let chain = fold.take_chain_for_base().unwrap();
        assert!(matches!(
            CsSession::new(
                chain,
                handoff.clone(),
                basis.clone(),
                plan.clone(),
                unsupported,
                switch_mask_code
            ),
            Err(Error::UnsupportedPrimitive)
        ));
        let mut kept = basis;
        kept.layout.members[0].kept = true;
        assert!(matches!(
            CsSession::new(
                fold_chain(&fixture, b"cs kept"),
                handoff,
                kept,
                plan,
                switch_code,
                switch_mask_code
            ),
            Err(Error::InvalidConfig)
        ));
    }

    fn fold_chain(fixture: &BatFoldFixture, seed: &[u8]) -> Chain {
        run_bat_fold_core(fixture, chain(seed), |_, _, answers| Ok(answers.to_vec()))
            .unwrap()
            .fold
            .take_chain_for_base()
            .unwrap()
    }

}
