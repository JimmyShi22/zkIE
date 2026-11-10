//! Isolated fixture handoff from a BAT group row to the FOLD stage.
//!
//! This module builds one deterministic fixture group that BAT and FOLD use
//! natively: the BAT group's base and mask codes are the FOLD shape's codes, the
//! member mask width is already a power of two, and the BAT layout is the FOLD
//! fixture layout. No field is copied from a width adapter and no code or tail
//! parameter is transformed. The BAT combined row is converted by the existing
//! BAT-to-FOLD input-row conversion and enters FOLD with the same main
//! coefficients, mask coefficients and target.
//!
//! The fixture has one source for the positive case. The group message tables
//! are fixed before the run and each BAT claim target is the inner product of
//! that claim's coefficients (mapped through BAT's own combination) with the
//! same group message tables. The combined target is therefore the true inner
//! product of the combined row against the fixture witness. Those group tables
//! are the prover's FOLD messages; they are plaintext fixture data and are not
//! bound to an asset root. The group also carries fixture sample-class rows
//! required by the BAT member registration; their targets are ordinary inner
//! products against the same tables, not authenticated send receipts.
//!
//! The fresh-mask witness is built from a deterministic numbered test source.
//! Its coefficients, encoding tail and mask rows produce the committed mask
//! tree root and the hiding sum carried by the FOLD start message; the fixture
//! independently re-commits the stored rows and tails and checks the same root.
//! The verifier's runtime registry receives that root before any challenge.
//! The values are public and reproducible, provide no hiding, and are only a
//! test fixture. The root is registered, not authenticated by openings.
//!
//! Order and ownership: the group is frozen, the BAT session draws `x` and is
//! consumed by `into_chain`, the returned chain is moved into the FOLD-only
//! session entry, the fixed row is bound, the start message fixes the fresh
//! root and sum before the epsilon challenge, each round message is absorbed
//! before its gamma, and the final output stays pending. The handoff records
//! the chain round count and value on both sides; it does not compare pointers.
//! A chain that was not advanced by a prior stage is rejected.
//!
//! The verifier advances only on the bound row and the checked start/round
//! messages. FOLD does not decide truth: a wrong BAT target still completes the
//! rounds and remains pending, and only a later relation or proximity check
//! could reject it. Wrong shape, code, oracle, root and order violations fail
//! closed.
//!
//! Unclosed: real SEND/G_u, authenticated asset roots, fresh-mask openings,
//! hiding, folded tails, a proximity/relation connection, single-shot
//! semantics and production wiring. The plan's reader and binding are a valid
//! but inert scaffold required by the existing plan type; the FOLD-only entry
//! never reads them and they are not a witness or an asset binding.
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
    fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_mask_sum, fold_start_prove,
    validate_fold_mask_shape, FoldGroupLayout, FoldInputRow, FoldMask, FoldOutput,
    FoldProverSession, FoldShape, FoldStartMessage, MaskCoins, MaskOracle, OracleMetadata,
    RuntimeOracles, VerifiedGroup, VerifiedGroups, VerifiedStatement,
};
use zkie_engine::pcs::request::foldeq::{
    reader_table_identity, FoldeqBinding, FoldeqEdgeBinding, FoldeqPlan, FoldeqSession, GammaPath,
};
use zkie_engine::pcs::request::params::{RequestParams, LEAF_SALT_MIN_BYTES};
use zkie_engine::pcs::request::read::ReadLedger;
use zkie_engine::pcs::request::runtime_tree::{MaskTree, MaskTreePlan};
use zkie_engine::pcs::request::table::{
    commit_table, IndexReader, IndexShape, TableDistances,
};
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, TEST_COINS};

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
        let digest = hash_labeled(
            tag_word(TEST_COINS, 0),
            number as u64,
            Some(&self.seed),
            &[],
        )?;
        let mut coefficients = [Fp::ZERO; E_DEGREE];
        for (index, coefficient) in coefficients.iter_mut().enumerate() {
            let bytes = &digest.0[index * 8..(index + 1) * 8];
            let word = u64::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidEncoding)?);
            *coefficient = Fp::from_u64(word % MODULUS);
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

fn message_table(len: usize, base: u64, tag: u64, oracle: u64) -> Vec<Fp3> {
    (0..len)
        .map(|index| {
            Fp3::from_u64(
                base.wrapping_add(tag.wrapping_mul(0x100))
                    .wrapping_add(oracle.wrapping_mul(0x40))
                    .wrapping_add(index as u64),
            )
        })
        .collect()
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

/// Builds the group message table from the per-member fixture messages. The
/// index mapping is the layout's own: member entries are column-major within
/// the member table and placed at the member's first group row, exactly where
/// BAT carries the coefficients.
fn build_group_table(
    layout: &FoldGroupLayout,
    bat_layout: &BatGroupLayout,
    part: BatPart,
    ell: u64,
    tag: u64,
    base: u64,
) -> Result<Vec<Fp3>> {
    let group_bits = bat_layout.group_row_bits(part);
    let group_len = bat_layout.group_len(part)?;
    let mut values = vec![Fp3::ZERO; group_len];
    if layout.members.len() != bat_layout.members.len() {
        return Err(Error::StatementMismatch);
    }
    for (fold_member, bat_member) in layout.members.iter().zip(&bat_layout.members) {
        if fold_member.oracle != bat_member.oracle || fold_member.kept {
            return Err(Error::UnsupportedPrimitive);
        }
        let rows = match part {
            BatPart::Main => fold_member.main_rows,
            BatPart::Mask => fold_member.mask_rows,
        };
        if rows == 0 {
            continue;
        }
        let first = match part {
            BatPart::Main => bat_member.main_first,
            BatPart::Mask => bat_member.mask_first,
        }
        .ok_or(Error::StatementMismatch)?;
        let bits = row_bits(rows);
        let row_mask = (1u64 << bits) - 1;
        let table = message_table(table_len(rows, ell)?, base, tag, fold_member.oracle);
        for (index, value) in table.iter().enumerate() {
            let row = index as u64 & row_mask;
            let place = (index as u64) >> bits;
            let at = first
                .checked_add(row)
                .and_then(|value| value.checked_add(place.checked_shl(group_bits)?))
                .ok_or(Error::ResourceLimit)?;
            let slot = values.get_mut(at as usize).ok_or(Error::InvalidIndex)?;
            *slot = *value;
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

/// One deterministic fixture group shared natively by BAT and FOLD.
pub struct BatFoldFixture {
    tag: u64,
    target_shift: Fp3,
    shape: FoldShape,
    layout: FoldGroupLayout,
    bat_statement: BatStatement,
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
        let mut bat_statement = BatStatement::new();
        for entry in &shapes {
            bat_statement.add_oracle(*entry)?;
        }
        let layout = probe.to_fold_layout_checked(FIXTURE_GROUP, &shape)?;
        if layout.group != FIXTURE_GROUP
            || layout.members.iter().any(|member| member.kept)
            || bat_layout.k != shape.k
            || bat_layout.h != shape.h
        {
            return Err(Error::UnsupportedPrimitive);
        }
        let main_len = table_len(FIXTURE_MAIN_ROWS, shape.code.ell)?;
        let mask_len = table_len(FIXTURE_MASK_ROWS, shape.mask_code.ell)?;
        let group_main = build_group_table(
            &layout,
            &bat_layout,
            BatPart::Main,
            shape.code.ell,
            tag,
            0x1000,
        )?;
        let group_mask = build_group_table(
            &layout,
            &bat_layout,
            BatPart::Mask,
            shape.mask_code.ell,
            tag,
            0x2000,
        )?;

        // Fixture sample-class rows satisfy BAT's sampled-member registration.
        // Their targets are the true inner products of their unit coefficients
        // against the same group witness; they are not authenticated send
        // receipts.
        for oracle in 0..FIXTURE_MEMBERS {
            let mut sample = BatRow::new(BatRowKind::Sample);
            sample.add_unit(
                oracle,
                BatPart::Main,
                oracle,
                Fp3::from_u64(0x600 + tag % 71 + oracle),
            );
            sample.add_unit(
                oracle,
                BatPart::Mask,
                oracle + 1,
                Fp3::from_u64(0x700 + tag % 73 + oracle),
            );
            sample.target = claim_target(&probe, &bat_layout, &sample, &group_main, &group_mask)?;
            bat_statement.sample(oracle, vec![sample])?;
        }
        let group = bat_statement.new_group(shape.code, shape.mask_code)?;
        for oracle in 0..FIXTURE_MEMBERS {
            bat_statement.join(group, oracle, false)?;
        }
        if bat_statement.layout(group)? != bat_layout {
            return Err(Error::StatementMismatch);
        }

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
            &bat_statement,
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
            &bat_statement,
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
            &bat_statement,
            &bat_layout,
            &third,
            &group_main,
            &group_mask,
        )?;

        for row in [first, second, third] {
            bat_statement.add_row(group, row)?;
        }

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
                group,
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
        let oracle = MaskOracle::prover_view(FIXTURE_FRESH, group, &notice);
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
            bat_statement,
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

    pub fn bat_statement(&self) -> &BatStatement {
        &self.bat_statement
    }

    pub fn group(&self) -> u64 {
        FIXTURE_GROUP
    }

    pub fn planned_rows(&self) -> u64 {
        FIXTURE_PLANNED_ROWS
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
    let table = commit_table(shape, configuration, &messages, &tails, &salts)?;
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

/// The result of one fixture BAT-to-FOLD run. The FOLD output is a pending
/// candidate, not an acceptance.
pub struct BatFoldOutcome {
    pub tag: u64,
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub bat_layout: BatGroupLayout,
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

/// Runs the isolated fixture: BAT freezes the shared group, draws `x`, hands
/// its advanced chain to the FOLD-only entry, the fixed combined row enters
/// FOLD unchanged, and the rounds finish with a pending output.
pub fn run_bat_fold(fixture: &BatFoldFixture, chain: Chain) -> Result<BatFoldOutcome> {
    if fixture.recompute_root()? != fixture.start.root {
        return Err(Error::StatementMismatch);
    }
    if fold_mask_sum(fixture.shape.k, fixture.mask.coefficients())? != fixture.start.sum {
        return Err(Error::StatementMismatch);
    }

    let mut bat = BatSession::new(
        fixture.bat_statement.clone(),
        chain,
        fixture.planned_rows(),
    )?;
    let input = bat.begin_run(fixture.group())?;
    let bat_layout = input.layout.clone();
    let bat_order = input.rows.iter().map(|row| row.kind).collect::<Vec<_>>();
    let bat_input_targets = input.rows.iter().map(|row| row.target).collect::<Vec<_>>();
    let combined = bat.finish_run(input.clone())?;
    let x = bat.last_x().ok_or(Error::InvalidSessionState)?;
    let fold_row = bat
        .statement()
        .to_fold_input_row(&combined, &bat_layout)?;
    if fold_row.coefficients != combined.main
        || fold_row.mask_coefficients != combined.mask
        || fold_row.target != combined.target
    {
        return Err(Error::StatementMismatch);
    }
    let bat_chain = bat.into_chain()?;
    let bat_chain_rounds = bat_chain.rounds();
    let bat_chain_value = *bat_chain.value();

    let combined_value = dot(&combined.main, &fixture.group_main)?
        .add(&dot(&combined.mask, &fixture.group_mask)?);
    let combined_matches_witness = combined.target == combined_value;

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
        &fold_row,
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
    Ok(BatFoldOutcome {
        tag: fixture.tag,
        shape: fixture.shape,
        layout: fixture.layout.clone(),
        bat_layout,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

    fn chain(seed: &[u8]) -> Chain {
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest::ZERO, seed, &[], None).unwrap()
    }

    fn bat_stage(fixture: &BatFoldFixture, seed: &[u8]) -> Result<(Chain, FoldInputRow)> {
        let mut bat = BatSession::new(
            fixture.bat_statement.clone(),
            chain(seed),
            fixture.planned_rows(),
        )?;
        let input = bat.begin_run(fixture.group())?;
        let combined = bat.finish_run(input.clone())?;
        let row = bat.statement().to_fold_input_row(&combined, &input.layout)?;
        Ok((bat.into_chain()?, row))
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
        assert_eq!(
            fixture
                .bat_statement
                .to_fold_layout_checked(FIXTURE_GROUP, &fixture.shape)
                .unwrap(),
            fixture.layout
        );
        let bat_layout = fixture.bat_statement.layout(FIXTURE_GROUP).unwrap();
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
        assert!(!run.real_assets_bound);
        assert!(!run.mask_root_authenticated);
        assert!(!run.production_wired);
        assert_eq!(run.gamma.len(), fixture.shape.k as usize);
        assert_eq!(run.fold_entry_rounds, run.bat_chain_rounds);
        assert_eq!(run.fold_entry_value, run.bat_chain_value);
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
            fixture.bat_statement.clone(),
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
        changed.rows[0].target = Fp3::from_u64(9999);
        assert!(matches!(
            bat.finish_run(changed),
            Err(Error::StatementMismatch)
        ));
        assert!(bat.finish_run(input).is_err());

        let mut bat = BatSession::new(
            fixture.bat_statement.clone(),
            chain(b"frozen layout"),
            fixture.planned_rows(),
        )
        .unwrap();
        let input = bat.begin_run(fixture.group()).unwrap();
        let mut changed_layout = input.clone();
        changed_layout.layout.k += 1;
        assert!(bat.finish_run(changed_layout).is_err());

        let mut bat = BatSession::new(
            fixture.bat_statement.clone(),
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
}
