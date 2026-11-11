//! Main-row/message demonstration for the isolated GroupRow source and FOLD
//! component.
//!
//! This module builds a candidate main-row witness from the complete
//! authenticated asset opening, compares the main rows against an independent
//! verifier-side reconstruction, and runs the isolated FOLD component over the
//! same sources. Folding-tail derivation (`fold_low`/`fold_gather_mask`
//! equivalent) is not implemented: the stored tail fields are fixture values,
//! they do not participate in FOLD folding, and the outcome must not be read as
//! a completed witness fold. The candidate witness derivation conditions are
//! incomplete for that reason; this does not change the T-002 final
//! acceptance. It does not run G_u or BAT, does not authenticate the fresh
//! mask as a production commitment, and does not accept a proof. The production
//! verify entry is unchanged.
//!
//! Source table for the candidate GroupRow:
//!
//! | GroupRow component | source | shape |
//! | --- | --- | --- |
//! | main messages | authenticated asset opening rows, fixed slice/encoding | `2^k * ell` |
//! | main random tails | fixed zero tail fixture; **not folded, not checked** | `2^k * r` |
//! | old-mask messages | fixed old-mask fixture | `2^h * mask_ell` |
//! | old-mask random tail | fixed old-mask tail fixture; **not folded, not checked** | `2^h * mask_r` |
//! | fresh-mask rows | derived from the fixed fresh-mask coefficients | `2^h * mask_ell` |
//! | fresh-mask random tail | fixed fresh-mask tail fixture; **not folded, not checked** | `2^h * mask_r` |
//! | main claim coefficients | fixed plan values | `2^k * ell` |
//! | old-mask claim coefficients | fixed plan values | `2^h * mask_ell` |
//! | kept forms | none (no-kept subset) | 0 |
//! | target | main and old-mask claim dots | scalar |
//!
//! The fixed plan order is: fixed fresh commitment and sum, then epsilon, then
//! one fixed round message followed by that round's gamma. Main, old-mask,
//! fresh-mask and random tails are retained separately and are never merged in
//! the outcome. The challenges are used according to each component's role;
//! not every witness entry depends on epsilon.
//!
//! Two comparisons are distinguished:
//!
//! - the main rows are compared in full against the asset opening under the
//!   anchor premise and the verifier's fixed shape/configuration. This is the
//!   comparison path;
//! - the fresh fixture is only checked for local consistency with a root
//!   rebuilt from the same fixture data. This is not trusted fresh
//!   authentication.
//!
//! Shape and configuration come from the verifier's fixed `GroupRowPlan`, not
//! from the source's self-report; the source construction checks the table
//! against that plan. The `Arc::ptr_eq` assertion only records that the demo
//! consumes the same object as the comparison; it is not independent
//! authentication. The complete opening is resident; this is intentionally
//! expensive and is not a production authentication or memory path.
//!
//! G_u/BAT and the real GroupRow/fresh authentication remain unclosed. The 3b
//! projection handoff continues to reject; passing this comparison does not
//! authenticate the fresh term or a complete FOLD statement.

use std::sync::Arc;

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::build_code_plan;
use zkie_engine::pcs::request::field::{Fp, Fp3};
use zkie_engine::pcs::request::fold::{
    fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_mask_shape_fault, fold_mask_table,
    fold_output, FoldGroupLayout, FoldInputRow, FoldMask, FoldMemberLayout,
    FoldOutput, FoldProverSession, FoldRoundTranscript, FoldRoundVerifier, FoldShape,
    FoldStartMessage, MaskCoins,
};
use zkie_engine::pcs::request::runtime_tree::{MaskTree, MaskTreePlan};
use zkie_engine::pcs::request::table::{
    commit_table_streaming, IndexShape, VecTableSource,
};
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::proof::column_session::{
    build_authenticated_encoding_table_with_configuration, fp_from_signed_word, ColumnBindingInput,
};
use crate::proof::fold_handoff::{FoldHandoff, FoldHandoffAnchor};
use crate::proof::model_commitment::{verify_model_anchor, AssetOpening, ModelCommitmentPublic};
use crate::proof::session_baseline::TrustedModel;

const GROUP: u64 = 5;
const FRESH_ID: u64 = 99;
const TREE_NUMBER: u64 = 7;
const FOLD_K: u32 = 2;
const FOLD_H: u32 = 0;
const MASK_ELL: u64 = 6;
const MASK_R: u64 = 1;
const TREE_RUN: Digest = Digest([0x33; 32]);
const TREE_CONFIGURATION: Digest = Digest([0x34; 32]);
const MAIN_ORACLE: u64 = 11;
const STATIC_ORACLE: u64 = 12;

/// Fixed numbered source for the test only. It is not a random mask and is not
/// evidence of hiding or a soundness bound; production randomness is unclosed.
struct FixedCoins {
    next: u32,
}

impl Default for FixedCoins {
    fn default() -> Self {
        Self { next: 0 }
    }
}

impl MaskCoins for FixedCoins {
    fn draw(&mut self) -> Result<(Fp3, u32)> {
        let number = self.next;
        self.next = self.next.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok((Fp3::from_u64(0x400 + number as u64), number))
    }

    fn redraw(&self, number: u32) -> Result<Fp3> {
        Ok(Fp3::from_u64(0x400 + number as u64))
    }
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

fn mask_tree_plan(log_n: u32, salt_bytes: usize) -> Result<MaskTreePlan> {
    let plan = MaskTreePlan {
        tree_number: TREE_NUMBER,
        leaf_class: 0,
        log_n,
        mask_rows: 1,
        salt_bytes,
        run: TREE_RUN,
        configuration: TREE_CONFIGURATION,
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(plan)
}

fn table_salts(shape: IndexShape) -> Result<Vec<Vec<u8>>> {
    let columns = usize::try_from(shape.code.n()).map_err(|_| Error::ResourceLimit)?;
    Ok((0..columns)
        .map(|position| vec![(position as u8) ^ 0xA5; shape.salt_bytes])
        .collect())
}

/// The verifier's fixed shape and configuration for this demonstration. It is
/// supplied independently of the source, which must match it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRowPlan {
    pub table_shape: IndexShape,
    pub configuration: Digest,
}

impl GroupRowPlan {
    pub fn new(configuration: Digest) -> Result<Self> {
        let table_shape = IndexShape {
            code: build_code_plan(3, 4, 1)?,
            v: 2,
            answers: 1,
            salt_bytes: zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES,
            distances: zkie_engine::pcs::request::table::TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        Ok(Self {
            table_shape,
            configuration,
        })
    }
}

/// The candidate GroupRow source. Main, old-mask, fresh-mask and random tails
/// are separate fields.
pub struct GroupRowSource {
    pub plan: GroupRowPlan,
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub group: u64,
    pub fresh: u64,
    pub tree_number: u64,
    pub table_root: Digest,
    pub table_messages: Vec<Vec<Fp>>,
    pub table_tails: Vec<Vec<Fp>>,
    pub main_messages: Vec<Fp3>,
    pub main_tails: Vec<Fp3>,
    pub old_mask_messages: Vec<Fp3>,
    pub old_mask_tail: Vec<Fp3>,
    pub fresh_mask: Vec<Fp3>,
    pub fresh_tail: Vec<Fp3>,
    pub mask: FoldMask,
    pub fresh_root: Digest,
    pub coefficients: Vec<Fp3>,
    pub mask_coefficients: Vec<Fp3>,
    pub target: Fp3,
    pub start_sum: Fp3,
    pub row: FoldInputRow,
}

impl GroupRowSource {
    pub fn mask_tree_plan(&self) -> Result<MaskTreePlan> {
        mask_tree_plan(
            self.plan.table_shape.code.log_n,
            self.plan.table_shape.salt_bytes,
        )
    }
}

fn source_from_table(
    table: &crate::proof::column_session::AssetEncodingTable,
    plan: GroupRowPlan,
) -> Result<GroupRowSource> {
    source_from_table_with_old_mask(table, plan, None, None)
}

fn source_from_table_with_old_mask(
    table: &crate::proof::column_session::AssetEncodingTable,
    plan: GroupRowPlan,
    old_mask_override: Option<Vec<Fp3>>,
    old_mask_tail_override: Option<Vec<Fp3>>,
) -> Result<GroupRowSource> {
    if table.shape != plan.table_shape || table.reader.configuration != plan.configuration {
        return Err(Error::StatementMismatch);
    }
    let mask_code = build_code_plan(plan.table_shape.code.log_n, MASK_ELL, MASK_R)?;
    let shape = FoldShape {
        code: table.shape.code,
        mask_code,
        k: FOLD_K,
        h: FOLD_H,
        mask_leaf_class: 0,
    };
    if fold_mask_shape_fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    let layout = FoldGroupLayout {
        group: GROUP,
        k: FOLD_K,
        h: FOLD_H,
        members: vec![
            FoldMemberLayout {
                oracle: MAIN_ORACLE,
                main_rows: 1,
                mask_rows: 0,
                kept: false,
            },
            FoldMemberLayout {
                oracle: STATIC_ORACLE,
                main_rows: 1,
                mask_rows: 0,
                kept: false,
            },
        ],
    };
    if layout.fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    let rows =
        usize::try_from(plan.table_shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let r = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    if table.messages.len() != rows || table.tails.len() != rows {
        return Err(Error::ShapeMismatch);
    }
    let mut main_messages = Vec::with_capacity(rows * ell);
    let mut main_tails = Vec::with_capacity(rows * r);
    for row in &table.messages {
        if row.len() != ell {
            return Err(Error::ShapeMismatch);
        }
        main_messages.extend(row.iter().map(|value| Fp3::from_base(*value)));
    }
    for row in &table.tails {
        if row.len() != r {
            return Err(Error::ShapeMismatch);
        }
        main_tails.extend(row.iter().map(|value| Fp3::from_base(*value)));
    }
    let mask_rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let mask_len = usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_tail_len =
        usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let old_mask_messages: Vec<Fp3> = match old_mask_override {
        Some(values) => values,
        None => (0..mask_rows * mask_len)
            .map(|index| Fp3::from_u64(0x200 + index as u64))
            .collect(),
    };
    if old_mask_messages.len() != mask_rows * mask_len {
        return Err(Error::ShapeMismatch);
    }
    let old_mask_tail: Vec<Fp3> = match old_mask_tail_override {
        Some(values) => values,
        None => (0..mask_rows * mask_tail_len)
            .map(|index| Fp3::from_u64(0x300 + index as u64))
            .collect(),
    };
    if old_mask_tail.len() != mask_rows * mask_tail_len {
        return Err(Error::ShapeMismatch);
    }
    let coefficients: Vec<Fp3> = (0..rows * ell)
        .map(|index| Fp3::from_u64(1 + index as u64))
        .collect();
    let mask_coefficients: Vec<Fp3> = (0..mask_rows * mask_len)
        .map(|index| Fp3::from_u64(0x100 + index as u64))
        .collect();
    let target = dot(&coefficients, &main_messages)?.add(&dot(&mask_coefficients, &old_mask_messages)?);
    let row = FoldInputRow {
        group: GROUP,
        k: FOLD_K,
        h: FOLD_H,
        members: vec![MAIN_ORACLE, STATIC_ORACLE],
        coefficients: coefficients.clone(),
        mask_coefficients: mask_coefficients.clone(),
        kept: Vec::new(),
        target,
    };
    row.check(&shape, &layout)?;
    let mut coins = FixedCoins::default();
    let mut mask = fold_mask_make(&shape, &mut coins)?;
    fold_mask_make_tail(&shape, &mut mask, &mut coins)?;
    mask.sent_as(FRESH_ID)?;
    let fresh_mask = fold_mask_table(
        FOLD_K,
        FOLD_H,
        mask_len,
        mask.coefficients(),
    )?;
    let fresh_tail = mask
        .tail()
        .ok_or(Error::ProverStateMissing)?
        .values
        .clone();
    let salts = table_salts(plan.table_shape)?;
    let fresh_tree = MaskTree::commit(
        mask_tree_plan(plan.table_shape.code.log_n, plan.table_shape.salt_bytes)?,
        &shape,
        &[fresh_mask.clone()],
        &[fresh_tail.clone()],
        &salts,
    )?;
    let start_sum = {
        let prover = FoldProverSession::new(
            &shape,
            &layout,
            &mask,
            row.clone(),
            main_messages.clone(),
            old_mask_messages.clone(),
            Vec::new(),
        )?;
        prover.sum()
    };
    Ok(GroupRowSource {
        plan,
        shape,
        layout,
        group: GROUP,
        fresh: FRESH_ID,
        tree_number: TREE_NUMBER,
        table_root: table.reader.root,
        table_messages: table.messages.clone(),
        table_tails: table.tails.clone(),
        main_messages,
        main_tails,
        old_mask_messages,
        old_mask_tail,
        fresh_mask,
        fresh_tail,
        mask,
        fresh_root: fresh_tree.root,
        coefficients,
        mask_coefficients,
        target,
        start_sum,
        row,
    })
}

/// The independent verifier-side comparison result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRowComparison {
    pub asset_opening: bool,
    pub main_rows: bool,
    pub table_commitment: bool,
    /// Local consistency of the fresh fixture with a root rebuilt from the same
    /// fixture data. This is not trusted fresh authentication.
    pub fresh_local_consistency: bool,
}

fn compare_main_rows(
    expected_main: &[Fp3],
    expected_tails: &[Fp3],
    source: &GroupRowSource,
) -> Result<()> {
    if expected_main != source.main_messages || expected_tails != source.main_tails {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

fn compare_source(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    asset_path: &str,
    opening: &AssetOpening,
    source: &GroupRowSource,
    plan: &GroupRowPlan,
) -> Result<GroupRowComparison> {
    if source.plan != *plan {
        return Err(Error::StatementMismatch);
    }
    verify_model_anchor(
        public,
        trusted.anchor_id,
        trusted.model_id,
        trusted.parameters_digest,
        trusted.asset_specs,
    )?;
    let meta = public
        .assets
        .iter()
        .find(|asset| asset.path == asset_path)
        .ok_or(Error::AssetMismatch)?;
    if opening.asset_id != meta.asset_id
        || opening.start != meta.start
        || opening.root != meta.root
        || opening.row_count != meta.rows
        || opening.rows.len() != meta.rows as usize
    {
        return Err(Error::StatementMismatch);
    }
    public.verify_opening(opening)?;

    let shape = plan.table_shape;
    let rows = usize::try_from(shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let r = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let plain = usize::try_from(shape.plain()).map_err(|_| Error::ResourceLimit)?;
    let mut expected_messages = vec![vec![Fp::ZERO; ell]; rows];
    let expected_tail_rows = vec![vec![Fp::ZERO; r]; rows];
    for (row, message) in expected_messages.iter_mut().enumerate() {
        let source_row = opening.rows.get(row).ok_or(Error::IncompleteArtifact)?;
        if source_row.row != row as u64 || source_row.values.len() != meta.row_width {
            return Err(Error::ShapeMismatch);
        }
        for (column, entry) in message.iter_mut().take(plain).enumerate() {
            let value = *source_row.values.get(column).ok_or(Error::ShapeMismatch)?;
            *entry = fp_from_signed_word(value);
        }
    }
    let expected_main: Vec<Fp3> = expected_messages
        .iter()
        .flat_map(|row| row.iter().map(|value| Fp3::from_base(*value)))
        .collect();
    let expected_tails: Vec<Fp3> = expected_tail_rows
        .iter()
        .flat_map(|row| row.iter().map(|value| Fp3::from_base(*value)))
        .collect();
    compare_main_rows(&expected_main, &expected_tails, source)?;
    let salts = table_salts(shape)?;
    let mut expected_source = VecTableSource::new(expected_messages, expected_tail_rows)?;
    let expected_table = commit_table_streaming(
        &mut expected_source,
        shape,
        plan.configuration,
        &salts,
        64,
        u64::MAX,
    )?;
    if expected_table.reader.root != source.table_root {
        return Err(Error::StatementMismatch);
    }
    let fresh_tree = MaskTree::commit(
        source.mask_tree_plan()?,
        &source.shape,
        &[source.fresh_mask.clone()],
        &[source.fresh_tail.clone()],
        &salts,
    )?;
    if fresh_tree.root != source.fresh_root {
        return Err(Error::StatementMismatch);
    }
    Ok(GroupRowComparison {
        asset_opening: true,
        main_rows: true,
        table_commitment: true,
        fresh_local_consistency: true,
    })
}

/// The separate retained components and the FOLD result.
pub struct GroupRowOutcome {
    pub asset_root: Digest,
    pub table_root: Digest,
    pub fresh_root: Digest,
    pub comparison: GroupRowComparison,
    pub same_source: bool,
    pub main_messages: Vec<Fp3>,
    /// Fixed fixture tail; not folded and not checked in this step.
    pub main_tails: Vec<Fp3>,
    pub old_mask_messages: Vec<Fp3>,
    /// Fixed fixture tail; not folded and not checked in this step.
    pub old_mask_tail: Vec<Fp3>,
    pub fresh_mask: Vec<Fp3>,
    /// Fixed fixture tail; not folded and not checked in this step.
    pub fresh_tail: Vec<Fp3>,
    pub coefficients: Vec<Fp3>,
    pub mask_coefficients: Vec<Fp3>,
    pub target: Fp3,
    pub eps: Fp3,
    pub gammas: Vec<Fp3>,
    pub round_messages: Vec<(Fp3, Fp3)>,
    pub order: Vec<&'static str>,
    pub fold_output: FoldOutput,
    pub fold_next_stage_closed: bool,
    pub handoff_rejected: bool,
    pub main_tail_folded: bool,
    pub old_mask_tail_folded: bool,
    pub fresh_tail_folded: bool,
}

fn start_bytes(source: &GroupRowSource, sum: Fp3) -> Result<Vec<u8>> {
    let notice = fold_mask_notice(&source.shape)?;
    let message = FoldStartMessage {
        id: source.fresh,
        claimed_group: source.group,
        tree_number: source.tree_number,
        notice,
        root: source.fresh_root,
        sum,
    };
    let mut writer = CanonicalWriter::new();
    writer.put_u64(message.id);
    writer.put_u64(message.claimed_group);
    writer.put_u64(message.tree_number);
    writer.put_u64(message.notice.mask_rows);
    writer.put_u64(message.notice.mask_code.log_n as u64);
    writer.put_u64(message.notice.mask_code.ell);
    writer.put_u64(message.notice.mask_code.r);
    writer.put_u64(message.notice.answers as u64);
    writer.put_u64(message.notice.leaf_class as u64);
    writer.put_digest(&message.root);
    writer.put_fp3(&message.sum);
    Ok(writer.into_bytes())
}

struct ComponentRun {
    source: Arc<GroupRowSource>,
    eps: Fp3,
    gammas: Vec<Fp3>,
    messages: Vec<(Fp3, Fp3)>,
    order: Vec<&'static str>,
    output: FoldOutput,
    handoff_rejected: bool,
    main_tail_folded: bool,
    old_mask_tail_folded: bool,
    fresh_tail_folded: bool,
}

fn run_component(source: Arc<GroupRowSource>) -> Result<ComponentRun> {
    let prover = FoldProverSession::new(
        &source.shape,
        &source.layout,
        &source.mask,
        source.row.clone(),
        source.main_messages.clone(),
        source.old_mask_messages.clone(),
        Vec::new(),
    )?;
    let start_sum = prover.sum();
    if start_sum != source.start_sum {
        return Err(Error::StatementMismatch);
    }
    let bytes = start_bytes(&source, start_sum)?;
    let mut transcript = FoldRoundTranscript::new(source.shape.k, 0x4752, &bytes)?;
    let eps = transcript.draw_epsilon(&bytes)?;
    let mut verifier = FoldRoundVerifier::new(start_sum, eps, source.row.target);
    let mut prover = prover;
    prover.start(eps, &FixedCoins::default())?;
    let mut order = vec!["fresh_commit", "start_sum", "epsilon"];
    let mut gammas = Vec::with_capacity(source.shape.k as usize);
    let mut messages = Vec::with_capacity(source.shape.k as usize);
    for _ in 0..source.shape.k {
        let q = prover.round()?;
        order.push("round_message");
        let gamma = transcript.draw_round(&q[0], &q[2])?;
        order.push("gamma");
        verifier.round(q[0], q[2], gamma);
        prover.bind(gamma)?;
        messages.push((q[0], q[2]));
        gammas.push(gamma);
    }
    let end = prover.end()?;
    let main_end = prover.main_messages().to_vec();
    let output = fold_output(
        &source.shape,
        &source.layout,
        &source.row,
        source.fresh,
        source.fresh_root,
        source.tree_number,
        eps,
        start_sum,
        &gammas,
        verifier.claim(),
    )?;
    if output.target != verifier.claim() {
        return Err(Error::InvalidProof);
    }
    let end_value = dot(&output.coefficients, &main_end)?
        .add(&dot(&output.mask_coefficients, &end.mu)?)
        .add(&dot(&output.fresh_coefficients, &end.fresh)?);
    if end_value != output.target {
        return Err(Error::InvalidProof);
    }
    let anchor = FoldHandoffAnchor {
        shape: source.shape,
        group: source.group,
        members: source.row.members.clone(),
        fresh: source.fresh,
        fresh_root: source.fresh_root,
        tree_number: source.tree_number,
        layout: source.layout.clone(),
        fold_gamma: gammas.clone(),
        fold_epsilon: eps,
        fold_sum: start_sum,
    };
    let handoff_rejected = match FoldHandoff::from_output(&output, &anchor) {
        Ok(handoff) => handoff.handoff_statement().is_err(),
        Err(_) => false,
    };
    Ok(ComponentRun {
        source,
        eps,
        gammas,
        messages,
        order,
        output,
        handoff_rejected,
        main_tail_folded: false,
        old_mask_tail_folded: false,
        fresh_tail_folded: false,
    })
}

/// Runs the full comparison and the isolated FOLD component over the same
/// candidate source.
pub fn verify_group_row_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<GroupRowOutcome> {
    let configuration = input.parameters.digest()?;
    let plan = GroupRowPlan::new(configuration)?;
    let table = build_authenticated_encoding_table_with_configuration(
        public,
        trusted,
        asset_path,
        opening,
        configuration,
    )?;
    let source = Arc::new(source_from_table(&table, plan)?);
    let comparison = compare_source(public, trusted, asset_path, opening, &source, &plan)?;
    let run = run_component(Arc::clone(&source))?;
    let same_source = Arc::ptr_eq(&source, &run.source);
    if !same_source {
        return Err(Error::InvalidSessionState);
    }
    let fold_next_stage_closed = run.output.next_stage().is_err();
    if !fold_next_stage_closed {
        return Err(Error::InvalidSessionState);
    }
    Ok(GroupRowOutcome {
        asset_root: table.asset_root,
        table_root: table.reader.root,
        fresh_root: source.fresh_root,
        comparison,
        same_source,
        main_messages: source.main_messages.clone(),
        main_tails: source.main_tails.clone(),
        old_mask_messages: source.old_mask_messages.clone(),
        old_mask_tail: source.old_mask_tail.clone(),
        fresh_mask: source.fresh_mask.clone(),
        fresh_tail: source.fresh_tail.clone(),
        coefficients: source.coefficients.clone(),
        mask_coefficients: source.mask_coefficients.clone(),
        target: source.target,
        eps: run.eps,
        gammas: run.gammas,
        round_messages: run.messages,
        order: run.order,
        fold_output: run.output,
        fold_next_stage_closed,
        handoff_rejected: run.handoff_rejected,
        main_tail_folded: run.main_tail_folded,
        old_mask_tail_folded: run.old_mask_tail_folded,
        fresh_tail_folded: run.fresh_tail_folded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use zkie_engine::pcs::request::table::TableDistances;

    fn demo_shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn table_from(messages: Vec<Vec<Fp>>) -> crate::proof::column_session::AssetEncodingTable {
        let shape = demo_shape();
        let tails = vec![vec![Fp::ZERO; 1]; messages.len()];
        let salts = table_salts(shape).unwrap();
        let mut source = VecTableSource::new(messages.clone(), tails.clone()).unwrap();
        let table = commit_table_streaming(
            &mut source,
            shape,
            Digest([0x61; 32]),
            &salts,
            64,
            u64::MAX,
        )
        .unwrap();
        crate::proof::column_session::AssetEncodingTable {
            reader: table.reader.clone(),
            table,
            source: std::cell::RefCell::new(source),
            asset_root: Digest([0x62; 32]),
            shape,
            messages,
            tails,
        }
    }

    fn plan_fixture() -> GroupRowPlan {
        GroupRowPlan::new(Digest([0x61; 32])).unwrap()
    }

    fn table_fixture() -> crate::proof::column_session::AssetEncodingTable {
        let messages = (0..4)
            .map(|row| {
                (0..4)
                    .map(|column| Fp::from_u64(1 + row * 4 + column))
                    .collect()
            })
            .collect();
        table_from(messages)
    }

    fn source_fixture() -> GroupRowSource {
        source_from_table(&table_fixture(), plan_fixture()).unwrap()
    }

    fn fresh_root_matches(source: &GroupRowSource) -> bool {
        MaskTree::commit(
            source.mask_tree_plan().unwrap(),
            &source.shape,
            &[source.fresh_mask.clone()],
            &[source.fresh_tail.clone()],
            &table_salts(source.plan.table_shape).unwrap(),
        )
        .map(|tree| tree.root == source.fresh_root)
        .unwrap_or(false)
    }

    #[test]
    fn source_components_and_fixed_randomness_are_separate() {
        let source = source_fixture();
        assert_eq!(source.main_messages.len(), 16);
        assert_eq!(source.old_mask_messages.len(), 6);
        assert_eq!(source.fresh_mask.len(), 6);
        assert_ne!(source.main_messages[0], source.old_mask_messages[0]);
        assert_ne!(source.old_mask_messages[0], source.fresh_mask[0]);
        assert_eq!(source.mask.coefficients().len(), 5);
        assert_eq!(source.row.kept.len(), 0);
        assert_eq!(
            dot(&source.coefficients, &source.main_messages)
                .unwrap()
                .add(&dot(&source.mask_coefficients, &source.old_mask_messages).unwrap()),
            source.target
        );
    }

    #[test]
    fn component_order_and_fold_output_are_consistent() {
        let source = Arc::new(source_fixture());
        let run = run_component(Arc::clone(&source)).unwrap();
        assert_eq!(&run.order[..3], &["fresh_commit", "start_sum", "epsilon"][..]);
        let first_message = run.order.iter().position(|event| *event == "round_message").unwrap();
        let first_gamma = run.order.iter().position(|event| *event == "gamma").unwrap();
        assert!(first_message < first_gamma);
        assert_eq!(run.gammas.len(), 2);
        assert_eq!(run.messages.len(), 2);
        assert_eq!(run.output.sum, run.source.start_sum);
        assert_eq!(run.output.eps, run.eps);
        assert_eq!(run.output.coefficients.len(), 4);
        assert_eq!(run.output.mask_coefficients.len(), 6);
        assert_eq!(run.output.fresh_coefficients.len(), 6);
        assert!(run.output.next_stage().is_err());
        assert!(run.handoff_rejected);
        assert!(!run.main_tail_folded);
        assert!(!run.old_mask_tail_folded);
        assert!(!run.fresh_tail_folded);
        assert_eq!(run.output.mask_coefficients.len(), 6);
        assert_eq!(run.output.fresh_coefficients.len(), 6);
    }

    #[test]
    fn independent_reconstruction_rejects_a_valid_recommitment_mismatch() {
        let source = source_fixture();
        let plan = plan_fixture();
        let shape = plan.table_shape;
        let mut messages = source.table_messages.clone();
        messages[0][0] = Fp::from_u64(999);
        let salts = table_salts(shape).unwrap();
        let tails = source.table_tails.clone();
        let mut changed_source = VecTableSource::new(messages.clone(), tails.clone()).unwrap();
        let changed = commit_table_streaming(
            &mut changed_source,
            shape,
            plan.configuration,
            &salts,
            64,
            u64::MAX,
        )
        .unwrap();
        assert_ne!(changed.reader.root, source.table_root);
        let expected: Vec<Fp3> = source
            .table_messages
            .iter()
            .flat_map(|row| row.iter().map(|value| Fp3::from_base(*value)))
            .collect();
        let altered: Vec<Fp3> = messages
            .iter()
            .flat_map(|row| row.iter().map(|value| Fp3::from_base(*value)))
            .collect();
        assert_ne!(expected, altered);
        assert!(matches!(
            compare_main_rows(&expected, &source.main_tails, &source),
            Ok(())
        ));
        assert!(matches!(
            compare_main_rows(&altered, &source.main_tails, &source),
            Err(Error::StatementMismatch)
        ));
        let altered_table = table_from(messages);
        let altered_source = source_from_table(&altered_table, plan).unwrap();
        assert_ne!(altered_source.table_root, source.table_root);
        assert!(matches!(
            compare_main_rows(&expected, &altered_source.main_tails, &altered_source),
            Err(Error::StatementMismatch)
        ));
    }

    #[test]
    fn verifier_plan_shape_and_configuration_are_checked() {
        let table = table_fixture();
        let mut wrong_configuration = plan_fixture();
        wrong_configuration.configuration = Digest([0x63; 32]);
        assert!(source_from_table(&table, wrong_configuration).is_err());
        let mut wrong_shape = plan_fixture();
        wrong_shape.table_shape.code = build_code_plan(3, 6, 1).unwrap();
        assert!(source_from_table(&table, wrong_shape).is_err());
    }

    #[test]
    fn input_parts_change_their_own_layer_and_tails_are_not_folded() {
        let plan = plan_fixture();
        let source = source_fixture();
        let mut altered_main = source.main_messages.clone();
        altered_main[0] = altered_main[0].add(&Fp3::ONE);
        assert!(matches!(
            compare_main_rows(&altered_main, &source.main_tails, &source),
            Err(Error::StatementMismatch)
        ));
        let mut altered_main_tail = source.main_tails.clone();
        altered_main_tail[0] = altered_main_tail[0].add(&Fp3::ONE);
        assert!(matches!(
            compare_main_rows(&source.main_messages, &altered_main_tail, &source),
            Err(Error::StatementMismatch)
        ));

        let table = table_fixture();
        let mut altered_old = source.old_mask_messages.clone();
        altered_old[0] = altered_old[0].add(&Fp3::ONE);
        let altered_old_source = source_from_table_with_old_mask(
            &table,
            plan,
            Some(altered_old),
            None,
        )
        .unwrap();
        assert_ne!(altered_old_source.target, source.target);
        let altered_old_tail_source = source_from_table_with_old_mask(
            &table,
            plan,
            None,
            Some(vec![source.old_mask_tail[0].add(&Fp3::ONE)]),
        )
        .unwrap();
        assert_eq!(altered_old_tail_source.target, source.target);

        let mut altered_fresh = source_fixture();
        altered_fresh.fresh_mask[0] = altered_fresh.fresh_mask[0].add(&Fp3::ONE);
        assert!(!fresh_root_matches(&altered_fresh));
        let mut altered_fresh_tail = source_fixture();
        altered_fresh_tail.fresh_tail[0] =
            altered_fresh_tail.fresh_tail[0].add(&Fp3::ONE);
        assert!(!fresh_root_matches(&altered_fresh_tail));
    }

    #[test]
    fn order_violation_is_rejected_at_the_transcript_layer() {
        let mut transcript = FoldRoundTranscript::new(2, 0x4752, &[]).unwrap();
        assert!(matches!(
            transcript.draw_round(&Fp3::ONE, &Fp3::ONE),
            Err(Error::InvalidSessionState)
        ));
    }
}
