//! Isolated G_u declaration registration adapted to the existing BAT layer.
//!
//! This module runs the fixed pre-FOLD column session, takes the record, sample
//! and column declarations registered by that session, adapts them to BAT
//! claims, freezes one BAT group, closes the algebraic round on a snapshot of
//! the session chain, draws `x`, and lets BAT combine the declarations.
//!
//! The declaration adapter is semantic: a G_u relation row becomes a BAT claim
//! with a main term and, when present, a mask term. BAT requires power-of-two
//! code widths; the G_u mask code has six entries, so the BAT mask term is
//! zero-padded to the next power of two. The padding preserves the linear
//! functional because the padded coefficients are zero. The G_u oracle/group
//! ids are mapped to BAT's sequential registration ids; the original ids are
//! retained in the outcome. This is an adapter, not a second BAT.
//!
//! Chain ownership is transferred through the stages: `take_chain_for_bat`
//! consumes the FOLDEQ session after the declarations are joined, and
//! `BatSession::into_chain` consumes the BAT session and returns the advanced
//! chain. The old handles cannot advance independently. This is one continuous
//! state handoff, not a copied digest and not a forked snapshot. A missing
//! stage, an earlier stage, or a BAT transfer before a completed run is
//! rejected; consuming the handles makes reuse a type error.
//!
//! The BAT round is a candidate demonstration. BAT's close-algebraic step and
//! the session prefix are used, but this is not an exact reproduction of a
//! reference transcript and is not a permanent proof format. The binding basis
//! is the frozen `BatRunInput` (group layout, member order, declaration class
//! order and row count); the challenge domain is BAT's empty algebraic close;
//! the declaration order is BAT's fixed class order. No extra hashing or
//! redundant protocol round is introduced.
//!
//! Supported declarations come from the same session reader, G_u identity,
//! shape and opened columns:
//!
//! | declaration | source | BAT class |
//! | --- | --- | --- |
//! | sample | the session SEND answer path | `Sample` (fake demonstration answer) |
//! | record | the session record join | `Plain` |
//! | column | the session column read path | `Plain` (opens bind only the column root) |
//!
//! Not supported and listed: multi-oracle SEND, a real SEND provider and a
//! fresh-mask declaration. The G_u root is self-reported; the G_u row is a
//! pending declaration. BAT combines claims and does not judge truth. Without a
//! later proximity/relation check this is not a check pass.
//!
//! Declaration fixing follows the confirmed session order: the verifier takes
//! the declarations from the same session's bound table root/record, G_u
//! message, challenges and authenticated columns; the BAT group is frozen and
//! cannot be replaced; only then is `x` drawn. No arbitrary external
//! declaration is hashed to claim a trusted source, and no extra declaration
//! absorption round is added: the declarations depend on the already-fixed
//! inputs and are frozen by `BatRunInput`.
//!
//! The six-to-eight mask adapter is only for this local algebraic
//! demonstration. Its combined row must not flow into a path that claims a real
//! FOLD or proximity result; the adapter-only and downstream flags record that
//! isolation.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::bat::{
    BatGroupLayout, BatGroupRow, BatOracleShape, BatPart, BatRow, BatRowKind, BatSession,
    BatStatement,
};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::gu::GuShape;
use zkie_engine::pcs::request::relation::{GuRelationKind, GuRelationRow};
use zkie_engine::pcs::request::table::{IndexShape, SampleSeed};

use crate::proof::column_session::{
    bind_table, build_authenticated_encoding_table_with_configuration, open_gu_flow_with_answers,
    AssetEncodingTable, ColumnBindingInput, GU_NOTICE_ROOT,
};
use crate::proof::model_commitment::{AssetOpening, ModelCommitmentPublic};
use crate::proof::session_baseline::TrustedModel;

/// The BAT adapter shape for one G_u oracle. The oracle/group ids are BAT's
/// sequential ids; the original G_u ids are kept separately in the outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatAdapterShape {
    pub oracle: u64,
    pub group: u64,
    /// The original G_u oracle/group ids used by the session declarations.
    pub source_oracle: u64,
    pub source_group: u64,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub main_code: CodeParams,
    pub mask_code: CodeParams,
    /// The original G_u mask code width before power-of-two padding.
    pub mask_orig_ell: u64,
    pub answers: u32,
}

impl BatAdapterShape {
    fn from_gu(gu: &zkie_engine::pcs::request::gu::GuShape) -> Result<Self> {
        let original_mask_code = gu.u.mask_code.ok_or(Error::UnsupportedPrimitive)?;
        let mask_ell = original_mask_code
            .ell
            .checked_next_power_of_two()
            .ok_or(Error::ResourceLimit)?;
        // BAT requires power-of-two code widths. When padding leaves no room
        // for the original tail, use an empty tail for the BAT-only mask term;
        // the padded coefficients are zero and the original tail is not part
        // of the declaration target.
        let n = 1u64
            .checked_shl(original_mask_code.log_n)
            .ok_or(Error::ResourceLimit)?;
        let mask_r = if mask_ell.saturating_add(original_mask_code.r) <= n {
            original_mask_code.r
        } else {
            0
        };
        let mask_code = build_code_plan(original_mask_code.log_n, mask_ell, mask_r)?;
        if !gu.u.code.ell.is_power_of_two() || !mask_code.ell.is_power_of_two() {
            return Err(Error::UnsupportedPrimitive);
        }
        Ok(Self {
            oracle: 0,
            group: 0,
            source_oracle: gu.oracle,
            source_group: gu.group,
            main_rows: gu.u.main_rows,
            mask_rows: gu.u.mask_rows,
            main_code: gu.u.code,
            mask_code,
            mask_orig_ell: original_mask_code.ell,
            answers: gu.u.answers,
        })
    }

    fn oracle_shape(&self) -> BatOracleShape {
        BatOracleShape {
            oracle: self.oracle,
            main_rows: self.main_rows,
            mask_rows: self.mask_rows,
            main_code: Some(self.main_code),
            mask_code: Some(self.mask_code),
            answers: self.answers,
        }
    }

    fn main_table_len(&self) -> Result<usize> {
        let bits = self.oracle_shape().table_bits(BatPart::Main);
        1usize.checked_shl(bits).ok_or(Error::ResourceLimit)
    }

    fn mask_table_len(&self) -> Result<usize> {
        let bits = self.oracle_shape().table_bits(BatPart::Mask);
        1usize.checked_shl(bits).ok_or(Error::ResourceLimit)
    }
}

fn adapt_row(row: &GuRelationRow, shape: &BatAdapterShape) -> Result<BatRow> {
    if row.oracle != shape.source_oracle || row.group != shape.source_group {
        return Err(Error::StatementMismatch);
    }
    if row.main.len() as u64 != shape.main_rows.saturating_mul(shape.main_code.ell)
        || row.mask.len() as u64 != shape.mask_rows.saturating_mul(shape.mask_orig_ell)
    {
        return Err(Error::ShapeMismatch);
    }
    let kind = match row.kind {
        GuRelationKind::Sample => BatRowKind::Sample,
        GuRelationKind::Record | GuRelationKind::Column => BatRowKind::Plain,
    };
    let mut out = BatRow::new(kind);
    if !row.main.is_empty() {
        let mut main = row.main.clone();
        main.resize(shape.main_table_len()?, Fp3::ZERO);
        out.add_term(shape.oracle, BatPart::Main, main);
    }
    if !row.mask.is_empty() {
        let mut mask = row.mask.clone();
        mask.resize(shape.mask_table_len()?, Fp3::ZERO);
        out.add_term(shape.oracle, BatPart::Mask, mask);
    }
    out.target = row.target;
    Ok(out)
}

/// The frozen BAT input summary checked by the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatFrozenSummary {
    pub oracle: u64,
    pub group: u64,
    pub order: Vec<BatRowKind>,
    pub targets: Vec<Fp3>,
    pub main_len: usize,
    pub mask_len: usize,
    pub count: usize,
}

/// The BAT run result, its retained declaration order and the advanced chain.
pub struct BatCombineRun {
    pub layout: BatGroupLayout,
    pub combined: BatGroupRow,
    pub x: Fp3,
    pub order: Vec<BatRowKind>,
    pub input: BatFrozenSummary,
    pub returned_chain: Chain,
}

/// Adapts the G_u declarations to BAT, freezes the group, closes the BAT
/// algebraic round on the supplied chain snapshot, draws `x` and combines.
pub fn run_bat_combine(
    rows: &[GuRelationRow],
    shape: &BatAdapterShape,
    chain: Chain,
) -> Result<BatCombineRun> {
    if rows.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let planned_rows = u64::try_from(rows.len()).map_err(|_| Error::ResourceLimit)?;
    let mut bat = BatSession::new(BatStatement::new(), chain, planned_rows)?;
    let statement = bat.statement_mut();
    statement.add_oracle(shape.oracle_shape())?;
    let group = statement.new_group(shape.main_code, shape.mask_code)?;
    let mut samples = Vec::new();
    let mut plains = Vec::new();
    for row in rows {
        let adapted = adapt_row(row, shape)?;
        match adapted.kind {
            BatRowKind::Sample => samples.push(adapted),
            BatRowKind::Plain => plains.push(adapted),
            BatRowKind::Generator => return Err(Error::UnsupportedPrimitive),
        }
    }
    if samples.len() != shape.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    statement.sample(shape.oracle, samples)?;
    statement.join(group, shape.oracle, false)?;
    for row in plains {
        statement.add_row(group, row)?;
    }
    let layout = statement.layout(group)?;
    let input = bat.begin_run(group)?;
    let order = input.rows.iter().map(|row| row.kind).collect::<Vec<_>>();
    let targets = input.rows.iter().map(|row| row.target).collect::<Vec<_>>();
    let count = input.rows.len();
    let summary = BatFrozenSummary {
        oracle: shape.oracle,
        group: layout.group,
        order: order.clone(),
        targets,
        main_len: shape.main_table_len()?,
        mask_len: shape.mask_table_len()?,
        count,
    };
    let combined = bat.finish_run(input)?;
    let x = bat.last_x().ok_or(Error::InvalidSessionState)?;
    let returned_chain = bat.into_chain()?;
    Ok(BatCombineRun {
        layout,
        combined,
        x,
        order,
        input: summary,
        returned_chain,
    })
}

fn kind_name(kind: GuRelationKind) -> &'static str {
    match kind {
        GuRelationKind::Sample => "Sample",
        GuRelationKind::Record => "Record",
        GuRelationKind::Column => "Column",
    }
}

/// The declaration counts and the BAT combination result.
pub struct GuCombineOutcome {
    pub reader_root: Digest,
    pub configuration: Digest,
    pub shape: IndexShape,
    pub table_identity: Digest,
    pub gu_root: Digest,
    pub gu_oracle: u64,
    pub gu_group: u64,
    pub gu_tree_number: u64,
    pub queries: Vec<u64>,
    pub declarations: Vec<GuRelationRow>,
    pub declaration_names: Vec<&'static str>,
    pub sample_count: usize,
    pub record_count: usize,
    pub column_count: usize,
    pub unsupported: Vec<&'static str>,
    pub bat_group: u64,
    pub bat_oracle: u64,
    pub bat_layout: BatGroupLayout,
    pub bat_planned_rows: u64,
    pub bat_order: Vec<BatRowKind>,
    pub bat_input: BatFrozenSummary,
    pub combined: BatGroupRow,
    pub x: Fp3,
    pub returned_chain: Chain,
    pub chain_handoff: bool,
    pub transcript_after_columns: Digest,
    pub same_session: bool,
    pub column_root_only: bool,
    pub self_reported_gu_root: bool,
    pub fake_send_answer: bool,
    pub bat_decides_truth: bool,
    pub three_c_connected: bool,
    pub reference_exact: bool,
    pub permanent_format: bool,
    pub adapter_only: bool,
    pub downstream_fold_connected: bool,
    pub sample_answers_provider_constructed: bool,
    pub verifier_standard_answer_comparison: bool,
}

/// Runs the candidate G_u declaration registration and BAT combination over
/// one real session. The result is an isolated data-flow demonstration.
pub fn verify_gu_combine_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<GuCombineOutcome> {
    verify_gu_combine_session_with_provider(
        public,
        trusted,
        input,
        asset_path,
        opening,
        GU_NOTICE_ROOT,
        |gu, _seeds| {
            Ok((0..gu.u.answers)
                .map(|answer| Fp3::from_u64(0x31 + answer as u64))
                .collect())
        },
    )
}

/// Runs the same demonstration with a caller-supplied SEND answer provider.
/// The provider receives the actual session G_u shape and the seed set drawn
/// after the fixed root. The verifier registers the returned answers as
/// pending sample claims; it has no standard-answer comparison path.
pub fn verify_gu_combine_session_with_provider<F>(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
    gu_root: Digest,
    answer_provider: F,
) -> Result<GuCombineOutcome>
where
    F: FnOnce(&GuShape, &[SampleSeed]) -> Result<Vec<Fp3>>,
{
    let configuration = input.parameters.digest()?;
    let table = build_authenticated_encoding_table_with_configuration(
        public,
        trusted,
        asset_path,
        opening,
        configuration,
    )?;
    verify_gu_combine_session_with_table_and_provider(
        input,
        &table,
        gu_root,
        answer_provider,
    )
}

/// Runs the same demonstration over a table that the caller already built from
/// the complete authenticated opening. This is used by the SEND fixture, which
/// must build and fix the oracle commitment before the session seed.
pub fn verify_gu_combine_session_with_table_and_provider<F>(
    input: &ColumnBindingInput,
    table: &AssetEncodingTable,
    gu_root: Digest,
    answer_provider: F,
) -> Result<GuCombineOutcome>
where
    F: FnOnce(&GuShape, &[SampleSeed]) -> Result<Vec<Fp3>>,
{
    let configuration = input.parameters.digest()?;
    let binding = bind_table(table, input)?;
    let flow = open_gu_flow_with_answers(
        table,
        binding,
        input.edge.clone(),
        gu_root,
        answer_provider,
    )?;
    let crate::proof::column_session::GuFlow {
        session,
        gu_root,
        queries,
        table_identity,
        transcript_after_columns,
    } = flow;
    let plan = session.plan();
    let gu = plan.gu.as_ref().ok_or(Error::StatementMismatch)?.clone();
    let reader_root = plan.reader.root;
    let reader_configuration = plan.reader.configuration;
    let reader_shape = plan.reader.shape;
    if reader_root != table.reader.root
        || reader_configuration != configuration
        || table_identity != reader_table_identity(&plan.reader)?
        || transcript_after_columns != *session.transcript_value()
    {
        return Err(Error::StatementMismatch);
    }
    let mut rows = session.pending_gu_rows().to_vec();
    for draw in 0..session.pending_column_count() {
        rows.push(session.column_relation_row(draw)?);
    }
    if rows.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let shape = BatAdapterShape::from_gu(&gu)?;
    let main_len = usize::try_from(
        shape
            .main_rows
            .checked_mul(shape.main_code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mask_len = usize::try_from(
        shape
            .mask_rows
            .checked_mul(shape.mask_orig_ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    for row in &rows {
        if row.oracle != gu.oracle
            || row.group != gu.group
            || row.main.len() != main_len
            || row.mask.len() != mask_len
        {
            return Err(Error::StatementMismatch);
        }
    }
    let chain = session.take_chain_for_bat()?;
    let run = run_bat_combine(&rows, &shape, chain)?;
    let bat_planned_rows =
        u64::try_from(rows.len()).map_err(|_| Error::ResourceLimit)?;
    let sample_count = rows
        .iter()
        .filter(|row| row.kind == GuRelationKind::Sample)
        .count();
    let record_count = rows
        .iter()
        .filter(|row| row.kind == GuRelationKind::Record)
        .count();
    let column_count = rows
        .iter()
        .filter(|row| row.kind == GuRelationKind::Column)
        .count();
    Ok(GuCombineOutcome {
        reader_root,
        configuration: reader_configuration,
        shape: reader_shape,
        table_identity,
        gu_root,
        gu_oracle: gu.oracle,
        gu_group: gu.group,
        gu_tree_number: gu.tree.tree_number,
        queries,
        declaration_names: rows.iter().map(|row| kind_name(row.kind)).collect(),
        declarations: rows,
        sample_count,
        record_count,
        column_count,
        unsupported: vec![
            "multi-oracle SEND",
            "real SEND provider",
            "fresh-mask declaration",
        ],
        bat_group: run.layout.group,
        bat_oracle: shape.oracle,
        bat_layout: run.layout,
        bat_planned_rows,
        bat_order: run.order,
        bat_input: run.input,
        combined: run.combined,
        x: run.x,
        returned_chain: run.returned_chain,
        chain_handoff: true,
        transcript_after_columns,
        same_session: true,
        column_root_only: true,
        self_reported_gu_root: true,
        fake_send_answer: true,
        bat_decides_truth: false,
        three_c_connected: false,
        reference_exact: false,
        permanent_format: false,
        adapter_only: true,
        downstream_fold_connected: false,
        sample_answers_provider_constructed: true,
        verifier_standard_answer_comparison: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

    fn shape() -> BatAdapterShape {
        BatAdapterShape {
            oracle: 0,
            group: 0,
            source_oracle: 11,
            source_group: 5,
            main_rows: 2,
            mask_rows: 1,
            main_code: build_code_plan(3, 4, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            mask_orig_ell: 4,
            answers: 1,
        }
    }

    fn row(kind: GuRelationKind, target: u64) -> GuRelationRow {
        GuRelationRow {
            oracle: 11,
            group: 5,
            kind,
            main: (0..8).map(|index| Fp3::from_u64(index + 1)).collect(),
            mask: (0..4).map(|index| Fp3::from_u64(index + 10)).collect(),
            target: Fp3::from_u64(target),
        }
    }

    fn chain() -> Chain {
        Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            b"bat-adapter",
            &[],
            None,
        )
        .unwrap()
    }

    #[test]
    fn bat_combines_the_frozen_declarations() {
        let rows = vec![
            row(GuRelationKind::Sample, 7),
            row(GuRelationKind::Record, 8),
            row(GuRelationKind::Column, 9),
        ];
        let run = run_bat_combine(&rows, &shape(), chain()).unwrap();
        assert_eq!(
            run.order,
            vec![BatRowKind::Plain, BatRowKind::Plain, BatRowKind::Sample]
        );
        assert_eq!(run.input.count, 3);
        assert_eq!(run.input.order, run.order);
        assert_eq!(run.input.oracle, 0);
        assert_eq!(run.input.group, 0);
        assert_eq!(run.input.main_len, 8);
        assert_eq!(run.input.mask_len, 4);
        assert_eq!(run.combined.main.len(), 8);
        assert_eq!(run.combined.mask.len(), 4);
        assert_eq!(run.combined.group, 0);
        let x = run.x;
        let x2 = x.mul(&x);
        assert_eq!(
            run.combined.target,
            Fp3::from_u64(8)
                .add(&x.mul(&Fp3::from_u64(9)))
                .add(&x2.mul(&Fp3::from_u64(7)))
        );
    }

    #[test]
    fn a_wrong_declaration_is_still_pending() {
        let mut rows = vec![
            row(GuRelationKind::Sample, 7),
            row(GuRelationKind::Record, 8),
        ];
        rows[0].target = Fp3::from_u64(1000);
        let run = run_bat_combine(&rows, &shape(), chain()).unwrap();
        assert_eq!(
            run.combined.target,
            Fp3::from_u64(8).add(&run.x.mul(&Fp3::from_u64(1000)))
        );
        assert!(!run.combined.main.is_empty());
    }

    #[test]
    fn shape_mismatch_is_rejected_at_the_adapter() {
        let mut bad = row(GuRelationKind::Record, 8);
        bad.main.pop();
        assert!(matches!(
            run_bat_combine(&[bad], &shape(), chain()),
            Err(Error::ShapeMismatch)
        ));
        let mut bad = row(GuRelationKind::Record, 8);
        bad.mask.pop();
        assert!(matches!(
            run_bat_combine(&[bad], &shape(), chain()),
            Err(Error::ShapeMismatch)
        ));
        let mut bad = row(GuRelationKind::Record, 8);
        bad.group = 6;
        assert!(matches!(
            run_bat_combine(&[bad], &shape(), chain()),
            Err(Error::StatementMismatch)
        ));
        let mut bad = row(GuRelationKind::Record, 8);
        bad.oracle = 12;
        assert!(matches!(
            run_bat_combine(&[bad], &shape(), chain()),
            Err(Error::StatementMismatch)
        ));
    }

    #[test]
    fn a_closed_group_rejects_replacement() {
        let shape = shape();
        let rows = vec![
            row(GuRelationKind::Sample, 7),
            row(GuRelationKind::Record, 8),
        ];
        let mut bat = BatSession::new(BatStatement::new(), chain(), 2).unwrap();
        {
            let statement = bat.statement_mut();
            statement.add_oracle(shape.oracle_shape()).unwrap();
            let group = statement
                .new_group(shape.main_code, shape.mask_code)
                .unwrap();
            let sample = adapt_row(&rows[0], &shape).unwrap();
            let plain = adapt_row(&rows[1], &shape).unwrap();
            statement.sample(shape.oracle, vec![sample]).unwrap();
            statement.join(group, shape.oracle, false).unwrap();
            statement.add_row(group, plain).unwrap();
            bat.begin_run(group).unwrap();
        }
        assert!(matches!(
            bat.add_row(0, BatRow::new(BatRowKind::Plain)),
            Err(Error::InvalidSessionState)
        ));
    }

    #[test]
    fn bat_chain_transfer_requires_a_completed_run() {
        let bat = BatSession::new(BatStatement::new(), chain(), 0).unwrap();
        assert!(matches!(bat.into_chain(), Err(Error::InvalidSessionState)));
    }

    #[test]
    fn challenge_prefix_change_propagates_to_x_on_a_fixed_sample() {
        let rows = vec![row(GuRelationKind::Sample, 7)];
        let first_chain = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            b"bat-prefix-a",
            &[],
            None,
        )
        .unwrap();
        let second_chain = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            b"bat-prefix-b",
            &[],
            None,
        )
        .unwrap();
        let first = run_bat_combine(&rows, &shape(), first_chain).unwrap();
        let second = run_bat_combine(&rows, &shape(), second_chain).unwrap();
        // Fixed-sample propagation check only; this is not a general claim
        // about hash collisions or domain mapping.
        assert_ne!(first.x, second.x);
    }

    #[test]
    fn declaration_values_change_the_combination_not_the_challenge() {
        let rows = vec![row(GuRelationKind::Sample, 7)];
        let mut changed = rows.clone();
        changed[0].target = Fp3::from_u64(8);
        let first = run_bat_combine(&rows, &shape(), chain()).unwrap();
        let second = run_bat_combine(&changed, &shape(), chain()).unwrap();
        assert_eq!(first.x, second.x);
        assert_ne!(first.combined.target, second.combined.target);
    }
}
