//! Isolated G_u declaration registration and BAT combination demonstration.
//!
//! This module runs the fixed pre-FOLD column session, takes the record,
//! sample and column declarations that session registered, fixes the session
//! identity and declarations into the session chain snapshot, draws one
//! combination challenge `x`, and combines every declaration coefficient and
//! target by powers of `x`.
//!
//! Supported declarations come from the same session reader, G_u identity,
//! shape and opened columns:
//!
//! | declaration | source | note |
//! | --- | --- | --- |
//! | sample | the session SEND answer path | the answer is a fake demonstration value |
//! | record | the session record join under the bound index name | pending declaration |
//! | column | the session column read path | opens are authenticated against the column root only |
//!
//! Not supported in this step: multi-oracle SEND, a real SEND provider and a
//! fresh-mask declaration. These are listed rather than filled with extra
//! fixtures.
//!
//! The G_u row is a pending declaration. Registration binds the claimed G_u
//! root and the declaration coefficients/targets; it does not prove the G_u
//! value, its source or the declaration correctness. The G_u root is a
//! self-reported demonstration value and the SEND answer is a fake value; both
//! are recorded as such. Column openings are authenticated only against the
//! column root. BAT combines declarations and does not judge truth. Without a
//! later proximity/relation check, a successful combination is not a check
//! pass and no proof is accepted.
//!
//! The combined row is not connected to the 3c FOLD source in this step. The
//! 3c source has its own fixture claim/target relation; rewriting its target to
//! match this combination would not establish real consistency. The independent
//! registration/combination result is kept, and the 3c handoff remains
//! fail-closed.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::relation::{GuRelationKind, GuRelationRow};
use zkie_engine::pcs::request::table::IndexShape;
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::proof::column_session::{
    bind_table, build_authenticated_encoding_table_with_configuration, open_gu_flow,
    ColumnBindingInput,
};
use crate::proof::model_commitment::{AssetOpening, ModelCommitmentPublic};
use crate::proof::session_baseline::TrustedModel;

/// The combined declaration row. It is pending, not a validated relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombinedGuRow {
    pub oracle: u64,
    pub group: u64,
    pub main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub target: Fp3,
    pub kinds: Vec<GuRelationKind>,
    pub count: usize,
}

fn kind_word(kind: GuRelationKind) -> u64 {
    match kind {
        GuRelationKind::Sample => 1,
        GuRelationKind::Record => 2,
        GuRelationKind::Column => 3,
    }
}

/// Combines every declaration by powers of `x` in the order supplied.
/// Committed rows must share the oracle, group and coefficient shapes. The
/// function does not evaluate or judge the declarations.
pub fn combine_declarations(rows: &[GuRelationRow], x: Fp3) -> Result<CombinedGuRow> {
    let first = rows.first().ok_or(Error::InvalidConfig)?;
    let mut main = vec![Fp3::ZERO; first.main.len()];
    let mut mask = vec![Fp3::ZERO; first.mask.len()];
    let mut target = Fp3::ZERO;
    let mut power = Fp3::ONE;
    let mut kinds = Vec::with_capacity(rows.len());
    for row in rows {
        if row.oracle != first.oracle
            || row.group != first.group
            || row.main.len() != main.len()
            || row.mask.len() != mask.len()
        {
            return Err(Error::StatementMismatch);
        }
        for (sum, value) in main.iter_mut().zip(&row.main) {
            *sum = sum.add(&power.mul(value));
        }
        for (sum, value) in mask.iter_mut().zip(&row.mask) {
            *sum = sum.add(&power.mul(value));
        }
        target = target.add(&power.mul(&row.target));
        kinds.push(row.kind);
        power = power.mul(&x);
    }
    Ok(CombinedGuRow {
        oracle: first.oracle,
        group: first.group,
        main,
        mask,
        target,
        kinds,
        count: rows.len(),
    })
}

/// Absorbs only session-bound identity and declaration values. No additional
/// digest or unbound hash layer is introduced; the x challenge is the next
/// chain draw after these declarations.
fn absorb_declarations(
    chain: &mut Chain,
    gu_root: Digest,
    table_identity: Digest,
    reader_root: Digest,
    configuration: Digest,
    queries: &[u64],
    rows: &[GuRelationRow],
) -> Result<()> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(1);
    writer.put_digest(&gu_root);
    writer.put_digest(&table_identity);
    writer.put_digest(&reader_root);
    writer.put_digest(&configuration);
    writer.put_u64(queries.len() as u64);
    for query in queries {
        writer.put_u64(*query);
    }
    writer.put_u64(rows.len() as u64);
    for row in rows {
        writer.put_u64(kind_word(row.kind));
        writer.put_u64(row.oracle);
        writer.put_u64(row.group);
        writer.put_u64(row.main.len() as u64);
        for value in &row.main {
            writer.put_fp3(value);
        }
        writer.put_u64(row.mask.len() as u64);
        for value in &row.mask {
            writer.put_fp3(value);
        }
        writer.put_fp3(&row.target);
    }
    let bytes = writer.into_bytes();
    let mut message = chain.message();
    message.push_u64(bytes.len() as u64);
    for byte in bytes {
        message.push_u64(byte as u64);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_combine_challenge(
    chain: &mut Chain,
    gu_root: Digest,
    table_identity: Digest,
    reader_root: Digest,
    configuration: Digest,
    queries: &[u64],
    rows: &[GuRelationRow],
) -> Result<Fp3> {
    absorb_declarations(
        chain,
        gu_root,
        table_identity,
        reader_root,
        configuration,
        queries,
        rows,
    )?;
    chain.draw_e()
}

/// The declaration counts and the independent combination result.
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
    pub sample_count: usize,
    pub record_count: usize,
    pub column_count: usize,
    pub unsupported: Vec<&'static str>,
    pub combined: CombinedGuRow,
    pub x: Fp3,
    pub transcript_after_columns: Digest,
    pub same_session: bool,
    pub column_root_only: bool,
    pub self_reported_gu_root: bool,
    pub fake_send_answer: bool,
    pub bat_decides_truth: bool,
    pub three_c_connected: bool,
}

/// Runs the candidate G_u declaration registration and BAT combination over
/// one real session. The result is an isolated data-flow demonstration, not a
/// verification closure.
pub fn verify_gu_combine_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<GuCombineOutcome> {
    let configuration = input.parameters.digest()?;
    let table = build_authenticated_encoding_table_with_configuration(
        public,
        trusted,
        asset_path,
        opening,
        configuration,
    )?;
    let binding = bind_table(&table, input)?;
    let flow = open_gu_flow(&table, binding, input.edge.clone())?;
    let plan = flow.session.plan();
    let gu = plan.gu.as_ref().ok_or(Error::StatementMismatch)?;
    if plan.reader.root != table.reader.root
        || plan.reader.configuration != configuration
        || flow.table_identity != reader_table_identity(&plan.reader)?
        || flow.transcript_after_columns != *flow.session.transcript_value()
    {
        return Err(Error::StatementMismatch);
    }
    let rows = flow.session.pending_gu_rows().to_vec();
    if rows.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let main_len = usize::try_from(
        gu.u.main_rows
            .checked_mul(gu.u.code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mask_len = match gu.u.mask_code {
        Some(code) => usize::try_from(
            gu.u.mask_rows
                .checked_mul(code.ell)
                .ok_or(Error::ResourceLimit)?,
        )
        .map_err(|_| Error::ResourceLimit)?,
        None => 0,
    };
    for row in &rows {
        if row.oracle != gu.oracle
            || row.group != gu.group
            || row.main.len() != main_len
            || row.mask.len() != mask_len
        {
            return Err(Error::StatementMismatch);
        }
    }
    let mut chain = flow.session.chain().clone();
    let x = draw_combine_challenge(
        &mut chain,
        flow.gu_root,
        flow.table_identity,
        plan.reader.root,
        plan.reader.configuration,
        &flow.queries,
        &rows,
    )?;
    let combined = combine_declarations(&rows, x)?;
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
        reader_root: plan.reader.root,
        configuration: plan.reader.configuration,
        shape: plan.reader.shape,
        table_identity: flow.table_identity,
        gu_root: flow.gu_root,
        gu_oracle: gu.oracle,
        gu_group: gu.group,
        gu_tree_number: gu.tree.tree_number,
        queries: flow.queries,
        declarations: rows,
        sample_count,
        record_count,
        column_count,
        unsupported: vec![
            "multi-oracle SEND",
            "real SEND provider",
            "fresh-mask declaration",
        ],
        combined,
        x,
        transcript_after_columns: flow.transcript_after_columns,
        same_session: true,
        column_root_only: true,
        self_reported_gu_root: true,
        fake_send_answer: true,
        bat_decides_truth: false,
        three_c_connected: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

    fn row(kind: GuRelationKind, target: u64) -> GuRelationRow {
        GuRelationRow {
            oracle: 11,
            group: 5,
            kind,
            main: vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3)],
            mask: vec![Fp3::from_u64(4), Fp3::from_u64(5)],
            target: Fp3::from_u64(target),
        }
    }

    #[test]
    fn combination_uses_powers_of_x_and_keeps_kinds_separate() {
        let rows = vec![
            row(GuRelationKind::Sample, 7),
            row(GuRelationKind::Record, 8),
            row(GuRelationKind::Column, 9),
        ];
        let x = Fp3::from_u64(5);
        let combined = combine_declarations(&rows, x).unwrap();
        assert_eq!(combined.count, 3);
        assert_eq!(
            combined.kinds,
            vec![
                GuRelationKind::Sample,
                GuRelationKind::Record,
                GuRelationKind::Column
            ]
        );
        let x2 = x.mul(&x);
        let power_sum = Fp3::ONE.add(&x).add(&x2);
        for index in 0..3 {
            assert_eq!(
                combined.main[index],
                Fp3::from_u64(index as u64 + 1).mul(&power_sum)
            );
        }
        assert_eq!(
            combined.target,
            Fp3::from_u64(7)
                .add(&x.mul(&Fp3::from_u64(8)))
                .add(&x2.mul(&Fp3::from_u64(9)))
        );
    }

    #[test]
    fn a_wrong_declaration_is_still_pending_not_judged() {
        let mut rows = vec![
            row(GuRelationKind::Sample, 7),
            row(GuRelationKind::Record, 8),
        ];
        rows[0].target = Fp3::from_u64(1000);
        let x = Fp3::from_u64(3);
        let combined = combine_declarations(&rows, x).unwrap();
        assert_eq!(
            combined.target,
            Fp3::from_u64(1000).add(&x.mul(&Fp3::from_u64(8)))
        );
        assert_ne!(
            combined.target,
            Fp3::from_u64(7).add(&x.mul(&Fp3::from_u64(8)))
        );
    }

    #[test]
    fn mismatched_group_or_shape_is_rejected() {
        let rows = vec![row(GuRelationKind::Sample, 7)];
        let mut other = row(GuRelationKind::Record, 8);
        other.group = 6;
        assert!(combine_declarations(&[rows[0].clone(), other], Fp3::ONE).is_err());
        let mut other = row(GuRelationKind::Record, 8);
        other.main.pop();
        assert!(combine_declarations(&[rows[0].clone(), other], Fp3::ONE).is_err());
    }

    #[test]
    fn challenge_changes_with_the_declarations() {
        let rows = vec![row(GuRelationKind::Sample, 7)];
        let mut changed = rows.clone();
        changed[0].target = Fp3::from_u64(8);
        let mut first = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            b"gu-combine",
            &[],
            None,
        )
        .unwrap();
        let x1 = draw_combine_challenge(
            &mut first,
            Digest([0x47; 32]),
            Digest([0x48; 32]),
            Digest([0x49; 32]),
            Digest([0x4a; 32]),
            &[0, 1],
            &rows,
        )
        .unwrap();
        let mut second = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            b"gu-combine",
            &[],
            None,
        )
        .unwrap();
        let x2 = draw_combine_challenge(
            &mut second,
            Digest([0x47; 32]),
            Digest([0x48; 32]),
            Digest([0x49; 32]),
            Digest([0x4a; 32]),
            &[0, 1],
            &changed,
        )
        .unwrap();
        assert_ne!(x1, x2);
    }
}
