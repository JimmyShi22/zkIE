//! The output-head table's key material: the committed table, its record and
//! the mapping from the authenticated weights.
//!
//! The builder reads the authenticated weight rows through the read-only row
//! path, never the whole asset, and fills one table row at a time. The table
//! entry at row `r`, matrix row `y` and low column `j` is the source weight at
//! matrix row `y`, column `2^k_lo r + j`. Random tails, pads and salts are the
//! caller's private values; the builder checks their sizes and binds them
//! before the table's root is computed.

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::{Fp, E_DEGREE};
use zkie_engine::pcs::request::plan::RequestPlan;
use zkie_engine::pcs::request::table::{
    commit_table, CommittedTable, IndexReader, IndexShape,
};
use zkie_ops::proof::column_session::fp_from_signed_word;

use crate::s7_context::LmHeadContext;

/// The private values of one table: one random tail for each row, one salt for
/// each column and one pad for each answer of each row.
pub struct LmHeadKeyRandomness {
    pub tails: Vec<Vec<Fp>>,
    pub salts: Vec<Vec<u8>>,
    pub pads: Vec<Vec<Fp>>,
}

/// The committed table and its public reader.
pub struct LmHeadKey {
    pub shape: IndexShape,
    pub reader: IndexReader,
    pub table: CommittedTable,
    pub name: Vec<u8>,
}

/// The table entry of one matrix cell, read through the authenticated row path.
pub fn lm_head_entry(
    context: &LmHeadContext,
    plan: &RequestPlan,
    table_row: u64,
    matrix_row: u64,
    low: u64,
) -> Result<Fp> {
    let factor = 1u64
        .checked_shl(plan.lin.k_lo)
        .ok_or(Error::ResourceLimit)?;
    if table_row >= 1u64 << plan.table.v
        || matrix_row >= 1u64 << plan.lin.matrix_rows_log
        || low >= factor
    {
        return Err(Error::InvalidIndex);
    }
    let column = table_row
        .checked_mul(factor)
        .and_then(|value| value.checked_add(low))
        .ok_or(Error::ResourceLimit)?;
    let row = context.read_lm_head_row(matrix_row)?;
    let value = *row
        .get(usize::try_from(column).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ShapeMismatch)?;
    Ok(fp_from_signed_word(value as u64))
}

/// Builds the key material of the output-head table.
///
/// `max_bytes` bounds the derived buffers before any of them is allocated: the
/// messages, the codewords, the tails and the column tree.
pub fn build_lm_head_key(
    context: &LmHeadContext,
    plan: &RequestPlan,
    configuration: Digest,
    randomness: &LmHeadKeyRandomness,
    max_bytes: u64,
) -> Result<LmHeadKey> {
    plan.validate()?;
    if configuration == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let shape = plan.table_shape()?;
    let rows = usize::try_from(shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let factor = 1usize
        .checked_shl(plan.lin.k_lo)
        .ok_or(Error::ResourceLimit)?;
    let columns = 1usize
        .checked_shl(plan.lin.k_col)
        .ok_or(Error::ResourceLimit)?;
    if context.asset.rows != shape.rows()
        || context.asset.row_width < columns as u64
        || context.asset.element_count
            != context.asset.rows.saturating_mul(context.asset.row_width)
    {
        return Err(Error::ShapeMismatch);
    }
    if factor.checked_mul(rows).ok_or(Error::ResourceLimit)? != columns {
        return Err(Error::InvalidConfig);
    }
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let code_len = usize::try_from(shape.code.n()).map_err(|_| Error::ResourceLimit)?;
    let plain = usize::try_from(shape.plain()).map_err(|_| Error::ResourceLimit)?;
    if plain.checked_add(E_DEGREE * shape.answers as usize).ok_or(Error::ResourceLimit)? != ell {
        return Err(Error::InvalidConfig);
    }
    let messages_bytes = (rows as u64)
        .checked_mul(ell as u64)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let codewords_bytes = (rows as u64)
        .checked_mul(code_len as u64)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let tails_bytes = (rows as u64)
        .checked_mul(tail as u64)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let tree_bytes = (code_len as u64)
        .checked_mul((rows as u64).saturating_mul(8).saturating_add(shape.salt_bytes as u64))
        .ok_or(Error::ResourceLimit)?;
    let needed = messages_bytes
        .checked_add(codewords_bytes)
        .and_then(|value| value.checked_add(tails_bytes))
        .and_then(|value| value.checked_add(tree_bytes))
        .ok_or(Error::ResourceLimit)?;
    if needed > max_bytes {
        return Err(Error::ResourceLimit);
    }
    if randomness.pads.len() != rows
        || randomness.pads.iter().any(|row| row.len() != E_DEGREE * shape.answers as usize)
        || randomness.tails.len() != rows
        || randomness.tails.iter().any(|row| row.len() != tail)
        || randomness.salts.len() != code_len
        || randomness
            .salts
            .iter()
            .any(|salt| salt.len() != shape.salt_bytes)
    {
        return Err(Error::ShapeMismatch);
    }

    let mut messages = vec![vec![Fp::ZERO; ell]; rows];
    for row in 0..rows {
        for (offset, pad) in randomness.pads[row].iter().enumerate() {
            messages[row][plain + offset] = *pad;
        }
    }
    for matrix_row in 0..rows {
        let source = context.read_lm_head_row(matrix_row as u64)?;
        if source.len() < columns {
            return Err(Error::ShapeMismatch);
        }
        for table_row in 0..rows {
            let base = table_row
                .checked_mul(factor)
                .ok_or(Error::ResourceLimit)?;
            let destination = &mut messages[table_row];
            for low in 0..factor {
                let column = base.checked_add(low).ok_or(Error::ResourceLimit)?;
                let place = low
                    .checked_add(
                        matrix_row
                            .checked_mul(factor)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::ResourceLimit)?;
                destination[place] =
                    fp_from_signed_word(source[column] as u64);
            }
        }
    }
    let table = commit_table(
        shape,
        configuration,
        &messages,
        &randomness.tails,
        &randomness.salts,
    )?;
    let reader = table.reader.clone();
    let name = table.name.clone();
    Ok(LmHeadKey {
        shape,
        reader,
        table,
        name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    use crate::s7_context::load_gpt2_lm_head_context;
    use crate::provider::StatementData;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    fn artifacts() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
        let root = repo_root();
        let case = root.join("models/gpt2/case.toml");
        let policy = root.join("models/gpt2/case.policy");
        let base = root.join(
            "docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16",
        );
        let public = base.join("commit/public");
        let statement = base.join("proof/statement.bin");
        if case.exists()
            && policy.exists()
            && public.join("model-commitment.bin").exists()
            && statement.exists()
        {
            Some((case, policy, public, statement))
        } else {
            None
        }
    }

    fn context() -> Option<(LmHeadContext, RequestPlan)> {
        let (case, policy, public, statement) = artifacts()?;
        let statement_data: StatementData =
            bincode::deserialize(&std::fs::read(&statement).ok()?).ok()?;
        let context = load_gpt2_lm_head_context(
            &case,
            &policy,
            &public,
            &statement,
            statement_data.model_commitment,
        )
        .ok()?;
        let plan = zkie_models_gpt2::lm_head_plan::lm_head_plan().ok()?;
        let weight = context.asset_root.join(&context.asset.path);
        if !weight.exists() {
            return None;
        }
        Some((context, plan))
    }

    #[test]
    fn the_mapping_matches_sampled_source_entries() {
        let Some((context, plan)) = context() else {
            return;
        };
        for (table_row, matrix_row, low) in [(0, 0, 0), (0, 1023, 63), (1023, 0, 63), (17, 511, 31)] {
            let entry = lm_head_entry(&context, &plan, table_row, matrix_row, low)
                .expect("entry");
            let source = context.read_lm_head_row(matrix_row).expect("row");
            let column = (table_row * (1 << plan.lin.k_lo) + low) as usize;
            assert_eq!(entry, fp_from_signed_word(source[column] as u64));
        }
    }

    #[test]
    fn the_mapping_covers_the_matrix() {
        let Some((context, plan)) = context() else {
            return;
        };
        let rows = 1usize << plan.table.v;
        let factor = 1usize << plan.lin.k_lo;
        let columns = 1usize << plan.lin.k_col;
        assert_eq!(context.asset.rows as usize, rows);
        assert_eq!(rows * factor, columns);
        assert_eq!(context.asset.row_width as usize, columns);
    }

    #[test]
    fn a_wrong_asset_row_count_is_rejected() {
        let Some((mut context, plan)) = context() else {
            return;
        };
        context.asset.rows += 1;
        let randomness = empty_randomness();
        let error = build_lm_head_key(&context, &plan, Digest([0x42; 32]), &randomness, u64::MAX);
        assert!(error.is_err());
    }

    #[test]
    fn randomness_of_the_wrong_size_is_rejected() {
        let Some((context, plan)) = context() else {
            return;
        };
        let randomness = LmHeadKeyRandomness {
            tails: Vec::new(),
            salts: Vec::new(),
            pads: Vec::new(),
        };
        let error = build_lm_head_key(&context, &plan, Digest([0x42; 32]), &randomness, u64::MAX);
        assert!(error.is_err());
    }

    #[test]
    fn the_buffer_bound_is_checked_before_the_build() {
        let Some((context, plan)) = context() else {
            return;
        };
        let randomness = empty_randomness();
        let error = build_lm_head_key(&context, &plan, Digest([0x42; 32]), &randomness, 1);
        assert!(matches!(error, Err(Error::ResourceLimit)));
    }

    fn empty_randomness() -> LmHeadKeyRandomness {
        LmHeadKeyRandomness {
            tails: Vec::new(),
            salts: Vec::new(),
            pads: Vec::new(),
        }
    }
}
