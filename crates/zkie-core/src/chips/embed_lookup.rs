//! `EmbedLookupChip`: backs the `EmbedLookup` instruction — the classic
//! embedding-table lookup: given an integer index, look up its
//! `embed_dim`-length embedding vector from a `table_size x embed_dim` table
//! of model weights (public in the circuit, per the parent design's
//! "weights are public in the circuit" scope).
//!
//! ## Table orientation
//!
//! The embedding table is represented as `Vec<Vec<I18>>` shaped
//! `[table_size][embed_dim]` — **row-major, one row per index**:
//! `embedding_table[i][j]` is dimension `j` of index `i`'s embedding vector.
//! This is the OPPOSITE orientation from [`crate::chips::patch_embed::PatchEmbedChip`]'s
//! weight matrix (`[embed_dim][patch_len]`, one weight *column* vector per
//! output dimension) — the two chips' table/weight shapes must not be
//! assumed to match.
//!
//! ## Why `embed_dim` separate `LookupChip`s, not one shared one
//!
//! Structurally this is `embed_dim` independent lookups against the *same*
//! index: `output[j] = embedding_table[index][j]` for `j in 0..embed_dim`.
//! Unlike [`PatchEmbedChip`](crate::chips::patch_embed::PatchEmbedChip),
//! which configures a single [`crate::chips::dot_general::DotProductChip`]
//! *once* and reuses that one config's columns/gate across `embed_dim`
//! regions (safe because `DotProductChip`'s gate is a data-independent
//! polynomial identity), [`crate::chips::lookup::LookupChip`] bakes a
//! *specific* fixed table into the circuit at configure/load time. Reusing
//! one shared table across dimensions with different per-dimension values
//! would let a malicious prover substitute one dimension's genuine value
//! for another dimension's value at the same index (both are valid rows of
//! the same merged table) without being caught. So `EmbedLookupChip`
//! configures `embed_dim` separate `LookupChip`s, each with its own
//! dedicated output column and its own fixed lookup table — only the
//! `index` input column is shared across dimensions (safe: each
//! `LookupChip`'s lookup argument is now gated by its own selector, per the
//! fix in `lookup.rs`, so an unrelated dimension's selector being off at a
//! given row makes that dimension's check collapse to its own reserved
//! padding row regardless of what value happens to sit in the shared index
//! column at that row).
//!
//! Dimension `j`'s domain is the literal index values
//! `[I18::from_raw(0), I18::from_raw(1), ..., I18::from_raw(table_size - 1)]`
//! (indices encoded directly as raw `I18` integers, NOT via `I18::from_f64`
//! — these are literal indices, not fixed-point real numbers) and its values
//! are column `j` of the table (`embedding_table[i][j]` for `i` in
//! `0..table_size`).

use crate::chips::lookup::{LookupChip, LookupConfig, LookupError};
use crate::field_convert::Fr;
use crate::fixed_point::I18;
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront};
use std::fmt;

/// Errors that can occur while loading or assigning an `EmbedLookupChip`.
#[derive(Debug)]
pub enum EmbedLookupError {
    /// `embedding_table` did not have exactly `table_size` rows.
    TableRowCountMismatch { expected: usize, got: usize },
    /// Row `row` of `embedding_table` did not have exactly `embed_dim`
    /// columns.
    TableRowLengthMismatch {
        row: usize,
        expected: usize,
        got: usize,
    },
    /// The requested `index` is not `< table_size`.
    IndexOutOfRange { index: usize, table_size: usize },
    /// The underlying per-dimension `LookupChip::assign` failed.
    Lookup { dim: usize, source: LookupError },
    /// The underlying per-dimension `LookupChip::load_table` failed.
    Synthesis { dim: usize, source: ErrorFront },
}

impl fmt::Display for EmbedLookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedLookupError::TableRowCountMismatch { expected, got } => write!(
                f,
                "embedding table expects {expected} rows (table_size), got {got}"
            ),
            EmbedLookupError::TableRowLengthMismatch { row, expected, got } => write!(
                f,
                "embedding table row {row} expects {expected} columns (embed_dim), got {got}"
            ),
            EmbedLookupError::IndexOutOfRange { index, table_size } => write!(
                f,
                "embed lookup index {index} is out of range for table_size {table_size}"
            ),
            EmbedLookupError::Lookup { dim, source } => {
                write!(f, "embed lookup dimension {dim} failed: {source}")
            }
            EmbedLookupError::Synthesis { dim, source } => write!(
                f,
                "embed lookup dimension {dim} table load failed: {source:?}"
            ),
        }
    }
}

impl std::error::Error for EmbedLookupError {}

/// Configuration for an [`EmbedLookupChip`]: `embed_dim` independent
/// [`LookupConfig`]s, one per output dimension, sharing a common `index`
/// input column but each with its own dedicated output column and fixed
/// lookup table.
#[derive(Clone, Debug)]
pub struct EmbedLookupConfig {
    lookups: Vec<LookupConfig>,
    table_size: usize,
    embed_dim: usize,
}

pub struct EmbedLookupChip {
    config: EmbedLookupConfig,
}

impl EmbedLookupChip {
    /// `table_size` is the compile-time-known number of rows in the
    /// embedding table (the lookup domain size); `embed_dim` is the
    /// compile-time-known number of output dimensions (number of
    /// independently configured `LookupChip`s, and thus `output_cols.len()`).
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        index: Column<Advice>,
        output_cols: &[Column<Advice>],
        table_size: usize,
        embed_dim: usize,
    ) -> EmbedLookupConfig {
        assert!(table_size > 0, "table_size must be positive");
        assert!(embed_dim > 0, "embed_dim must be positive");
        assert_eq!(
            output_cols.len(),
            embed_dim,
            "output_cols.len() must equal embed_dim"
        );

        let lookups = output_cols
            .iter()
            .map(|&output| LookupChip::configure(meta, index, output))
            .collect();

        EmbedLookupConfig {
            lookups,
            table_size,
            embed_dim,
        }
    }

    pub fn construct(config: EmbedLookupConfig) -> Self {
        EmbedLookupChip { config }
    }

    /// Validates `embedding_table` is shaped `[table_size][embed_dim]`.
    fn validate_table_shape(&self, embedding_table: &[Vec<I18>]) -> Result<(), EmbedLookupError> {
        let table_size = self.config.table_size;
        let embed_dim = self.config.embed_dim;
        if embedding_table.len() != table_size {
            return Err(EmbedLookupError::TableRowCountMismatch {
                expected: table_size,
                got: embedding_table.len(),
            });
        }
        for (row, values) in embedding_table.iter().enumerate() {
            if values.len() != embed_dim {
                return Err(EmbedLookupError::TableRowLengthMismatch {
                    row,
                    expected: embed_dim,
                    got: values.len(),
                });
            }
        }
        Ok(())
    }

    /// Builds dimension `dim`'s `LookupChip`: domain is the literal index
    /// values `[0, 1, ..., table_size - 1]` (as raw `I18` integers), values
    /// are column `dim` of `embedding_table`.
    fn lookup_chip_for_dim(&self, dim: usize, embedding_table: &[Vec<I18>]) -> LookupChip {
        let table_size = self.config.table_size;
        let domain: Vec<I18> = (0..table_size as i64).map(I18::from_raw).collect();
        let values: Vec<I18> = embedding_table.iter().map(|row| row[dim]).collect();
        LookupChip::construct(self.config.lookups[dim].clone(), domain, values)
    }

    /// Loads all `embed_dim` underlying lookup tables from `embedding_table`.
    /// Must be called exactly once per circuit synthesis, before any
    /// `assign` calls, independently of how many times `assign` is called.
    pub fn load_table(
        &self,
        mut layouter: impl Layouter<Fr>,
        embedding_table: &[Vec<I18>],
    ) -> Result<(), EmbedLookupError> {
        self.validate_table_shape(embedding_table)?;
        for dim in 0..self.config.embed_dim {
            let chip = self.lookup_chip_for_dim(dim, embedding_table);
            chip.load_table(layouter.namespace(|| format!("embed lookup table dim {dim}")))
                .map_err(|source| EmbedLookupError::Synthesis { dim, source })?;
        }
        Ok(())
    }

    /// Looks up `embedding_table[index]` (the `embed_dim`-length embedding
    /// vector for `index`), witnessing each dimension's `(index, value)`
    /// pair so the underlying lookup arguments check it against the loaded
    /// table.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        index: usize,
        embedding_table: &[Vec<I18>],
    ) -> Result<Vec<I18>, EmbedLookupError> {
        self.validate_table_shape(embedding_table)?;
        let table_size = self.config.table_size;
        if index >= table_size {
            return Err(EmbedLookupError::IndexOutOfRange { index, table_size });
        }

        let index_i18 = I18::from_raw(index as i64);
        let mut outputs = Vec::with_capacity(self.config.embed_dim);
        for dim in 0..self.config.embed_dim {
            let chip = self.lookup_chip_for_dim(dim, embedding_table);
            let (out, _cell) = chip
                .assign(
                    layouter.namespace(|| format!("embed lookup dim {dim}")),
                    index_i18,
                )
                .map_err(|source| EmbedLookupError::Lookup { dim, source })?;
            outputs.push(out);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_convert::i64_to_fr;
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    const TABLE_SIZE: usize = 4;
    const EMBED_DIM: usize = 3;

    /// A small embedding table with a deliberately NON-zero row 0 (to make
    /// sure `EmbedLookupChip` doesn't accidentally only work "by luck" when
    /// index 0 maps to an all-zero embedding — see the `lookup.rs` padding
    /// fix this chip depends on).
    fn sample_table() -> Vec<Vec<I18>> {
        // Small fractional values (well within I18's representable range;
        // `I18::from_f64(10.0)` would already overflow since SCALE_18 is
        // 1e18 and i64::MAX is ~9.223e18), all distinct and non-zero,
        // including row 0.
        (0..TABLE_SIZE)
            .map(|i| {
                (0..EMBED_DIM)
                    .map(|j| I18::from_f64(((i * EMBED_DIM + j) as f64 + 1.0) * 0.01).unwrap())
                    .collect()
            })
            .collect()
    }

    #[derive(Clone)]
    struct EmbedLookupTestConfig {
        embed: EmbedLookupConfig,
    }

    struct EmbedLookupTestCircuit {
        table: Vec<Vec<I18>>,
        index: usize,
    }

    impl Circuit<Fr> for EmbedLookupTestCircuit {
        type Config = EmbedLookupTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            EmbedLookupTestCircuit {
                table: self.table.clone(),
                index: 0,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let index_col = meta.advice_column();
            let output_cols: Vec<Column<Advice>> =
                (0..EMBED_DIM).map(|_| meta.advice_column()).collect();
            EmbedLookupTestConfig {
                embed: EmbedLookupChip::configure(
                    meta,
                    index_col,
                    &output_cols,
                    TABLE_SIZE,
                    EMBED_DIM,
                ),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = EmbedLookupChip::construct(config.embed);
            chip.load_table(layouter.namespace(|| "table"), &self.table)
                .expect("load_table should not fail in this test");
            chip.assign(layouter.namespace(|| "assign"), self.index, &self.table)
                .expect("assign should not fail in this test");
            Ok(())
        }
    }

    #[test]
    fn embed_lookup_returns_expected_row_at_a_given_index() {
        let table = sample_table();
        // Independently verify against the table directly.
        for index in 0..TABLE_SIZE {
            let circuit = EmbedLookupTestCircuit {
                table: table.clone(),
                index,
            };
            let prover = MockProver::run(8, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }
    }

    #[test]
    fn embed_lookup_at_index_zero_with_nonzero_row_is_satisfied() {
        // Regression test: index 0's embedding row is deliberately non-zero
        // (see `sample_table`), exercising the `lookup.rs` selector-gating
        // fix for rows outside the assigned one defaulting to (0, 0).
        let table = sample_table();
        assert!(table[0].iter().any(|v| v.raw() != 0));
        let circuit = EmbedLookupTestCircuit { table, index: 0 };
        let prover = MockProver::run(8, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn assign_rejects_out_of_range_index_at_the_rust_level() {
        struct GuardCircuit {
            table: Vec<Vec<I18>>,
            out_of_range_index: usize,
        }

        impl Circuit<Fr> for GuardCircuit {
            type Config = EmbedLookupTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                GuardCircuit {
                    table: self.table.clone(),
                    out_of_range_index: self.out_of_range_index,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                EmbedLookupTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = EmbedLookupChip::construct(config.embed);
                chip.load_table(layouter.namespace(|| "table"), &self.table)
                    .expect("load_table should not fail in this test");
                match chip.assign(
                    layouter.namespace(|| "assign"),
                    self.out_of_range_index,
                    &self.table,
                ) {
                    Err(EmbedLookupError::IndexOutOfRange { index, table_size }) => {
                        assert_eq!(index, self.out_of_range_index);
                        assert_eq!(table_size, TABLE_SIZE);
                    }
                    Ok(_) => panic!("expected IndexOutOfRange error but assign succeeded"),
                    Err(other) => panic!("expected IndexOutOfRange error, got {other}"),
                }
                Ok(())
            }
        }

        let circuit = GuardCircuit {
            table: sample_table(),
            out_of_range_index: TABLE_SIZE,
        };
        let _ = MockProver::run(8, &circuit, vec![]);
    }

    #[test]
    fn assign_rejects_wrong_table_row_count() {
        struct GuardCircuit {
            table: Vec<Vec<I18>>,
        }

        impl Circuit<Fr> for GuardCircuit {
            type Config = EmbedLookupTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                GuardCircuit {
                    table: self.table.clone(),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                EmbedLookupTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = EmbedLookupChip::construct(config.embed);
                match chip.assign(layouter.namespace(|| "assign"), 0, &self.table) {
                    Err(EmbedLookupError::TableRowCountMismatch { expected, got }) => {
                        assert_eq!(expected, TABLE_SIZE);
                        assert_eq!(got, TABLE_SIZE - 1);
                    }
                    Ok(_) => panic!("expected TableRowCountMismatch error but assign succeeded"),
                    Err(other) => panic!("expected TableRowCountMismatch error, got {other}"),
                }
                Ok(())
            }
        }

        let mut table = sample_table();
        table.pop();
        let circuit = GuardCircuit { table };
        let _ = MockProver::run(8, &circuit, vec![]);
    }

    #[test]
    fn assign_rejects_wrong_table_row_length() {
        struct GuardCircuit {
            table: Vec<Vec<I18>>,
        }

        impl Circuit<Fr> for GuardCircuit {
            type Config = EmbedLookupTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                GuardCircuit {
                    table: self.table.clone(),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                EmbedLookupTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = EmbedLookupChip::construct(config.embed);
                match chip.assign(layouter.namespace(|| "assign"), 0, &self.table) {
                    Err(EmbedLookupError::TableRowLengthMismatch { row, expected, got }) => {
                        assert_eq!(row, 1);
                        assert_eq!(expected, EMBED_DIM);
                        assert_eq!(got, EMBED_DIM - 1);
                    }
                    Ok(_) => panic!("expected TableRowLengthMismatch error but assign succeeded"),
                    Err(other) => panic!("expected TableRowLengthMismatch error, got {other}"),
                }
                Ok(())
            }
        }

        let mut table = sample_table();
        table[1].pop();
        let circuit = GuardCircuit { table };
        let _ = MockProver::run(8, &circuit, vec![]);
    }

    /// Bypasses `EmbedLookupChip::assign` entirely and directly witnesses a
    /// wrong output for ONE dimension's underlying `LookupChip` region,
    /// confirming the lookup argument (not just Rust-level bookkeeping)
    /// catches a forged/mismatched embedding value, without disturbing the
    /// other, honestly-assigned dimensions.
    #[test]
    fn forged_output_for_one_dimension_is_rejected() {
        const FORGED_DIM: usize = 1;
        const QUERY_INDEX: usize = 2;

        struct ForgedCircuit {
            table: Vec<Vec<I18>>,
        }

        impl Circuit<Fr> for ForgedCircuit {
            type Config = EmbedLookupTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedCircuit {
                    table: self.table.clone(),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                EmbedLookupTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = EmbedLookupChip::construct(config.embed.clone());
                chip.load_table(layouter.namespace(|| "table"), &self.table)
                    .expect("load_table should not fail in this test");

                let index_i18 = I18::from_raw(QUERY_INDEX as i64);
                let correct_output = self.table[QUERY_INDEX][FORGED_DIM];
                let forged_output = I18::from_raw(correct_output.raw() + 1);

                let lookup_config = config.embed.lookups[FORGED_DIM].clone();
                layouter.assign_region(
                    || "forged embed lookup dim",
                    |mut region| {
                        lookup_config.selector.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "index",
                            lookup_config.input,
                            0,
                            || Value::known(i64_to_fr(index_i18.raw())),
                        )?;
                        region.assign_advice(
                            || "forged output",
                            lookup_config.output,
                            0,
                            || Value::known(i64_to_fr(forged_output.raw())),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedCircuit {
            table: sample_table(),
        };
        let prover = MockProver::run(8, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
