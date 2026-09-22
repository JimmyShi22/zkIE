//! `PatchEmbedChip`: backs the `PATCH_EMBED` instruction — TimesFM's
//! "patch → token" linear projection.
//!
//! TimesFM chunks a time series into fixed-length patches and projects each
//! patch (a length-`patch_len` vector of raw values) into an `embed_dim`
//! dimensional token via a matmul against a learned (public, per the v1
//! design's "weights are public in the circuit" scope) weight matrix shaped
//! `patch_len x embed_dim`.
//!
//! Structurally this is just `DOT_GENERAL` applied `embed_dim` times: the
//! same `patch_len`-length patch vector is dotted against `embed_dim`
//! different weight columns, one per output embedding dimension:
//! `output[j] = dot_product(patch, weight_column_j)` for `j in 0..embed_dim`.
//!
//! Implementation approach: a single `DotProductChip` is configured once
//! (`K = patch_len`) and its `assign` is invoked `embed_dim` times against
//! fresh, sequential layouter regions — one region per output dimension —
//! reusing the same columns rather than allocating `embed_dim` independent
//! sets of columns.

use crate::chips::dot_general::{DotProductChip, DotProductConfig, DotProductError};
use crate::field_convert::Fr;
use crate::fixed_point::I18;
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem};
use std::fmt;

/// Errors that can occur while assigning a `PatchEmbedChip` region.
#[derive(Debug)]
pub enum PatchEmbedError {
    /// `patch` did not have exactly the configured `patch_len` elements.
    PatchLengthMismatch { expected: usize, got: usize },
    /// `weights` did not have exactly the configured `embed_dim` columns.
    WeightCountMismatch { expected: usize, got: usize },
    /// The weight column for output dimension `dim` did not have exactly
    /// `patch_len` elements.
    WeightLengthMismatch {
        dim: usize,
        expected: usize,
        got: usize,
    },
    /// The underlying per-dimension `DotProductChip::assign` failed (raw
    /// overflow or a halo2 circuit-synthesis error).
    DotProduct { dim: usize, source: DotProductError },
}

impl fmt::Display for PatchEmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatchEmbedError::PatchLengthMismatch { expected, got } => write!(
                f,
                "patch embed expects a patch of length {expected}, got {got}"
            ),
            PatchEmbedError::WeightCountMismatch { expected, got } => write!(
                f,
                "patch embed expects {expected} weight columns (embed_dim), got {got}"
            ),
            PatchEmbedError::WeightLengthMismatch { dim, expected, got } => write!(
                f,
                "patch embed weight column {dim} expects length {expected}, got {got}"
            ),
            PatchEmbedError::DotProduct { dim, source } => {
                write!(f, "patch embed dot product for dim {dim} failed: {source}")
            }
        }
    }
}

impl std::error::Error for PatchEmbedError {}

#[derive(Clone, Debug)]
pub struct PatchEmbedConfig {
    dot: DotProductConfig,
    patch_len: usize,
    embed_dim: usize,
}

pub struct PatchEmbedChip {
    config: PatchEmbedConfig,
}

impl PatchEmbedChip {
    /// `patch_len` is the compile-time-known patch length (dot product `K`);
    /// `embed_dim` is the compile-time-known number of output embedding
    /// dimensions (number of sequential dot-product regions). Both are fixed
    /// per configured circuit, matching `DotProductChip::configure`'s `k`.
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        a: Column<Advice>,
        b: Column<Advice>,
        accumulator: Column<Advice>,
        q: Column<Advice>,
        r: Column<Advice>,
        slack: Column<Advice>,
        bits: Column<Advice>,
        patch_len: usize,
        embed_dim: usize,
    ) -> PatchEmbedConfig {
        assert!(embed_dim > 0, "embed_dim must be positive");
        let dot = DotProductChip::configure(meta, a, b, accumulator, q, r, slack, bits, patch_len);
        PatchEmbedConfig {
            dot,
            patch_len,
            embed_dim,
        }
    }

    pub fn construct(config: PatchEmbedConfig) -> Self {
        PatchEmbedChip { config }
    }

    /// Assigns `embed_dim` sequential dot-product regions projecting `patch`
    /// (length `patch_len`) through `weights` (`embed_dim` columns, each of
    /// length `patch_len` — i.e. `weights[j]` is the weight column for output
    /// dimension `j`), returning the length-`embed_dim` requantized I18
    /// output vector.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        patch: &[I18],
        weights: &[Vec<I18>],
    ) -> Result<Vec<I18>, PatchEmbedError> {
        let patch_len = self.config.patch_len;
        let embed_dim = self.config.embed_dim;

        if patch.len() != patch_len {
            return Err(PatchEmbedError::PatchLengthMismatch {
                expected: patch_len,
                got: patch.len(),
            });
        }
        if weights.len() != embed_dim {
            return Err(PatchEmbedError::WeightCountMismatch {
                expected: embed_dim,
                got: weights.len(),
            });
        }
        for (dim, weight_col) in weights.iter().enumerate() {
            if weight_col.len() != patch_len {
                return Err(PatchEmbedError::WeightLengthMismatch {
                    dim,
                    expected: patch_len,
                    got: weight_col.len(),
                });
            }
        }

        let mut outputs = Vec::with_capacity(embed_dim);
        for (dim, weight_col) in weights.iter().enumerate() {
            let dot_chip = DotProductChip::construct(self.config.dot.clone());
            let out = dot_chip
                .assign(
                    layouter.namespace(|| format!("patch embed dim {dim}")),
                    patch.to_vec(),
                    weight_col.clone(),
                )
                .map_err(|source| PatchEmbedError::DotProduct { dim, source })?;
            outputs.push(out);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness};
    use crate::fixed_point::{requantize_raw, SCALE_18};
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    const PATCH_LEN: usize = 3;
    const EMBED_DIM: usize = 2;

    #[derive(Clone)]
    struct PatchEmbedTestConfig {
        embed: PatchEmbedConfig,
    }

    struct PatchEmbedTestCircuit {
        patch: Vec<I18>,
        weights: Vec<Vec<I18>>,
    }

    impl Circuit<Fr> for PatchEmbedTestCircuit {
        type Config = PatchEmbedTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            PatchEmbedTestCircuit {
                patch: vec![I18::from_raw(0); PATCH_LEN],
                weights: vec![vec![I18::from_raw(0); PATCH_LEN]; EMBED_DIM],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let accumulator = meta.advice_column();
            let q = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let bits = meta.advice_column();
            PatchEmbedTestConfig {
                embed: PatchEmbedChip::configure(
                    meta,
                    a,
                    b,
                    accumulator,
                    q,
                    r,
                    slack,
                    bits,
                    PATCH_LEN,
                    EMBED_DIM,
                ),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = PatchEmbedChip::construct(config.embed);
            chip.assign(layouter, &self.patch, &self.weights)
                .map(|_| ())
                .map_err(|e| panic!("patch embed assign failed: {e}"))
        }
    }

    /// Independently computes `dot_product(patch, weight_col)` the same way
    /// `DotProductChip` does: raw Q36 accumulation, requantized once.
    fn expected_dot(patch: &[I18], weight_col: &[I18]) -> I18 {
        let raw_sum: i128 = patch
            .iter()
            .zip(weight_col.iter())
            .map(|(x, y)| (x.raw() as i128) * (y.raw() as i128))
            .sum();
        requantize_raw(raw_sum).unwrap().0
    }

    #[test]
    fn patch_embed_projects_patch_through_weight_matrix() {
        let patch = vec![
            I18::from_f64(1.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
        ];
        // weight column 0: [1, 0, 0] -> selects patch[0]
        // weight column 1: [0.5, 0.5, 1.0] -> 0.5*1 + 0.5*2 + 1*(-1.5) = -0.0
        let weights = vec![
            vec![
                I18::from_f64(1.0).unwrap(),
                I18::from_f64(0.0).unwrap(),
                I18::from_f64(0.0).unwrap(),
            ],
            vec![
                I18::from_f64(0.5).unwrap(),
                I18::from_f64(0.5).unwrap(),
                I18::from_f64(1.0).unwrap(),
            ],
        ];

        let expected: Vec<I18> = weights.iter().map(|w| expected_dot(&patch, w)).collect();

        let circuit = PatchEmbedTestCircuit {
            patch: patch.clone(),
            weights: weights.clone(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        assert!((expected[0].to_f64() - 1.0).abs() < 1e-9);
        assert!((expected[1].to_f64() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn patch_embed_all_zero_is_satisfied() {
        let patch = vec![I18::from_raw(0); PATCH_LEN];
        let weights = vec![vec![I18::from_raw(0); PATCH_LEN]; EMBED_DIM];
        let circuit = PatchEmbedTestCircuit { patch, weights };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn assign_rejects_wrong_patch_length() {
        struct LenTestCircuit {
            patch: Vec<I18>,
            weights: Vec<Vec<I18>>,
        }

        impl Circuit<Fr> for LenTestCircuit {
            type Config = PatchEmbedTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                LenTestCircuit {
                    patch: vec![I18::from_raw(0); PATCH_LEN],
                    weights: vec![vec![I18::from_raw(0); PATCH_LEN]; EMBED_DIM],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                PatchEmbedTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = PatchEmbedChip::construct(config.embed);
                match chip.assign(layouter, &self.patch, &self.weights) {
                    Err(PatchEmbedError::PatchLengthMismatch { expected, got }) => {
                        assert_eq!(expected, PATCH_LEN);
                        assert_eq!(got, PATCH_LEN - 1);
                    }
                    Ok(_) => panic!("expected PatchLengthMismatch error but assign succeeded"),
                    Err(other) => panic!("expected PatchLengthMismatch error, got {other}"),
                }
                Ok(())
            }
        }

        let circuit = LenTestCircuit {
            patch: vec![I18::from_raw(1); PATCH_LEN - 1],
            weights: vec![vec![I18::from_raw(1); PATCH_LEN]; EMBED_DIM],
        };
        let _ = MockProver::run(10, &circuit, vec![]);
    }

    #[test]
    fn assign_rejects_wrong_weight_count() {
        struct LenTestCircuit {
            patch: Vec<I18>,
            weights: Vec<Vec<I18>>,
        }

        impl Circuit<Fr> for LenTestCircuit {
            type Config = PatchEmbedTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                LenTestCircuit {
                    patch: vec![I18::from_raw(0); PATCH_LEN],
                    weights: vec![vec![I18::from_raw(0); PATCH_LEN]; EMBED_DIM],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                PatchEmbedTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = PatchEmbedChip::construct(config.embed);
                match chip.assign(layouter, &self.patch, &self.weights) {
                    Err(PatchEmbedError::WeightCountMismatch { expected, got }) => {
                        assert_eq!(expected, EMBED_DIM);
                        assert_eq!(got, EMBED_DIM - 1);
                    }
                    Ok(_) => panic!("expected WeightCountMismatch error but assign succeeded"),
                    Err(other) => panic!("expected WeightCountMismatch error, got {other}"),
                }
                Ok(())
            }
        }

        let circuit = LenTestCircuit {
            patch: vec![I18::from_raw(1); PATCH_LEN],
            weights: vec![vec![I18::from_raw(1); PATCH_LEN]; EMBED_DIM - 1],
        };
        let _ = MockProver::run(10, &circuit, vec![]);
    }

    #[test]
    fn assign_rejects_wrong_weight_column_length() {
        struct LenTestCircuit {
            patch: Vec<I18>,
            weights: Vec<Vec<I18>>,
        }

        impl Circuit<Fr> for LenTestCircuit {
            type Config = PatchEmbedTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                LenTestCircuit {
                    patch: vec![I18::from_raw(0); PATCH_LEN],
                    weights: vec![vec![I18::from_raw(0); PATCH_LEN]; EMBED_DIM],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                PatchEmbedTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = PatchEmbedChip::construct(config.embed);
                match chip.assign(layouter, &self.patch, &self.weights) {
                    Err(PatchEmbedError::WeightLengthMismatch { dim, expected, got }) => {
                        assert_eq!(dim, 1);
                        assert_eq!(expected, PATCH_LEN);
                        assert_eq!(got, PATCH_LEN - 1);
                    }
                    Ok(_) => panic!("expected WeightLengthMismatch error but assign succeeded"),
                    Err(other) => panic!("expected WeightLengthMismatch error, got {other}"),
                }
                Ok(())
            }
        }

        let mut weights = vec![vec![I18::from_raw(1); PATCH_LEN]; EMBED_DIM];
        weights[1] = vec![I18::from_raw(1); PATCH_LEN - 1];
        let circuit = LenTestCircuit {
            patch: vec![I18::from_raw(1); PATCH_LEN],
            weights,
        };
        let _ = MockProver::run(10, &circuit, vec![]);
    }

    /// Forges the final-row quotient witness for one output dimension's
    /// underlying dot-product region (bypassing `PatchEmbedChip::assign`
    /// entirely), mirroring `dot_general`'s own
    /// `dot_product_with_forged_final_quotient_is_rejected` test. This
    /// confirms that corrupting a single embedding dimension among several
    /// is caught, not just a lone dot product.
    #[test]
    fn forged_output_for_one_dimension_is_rejected() {
        const FORGED_DIM: usize = 1;

        struct ForgedPatchEmbedCircuit {
            patch: Vec<I18>,
            weights: Vec<Vec<I18>>,
        }

        impl Circuit<Fr> for ForgedPatchEmbedCircuit {
            type Config = PatchEmbedTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedPatchEmbedCircuit {
                    patch: vec![I18::from_raw(0); PATCH_LEN],
                    weights: vec![vec![I18::from_raw(0); PATCH_LEN]; EMBED_DIM],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                PatchEmbedTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let dot = &config.embed;
                let patch = &self.patch;

                for (dim, weight_col) in self.weights.iter().enumerate() {
                    let raw_sum: i128 = patch
                        .iter()
                        .zip(weight_col.iter())
                        .map(|(x, y)| (x.raw() as i128) * (y.raw() as i128))
                        .sum();
                    let mut partial_sums = Vec::with_capacity(PATCH_LEN);
                    let mut acc = 0i128;
                    for i in 0..PATCH_LEN {
                        acc += (patch[i].raw() as i128) * (weight_col[i].raw() as i128);
                        partial_sums.push(acc);
                    }
                    let (q, r) = requantize_raw(raw_sum).unwrap();
                    // Only forge the quotient for FORGED_DIM; every other
                    // dimension is assigned honestly.
                    let forged_q = if dim == FORGED_DIM {
                        q.raw() + 1
                    } else {
                        q.raw()
                    };
                    let slack = SCALE_18 - 1 - r;

                    layouter.assign_region(
                        || format!("patch embed forged dim {dim}"),
                        |mut region| {
                            for i in 0..PATCH_LEN {
                                region.assign_advice(
                                    || format!("a_{i}"),
                                    dot.dot.a,
                                    i,
                                    || Value::known(i64_to_fr(patch[i].raw())),
                                )?;
                                region.assign_advice(
                                    || format!("b_{i}"),
                                    dot.dot.b,
                                    i,
                                    || Value::known(i64_to_fr(weight_col[i].raw())),
                                )?;
                                region.assign_advice(
                                    || format!("accumulator_{i}"),
                                    dot.dot.accumulator,
                                    i,
                                    || Value::known(i128_to_fr(partial_sums[i])),
                                )?;
                                if i == 0 {
                                    dot.dot.s_acc_start.enable(&mut region, i)?;
                                } else {
                                    dot.dot.s_acc_step.enable(&mut region, i)?;
                                }
                            }
                            let last = PATCH_LEN - 1;
                            dot.dot.s_final.enable(&mut region, last)?;
                            dot.dot.s_slack.enable(&mut region, last)?;
                            let (forged_q_shift_fr, _) = shifted_i64_witness(forged_q);
                            region.assign_advice(|| "q", dot.dot.q, last, || forged_q_shift_fr)?;
                            region.assign_advice(
                                || "r",
                                dot.dot.r,
                                last,
                                || Value::known(i128_to_fr(r)),
                            )?;
                            region.assign_advice(
                                || "slack",
                                dot.dot.slack,
                                last,
                                || Value::known(i128_to_fr(slack)),
                            )
                        },
                    )?;
                }
                Ok(())
            }
        }

        let circuit = ForgedPatchEmbedCircuit {
            patch: vec![
                I18::from_f64(1.0).unwrap(),
                I18::from_f64(2.0).unwrap(),
                I18::from_f64(-1.5).unwrap(),
            ],
            weights: vec![
                vec![
                    I18::from_f64(1.0).unwrap(),
                    I18::from_f64(0.0).unwrap(),
                    I18::from_f64(0.0).unwrap(),
                ],
                vec![
                    I18::from_f64(0.5).unwrap(),
                    I18::from_f64(0.5).unwrap(),
                    I18::from_f64(1.0).unwrap(),
                ],
            ],
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
