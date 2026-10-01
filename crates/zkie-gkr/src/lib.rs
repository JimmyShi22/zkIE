//! `zkie-gkr`: an autotunable inference proving engine.
//!
//! The crate is organised as op-level proof primitives (`compose`, `projection`,
//! `layer_norm_centered`, `softmax_scaled`) built on the shared reduction
//! substrate (`sumcheck`, `matmul`, `logup_gkr`, `same_poly`, `mle`), with
//! `whir` / `committed` for polynomial commitments and `engine` / `autotune`
//! for the shard-granularity + per-stage-backend search.

pub mod field;
pub mod fixed_point;
pub mod mle;
pub mod sumcheck;
pub mod matmul;
pub mod logup_gkr;
pub mod same_poly;
pub mod claim;
pub mod whir;
pub mod batch_open;
pub mod committed;
pub mod par;
pub mod projection;
pub mod layer_norm_centered;
pub mod layernorm_chain;
pub mod softmax_scaled;
pub mod engine;
pub mod compose;

pub use field::Goldilocks;
