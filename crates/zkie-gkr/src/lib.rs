//! `zkie-gkr`: a minimal, self-contained prototype of the GKR/sum-check route
//! for proving a single matmul layer.
//!
//! The point of this crate is *not* production proving. It exists to make one
//! architectural claim concrete and measurable: for `C = A @ B`, the proof
//! overhead of a sum-check reduction is `O(m*k + k*n)` (sub-linear in the
//! matmul's `O(m*n*k)` work), whereas the current Halo2 `DotProductChip`
//! expands every output element into `K + 184` Plonkish rows.

pub mod field;
pub mod fixed_point;
pub mod committed;
pub mod gelu;
pub mod lookup;
pub mod matmul;
pub mod mle;
pub mod rmsnorm;
pub mod softmax;
pub mod sumcheck;
pub mod whir;

pub use field::Goldilocks;
