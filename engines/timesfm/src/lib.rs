//! Demonstrates the generic `zkie_compiler::dag` sharding/proving/linking
//! pipeline against a SYNTHETIC, TimesFM-*shaped* fixture (see
//! `fixtures`'s module docs) -- NOT the real TimesFM model. Compiling the
//! real `models/timesfm_1_0_200m.onnx` graph end-to-end is currently
//! blocked on `zkie-compiler`'s `op_mapper` missing several ops (`Split`,
//! `Sub`, and some prologue-only ops); see
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`
//! for details and follow-up plan.

pub mod fixtures;
pub mod partition;
