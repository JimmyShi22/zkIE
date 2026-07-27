//! Generic, model-agnostic DAG of independently-provable "shards" over a
//! `CompiledProgram`, plus a pluggable `Prover` and a
//! commitment-consistency `Linker`. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`.

pub mod model;

pub use model::{build_dag, BuildDagError, Dag, Edge, EdgeKind, Shard, ShardSpec};
