//! Loads a model's shard-boundary definition from a human-reviewed static
//! TOML file. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`
//! section 4 for why this is a static file, not a runtime detection
//! algorithm: instruction indices only depend on the model's architecture
//! and the compiler/exporter code, not on trained weight values, so they
//! only need to be regenerated (and re-reviewed) when that code changes --
//! not on every re-run.
//!
//! No real `engines/timesfm/partition.toml` exists yet for the actual
//! TimesFM model -- only the synthetic test fixture at
//! `tests/fixtures/synthetic_partition.toml` exists today (see the
//! `fixtures` module docs for why the real model isn't compiled yet).

use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use zkie_compiler::dag::ShardSpec;

#[derive(Debug, Deserialize)]
struct PartitionFile {
    shard: Vec<ShardDef>,
}

#[derive(Debug, Deserialize)]
struct ShardDef {
    name: String,
    /// Metadata for tooling/readability only (e.g. "this shard is one of
    /// the repeated layers") -- not consumed by `build_dag`.
    #[serde(default)]
    #[allow(dead_code)]
    group: Option<String>,
    start: usize,
    end: usize,
}

#[derive(Debug)]
pub enum PartitionError {
    Io(std::io::Error),
    Toml(toml::de::Error),
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PartitionError::Io(e) => write!(f, "failed to read partition file: {e}"),
            PartitionError::Toml(e) => write!(f, "failed to parse partition file: {e}"),
        }
    }
}

impl std::error::Error for PartitionError {}

impl From<std::io::Error> for PartitionError {
    fn from(e: std::io::Error) -> Self {
        PartitionError::Io(e)
    }
}

impl From<toml::de::Error> for PartitionError {
    fn from(e: toml::de::Error) -> Self {
        PartitionError::Toml(e)
    }
}

/// Loads a `partition.toml`-shaped file into the `ShardSpec`s
/// `zkie_compiler::dag::build_dag` expects, in file order.
pub fn load_partition_file(path: &Path) -> Result<Vec<ShardSpec>, PartitionError> {
    let raw = fs::read_to_string(path)?;
    let parsed: PartitionFile = toml::from_str(&raw)?;
    Ok(parsed
        .shard
        .into_iter()
        .map(|def| ShardSpec {
            name: def.name,
            range: def.start..def.end,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> &'static Path {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/synthetic_partition.toml"
        ))
    }

    fn malformed_fixture_path() -> &'static Path {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/malformed_partition.toml"
        ))
    }

    #[test]
    fn loads_fixture_partition_file() {
        let specs = load_partition_file(fixture_path()).expect("fixture should parse");

        assert_eq!(specs.len(), 5);
        assert_eq!(specs[0].name, "prologue");
        assert_eq!(specs[0].range, 0..3);
        assert_eq!(specs[1].name, "layer_0");
        assert_eq!(specs[1].range, 3..13);
        assert_eq!(specs[4].name, "epilogue");
        assert_eq!(specs[4].range, 33..34);
    }

    #[test]
    fn missing_file_is_a_typed_io_error() {
        let result = load_partition_file(Path::new("/does/not/exist.toml"));
        assert!(matches!(result, Err(PartitionError::Io(_))));
    }

    #[test]
    fn malformed_toml_is_a_typed_parse_error() {
        let result = load_partition_file(malformed_fixture_path());
        assert!(matches!(result, Err(PartitionError::Toml(_))));
    }
}
