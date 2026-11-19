//! Component tests for the timesfm bounded read path: the frozen rstd table
//! column is filled in bounded chunks and agrees bit for bit with the column
//! derived from the public file, reads outside the table bounds are refused,
//! and parameter assets above the read cap are refused before the file is
//! opened.

use std::path::{Path, PathBuf};

use zkie_core::error::Error;
use zkie_engine::pcs::request::field::Fp3;
use zkie_ops::proof::lin_session::fp3_from_signed_word;
use zkie_ops::proof::lookup::LookupTable;
use zkie_ops::proof::lookup_table::{PublicFileTable, RangeTable, TABLE_READ_CHUNK_ROWS};
use zkie_ops::proof::model_commitment::{read_asset_values_with_root, ModelAssetSpec};

const ROOT: &str = "/data/zkie-assets";
const RSTD_PATH: &str = "models/gpt2/weights/rsqrt_table_i32.bin";

fn rstd_table() -> RangeTable<PublicFileTable> {
    zkie::timesfm_norm_segment_provider::rstd_table_view(Path::new(ROOT))
        .expect("the frozen rstd table opens under the asset root")
}

fn unique_dir(tag: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let dir =
        std::env::temp_dir().join(format!("zkie-timesfm-reads-{tag}-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn write_i32(dir: &Path, path: &str, values: &[i32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let file = dir.join(path);
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).expect("asset dir");
    }
    std::fs::write(file, bytes).expect("asset file");
}

fn asset_spec(path: &str, element_count: u64) -> ModelAssetSpec {
    ModelAssetSpec {
        asset_id: 3,
        path: path.to_string(),
        element_count,
        row_width: 1024,
        encoding: "i32-le".to_string(),
    }
}

/// The chunked read of the frozen value column equals the column derived from
/// the public file (real rows, then first-row padding) bit for bit, and every
/// range read stays within the bounded chunk. The index column carries the
/// row index in the real range and zero in the padding.
#[test]
fn the_chunked_rstd_column_matches_the_public_file() {
    let table = rstd_table();
    let real = table.config().real_len;
    let rows = table.table_rows().unwrap();
    assert_eq!(rows, 1usize << zkie_models_timesfm::norm_profile::RSTD_TABLE_LOG);
    let bytes = std::fs::read(Path::new(ROOT).join(RSTD_PATH)).expect("the public rstd file");
    let values: Vec<i64> = bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i64)
        .collect();
    assert_eq!(values.len(), real);
    let first = fp3_from_signed_word(values[0] as u64);
    let mut expected: Vec<Fp3> = values
        .iter()
        .map(|value| fp3_from_signed_word(*value as u64))
        .collect();
    expected.resize(rows, first);
    let chunked = table.column_rows(1).unwrap();
    assert_eq!(chunked.len(), rows);
    assert_eq!(chunked, expected);
    assert!(table.stats().peak_rows <= TABLE_READ_CHUNK_ROWS as u64);
    let index = table.column_rows(0).unwrap();
    let expected_index: Vec<Fp3> = (0..rows)
        .map(|row| {
            if row < real {
                Fp3::from_u64(row as u64)
            } else {
                Fp3::ZERO
            }
        })
        .collect();
    assert_eq!(index, expected_index);
}

/// Reads at or past the real length, at the padded domain bound and past the
/// element index are refused; the last real row is still readable.
#[test]
fn reads_outside_the_rstd_table_bounds_are_refused() {
    let table = rstd_table();
    let real = table.config().real_len;
    let rows = table.table_rows().unwrap();
    assert!(table.row_value(real - 1).is_ok());
    assert!(table.row_value(real).is_err());
    assert!(table.row_value(rows).is_err());
    assert!(table.row_value(usize::MAX).is_err());
}

/// A parameter asset above the read cap is refused before the file is
/// opened (the missing file would otherwise be an asset mismatch), and a
/// truncated file is refused by the length check.
#[test]
fn parameter_asset_bounds_are_refused() {
    let dir = unique_dir("param");
    let over = asset_spec("params/missing_i32.bin", (1 << 20) / 4 + 1);
    assert!(matches!(
        read_asset_values_with_root(&dir, &over),
        Err(Error::ResourceLimit)
    ));
    write_i32(&dir, "params/short_i32.bin", &[1, 2, 3]);
    let short = asset_spec("params/short_i32.bin", 4);
    assert!(matches!(
        read_asset_values_with_root(&dir, &short),
        Err(Error::AssetMismatch)
    ));
    std::fs::remove_dir_all(&dir).ok();
}
