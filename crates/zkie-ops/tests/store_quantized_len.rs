use std::sync::Arc;

use zkie_core::common::quantized_io::{QuantDtype, SafetensorFile};
use zkie_ops::compose::Store;

#[test]
fn quantized_len_is_padded_rows_times_padded_cols() {
    // 3x5 FP8 weight -> Store::len must be 4 * 8 = 32.
    let path = std::env::temp_dir().join(format!(
        "zkie-store-len-{}.safetensors",
        std::process::id()
    ));
    let weight = [0x38u8; 15];
    let scale = [127u8];
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, dtype, shape, bytes) in [
        ("w", "F8_E4M3", vec![3usize, 5usize], weight.as_slice()),
        ("s", "F8_E8M0", vec![1usize, 1usize], scale.as_slice()),
    ] {
        let start = data.len();
        data.extend_from_slice(bytes);
        let end = data.len();
        let mut v = serde_json::Map::new();
        v.insert("dtype".to_string(), serde_json::Value::String(dtype.to_string()));
        v.insert(
            "shape".to_string(),
            serde_json::Value::Array(shape.iter().map(|&x| x.into()).collect()),
        );
        v.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(vec![start.into(), end.into()]),
        );
        header.insert(name.to_string(), serde_json::Value::Object(v));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(&path, &file).unwrap();

    let mut store = Store::new();
    let id = store.push_quantized(
        Arc::new(SafetensorFile::open(&path).unwrap()),
        "w".to_string(),
        "s".to_string(),
        QuantDtype::Fp8E4M3,
        5,
        5,
    );
    assert_eq!(store.len(id), 32);
    assert_eq!(store.materialize(id).len(), 32);
    std::fs::remove_file(path).unwrap();
}
