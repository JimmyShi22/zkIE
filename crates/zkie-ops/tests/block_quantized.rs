use std::sync::Arc;

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::common::quantized_io::{QuantDtype, SafetensorFile};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

fn write_safetensors() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-block-quantized-{}-{}.safetensors",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    // 4x4 FP8 weight, all 1.0; one E8M0 scale 1.0.
    let weight = [0x38u8; 16];
    let scale = [127u8];
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, dtype, shape, bytes) in [
        ("w", "F8_E4M3", vec![4usize, 4usize], weight.as_slice()),
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
    path
}

#[test]
fn block_quantized_matmul_roundtrip() {
    let path = write_safetensors();
    let file = Arc::new(SafetensorFile::open(&path).unwrap());
    let m = 1usize;
    let k = 4usize;
    let n = 4usize;

    let mut store = Store::new();
    let x = store.push(vec![from_i64(1 << 16); m * k]);
    let w = store.push_quantized(
        file,
        "w".to_string(),
        "s".to_string(),
        QuantDtype::Fp8E4M3,
        k,
        4,
    );
    let c = store.push(vec![]);
    let ops = vec![Op::BlockQuantizedMatMul {
        a: x,
        w,
        c,
        m,
        k,
        n,
        block_size: 4,
    }];
    let mut rng = XorShift64::new(0xAB12);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    std::fs::remove_file(path).unwrap();
}
