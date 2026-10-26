//! Minimal streaming reader for the quantized safetensors used by
//! DeepSeek-V4.1-Flash.
//!
//! The HuggingFace checkpoint stores most weights in the native inference
//! stack's formats: non-expert weights are FP8 E4M3 with E8M0 block scales
//! (block 32x32), routed expert weights are packed FP4 E2M1 with E8M0 scales,
//! and the remaining small tensors are BF16/F32. We never decode the whole
//! file: the header is parsed once, and each tensor is materialized on demand
//! from a memory-mapped view.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use serde::Deserialize;

use crate::common::field::{Goldilocks, PrimeCharacteristicRing};
use crate::common::fixed_point::from_i64;

const FIXED_SCALE: f64 = 65536.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantDtype {
    Fp8E4M3,
    Fp8E8M0,
    Int8,
    F32,
    Bf16,
}

impl QuantDtype {
    fn from_safetensors(s: &str) -> Option<Self> {
        match s {
            "F8_E4M3" => Some(Self::Fp8E4M3),
            "F8_E8M0" => Some(Self::Fp8E8M0),
            "I8" => Some(Self::Int8),
            "F32" => Some(Self::F32),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: QuantDtype,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// One opened safetensors shard. `mmap` keeps the file alive; tensor metadata
/// is available without reading any tensor bytes.
#[derive(Clone)]
pub struct SafetensorFile {
    mmap: Arc<Mmap>,
    tensors: Arc<HashMap<String, TensorInfo>>,
    data_base: usize,
}

impl SafetensorFile {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "safetensors file shorter than 8-byte header length",
            ));
        }
        let header_len = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "header overflow"))?;
        if header_end > mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "safetensors header extends past end of file",
            ));
        }
        let raw: HashMap<String, RawTensorInfo> =
            serde_json::from_slice(&mmap[8..header_end]).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
            })?;
        let data_base = header_end;
        let mut tensors = HashMap::with_capacity(raw.len());
        for (name, info) in raw {
            let dtype = QuantDtype::from_safetensors(&info.dtype).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unsupported safetensors dtype {}", info.dtype),
                )
            })?;
            if info.data_offsets[1] < info.data_offsets[0]
                || info.data_offsets[1] > mmap.len()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid data offsets for {name}"),
                ));
            }
            tensors.insert(
                name,
                TensorInfo {
                    dtype,
                    shape: info.shape,
                    data_offsets: info.data_offsets,
                },
            );
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            tensors: Arc::new(tensors),
            data_base,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn raw(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        Some(
            &self.mmap[self.data_base + info.data_offsets[0]..self.data_base + info.data_offsets[1]],
        )
    }

    pub fn read_f32(&self, name: &str) -> Vec<Goldilocks> {
        let raw = self.raw(name).expect("missing tensor");
        raw.chunks_exact(4)
            .map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                from_i64((v as f64 * FIXED_SCALE).round() as i64)
            })
            .collect()
    }

    pub fn read_bf16(&self, name: &str) -> Vec<Goldilocks> {
        let raw = self.raw(name).expect("missing tensor");
        raw.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                let v = bf16_to_f32(bits) as f64;
                from_i64((v * FIXED_SCALE).round() as i64)
            })
            .collect()
    }

    /// Dequantize an FP8 E4M3 block weight. `scale_name` is the matching E8M0
    /// scale tensor. `block_size` is 32 for these checkpoints.
    pub fn read_fp8_weight(
        &self,
        name: &str,
        scale_name: &str,
        block_size: usize,
    ) -> Vec<Goldilocks> {
        let info = self.tensor(name).expect("missing fp8 weight");
        let scale_info = self.tensor(scale_name).expect("missing fp8 scale");
        let raw = self.raw(name).expect("missing fp8 weight");
        let scale_raw = self.raw(scale_name).expect("missing fp8 scale");
        assert_eq!(info.shape.len(), 2, "fp8 weight must be 2D");
        assert_eq!(scale_info.shape.len(), 2, "fp8 scale must be 2D");
        let rows = info.shape[0];
        let cols = info.shape[1];
        let padded_rows = rows.next_power_of_two();
        let padded_cols = cols.next_power_of_two();
        let nb_cols = scale_info.shape[1];
        let mut out = vec![Goldilocks::ZERO; padded_rows * padded_cols];
        for r in 0..rows {
            let sr = r / block_size;
            for c in 0..cols {
                let sc = c / block_size;
                let scale = e8m0_to_f64(scale_raw[sr * nb_cols + sc]);
                let w = e4m3_to_f64(raw[r * cols + c]);
                out[r * padded_cols + c] =
                    from_i64((w * scale * FIXED_SCALE).round() as i64);
            }
        }
        out
    }

    /// Dequantize a packed FP4 E2M1 block weight. The safetensors shape's
    /// second dimension is the packed byte width (`logical_cols / 2`).
    pub fn read_fp4_weight(
        &self,
        name: &str,
        scale_name: &str,
        logical_cols: usize,
        block_size: usize,
    ) -> Vec<Goldilocks> {
        let info = self.tensor(name).expect("missing fp4 weight");
        let scale_info = self.tensor(scale_name).expect("missing fp4 scale");
        let raw = self.raw(name).expect("missing fp4 weight");
        let scale_raw = self.raw(scale_name).expect("missing fp4 scale");
        assert_eq!(info.shape.len(), 2, "fp4 weight must be 2D");
        assert_eq!(info.shape[1] * 2, logical_cols, "fp4 packed width mismatch");
        let rows = info.shape[0];
        let packed_cols = info.shape[1];
        let padded_rows = rows.next_power_of_two();
        let padded_cols = logical_cols.next_power_of_two();
        let nb_cols = scale_info.shape[1];
        assert_eq!(logical_cols / block_size, nb_cols, "fp4 scale block count mismatch");
        let mut out = vec![Goldilocks::ZERO; padded_rows * padded_cols];
        for r in 0..rows {
            for c in 0..logical_cols {
                let sc = c / block_size;
                // FP4 E8M0 scale is block-quantized over columns only; its row
                // dimension is the full weight row count (shape [rows, cols/block]).
                let scale = e8m0_to_f64(scale_raw[r * nb_cols + sc]);
                let byte = raw[r * packed_cols + c / 2];
                let nibble = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let w = e2m1_to_f64(nibble);
                out[r * padded_cols + c] =
                    from_i64((w * scale * FIXED_SCALE).round() as i64);
            }
        }
        out
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn e8m0_to_f64(code: u8) -> f64 {
    2f64.powi(code as i32 - 127)
}

pub fn e4m3_to_f64(code: u8) -> f64 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = ((code >> 3) & 0x0F) as i32;
    let mantissa = (code & 0x07) as f64;
    let value = if exponent == 0 {
        mantissa * 2f64.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    };
    sign * value
}

pub fn e2m1_to_f64(code: u8) -> f64 {
    const TABLE: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    TABLE[(code & 0x0F) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::fixed_point::to_i64;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn write_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> std::path::PathBuf {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zkie-safetensors-{}-{}.safetensors",
            std::process::id(),
            seq
        ));
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            let end = data.len();
            let mut v = serde_json::Map::new();
            v.insert("dtype".to_string(), serde_json::Value::String((*dtype).to_string()));
            v.insert(
                "shape".to_string(),
                serde_json::Value::Array(shape.iter().map(|&s| s.into()).collect()),
            );
            v.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![start.into(), end.into()]),
            );
            header.insert((*name).to_string(), serde_json::Value::Object(v));
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
    fn reads_bf16_and_f32() {
        let f32_1 = 1.5f32.to_le_bytes();
        let bf16_1 = (1.5f32.to_bits() >> 16) as u16;
        let path = write_safetensors(&[
            ("f32", "F32", &[1], &f32_1),
            ("bf16", "BF16", &[1], &bf16_1.to_le_bytes()),
        ]);
        let file = SafetensorFile::open(&path).unwrap();
        assert_eq!(to_i64(file.read_f32("f32")[0]), 98304);
        assert_eq!(to_i64(file.read_bf16("bf16")[0]), 98304);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dequantizes_fp8_block_weight() {
        // E4M3 codes: 0x38 = 1.0, 0x3C = 1.5, 0x40 = 2.0, 0x44 = 3.0.
        let weight = [0x38u8, 0x3C, 0x40, 0x44];
        // E8M0 code 127 = scale 1.0.
        let scale = [127u8];
        let path = write_safetensors(&[
            ("w", "F8_E4M3", &[1, 4], &weight),
            ("s", "F8_E8M0", &[1, 1], &scale),
        ]);
        let file = SafetensorFile::open(&path).unwrap();
        let out = file.read_fp8_weight("w", "s", 4);
        let got: Vec<i64> = out.iter().map(|&x| to_i64(x)).collect();
        assert_eq!(got, vec![65536, 98304, 131072, 196608]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dequantizes_packed_fp4_low_then_high() {
        // Byte 0x10 -> low=0, high=1 (0, 0.5).
        // Byte 0x1F -> low=15, high=1 (-6, 0.5).
        let weight = [0x10u8, 0x1F];
        let scale = [127u8];
        let path = write_safetensors(&[
            ("w", "I8", &[1, 2], &weight),
            ("s", "F8_E8M0", &[1, 1], &scale),
        ]);
        let file = SafetensorFile::open(&path).unwrap();
        let out = file.read_fp4_weight("w", "s", 4, 4);
        let got: Vec<i64> = out.iter().map(|&x| to_i64(x)).collect();
        assert_eq!(got, vec![0, 32768, -393216, 32768]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fp4_scale_rows_are_not_blocked() {
        // FP4 E8M0 scale is [rows, cols/block]: rows are NOT blocked, unlike
        // FP8's [rows/block, cols/block]. Every nibble is 0.5; the per-column
        // block scale differs per row, so the wrong row-blocking would reuse
        // row 0's scale for row 1.
        let weight = [0x11u8, 0x11, 0x11, 0x11]; // 2 rows x 2 packed cols
        let scale = [127u8, 126, 125, 124]; // [2,2]: 1.0, 0.5, 0.25, 0.125
        let path = write_safetensors(&[
            ("w", "I8", &[2, 2], &weight),
            ("s", "F8_E8M0", &[2, 2], &scale),
        ]);
        let file = SafetensorFile::open(&path).unwrap();
        let out = file.read_fp4_weight("w", "s", 4, 2);
        let got: Vec<i64> = out.iter().map(|&x| to_i64(x)).collect();
        // row0: 0.5*1.0, 0.5*1.0, 0.5*0.5, 0.5*0.5
        // row1: 0.5*0.25, 0.5*0.25, 0.5*0.125, 0.5*0.125
        assert_eq!(got, vec![32768, 32768, 16384, 16384, 8192, 8192, 4096, 4096]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn pads_fp8_rows_and_cols_independently() {
        // 3x5 weight, all 1.0. Padded rows=4, cols=8 => 32 elements, with the
        // tail of each row zero and the trailing row all zero.
        let weight = [0x38u8; 15];
        let scale = [127u8];
        let path = write_safetensors(&[
            ("w", "F8_E4M3", &[3, 5], &weight),
            ("s", "F8_E8M0", &[1, 1], &scale),
        ]);
        let file = SafetensorFile::open(&path).unwrap();
        let out = file.read_fp8_weight("w", "s", 5);
        assert_eq!(out.len(), 32);
        let got: Vec<i64> = out.iter().map(|&x| to_i64(x)).collect();
        for r in 0..3 {
            for c in 0..5 {
                assert_eq!(got[r * 8 + c], 65536);
            }
            for c in 5..8 {
                assert_eq!(got[r * 8 + c], 0);
            }
        }
        for c in 0..8 {
            assert_eq!(got[3 * 8 + c], 0);
        }
        std::fs::remove_file(path).unwrap();
    }
}
