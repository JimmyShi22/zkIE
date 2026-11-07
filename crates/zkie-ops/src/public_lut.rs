//! Public lookup tables derived by rule on the verifier side.
//!
//! The tables are still committed as model assets, but a proof no longer opens
//! them: the verifier reconstructs the values from the frozen formulas and
//! checks the reconstructed table against the committed Merkle root. Index
//! domains are fixed; rsqrt is padded to a power of two for shape but its valid
//! index range is never extended.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use zkie_core::error::{Error, Result};

pub const EXP_LEN: u64 = 1 << 21;
pub const GELU_LEN: u64 = 1 << 24;
pub const SILU_LEN: u64 = 1 << 20;
pub const RSQRT_LEN: u64 = 7_340_032;
pub const RSQRT_EXT_LEN: u64 = 14_849_263;

const SCALE: f64 = 65_536.0;

extern "C" {
    fn exp(x: f64) -> f64;
    fn tanh(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

fn libm_exp(x: f64) -> f64 {
    unsafe { exp(x) }
}

fn libm_tanh(x: f64) -> f64 {
    unsafe { tanh(x) }
}

fn libm_pow(x: f64, y: f64) -> f64 {
    unsafe { pow(x, y) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PublicLutKind {
    Exp,
    Gelu,
    Rsqrt,
    RsqrtExtension,
    Silu,
}

impl PublicLutKind {
    pub const fn tag(self) -> i64 {
        match self {
            Self::Exp => 1,
            Self::Gelu => 2,
            Self::Rsqrt => 3,
            Self::RsqrtExtension => 4,
            Self::Silu => 5,
        }
    }

    pub const fn expected_len(self) -> u64 {
        match self {
            Self::Exp => EXP_LEN,
            Self::Gelu => GELU_LEN,
            Self::Rsqrt => RSQRT_LEN,
            Self::RsqrtExtension => RSQRT_EXT_LEN,
            Self::Silu => SILU_LEN,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PublicLutRef {
    pub asset_id: u64,
    pub kind: PublicLutKind,
    pub valid_len: u64,
    pub padded_len: u64,
}

/// Classify a committed table path. Only the frozen public LUT files match;
/// everything else remains an ordinary model asset.
pub fn classify_path(path: &str) -> Option<PublicLutKind> {
    let file = path.rsplit('/').next().unwrap_or(path);
    match file {
        "exp_table_i32.bin" => Some(PublicLutKind::Exp),
        "gelu_table_i32.bin" => Some(PublicLutKind::Gelu),
        "rsqrt_table_i32.bin" => Some(PublicLutKind::Rsqrt),
        "rsqrt_ext_i32.bin" => Some(PublicLutKind::RsqrtExtension),
        "silu_table_i32.bin" => Some(PublicLutKind::Silu),
        _ => None,
    }
}

pub fn padded_len(valid_len: u64) -> u64 {
    valid_len.max(1).next_power_of_two()
}

fn round_half_up(value: f64) -> i64 {
    let rounded = (value + 0.5).floor();
    if rounded > i32::MAX as f64 {
        i32::MAX as i64
    } else if rounded < i32::MIN as f64 {
        i32::MIN as i64
    } else {
        rounded as i64
    }
}

fn exp_value(index: u64) -> i64 {
    let x = (index as f64 - EXP_LEN as f64) / SCALE;
    round_half_up(libm_exp(x) * SCALE)
}

fn gelu_value(index: u64) -> i64 {
    let x = (index as f64 - GELU_LEN as f64 / 2.0) / SCALE;
    let cube = libm_pow(x, 3.0);
    let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * cube);
    let y = 0.5 * x * (1.0 + libm_tanh(inner));
    round_half_up(y * SCALE)
}

fn silu_value(index: u64) -> i64 {
    let x = (index as f64 - SILU_LEN as f64 / 2.0) / SCALE;
    let y = x / (1.0 + libm_exp(-x));
    round_half_up(y * SCALE)
}

const FINE: u64 = 1 << 20;
const COARSE: u64 = 1 << 22;
const MEDIUM_END: u64 = FINE + COARSE;
const RSQRT_EXT_OFFSET: u64 = RSQRT_LEN;

fn rsqrt_raw(global_index: u64) -> u64 {
    if global_index < FINE {
        global_index
    } else if global_index < MEDIUM_END {
        FINE + (global_index - FINE) * 256
    } else {
        FINE + COARSE * 256 + (global_index - MEDIUM_END) * 65_536
    }
}

fn rsqrt_value(global_index: u64) -> i64 {
    let raw = rsqrt_raw(global_index) as f64;
    let value = SCALE / (raw / 16_384.0 + 1e-6).sqrt();
    round_half_up(value)
}

fn generate(kind: PublicLutKind, valid_len: u64) -> Result<Vec<i64>> {
    if valid_len != kind.expected_len() {
        return Err(Error::InvalidConfig);
    }
    let mut values = Vec::with_capacity(valid_len as usize);
    match kind {
        PublicLutKind::Exp => {
            for index in 0..valid_len {
                values.push(exp_value(index));
            }
        }
        PublicLutKind::Gelu => {
            for index in 0..valid_len {
                values.push(gelu_value(index));
            }
        }
        PublicLutKind::Silu => {
            for index in 0..valid_len {
                values.push(silu_value(index));
            }
        }
        PublicLutKind::Rsqrt => {
            for index in 0..valid_len {
                values.push(rsqrt_value(index));
            }
        }
        PublicLutKind::RsqrtExtension => {
            for index in 0..valid_len {
                values.push(rsqrt_value(RSQRT_EXT_OFFSET + index));
            }
        }
    }
    Ok(values)
}

fn cache() -> &'static Mutex<BTreeMap<(PublicLutKind, u64), Arc<Vec<i64>>>> {
    static CACHE: OnceLock<Mutex<BTreeMap<(PublicLutKind, u64), Arc<Vec<i64>>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Generated values for a public LUT, cached per process. The returned vector
/// has exactly `valid_len` entries; padding is a shape rule only and never
/// extends the valid index range.
pub fn values(reference: &PublicLutRef) -> Result<Arc<Vec<i64>>> {
    if reference.valid_len != reference.kind.expected_len()
        || reference.padded_len != padded_len(reference.valid_len)
    {
        return Err(Error::InvalidConfig);
    }
    let key = (reference.kind, reference.valid_len);
    if let Some(values) = cache()
        .lock()
        .map_err(|_| Error::SynchronizationFailed)?
        .get(&key)
        .cloned()
    {
        return Ok(values);
    }
    let generated = Arc::new(generate(reference.kind, reference.valid_len)?);
    let mut guard = cache()
        .lock()
        .map_err(|_| Error::SynchronizationFailed)?;
    if let Some(existing) = guard.get(&key) {
        return Ok(Arc::clone(existing));
    }
    guard.insert(key, Arc::clone(&generated));
    Ok(generated)
}

/// Identity words appended to the relation config before the relation label is
/// derived. The committed Merkle root binds the table before any proof roots.
pub fn identity_words(reference: &PublicLutRef, root: &zkie_core::types::Digest) -> Vec<i64> {
    let mut words = Vec::with_capacity(7);
    words.push(reference.kind.tag());
    words.push(reference.valid_len as i64);
    words.push(reference.padded_len as i64);
    for chunk in root.0.chunks_exact(8) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        words.push(u64::from_le_bytes(bytes) as i64);
    }
    words
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_rule_is_fixed() {
        assert_eq!(padded_len(RSQRT_LEN), 8_388_608);
        assert_eq!(padded_len(EXP_LEN), EXP_LEN);
        assert_eq!(padded_len(GELU_LEN), GELU_LEN);
        assert_eq!(padded_len(SILU_LEN), SILU_LEN);
        assert_eq!(padded_len(RSQRT_EXT_LEN), 16_777_216);
    }

    #[test]
    fn sample_values_match_frozen_assets() {
        assert_eq!(generate(PublicLutKind::Exp, EXP_LEN).unwrap()[0], 0);
        assert_eq!(
            generate(PublicLutKind::Exp, EXP_LEN).unwrap()[EXP_LEN as usize - 1],
            65_535
        );
        let gelu = generate(PublicLutKind::Gelu, GELU_LEN).unwrap();
        assert_eq!(gelu[0], 0);
        assert_eq!(gelu[GELU_LEN as usize / 2], 0);
        assert_eq!(gelu[GELU_LEN as usize - 1], 8_388_607);
        let rsqrt = generate(PublicLutKind::Rsqrt, RSQRT_LEN).unwrap();
        assert_eq!(rsqrt[0], 65_536_000);
        assert_eq!(rsqrt[RSQRT_LEN as usize - 1], 23);
        let silu = generate(PublicLutKind::Silu, SILU_LEN).unwrap();
        assert_eq!(silu[SILU_LEN as usize / 2], 0);
    }
}
