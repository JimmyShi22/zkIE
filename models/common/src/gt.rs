//! Ground-truth loading, tie-break and dual-track comparison.
//!
//! Original float GT and independent integer GT are separate tracks. A blocked
//! track never becomes an empty PASS: it carries its reason.

use std::path::Path;

use zkie_core::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GtStatus {
    Available,
    Blocked(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgmaxGt {
    pub values: Vec<u32>,
    pub status: GtStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionGt {
    pub values: Vec<f64>,
    pub shape: Vec<u64>,
    pub status: GtStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgmaxReport {
    pub matches: u64,
    pub total: u64,
    pub differing_positions: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionMetrics {
    pub count: u64,
    pub mae: f64,
    pub rmse: f64,
    pub max_abs: f64,
    pub all_exact: bool,
}

/// First index of the maximum value; ties take the smallest index.
pub fn first_argmax(values: &[i64]) -> Result<u32> {
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    if values.is_empty() {
        return Err(Error::InvalidConfig);
    }
    u32::try_from(best).map_err(|_| Error::ResourceLimit)
}

pub fn load_argmax_i32(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).map_err(|_| Error::AssetMismatch)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidEncoding);
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = i32::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
        if value < 0 {
            return Err(Error::InvalidEncoding);
        }
        values.push(value as u32);
    }
    Ok(values)
}

pub fn load_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|_| Error::AssetMismatch)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidEncoding);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
        .collect())
}

pub fn compare_argmax(original: &[u32], integer: &[u32]) -> Result<ArgmaxReport> {
    if original.len() != integer.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut differing_positions = Vec::new();
    let mut matches = 0u64;
    for (index, (left, right)) in original.iter().zip(integer).enumerate() {
        if left == right {
            matches += 1;
        } else {
            differing_positions.push(index as u64);
        }
    }
    Ok(ArgmaxReport {
        matches,
        total: original.len() as u64,
        differing_positions,
    })
}

pub fn regression_metrics(original: &[f64], integer: &[f64]) -> Result<RegressionMetrics> {
    if original.len() != integer.len() || original.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum_abs = 0.0f64;
    let mut sum_square = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut all_exact = true;
    for (left, right) in original.iter().zip(integer) {
        let difference = (left - right).abs();
        sum_abs += difference;
        sum_square += difference * difference;
        if difference > max_abs {
            max_abs = difference;
        }
        if difference != 0.0 {
            all_exact = false;
        }
    }
    let count = original.len() as f64;
    Ok(RegressionMetrics {
        count: original.len() as u64,
        mae: sum_abs / count,
        rmse: (sum_square / count).sqrt(),
        max_abs,
        all_exact,
    })
}
