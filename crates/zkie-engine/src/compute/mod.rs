//! Exact CPU kernels shared by forward execution and proof generation.
//!
//! Storage sources are read in explicit tiles through [`TensorReader`]; there
//! is no whole-file upload path. The kernels themselves operate on exact
//! integers or on the protocol fields.

pub mod cpu;

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::storage::TensorReader;
use zkie_core::types::DType;

use crate::pcs::request::field::{Fp, Fp3, MODULUS};

pub const DEFAULT_TILE_ELEMENTS: usize = 4096;

fn is_i64(dtype: DType) -> bool {
    matches!(dtype, DType::I64)
}

fn is_base_field(dtype: DType) -> bool {
    matches!(dtype, DType::BaseField(_))
}

fn is_extension_field(dtype: DType) -> bool {
    matches!(dtype, DType::ExtensionField(descriptor) if descriptor.extension_degree == 3)
}

fn tile_count(total: usize, tile_elements: usize) -> Result<usize> {
    if tile_elements == 0 {
        return Err(Error::InvalidConfig);
    }
    Ok(total.div_ceil(tile_elements))
}

/// Read an integer range into a caller-provided tile buffer.
pub fn read_i64_range(
    reader: &dyn TensorReader,
    start: u64,
    count: usize,
    dst: &mut [i64],
) -> Result<()> {
    if !is_i64(reader.spec().dtype) {
        return Err(Error::UnsupportedPrimitive);
    }
    if dst.len() != count {
        return Err(Error::ShapeMismatch);
    }
    let byte_len = count.checked_mul(8).ok_or(Error::ResourceLimit)?;
    let mut bytes = vec![0u8; byte_len];
    reader.read_range(start, count, &mut bytes)?;
    for (index, chunk) in bytes.chunks_exact(8).enumerate() {
        dst[index] = i64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
    }
    Ok(())
}

/// Read a base-field range. Non-canonical words are rejected, never reduced.
pub fn read_fp_range(
    reader: &dyn TensorReader,
    start: u64,
    count: usize,
    dst: &mut [Fp],
) -> Result<()> {
    if !is_base_field(reader.spec().dtype) {
        return Err(Error::UnsupportedPrimitive);
    }
    if dst.len() != count {
        return Err(Error::ShapeMismatch);
    }
    let byte_len = count.checked_mul(8).ok_or(Error::ResourceLimit)?;
    let mut bytes = vec![0u8; byte_len];
    reader.read_range(start, count, &mut bytes)?;
    for (index, chunk) in bytes.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
        if word >= MODULUS {
            return Err(Error::InvalidEncoding);
        }
        dst[index] = Fp::from_u64(word);
    }
    Ok(())
}

/// Read an extension-field range of canonical 24-byte elements.
pub fn read_fp3_range(
    reader: &dyn TensorReader,
    start: u64,
    count: usize,
    dst: &mut [Fp3],
) -> Result<()> {
    if !is_extension_field(reader.spec().dtype) {
        return Err(Error::UnsupportedPrimitive);
    }
    if dst.len() != count {
        return Err(Error::ShapeMismatch);
    }
    let byte_len = count.checked_mul(24).ok_or(Error::ResourceLimit)?;
    let mut bytes = vec![0u8; byte_len];
    reader.read_range(start, count, &mut bytes)?;
    for (index, chunk) in bytes.chunks_exact(24).enumerate() {
        dst[index] = Fp3::decode(chunk)?;
    }
    Ok(())
}

/// Read the whole integer tensor in fixed-size tiles. The function is
/// explicit about the tile size; the underlying reader still sees ranged
/// reads only.
pub fn read_i64_tiles(reader: &dyn TensorReader, tile_elements: usize) -> Result<Vec<i64>> {
    let total = usize::try_from(reader.element_count()).map_err(|_| Error::ResourceLimit)?;
    let _ = tile_count(total, tile_elements)?;
    let mut values = vec![0i64; total];
    let mut start = 0usize;
    while start < total {
        let count = core::cmp::min(tile_elements, total - start);
        read_i64_range(
            reader,
            start as u64,
            count,
            &mut values[start..start + count],
        )?;
        start += count;
    }
    Ok(values)
}

pub fn read_fp_tiles(reader: &dyn TensorReader, tile_elements: usize) -> Result<Vec<Fp>> {
    let total = usize::try_from(reader.element_count()).map_err(|_| Error::ResourceLimit)?;
    let _ = tile_count(total, tile_elements)?;
    let mut values = vec![Fp::ZERO; total];
    let mut start = 0usize;
    while start < total {
        let count = core::cmp::min(tile_elements, total - start);
        read_fp_range(
            reader,
            start as u64,
            count,
            &mut values[start..start + count],
        )?;
        start += count;
    }
    Ok(values)
}

pub fn read_fp3_tiles(reader: &dyn TensorReader, tile_elements: usize) -> Result<Vec<Fp3>> {
    let total = usize::try_from(reader.element_count()).map_err(|_| Error::ResourceLimit)?;
    let _ = tile_count(total, tile_elements)?;
    let mut values = vec![Fp3::ZERO; total];
    let mut start = 0usize;
    while start < total {
        let count = core::cmp::min(tile_elements, total - start);
        read_fp3_range(
            reader,
            start as u64,
            count,
            &mut values[start..start + count],
        )?;
        start += count;
    }
    Ok(values)
}

/// Encode integer values as canonical little-endian bytes for scratch or
/// resident buffers.
pub fn encode_i64(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
