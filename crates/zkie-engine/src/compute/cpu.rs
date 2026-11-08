//! The exact kernel set. Every kernel is deterministic: loop order, tile order
//! and worker count never change the result.

use std::sync::Arc;

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::parallel::{run_deterministic, LogicalTaskKey, TaskExecutor, TaskPhase};
use crate::pcs::request::field::{fp_from_u64, fp_pow, Fp, Fp3, MODULUS};
use crate::primitives::transcript::{
    hash_labeled, tag_word, LEAF_BINDING, LEAF_MAIN, LEAF_STORED_MAIN, TREE_NODE,
};

fn checked_matrix_len(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols).ok_or(Error::ResourceLimit)
}

fn check_slice(values_len: usize, expected: usize) -> Result<()> {
    if values_len != expected {
        return Err(Error::ShapeMismatch);
    }
    Ok(())
}

/// Exact integer matrix product with checked `i128` accumulation.
///
/// The accumulation is never reduced modulo a field. An accumulator overflow
/// is an error; callers that need wider results must use the limb routines.
pub fn fixed_matmul(
    a: &[i64],
    b: &[i64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<i128>> {
    fixed_matmul_rows(a, b, m, n, k, 0, m)
}

/// A contiguous block of output rows, for deterministic tiled execution.
pub fn fixed_matmul_rows(
    a: &[i64],
    b: &[i64],
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
) -> Result<Vec<i128>> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Matmul);
    if row_end < row_start || row_end > m {
        return Err(Error::InvalidIndex);
    }
    check_slice(a.len(), checked_matrix_len(m, k)?)?;
    check_slice(b.len(), checked_matrix_len(k, n)?)?;
    let rows = row_end - row_start;
    let mut out = vec![0i128; checked_matrix_len(rows, n)?];
    for row in row_start..row_end {
        for inner in 0..k {
            let left = a[row * k + inner] as i128;
            if left == 0 {
                continue;
            }
            for column in 0..n {
                let term = left
                    .checked_mul(b[inner * n + column] as i128)
                    .ok_or(Error::BoundViolation)?;
                let slot = &mut out[(row - row_start) * n + column];
                *slot = slot.checked_add(term).ok_or(Error::BoundViolation)?;
            }
        }
    }
    Ok(out)
}

/// Run row blocks on the deterministic pool. Results are assembled in key
/// order, so worker count and completion order are invisible.
pub fn parallel_fixed_matmul<E: TaskExecutor>(
    executor: &E,
    a: Vec<i64>,
    b: Vec<i64>,
    m: usize,
    n: usize,
    k: usize,
    block_rows: usize,
) -> Result<Vec<i128>> {
    if block_rows == 0 || m == 0 {
        return Err(Error::InvalidConfig);
    }
    let a = Arc::new(a);
    let b = Arc::new(b);
    let mut tasks = Vec::new();
    let mut block = 0u32;
    let mut start = 0usize;
    while start < m {
        let end = core::cmp::min(start + block_rows, m);
        let a = Arc::clone(&a);
        let b = Arc::clone(&b);
        tasks.push((
            LogicalTaskKey::new(0, TaskPhase::Forward, 0, 0, 0, block),
            move || fixed_matmul_rows(&a, &b, m, n, k, start, end),
        ));
        block = block.checked_add(1).ok_or(Error::ResourceLimit)?;
        start = end;
    }
    let results = run_deterministic(executor, tasks)?;
    let mut out = Vec::with_capacity(checked_matrix_len(m, n)?);
    for result in results {
        out.extend(result.value);
    }
    Ok(out)
}

pub fn field_matmul_fp(
    a: &[Fp],
    b: &[Fp],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<Fp>> {
    check_slice(a.len(), checked_matrix_len(m, k)?)?;
    check_slice(b.len(), checked_matrix_len(k, n)?)?;
    let mut out = vec![Fp::ZERO; checked_matrix_len(m, n)?];
    for row in 0..m {
        for inner in 0..k {
            let left = a[row * k + inner];
            if left == Fp::ZERO {
                continue;
            }
            for column in 0..n {
                out[row * n + column] = out[row * n + column] + left * b[inner * n + column];
            }
        }
    }
    Ok(out)
}

pub fn field_matmul_fp3(
    a: &[Fp3],
    b: &[Fp3],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<Fp3>> {
    check_slice(a.len(), checked_matrix_len(m, k)?)?;
    check_slice(b.len(), checked_matrix_len(k, n)?)?;
    let mut out = vec![Fp3::ZERO; checked_matrix_len(m, n)?];
    for row in 0..m {
        for inner in 0..k {
            let left = a[row * k + inner];
            if left.is_zero() {
                continue;
            }
            for column in 0..n {
                out[row * n + column] = out[row * n + column].add(&left.mul(&b[inner * n + column]));
            }
        }
    }
    Ok(out)
}

/// Batched product. Every batch uses the same fixed loop order as the single
/// product, so the concatenation equals per-batch calls.
pub fn batched_matmul_fp(
    a: &[Fp],
    b: &[Fp],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<Fp>> {
    let a_per_batch = checked_matrix_len(m, k)?;
    let b_per_batch = checked_matrix_len(k, n)?;
    let out_per_batch = checked_matrix_len(m, n)?;
    check_slice(a.len(), batch.checked_mul(a_per_batch).ok_or(Error::ResourceLimit)?)?;
    check_slice(b.len(), batch.checked_mul(b_per_batch).ok_or(Error::ResourceLimit)?)?;
    let mut out = Vec::with_capacity(
        batch
            .checked_mul(out_per_batch)
            .ok_or(Error::ResourceLimit)?,
    );
    for index in 0..batch {
        let product = field_matmul_fp(
            &a[index * a_per_batch..(index + 1) * a_per_batch],
            &b[index * b_per_batch..(index + 1) * b_per_batch],
            m,
            n,
            k,
        )?;
        out.extend(product);
    }
    Ok(out)
}

/// Gather rows from an embedding table. Every index is authenticated by being
/// below the table's row count; the complete table is the only weight source.
pub fn gather_matmul(
    table: &[i64],
    rows: usize,
    cols: usize,
    indices: &[u32],
) -> Result<Vec<i64>> {
    check_slice(table.len(), checked_matrix_len(rows, cols)?)?;
    let mut out = Vec::with_capacity(
        indices
            .len()
            .checked_mul(cols)
            .ok_or(Error::ResourceLimit)?,
    );
    for index in indices {
        let row = *index as usize;
        if row >= rows {
            return Err(Error::InvalidIndex);
        }
        out.extend_from_slice(&table[row * cols..(row + 1) * cols]);
    }
    Ok(out)
}

/// Multilinear fold in a fixed variable order. Variable `i` is the bit `i` of
/// the evaluation index; each challenge halves the vector.
pub fn mle_fold_fp(values: &[Fp], challenges: &[Fp]) -> Result<Fp> {
    let expected = 1usize
        .checked_shl(challenges.len() as u32)
        .ok_or(Error::ResourceLimit)?;
    check_slice(values.len(), expected)?;
    let mut current = values.to_vec();
    for challenge in challenges {
        let next_len = current.len() / 2;
        let mut next = vec![Fp::ZERO; next_len];
        for index in 0..next_len {
            let zero = current[2 * index];
            let one = current[2 * index + 1];
            next[index] = zero + *challenge * (one - zero);
        }
        current = next;
    }
    Ok(current[0])
}

pub fn mle_fold_fp3(values: &[Fp3], challenges: &[Fp3]) -> Result<Fp3> {
    let expected = 1usize
        .checked_shl(challenges.len() as u32)
        .ok_or(Error::ResourceLimit)?;
    check_slice(values.len(), expected)?;
    let mut current = values.to_vec();
    for challenge in challenges {
        let next_len = current.len() / 2;
        let mut next = vec![Fp3::ZERO; next_len];
        for index in 0..next_len {
            let zero = current[2 * index];
            let one = current[2 * index + 1];
            next[index] = zero.add(&challenge.mul(&one.sub(&zero)));
        }
        current = next;
    }
    Ok(current[0])
}

/// Exact integer sum; an `i128` overflow is an error, never a field wrap.
pub fn integer_sum(values: &[i64]) -> Result<i128> {
    let mut total = 0i128;
    for value in values {
        total = total
            .checked_add(*value as i128)
            .ok_or(Error::BoundViolation)?;
    }
    Ok(total)
}

/// The maximum and the first index that attains it.
pub fn integer_max(values: &[i64]) -> Option<(i64, usize)> {
    let mut best: Option<(i64, usize)> = None;
    for (index, value) in values.iter().enumerate() {
        match best {
            None => best = Some((*value, index)),
            Some((current, _)) if *value > current => best = Some((*value, index)),
            Some(_) => {}
        }
    }
    best
}

pub fn field_sum_fp(values: &[Fp]) -> Fp {
    values.iter().fold(Fp::ZERO, |accumulator, value| accumulator + *value)
}

pub fn field_sum_fp3(values: &[Fp3]) -> Fp3 {
    values
        .iter()
        .fold(Fp3::ZERO, |accumulator, value| accumulator.add(value))
}

/// The primitive `2^log_n`-th root of unity. The generator and stage order are
/// fixed by the protocol.
pub fn dft_root(log_n: u32) -> Result<Fp> {
    if log_n > 32 {
        return Err(Error::InvalidConfig);
    }
    let order = 1u64 << log_n;
    Ok(fp_pow(fp_from_u64(7), (MODULUS - 1) / order))
}

fn bit_reverse(mut value: usize, bits: u32) -> usize {
    let mut out = 0usize;
    for _ in 0..bits {
        out = (out << 1) | (value & 1);
        value >>= 1;
    }
    out
}

/// In-place-style DFT on the base field: natural-order input, bit-reversal
/// permutation, then fixed stages of doubling length. Output is natural order.
pub fn dft_fp(values: &[Fp]) -> Result<Vec<Fp>> {
    let n = values.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let log_n = n.trailing_zeros();
    let root = dft_root(log_n)?;
    let mut out = vec![Fp::ZERO; n];
    for (index, value) in values.iter().enumerate() {
        out[bit_reverse(index, log_n)] = *value;
    }
    let mut len = 2usize;
    while len <= n {
        let step = fp_pow(root, (n / len) as u64);
        let half = len / 2;
        let mut start = 0usize;
        while start < n {
            let mut twiddle = Fp::ONE;
            for offset in 0..half {
                let even = out[start + offset];
                let odd = out[start + offset + half] * twiddle;
                out[start + offset] = even + odd;
                out[start + offset + half] = even - odd;
                twiddle = twiddle * step;
            }
            start += len;
        }
        len <<= 1;
    }
    Ok(out)
}

fn is_leaf_domain(domain: u16) -> bool {
    matches!(domain, LEAF_MAIN | LEAF_BINDING | LEAF_STORED_MAIN)
}

fn is_vertex_domain(domain: u16) -> bool {
    domain == TREE_NODE
}

/// The tag word of a leaf, including its symbol count and salt length.
pub fn leaf_tag(domain: u16, tree: u16, symbols: u64, salt_bytes: usize) -> Result<u64> {
    if !is_leaf_domain(domain) {
        return Err(Error::InvalidEncoding);
    }
    if symbols == 0 || symbols >= (1u64 << 24) {
        return Err(Error::InvalidConfig);
    }
    if salt_bytes < 40 || salt_bytes > 64 || salt_bytes % 8 != 0 {
        return Err(Error::InvalidConfig);
    }
    Ok(domain as u64
        | ((tree as u64) << 16)
        | (symbols << 32)
        | (((salt_bytes / 8) as u64) << 56))
}

fn leaf_salt_at(symbols: usize, salt_bytes: usize) -> Result<usize> {
    let without_salt = 16usize
        .checked_add(symbols.checked_mul(8).ok_or(Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    if without_salt % 128 + salt_bytes <= 128 {
        Ok(without_salt)
    } else {
        Ok(((without_salt / 128) + 1) * 128)
    }
}

/// A leaf hash: symbols in canonical little-endian order, zero fill, then the
/// whole salt in the last compression block.
pub fn merkle_leaf(
    tag: u64,
    index: u64,
    start: &Digest,
    symbols: &[u64],
    salt: &[u8],
) -> Result<Digest> {
    let domain = (tag & 0xffff) as u16;
    let tag_symbols = ((tag >> 32) & 0x00ff_ffff) as usize;
    let tag_salt_bytes = ((tag >> 56) * 8) as usize;
    if !is_leaf_domain(domain)
        || tag_symbols != symbols.len()
        || tag_salt_bytes != salt.len()
    {
        return Err(Error::InvalidEncoding);
    }
    let salt_at = leaf_salt_at(symbols.len(), salt.len())?;
    let mut data = Vec::with_capacity(salt_at + salt.len());
    for symbol in symbols {
        data.extend_from_slice(&symbol.to_le_bytes());
    }
    let fill_to = salt_at - 16;
    while data.len() < fill_to {
        data.push(0);
    }
    data.extend_from_slice(salt);
    hash_labeled(tag, index, Some(start), &data)
}

/// A vertex hash: the label and two child digests.
pub fn merkle_node(
    tag: u64,
    index: u64,
    start: &Digest,
    left: &Digest,
    right: &Digest,
) -> Result<Digest> {
    if !is_vertex_domain((tag & 0xffff) as u16) {
        return Err(Error::InvalidEncoding);
    }
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&left.0);
    data.extend_from_slice(&right.0);
    hash_labeled(tag, index, Some(start), &data)
}

#[derive(Clone, Debug)]
pub struct MerkleTree {
    pub root: Digest,
    pub leaf_count: usize,
    pub padded_count: usize,
    pub levels: Vec<Vec<Digest>>,
}

/// Build a perfect binary tree of exactly `leaves.len()` leaves, which must be
/// a power of two. Leaves keep their order; there is no padding at this layer.
pub fn merkle_build(start: &Digest, leaves: &[Digest]) -> Result<MerkleTree> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Merkle);
    // The protocol's trees are perfect binary trees: the caller fixes the
    // depth and supplies exactly 2^depth leaves. There is no tree-level
    // padding rule; a non-power-of-two leaf count is rejected.
    if leaves.is_empty() || !leaves.len().is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let padded_count = leaves.len();
    let level = leaves.to_vec();
    let mut levels = vec![level];
    let node_tag = tag_word(TREE_NODE, 0);
    while levels.last().expect("one level").len() > 1 {
        let previous = levels.last().expect("one level");
        let mut next = Vec::with_capacity(previous.len() / 2);
        for pair in previous.chunks_exact(2) {
            next.push(merkle_node(node_tag, 0, start, &pair[0], &pair[1])?);
        }
        levels.push(next);
    }
    let root = levels.last().expect("one level")[0];
    Ok(MerkleTree {
        root,
        leaf_count: leaves.len(),
        padded_count,
        levels,
    })
}

impl MerkleTree {
    pub fn open(&self, index: usize) -> Result<Vec<Digest>> {
        if index >= self.leaf_count {
            return Err(Error::InvalidIndex);
        }
        let mut path = Vec::with_capacity(self.levels.len().saturating_sub(1));
        let mut position = index;
        for level in &self.levels[..self.levels.len() - 1] {
            path.push(level[position ^ 1]);
            position >>= 1;
        }
        Ok(path)
    }
}

pub fn merkle_verify(
    start: &Digest,
    root: &Digest,
    index: usize,
    leaf: &Digest,
    path: &[Digest],
) -> Result<bool> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Merkle);
    let node_tag = tag_word(TREE_NODE, 0);
    let mut current = *leaf;
    let mut position = index;
    for sibling in path {
        current = if position & 1 == 0 {
            merkle_node(node_tag, 0, start, &current, sibling)?
        } else {
            merkle_node(node_tag, 0, start, sibling, &current)?
        };
        position >>= 1;
    }
    Ok(&current == root)
}

/// Increment exact counts. A count overflow is an error.
pub fn lookup_accumulate_counts(counts: &mut [u64], indices: &[u32]) -> Result<()> {
    for index in indices {
        let slot = counts
            .get_mut(*index as usize)
            .ok_or(Error::InvalidIndex)?;
        *slot = slot.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    Ok(())
}

/// Field accumulation is kept separate from integer counting.
pub fn lookup_accumulate_fp(table: &mut [Fp], indices: &[u32]) -> Result<()> {
    for index in indices {
        let slot = table.get_mut(*index as usize).ok_or(Error::InvalidIndex)?;
        *slot = *slot + Fp::ONE;
    }
    Ok(())
}

/// Floor quotient and non-negative remainder for a positive divisor.
pub fn div_rem_floor(a: i128, divisor: i128) -> Result<(i128, i128)> {
    if divisor == 0 {
        return Err(Error::DivisionByZero);
    }
    if divisor < 0 {
        return Err(Error::InvalidConfig);
    }
    Ok((a.div_euclid(divisor), a.rem_euclid(divisor)))
}

/// Exact floor square root of an unsigned 128-bit integer.
pub fn isqrt_u128(value: u128) -> u128 {
    let mut low = 0u128;
    let mut high = 1u128 << 64;
    while low < high {
        let middle = low + (high - low + 1) / 2;
        match middle.checked_mul(middle) {
            Some(square) if square <= value => low = middle,
            _ => high = middle - 1,
        }
    }
    low
}

/// Split a non-negative value into limbs of `base_bits` bits each.
pub fn limb_decompose(value: i128, base_bits: u32) -> Result<Vec<u64>> {
    if value < 0 {
        return Err(Error::BoundViolation);
    }
    if base_bits == 0 || base_bits > 32 {
        return Err(Error::InvalidConfig);
    }
    let mask = (1u128 << base_bits) - 1;
    let mut remaining = value as u128;
    let mut limbs = Vec::new();
    loop {
        limbs.push((remaining & mask) as u64);
        remaining >>= base_bits;
        if remaining == 0 {
            break;
        }
    }
    Ok(limbs)
}

/// Reassemble limbs. A limb wider than `base_bits` is an encoding error.
pub fn limb_recompose(limbs: &[u64], base_bits: u32) -> Result<i128> {
    if base_bits == 0 || base_bits > 32 {
        return Err(Error::InvalidConfig);
    }
    let limit = 1u64 << base_bits;
    let mut value = 0u128;
    for limb in limbs.iter().rev() {
        if *limb >= limit {
            return Err(Error::InvalidEncoding);
        }
        value = value
            .checked_shl(base_bits)
            .ok_or(Error::BoundViolation)?
            | *limb as u128;
    }
    if value > i128::MAX as u128 {
        return Err(Error::BoundViolation);
    }
    Ok(value as i128)
}

/// Limb-wise addition with a carry out of the most significant limb.
pub fn limb_add(a: &[u64], b: &[u64], base_bits: u32) -> Result<(Vec<u64>, u64)> {
    if base_bits == 0 || base_bits > 32 {
        return Err(Error::InvalidConfig);
    }
    let limit = 1u128 << base_bits;
    let len = core::cmp::max(a.len(), b.len());
    let mut out = Vec::with_capacity(len);
    let mut carry = 0u128;
    for index in 0..len {
        let left = a.get(index).copied().unwrap_or(0) as u128;
        let right = b.get(index).copied().unwrap_or(0) as u128;
        if left >= limit || right >= limit {
            return Err(Error::InvalidEncoding);
        }
        let sum = left + right + carry;
        out.push((sum & (limit - 1)) as u64);
        carry = sum >> base_bits;
    }
    Ok((out, carry as u64))
}

pub fn transpose_2d<T: Copy>(input: &[T], rows: usize, cols: usize) -> Result<Vec<T>> {
    check_slice(input.len(), checked_matrix_len(rows, cols)?)?;
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = vec![input[0]; input.len()];
    for row in 0..rows {
        for column in 0..cols {
            out[column * rows + row] = input[row * cols + column];
        }
    }
    Ok(out)
}

pub fn reshape_row_major<T: Copy>(input: &[T], from: &[u64], to: &[u64]) -> Result<Vec<T>> {
    let from_count = from
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or(Error::ShapeMismatch)?;
    let to_count = to
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or(Error::ShapeMismatch)?;
    if from_count != to_count || usize::try_from(from_count).ok() != Some(input.len()) {
        return Err(Error::ShapeMismatch);
    }
    Ok(input.to_vec())
}

pub fn broadcast_row_major<T: Copy>(input: &[T], from: &[u64], to: &[u64]) -> Result<Vec<T>> {
    if from.len() != to.len() {
        return Err(Error::ShapeMismatch);
    }
    for (source, target) in from.iter().zip(to) {
        if *source != *target && *source != 1 {
            return Err(Error::ShapeMismatch);
        }
    }
    let from_count = from
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or(Error::ShapeMismatch)?;
    if usize::try_from(from_count).ok() != Some(input.len()) {
        return Err(Error::ShapeMismatch);
    }
    let to_count = to
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or(Error::ShapeMismatch)?;
    let to_count = usize::try_from(to_count).map_err(|_| Error::ResourceLimit)?;
    let mut source_strides = vec![1u64; from.len()];
    for index in (0..from.len()).rev().skip(1) {
        source_strides[index] = source_strides[index + 1]
            .checked_mul(from[index + 1])
            .ok_or(Error::ResourceLimit)?;
    }
    let mut out = Vec::with_capacity(to_count);
    for flat in 0..to_count {
        let mut remainder = flat as u64;
        let mut offset = 0u64;
        for axis in (0..to.len()).rev() {
            let coordinate = remainder % to[axis];
            remainder /= to[axis];
            let source_coordinate = if from[axis] == 1 { 0 } else { coordinate };
            offset = offset
                .checked_add(source_coordinate.checked_mul(source_strides[axis]).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ResourceLimit)?;
        }
        out.push(
            *input
                .get(offset as usize)
                .ok_or(Error::InvalidIndex)?,
        );
    }
    Ok(out)
}
