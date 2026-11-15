//! The public integer range of an encoded table's entries.
//!
//! A table's entries are signed integers stored as field elements. This module
//! derives, from the public configuration alone, a lookup geometry that proves
//! every matrix entry lies in one declared range without reading the table's
//! values at verification time:
//!
//! - the table is the matrix of the committed table object, `rows` rows of
//!   `2^k_col` entries, each entry one signed integer;
//! - one lookup covers every entry: the key of an entry is the entry plus a
//!   public offset, and a key is a row of the public range table `0 .. 2^d - 1`;
//! - the declared range is `[lo, lo + 2^d)`, a power-of-two width, so a key in
//!   the range table is exactly an entry in the declared range (a width that
//!   is not a power of two would leave padded table rows that accept entries
//!   above the declared upper bound);
//! - every multiplicity is below the number of keys, which fixes the bit
//!   planes of the multiplicity message;
//! - one lookup with one part leaves one pending row: a single claim needs no
//!   batching.
//!
//! Nothing here fixes a production range or a production limit: a caller
//! passes the declaration and the resource limits, and the derivation either
//! returns the plan or refuses it with the dimension that failed.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::column_session::fp_from_signed_word;
use crate::proof::lookup_table::{content_digest, TableEncoding};

/// Version of the range declaration and the derived plan.
pub const ENTRY_RANGE_PLAN_VERSION: u64 = 1;

/// Domain separation of the public range table's identity.
const RANGE_TABLE_DOMAIN: &[u8] = b"zkie/entry-range-table/v1";

/// The largest table the canonical row encoding admits: the rows are the
/// integers `0 .. 2^table_log - 1`, written as signed 32-bit values.
const RANGE_TABLE_MAX_LOG: u32 = 30;

/// A declared range of an entry: `lo` up to and including `hi`. The width
/// `hi - lo + 1` must be a power of two (`2^d`, `d >= 1`), and both bounds must
/// fit the signed 32-bit entries of the encoded table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryRange {
    lo: i64,
    hi: i64,
}

impl EntryRange {
    /// Checks the declaration by itself: ordered bounds inside the signed
    /// 32-bit domain and a power-of-two width.
    pub fn new(lo: i64, hi: i64) -> Result<Self> {
        if lo > hi || lo < i32::MIN as i64 || hi > i32::MAX as i64 {
            return Err(Error::InvalidConfig);
        }
        let width = (hi - lo) as u64 + 1;
        if !width.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self { lo, hi })
    }

    pub fn lo(&self) -> i64 {
        self.lo
    }

    pub fn hi(&self) -> i64 {
        self.hi
    }

    pub fn width(&self) -> u64 {
        (self.hi - self.lo) as u64 + 1
    }

    pub fn table_log(&self) -> u32 {
        self.width().trailing_zeros()
    }
}

/// The resources a plan may use. The caller chooses them; nothing here
/// reduces a limit on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryRangeLimits {
    /// The largest range table: `2^table_log` rows.
    pub max_table_log: u32,
    /// The largest multiplicity-bit message: `2^(mu + table_log)` entries.
    pub max_bit_entries: u64,
}

/// The dimensions a declaration implies, before any limit applies. A caller
/// that reports an infeasible declaration uses this to say how large it would
/// be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryRangeDimensions {
    pub table_log: u32,
    pub mu: u32,
    pub key_bits: u32,
    pub place_bits: u32,
    pub bit_entries: u64,
}

/// One lookup covering every matrix entry, and the message and mask demand
/// that follow from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryRangePlan {
    pub version: u64,
    pub range: EntryRange,
    /// The range table has `2^table_log` rows, the integers `0 .. 2^table_log`.
    pub table_log: u32,
    /// Multiplicity planes: `2^mu` bits, one bit per plane.
    pub mu: u32,
    /// Value columns of the public table: one, the entry's range row.
    pub columns: u8,
    /// Key number bits: the matrix row bits and the matrix column bits.
    pub key_bits: u32,
    /// Columns of the matrix: the low coordinates of a key's number.
    pub column_bits: u32,
    /// Place bits of the one part: one place for each matrix row.
    pub place_bits: u32,
    pub matrix_rows: u64,
    /// The matrix row stride of one key's digits: one digit, no stride.
    pub stride: u64,
    /// The weight of the entry's single digit: one.
    pub digit_weight: Fp3,
    /// The public offset added to every entry: `-lo`.
    pub offset: Fp3,
    /// The multiplicity-bit message: `2^(mu + table_log)` entries.
    pub bit_entries: u64,
    /// Pending rows the lookup leaves on the matrix: one.
    pub pending_rows: usize,
    /// Entries of the index-mask message the lookup takes: two for the one
    /// part.
    pub index_mask_entries: usize,
    /// Batching rounds for the pending rows: none for a single row.
    pub batt_rounds: usize,
    /// The public range table's trusted identity.
    pub identity: Digest,
}

fn matrix_bits(rows: u64, k_col: u32) -> Result<(u32, u64)> {
    if rows == 0 || !rows.is_power_of_two() || rows > (1u64 << 40) || k_col > 40 {
        return Err(Error::InvalidConfig);
    }
    let key_bits = rows
        .trailing_zeros()
        .checked_add(k_col)
        .ok_or(Error::ResourceLimit)?;
    if key_bits == 0 || key_bits >= 63 {
        return Err(Error::ResourceLimit);
    }
    Ok((key_bits, 1u64 << key_bits))
}

/// The multiplicity planes: the smallest `mu` with `keys < 2^(2^mu)`, so that
/// one bit plane per multiplicity is enough for every key.
fn multiplicity_planes(keys: u64) -> Result<u32> {
    let mut mu = 1u32;
    loop {
        let planes = 1u64 << mu;
        if planes >= 63 {
            return Err(Error::ResourceLimit);
        }
        if keys < (1u64 << planes) {
            return Ok(mu);
        }
        mu = mu.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
}

/// The dimensions a declaration implies, with the range table and the
/// multiplicity message sized from the matrix and the range.
pub fn entry_range_dimensions(
    rows: u64,
    k_col: u32,
    range: EntryRange,
) -> Result<EntryRangeDimensions> {
    let (key_bits, keys) = matrix_bits(rows, k_col)?;
    let table_log = range.table_log();
    if table_log == 0 {
        return Err(Error::InvalidConfig);
    }
    let mu = multiplicity_planes(keys)?;
    let place_bits = table_log.checked_add(mu).ok_or(Error::ResourceLimit)?;
    if place_bits >= 63 {
        return Err(Error::ResourceLimit);
    }
    let bit_entries = 1u64 << place_bits;
    let place_bits = rows.trailing_zeros();
    Ok(EntryRangeDimensions {
        table_log,
        mu,
        key_bits,
        place_bits,
        bit_entries,
    })
}

/// The public range table's identity. The table is the integers
/// `0 .. 2^table_log - 1` in the canonical row encoding, so the identity binds
/// the width and that encoding, never a caller's choice of entries.
pub fn entry_range_table_identity(table_log: u32) -> Result<Digest> {
    if table_log == 0 || table_log > RANGE_TABLE_MAX_LOG {
        return Err(Error::InvalidConfig);
    }
    let mut bytes = Vec::with_capacity(RANGE_TABLE_DOMAIN.len() + 20);
    bytes.extend_from_slice(RANGE_TABLE_DOMAIN);
    bytes.extend_from_slice(&ENTRY_RANGE_PLAN_VERSION.to_le_bytes());
    bytes.extend_from_slice(&table_log.to_le_bytes());
    bytes.extend_from_slice(&(TableEncoding::I32Le.width() as u64).to_le_bytes());
    Ok(digest_bytes(&bytes))
}

/// The canonical content root of the range table: the digest of the
/// `2^table_log` rows `0 .. 2^table_log - 1` in the canonical row encoding.
/// It costs one pass over the table, so callers that only need the identity
/// do not call it.
pub fn entry_range_content_root(table_log: u32, max_rows: u64) -> Result<Digest> {
    if table_log == 0 || table_log > RANGE_TABLE_MAX_LOG {
        return Err(Error::InvalidConfig);
    }
    let rows = 1u64 << table_log;
    if rows > max_rows {
        return Err(Error::ResourceLimit);
    }
    let count = usize::try_from(rows).map_err(|_| Error::ResourceLimit)?;
    let values: Vec<i64> = (0..count as i64).collect();
    content_digest(TableEncoding::I32Le, &values)
}

/// Derives the entry range plan from the matrix geometry, the declared range
/// and the limits. One lookup with one part covers every matrix row; a
/// declaration whose table or multiplicity message exceeds a limit is refused
/// before any message is built.
pub fn derive_entry_range(
    rows: u64,
    k_col: u32,
    range: EntryRange,
    limits: EntryRangeLimits,
) -> Result<EntryRangePlan> {
    let dimensions = entry_range_dimensions(rows, k_col, range)?;
    if dimensions.table_log > RANGE_TABLE_MAX_LOG {
        // The range table's rows are the integers of the canonical 32-bit row
        // encoding; a wider table cannot be one.
        return Err(Error::InvalidConfig);
    }
    if dimensions.table_log > limits.max_table_log {
        return Err(Error::ResourceLimit);
    }
    if dimensions.bit_entries > limits.max_bit_entries {
        return Err(Error::ResourceLimit);
    }
    let offset = Fp3::ZERO.sub(&Fp3::from_base(fp_from_signed_word(range.lo() as u64)));
    Ok(EntryRangePlan {
        version: ENTRY_RANGE_PLAN_VERSION,
        range,
        table_log: dimensions.table_log,
        mu: dimensions.mu,
        columns: 1,
        key_bits: dimensions.key_bits,
        column_bits: k_col,
        place_bits: dimensions.place_bits,
        matrix_rows: rows,
        stride: 0,
        digit_weight: Fp3::ONE,
        offset,
        bit_entries: dimensions.bit_entries,
        // One lookup with one part: one pending row, no batching.
        pending_rows: 1,
        // The index-mask message carries two entries for the one part.
        index_mask_entries: 2,
        batt_rounds: 0,
        identity: entry_range_table_identity(dimensions.table_log)?,
    })
}

impl EntryRangePlan {
    /// The key of one entry value: the public offset plus the entry, in the
    /// extension field the messages use.
    pub fn entry_key(&self, value: i64) -> Fp3 {
        self.offset
            .add(&Fp3::from_base(fp_from_signed_word(value as u64)))
    }

    /// Whether a key is a row of the public range table: a base-field value
    /// below `2^table_log`. A key outside that set is not in the table.
    pub fn is_table_key(&self, key: &Fp3) -> bool {
        key.c[1].as_canonical_u64() == 0
            && key.c[2].as_canonical_u64() == 0
            && key.c[0].as_canonical_u64() < (1u64 << self.table_log)
    }

    /// Whether the one part's run covers every matrix row: one digit, no
    /// stride, one place for each row, and every key number uses the matrix's
    /// row and column bits.
    pub fn covers_every_row(&self) -> bool {
        self.stride == 0
            && self.digit_weight == Fp3::ONE
            && (1u64 << self.place_bits) == self.matrix_rows
            && self.place_bits + self.column_bits == self.key_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: u64 = 1024;
    const K_COL: u32 = 10;
    const LIMITS: EntryRangeLimits = EntryRangeLimits {
        max_table_log: 30,
        max_bit_entries: 1 << 24,
    };

    fn signed(bits: u32) -> EntryRange {
        EntryRange::new(-(1i64 << bits), (1i64 << bits) - 1).unwrap()
    }

    #[test]
    fn the_plan_covers_every_matrix_row_and_entry() {
        let plan = derive_entry_range(ROWS, K_COL, signed(15), LIMITS).unwrap();
        assert_eq!(plan.table_log, 16);
        assert_eq!(plan.mu, 5);
        assert_eq!(plan.key_bits, 20);
        assert_eq!(plan.column_bits, K_COL);
        assert_eq!(plan.place_bits, 10);
        assert_eq!(plan.matrix_rows, ROWS);
        assert_eq!(plan.stride, 0);
        assert_eq!(plan.digit_weight, Fp3::ONE);
        assert_eq!(plan.bit_entries, 1 << 21);
        assert_eq!(plan.pending_rows, 1);
        assert_eq!(plan.batt_rounds, 0);
        assert_eq!(plan.index_mask_entries, 2);
        assert!(plan.covers_every_row());
        let lo = plan.range.lo();
        let hi = plan.range.hi();
        assert!(plan.is_table_key(&plan.entry_key(lo)));
        assert!(plan.is_table_key(&plan.entry_key(hi)));
        assert_eq!(plan.entry_key(lo), Fp3::ZERO);
        assert_eq!(
            plan.entry_key(hi).c[0].as_canonical_u64(),
            plan.range.width() - 1
        );
    }

    #[test]
    fn the_same_declaration_derives_the_same_plan() {
        let first = derive_entry_range(ROWS, K_COL, signed(15), LIMITS).unwrap();
        let second = derive_entry_range(ROWS, K_COL, signed(15), LIMITS).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn another_declaration_changes_the_table_identity() {
        let small = derive_entry_range(ROWS, K_COL, signed(15), LIMITS).unwrap();
        let large = derive_entry_range(ROWS, K_COL, signed(17), LIMITS).unwrap();
        assert_ne!(small.table_log, large.table_log);
        assert_ne!(small.identity, large.identity);
        assert_ne!(
            entry_range_table_identity(16).unwrap(),
            entry_range_table_identity(17).unwrap()
        );
    }

    #[test]
    fn a_width_that_is_not_a_power_of_two_is_refused() {
        assert!(EntryRange::new(-1000, 1000).is_err());
        assert!(EntryRange::new(-(1i64 << 15), (1i64 << 15)).is_err());
    }

    #[test]
    fn values_beyond_the_declared_range_have_keys_outside_the_table() {
        let plan = derive_entry_range(ROWS, K_COL, signed(15), LIMITS).unwrap();
        let hi = plan.range.hi();
        let lo = plan.range.lo();
        assert!(!plan.is_table_key(&plan.entry_key(hi + 1)));
        assert!(!plan.is_table_key(&plan.entry_key(lo - 1)));
        assert!(!plan.is_table_key(&plan.entry_key(i32::MAX as i64)));
        assert!(!plan.is_table_key(&plan.entry_key(i32::MIN as i64)));
    }

    #[test]
    fn the_full_signed_32_bit_range_exceeds_the_table_limit() {
        let full = EntryRange::new(i32::MIN as i64, i32::MAX as i64).unwrap();
        let dimensions = entry_range_dimensions(ROWS, K_COL, full).unwrap();
        assert_eq!(dimensions.table_log, 32);
        assert_eq!(dimensions.bit_entries, 1 << 37);
        assert_eq!(
            derive_entry_range(ROWS, K_COL, full, LIMITS),
            Err(Error::InvalidConfig)
        );
        // A wider limit does not admit it: the canonical range table's rows
        // are signed 32-bit values, so the table itself is limited to 2^30
        // rows. The dimensions above still report what the declaration implies.
        let wide = EntryRangeLimits {
            max_table_log: 32,
            max_bit_entries: 1 << 37,
        };
        assert_eq!(
            derive_entry_range(ROWS, K_COL, full, wide),
            Err(Error::InvalidConfig)
        );
    }

    #[test]
    fn a_bit_message_above_the_limit_is_refused() {
        let tight = EntryRangeLimits {
            max_table_log: 30,
            max_bit_entries: 1 << 20,
        };
        assert_eq!(
            derive_entry_range(ROWS, K_COL, signed(15), tight),
            Err(Error::ResourceLimit)
        );
    }

    #[test]
    fn the_multiplicity_planes_follow_the_key_count() {
        let tiny = derive_entry_range(4, 2, signed(3), LIMITS).unwrap();
        assert_eq!(tiny.key_bits, 4);
        assert_eq!(tiny.mu, 3);
        assert_eq!(tiny.place_bits, 2);
        assert_eq!(tiny.table_log, 4);
        assert_eq!(tiny.bit_entries, 1 << 7);
        assert!(tiny.covers_every_row());
    }

    #[test]
    fn the_range_table_content_root_is_stable_and_bounded() {
        let root = entry_range_content_root(8, 1 << 10).unwrap();
        assert_eq!(root, entry_range_content_root(8, 1 << 10).unwrap());
        assert_ne!(root, entry_range_content_root(9, 1 << 10).unwrap());
        assert_eq!(
            entry_range_content_root(8, 1 << 7),
            Err(Error::ResourceLimit)
        );
        assert_eq!(
            entry_range_content_root(31, 1 << 40),
            Err(Error::InvalidConfig)
        );
    }
}
