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

/// The canonical row encoding of the range table, bound into its identity by
/// its semantics so that another encoding of the same width cannot share it.
const RANGE_TABLE_ENCODING: &[u8] = b"i32-le";

/// The largest table the canonical row encoding admits: the rows are the
/// integers `0 .. 2^table_log - 1`, written as signed 32-bit values, so the
/// last row is at most `2^31 - 1`. This is the encoding's own bound; the
/// resource limit a caller passes may be smaller.
const RANGE_TABLE_MAX_LOG: u32 = 31;

/// The base field's bit width. A key is a field element whose integer form
/// must stay inside the field, so the range table's rows and the
/// multiplicity planes together stay below it.
const FIELD_BITS: u32 = 64;

/// The admitted number of multiplicity bit positions: at least two and at
/// most 62, so that every multiplicity fits a plane count the field admits.
const PLANES_MIN: u32 = 2;
const PLANES_MAX: u32 = 62;

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

// ---------------------------------------------------------------------------
// The public range configuration of the first production weight table
// ---------------------------------------------------------------------------

/// The declared integer interval of the first production weight table. The
/// interval enters the table configuration and the key parameters; a value
/// outside it is refused when the table is built and again by the lookup.
/// The width is fixed by the public declaration: the measured entries of the
/// first table stay inside it and the width is not chosen from the proof or
/// from the private extremes of a run.
pub const PRODUCTION_TABLE_RANGE_LO: i64 = -262_144;
pub const PRODUCTION_TABLE_RANGE_HI: i64 = 262_143;
/// The matrix geometry: 1024 rows by 1024 columns.
pub const PRODUCTION_TABLE_MATRIX_ROWS: u64 = 1_024;
pub const PRODUCTION_TABLE_COLUMN_BITS: u32 = 10;
/// The admitted resource limits of this implementation. They are resource
/// bounds, not a wider protocol range.
pub const PRODUCTION_TABLE_MAX_TABLE_LOG: u32 = 19;
pub const PRODUCTION_TABLE_MAX_BIT_ENTRIES: u64 = 1 << 24;

/// The public range plan of the first production weight table, derived from
/// the fixed public configuration above.
pub fn production_table_range() -> Result<EntryRangePlan> {
    derive_entry_range(
        PRODUCTION_TABLE_MATRIX_ROWS,
        PRODUCTION_TABLE_COLUMN_BITS,
        EntryRange::new(PRODUCTION_TABLE_RANGE_LO, PRODUCTION_TABLE_RANGE_HI)?,
        EntryRangeLimits {
            max_table_log: PRODUCTION_TABLE_MAX_TABLE_LOG,
            max_bit_entries: PRODUCTION_TABLE_MAX_BIT_ENTRIES,
        },
    )
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
    // The smallest plane count whose bits can hold a multiplicity of every
    // key. A key count that no admitted plane count holds is refused here,
    // before any resource limit is consulted.
    for mu in 1u32..=6 {
        let planes = 1u64 << mu;
        // A plane count of 64 holds every key; the shift by 64 below is why
        // this case is taken first.
        if planes >= 64 || keys < (1u64 << planes) {
            return Ok(mu);
        }
    }
    Err(Error::InvalidConfig)
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
    let bit_bits = table_log.checked_add(mu).ok_or(Error::ResourceLimit)?;
    if bit_bits >= FIELD_BITS {
        return Err(Error::ResourceLimit);
    }
    let bit_entries = 1u64 << bit_bits;
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
    bytes.extend_from_slice(RANGE_TABLE_ENCODING);
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
    let planes = 1u32 << dimensions.mu;
    // The lookup's own domain, from its declared range plan: the planes are
    // between two and 62, the table has at least one row, a key number stays
    // below the plane count (a part has at most as many keys as a
    // multiplicity can count), and the table rows plus the planes stay inside
    // the base field. Sizes alone, from entry_range_dimensions(), grant
    // nothing.
    if planes < PLANES_MIN
        || planes > PLANES_MAX
        || dimensions.table_log < 1
        || dimensions.key_bits >= planes
        || dimensions
            .table_log
            .checked_add(planes)
            .map_or(true, |bits| bits >= FIELD_BITS)
    {
        return Err(Error::InvalidConfig);
    }
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

// ---------------------------------------------------------------------------
// The range lookup over the committed matrix
// ---------------------------------------------------------------------------

use crate::proof::hidden_member::{MaskTerm as PendingMaskTerm, PendingRow, WeightBlock};
use crate::proof::lookup::{eq_weights, LookupPlan, LookupTable};
use crate::proof::lookup_table::{TableConfig, TABLE_ENCODING_VERSION};
use crate::proof::psc_general::GeneralClaim;

/// The public range table `0 .. 2^table_log - 1` as a lookup table. The rows
/// are the integers in the canonical signed 32-bit encoding, so every row is
/// public and no value is committed. The verifier evaluates the table's
/// multilinear extension at a point in closed form; a caller that wants the
/// actual table-side checks runs the lookup's table levels, which this table
/// answers at every challenge point.
pub struct IntegerRangeTable {
    identity: Digest,
    config: TableConfig,
    table_log: u32,
    max_root_rows: u64,
}

impl IntegerRangeTable {
    /// Opens the range table of one plan. The content root costs one pass over
    /// the table, bounded by `max_root_rows`.
    pub fn new(plan: &EntryRangePlan, max_root_rows: u64) -> Result<Self> {
        if plan.table_log == 0 || plan.table_log > RANGE_TABLE_MAX_LOG {
            return Err(Error::InvalidConfig);
        }
        if plan.identity != entry_range_table_identity(plan.table_log)? {
            return Err(Error::InvalidConfig);
        }
        let row_count = 1u64 << plan.table_log;
        let content_root = entry_range_content_root(plan.table_log, max_root_rows)?;
        Ok(Self {
            identity: plan.identity,
            config: TableConfig {
                identity: plan.identity,
                content_root,
                real_len: usize::try_from(row_count).map_err(|_| Error::ResourceLimit)?,
                encoding: TableEncoding::I32Le,
                version: TABLE_ENCODING_VERSION,
            },
            table_log: plan.table_log,
            max_root_rows,
        })
    }

    pub fn row_count(&self) -> u64 {
        1u64 << self.table_log
    }
}

impl LookupTable for IntegerRangeTable {
    fn columns(&self) -> usize {
        1
    }

    fn row_count(&self) -> usize {
        1usize << self.table_log
    }

    fn identity(&self) -> Digest {
        self.identity
    }

    /// The multilinear extension of the identity function `f(t) = t`: at a
    /// point it is `sum over the bits of 2^i * r_i`.
    fn column_at(&self, column: usize, point: &[Fp3]) -> Result<Fp3> {
        if column != 0 || point.len() != self.table_log as usize {
            return Err(Error::InvalidConfig);
        }
        let mut value = Fp3::ZERO;
        for (bit, coordinate) in point.iter().enumerate() {
            value = value.add(&coordinate.mul(&Fp3::from_u64(1u64 << bit)));
        }
        Ok(value)
    }

    fn column_rows(&self, column: usize) -> Result<Vec<Fp3>> {
        if column != 0 {
            return Err(Error::InvalidIndex);
        }
        Ok((0..self.row_count()).map(Fp3::from_u64).collect())
    }

    fn config(&self) -> TableConfig {
        self.config.clone()
    }

    fn check_content(&self) -> Result<()> {
        let root = entry_range_content_root(self.table_log, self.max_root_rows)?;
        if root != self.config.content_root {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// The lookup plan that proves the plan's declared range over the committed
/// matrix: one key for each matrix entry, one value column, the range table's
/// rows, and the multiplicity planes the key count requires.
pub fn range_lookup_plan(plan: &EntryRangePlan) -> Result<LookupPlan> {
    let lookup = LookupPlan {
        key_bits: plan.key_bits,
        mu: plan.mu,
        table_log: plan.table_log,
        columns: 1,
    };
    lookup.validate()?;
    Ok(lookup)
}

/// The key of every matrix entry, in row-major order: the entry plus the
/// public offset, in the extension field the messages use.
pub fn range_key_values(plan: &EntryRangePlan, entries: &[Fp3]) -> Result<Vec<Fp3>> {
    let count = plan
        .matrix_rows
        .checked_mul(1u64 << plan.column_bits)
        .ok_or(Error::ResourceLimit)?;
    if entries.len() as u64 != count {
        return Err(Error::ShapeMismatch);
    }
    Ok(entries.iter().map(|entry| plan.offset.add(entry)).collect())
}

/// The multiplicity of every row of the range table: how often that range
/// value occurs among the keys. A key outside the table is refused.
pub fn range_multiplicities(plan: &EntryRangePlan, keys: &[Fp3]) -> Result<Vec<u64>> {
    let rows = 1usize << plan.table_log;
    let mut counts = vec![0u64; rows];
    for key in keys {
        if !plan.is_table_key(key) {
            return Err(Error::BoundViolation);
        }
        let row = usize::try_from(key.c[0].as_canonical_u64()).map_err(|_| Error::ResourceLimit)?;
        let slot = counts.get_mut(row).ok_or(Error::BoundViolation)?;
        *slot = slot.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    Ok(counts)
}

/// The multiplicity-bit values of the lookup's message, plane-major:
/// `bits[j + planes * row]` is bit `j` of the row's multiplicity.
pub fn range_bit_values(plan: &EntryRangePlan, counts: &[u64]) -> Result<Vec<Fp3>> {
    let planes = 1usize << plan.mu;
    let rows = 1usize << plan.table_log;
    if counts.len() != rows {
        return Err(Error::ShapeMismatch);
    }
    let mut bits = vec![Fp3::ZERO; planes * rows];
    for (row, count) in counts.iter().enumerate() {
        for plane in 0..planes {
            if (count >> plane) & 1 == 1 {
                bits[plane + planes * row] = Fp3::ONE;
            }
        }
    }
    Ok(bits)
}

/// The pending claim of the declared range configuration: the same
/// derivation as , from the public declaration alone.
pub fn range_pending_row_with(
    config: &TableKeyRangeConfig,
    claim: &GeneralClaim,
    index_oracle: u64,
) -> Result<PendingRow> {
    if claim.point.len() != config.key_bits as usize {
        return Err(Error::ShapeMismatch);
    }
    let column_bits = config.column_bits as usize;
    let place_point = &claim.point[column_bits..];
    let weights = eq_weights(place_point);
    if weights.len() != (1usize << config.place_bits) {
        return Err(Error::ShapeMismatch);
    }
    let mut covered = Fp3::ZERO;
    for weight in &weights {
        covered = covered.add(weight);
    }
    let blocks = if config.place_bits == 0 {
        vec![WeightBlock::Scalar {
            row: 0,
            value: weights[0],
        }]
    } else {
        vec![WeightBlock::Dense {
            first_row: 0,
            log_rows: config.place_bits,
            values: weights,
        }]
    };
    let ps: Vec<Fp3> = claim.point[..column_bits].to_vec();
    let target = claim.value.sub(&covered.mul(&config.offset));
    let masks = claim
        .masks
        .iter()
        .map(|term| PendingMaskTerm {
            oracle: index_oracle,
            slot: term.slot as u64,
            coeff: term.coefficient,
        })
        .collect();
    Ok(PendingRow {
        weights: blocks,
        psi: ps,
        target,
        masks,
    })
}

/// The pending claim the range lookup leaves on the matrix, from the lookup's
/// key claim: the matrix is weighted by the equality weights of the key
/// point's place coordinates, the column point is the key point's column
/// coordinates, the covered places' share of the offset moves into the target,
/// and the claim's mask terms are kept with their index-mask oracle.
pub fn range_pending_row(
    plan: &EntryRangePlan,
    claim: &GeneralClaim,
    index_oracle: u64,
) -> Result<PendingRow> {
    if claim.point.len() != plan.key_bits as usize {
        return Err(Error::ShapeMismatch);
    }
    let column_bits = plan.column_bits as usize;
    let place_point = &claim.point[column_bits..];
    let weights = eq_weights(place_point);
    if weights.len() != (1usize << plan.place_bits) {
        return Err(Error::ShapeMismatch);
    }
    let mut covered = Fp3::ZERO;
    for weight in &weights {
        covered = covered.add(weight);
    }
    let blocks = if plan.place_bits == 0 {
        vec![WeightBlock::Scalar {
            row: 0,
            value: weights[0],
        }]
    } else {
        vec![WeightBlock::Dense {
            first_row: 0,
            log_rows: plan.place_bits,
            values: weights,
        }]
    };
    let ps: Vec<Fp3> = claim.point[..column_bits].to_vec();
    let target = claim.value.sub(&covered.mul(&plan.offset));
    let masks = claim
        .masks
        .iter()
        .map(|term| PendingMaskTerm {
            oracle: index_oracle,
            slot: term.slot as u64,
            coeff: term.coefficient,
        })
        .collect();
    Ok(PendingRow {
        weights: blocks,
        psi: ps,
        target,
        masks,
    })
}

// ---------------------------------------------------------------------------
// The lookup message of the range plan
// ---------------------------------------------------------------------------

use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::field::E_DEGREE;

use crate::proof::hidden_send_session::WShape;
use crate::proof::lookup::mask_count;
use crate::proof::proximity_planner::smallest_mask_rows;

/// The lookup message of one range plan: the multiplicity-bit message, its
/// shape and the mask message's size. The caller fixes the codes and the
/// message identities; the plan fixes the entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeLookupMessage {
    pub lookup: LookupPlan,
    /// Entries of the bit message: `2^(mu + table_log)`.
    pub bits_entries: u64,
    /// The bit message's shape: one message tree, one sample answer.
    pub bits_shape: WShape,
    /// Mask entries the lookup takes.
    pub mask_entries: usize,
    /// Rows of the mask message: the smallest power of two that holds the
    /// entries beside the SEND pad.
    pub mask_rows: u64,
}

/// Derives the range lookup's message from the plan and the codes. The bit
/// message carries the plan's bit entries; the mask message carries the
/// lookup's mask entries.
#[allow(clippy::too_many_arguments)]
pub fn derive_range_lookup_message(
    plan: &EntryRangePlan,
    code: CodeParams,
    mask_code: CodeParams,
    salt_bytes: usize,
    bits_tree: u64,
    bits_leaf_class: u32,
    run: Digest,
    configuration: Digest,
) -> Result<RangeLookupMessage> {
    let lookup = range_lookup_plan(plan)?;
    let bits_entries = plan.bit_entries;
    let columns = usize::try_from(bits_entries).map_err(|_| Error::ResourceLimit)?;
    let mask_entries = mask_count(&lookup)?;
    let mask_rows = smallest_mask_rows(mask_entries, mask_code.ell)?;
    // The mask tree's part is `mask_rows * E_DEGREE` symbols, and the mask
    // code's tree depth must hold it.
    let symbols = mask_rows
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    if symbols > (1u64 << mask_code.log_n) {
        return Err(Error::ResourceLimit);
    }
    let bits_shape = WShape::new(
        bits_entries,
        code,
        1,
        1,
        salt_bytes,
        bits_tree,
        bits_leaf_class,
        run,
        configuration,
    )?;
    if bits_shape.fault(bits_entries, 1).is_some() {
        return Err(Error::InvalidConfig);
    }
    let main_symbols = bits_shape
        .main_rows
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    if main_symbols > (1u64 << code.log_n) {
        return Err(Error::ResourceLimit);
    }
    let _ = columns;
    Ok(RangeLookupMessage {
        lookup,
        bits_entries,
        bits_shape,
        mask_entries,
        mask_rows,
    })
}

// ---------------------------------------------------------------------------
// The public declaration of one table's entry range
// ---------------------------------------------------------------------------

use zkie_core::common::codec::DecodeLimits;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

/// Version of the range declaration.
pub const TABLE_KEY_RANGE_VERSION: u64 = 1;

/// The public declaration of one table's entry range: the integers the table
/// is to satisfy, the matrix geometry the lookup covers, and the message
/// counts the plan takes. It is separate from the range table's identity: the
/// same table serves any offset, so this declaration binds the bounds, the
/// offset and the geometry beside the table identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableKeyRangeConfig {
    pub version: u64,
    pub lo: i64,
    pub hi: i64,
    pub matrix_rows: u64,
    pub column_bits: u32,
    pub place_bits: u32,
    pub table_log: u32,
    pub mu: u32,
    pub key_bits: u32,
    pub offset: Fp3,
    pub digit_weight: Fp3,
    pub stride: u64,
    pub range_table: Digest,
    pub pending_rows: usize,
    pub index_mask_entries: usize,
    pub batt_rounds: usize,
}

/// The declaration of one derived plan.
pub fn table_key_range_config(plan: &EntryRangePlan) -> TableKeyRangeConfig {
    TableKeyRangeConfig {
        version: TABLE_KEY_RANGE_VERSION,
        lo: plan.range.lo(),
        hi: plan.range.hi(),
        matrix_rows: plan.matrix_rows,
        column_bits: plan.column_bits,
        place_bits: plan.place_bits,
        table_log: plan.table_log,
        mu: plan.mu,
        key_bits: plan.key_bits,
        offset: plan.offset,
        digit_weight: plan.digit_weight,
        stride: plan.stride,
        range_table: plan.identity,
        pending_rows: plan.pending_rows,
        index_mask_entries: plan.index_mask_entries,
        batt_rounds: plan.batt_rounds,
    }
}

impl TableKeyRangeConfig {
    /// The canonical bytes of the declaration.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.version);
        writer.put_u64(self.lo as u64);
        writer.put_u64(self.hi as u64);
        writer.put_u64(self.matrix_rows);
        writer.put_u64(self.column_bits as u64);
        writer.put_u64(self.place_bits as u64);
        writer.put_u64(self.table_log as u64);
        writer.put_u64(self.mu as u64);
        writer.put_u64(self.key_bits as u64);
        writer.put_fp3(&self.offset);
        writer.put_fp3(&self.digit_weight);
        writer.put_u64(self.stride);
        writer.put_digest(&self.range_table);
        writer.put_u64(self.pending_rows as u64);
        writer.put_u64(self.index_mask_entries as u64);
        writer.put_u64(self.batt_rounds as u64);
        writer.into_bytes()
    }

    pub fn digest(&self) -> Digest {
        digest_bytes(&self.encode())
    }

    /// Decodes the canonical bytes strictly: the version must match, and
    /// truncation and trailing bytes are rejected.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let limits = DecodeLimits {
            max_bytes: 1 << 20,
            max_elements: 1 << 12,
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        let version = reader.get_u64()?;
        if version != TABLE_KEY_RANGE_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let config = Self {
            version,
            lo: reader.get_u64()? as i64,
            hi: reader.get_u64()? as i64,
            matrix_rows: reader.get_u64()?,
            column_bits: u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            place_bits: u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            table_log: u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            mu: u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            key_bits: u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            offset: reader.get_fp3()?,
            digit_weight: reader.get_fp3()?,
            stride: reader.get_u64()?,
            range_table: reader.get_digest()?,
            pending_rows: usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            index_mask_entries: usize::try_from(reader.get_u64()?)
                .map_err(|_| Error::InvalidEncoding)?,
            batt_rounds: usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
        };
        reader.finish()?;
        Ok(config)
    }

    /// The declaration must be the one the plan derives: same bounds, same
    /// geometry, same table identity and the same message counts.
    pub fn validate(&self, plan: &EntryRangePlan) -> Result<()> {
        if *self != table_key_range_config(plan) {
            return Err(Error::StatementMismatch);
        }
        Ok(())
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
        // A wider limit does not admit it: 32 table rows and 32 planes are not
        // below the 64-bit field, and the canonical row encoding only admits
        // the integers 0 .. 2^31 - 1. The dimensions above still report what
        // the declaration would imply.
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
    fn a_key_count_beyond_the_admitted_planes_is_refused() {
        let range = signed(3);
        // Sizing still reports what the declaration implies: 2^32 keys need
        // six planes (64 bit positions).
        let dimensions = entry_range_dimensions(1u64 << 32, 0, range).unwrap();
        assert_eq!(dimensions.mu, 6);
        // The plan refuses it: the lookup admits at most 62 planes.
        assert_eq!(
            derive_entry_range(
                1u64 << 32,
                0,
                range,
                EntryRangeLimits {
                    max_table_log: 19,
                    max_bit_entries: 1 << 24,
                }
            ),
            Err(Error::InvalidConfig)
        );
    }

    #[test]
    fn the_frozen_candidate_derives_its_declared_geometry() {
        // The candidate the public configuration declares: the integers
        // -131072 ..= 131071, one key per matrix entry of 1024 by 1024.
        let range = EntryRange::new(-131072, 131071).unwrap();
        let plan = derive_entry_range(
            ROWS,
            K_COL,
            range,
            EntryRangeLimits {
                max_table_log: 19,
                max_bit_entries: 1 << 24,
            },
        )
        .unwrap();
        assert_eq!(plan.table_log, 18);
        assert_eq!(plan.mu, 5);
        assert_eq!(plan.key_bits, 20);
        assert_eq!(plan.place_bits, 10);
        assert_eq!(plan.bit_entries, 1 << 23);
        assert_eq!(plan.pending_rows, 1);
        assert_eq!(plan.index_mask_entries, 2);
        assert_eq!(plan.identity, entry_range_table_identity(18).unwrap());
        assert_eq!(plan.entry_key(-131072), Fp3::ZERO);
        assert_eq!(plan.entry_key(131071).c[0].as_canonical_u64(), 262143);
        assert!(!plan.is_table_key(&plan.entry_key(131072)));
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
            entry_range_content_root(31, 1 << 30),
            Err(Error::ResourceLimit)
        );
        assert_eq!(
            entry_range_content_root(32, 1 << 40),
            Err(Error::InvalidConfig)
        );
    }
}

#[cfg(test)]
mod range_lookup_tests {
    use super::*;
    use crate::proof::psc_general::MaskTerm as ClaimMaskTerm;

    const ROWS: u64 = 4;
    const K_COL: u32 = 2;

    fn signed(bits: u32) -> EntryRange {
        EntryRange::new(-(1i64 << bits), (1i64 << bits) - 1).unwrap()
    }

    fn plan(bits: u32) -> EntryRangePlan {
        derive_entry_range(
            ROWS,
            K_COL,
            signed(bits),
            EntryRangeLimits {
                max_table_log: 30,
                max_bit_entries: 1 << 20,
            },
        )
        .unwrap()
    }

    fn eq_at(point: &[Fp3], index: u64) -> Fp3 {
        let mut value = Fp3::ONE;
        for (bit, coordinate) in point.iter().enumerate() {
            let one_minus = Fp3::ONE.sub(coordinate);
            let factor = if (index >> bit) & 1 == 1 {
                *coordinate
            } else {
                one_minus
            };
            value = value.mul(&factor);
        }
        value
    }

    fn field(value: i64) -> Fp3 {
        Fp3::from_base(fp_from_signed_word(value as u64))
    }

    #[test]
    fn the_range_table_evaluates_like_its_rows() {
        let plan = plan(3);
        let table = IntegerRangeTable::new(&plan, 1 << 8).unwrap();
        assert_eq!(table.row_count(), 1 << plan.table_log);
        assert_eq!(table.identity(), plan.identity);
        assert_eq!(table.config().real_len, 1 << plan.table_log);
        table.check_content().unwrap();
        let point = [
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
            Fp3::from_u64(11),
        ];
        let mut brute = Fp3::ZERO;
        for row in 0..(1u64 << plan.table_log) {
            brute = brute.add(&Fp3::from_u64(row).mul(&eq_at(&point, row)));
        }
        assert_eq!(table.column_at(0, &point).unwrap(), brute);
        assert!(table.column_at(1, &point).is_err());
        assert!(table.column_at(0, &point[..2]).is_err());
        let rows = table.column_rows(0).unwrap();
        assert_eq!(rows.len(), 1 << plan.table_log);
        assert_eq!(rows[3], Fp3::from_u64(3));
    }

    #[test]
    fn the_lookup_plan_matches_the_range_plan() {
        let plan = plan(3);
        let lookup = range_lookup_plan(&plan).unwrap();
        assert_eq!(lookup.key_bits, plan.key_bits);
        assert_eq!(lookup.mu, plan.mu);
        assert_eq!(lookup.table_log, plan.table_log);
        assert_eq!(lookup.columns, 1);
    }

    #[test]
    fn the_multiplicity_bits_reconstruct_the_counts() {
        let plan = plan(3);
        let matrix = [-8i64, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, 0, 1, 2, 3, 4];
        let entries: Vec<Fp3> = matrix.iter().map(|value| field(*value)).collect();
        let keys = range_key_values(&plan, &entries).unwrap();
        let counts = range_multiplicities(&plan, &keys).unwrap();
        assert_eq!(counts.iter().sum::<u64>(), 16);
        let bits = range_bit_values(&plan, &counts).unwrap();
        assert_eq!(bits.len() as u64, plan.bit_entries);
        let planes = 1usize << plan.mu;
        for (row, count) in counts.iter().enumerate() {
            let mut rebuilt = 0u64;
            for plane in 0..planes {
                if bits[plane + planes * row] == Fp3::ONE {
                    rebuilt |= 1 << plane;
                }
            }
            assert_eq!(rebuilt, *count);
        }
    }

    #[test]
    fn an_entry_outside_the_declared_range_has_no_key() {
        let plan = plan(3);
        let mut entries = vec![Fp3::ZERO; (ROWS << K_COL) as usize];
        entries[5] = field(plan.range.hi() + 1);
        let keys = range_key_values(&plan, &entries).unwrap();
        assert_eq!(
            range_multiplicities(&plan, &keys),
            Err(Error::BoundViolation)
        );
        entries[5] = field(plan.range.lo() - 1);
        let keys = range_key_values(&plan, &entries).unwrap();
        assert_eq!(
            range_multiplicities(&plan, &keys),
            Err(Error::BoundViolation)
        );
    }

    #[test]
    fn the_pending_row_rebuilds_the_matrix_claim() {
        let plan = plan(3);
        let columns = 1usize << K_COL;
        let matrix: Vec<Vec<i64>> = vec![
            vec![-8, -7, -6, -5],
            vec![-4, -3, -2, -1],
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
        ];
        let point = [
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let mut key_value = Fp3::ZERO;
        for row in 0..ROWS as usize {
            for column in 0..columns {
                let number = column as u64 + ((row as u64) << K_COL);
                let key = plan.offset.add(&field(matrix[row][column]));
                key_value = key_value.add(&key.mul(&eq_at(&point, number)));
            }
        }
        let mask_value = field(37);
        let claim = GeneralClaim {
            point: point.to_vec(),
            value: key_value.add(&field(5).mul(&mask_value)),
            masks: vec![ClaimMaskTerm {
                slot: 7,
                coefficient: field(5),
            }],
        };
        let row = range_pending_row(&plan, &claim, 60).unwrap();
        assert_eq!(row.psi, point[..K_COL as usize].to_vec());
        assert_eq!(row.masks.len(), 1);
        assert_eq!(row.masks[0].oracle, 60);
        assert_eq!(row.masks[0].slot, 7);
        assert_eq!(row.masks[0].coeff, field(5));
        assert_eq!(row.weights.len(), 1);
        let (block, weights) = match &row.weights[0] {
            WeightBlock::Dense {
                first_row,
                log_rows,
                values,
            } => (*first_row, (*log_rows, values)),
            _ => panic!("the one part covers a dense row block"),
        };
        assert_eq!(block, 0);
        assert_eq!(weights.0, plan.place_bits);
        // The row's weighted matrix sum plus its mask terms must be the target.
        let mut matrix_sum = Fp3::ZERO;
        for (place, weight) in weights.1.iter().enumerate() {
            for column in 0..columns {
                let column_weight = eq_at(&point[..K_COL as usize], column as u64);
                matrix_sum = matrix_sum.add(
                    &weight
                        .mul(&column_weight)
                        .mul(&field(matrix[place][column])),
                );
            }
        }
        let lhs = matrix_sum.add(&field(5).mul(&mask_value));
        assert_eq!(lhs, row.target);
    }

    #[test]
    fn a_key_point_of_the_wrong_size_is_refused() {
        let plan = plan(3);
        let claim = GeneralClaim {
            point: vec![Fp3::ONE; 3],
            value: Fp3::ZERO,
            masks: Vec::new(),
        };
        assert_eq!(
            range_pending_row(&plan, &claim, 60),
            Err(Error::ShapeMismatch)
        );
    }
}

#[cfg(test)]
mod range_message_tests {
    use super::*;
    use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};

    const SALT: usize = 40;

    fn frozen_plan() -> EntryRangePlan {
        derive_entry_range(
            1024,
            10,
            EntryRange::new(-131072, 131071).unwrap(),
            EntryRangeLimits {
                max_table_log: 19,
                max_bit_entries: 1 << 24,
            },
        )
        .unwrap()
    }

    #[test]
    fn the_range_message_fits_the_frozen_geometry() {
        let plan = frozen_plan();
        let code = zkie_engine::pcs::request::code::build_code_plan(15, 2048, 128).unwrap();
        let mask_code = zkie_engine::pcs::request::code::build_code_plan(15, 256, 66).unwrap();
        let message = derive_range_lookup_message(
            &plan,
            code,
            mask_code,
            SALT,
            100,
            1,
            Digest([0x41; 32]),
            Digest([0x51; 32]),
        )
        .unwrap();
        assert_eq!(message.bits_entries, 1 << 23);
        // The last place of a full row falls on the row's sample pad, so the
        // message takes one more power-of-two row.
        assert_eq!(message.bits_shape.main_rows, 8192);
        assert_eq!(message.bits_shape.answers, 1);
        assert!(message.mask_entries > 0);
        assert!(message.mask_rows.is_power_of_two());
        assert!(message.mask_rows * mask_code.ell - 1 >= message.mask_entries as u64);
        assert!(message.mask_rows * E_DEGREE as u64 <= (1u64 << mask_code.log_n));
        let commit = crate::proof::lookup_messages::LookupCommitPlan {
            bits_shape: message.bits_shape,
            bits_oracle: 1,
            group: 7,
            bits_openings: 1,
            mask_oracle: 2,
            mask_code,
            mask_rows: message.mask_rows,
            mask_openings: 1,
            mask_tree: RuntimeTreePlan {
                tree_number: 101,
                leaf_class: 3,
                log_n: mask_code.log_n,
                log_n_zk: mask_code.log_n,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: message.mask_rows * E_DEGREE as u64,
                }],
                salt_bytes: SALT,
                run: Digest([0x42; 32]),
                configuration: Digest([0x52; 32]),
            },
        };
        commit
            .validate(message.bits_entries as usize, message.mask_entries)
            .unwrap();
    }

    #[test]
    fn a_code_without_room_for_the_bit_tree_is_refused() {
        let plan = frozen_plan();
        let code = zkie_engine::pcs::request::code::build_code_plan(12, 2048, 128).unwrap();
        let mask_code = zkie_engine::pcs::request::code::build_code_plan(15, 256, 66).unwrap();
        assert_eq!(
            derive_range_lookup_message(
                &plan,
                code,
                mask_code,
                SALT,
                100,
                1,
                Digest([0x41; 32]),
                Digest([0x51; 32]),
            ),
            Err(Error::ResourceLimit)
        );
    }

    #[test]
    fn the_mask_rows_hold_the_entries() {
        assert_eq!(smallest_mask_rows(255, 256).unwrap(), 1);
        assert_eq!(smallest_mask_rows(256, 256).unwrap(), 2);
        assert_eq!(smallest_mask_rows(0, 256), Err(Error::InvalidConfig));
        assert_eq!(smallest_mask_rows(3, 0), Err(Error::InvalidConfig));
    }
}

#[cfg(test)]
mod key_range_config_tests {
    use super::*;

    fn plan(lo: i64, hi: i64) -> EntryRangePlan {
        derive_entry_range(
            1024,
            10,
            EntryRange::new(lo, hi).unwrap(),
            EntryRangeLimits {
                max_table_log: 19,
                max_bit_entries: 1 << 24,
            },
        )
        .unwrap()
    }

    #[test]
    fn the_declaration_round_trips_and_matches_its_plan() {
        let plan = plan(-131072, 131071);
        let config = table_key_range_config(&plan);
        config.validate(&plan).unwrap();
        let bytes = config.encode();
        assert_eq!(TableKeyRangeConfig::decode(&bytes).unwrap(), config);
        assert_eq!(config.digest(), digest_bytes(&bytes));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(TableKeyRangeConfig::decode(&trailing).is_err());
        let mut short = bytes;
        short.pop();
        assert!(TableKeyRangeConfig::decode(&short).is_err());
    }

    #[test]
    fn the_same_table_serves_another_offset() {
        let low = plan(-131072, 131071);
        let shifted = plan(-131071, 131072);
        assert_eq!(low.table_log, shifted.table_log);
        assert_eq!(low.identity, shifted.identity);
        assert_ne!(low.offset, shifted.offset);
        let low_config = table_key_range_config(&low);
        let shifted_config = table_key_range_config(&shifted);
        assert_ne!(low_config.digest(), shifted_config.digest());
        assert_eq!(low_config.validate(&shifted), Err(Error::StatementMismatch));
        assert_eq!(shifted_config.validate(&low), Err(Error::StatementMismatch));
    }

    #[test]
    fn the_public_production_range_derives_the_frozen_geometry() {
        let plan = production_table_range().expect("the public configuration derives");
        assert_eq!(plan.range.lo(), PRODUCTION_TABLE_RANGE_LO);
        assert_eq!(plan.range.hi(), PRODUCTION_TABLE_RANGE_HI);
        assert_eq!(plan.table_log, 19);
        assert_eq!(plan.mu, 5);
        assert_eq!(plan.key_bits, 20);
        assert_eq!(plan.bit_entries, 1 << 24);
        assert_eq!(plan.offset, Fp3::from_u64(262_144));
        // A wider table than the admitted resource limit is refused.
        let wider = derive_entry_range(
            PRODUCTION_TABLE_MATRIX_ROWS,
            PRODUCTION_TABLE_COLUMN_BITS,
            EntryRange::new(-(1 << 19), (1 << 19) - 1).unwrap(),
            EntryRangeLimits {
                max_table_log: PRODUCTION_TABLE_MAX_TABLE_LOG,
                max_bit_entries: PRODUCTION_TABLE_MAX_BIT_ENTRIES,
            },
        );
        assert!(wider.is_err());
    }

    #[test]
    fn another_geometry_is_refused() {
        let plan = plan(-131072, 131071);
        let config = table_key_range_config(&plan);
        let mut other = config.clone();
        other.column_bits += 1;
        assert_eq!(other.validate(&plan), Err(Error::StatementMismatch));
        let mut other = config.clone();
        other.pending_rows = 2;
        assert_eq!(other.validate(&plan), Err(Error::StatementMismatch));
        let mut other = config;
        other.version = TABLE_KEY_RANGE_VERSION + 1;
        assert!(TableKeyRangeConfig::decode(&other.encode()).is_err());
    }
}
