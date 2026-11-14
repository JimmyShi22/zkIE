//! The packed symbol array of one compute matrix.
//!
//! A matrix's integer cells are committed as one symbol array: cells of the
//! same column are packed in groups of at most three consecutive rows,
//! `P(s, c) = A(3s, c) + B * A(3s + 1, c) + B^2 * A(3s + 2, c)` with
//! `B = 2^20`; the last group holds only the valid rows. The array position is
//! `placeP(s, c) = (s / J, c * J + s % J)` with one place per group here
//! (`J = 1`), so the position is `(s, c)`. The number of groups is
//! `ceil(rows / 3)` and the number of columns is kept.
//!
//! The packing is an integer identity, not a field reduction: every cell must
//! be in `[0, 2^20)` so that the packed value is injective and below
//! `2^60`, and a cell outside that range is refused instead of being reduced
//! into the field. The plan fixes the geometry before any challenge; the
//! verifier consumes the packed array through the same plan and never
//! receives the original cells.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_engine::pcs::request::field::{fp_from_u64, Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::runtime_tree::{
    verify_runtime_opening, RuntimePartRows, RuntimeTree, RuntimeTreeOpening, RuntimeTreePlan,
};

/// The base of the packing digit, `2^20`.
pub const PACK_BASE: i128 = 1 << 20;
/// The rows one packed group may hold.
pub const PACK_ROWS: u64 = 3;

/// The geometry of one packed symbol array: the original matrix's rows and
/// columns, the group count and the packed value bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedMatrixPlan {
    pub rows: u64,
    pub columns: u64,
    pub groups: u64,
}

impl PackedMatrixPlan {
    pub fn new(rows: u64, columns: u64) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(Error::InvalidConfig);
        }
        let groups = rows
            .checked_add(PACK_ROWS - 1)
            .ok_or(Error::ResourceLimit)?
            / PACK_ROWS;
        Ok(Self {
            rows,
            columns,
            groups,
        })
    }

    /// The packed array position of group `s` and column `c`, with one place
    /// per group: `(s / J, c * J + s % J)` at `J = 1`.
    pub fn position(&self, group: u64, column: u64) -> Result<(u64, u64)> {
        if group >= self.groups || column >= self.columns {
            return Err(Error::InvalidIndex);
        }
        Ok((group, column))
    }

    /// The largest packed value the plan admits, `2^60 - 1`.
    pub fn packed_bound(&self) -> i128 {
        PACK_BASE * PACK_BASE * PACK_BASE - 1
    }
}

/// Packs the row-major integer cells of the matrix into the symbol array,
/// row-major over `(group, column)`. Each cell must be in `[0, 2^20)`; the
/// last group holds only the valid rows. No field reduction is applied.
pub fn pack_columns(cells: &[i128], plan: &PackedMatrixPlan) -> Result<Vec<i128>> {
    let rows = usize::try_from(plan.rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?;
    let groups = usize::try_from(plan.groups).map_err(|_| Error::ResourceLimit)?;
    let expected = rows.checked_mul(columns).ok_or(Error::ResourceLimit)?;
    if cells.len() != expected {
        return Err(Error::ShapeMismatch);
    }
    for cell in cells {
        if *cell < 0 || *cell >= PACK_BASE {
            return Err(Error::BoundViolation);
        }
    }
    let mut packed = vec![0i128; groups * columns];
    for group in 0..groups {
        for column in 0..columns {
            let mut value = 0i128;
            let mut scale = 1i128;
            for offset in 0..PACK_ROWS as usize {
                let row = group * PACK_ROWS as usize + offset;
                if row >= rows {
                    break;
                }
                value += scale * cells[row * columns + column];
                scale *= PACK_BASE;
            }
            packed[group * columns + column] = value;
        }
    }
    Ok(packed)
}


/// Commits the packed values under the plan's tree: one main row of the packed
/// entries, padded with zeros to the tree's columns. The values are integers
/// in `[0, 2^60)` embedded as base-field elements; no packed value is reduced.
pub fn commit_packed_array(
    plan: &PackedMatrixPlan,
    values: &[i128],
    tree: RuntimeTreePlan,
    salts: &[Vec<u8>],
) -> Result<RuntimeTree> {
    let entries = usize::try_from(plan.groups)
        .map_err(|_| Error::ResourceLimit)?
        .checked_mul(usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    if values.len() != entries {
        return Err(Error::ShapeMismatch);
    }
    for value in values {
        if *value < 0 || *value > plan.packed_bound() {
            return Err(Error::BoundViolation);
        }
    }
    let columns = usize::try_from(tree.columns()).map_err(|_| Error::ResourceLimit)?;
    if entries > columns || salts.len() != columns {
        return Err(Error::ShapeMismatch);
    }
    let mut row = vec![Fp3::ZERO; columns];
    for (slot, value) in row.iter_mut().zip(values) {
        *slot = Fp3::from_base(fp_from_u64(*value as u64));
    }
    let parts = [RuntimePartRows {
        main: vec![row],
        mask: Vec::new(),
        main_deg: E_DEGREE as u32,
    }];
    RuntimeTree::commit(tree, &parts, salts)
}

/// Verifies the packed array's opening at `positions` against `root` and
/// returns the packed values at those positions in the opening's order.
pub fn verify_packed_opening(
    plan: &PackedMatrixPlan,
    tree: &RuntimeTreePlan,
    root: &Digest,
    positions: &[u64],
    opening: &RuntimeTreeOpening,
) -> Result<Vec<i128>> {
    verify_runtime_opening(tree, root, positions, opening)?;
    let width = usize::try_from(tree.width()?).map_err(|_| Error::ResourceLimit)?;
    if width != E_DEGREE as usize || opening.columns.len() != positions.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut values = Vec::with_capacity(positions.len());
    for (at, column) in opening.columns.iter().enumerate() {
        if column.position != positions[at]
            || positions[at] >= tree.columns()
            || column.symbols.len() != width
        {
            return Err(Error::ShapeMismatch);
        }
        let value = Fp3::new(column.symbols[0], column.symbols[1], column.symbols[2]);
        if value.c[1] != Fp::ZERO || value.c[2] != Fp::ZERO {
            return Err(Error::InvalidProof);
        }
        let packed = value.c[0].as_canonical_u64();
        if packed > plan.packed_bound() as u64 {
            return Err(Error::BoundViolation);
        }
        values.push(packed as i128);
    }
    Ok(values)
}

#[cfg(test)]
mod commit_tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;

    pub(crate) fn tree() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 901,
            leaf_class: 1,
            log_n: 2,
            log_n_zk: 2,
            parts: vec![zkie_engine::pcs::request::runtime_tree::RuntimeTreePart {
                main_symbols: E_DEGREE as u64,
                mask_symbols: 0,
            }],
            salt_bytes: 40,
            run: Digest([0x91; 32]),
            configuration: Digest([0x92; 32]),
        }
    }

    pub(crate) fn salts() -> Vec<Vec<u8>> {
        (0..4).map(|index| vec![index as u8 ^ 0x5A; 40]).collect()
    }

    #[test]
    fn the_packed_array_opens_at_the_planned_positions() {
        let plan = PackedMatrixPlan::new(4, 2).unwrap();
        let packed = pack_columns(&super::tests_cells(), &plan).unwrap();
        let tree = tree();
        let committed = commit_packed_array(&plan, &packed, tree.clone(), &salts()).unwrap();
        let positions = [0u64, 2];
        let opening = committed.open(&positions).unwrap();
        let values = verify_packed_opening(&plan, &tree, committed.root(), &positions, &opening)
            .unwrap();
        assert_eq!(values, vec![packed[0], packed[2]]);
        // A changed root, a changed position and a zero packed value outside
        // the plan are refused.
        assert!(verify_packed_opening(
            &plan,
            &tree,
            &Digest([0xEE; 32]),
            &positions,
            &opening
        )
        .is_err());
        assert!(verify_packed_opening(&plan, &tree, committed.root(), &[3], &opening).is_err());
        let mut large = packed.clone();
        large[0] = plan.packed_bound() + 1;
        assert_eq!(
            commit_packed_array(&plan, &large, tree, &salts()).unwrap_err(),
            Error::BoundViolation
        );
    }
}

#[cfg(test)]
pub(crate) fn tests_cells() -> Vec<i128> {
    vec![1, 2, 3, 4, 5, 6, 7, 8]
}


/// The range transcript's combination object: for every column of the original
/// matrix and every combination plane, one value is stored as a low digit and a
/// high digit, `value = low + 2^low_bits * high`. The column coordinates are
/// `plane * columns + offset + index`; the offset and the digit width are plan
/// data, so the prover and the verifier address the same positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeCombinationPlan {
    pub columns: u64,
    pub planes: u64,
    pub low_bits: u32,
    pub offset: u64,
}

impl RangeCombinationPlan {
    pub fn new(columns: u64, planes: u64, low_bits: u32, offset: u64) -> Result<Self> {
        if columns == 0 || planes == 0 || low_bits == 0 || low_bits >= 63 {
            return Err(Error::InvalidConfig);
        }
        let last = planes
            .checked_mul(columns)
            .and_then(|value| value.checked_add(offset))
            .ok_or(Error::ResourceLimit)?;
        if last.checked_add(columns).is_none() {
            return Err(Error::ResourceLimit);
        }
        Ok(Self {
            columns,
            planes,
            low_bits,
            offset,
        })
    }

    /// The digit base `d`, one above the low digit's range.
    pub fn digit_base(&self) -> i128 {
        1i128 << self.low_bits
    }

    /// The stored column of plane `plane`, original column `index`:
    /// `plane * columns + offset + index`.
    pub fn zz_column(&self, plane: u64, index: u64) -> Result<u64> {
        if plane >= self.planes || index >= self.columns {
            return Err(Error::InvalidIndex);
        }
        plane
            .checked_mul(self.columns)
            .and_then(|value| value.checked_add(self.offset))
            .and_then(|value| value.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// Splits a nonnegative combination value into its low and high digits and
    /// checks the integer identity `value = low + d * high`.
    pub fn split(&self, value: i128) -> Result<(i128, i128)> {
        if value < 0 {
            return Err(Error::BoundViolation);
        }
        let base = self.digit_base();
        let low = value % base;
        let high = value / base;
        if low + base * high != value {
            return Err(Error::BoundViolation);
        }
        Ok((low, high))
    }
}



/// One range message of the combination object: the committed root and the
/// opening at the positions the plan samples. The values are the combination
/// object's stored entries, low and high digits in one integer each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeMessage {
    pub root: Digest,
    pub positions: Vec<u64>,
    pub opening: RuntimeTreeOpening,
}

/// The per-column combination values of the original matrix:
/// `offset + sum_row coeff[row] * A(row, column)`. Every coefficient and cell
/// is an integer; the value must be nonnegative, so the digit split is the
/// integer split. The coefficients come from the range transcript's
/// challenges and are public inputs to this evaluation.
pub fn combination_values(
    plan: &PackedMatrixPlan,
    cells: &[i128],
    coefficients: &[i128],
    offset: i128,
) -> Result<Vec<i128>> {
    let rows = usize::try_from(plan.rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?;
    if cells.len() != rows * columns || coefficients.len() != rows || offset < 0 {
        return Err(Error::ShapeMismatch);
    }
    if cells.iter().any(|cell| *cell < 0 || *cell >= PACK_BASE) {
        return Err(Error::BoundViolation);
    }
    let mut values = vec![offset; columns];
    for row in 0..rows {
        let coefficient = coefficients[row];
        if coefficient < 0 {
            return Err(Error::BoundViolation);
        }
        for (column, value) in values.iter_mut().enumerate() {
            let cell = cells[row * columns + column];
            *value = value
                .checked_add(coefficient.checked_mul(cell).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ResourceLimit)?;
        }
    }
    Ok(values)
}

/// Commits the stored combination values under the range plan's tree and
/// returns the opening at `positions` with the message that carries them.
pub fn commit_range_values(
    range: &RangeCombinationPlan,
    values: &[i128],
    tree: RuntimeTreePlan,
    salts: &[Vec<u8>],
    positions: &[u64],
) -> Result<(RuntimeTree, RangeMessage)> {
    let stored = usize::try_from(range.planes)
        .map_err(|_| Error::ResourceLimit)?
        .checked_mul(usize::try_from(range.columns).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    if values.len() != stored {
        return Err(Error::ShapeMismatch);
    }
    let bound = (1i128 << 62) - 1;
    for value in values {
        if *value < 0 || *value > bound {
            return Err(Error::BoundViolation);
        }
    }
    let columns = usize::try_from(tree.columns()).map_err(|_| Error::ResourceLimit)?;
    if stored > columns || salts.len() != columns {
        return Err(Error::ShapeMismatch);
    }
    let mut row = vec![Fp3::ZERO; columns];
    for (slot, value) in row.iter_mut().zip(values) {
        *slot = Fp3::from_base(fp_from_u64(*value as u64));
    }
    let parts = [RuntimePartRows {
        main: vec![row],
        mask: Vec::new(),
        main_deg: E_DEGREE as u32,
    }];
    let committed = RuntimeTree::commit(tree, &parts, salts)?;
    let opening = committed.open(positions)?;
    let message = RangeMessage {
        root: *committed.root(),
        positions: positions.to_vec(),
        opening,
    };
    Ok((committed, message))
}

/// Verifies a range message against the plan, tree and root and returns the
/// low/high digits of the stored columns at the message's positions.
pub fn verify_range_message(
    range: &RangeCombinationPlan,
    tree: &RuntimeTreePlan,
    message: &RangeMessage,
) -> Result<Vec<(i128, i128)>> {
    let total = range
        .planes
        .checked_mul(range.columns)
        .and_then(|value| value.checked_add(range.offset))
        .ok_or(Error::ResourceLimit)?;
    if message
        .positions
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
        || message.positions.iter().any(|position| *position >= total)
    {
        return Err(Error::InvalidIndex);
    }
    let values = verify_packed_opening(
        &PackedMatrixPlan::new(range.planes * range.columns, 1)?,
        tree,
        &message.root,
        &message.positions,
        &message.opening,
    )?;
    let _ = total;
    values.into_iter().map(|value| range.split(value)).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    fn cells() -> Vec<i128> {
        vec![
            1, 2, // row 0
            3, 4, // row 1
            5, 6, // row 2
            7, 8, // row 3
        ]
    }

    #[test]
    fn the_packing_follows_the_group_and_column_order() {
        let plan = PackedMatrixPlan::new(4, 2).unwrap();
        assert_eq!(plan.groups, 2);
        let packed = pack_columns(&cells(), &plan).unwrap();
        // Group 0 packs rows 0..3: 1 + B*3 + B^2*5 and 2 + B*4 + B^2*6.
        let b = PACK_BASE;
        assert_eq!(packed[0], 1 + b * 3 + b * b * 5);
        assert_eq!(packed[1], 2 + b * 4 + b * b * 6);
        // Group 1 holds the single valid row 3.
        assert_eq!(packed[2], 7);
        assert_eq!(packed[3], 8);
        assert!(packed.iter().all(|value| *value <= plan.packed_bound()));
        assert_eq!(plan.position(1, 1).unwrap(), (1, 1));
        assert!(plan.position(2, 0).is_err());
        assert!(plan.position(0, 2).is_err());
    }


    #[test]
    fn the_range_combination_object_addresses_and_splits() {
        let plan = RangeCombinationPlan::new(4, 2, 20, 7).unwrap();
        assert_eq!(plan.digit_base(), 1 << 20);
        assert_eq!(plan.zz_column(0, 0).unwrap(), 7);
        assert_eq!(plan.zz_column(1, 3).unwrap(), 4 + 7 + 3);
        assert!(plan.zz_column(2, 0).is_err());
        assert!(plan.zz_column(0, 4).is_err());
        let (low, high) = plan.split(0x12345).unwrap();
        assert_eq!(low, 0x12345);
        assert_eq!(high, 0);
        let value = 0x12345 + (1 << 20) * 9;
        let (low, high) = plan.split(value).unwrap();
        assert_eq!(low, 0x12345);
        assert_eq!(high, 9);
        assert_eq!(low + plan.digit_base() * high, value);
        assert_eq!(plan.split(-1).unwrap_err(), Error::BoundViolation);
        assert!(RangeCombinationPlan::new(0, 1, 20, 0).is_err());
        assert!(RangeCombinationPlan::new(1, 0, 20, 0).is_err());
        assert!(RangeCombinationPlan::new(1, 1, 0, 0).is_err());
        assert!(RangeCombinationPlan::new(1, 1, 63, 0).is_err());
    }


    #[test]
    fn the_range_values_are_combined_split_and_messaged() {
        let plan = PackedMatrixPlan::new(2, 2).unwrap();
        let cells = vec![1, 2, 3, 4];
        let coefficients = vec![2, 1];
        let values = combination_values(&plan, &cells, &coefficients, 5).unwrap();
        assert_eq!(values, vec![5 + 2 * 1 + 3, 5 + 2 * 2 + 4]);
        let range = RangeCombinationPlan::new(2, 2, 4, 0).unwrap();
        let stored = vec![values[0], values[1], values[1], values[0]];
        assert!(RangeCombinationPlan::new(2, 2, 4, 0).is_ok());
        let tree = super::commit_tests::tree();
        let salts = super::commit_tests::salts();
        let positions = [0u64, 3];
        let (_, message) =
            commit_range_values(&range, &stored, tree.clone(), &salts, &positions).unwrap();
        let digits = verify_range_message(&range, &tree, &message).unwrap();
        assert_eq!(digits.len(), 2);
        assert_eq!(digits[0], range.split(stored[0]).unwrap());
        assert_eq!(digits[1], range.split(stored[3]).unwrap());
        let mut changed = message.clone();
        changed.positions = vec![0, 0];
        assert!(verify_range_message(&range, &tree, &changed).is_err());
        let mut negative = cells.clone();
        negative[0] = -1;
        assert!(combination_values(&plan, &negative, &coefficients, 0).is_err());
    }

    #[test]
    fn out_of_range_cells_and_wrong_sizes_are_refused() {
        let plan = PackedMatrixPlan::new(4, 2).unwrap();
        let mut large = cells();
        large[0] = PACK_BASE;
        assert_eq!(pack_columns(&large, &plan).unwrap_err(), Error::BoundViolation);
        let mut negative = cells();
        negative[3] = -1;
        assert_eq!(
            pack_columns(&negative, &plan).unwrap_err(),
            Error::BoundViolation
        );
        assert_eq!(
            pack_columns(&cells()[..7], &plan).unwrap_err(),
            Error::ShapeMismatch
        );
        assert!(PackedMatrixPlan::new(0, 1).is_err());
        assert!(PackedMatrixPlan::new(1, 0).is_err());
    }
}
