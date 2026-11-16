//! Projection consumer claims.
//!
//! A projection consumer reduces its activation rows against one weight
//! matrix in the encoded table layout. This module keeps the two directions of
//! that reduction in one place: the prover side turns the verified activation
//! rows and the operator's actual output rows into the row-weight and output
//! claims, and the verifier side recomputes the same reduction from those
//! claims and the trusted table rows. No prover-supplied equation is trusted:
//! both sides derive the row weights from the activation values.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

/// The trusted geometry of one projection consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionShape {
    /// Rows of the weight matrix, which are also the table's row count.
    pub matrix_rows: u64,
    /// The low column bits that select a table row; `columns = matrix_rows << k_lo`.
    pub k_lo: u32,
    /// Output columns per position.
    pub columns: u64,
    /// Activation/output positions.
    pub positions: u32,
}

impl ProjectionShape {
    pub fn check(&self) -> Result<()> {
        if self.matrix_rows == 0 || self.columns == 0 || self.positions == 0 || self.k_lo >= 32 {
            return Err(Error::InvalidConfig);
        }
        let expected = self
            .matrix_rows
            .checked_shl(self.k_lo)
            .ok_or(Error::ResourceLimit)?;
        if expected != self.columns {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    /// The number of entries of each table row message used by the reduction.
    pub fn entries(&self) -> u64 {
        self.columns
    }

    /// The number of table rows.
    pub fn table_rows(&self) -> u64 {
        self.columns >> self.k_lo
    }
}

/// The claims of one projection consumer: one row-weight vector per position
/// and one output row per position.
#[derive(Clone, Debug)]
pub struct ProjectionClaims {
    pub weights: Vec<Vec<Fp3>>,
    pub output: Vec<Vec<Fp3>>,
}

fn rows<T: Clone>(values: &[T], width: usize, count: usize) -> Result<Vec<Vec<T>>> {
    if width == 0 || values.len() != width.checked_mul(count).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    Ok(values.chunks_exact(width).map(|row| row.to_vec()).collect())
}

/// Builds the claims from the verified activation rows and the operator's
/// actual output rows. The caller passes values that were verified elsewhere;
/// this function never accepts a prover-chosen equation.
pub fn projection_claims(
    shape: ProjectionShape,
    activation: &[Fp3],
    output: &[Fp3],
) -> Result<ProjectionClaims> {
    shape.check()?;
    let matrix_rows = usize::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(shape.columns).map_err(|_| Error::ResourceLimit)?;
    let weights = rows(activation, matrix_rows, shape.positions as usize)?;
    let output = rows(output, columns, shape.positions as usize)?;
    Ok(ProjectionClaims { weights, output })
}

fn entry(shape: &ProjectionShape, column: u64, row: u64) -> Result<u64> {
    let low_mask = (1u64 << shape.k_lo) - 1;
    let low = column & low_mask;
    let shifted = row.checked_shl(shape.k_lo).ok_or(Error::ResourceLimit)?;
    low.checked_add(shifted).ok_or(Error::ResourceLimit)
}

/// Recomputes every output entry from the row weights and the trusted table
/// rows in the encoded layout, and compares it with the claimed output.
pub fn check_projection_reduction(
    shape: ProjectionShape,
    weights: &[Vec<Fp3>],
    table: &[Vec<Fp3>],
    output: &[Vec<Fp3>],
) -> Result<()> {
    shape.check()?;
    let matrix_rows = usize::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(shape.columns).map_err(|_| Error::ResourceLimit)?;
    let table_rows = usize::try_from(shape.table_rows()).map_err(|_| Error::ResourceLimit)?;
    if weights.len() != shape.positions as usize
        || output.len() != shape.positions as usize
        || table.len() != table_rows
    {
        return Err(Error::ShapeMismatch);
    }
    if weights.iter().any(|row| row.len() != matrix_rows)
        || output.iter().any(|row| row.len() != columns)
        || table.iter().any(|row| row.len() < columns)
    {
        return Err(Error::ShapeMismatch);
    }
    for position in 0..shape.positions as usize {
        for column in 0..columns {
            let table_row = column >> shape.k_lo as usize;
            let mut sum = Fp3::ZERO;
            for row in 0..matrix_rows {
                let index = entry(&shape, column as u64, row as u64)?;
                let index = usize::try_from(index).map_err(|_| Error::ResourceLimit)?;
                let value = table[table_row]
                    .get(index)
                    .copied()
                    .ok_or(Error::ShapeMismatch)?;
                sum = sum.add(&weights[position][row].mul(&value));
            }
            if sum != output[position][column] {
                return Err(Error::InvalidProof);
            }
        }
    }
    Ok(())
}

/// The authenticated table matrix in its stored-record orientation: one call
/// writes `M[place, record] = A[record, place]` for every place of one stored
/// row. The real prover streams exactly one stored row at a time through the
/// asset mapping; no caller holds the whole matrix. The component fixtures
/// expose their place-major rows through the same interface.
pub trait ConsumerMatrixColumns {
    fn places(&self) -> u64;
    fn records(&self) -> u64;
    fn row(&self, record: u64, out: &mut [Fp3]) -> Result<()>;
}

/// The place-major fixture matrix behind the stored-row interface. It exists
/// for component tests only; the production path reads the asset mapping.
pub struct PlaceRows<'a> {
    rows: &'a [Vec<Fp3>],
}

impl<'a> PlaceRows<'a> {
    pub fn new(rows: &'a [Vec<Fp3>]) -> Self {
        Self { rows }
    }

    fn records_of(&self) -> usize {
        self.rows.first().map_or(0, |row| row.len())
    }
}

impl ConsumerMatrixColumns for PlaceRows<'_> {
    fn places(&self) -> u64 {
        self.rows.len() as u64
    }

    fn records(&self) -> u64 {
        self.records_of() as u64
    }

    fn row(&self, record: u64, out: &mut [Fp3]) -> Result<()> {
        let record = usize::try_from(record).map_err(|_| Error::ResourceLimit)?;
        if out.len() != self.rows.len()
            || self.rows.iter().any(|row| row.len() != self.records_of())
        {
            return Err(Error::ShapeMismatch);
        }
        for (place, row) in self.rows.iter().enumerate() {
            out[place] = *row.get(record).ok_or(Error::ShapeMismatch)?;
        }
        Ok(())
    }
}

/// The multilinear equality weight of one index at a point.
fn eq_at(point: &[Fp3], index: u64) -> Fp3 {
    let mut weight = Fp3::ONE;
    for (bit, coordinate) in point.iter().enumerate() {
        let chosen = if (index >> bit) & 1 == 1 {
            *coordinate
        } else {
            Fp3::ONE.sub(coordinate)
        };
        weight = weight.mul(&chosen);
    }
    weight
}

/// One projection consumer's reduction claim: the public activation rows are
/// reduced by the challenge coefficients of the positions, the raw output
/// columns of one head slice by the challenge coefficients of the columns.
/// The claim is about the same matrix `M` the key authenticates: the matrix's
/// rows are the table places and its columns are the table records
/// (`M[place, record] = A[record, place]` for the weight file `A`).
///
/// The identity the product-sum proof closes is
/// `Σ_p Σ_n a[p] b[n] Y[p,n] = Σ_k (Σ_p a[p] X[p,k]) · (Σ_n b[n] M[n,k])`,
/// where `Y` is the raw integer accumulator of the production projection,
/// not its requantized output. `b` is zero outside the consumed slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerReduction {
    /// `U[k] = Σ_p a[p] X[p,k]`: the shown side of the product sum, over the
    /// contraction index (the matrix's columns).
    pub shown: Vec<Fp3>,
    /// The column coefficients `b` padded to the full table columns; entries
    /// outside the consumed slice are zero.
    pub weights: Vec<Fp3>,
    /// `Σ_p Σ_n a[p] b[n] Y[p,n]` over the raw output.
    pub target: Fp3,
}

impl ConsumerReduction {
    /// The right side on a known matrix: `Σ_k shown[k] · Σ_n weights[n]
    /// M[n,k]`. The matrix is streamed one stored record at a time.
    pub fn identity_on(&self, matrix: &dyn ConsumerMatrixColumns) -> Result<Fp3> {
        let columns = self.column_values(matrix)?;
        let mut total = Fp3::ZERO;
        for (shown, column) in self.shown.iter().zip(&columns) {
            total = total.add(&shown.mul(column));
        }
        Ok(total)
    }

    /// The private second map of the product sum, `V[k] = Σ_n weights[n]
    /// M[n,k]`: one value per contraction index. The matrix is streamed one
    /// stored record at a time, so the memory stays one row of places.
    pub fn column_values(&self, matrix: &dyn ConsumerMatrixColumns) -> Result<Vec<Fp3>> {
        let places = usize::try_from(matrix.places()).map_err(|_| Error::ResourceLimit)?;
        let records = usize::try_from(matrix.records()).map_err(|_| Error::ResourceLimit)?;
        if self.weights.len() != places || self.shown.len() != records {
            return Err(Error::ShapeMismatch);
        }
        let mut columns = vec![Fp3::ZERO; records];
        let mut row = vec![Fp3::ZERO; places];
        for (record, column) in columns.iter_mut().enumerate() {
            matrix.row(record as u64, &mut row)?;
            let mut total = Fp3::ZERO;
            for (weight, value) in self.weights.iter().zip(&row) {
                if !weight.is_zero() {
                    total = total.add(&weight.mul(value));
                }
            }
            *column = total;
        }
        Ok(columns)
    }
}

/// The public first map of the product sum, `U[k] = Σ_p a[p] X[p,k]`: the
/// activation rows reduced by the position coefficients over the contraction
/// index. The activation rows are public verified values; both sides derive
/// this vector, and the raw output never enters it.
pub fn activation_shown(positions: usize, records: u64, x: &[Fp3], a: &[Fp3]) -> Result<Vec<Fp3>> {
    let contraction = usize::try_from(records).map_err(|_| Error::ResourceLimit)?;
    if positions == 0 || contraction == 0 {
        return Err(Error::InvalidConfig);
    }
    if x.len()
        != positions
            .checked_mul(contraction)
            .ok_or(Error::ResourceLimit)?
    {
        return Err(Error::ShapeMismatch);
    }
    if a.len() != positions {
        return Err(Error::ShapeMismatch);
    }
    let mut shown = vec![Fp3::ZERO; contraction];
    for position in 0..positions {
        let coefficient = a[position];
        if coefficient.is_zero() {
            continue;
        }
        for k in 0..contraction {
            shown[k] = shown[k].add(&coefficient.mul(&x[position * contraction + k]));
        }
    }
    Ok(shown)
}

/// The left side of the consumer identity on the raw output rows:
/// `Σ_p a[p] · Σ_n b[n] Y[p,n]`. The rows are the operator's actual raw
/// accumulator over the consumed slice; the product sumcheck proves the two
/// reduced maps multiply to this value.
pub fn raw_target(
    positions: usize,
    slice_len: usize,
    y_raw: &[Fp3],
    a: &[Fp3],
    b: &[Fp3],
) -> Result<Fp3> {
    if positions == 0 || slice_len == 0 || a.len() != positions || b.len() != slice_len {
        return Err(Error::InvalidConfig);
    }
    if y_raw.len()
        != positions
            .checked_mul(slice_len)
            .ok_or(Error::ResourceLimit)?
    {
        return Err(Error::ShapeMismatch);
    }
    let mut target = Fp3::ZERO;
    for position in 0..positions {
        let mut row = Fp3::ZERO;
        for index in 0..slice_len {
            let coefficient = b[index];
            if coefficient.is_zero() {
                continue;
            }
            row = row.add(&coefficient.mul(&y_raw[position * slice_len + index]));
        }
        target = target.add(&a[position].mul(&row));
    }
    Ok(target)
}

/// Builds one head slice's reduction claim from the public activation rows,
/// the raw output and the challenge coefficients. The position count, the
/// contraction length, the table columns and the consumed slice are the fixed
/// public plan data; `b_slice` is the slice's challenge coefficients and the
/// padded weights are zero outside it.
#[allow(clippy::too_many_arguments)]
pub fn consumer_reduction(
    positions: usize,
    matrix_rows: u64,
    columns: u64,
    slice: (u64, u64),
    x: &[Fp3],
    y_raw: &[Fp3],
    a: &[Fp3],
    b_slice: &[Fp3],
) -> Result<ConsumerReduction> {
    let contraction = usize::try_from(matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let full_columns = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    let (slice_start, slice_end) = slice;
    if positions == 0 || contraction == 0 || full_columns == 0 {
        return Err(Error::InvalidConfig);
    }
    if slice_end <= slice_start || slice_end > columns {
        return Err(Error::InvalidConfig);
    }
    let slice_len = usize::try_from(slice_end - slice_start).map_err(|_| Error::ResourceLimit)?;
    let expected_x = positions
        .checked_mul(contraction)
        .ok_or(Error::ResourceLimit)?;
    let expected_y = positions
        .checked_mul(slice_len)
        .ok_or(Error::ResourceLimit)?;
    if x.len() != expected_x || y_raw.len() != expected_y {
        return Err(Error::ShapeMismatch);
    }
    if a.len() != positions || b_slice.len() != slice_len {
        return Err(Error::ShapeMismatch);
    }
    let shown = activation_shown(positions, matrix_rows, x, a)?;
    let mut weights = vec![Fp3::ZERO; full_columns];
    let start = usize::try_from(slice_start).map_err(|_| Error::ResourceLimit)?;
    weights[start..start + slice_len].copy_from_slice(b_slice);
    let target = raw_target(positions, slice_len, y_raw, a, b_slice)?;
    Ok(ConsumerReduction {
        shown,
        weights,
        target,
    })
}

/// One consumed head slice of a projection consumer plan: the head index and
/// the table columns `[start, end)` the head's projection reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerSlice {
    pub head: u32,
    pub columns: (u64, u64),
}

/// The fixed coverage of one real request's projection consumers: every head
/// slice in order, exactly the consumed columns, no overlap, and no stored
/// padding column claimed as consumed. A table may store more columns than the
/// request uses; the coverage never rounds the used region up to the stored
/// width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerCoverage {
    positions: u32,
    head_dim: u64,
    consumed_columns: u64,
    table_columns: u64,
    slices: Vec<ConsumerSlice>,
}

impl ConsumerCoverage {
    /// Builds the coverage of `heads` head slices of `head_dim` columns each,
    /// consuming exactly `hidden` columns of a table with `table_columns`
    /// stored columns. `heads * head_dim` must equal `hidden`, and `hidden`
    /// must not exceed the stored width.
    pub fn new(
        positions: u32,
        head_dim: u64,
        heads: u32,
        hidden: u64,
        table_columns: u64,
    ) -> Result<Self> {
        if positions == 0 || head_dim == 0 || heads == 0 || hidden == 0 || table_columns == 0 {
            return Err(Error::InvalidConfig);
        }
        let total = u64::from(heads)
            .checked_mul(head_dim)
            .ok_or(Error::ResourceLimit)?;
        if total != hidden || hidden > table_columns {
            return Err(Error::InvalidConfig);
        }
        let mut slices = Vec::with_capacity(heads as usize);
        for head in 0..heads {
            let start = u64::from(head)
                .checked_mul(head_dim)
                .ok_or(Error::ResourceLimit)?;
            let end = start.checked_add(head_dim).ok_or(Error::ResourceLimit)?;
            slices.push(ConsumerSlice {
                head,
                columns: (start, end),
            });
        }
        Ok(Self {
            positions,
            head_dim,
            consumed_columns: hidden,
            table_columns,
            slices,
        })
    }

    pub fn slices(&self) -> &[ConsumerSlice] {
        &self.slices
    }

    pub fn positions(&self) -> u32 {
        self.positions
    }

    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }

    pub fn consumed_columns(&self) -> u64 {
        self.consumed_columns
    }

    pub fn table_columns(&self) -> u64 {
        self.table_columns
    }

    /// The stored columns the request does not consume; they are never
    /// reported as used and no head slice covers them.
    pub fn padding_columns(&self) -> u64 {
        self.table_columns - self.consumed_columns
    }

    /// The number of real references of the request: one output entry per
    /// position and consumed column.
    pub fn references(&self) -> Result<u64> {
        u64::from(self.positions)
            .checked_mul(self.consumed_columns)
            .ok_or(Error::ResourceLimit)
    }

    /// Whether a stored column is consumed by some head slice.
    pub fn covers(&self, column: u64) -> bool {
        column < self.consumed_columns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::field::fp_from_u64;

    fn value(word: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(word))
    }

    fn real_size_shape() -> ProjectionShape {
        ProjectionShape {
            matrix_rows: 1024,
            k_lo: 0,
            columns: 1024,
            positions: 2,
        }
    }

    fn activation(shape: &ProjectionShape) -> Vec<Fp3> {
        let count = shape.positions as usize * shape.matrix_rows as usize;
        (0..count).map(|index| value((index as u64) % 17)).collect()
    }

    fn table(shape: &ProjectionShape) -> Vec<Vec<Fp3>> {
        let rows = shape.table_rows() as usize;
        let entries = shape.entries() as usize;
        (0..rows)
            .map(|row| {
                (0..entries)
                    .map(|column| value(((row * 7 + column * 3) % 19) as u64))
                    .collect()
            })
            .collect()
    }

    fn reduce(shape: &ProjectionShape, activation: &[Fp3], table: &[Vec<Fp3>]) -> Vec<Fp3> {
        let matrix_rows = shape.matrix_rows as usize;
        let columns = shape.columns as usize;
        let mut output = vec![Fp3::ZERO; shape.positions as usize * columns];
        for position in 0..shape.positions as usize {
            for column in 0..columns {
                let table_row = column >> shape.k_lo as usize;
                let mut sum = Fp3::ZERO;
                for row in 0..matrix_rows {
                    let index = entry(shape, column as u64, row as u64).unwrap() as usize;
                    sum = sum.add(
                        &activation[position * matrix_rows + row].mul(&table[table_row][index]),
                    );
                }
                output[position * columns + column] = sum;
            }
        }
        output
    }

    #[test]
    fn real_size_projection_reduction_round_trips() {
        let shape = real_size_shape();
        let activation = activation(&shape);
        let table = table(&shape);
        let output = reduce(&shape, &activation, &table);
        let claims = projection_claims(shape, &activation, &output).unwrap();
        check_projection_reduction(shape, &claims.weights, &table, &claims.output).unwrap();
    }

    #[test]
    fn changed_output_or_shape_is_rejected() {
        let shape = real_size_shape();
        let activation = activation(&shape);
        let table = table(&shape);
        let output = reduce(&shape, &activation, &table);
        let mut claims = projection_claims(shape, &activation, &output).unwrap();
        claims.output[0][0] = claims.output[0][0].add(&value(1));
        assert!(
            check_projection_reduction(shape, &claims.weights, &table, &claims.output).is_err()
        );

        let mut bad_shape = shape;
        bad_shape.columns += 1;
        assert!(bad_shape.check().is_err());

        let mut short_activation = activation.clone();
        short_activation.pop();
        assert!(projection_claims(shape, &short_activation, &output).is_err());
        let mut short_output = output.clone();
        short_output.pop();
        assert!(projection_claims(shape, &activation, &short_output).is_err());
    }

    fn consumer_fixture() -> (
        Vec<Vec<Fp3>>,
        Vec<Vec<Fp3>>,
        Vec<Fp3>,
        Vec<Fp3>,
        Vec<Fp3>,
        Vec<Fp3>,
    ) {
        let positions = 3usize;
        let contraction = 8usize;
        let columns = 16usize;
        let (c0, c1) = (4usize, 8usize);
        // A[record][place] as the committed records; M[place][record] is the
        // plan's matrix the LIN claim is about.
        let records: Vec<Vec<Fp3>> = (0..contraction)
            .map(|record| {
                (0..columns)
                    .map(|place| value(((record * 7 + place * 3 + 1) % 19) as u64))
                    .collect()
            })
            .collect();
        let matrix: Vec<Vec<Fp3>> = (0..columns)
            .map(|place| {
                (0..contraction)
                    .map(|record| records[record][place])
                    .collect()
            })
            .collect();
        let x: Vec<Fp3> = (0..positions * contraction)
            .map(|index| value(((index * 5 + 2) % 13) as u64))
            .collect();
        let mut y_raw = vec![Fp3::ZERO; positions * (c1 - c0)];
        for position in 0..positions {
            for place in c0..c1 {
                let mut sum = Fp3::ZERO;
                for record in 0..contraction {
                    sum = sum.add(&x[position * contraction + record].mul(&records[record][place]));
                }
                y_raw[position * (c1 - c0) + (place - c0)] = sum;
            }
        }
        let a: Vec<Fp3> = (0..positions)
            .map(|index| value(index as u64 + 1))
            .collect();
        let b: Vec<Fp3> = (0..c1 - c0).map(|index| value(index as u64 + 4)).collect();
        (matrix, records, x, y_raw, a, b)
    }

    #[test]
    fn the_consumer_reduction_identity_holds_on_the_raw_output() {
        let (matrix, _records, x, y_raw, a, b) = consumer_fixture();
        let reduction = consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a, &b).unwrap();
        assert_eq!(reduction.weights.len(), 16);
        assert!(reduction.weights[..4].iter().all(|weight| weight.is_zero()));
        assert!(reduction.weights[8..].iter().all(|weight| weight.is_zero()));
        assert_eq!(reduction.target, reduction.identity_on(&PlaceRows::new(&matrix)).unwrap());
    }

    #[test]
    fn the_consumer_column_values_carry_the_identity() {
        let (matrix, _records, x, y_raw, a, b) = consumer_fixture();
        let reduction = consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a, &b).unwrap();
        let columns = reduction.column_values(&PlaceRows::new(&matrix)).unwrap();
        assert_eq!(columns.len(), 8);
        assert_eq!(activation_shown(3, 8, &x, &a).unwrap(), reduction.shown);
        let mut total = Fp3::ZERO;
        for (shown, column) in reduction.shown.iter().zip(&columns) {
            total = total.add(&shown.mul(column));
        }
        assert_eq!(total, reduction.identity_on(&PlaceRows::new(&matrix)).unwrap());
    }

    #[test]
    fn the_raw_target_is_the_reduction_target() {
        let (_matrix, _records, x, y_raw, a, b) = consumer_fixture();
        let reduction = consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a, &b).unwrap();
        assert_eq!(raw_target(3, 4, &y_raw, &a, &b).unwrap(), reduction.target);
        let mut changed = y_raw.clone();
        changed[0] = changed[0].add(&value(1));
        assert_ne!(
            raw_target(3, 4, &changed, &a, &b).unwrap(),
            reduction.target
        );
        assert!(raw_target(0, 4, &y_raw, &a, &b).is_err());
        assert!(raw_target(3, 4, &y_raw[..11], &a, &b).is_err());
        assert!(raw_target(3, 4, &y_raw, &a[..2], &b).is_err());
        assert!(raw_target(3, 4, &y_raw, &a, &b[..3]).is_err());
    }

    #[test]
    fn consumer_maps_refuse_broken_shapes() {
        let (matrix, _records, x, y_raw, a, b) = consumer_fixture();
        let reduction = consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a, &b).unwrap();
        assert!(reduction.column_values(&PlaceRows::new(&matrix[..15])).is_err());
        let mut broken = matrix.clone();
        broken[0].pop();
        assert!(reduction.column_values(&PlaceRows::new(&broken)).is_err());
        assert!(activation_shown(3, 8, &x[..23], &a).is_err());
        assert!(activation_shown(3, 8, &x, &a[..2]).is_err());
        assert!(activation_shown(0, 8, &x, &a).is_err());
    }

    #[test]
    fn the_requantized_output_does_not_satisfy_the_raw_identity() {
        let (matrix, _records, x, mut y_raw, a, b) = consumer_fixture();
        let honest = consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a, &b).unwrap();
        assert_eq!(honest.target, honest.identity_on(&PlaceRows::new(&matrix)).unwrap());
        // A rounding or truncation change to one raw entry breaks the
        // identity; the reduction is over the raw accumulator, never the
        // requantized output.
        y_raw[1] = y_raw[1].add(&value(1));
        let changed = consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a, &b).unwrap();
        assert_ne!(changed.target, changed.identity_on(&PlaceRows::new(&matrix)).unwrap());
    }

    #[test]
    fn a_consumer_reduction_fault_is_refused() {
        let (_matrix, _records, x, y_raw, a, b) = consumer_fixture();
        assert!(consumer_reduction(3, 8, 16, (8, 10), &x, &y_raw, &a, &b).is_err());
        assert!(consumer_reduction(3, 8, 16, (4, 20), &x, &y_raw, &a, &b).is_err());
        assert!(consumer_reduction(4, 8, 16, (4, 8), &x, &y_raw, &a, &b).is_err());
        assert!(consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a[..2], &b).is_err());
        assert!(consumer_reduction(3, 8, 16, (4, 8), &x, &y_raw, &a, &b[..3]).is_err());
    }

    #[test]
    fn the_production_coverage_does_not_claim_the_padding_columns() {
        // The real GPT-2 hidden width is 768 over a 1024-column stored table;
        // twelve 64-column head slices consume [0,768) and never [768,1024).
        let coverage = ConsumerCoverage::new(512, 64, 12, 768, 1024).unwrap();
        assert_eq!(coverage.slices().len(), 12);
        assert_eq!(coverage.consumed_columns(), 768);
        assert_eq!(coverage.padding_columns(), 256);
        assert_eq!(coverage.references().unwrap(), 512 * 768);
        assert!(coverage.covers(767));
        assert!(!coverage.covers(768));
        for (index, slice) in coverage.slices().iter().enumerate() {
            assert_eq!(slice.head as usize, index);
            assert_eq!(
                slice.columns,
                ((index as u64) * 64, (index as u64 + 1) * 64)
            );
        }
        let sixteen = ConsumerCoverage::new(16, 64, 12, 768, 1024).unwrap();
        assert_eq!(sixteen.references().unwrap(), 16 * 768);
    }

    #[test]
    fn a_coverage_that_claims_padding_or_mismatches_is_refused() {
        // Sixteen heads would claim the stored pad columns as consumed.
        assert!(ConsumerCoverage::new(16, 64, 16, 768, 1024).is_err());
        // The consumed width cannot exceed the stored width.
        assert!(ConsumerCoverage::new(16, 64, 12, 768, 512).is_err());
        assert!(ConsumerCoverage::new(0, 64, 12, 768, 1024).is_err());
        assert!(ConsumerCoverage::new(16, 0, 12, 768, 1024).is_err());
        assert!(ConsumerCoverage::new(16, 64, 0, 768, 1024).is_err());
        assert!(ConsumerCoverage::new(16, 64, 12, 0, 1024).is_err());
        assert!(ConsumerCoverage::new(16, 64, 12, 768, 0).is_err());
    }
}
