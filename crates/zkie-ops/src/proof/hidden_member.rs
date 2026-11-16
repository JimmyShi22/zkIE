//! Pending claims and reduced-witness tables of the hidden consumer chain.
//!
//! A consumer's sub-proof leaves linear claims on the committed matrix: row
//! weights over the matrix rows, one equality point over the columns, a
//! masked target and the index-mask entries that hide the target. One claim
//! is `<g (x) eq(psi, .), M> + sum beta_j c_j = v`. A batch of one passes the
//! claim through unchanged; this chain's consumer is a batch of one, so no
//! batching sumcheck runs and the coefficient is one.
//!
//! The consumer's other claims read the matrix only through one reduced
//! witness: a table whose entries are the scalar products of a run of matrix
//! cells with a point. The table fixes per entry the matrix, the run length,
//! the point and the entry's offset; the points are set before the witness
//! root is committed, no entry may be read before the root, and the batching
//! challenge is drawn only after it. Every cell of a table's matrix part is
//! covered by exactly one read with a non-zero weight.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::field::Fp3;

/// One block of row weights inside the matrix row cube: a single row's weight,
/// or an aligned block of rows whose values are a slice of the consumer's
/// weight vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightBlock {
    /// One row, with its weight read at construction.
    Scalar { row: u64, value: Fp3 },
    /// `2^log_rows` consecutive rows beginning at an aligned `first_row`.
    Dense {
        first_row: u64,
        log_rows: u32,
        values: Vec<Fp3>,
    },
}

impl WeightBlock {
    fn rows(&self) -> u64 {
        match self {
            WeightBlock::Scalar { .. } => 1,
            WeightBlock::Dense { log_rows, .. } => 1u64 << log_rows,
        }
    }
}

/// Cuts a dense weight range into aligned blocks of rows, at most two of each
/// size: a block of two or more rows keeps a slice of the shared vector, and a
/// block of one row keeps its value. The range must lie inside the row cube.
pub fn stretch_weights(k_row: u32, first_row: u64, values: &[Fp3]) -> Result<Vec<WeightBlock>> {
    if k_row >= 63 || values.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let cube = 1u64 << k_row;
    let end = first_row
        .checked_add(values.len() as u64)
        .ok_or(Error::ResourceLimit)?;
    if end > cube {
        return Err(Error::BoundViolation);
    }
    let mut blocks = Vec::new();
    let mut at = first_row;
    while at < end {
        let mut log_rows = 0u32;
        while log_rows < k_row
            && at & ((2u64 << log_rows) - 1) == 0
            && at + (2u64 << log_rows) <= end
        {
            log_rows += 1;
        }
        if log_rows == 0 {
            blocks.push(WeightBlock::Scalar {
                row: at,
                value: values[(at - first_row) as usize],
            });
            at += 1;
        } else {
            let length = 1usize << log_rows;
            let start = (at - first_row) as usize;
            blocks.push(WeightBlock::Dense {
                first_row: at,
                log_rows,
                values: values[start..start + length].to_vec(),
            });
            at += length as u64;
        }
    }
    Ok(blocks)
}

/// One index-mask entry consumed by a pending claim: the entry's oracle, its
/// slot in that message, and the coefficient that multiplies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskTerm {
    pub oracle: u64,
    pub slot: u64,
    pub coeff: Fp3,
}

/// One pending claim left by a consumer's sub-proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingRow {
    pub weights: Vec<WeightBlock>,
    pub psi: Vec<Fp3>,
    pub target: Fp3,
    pub masks: Vec<MaskTerm>,
}

impl PendingRow {
    /// The row cube's size in bits, from the weights' extent: the smallest
    /// cube that holds every block.
    pub fn row_bits(&self) -> Result<u32> {
        let mut bits = 0u32;
        for block in &self.weights {
            let end = match block {
                WeightBlock::Scalar { row, .. } => row.saturating_add(1),
                WeightBlock::Dense {
                    first_row,
                    log_rows,
                    ..
                } => first_row.saturating_add(1u64 << log_rows),
            };
            let need = if end <= 1 {
                0
            } else {
                64 - (end - 1).leading_zeros()
            };
            bits = bits.max(need);
        }
        Ok(bits)
    }
}

/// The index-mask message's trusted shape, as the pending claims must name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexMaskRef {
    pub oracle: u64,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub answers: u32,
    pub mask_code: CodeParams,
}

/// Checks the pending claims against the matrix geometry and the index-mask
/// message. Both sides run this first; a mismatch is reported with the rule
/// that failed.
pub fn rows_fit(
    index: &IndexMaskRef,
    own_mask_code: CodeParams,
    k_row: u32,
    k_col: u32,
    pending: &[PendingRow],
) -> std::result::Result<(), &'static str> {
    if index.main_rows != 0
        || index.mask_rows == 0
        || index.answers == 0
        || index.mask_code != own_mask_code
    {
        return Err("the index-mask message is no sampled message of mask rows alone under the own group's mask code");
    }
    if pending.is_empty() {
        return Err("no pending row");
    }
    for row in pending {
        if row.psi.len() != k_col as usize {
            return Err("a pending row's column point does not fit the matrix's columns");
        }
        if row.weights.iter().map(WeightBlock::rows).sum::<u64>() == 0 {
            return Err("a pending row has no row weights");
        }
        for block in &row.weights {
            match block {
                WeightBlock::Scalar { row, .. } => {
                    if *row >= (1u64 << k_row) {
                        return Err("a pending row's weights do not fit the matrix's rows");
                    }
                }
                WeightBlock::Dense {
                    first_row,
                    log_rows,
                    values,
                } => {
                    if *log_rows >= k_row
                        || *first_row & ((1u64 << log_rows) - 1) != 0
                        || first_row
                            .checked_add(1u64 << log_rows)
                            .map(|end| end > (1u64 << k_row))
                            .unwrap_or(true)
                        || values.len() != (1usize << log_rows)
                    {
                        return Err("a pending row's weights do not fit the matrix's rows");
                    }
                }
            }
        }
        for term in &row.masks {
            if term.oracle != index.oracle {
                return Err("a pending row names a mask entry outside the index-mask message");
            }
        }
    }
    Ok(())
}

/// The claim that LIN's first part consumes: one row-weight vector, one
/// column point, one target and the mask terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombinedClaim {
    pub weights: Vec<WeightBlock>,
    pub psi: Vec<Fp3>,
    pub target: Fp3,
    pub masks: Vec<MaskTerm>,
}

/// The claim a batch leaves. A chain of one pending row passes it through
/// with coefficient one; more than one is not part of this consumer's chain
/// and is refused instead of being batched here.
pub fn claim_of(pending: &[PendingRow], coefficients: &[Fp3]) -> Result<CombinedClaim> {
    if pending.is_empty() || coefficients.len() != pending.len() {
        return Err(Error::InvalidConfig);
    }
    if pending.len() > 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    if coefficients[0] != Fp3::from_u64(1) {
        return Err(Error::InvalidConfig);
    }
    let row = &pending[0];
    Ok(CombinedClaim {
        weights: row.weights.clone(),
        psi: row.psi.clone(),
        target: row.target,
        masks: row.masks.clone(),
    })
}

/// The multilinear extension of one matrix row at the column point: bit `j`
/// of the column index pairs with `psi[j]`, the lowest column bit first.
pub fn row_at(row: &[Fp3], psi: &[Fp3]) -> Result<Fp3> {
    if psi.is_empty() || psi.len() >= 32 || row.len() != (1usize << psi.len()) {
        return Err(Error::ShapeMismatch);
    }
    let mut value = Fp3::ZERO;
    for (index, entry) in row.iter().enumerate() {
        let mut factor = Fp3::ONE;
        for (bit, coordinate) in psi.iter().enumerate() {
            let chosen = if (index >> bit) & 1 == 1 {
                *coordinate
            } else {
                Fp3::ONE.sub(coordinate)
            };
            factor = factor.mul(&chosen);
        }
        value = value.add(&factor.mul(entry));
    }
    Ok(value)
}

/// The equality weight of one record index at the pending column point.
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

/// The pending claim's left side: the row weights against the matrix at the
/// column point, plus the index-mask entries. The matrix is streamed one
/// stored record at a time; the memory stays one row of places and the
/// stretched weight vector.
pub fn pending_value(
    pending: &PendingRow,
    matrix: &dyn crate::proof::projection_claim::ConsumerMatrixColumns,
    mask_value: &dyn Fn(u64, u64) -> Result<Fp3>,
) -> Result<Fp3> {
    if pending.weights.is_empty() || pending.psi.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let cube = 1u64 << pending.row_bits()?;
    let places = usize::try_from(matrix.places()).map_err(|_| Error::ResourceLimit)?;
    if cube > matrix.places() {
        return Err(Error::ShapeMismatch);
    }
    let records = usize::try_from(matrix.records()).map_err(|_| Error::ResourceLimit)?;
    if records != 1usize << pending.psi.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut weights = vec![Fp3::ZERO; places];
    for block in &pending.weights {
        match block {
            WeightBlock::Scalar { row, value: weight } => {
                if *row >= cube {
                    return Err(Error::BoundViolation);
                }
                weights[*row as usize] = *weight;
            }
            WeightBlock::Dense {
                first_row,
                log_rows,
                values,
            } => {
                if values.len() as u64 != 1u64 << log_rows {
                    return Err(Error::ShapeMismatch);
                }
                for (offset, weight) in values.iter().enumerate() {
                    let row = first_row
                        .checked_add(offset as u64)
                        .ok_or(Error::ResourceLimit)?;
                    if row >= cube {
                        return Err(Error::BoundViolation);
                    }
                    weights[row as usize] = *weight;
                }
            }
        }
    }
    let mut value = Fp3::ZERO;
    let mut row = vec![Fp3::ZERO; places];
    for record in 0..records {
        matrix.row(record as u64, &mut row)?;
        let mut total = Fp3::ZERO;
        for (weight, entry) in weights.iter().zip(&row) {
            if !weight.is_zero() {
                total = total.add(&weight.mul(entry));
            }
        }
        value = value.add(&eq_at(&pending.psi, record as u64).mul(&total));
    }
    for term in &pending.masks {
        value = value.add(&term.coeff.mul(&mask_value(term.oracle, term.slot)?));
    }
    Ok(value)
}

/// Checks that the plan's member claim is the claim the request chain derived
/// from its verified prefix: the same row weights, the same column point, the
/// same target and the same mask terms, in the same order. A plan claim that
/// differs is another source and is refused rather than trusted.
pub fn check_member_claim(plan: &PendingRow, chain: &PendingRow) -> Result<()> {
    if plan.weights != chain.weights
        || plan.psi != chain.psi
        || plan.target != chain.target
        || plan.masks != chain.masks
    {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

/// Checks the pending claim: the left side must equal the target.
pub fn check_pending(
    pending: &PendingRow,
    matrix: &dyn crate::proof::projection_claim::ConsumerMatrixColumns,
    mask_value: &dyn Fn(u64, u64) -> Result<Fp3>,
) -> Result<()> {
    if pending_value(pending, matrix, mask_value)? != pending.target {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The merge that batches a sub-proof's weight claims into one pending claim.
/// The claims' end values are registered as slots first; one challenge batches
/// them, a product-sumcheck over the matrix column bits gives the point, and
/// the masked end value leaves one pending claim. The plan fixes the order and
/// the relations; the transcript supplies the challenge values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MergePhase {
    Empty,
    Claims,
    Batched,
    Pointed,
    Ended,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergePlan {
    claims: u32,
    column_bits: u32,
    phase: MergePhase,
    claim_slots: Vec<u64>,
    batch: Vec<Fp3>,
    point: Vec<Fp3>,
    end_slot: u64,
    beta: Fp3,
    target: Fp3,
}

impl MergePlan {
    pub fn new(claims: u32, column_bits: u32) -> Result<Self> {
        if claims == 0 || column_bits == 0 || column_bits > 62 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            claims,
            column_bits,
            phase: MergePhase::Empty,
            claim_slots: Vec::new(),
            batch: Vec::new(),
            point: Vec::new(),
            end_slot: 0,
            beta: Fp3::ZERO,
            target: Fp3::ZERO,
        })
    }

    pub fn claims(&self) -> u32 {
        self.claims
    }

    pub fn column_bits(&self) -> u32 {
        self.column_bits
    }

    pub fn batch(&self) -> &[Fp3] {
        &self.batch
    }

    /// The merge's point, the highest column bit first, as the transcript
    /// produced it.
    pub fn point(&self) -> &[Fp3] {
        &self.point
    }

    pub fn beta(&self) -> Fp3 {
        self.beta
    }

    pub fn target(&self) -> Fp3 {
        self.target
    }

    /// Registers the claims' end-value slots, once, before any challenge.
    pub fn register_claims(&mut self, slots: &[u64]) -> Result<()> {
        if self.phase != MergePhase::Empty || slots.len() != self.claims as usize {
            return Err(Error::InvalidConfig);
        }
        let mut sorted = slots.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != slots.len() {
            return Err(Error::InvalidConfig);
        }
        self.claim_slots = slots.to_vec();
        self.phase = MergePhase::Claims;
        Ok(())
    }

    /// Draws the challenge whose powers batch the claims.
    pub fn draw_batch(&mut self, batch: &[Fp3]) -> Result<()> {
        if self.phase != MergePhase::Claims || batch.len() != self.claims as usize {
            return Err(Error::InvalidConfig);
        }
        self.batch = batch.to_vec();
        self.phase = MergePhase::Batched;
        Ok(())
    }

    /// Fixes the sumcheck's point over the matrix's column bits.
    pub fn set_point(&mut self, point: &[Fp3]) -> Result<()> {
        if self.phase != MergePhase::Batched || point.len() != self.column_bits as usize {
            return Err(Error::InvalidConfig);
        }
        self.point = point.to_vec();
        self.phase = MergePhase::Pointed;
        Ok(())
    }

    /// Records the masked end value and its mask factor. The end value's slot
    /// is not one of the claims' slots: no claim of the sub-proof may name it.
    pub fn end(&mut self, end_slot: u64, beta: Fp3, target: Fp3) -> Result<()> {
        if self.phase != MergePhase::Pointed || beta.is_zero() {
            return Err(Error::InvalidConfig);
        }
        if self.claim_slots.contains(&end_slot) {
            return Err(Error::InvalidConfig);
        }
        self.end_slot = end_slot;
        self.beta = beta;
        self.target = target;
        self.phase = MergePhase::Ended;
        Ok(())
    }

    /// The pending claim the merge leaves, with the caller's row weights and
    /// the index-mask slot the target is masked by.
    pub fn pending_claim(
        &self,
        weights: Vec<WeightBlock>,
        index_oracle: u64,
        mask_slot: u64,
    ) -> Result<PendingRow> {
        if self.phase != MergePhase::Ended || weights.is_empty() {
            return Err(Error::InvalidConfig);
        }
        // The point's coordinates are consumed lowest column bit first.
        let mut psi = self.point.clone();
        psi.reverse();
        Ok(PendingRow {
            weights,
            psi,
            target: self.target,
            masks: vec![MaskTerm {
                oracle: index_oracle,
                slot: mask_slot,
                coeff: self.beta,
            }],
        })
    }
}

/// One reduced-witness table: a matrix part whose entries are the run scalar
/// products. `phi` is the point of the run; it is empty until the point is
/// known, and every table's point must be set before the witness root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionTable {
    pub matrix: u64,
    pub parts: u64,
    pub run_length: u64,
    pub rows_per_part: u64,
    pub columns: u64,
    pub at: u64,
    pub entries: u64,
    pub phi: Vec<Fp3>,
    pub later: bool,
}

/// The scalar product of one run of matrix cells with the run's point.
pub fn entry_value(cells: &[Fp3], phi: &[Fp3]) -> Result<Fp3> {
    if cells.is_empty() || cells.len() != phi.len() {
        return Err(Error::InvalidConfig);
    }
    let mut sum = Fp3::ZERO;
    for (cell, weight) in cells.iter().zip(phi) {
        sum = sum.add(&cell.mul(weight));
    }
    Ok(sum)
}

/// The entries of one table in row-major then run order: row `r`, run `q`
/// covers columns `q * run_length .. (q + 1) * run_length` of that row. The
/// matrix is row-major with `columns` entries per row; every entry is the run's
/// scalar product with the point.
pub fn table_entries(
    cells: &[Fp3],
    rows: u64,
    columns: u64,
    run_length: u64,
    phi: &[Fp3],
) -> Result<Vec<Fp3>> {
    if rows == 0 || columns == 0 || run_length == 0 || columns % run_length != 0 {
        return Err(Error::InvalidConfig);
    }
    if phi.len() as u64 != run_length {
        return Err(Error::InvalidConfig);
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(Error::ResourceLimit)?;
    if cells.len() as u64 != expected {
        return Err(Error::ShapeMismatch);
    }
    let rows = usize::try_from(rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    let run = usize::try_from(run_length).map_err(|_| Error::ResourceLimit)?;
    let mut entries = Vec::with_capacity(rows * (columns / run));
    for row in 0..rows {
        let base = row.checked_mul(columns).ok_or(Error::ResourceLimit)?;
        for q in 0..columns / run {
            let start = base + q * run;
            entries.push(entry_value(&cells[start..start + run], phi)?);
        }
    }
    Ok(entries)
}

/// One read of the reduced witness: `weights[i]` multiplies entry
/// `first_entry + i` of the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionRead {
    pub table: usize,
    pub first_entry: u64,
    pub weights: Vec<Fp3>,
}

/// The reads of the reduced witness as one plan object: the caller fixes the
/// per-entry weights, and the set is checked against the tables' geometry and
/// coverage before any read is used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionReadSet {
    reads: Vec<ReductionRead>,
}

impl ReductionReadSet {
    pub fn new(reads: Vec<ReductionRead>) -> Self {
        Self { reads }
    }

    pub fn reads(&self) -> &[ReductionRead] {
        &self.reads
    }

    /// One table's reads, cut into segments of `segment` entries: `weights[i]`
    /// is the coefficient of entry `first_entry + i`. The caller supplies the
    /// coefficients in entry order.
    pub fn segmented(
        table: usize,
        first_entry: u64,
        segment: u64,
        weights: &[Fp3],
    ) -> Result<Self> {
        if segment == 0 || weights.is_empty() || weights.len() as u64 % segment != 0 {
            return Err(Error::InvalidConfig);
        }
        let mut reads = Vec::with_capacity(weights.len() / segment as usize);
        for (index, chunk) in weights.chunks(segment as usize).enumerate() {
            let offset = (index as u64)
                .checked_mul(segment)
                .ok_or(Error::ResourceLimit)?;
            reads.push(ReductionRead {
                table,
                first_entry: first_entry
                    .checked_add(offset)
                    .ok_or(Error::ResourceLimit)?,
                weights: chunk.to_vec(),
            });
        }
        Ok(Self { reads })
    }

    /// Checks the set against the witness plan: every entry of every table is
    /// covered exactly once with a non-zero weight, and no read leaves a table.
    pub fn cover(&self, plan: &ReductionWitnessPlan) -> Result<()> {
        plan.check_reads(&self.reads)
    }
}

/// Where the reduced witness is in its order: built, points set, root
/// committed, batching challenge drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WitnessPhase {
    Built,
    Points,
    Committed,
    Batched,
}

/// The reduced witness's trusted plan and order. The plan allocates each
/// table's offsets, fixes the points, enforces that the root follows every
/// point and that the batching challenge follows the root, and checks that
/// the reads cover each table exactly once with a non-zero weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionWitnessPlan {
    tables: Vec<ReductionTable>,
    next_at: u64,
    phase: WitnessPhase,
    root: Option<Digest>,
    batch_challenge: Option<Fp3>,
}

impl ReductionWitnessPlan {
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            next_at: 0,
            phase: WitnessPhase::Built,
            root: None,
            batch_challenge: None,
        }
    }

    /// Builds the runtime plan from settled tables: the offsets must follow
    /// each other from zero, the entry counts must match the geometry, and
    /// every point must be set.
    pub fn from_tables(tables: Vec<ReductionTable>) -> Result<Self> {
        let mut next_at = 0u64;
        for table in &tables {
            if table.phi.is_empty()
                || table.phi.len() as u64 != table.run_length
                || table.parts == 0
                || table.run_length == 0
                || table.columns == 0
                || table.columns % table.run_length != 0
                || table.rows_per_part == 0
                || table.at != next_at
            {
                return Err(Error::InvalidConfig);
            }
            let expected = table
                .rows_per_part
                .checked_mul(table.columns / table.run_length)
                .and_then(|value| value.checked_mul(table.parts))
                .ok_or(Error::ResourceLimit)?;
            if table.entries != expected {
                return Err(Error::InvalidConfig);
            }
            next_at = next_at
                .checked_add(table.entries)
                .ok_or(Error::ResourceLimit)?;
        }
        let mut plan = Self {
            tables,
            next_at,
            phase: WitnessPhase::Built,
            root: None,
            batch_challenge: None,
        };
        plan.points_known()?;
        Ok(plan)
    }

    pub fn tables(&self) -> &[ReductionTable] {
        &self.tables
    }

    /// The total number of entries of all tables.
    pub fn total_entries(&self) -> u64 {
        self.next_at
    }

    pub fn root(&self) -> Option<Digest> {
        self.root
    }

    pub fn batch_challenge(&self) -> Option<Fp3> {
        self.batch_challenge
    }

    /// Adds one table. Its point is supplied now for a table whose point is
    /// already known, or left empty and set later; `later` records which.
    pub fn add_table(
        &mut self,
        matrix: u64,
        parts: u64,
        run_length: u64,
        rows_per_part: u64,
        columns: u64,
        phi: Vec<Fp3>,
        later: bool,
    ) -> Result<usize> {
        if self.phase != WitnessPhase::Built {
            return Err(Error::InvalidConfig);
        }
        if parts == 0
            || run_length == 0
            || columns == 0
            || columns % run_length != 0
            || rows_per_part == 0
            || (!phi.is_empty() && phi.len() as u64 != run_length)
        {
            return Err(Error::InvalidConfig);
        }
        let per_part = rows_per_part
            .checked_mul(columns / run_length)
            .ok_or(Error::ResourceLimit)?;
        let entries = per_part
            .checked_mul(parts)
            .ok_or(Error::ResourceLimit)?;
        let at = self.next_at;
        self.next_at = self
            .next_at
            .checked_add(entries)
            .ok_or(Error::ResourceLimit)?;
        self.tables.push(ReductionTable {
            matrix,
            parts,
            run_length,
            rows_per_part,
            columns,
            at,
            entries,
            phi,
            later,
        });
        Ok(self.tables.len() - 1)
    }

    /// Sets the point of a table that was added without one. Only before the
    /// root; a table whose point is already set is not overwritten.
    pub fn set_point(&mut self, table: usize, phi: Vec<Fp3>) -> Result<()> {
        if self.phase != WitnessPhase::Built {
            return Err(Error::InvalidConfig);
        }
        let entry = self.tables.get_mut(table).ok_or(Error::InvalidIndex)?;
        if !entry.phi.is_empty()
            || phi.len() as u64 != entry.run_length
            || !entry.later
        {
            return Err(Error::InvalidConfig);
        }
        entry.phi = phi;
        Ok(())
    }

    /// Every point is known: fixes the order so the root can follow. A table
    /// whose point is still empty refuses here.
    pub fn points_known(&mut self) -> Result<()> {
        if self.phase != WitnessPhase::Built {
            return Err(Error::InvalidConfig);
        }
        if self.tables.iter().any(|table| table.phi.is_empty()) {
            return Err(Error::InvalidConfig);
        }
        self.phase = WitnessPhase::Points;
        Ok(())
    }

    /// Commits the witness root. Only after every point, and only once.
    pub fn commit_root(&mut self, root: Digest) -> Result<()> {
        if self.phase != WitnessPhase::Points || root == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        self.root = Some(root);
        self.phase = WitnessPhase::Committed;
        Ok(())
    }

    /// Draws the challenge that batches the claims. Only after the root.
    pub fn draw_batch(&mut self, challenge: Fp3) -> Result<()> {
        if self.phase != WitnessPhase::Committed || challenge.is_zero() {
            return Err(Error::InvalidConfig);
        }
        self.batch_challenge = Some(challenge);
        self.phase = WitnessPhase::Batched;
        Ok(())
    }

    /// The witness entry a claim reads, as an absolute offset. Only after the
    /// root, and only inside a table.
    pub fn read(&self, table: usize, entry: u64) -> Result<u64> {
        if self.phase != WitnessPhase::Committed && self.phase != WitnessPhase::Batched {
            return Err(Error::InvalidConfig);
        }
        let table = self.tables.get(table).ok_or(Error::InvalidIndex)?;
        if entry >= table.entries {
            return Err(Error::InvalidIndex);
        }
        Ok(table.at + entry)
    }

    /// Checks that the reads cover every entry of every table exactly once
    /// with a non-zero weight, and that no read leaves its table.
    pub fn check_reads(&self, reads: &[ReductionRead]) -> Result<()> {
        if self.phase != WitnessPhase::Committed && self.phase != WitnessPhase::Batched {
            return Err(Error::InvalidConfig);
        }
        let total = usize::try_from(self.next_at).map_err(|_| Error::ResourceLimit)?;
        let mut covered = vec![0u32; total];
        for read in reads {
            let table = self.tables.get(read.table).ok_or(Error::InvalidIndex)?;
            if read.weights.is_empty() {
                return Err(Error::InvalidConfig);
            }
            let end = read
                .first_entry
                .checked_add(read.weights.len() as u64)
                .ok_or(Error::ResourceLimit)?;
            if end > table.entries {
                return Err(Error::InvalidIndex);
            }
            for (index, weight) in read.weights.iter().enumerate() {
                if weight.is_zero() {
                    return Err(Error::InvalidConfig);
                }
                let at = usize::try_from(table.at + read.first_entry + index as u64)
                    .map_err(|_| Error::ResourceLimit)?;
                let slot = covered.get_mut(at).ok_or(Error::InvalidIndex)?;
                *slot = slot.checked_add(1).ok_or(Error::ResourceLimit)?;
            }
        }
        if covered.iter().any(|count| *count != 1) {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::projection_claim::PlaceRows;

    fn fp(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    fn mask() -> IndexMaskRef {
        IndexMaskRef {
            oracle: 60,
            main_rows: 0,
            mask_rows: 4,
            answers: 1,
            mask_code: CodeParams {
                log_n: 14,
                ell: 256,
                r: 8,
            },
        }
    }

    fn pending() -> PendingRow {
        PendingRow {
            weights: vec![WeightBlock::Dense {
                first_row: 0,
                log_rows: 1,
                values: vec![fp(1), fp(2)],
            }],
            psi: vec![fp(5), fp(6), fp(7)],
            target: fp(9),
            masks: vec![MaskTerm {
                oracle: 60,
                slot: 3,
                coeff: fp(11),
            }],
        }
    }

    #[test]
    fn weights_are_cut_into_aligned_blocks() {
        let values: Vec<Fp3> = (0..6).map(fp).collect();
        let blocks = stretch_weights(4, 0, &values).unwrap();
        // 0..4 is one aligned block of four, 4 is aligned to two, 5 is one row.
        assert_eq!(
            blocks,
            vec![
                WeightBlock::Dense {
                    first_row: 0,
                    log_rows: 2,
                    values: vec![fp(0), fp(1), fp(2), fp(3)],
                },
                WeightBlock::Dense {
                    first_row: 4,
                    log_rows: 1,
                    values: vec![fp(4), fp(5)],
                },
            ]
        );
        let blocks = stretch_weights(3, 5, &[fp(7), fp(8), fp(9)]).unwrap();
        // Row 5 is aligned to one row only; rows 6 and 7 begin an aligned block
        // of two.
        assert_eq!(
            blocks,
            vec![
                WeightBlock::Scalar {
                    row: 5,
                    value: fp(7)
                },
                WeightBlock::Dense {
                    first_row: 6,
                    log_rows: 1,
                    values: vec![fp(8), fp(9)],
                },
            ]
        );
        // A range outside the row cube is refused.
        assert!(stretch_weights(2, 2, &[fp(1), fp(2), fp(3)]).is_err());
        assert!(stretch_weights(4, 15, &[fp(1), fp(2)]).is_err());
    }

    #[test]
    fn pending_rows_fit_the_geometry_and_the_mask() {
        let rows = [pending()];
        assert!(rows_fit(&mask(), mask().mask_code, 2, 3, &rows).is_ok());
        // The index-mask message must be mask rows only, sampled, under the own
        // group's mask code.
        let mut bad = mask();
        bad.main_rows = 1;
        assert!(rows_fit(&bad, mask().mask_code, 2, 3, &rows).is_err());
        let mut bad = mask();
        bad.mask_rows = 0;
        assert!(rows_fit(&bad, mask().mask_code, 2, 3, &rows).is_err());
        let mut bad = mask();
        bad.answers = 0;
        assert!(rows_fit(&bad, mask().mask_code, 2, 3, &rows).is_err());
        let mut bad = mask();
        bad.mask_code.r += 1;
        assert!(rows_fit(&bad, mask().mask_code, 2, 3, &rows).is_err());
        // No pending row, a point of another width, a mask entry of another
        // message, and a weight range outside the cube are each refused.
        assert!(rows_fit(&mask(), mask().mask_code, 2, 3, &[]).is_err());
        assert!(rows_fit(&mask(), mask().mask_code, 2, 4, &rows).is_err());
        let mut bad = pending();
        bad.masks[0].oracle = 61;
        assert!(rows_fit(&mask(), mask().mask_code, 2, 3, &[bad.clone()]).is_err());
        let mut bad = pending();
        bad.weights[0] = WeightBlock::Dense {
            first_row: 0,
            log_rows: 3,
            values: vec![fp(1); 8],
        };
        assert!(rows_fit(&mask(), mask().mask_code, 2, 3, &[bad]).is_err());
    }

    #[test]
    fn one_pending_row_passes_through_and_a_batch_is_refused() {
        let claim = claim_of(&[pending()], &[fp(1)]).unwrap();
        assert_eq!(claim.target, fp(9));
        assert_eq!(claim.psi.len(), 3);
        assert_eq!(claim.masks.len(), 1);
        // The coefficient of a chain of one is one; another coefficient, and a
        // second claim, are not this consumer's chain.
        assert!(claim_of(&[pending()], &[fp(2)]).is_err());
        assert!(claim_of(&[pending(), pending()], &[fp(1), fp(1)]).is_err());
        assert!(claim_of(&[], &[]).is_err());
    }

    fn witness() -> ReductionWitnessPlan {
        let mut plan = ReductionWitnessPlan::new();
        // One table of two parts, runs of four cells over two rows of eight.
        let at = plan
            .add_table(0, 2, 4, 2, 8, vec![fp(1), fp(2), fp(3), fp(4)], false)
            .unwrap();
        assert_eq!(at, 0);
        assert_eq!(plan.tables()[0].entries, 8);
        plan.points_known().unwrap();
        plan
    }

    #[test]
    fn the_witness_order_holds() {
        let mut plan = witness();
        assert!(plan.read(0, 0).is_err());
        assert!(plan.commit_root(Digest([0x11; 32])).is_ok());
        assert!(plan.read(0, 7).is_ok());
        assert!(plan.read(0, 8).is_err());
        assert!(plan.draw_batch(fp(3)).is_ok());
        assert_eq!(plan.batch_challenge(), Some(fp(3)));
        // The root is fixed once, and the challenge only after it.
        assert!(plan.commit_root(Digest([0x12; 32])).is_err());
        assert!(plan.draw_batch(fp(4)).is_err());
    }

    #[test]
    fn a_later_point_must_be_set_before_the_root() {
        let mut plan = ReductionWitnessPlan::new();
        plan.add_table(1, 1, 2, 1, 4, Vec::new(), true).unwrap();
        assert!(plan.points_known().is_err());
        assert!(plan.commit_root(Digest([0x21; 32])).is_err());
        assert!(plan.set_point(0, vec![fp(1)]).is_err());
        plan.set_point(0, vec![fp(1), fp(2)]).unwrap();
        plan.points_known().unwrap();
        assert!(plan.commit_root(Digest([0x22; 32])).is_ok());
        // A point of another length and a table whose point is already set are
        // not overwritten.
        let mut plan = ReductionWitnessPlan::new();
        plan.add_table(1, 1, 2, 1, 4, Vec::new(), true).unwrap();
        assert!(plan.set_point(0, vec![fp(1)]).is_err());
        let mut plan = ReductionWitnessPlan::new();
        plan.add_table(1, 1, 2, 1, 4, vec![fp(1), fp(2)], false)
            .unwrap();
        assert!(plan.set_point(0, vec![fp(3), fp(4)]).is_err());
    }

    #[test]
    fn the_merge_orders_its_parts_and_builds_one_pending_claim() {
        let mut merge = MergePlan::new(2, 3).unwrap();
        assert!(merge.draw_batch(&[fp(1), fp(2)]).is_err());
        merge.register_claims(&[10, 12]).unwrap();
        // Distinct slots and the exact count.
        assert!(MergePlan::new(2, 3).unwrap().register_claims(&[10]).is_err());
        let mut dup = MergePlan::new(2, 3).unwrap();
        assert!(dup.register_claims(&[10, 10]).is_err());
        merge.draw_batch(&[fp(1), fp(2)]).unwrap();
        assert!(merge.set_point(&[fp(1), fp(2)]).is_err());
        merge.set_point(&[fp(7), fp(8), fp(9)]).unwrap();
        // The merge's end value may not reuse a claim slot, and its factor is
        // not zero.
        assert!(merge.end(10, fp(5), fp(4)).is_err());
        assert!(merge.end(20, Fp3::ZERO, fp(4)).is_err());
        merge.end(20, fp(5), fp(4)).unwrap();
        assert!(merge.end(20, fp(5), fp(4)).is_err());
        let weights = vec![WeightBlock::Scalar { row: 1, value: fp(3) }];
        let pending = merge.pending_claim(weights.clone(), 60, 9).unwrap();
        // The point's coordinates are consumed lowest column bit first.
        assert_eq!(pending.psi, vec![fp(9), fp(8), fp(7)]);
        assert_eq!(pending.target, fp(4));
        assert_eq!(
            pending.masks,
            vec![MaskTerm {
                oracle: 60,
                slot: 9,
                coeff: fp(5)
            }]
        );
        assert_eq!(pending.weights, weights);
        assert!(merge.pending_claim(Vec::new(), 60, 9).is_err());
    }

    #[test]
    fn settled_tables_build_the_runtime_plan_in_order() {
        let table = |at: u64, entries: u64| ReductionTable {
            matrix: 0,
            parts: 1,
            run_length: 2,
            rows_per_part: 2,
            columns: 4,
            at,
            entries,
            phi: vec![fp(1), fp(2)],
            later: false,
        };
        let plan = ReductionWitnessPlan::from_tables(vec![table(0, 4), table(4, 4)]).unwrap();
        assert_eq!(plan.tables().len(), 2);
        assert!(plan.read(0, 0).is_err());
        // A gap, an overlap, a wrong entry count and an unset point are each
        // refused.
        assert!(ReductionWitnessPlan::from_tables(vec![table(0, 4), table(5, 4)]).is_err());
        assert!(ReductionWitnessPlan::from_tables(vec![table(0, 4), table(3, 4)]).is_err());
        assert!(ReductionWitnessPlan::from_tables(vec![table(0, 3)]).is_err());
        let mut unset = table(0, 4);
        unset.phi.clear();
        assert!(ReductionWitnessPlan::from_tables(vec![unset]).is_err());
    }

    #[test]
    fn table_entries_are_the_run_inner_products() {
        // Two rows of eight cells, runs of four: four entries in row-major,
        // then run order.
        let cells: Vec<Fp3> = (1..=16).map(fp).collect();
        let phi = [fp(1), fp(2), fp(3), fp(4)];
        let entries = table_entries(&cells, 2, 8, 4, &phi).unwrap();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0], entry_value(&[fp(1), fp(2), fp(3), fp(4)], &phi).unwrap());
        assert_eq!(entries[1], entry_value(&[fp(5), fp(6), fp(7), fp(8)], &phi).unwrap());
        assert_eq!(entries[2], entry_value(&[fp(9), fp(10), fp(11), fp(12)], &phi).unwrap());
        // A point of another length, a run that does not divide the columns, a
        // wrong cell count and an empty matrix are each refused.
        assert!(table_entries(&cells, 2, 8, 4, &phi[..3]).is_err());
        assert!(table_entries(&cells, 2, 8, 3, &[fp(1), fp(2), fp(3)]).is_err());
        assert!(table_entries(&cells[..15], 2, 8, 4, &phi).is_err());
        assert!(table_entries(&cells, 0, 8, 4, &phi).is_err());
        assert!(entry_value(&[], &[]).is_err());
        assert!(entry_value(&[fp(1)], &[fp(1), fp(2)]).is_err());
    }

    #[test]
    fn a_read_set_is_checked_against_the_witness() {
        let mut plan = witness();
        plan.commit_root(Digest([0x41; 32])).unwrap();
        let weights: Vec<Fp3> = (0..8).map(|_| fp(2)).collect();
        let reads = ReductionReadSet::segmented(0, 0, 4, &weights).unwrap();
        assert_eq!(reads.reads().len(), 2);
        reads.cover(&plan).unwrap();
        // A set that stops short of the table's last entry, a read outside the
        // table and a zero coefficient are each refused.
        let short = ReductionReadSet::segmented(0, 0, 4, &weights[..4]).unwrap();
        assert!(short.cover(&plan).is_err());
        let outside = ReductionReadSet::segmented(0, 4, 4, &weights).unwrap();
        assert!(outside.cover(&plan).is_err());
        let mut zero = weights.clone();
        zero[2] = Fp3::ZERO;
        let zero = ReductionReadSet::segmented(0, 0, 4, &zero).unwrap();
        assert!(zero.cover(&plan).is_err());
        // A segment that does not divide the weight list is not a read set.
        assert!(ReductionReadSet::segmented(0, 0, 3, &weights).is_err());
        assert!(ReductionReadSet::segmented(0, 0, 0, &weights).is_err());
    }

    #[test]
    fn reads_must_cover_each_entry_once_with_a_weight() {
        let mut plan = witness();
        plan.commit_root(Digest([0x31; 32])).unwrap();
        let ones = vec![fp(1); 8];
        let read = ReductionRead {
            table: 0,
            first_entry: 0,
            weights: ones.clone(),
        };
        assert!(plan.check_reads(&[read.clone()]).is_ok());
        // A missing entry, a duplicate, a zero weight, an out-of-table read and
        // a missing table are each refused.
        let mut short = read.clone();
        short.weights.pop();
        assert!(plan.check_reads(&[short]).is_err());
        assert!(plan
            .check_reads(&[read.clone(), ReductionRead {
                table: 0,
                first_entry: 0,
                weights: vec![fp(1)],
            }])
            .is_err());
        let mut zero = read.clone();
        zero.weights[3] = Fp3::ZERO;
        assert!(plan.check_reads(&[zero]).is_err());
        assert!(plan
            .check_reads(&[ReductionRead {
                table: 0,
                first_entry: 1,
                weights: vec![fp(1); 8],
            }])
            .is_err());
        assert!(plan
            .check_reads(&[ReductionRead {
                table: 1,
                first_entry: 0,
                weights: vec![fp(1)],
            }])
            .is_err());
    }

    #[test]
    fn the_column_point_pairs_bits_low_first() {
        let row = [fp(10), fp(20), fp(30), fp(40)];
        let zero = fp(0);
        let one = fp(1);
        assert_eq!(row_at(&row, &[zero, zero]).unwrap(), fp(10));
        assert_eq!(row_at(&row, &[one, zero]).unwrap(), fp(20));
        assert_eq!(row_at(&row, &[zero, one]).unwrap(), fp(30));
        assert_eq!(row_at(&row, &[one, one]).unwrap(), fp(40));
        assert!(row_at(&row, &[]).is_err());
        assert!(row_at(&row[..3], &[zero, zero]).is_err());
    }

    #[test]
    fn the_plan_claim_must_be_the_chain_claim() {
        let chain = pending();
        check_member_claim(&chain, &chain).unwrap();
        let mut changed = chain.clone();
        changed.weights.pop();
        assert!(check_member_claim(&changed, &chain).is_err());
        let mut changed = chain.clone();
        changed.psi[0] = changed.psi[0].add(&Fp3::ONE);
        assert!(check_member_claim(&changed, &chain).is_err());
        let mut changed = chain.clone();
        changed.target = changed.target.add(&Fp3::ONE);
        assert!(check_member_claim(&changed, &chain).is_err());
        let mut changed = chain.clone();
        changed.masks[0].coeff = changed.masks[0].coeff.add(&Fp3::ONE);
        assert!(check_member_claim(&changed, &chain).is_err());
    }

    #[test]
    fn the_pending_claim_holds_on_the_matrix_and_masks() {
        // Four rows of four columns; the column point is zero, so each row's
        // value is its first column.
        let matrix: Vec<Vec<Fp3>> = (0..4)
            .map(|row| {
                (0..4)
                    .map(|column| fp(1 + 4 * row as u64 + column as u64))
                    .collect()
            })
            .collect();
        let weights = stretch_weights(2, 0, &[fp(2), fp(3), fp(5), fp(7)]).unwrap();
        let zero = fp(0);
        let mask_value = |oracle: u64, slot: u64| Ok(fp(oracle + slot + 100));
        let pending = PendingRow {
            weights,
            psi: vec![zero, zero],
            target: fp(2 * 1 + 3 * 5 + 5 * 9 + 7 * 13 + 7 * (60 + 3 + 100)),
            masks: vec![MaskTerm {
                oracle: 60,
                slot: 3,
                coeff: fp(7),
            }],
        };
        check_pending(&pending, &PlaceRows::new(&matrix), &mask_value).unwrap();
        // Another matrix value or mask value breaks the claim.
        let mut changed = matrix.clone();
        changed[1][0] = changed[1][0].add(&fp(1));
        assert!(check_pending(&pending, &PlaceRows::new(&changed), &mask_value).is_err());
        let other_mask = |oracle: u64, slot: u64| Ok(fp(oracle + slot + 101));
        assert!(check_pending(&pending, &PlaceRows::new(&matrix), &other_mask).is_err());
        // A scalar block with a gap and a block outside the row cube.
        let scalar = PendingRow {
            weights: vec![WeightBlock::Scalar {
                row: 1,
                value: fp(3),
            }],
            psi: vec![zero, zero],
            target: fp(3 * 5 + 7 * (60 + 3 + 100)),
            masks: vec![MaskTerm {
                oracle: 60,
                slot: 3,
                coeff: fp(7),
            }],
        };
        check_pending(&scalar, &PlaceRows::new(&matrix), &mask_value).unwrap();
        let outside = PendingRow {
            weights: vec![WeightBlock::Scalar {
                row: 4,
                value: fp(3),
            }],
            ..scalar.clone()
        };
        assert!(check_pending(&outside, &PlaceRows::new(&matrix), &mask_value).is_err());
    }
}
