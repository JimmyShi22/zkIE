//! The request-level chain's transcript section.
//!
//! A request's sub-proof leaves three masked sumchecks: the kept token
//! sumcheck, the batched stack and the weight merge. Their proofs travel as
//! one section, after the sub-proof's other messages and before the chain
//! value. The section carries its own version; the round and end-value counts
//! come from the plan's claim shapes, and a section that does not match them
//! is refused rather than reinterpreted.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::hidden_member::PendingRow;
use crate::proof::psc::PscRound;
use crate::proof::psc_general::{
    GeneralChallenges, GeneralMaskEntries, GeneralOutput, GeneralProof, MaskTerm,
};
use crate::proof::request_reduction::{prove_token_kept, verify_token_kept};
use crate::proof::request_stack::{
    prove_stack_group, verify_stack_group, GroupValues, StackReduction,
};
use crate::proof::request_start::{token_start, StartForm};
use crate::proof::request_weights::{
    prove_merge, verify_merge, MergeValue, WeightClaim, WeightLayout,
};
use crate::proof::packed_matrix::{PackedMatrixPlan, RangeCombinationPlan};
use crate::proof::row_claim::RowForm;

/// One named layout quantity of the request chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutValue {
    pub name: &'static str,
    pub value: u64,
}

/// The request chain's layout registration. One table names every quantity the
/// layout fixes, so the request-level re-derivation fills values in one place
/// instead of scattered constants. The skeleton starts with every value unset
/// (zero); a zero is not a final parameter, and the frozen layout replaces it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestLayout {
    values: Vec<LayoutValue>,
}

impl RequestLayout {
    /// Every quantity the request chain's layout fixes.
    pub const NAMES: [&'static str; 9] = [
        "mask_rows",
        "mask_slots",
        "end_values",
        "stack_rounds",
        "merge_rounds",
        "opening_budget",
        "sample_plan",
        "tree_layout",
        "stage_plan",
    ];

    /// The skeleton: every value unset until the layout is frozen.
    pub fn skeleton() -> Self {
        Self {
            values: Self::NAMES
                .iter()
                .map(|name| LayoutValue { name, value: 0 })
                .collect(),
        }
    }

    pub fn entries(&self) -> &[LayoutValue] {
        &self.values
    }

    /// Sets one named quantity. An unknown name is refused, and the table
    /// never grows: the layout's list is fixed by the skeleton.
    pub fn set(&mut self, name: &str, value: u64) -> Result<()> {
        let entry = self
            .values
            .iter_mut()
            .find(|entry| entry.name == name)
            .ok_or(Error::InvalidConfig)?;
        entry.value = value;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<u64> {
        self.values
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.value)
            .ok_or(Error::InvalidConfig)
    }

    /// True while every value is unset: the skeleton, not the frozen layout.
    pub fn is_skeleton(&self) -> bool {
        self.values.iter().all(|entry| entry.value == 0)
    }
}

/// The section's version.
pub const REQUEST_CHAIN_VERSION: u64 = 3;

/// One request chain's proofs, in the order the chain runs them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestChainMessages {
    pub token: GeneralProof,
    pub stack: GeneralProof,
    pub merge: GeneralProof,
}

/// The round counts and shown end-value counts the plan expects of each proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestChainShape {
    pub token_rounds: usize,
    pub token_ends: usize,
    pub stack_rounds: usize,
    pub stack_ends: usize,
    pub merge_rounds: usize,
    pub merge_ends: usize,
}

impl RequestChainMessages {
    /// Checks the stored counts against the plan's shape.
    pub fn check_shape(&self, shape: RequestChainShape) -> Result<()> {
        if self.token.rounds.len() != shape.token_rounds
            || self.token.end_values.len() != shape.token_ends
            || self.stack.rounds.len() != shape.stack_rounds
            || self.stack.end_values.len() != shape.stack_ends
            || self.merge.rounds.len() != shape.merge_rounds
            || self.merge.end_values.len() != shape.merge_ends
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    pub fn encode(&self, shape: RequestChainShape) -> Result<Vec<u8>> {
        self.check_shape(shape)?;
        let mut writer = CanonicalWriter::new();
        self.encode_into(&mut writer);
        Ok(writer.into_bytes())
    }

    /// Writes the section without the shape check; the decoder still checks
    /// every stored count against the plan.
    pub fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(REQUEST_CHAIN_VERSION);
        self.token.encode_into(writer);
        self.stack.encode_into(writer);
        self.merge.encode_into(writer);
    }

    pub fn decode(reader: &mut CanonicalReader, shape: RequestChainShape) -> Result<Self> {
        if reader.get_u64()? != REQUEST_CHAIN_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let token = GeneralProof::decode(reader, shape.token_rounds, shape.token_ends)?;
        let stack = GeneralProof::decode(reader, shape.stack_rounds, shape.stack_ends)?;
        let merge = GeneralProof::decode(reader, shape.merge_rounds, shape.merge_ends)?;
        Ok(Self {
            token,
            stack,
            merge,
        })
    }
}

/// The committed mask table of one request chain's sub-proof segment: one
/// oracle owns the masks of the token, stack and merge sumchecks, with its own
/// tree and opening ledger. The index-mask oracle of the request's pending
/// claim is a different object; a slot number is never read across the two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainMaskPlan {
    pub oracle: u64,
    pub group: u64,
    pub mask_code: zkie_engine::pcs::request::code::CodeParams,
    pub mask_rows: u64,
    pub entries: usize,
    /// The number of positions one test opens in the table; the table's code
    /// tail must cover them.
    pub openings: usize,
    pub tree: zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan,
}

impl ChainMaskPlan {
    /// The structural checks of the sub-proof mask table.
    pub fn fault(&self) -> Option<&'static str> {
        if self.oracle == 0 || self.group == 0 {
            return Some("the sub-proof mask table has no identity");
        }
        if zkie_engine::pcs::request::code::build_code_plan(
            self.mask_code.log_n,
            self.mask_code.ell,
            self.mask_code.r,
        )
        .is_err()
        {
            return Some("the sub-proof mask code is not a code");
        }
        if self.mask_rows == 0 || !self.mask_rows.is_power_of_two() {
            return Some("the sub-proof mask table has no power-of-two rows");
        }
        if self.entries == 0 {
            return Some("the sub-proof mask table has no entries");
        }
        if self.openings == 0 || self.openings as u64 > self.mask_code.r {
            return Some("the sub-proof mask opening count exceeds its tail");
        }
        if self.tree.fault().is_some() {
            return Some("the sub-proof mask tree is not a tree");
        }
        if self.tree.log_n != self.mask_code.log_n
            || self.tree.log_n_zk != self.mask_code.log_n
            || self.tree.parts.len() != 1
        {
            return Some("the sub-proof mask tree does not match its code");
        }
        let mask_symbols = match self.mask_rows.checked_mul(3) {
            Some(value) => value,
            None => return Some("the sub-proof mask row width overflows"),
        };
        if self.tree.parts[0].main_symbols != 0
            || self.tree.parts[0].mask_symbols != mask_symbols
        {
            return Some("the sub-proof mask tree does not carry its mask rows");
        }
        let free = match self.free_entries() {
            Ok(free) => free,
            Err(_) => return Some("the sub-proof mask rows are shorter than the entries"),
        };
        if free < self.entries {
            return Some("the sub-proof mask rows are shorter than the entries");
        }
        None
    }

    /// The free mask entries, excluding the SEND pad of the last row.
    pub fn free_entries(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        if ell == 0 {
            return Err(Error::InvalidConfig);
        }
        rows.checked_mul(ell.checked_sub(1).ok_or(Error::InvalidConfig)?)
            .and_then(|value| value.checked_add(rows.saturating_sub(1)))
            .ok_or(Error::ResourceLimit)
    }

    /// The BAT virtual table index of an entry: `row + place * 2^row_bits`,
    /// the layout the mask group uses.
    pub fn entry_virtual_index(&self, slot: usize) -> Result<u64> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let length = rows.checked_mul(ell).ok_or(Error::ResourceLimit)?;
        if slot >= length || slot >= self.free_entries()? {
            return Err(Error::InvalidIndex);
        }
        let row = slot / ell;
        let place = slot % ell;
        let row_bits = if self.mask_rows <= 1 {
            0
        } else {
            64 - (self.mask_rows - 1).leading_zeros()
        };
        (row as u64)
            .checked_add((place as u64) << row_bits)
            .ok_or(Error::ResourceLimit)
    }
}

/// One limb matrix of the original computation matrix, with its packed symbol
/// array's SEND message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeObject {
    /// The limb matrix in the functional's read orientation: its cell rows are
    /// the activation tensor's hidden features, its cell columns the positions.
    pub matrix: PackedMatrixPlan,
    pub array_oracle: u64,
    pub array_shape: crate::proof::hidden_send_session::WShape,
}

/// The smallness objects of the original computation matrix: one packed
/// symbol array per balanced base-256 limb, one reduced read table (U3)
/// covering every limb's functionals, and one digit object (ZZ) whose planes
/// cover all limb matrices at their offsets. The carrier is drawn once for the
/// tensor's geometry. BZ and the digit lookup are not part of this object set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangePlan {
    /// One object per limb; every limb matrix has the same geometry.
    pub objects: Vec<RangeObject>,
    /// The digit object's addresses: two planes over the summed cell rows of
    /// all limb matrices, each matrix at its offset.
    pub combination: RangeCombinationPlan,
    /// The number of random column combinations. The supported set is the
    /// powers of two in `2..=256`; the carrier is drawn from the segment's
    /// chain once, before the digit object and after every symbol array and
    /// the reduced read table. The batching element is drawn nonzero from the
    /// same chain after the digit object's sample answers.
    pub combinations: u32,
    /// The high digit's width. The low digit's width is the combination plan's.
    pub high_bits: u32,
    /// The combination window `2^LBZ`: a combination value plus this offset is
    /// split into the two digits.
    pub window: i128,
    pub group: u64,
    /// The reduced read table's SEND message.
    pub unit_oracle: u64,
    pub unit_shape: crate::proof::hidden_send_session::WShape,
    /// The digit object's SEND message: `combinations` rows over
    /// `2 * cell_rows()` columns.
    pub digit_oracle: u64,
    pub digit_shape: crate::proof::hidden_send_session::WShape,
}

/// The digit object's real entry count: `combinations` rows over
/// `2 * cell_rows` columns.
pub fn range_digit_values(combinations: u32, cell_rows: u64) -> Result<u64> {
    (combinations as u64)
        .checked_mul(2)
        .and_then(|value| value.checked_mul(cell_rows))
        .ok_or(Error::ResourceLimit)
}

/// The digit message's committed width: the real entry count padded to a
/// power of two, like every other value message of the session.
pub fn range_digit_columns(combinations: u32, cell_rows: u64) -> Result<u64> {
    range_digit_values(combinations, cell_rows)?
        .checked_next_power_of_two()
        .ok_or(Error::ResourceLimit)
}

/// The digit message's derived row count for a committed width, a code length
/// and a sample-answer count: the smallest power of two whose places stay off
/// the sample pad (`(width - 1) / rows < ell - answers`) and whose capacity
/// covers the width.
pub fn range_digit_rows(width: u64, ell: u64, answers: u32) -> Result<u64> {
    if width == 0 || !width.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let answers = answers as u64;
    if ell == 0 || answers >= ell {
        return Err(Error::InvalidConfig);
    }
    let span = ell - answers;
    let needed = width
        .checked_add(span - 1)
        .ok_or(Error::ResourceLimit)?
        / span;
    let rows = needed
        .checked_next_power_of_two()
        .ok_or(Error::ResourceLimit)?;
    if rows.checked_mul(ell).ok_or(Error::ResourceLimit)? < width {
        return Err(Error::ResourceLimit);
    }
    Ok(rows)
}

impl RangePlan {
    /// The number of limb matrices.
    pub fn limbs(&self) -> u64 {
        self.objects.len() as u64
    }

    /// The cell rows of one limb matrix.
    pub fn rows_cell(&self) -> Result<u64> {
        self.objects
            .first()
            .map(|object| object.matrix.rows)
            .ok_or(Error::InvalidConfig)
    }

    /// The cell columns of one limb matrix.
    pub fn columns_cell(&self) -> Result<u64> {
        self.objects
            .first()
            .map(|object| object.matrix.columns)
            .ok_or(Error::InvalidConfig)
    }

    /// The summed cell rows of every limb matrix: the digit object's
    /// per-plane column count.
    pub fn cell_rows(&self) -> Result<u64> {
        self.rows_cell()?
            .checked_mul(self.limbs())
            .ok_or(Error::ResourceLimit)
    }

    /// The first cell row of one limb matrix within the digit object.
    pub fn offset(&self, limb: usize) -> Result<u64> {
        if limb >= self.objects.len() {
            return Err(Error::InvalidIndex);
        }
        (limb as u64)
            .checked_mul(self.rows_cell()?)
            .ok_or(Error::ResourceLimit)
    }

    /// The entries of one limb's packed symbol array before padding.
    pub fn array_entries(&self, limb: usize) -> Result<u64> {
        let object = self.objects.get(limb).ok_or(Error::InvalidIndex)?;
        object
            .matrix
            .groups
            .checked_mul(object.matrix.columns)
            .ok_or(Error::ResourceLimit)
    }

    /// The value-vector length of one limb's packed array: its entries padded
    /// to a power of two.
    pub fn array_columns(&self, limb: usize) -> Result<u64> {
        self.array_entries(limb)?
            .checked_next_power_of_two()
            .ok_or(Error::ResourceLimit)
    }

    /// The digit object's real entry count: `combinations` rows over
    /// `2 * cell_rows()` columns.
    pub fn digit_values(&self) -> Result<u64> {
        range_digit_values(self.combinations, self.cell_rows()?)
    }

    /// The digit message's committed width: the real entries padded to a
    /// power of two, like every other value message of the session.
    pub fn digit_columns(&self) -> Result<u64> {
        range_digit_columns(self.combinations, self.cell_rows()?)
    }

    /// The digit message's derived row count: the smallest power of two whose
    /// places stay off the sample pad and whose capacity covers the width.
    pub fn digit_rows(&self) -> Result<u64> {
        range_digit_rows(
            self.digit_columns()?,
            self.digit_shape.code.ell,
            self.digit_shape.answers,
        )
    }

    /// The digit object's stored column of a plane and a cell row. The index
    /// is the limb matrix's offset plus its cell row.
    pub fn digit_place(&self, plane: u32, index: u64) -> Result<u64> {
        if plane >= 2 {
            return Err(Error::InvalidIndex);
        }
        self.combination.zz_column(plane as u64, index)
    }

    /// The dense value-vector index of a digit entry: place times the
    /// combination count plus the combination, the order the key vectors
    /// follow.
    pub fn digit_dense(&self, plane: u32, index: u64, combination: u32) -> Result<u64> {
        if combination >= self.combinations {
            return Err(Error::InvalidIndex);
        }
        self.digit_place(plane, index)?
            .checked_mul(self.combinations as u64)
            .and_then(|value| value.checked_add(combination as u64))
            .ok_or(Error::ResourceLimit)
    }

    /// The BAT virtual index of a digit entry: its dense index read through
    /// the digit message's own row count.
    pub fn digit_virtual(&self, plane: u32, index: u64, combination: u32) -> Result<u64> {
        let dense = self.digit_dense(plane, index, combination)?;
        Self::message_virtual(&self.digit_shape, dense)
    }

    /// The BAT virtual index of a dense entry of one value message.
    pub fn message_virtual(shape: &crate::proof::hidden_send_session::WShape, index: u64) -> Result<u64> {
        let rows = shape.main_rows;
        if rows == 0 || !rows.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        let row_bits = 64 - (rows - 1).leading_zeros();
        let row = index % rows;
        let place = index / rows;
        row.checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)
    }

    /// The structural checks of the range objects. The message capacities
    /// that depend on the member's read tables are checked by the session
    /// plan.
    pub fn fault(&self) -> Option<&'static str> {
        let Some(first) = self.objects.first() else {
            return Some("the range objects are empty");
        };
        let matrix = first.matrix;
        if matrix.rows == 0 || matrix.columns == 0 || matrix.groups == 0 {
            return Some("the range matrix has no geometry");
        }
        if self.group == 0 || self.unit_oracle == 0 || self.digit_oracle == 0 {
            return Some("the range objects have no identity");
        }
        if self.unit_oracle == self.digit_oracle {
            return Some("two range objects share an oracle");
        }
        let mut oracles = vec![self.unit_oracle, self.digit_oracle];
        let mut trees = vec![self.unit_shape.tree_number, self.digit_shape.tree_number];
        if self.unit_shape.tree_number == self.digit_shape.tree_number {
            return Some("two range objects share a tree number");
        }
        for object in &self.objects {
            if object.array_oracle == 0 {
                return Some("a range message has no identity");
            }
            if object.matrix != matrix {
                return Some("the limb matrices have different geometries");
            }
            if oracles.contains(&object.array_oracle) {
                return Some("two range objects share an oracle");
            }
            if trees.contains(&object.array_shape.tree_number) {
                return Some("two range objects share a tree number");
            }
            oracles.push(object.array_oracle);
            trees.push(object.array_shape.tree_number);
        }
        let mz = match matrix.rows.checked_mul(self.limbs()) {
            Some(value) => value,
            None => return Some("the range matrix sizes overflow"),
        };
        if self.combination.columns != mz
            || self.combination.planes != 2
            || self.combination.low_bits != 10
            || self.high_bits != 8
            || self.window != 1i128 << 17
        {
            return Some("the digit object has another window");
        }
        if self.combinations < 2
            || self.combinations > 256
            || !(self.combinations as u64).is_power_of_two()
        {
            return Some("the combination count is not a power of two in range");
        }
        if !self.unit_shape.code.ell.is_power_of_two()
            || !self.digit_shape.code.ell.is_power_of_two()
        {
            return Some("a range message code has no power-of-two message length");
        }
        for (limb, object) in self.objects.iter().enumerate() {
            if !object.array_shape.code.ell.is_power_of_two() {
                return Some("a range message code has no power-of-two message length");
            }
            let array_columns = match self.array_columns(limb) {
                Ok(columns) => columns,
                Err(_) => return Some("the packed array size overflows"),
            };
            if object.array_shape.fault(array_columns, 1).is_some() {
                return Some("the packed array message shape is not a message");
            }
        }
        let digit_columns = match self.digit_columns() {
            Ok(columns) => columns,
            Err(_) => return Some("the digit object size overflows"),
        };
        let digit_rows = match self.digit_rows() {
            Ok(rows) => rows,
            Err(_) => return Some("the digit message row count overflows"),
        };
        if self.digit_shape.main_rows != digit_rows {
            return Some("the digit message does not carry its rows");
        }
        if self.digit_shape.fault(digit_columns, 1).is_some() {
            return Some("the digit message shape is not a message");
        }
        None
    }
}

/// The read ledger of one sub-proof mask table: every test that opens the table
/// adds its position count here, and the accumulated count may not pass the
/// table's code tail. The ledger belongs to this object alone; another mask
/// object's openings never enter it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChainMaskReads {
    pub used: usize,
}

impl ChainMaskReads {
    /// Records one test's openings against the table's tail.
    pub fn open(&mut self, plan: &ChainMaskPlan, count: usize) -> Result<()> {
        if plan.fault().is_some() || count == 0 {
            return Err(Error::InvalidConfig);
        }
        let used = self.used.checked_add(count).ok_or(Error::ResourceLimit)?;
        if used as u64 > plan.mask_code.r {
            return Err(Error::BoundViolation);
        }
        self.used = used;
        Ok(())
    }
}

impl ChainMaskPlan {
    /// The number of seed coordinates the mask sample form reads: the row bits
    /// of the mask table plus the mask code message bits.
    pub fn sample_bits(&self) -> Result<u32> {
        if self.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let row_bits = if self.mask_rows <= 1 {
            0
        } else {
            64 - (self.mask_rows - 1).leading_zeros()
        };
        row_bits
            .checked_add(self.mask_code.log_ell())
            .ok_or(Error::ResourceLimit)
    }

    /// The sampled answer of the table under one seed: the mask rows are read
    /// through the sample cube, the last entry takes the seed factor instead
    /// of its cube coefficient, and the answer is the masked row sum.
    pub fn answer(
        &self,
        rows: &[Vec<Fp3>],
        seed: &zkie_engine::pcs::request::table::SampleSeed,
    ) -> Result<Fp3> {
        if self.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let bits = self.sample_bits()?;
        if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
            return Err(Error::ShapeMismatch);
        }
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let mask_rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        if rows.len() != mask_rows || rows.iter().any(|row| row.len() != ell) {
            return Err(Error::ShapeMismatch);
        }
        let row_bits = if self.mask_rows <= 1 {
            0
        } else {
            64 - (self.mask_rows - 1).leading_zeros()
        };
        let point = &seed.z[..bits as usize];
        let count = 1u64
            .checked_shl(bits)
            .ok_or(Error::ResourceLimit)?;
        let eq = crate::proof::output_closure::eq_vector(point)?;
        let mut answer = Fp3::ZERO;
        for index in 0..count {
            let row = (index & ((1u64 << row_bits) - 1)) as usize;
            let place = (index >> row_bits) as usize;
            if row >= mask_rows || place >= ell {
                continue;
            }
            let coefficient = if row + 1 == mask_rows && place + 1 == ell {
                seed.beta
            } else {
                *eq.get(index as usize).ok_or(Error::InvalidIndex)?
            };
            answer = answer.add(&coefficient.mul(&rows[row][place]));
        }
        Ok(answer)
    }

    /// Checks one test's opening list of the table and records it in the
    /// table's read ledger: the count is the plan's openings, every position
    /// lies in the tree's columns, no position repeats, and the accumulated
    /// reads stay inside the code tail. The opening itself is checked against
    /// the root, which this function never reads.
    pub fn check_positions(
        &self,
        positions: &[u64],
        reads: &mut ChainMaskReads,
    ) -> Result<()> {
        if self.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if positions.len() != self.openings {
            return Err(Error::ShapeMismatch);
        }
        let columns = self.tree.columns();
        let mut seen = std::collections::BTreeSet::new();
        for position in positions {
            if *position >= columns {
                return Err(Error::InvalidIndex);
            }
            if !seen.insert(*position) {
                return Err(Error::InvalidConfig);
            }
        }
        reads.open(self, positions.len())
    }
}

/// The two source rows of the stack group: the shown A0 against the left table
/// side with the equation weights, and the shown B0 against the right side,
/// each carrying its own sub-proof mask slot with the stack end factor. The
/// cell entries and the owner of the table are trusted plan data.
pub fn stack_source_rows(
    left_owner: u64,
    left_entries: &[u64],
    left_weights: &[Fp3],
    right_owner: u64,
    right_entries: &[u64],
    right_weights: &[Fp3],
    a0: Fp3,
    b0: Fp3,
    mask_member: u64,
    a0_slot: u64,
    b0_slot: u64,
    beta_stack: Fp3,
) -> Result<(
    zkie_engine::pcs::request::bat::BatRow,
    zkie_engine::pcs::request::bat::BatRow,
)> {
    use zkie_engine::pcs::request::bat::{BatPart, BatRow, BatRowKind};
    if left_entries.len() != left_weights.len()
        || right_entries.len() != right_weights.len()
        || beta_stack.is_zero()
    {
        return Err(Error::ShapeMismatch);
    }
    let build = |owner: u64,
                 entries: &[u64],
                 weights: &[Fp3],
                 shown: Fp3,
                 slot: u64|
     -> zkie_engine::pcs::request::bat::BatRow {
        let mut row = BatRow::new(BatRowKind::Plain);
        for (entry, weight) in entries.iter().zip(weights) {
            row.add_unit(owner, BatPart::Main, *entry, *weight);
        }
        row.add_unit(mask_member, BatPart::Mask, slot, beta_stack);
        row.target = shown;
        row
    };
    let left = build(left_owner, left_entries, left_weights, a0, a0_slot);
    let right = build(right_owner, right_entries, right_weights, b0, b0_slot);
    Ok((left, right))
}

/// Lands one sub-proof row form on the chain mask table: the hiding entries
/// live at the table's first entries in order, each counted mask slot maps
/// through the trusted slot-to-entry list, and kept values keep their own
/// owners. This is the token kept relation's first landing point; the slot
/// list is plan data, never a proof byte.
pub fn chain_mask_row_form(
    plan: &ChainMaskPlan,
    form: &crate::proof::row_claim::RowForm,
    slot_entries: &[usize],
    kept: &[(u64, u64)],
) -> Result<zkie_engine::pcs::request::bat::BatRow> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let hiding: Vec<u64> = (0..form.hiding.len() as u64).collect();
    let mut slots = Vec::with_capacity(form.slots.len());
    for term in &form.slots {
        let entry = *slot_entries.get(term.slot).ok_or(Error::InvalidIndex)?;
        slots.push(plan.entry_virtual_index(entry)?);
    }
    let binding = crate::proof::segment_slots::RowBinding {
        mask_member: plan.oracle,
        hiding_entries: &hiding,
        slot_entries: &slots,
        kept,
    };
    crate::proof::segment_slots::row_form_row(&binding, form)
}

/// Lands the merge sumcheck's remaining row on the sub-proof mask table: the
/// output row form (hiding coefficients, counted mask slots, kept-map terms and
/// the row target) goes through the same trusted slot list and kept owners as
/// the token kept row. This is the wsForkLin remaining relation's landing
/// point.
pub fn merge_remaining_row(
    plan: &ChainMaskPlan,
    output: &crate::proof::psc_general::GeneralOutput,
    slot_entries: &[usize],
    kept: &[(u64, u64)],
) -> Result<zkie_engine::pcs::request::bat::BatRow> {
    let form = crate::proof::row_claim::RowForm::from_output(output);
    chain_mask_row_form(plan, &form, slot_entries, kept)
}

/// Verifies one opening of the sub-proof mask table against its root: the plan
/// must be a tree, and the opening's positions and columns are checked against
/// the root by the runtime tree. The read ledger and the position list policy
/// are checked separately, so a proof byte never chooses the list.
pub fn verify_chain_mask_opening(
    plan: &ChainMaskPlan,
    root: &zkie_core::types::Digest,
    list: &[u64],
    opening: &zkie_engine::pcs::request::runtime_tree::RuntimeTreeOpening,
) -> Result<()> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    zkie_engine::pcs::request::runtime_tree::verify_runtime_opening(
        &plan.tree, root, list, opening,
    )
}

/// The owner binding of the sub-proof segment's mask slots: the committed
/// member is the plan oracle and every slot maps through the chain mask
/// layout. Duplicate slots are refused; the slots are the sub-proof's, never
/// the request index-mask's.
pub fn chain_mask_entries(plan: &ChainMaskPlan, slots: &[usize]) -> Result<Vec<(u64, u64)>> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut entries = Vec::with_capacity(slots.len());
    for slot in slots {
        if !seen.insert(*slot) {
            return Err(Error::InvalidConfig);
        }
        entries.push((plan.oracle, plan.entry_virtual_index(*slot)?));
    }
    Ok(entries)
}

/// Mask slots at and above this one name the start form's hiding entries.
pub const START_HIDING_SLOT: usize = 1 << 24;

/// Everything the chain proves and checks, in one place. The witnesses are the
/// maps of the three sumchecks; the verifier supplies the same values except
/// the kept map of the token sumcheck.
#[derive(Clone, Copy)]
pub struct ChainInputs<'a> {
    pub token_point: &'a [Fp3],
    pub token_shown: &'a [Fp3],
    pub token_kept: &'a [Fp3],
    pub token_claim_masks: &'a [MaskTerm],
    pub token_shown_claim: Fp3,
    pub token_claim_slot: usize,
    pub token_masks: &'a GeneralMaskEntries,
    pub reduction: &'a StackReduction,
    pub stack_left: &'a [Vec<Fp3>],
    pub stack_right: &'a [Vec<Fp3>],
    pub om: &'a [Fp3],
    pub stack_masks: &'a GeneralMaskEntries,
    pub layout: &'a WeightLayout,
    pub claims: &'a [WeightClaim],
    pub claim_maps: &'a [Vec<Fp3>],
    /// One mask-row slot per sent end value, in the group's order.
    pub end_slots: &'a [usize],
    pub merge_masks: &'a GeneralMaskEntries,
    pub mask_value: &'a dyn Fn(usize) -> Result<Fp3>,
    pub index_oracle: u64,
    pub index_slot: u64,
}

/// One request chain's proofs, in the order the chain runs them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainProof {
    pub token: GeneralProof,
    pub stack: GeneralProof,
    pub merge: GeneralProof,
}

/// What the chain's verifier returns: the sub-proofs' outputs, the merge's row
/// weights and the pending claim the request's member chain consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainOutputs {
    pub token: GeneralOutput,
    pub stack: GeneralOutput,
    pub merge: GeneralOutput,
    pub gamma: Vec<Fp3>,
    pub pending: PendingRow,
}

/// Records the challenges drawn while one sub-proof runs, so the token
/// sumcheck's discount factor is available to the start form.
struct Recorder<'a> {
    inner: &'a mut dyn GeneralChallenges,
    draws: Vec<Fp3>,
}

impl Recorder<'_> {
    fn epsilon(&self) -> Result<Fp3> {
        self.draws
            .first()
            .copied()
            .filter(|value| !value.is_zero())
            .ok_or(Error::InvalidConfig)
    }
}

impl GeneralChallenges for Recorder<'_> {
    fn draw(&mut self) -> Result<Fp3> {
        let value = self.inner.draw()?;
        self.draws.push(value);
        Ok(value)
    }
    fn draw_nonzero(&mut self) -> Result<Fp3> {
        let value = self.inner.draw_nonzero()?;
        self.draws.push(value);
        Ok(value)
    }
    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()> {
        self.inner.send_hiding_sum(value)
    }
    fn send_round(&mut self, index: usize, round: &PscRound) -> Result<()> {
        self.inner.send_round(index, round)
    }
    fn send_end_values(&mut self, values: &[Fp3]) -> Result<()> {
        self.inner.send_end_values(values)
    }
}

/// The token claim's factor: the coefficient of its slot in the token claim
/// masks. Exactly one term may name the slot.
fn claim_factor(inputs: &ChainInputs) -> Result<Fp3> {
    let mut factor = None;
    for term in inputs.token_claim_masks {
        if term.slot == inputs.token_claim_slot {
            if factor.is_some() {
                return Err(Error::InvalidConfig);
            }
            factor = Some(term.coefficient);
        }
    }
    factor.ok_or(Error::InvalidConfig)
}

/// The stack's claim masks: the start form's slot terms, then its hiding
/// coefficients at `START_HIDING_SLOT..`.
fn start_terms(start: &StartForm) -> Vec<MaskTerm> {
    let mut terms: Vec<MaskTerm> = start
        .claims
        .iter()
        .map(|term| MaskTerm {
            slot: term.slot,
            coefficient: term.coefficient,
        })
        .collect();
    for (index, coefficient) in start.hiding.iter().enumerate() {
        terms.push(MaskTerm {
            slot: START_HIDING_SLOT + index,
            coefficient: *coefficient,
        });
    }
    terms
}

fn start_form(token: &GeneralOutput, epsilon: Fp3, inputs: &ChainInputs) -> Result<StartForm> {
    token_start(
        token,
        epsilon,
        inputs.token_claim_slot,
        inputs.token_shown_claim,
        claim_factor(inputs)?,
        inputs.om.first().copied().ok_or(Error::InvalidConfig)?,
    )
}

/// Derives the hidden end group from the received end values. Every end value
/// carries the same factor, each slot's entry must be the mask the stack used,
/// and a hidden value is the sent one minus the factor times its entry. The
/// merge's input values are the sent weight values with their slots and the
/// stack's factor.
fn group_from_ends(
    output: &GeneralOutput,
    masks: &GeneralMaskEntries,
    end_slots: &[usize],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
) -> Result<(GroupValues, Vec<MergeValue>, Fp3)> {
    if output.ends.len() < 2
        || output.ends.len() != end_slots.len()
        || masks.shown.len() != output.ends.len()
    {
        return Err(Error::ShapeMismatch);
    }
    // Each sent value has its own mask slot: a repeated slot would let one
    // value stand in for another.
    let mut unique = end_slots.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != end_slots.len() {
        return Err(Error::InvalidConfig);
    }
    let beta = output.ends[0].mask_scale;
    let mut hidden = Vec::with_capacity(output.ends.len());
    for (index, (end, slot)) in output.ends.iter().zip(end_slots).enumerate() {
        if end.mask_scale != beta {
            return Err(Error::StatementMismatch);
        }
        let entry = mask_value(*slot)?;
        if entry != masks.shown[index] {
            return Err(Error::StatementMismatch);
        }
        hidden.push(end.value.sub(&beta.mul(&entry)));
    }
    let values = output.ends[2..]
        .iter()
        .zip(&end_slots[2..])
        .map(|(end, slot)| MergeValue::from_stack_end(end.value, *slot, beta))
        .collect();
    Ok((
        GroupValues {
            a0: hidden[0],
            b0: hidden[1],
            v: hidden[2..].to_vec(),
        },
        values,
        beta,
    ))
}

/// Proves one request chain in the order the transcript carries it: the kept
/// token sumcheck, the batched stack whose start absorbs the token claim, and
/// the weight merge. The prover checks its own row claims before handing them
/// on.
pub fn prove_request_chain(
    inputs: &ChainInputs,
    challenges: &mut dyn GeneralChallenges,
) -> Result<(ChainProof, ChainOutputs)> {
    let (token, token_output, epsilon) = {
        let mut recorder = Recorder {
            inner: challenges,
            draws: Vec::new(),
        };
        let (proof, output) = prove_token_kept(
            inputs.token_point,
            inputs.token_shown,
            inputs.token_kept,
            inputs.token_claim_masks,
            inputs.mask_value,
            inputs.token_shown_claim,
            inputs.token_masks,
            &mut recorder,
        )
        ?;
        let epsilon = recorder.epsilon()?;
        (proof, output, epsilon)
    };
    let start = start_form(&token_output, epsilon, inputs)
        ?;
    let terms = start_terms(&start);
    let (stack, stack_output) = prove_stack_group(
        inputs.reduction,
        inputs.stack_left,
        inputs.stack_right,
        inputs.om,
        &terms,
        inputs.mask_value,
        start.constant,
        inputs.stack_masks,
        challenges,
    )
    ?;
    let (group, values, _beta) = group_from_ends(
        &stack_output,
        inputs.stack_masks,
        inputs.end_slots,
        inputs.mask_value,
    )
    ?;
    inputs.reduction.group_bases(
        inputs.stack_left,
        inputs.stack_right,
        inputs.om,
        &stack_output.point,
        &group,
    )
    ?;
    let outcome = prove_merge(
        inputs.layout,
        inputs.claims,
        inputs.claim_maps,
        &values,
        inputs.mask_value,
        inputs.merge_masks,
        challenges,
    )
    ?;
    RowForm::from_output(&stack_output).check(
        &inputs.stack_masks.hiding,
        inputs.mask_value,
        &[],
    )?;
    RowForm::from_output(&outcome.output).check(
        &inputs.merge_masks.hiding,
        inputs.mask_value,
        &[],
    )?;
    // The prover derives the same pending claim the verifier will: the merge's
    // row weights at its point, the final masked value and the request's
    // index-mask slot. The request chain's member consumes this row.
    let end = outcome.output.ends.first().ok_or(Error::ShapeMismatch)?;
    let pending = inputs.layout.pending_row(
        &outcome.gamma,
        &outcome.output.point,
        end.mask_scale,
        end.value,
        inputs.index_oracle,
        inputs.index_slot,
    )
    ?;
    let proof = ChainProof {
        token,
        stack,
        merge: outcome.proof,
    };
    let outputs = ChainOutputs {
        token: token_output,
        stack: stack_output,
        merge: outcome.output,
        gamma: outcome.gamma,
        pending,
    };
    Ok((proof, outputs))
}

/// Verifies one request chain: the three sub-proofs in transcript order, the
/// two row claims on the transcript's mask entries, and the pending claim the
/// merge leaves.
pub fn verify_request_chain(
    inputs: &ChainInputs,
    proof: &ChainProof,
    challenges: &mut dyn GeneralChallenges,
) -> Result<ChainOutputs> {
    let (token, epsilon) = {
        let mut recorder = Recorder {
            inner: challenges,
            draws: Vec::new(),
        };
        let output = verify_token_kept(
            inputs.token_point,
            inputs.token_claim_masks,
            inputs.token_shown_claim,
            &proof.token,
            inputs.token_masks.hiding.len(),
            &mut recorder,
        )?;
        let epsilon = recorder.epsilon()?;
        (output, epsilon)
    };
    let start = start_form(&token, epsilon, inputs)?;
    let terms = start_terms(&start);
    let stack = verify_stack_group(
        inputs.reduction,
        &terms,
        start.constant,
        &proof.stack,
        inputs.stack_masks.hiding.len(),
        challenges,
    )?;
    // The verifier reads only the received shown end values: each carries the
    // stack's public factor and its own slot. No private mask value is
    // recovered here; the relation rows bind these values to the committed
    // members.
    if stack.ends.len() != inputs.end_slots.len() || stack.ends.len() < 3 {
        return Err(Error::ShapeMismatch);
    }
    let beta = stack.ends[0].mask_scale;
    let mut unique = inputs.end_slots.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != inputs.end_slots.len() {
        return Err(Error::InvalidConfig);
    }
    if stack.ends.iter().any(|end| end.mask_scale != beta) {
        return Err(Error::StatementMismatch);
    }
    let values: Vec<MergeValue> = stack.ends[2..]
        .iter()
        .zip(&inputs.end_slots[2..])
        .map(|(end, slot)| MergeValue::from_stack_end(end.value, *slot, beta))
        .collect();
    let (merge, _rows, gamma) = verify_merge(
        inputs.layout,
        inputs.claims,
        &values,
        &proof.merge,
        inputs.merge_masks.hiding.len(),
        challenges,
    )?;
    let end = merge.ends.first().ok_or(Error::ShapeMismatch)?;
    let pending = inputs.layout.pending_row(
        &gamma,
        &merge.point,
        end.mask_scale,
        end.value,
        inputs.index_oracle,
        inputs.index_slot,
    )?;
    Ok(ChainOutputs {
        token,
        stack,
        merge,
        gamma,
        pending,
    })
}

/// The direct-evaluation verifier used by the prover and the fixtures only:
/// it recovers the hidden end group from the caller's mask entries, checks the
/// two row forms against them and the source ties with the read cells. The
/// formal session verifier never calls this; the committed relation rows carry
/// the same checks.
pub fn verify_request_chain_direct(
    inputs: &ChainInputs,
    proof: &ChainProof,
    challenges: &mut dyn GeneralChallenges,
) -> Result<ChainOutputs> {
    let (token, epsilon) = {
        let mut recorder = Recorder {
            inner: challenges,
            draws: Vec::new(),
        };
        let output = verify_token_kept(
            inputs.token_point,
            inputs.token_claim_masks,
            inputs.token_shown_claim,
            &proof.token,
            inputs.token_masks.hiding.len(),
            &mut recorder,
        )?;
        let epsilon = recorder.epsilon()?;
        (output, epsilon)
    };
    let start = start_form(&token, epsilon, inputs)?;
    let terms = start_terms(&start);
    let stack = verify_stack_group(
        inputs.reduction,
        &terms,
        start.constant,
        &proof.stack,
        inputs.stack_masks.hiding.len(),
        challenges,
    )?;
    let (group, values, _beta) = group_from_ends(
        &stack,
        inputs.stack_masks,
        inputs.end_slots,
        inputs.mask_value,
    )?;
    inputs.reduction.group_bases(
        inputs.stack_left,
        inputs.stack_right,
        inputs.om,
        &stack.point,
        &group,
    )?;
    let (merge, _rows, gamma) = verify_merge(
        inputs.layout,
        inputs.claims,
        &values,
        &proof.merge,
        inputs.merge_masks.hiding.len(),
        challenges,
    )?;
    RowForm::from_output(&stack).check(&inputs.stack_masks.hiding, inputs.mask_value, &[])?;
    RowForm::from_output(&merge).check(&inputs.merge_masks.hiding, inputs.mask_value, &[])?;
    let end = merge.ends.first().ok_or(Error::ShapeMismatch)?;
    let pending = inputs.layout.pending_row(
        &gamma,
        &merge.point,
        end.mask_scale,
        end.value,
        inputs.index_oracle,
        inputs.index_slot,
    )?;
    Ok(ChainOutputs {
        token,
        stack,
        merge,
        gamma,
        pending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::projection_claim::PlaceRows;

    use zkie_core::common::codec::DecodeLimits;
    use zkie_engine::pcs::request::field::Fp3;
    use zkie_engine::pcs::request::field::fp_from_u64;

    use crate::proof::psc::PscRound;

    fn f(value: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(value))
    }

    fn proof(rounds: usize, ends: usize, base: u64) -> GeneralProof {
        GeneralProof {
            hiding_sum: f(base),
            rounds: (0..rounds)
                .map(|index| PscRound {
                    coefficients: vec![f(base + 1 + index as u64)],
                })
                .collect(),
            y_round: PscRound {
                coefficients: vec![f(base + 100)],
            },
            end_values: (0..ends).map(|index| f(base + 200 + index as u64)).collect(),
        }
    }

    fn shape() -> RequestChainShape {
        RequestChainShape {
            token_rounds: 2,
            token_ends: 1,
            stack_rounds: 3,
            stack_ends: 2,
            merge_rounds: 4,
            merge_ends: 1,
        }
    }

    fn messages() -> RequestChainMessages {
        RequestChainMessages {
            token: proof(2, 1, 10),
            stack: proof(3, 2, 20),
            merge: proof(4, 1, 30),
        }
    }

    #[test]
    fn the_layout_registers_every_quantity_by_name() {
        let mut layout = RequestLayout::skeleton();
        assert_eq!(layout.entries().len(), RequestLayout::NAMES.len());
        assert!(layout.is_skeleton());
        layout.set("mask_rows", 64).unwrap();
        layout.set("end_values", 4).unwrap();
        assert_eq!(layout.get("mask_rows").unwrap(), 64);
        assert_eq!(layout.get("end_values").unwrap(), 4);
        assert!(!layout.is_skeleton());
        // Unknown names are refused and the table does not grow.
        assert!(layout.set("unknown", 1).is_err());
        assert!(layout.get("unknown").is_err());
        assert_eq!(layout.entries().len(), RequestLayout::NAMES.len());
    }

    #[test]
    fn the_range_plan_binds_the_three_messages_to_their_trees() {
        use crate::proof::hidden_send_session::WShape;
        use zkie_engine::pcs::request::code::build_code_plan;
        let shape = |columns: u64, log_n: u32, ell: u64, r: u64, tree_number: u64| {
            WShape::new(
                columns,
                build_code_plan(log_n, ell, r).unwrap(),
                1,
                1,
                40,
                tree_number,
                4,
                zkie_core::types::Digest([0x81; 32]),
                zkie_core::types::Digest([0x82; 32]),
            )
            .unwrap()
        };
        let object = |tree_number: u64, oracle: u64| RangeObject {
            matrix: PackedMatrixPlan::new(4, 2).unwrap(),
            array_oracle: oracle,
            array_shape: shape(4, 3, 4, 2, tree_number),
        };
        let plan = RangePlan {
            objects: vec![object(910, 71), object(913, 74)],
            combination: RangeCombinationPlan::new(8, 2, 10, 0).unwrap(),
            combinations: 2,
            high_bits: 8,
            window: 1i128 << 17,
            group: 7,
            unit_oracle: 72,
            unit_shape: shape(4, 2, 4, 0, 911),
            digit_oracle: 73,
            digit_shape: shape(32, 5, 32, 0, 912),
        };
        assert!(plan.fault().is_none());
        assert_eq!(plan.limbs(), 2);
        assert_eq!(plan.offset(1).unwrap(), 4);
        assert_eq!(plan.cell_rows().unwrap(), 8);
        // A shared oracle or tree, another window, a combination count that is
        // not a power of two, a message that cannot carry its entries and a
        // digit message with another row count are refused. Two limb matrices
        // must share one geometry and the digit object must cover every row.
        let mut broken = plan.clone();
        broken.objects[1].array_oracle = broken.unit_oracle;
        assert!(broken.fault().is_some());
        let mut broken = plan.clone();
        broken.window = 1i128 << 16;
        assert!(broken.fault().is_some());
        let mut broken = plan.clone();
        broken.combinations = 3;
        assert!(broken.fault().is_some());
        let mut broken = plan.clone();
        broken.objects[0].array_shape.main_rows = 0;
        assert!(broken.fault().is_some());
        let mut broken = plan.clone();
        broken.digit_shape.main_rows = 1;
        assert!(broken.fault().is_some());
        let mut broken = plan.clone();
        broken.objects[1].matrix = PackedMatrixPlan::new(4, 4).unwrap();
        assert!(broken.fault().is_some());
        let mut broken = plan.clone();
        broken.combination = RangeCombinationPlan::new(4, 2, 10, 0).unwrap();
        assert!(broken.fault().is_some());
        let mut broken = plan;
        broken.unit_shape = shape(4, 2, 4, 0, 910);
        assert!(broken.fault().is_some());
    }

    #[test]
    fn the_section_round_trips_with_its_shape() {
        let messages = messages();
        let bytes = messages.encode(shape()).unwrap();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert_eq!(
            RequestChainMessages::decode(&mut reader, shape()).unwrap(),
            messages
        );
    }

    #[test]
    fn another_version_or_shape_is_refused() {
        let messages = messages();
        let mut bytes = messages.encode(shape()).unwrap();
        // The first eight bytes are the version: the current one decodes, the
        // previous one and any other value are refused.
        assert_eq!(REQUEST_CHAIN_VERSION, 3);
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(RequestChainMessages::decode(&mut reader, shape()).is_ok());
        let mut previous = bytes.clone();
        previous[..8].copy_from_slice(&2u64.to_le_bytes());
        let mut reader = CanonicalReader::new(&previous, DecodeLimits::unlimited());
        assert!(RequestChainMessages::decode(&mut reader, shape()).is_err());
        bytes[7] = 9;
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(RequestChainMessages::decode(&mut reader, shape()).is_err());
        let bytes = messages.encode(shape()).unwrap();
        // The encoder and the shape check refuse proofs of another shape.
        let mut wrong = shape();
        wrong.stack_rounds = 2;
        assert!(messages.encode(wrong).is_err());
        assert!(messages.check_shape(wrong).is_err());
        assert!(messages.check_shape(shape()).is_ok());
        // The decoder refuses stored counts that differ from the plan's.
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        let mut wrong = shape();
        wrong.merge_ends = 2;
        assert!(RequestChainMessages::decode(&mut reader, wrong).is_err());
        // Truncated bytes are refused.
        let mut reader = CanonicalReader::new(&bytes[..bytes.len() - 4], DecodeLimits::unlimited());
        assert!(RequestChainMessages::decode(&mut reader, shape()).is_err());
    }

    use crate::proof::psc_general::{
        general_variables, GeneralMaskEntries, Summand,
    };
    use crate::proof::request_reduction::{prove_token_kept, token_kept_claim, verify_token_kept};
    use crate::proof::request_stack::StackPanel;
    use crate::proof::request_start::token_start;
    use crate::proof::request_weights::{
        ClaimFactor, MergeValue, WeightClaim, WeightGroup, WeightLayout,
    };

    struct Source(u64);

    impl GeneralChallenges for Source {
        fn draw(&mut self) -> Result<Fp3> {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            Ok(Fp3::from_u64(self.0))
        }
        fn draw_nonzero(&mut self) -> Result<Fp3> {
            for _ in 0..64 {
                let value = self.draw()?;
                if value != Fp3::ZERO {
                    return Ok(value);
                }
            }
            Err(Error::EntropyUnavailable)
        }
        fn send_hiding_sum(&mut self, _value: Fp3) -> Result<()> {
            Ok(())
        }
        fn send_round(&mut self, _index: usize, _round: &PscRound) -> Result<()> {
            Ok(())
        }
        fn send_end_values(&mut self, _values: &[Fp3]) -> Result<()> {
            Ok(())
        }
    }

    fn chain_mask(slot: usize) -> Result<Fp3> {
        if (20..24).contains(&slot) {
            Ok(f(800 + (slot - 20) as u64))
        } else if slot == 3 {
            Ok(f(953))
        } else if slot >= START_HIDING_SLOT {
            Ok(f(700 + (slot - START_HIDING_SLOT) as u64))
        } else {
            Ok(f(950 + slot as u64))
        }
    }

    fn index_mask(_oracle: u64, slot: u64) -> Result<Fp3> {
        Ok(f(600 + slot))
    }

    fn hiding_count(variables: &[crate::proof::psc::Weight], summand: &Summand) -> usize {
        crate::proof::psc::hp_entries(&general_variables(variables, summand).unwrap()).unwrap()
    }

    #[test]
    fn the_chain_verifies_and_leaves_the_pending_claim() {
        let point = [f(2), f(3)];
        let token_shown = vec![f(1), f(2), f(3), f(4)];
        let token_kept = vec![f(5), f(6), f(7), f(8)];
        let hidden = token_kept_claim(&point, &token_shown, &token_kept).unwrap();
        let factor = f(11);
        let shown_claim = hidden.add(&factor.mul(&f(953)));
        let claim_masks = [MaskTerm {
            slot: 3,
            coefficient: factor,
        }];
        let token_variables = crate::proof::request_reduction::token_variables(&point).unwrap();
        let token_masks = GeneralMaskEntries {
            hiding: (0..hiding_count(&token_variables, &Summand::ShownKept))
                .map(|i| f(700 + i as u64))
                .collect(),
            shown: vec![f(800)],
        };
        // The token proof at the dry run gives the start form the stack needs.
        let (token_proof, _) = prove_token_kept(
            &point,
            &token_shown,
            &token_kept,
            &claim_masks,
            &chain_mask,
            shown_claim,
            &token_masks,
            &mut Source(61),
        )
        .unwrap();
        let token_output = verify_token_kept(
            &point,
            &claim_masks,
            shown_claim,
            &token_proof,
            token_masks.hiding.len(),
            &mut Source(61),
        )
        .unwrap();
        let mut source = Source(61);
        let epsilon = source.draw().unwrap();
        let batch = f(5);
        let start = token_start(&token_output, epsilon, 3, shown_claim, factor, batch).unwrap();
        let terms = start_terms(&start);
        let mut hidden_cstart = start.constant;
        for term in &terms {
            hidden_cstart =
                hidden_cstart.sub(&term.coefficient.mul(&chain_mask(term.slot).unwrap()));
        }
        let reduction = StackReduction::new(vec![
            StackPanel {
                claim: 0,
                ec: Fp3::ONE,
                ni: 2,
                ny: 1,
                weight: vec![Fp3::ONE, Fp3::ONE],
                first_claim: None,
                w: 0,
                off: 0,
            },
            StackPanel {
                claim: 0,
                ec: Fp3::ONE,
                ni: 2,
                ny: 1,
                weight: vec![Fp3::ONE, Fp3::ONE],
                first_claim: Some(0),
                w: 0,
                off: 0,
            },
        ])
        .unwrap();
        // The weight panel's rows are the merge's hidden values; the read
        // panel's second side is distinct per row so a swap is observable.
        let stack_right = vec![vec![f(2), f(3)], vec![f(10), f(20)]];
        // The packed product equals the hidden start: the read panel's first
        // cell carries the balance after the weight rows' contribution.
        let read_scale = batch.mul(&f(2));
        let panel_value = hidden_cstart
            .sub(&batch.mul(&f(30)))
            .mul(&read_scale.inverse().unwrap());
        let stack_left = vec![
            vec![panel_value, Fp3::ZERO],
            vec![Fp3::ONE, Fp3::ONE],
        ];
        let om = vec![batch];
        let stack_variables = (0..reduction.lam)
            .map(|_| crate::proof::psc::Weight::plain(3))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let stack_summand = Summand::PanelProduct {
            lam: reduction.lam,
            panels: reduction.panel_ends(),
        };
        let stack_masks = GeneralMaskEntries {
            hiding: (0..hiding_count(&stack_variables, &stack_summand))
                .map(|i| f(600 + i as u64))
                .collect(),
            shown: vec![f(800), f(801), f(802), f(803)],
        };
        let end_slots = [20usize, 21, 22, 23];
        // The weight merge: two claims over a two-bit column cube.
        let mut layout = WeightLayout::new(2).unwrap();
        layout
            .add(0, 0, 1, vec![WeightGroup::e("x", 4, false).unwrap()])
            .unwrap();
        let claims = vec![
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![f(2)], vec![f(3), f(5)])],
            },
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![f(7)], vec![f(11), f(13)])],
            },
        ];
        let maps = vec![
            vec![f(1), f(2), f(3), f(4)],
            vec![f(4), f(6), f(4), f(6)],
        ];
        let values = vec![
            MergeValue {
                shown: f(10),
                mask_slot: 0,
                mask_factor: f(3),
            },
            MergeValue {
                shown: f(20),
                mask_slot: 1,
                mask_factor: f(5),
            },
        ];
        let merge_variables = (0..layout.lk)
            .map(|_| crate::proof::psc::Weight::plain(2))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let merge_masks = GeneralMaskEntries {
            hiding: (0..hiding_count(&merge_variables, &Summand::Map))
                .map(|i| f(500 + i as u64))
                .collect(),
            shown: vec![f(801)],
        };
        let inputs = ChainInputs {
            token_point: &point,
            token_shown: &token_shown,
            token_kept: &token_kept,
            token_claim_masks: &claim_masks,
            token_shown_claim: shown_claim,
            token_claim_slot: 3,
            token_masks: &token_masks,
            reduction: &reduction,
            stack_left: &stack_left,
            stack_right: &stack_right,
            om: &om,
            stack_masks: &stack_masks,
            layout: &layout,
            claims: &claims,
            claim_maps: &maps,
            end_slots: &end_slots,
            merge_masks: &merge_masks,
            mask_value: &chain_mask,
            index_oracle: 60,
            index_slot: 7,
        };
        let (proof, proven) = prove_request_chain(&inputs, &mut Source(61)).unwrap();
        let outputs = verify_request_chain_direct(&inputs, &proof, &mut Source(61)).unwrap();
        // The formal verifier reads no private mask value and leaves the same
        // pending claim.
        let formal = verify_request_chain(&inputs, &proof, &mut Source(61)).unwrap();
        assert_eq!(formal, outputs);
        // The prover's pending claim is the verifier's: same transcript, same
        // row weights and mask slot.
        assert_eq!(proven, outputs);
        // The pending claim holds on a matrix built to satisfy it.
        let mut mask_total = Fp3::ZERO;
        for term in &outputs.pending.masks {
            mask_total = mask_total.add(&term.coeff.mul(&index_mask(term.oracle, term.slot).unwrap()));
        }
        let needed = outputs.pending.target.sub(&mask_total);
        let mut matrix = vec![vec![Fp3::ZERO; 4]; outputs.gamma.len()];
        let row = outputs
            .gamma
            .iter()
            .position(|value| *value != Fp3::ZERO)
            .unwrap();
        let cell = needed.mul(&outputs.gamma[row].inverse().unwrap());
        matrix[row] = vec![cell; 4];
        crate::proof::hidden_member::check_pending(&outputs.pending, &PlaceRows::new(&matrix), &index_mask).unwrap();
        // Another matrix value or mask value breaks the pending claim.
        let mut changed = matrix.clone();
        changed[row][0] = changed[row][0].add(&Fp3::ONE);
        assert!(crate::proof::hidden_member::check_pending(&outputs.pending, &PlaceRows::new(&changed), &index_mask).is_err());
        let other_mask = |_oracle: u64, slot: u64| Ok(f(601 + slot));
        assert!(crate::proof::hidden_member::check_pending(&outputs.pending, &PlaceRows::new(&matrix), &other_mask).is_err());
        // Another mask source, or a repeated end slot, is refused: every end
        // value has its own slot and its own entry.
        let other_chain_mask = |slot: usize| -> Result<Fp3> {
            let mut value = chain_mask(slot)?;
            if (20..24).contains(&slot) {
                value = value.add(&Fp3::ONE);
            }
            Ok(value)
        };
        let mut other_inputs = ChainInputs {
            token_point: &point,
            token_shown: &token_shown,
            token_kept: &token_kept,
            token_claim_masks: &claim_masks,
            token_shown_claim: shown_claim,
            token_claim_slot: 3,
            token_masks: &token_masks,
            reduction: &reduction,
            stack_left: &stack_left,
            stack_right: &stack_right,
            om: &om,
            stack_masks: &stack_masks,
            layout: &layout,
            claims: &claims,
            claim_maps: &maps,
            end_slots: &end_slots,
            merge_masks: &merge_masks,
            mask_value: &chain_mask,
            index_oracle: 60,
            index_slot: 7,
        };
        let other_ptr: &dyn Fn(usize) -> Result<Fp3> = &other_chain_mask;
        other_inputs.mask_value = other_ptr;
        assert!(verify_request_chain_direct(&other_inputs, &proof, &mut Source(61)).is_err());
        // The repeated slot: two end values naming the same mask slot.
        let repeated = [20usize, 21, 22, 22];
        let mut repeated_inputs = ChainInputs { end_slots: &repeated, ..other_inputs };
        repeated_inputs.mask_value = &chain_mask;
        repeated_inputs.end_slots = &repeated;
        assert!(verify_request_chain_direct(&repeated_inputs, &proof, &mut Source(61)).is_err());
        // Another read cell, swapped weight rows, or another weight cell
        // breaks the group's source ties or the merge's claim: these are the
        // changed-cell, swapped-row and swapped-object negatives at the chain.
        let mut changed_left = stack_left.clone();
        changed_left[0][0] = changed_left[0][0].add(&Fp3::ONE);
        let changed_left_inputs = ChainInputs {
            stack_left: &changed_left,
            ..inputs
        };
        assert!(verify_request_chain_direct(&changed_left_inputs, &proof, &mut Source(61)).is_err());
        // The read panel's second side is the verifier's read; swapping its
        // rows or changing a cell breaks the same tie. The weight cells are
        // the prover's maps; a changed weight cell is the matrix negative the
        // pending check already refuses.
        let mut swapped_right = stack_right.clone();
        swapped_right[0].swap(0, 1);
        let swapped_inputs = ChainInputs {
            stack_right: &swapped_right,
            ..inputs
        };
        assert!(verify_request_chain_direct(&swapped_inputs, &proof, &mut Source(61)).is_err());
        let mut changed_read = stack_right.clone();
        changed_read[0][1] = changed_read[0][1].add(&Fp3::ONE);
        let changed_read_inputs = ChainInputs {
            stack_right: &changed_read,
            ..inputs
        };
        assert!(verify_request_chain_direct(&changed_read_inputs, &proof, &mut Source(61)).is_err());
        // Another batch coefficient or a swapped weight slot is refused: the
        // batch is public plan data and each slot names its own entry.
        let changed_om = [batch.add(&Fp3::ONE)];
        let changed_om_inputs = ChainInputs { om: &changed_om, ..inputs };
        assert!(verify_request_chain_direct(&changed_om_inputs, &proof, &mut Source(61)).is_err());
        let swapped_slots = [20usize, 21, 23, 22];
        let swapped_slot_inputs = ChainInputs { end_slots: &swapped_slots, ..inputs };
        assert!(verify_request_chain_direct(&swapped_slot_inputs, &proof, &mut Source(61)).is_err());
        // A changed token proof breaks the stack's start.
        let mut tampered = proof.clone();
        tampered.token.rounds[0].coefficients[0] =
            tampered.token.rounds[0].coefficients[0].add(&Fp3::ONE);
        assert!(verify_request_chain_direct(&inputs, &tampered, &mut Source(61)).is_err());
        // A changed merge proof fails the merge's own row claim.
        let mut tampered = proof.clone();
        tampered.merge.rounds[0].coefficients[0] =
            tampered.merge.rounds[0].coefficients[0].add(&Fp3::ONE);
        assert!(verify_request_chain_direct(&inputs, &tampered, &mut Source(61)).is_err());
    }
}

#[cfg(test)]
mod chain_mask_tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};

    fn plan() -> ChainMaskPlan {
        ChainMaskPlan {
            oracle: 41,
            group: 7,
            mask_code: build_code_plan(8, 16, 4).unwrap(),
            mask_rows: 4,
            entries: 8,
            openings: 4,
            tree: RuntimeTreePlan {
                tree_number: 700,
                leaf_class: 9,
                log_n: 8,
                log_n_zk: 8,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 12,
                }],
                salt_bytes: 40,
                run: zkie_core::types::Digest([0x11; 32]),
                configuration: zkie_core::types::Digest([0x22; 32]),
            },
        }
    }

    #[test]
    fn the_chain_mask_layout_maps_slots_to_virtual_entries() {
        let plan = plan();
        assert!(plan.fault().is_none());
        // Four rows of sixteen places: the pad is the last place of the last
        // row, so the free entries are 4*15 + 3.
        assert_eq!(plan.free_entries().unwrap(), 63);
        assert_eq!(plan.entry_virtual_index(0).unwrap(), 0);
        assert_eq!(plan.entry_virtual_index(16).unwrap(), 1);
        assert_eq!(plan.entry_virtual_index(1).unwrap(), 4);
        let entries = chain_mask_entries(&plan, &[0, 16, 1]).unwrap();
        assert_eq!(entries[0], (41, 0));
        assert_eq!(entries[1], (41, 1));
        assert_eq!(entries[2], (41, 4));
        // A duplicate slot, a slot past the free entries and a broken plan are
        // refused.
        assert!(chain_mask_entries(&plan, &[2, 2]).is_err());
        assert!(chain_mask_entries(&plan, &[63]).is_err());
        let mut broken = plan.clone();
        broken.entries = 64;
        assert!(broken.fault().is_some());
        assert!(chain_mask_entries(&broken, &[0]).is_err());
        // The read ledger: the table's own tail bounds the accumulated
        // openings; another object never enters this count.
        let mut reads = ChainMaskReads::default();
        reads.open(&plan, 2).unwrap();
        reads.open(&plan, 2).unwrap();
        assert!(reads.open(&plan, 1).is_err());
        let mut zero = reads;
        assert!(zero.open(&plan, 0).is_err());
        // The sampled answer of a two-row table at a chosen seed: the cube
        // coefficients on the two rows plus the seed factor on the last entry.
        let mut small = plan.clone();
        small.mask_rows = 2;
        small.entries = 4;
        small.tree.parts[0].mask_symbols = 2 * 3;
        let seed = zkie_engine::pcs::request::table::SampleSeed {
            z: vec![
                Fp3::from_u64(2),
                Fp3::from_u64(3),
                Fp3::from_u64(5),
                Fp3::from_u64(7),
                Fp3::from_u64(9),
            ],
            beta: Fp3::from_u64(11),
        };
        let mut rows = vec![vec![Fp3::ZERO; 16]; 2];
        rows[0][0] = Fp3::from_u64(13);
        rows[1][3] = Fp3::from_u64(17);
        let answer = small.answer(&rows, &seed).unwrap();
        // Recompute the same cube independently.
        let bits = 1 + 4;
        let eq = crate::proof::output_closure::eq_vector(&seed.z[..bits]).unwrap();
        let mut expected = Fp3::ZERO;
        for index in 0u64..(1 << bits) {
            let row = (index & 1) as usize;
            let place = (index >> 1) as usize;
            if place >= 16 {
                continue;
            }
            let coefficient = if row == 1 && place == 15 {
                seed.beta
            } else {
                eq[index as usize]
            };
            expected = expected.add(&coefficient.mul(&rows[row][place]));
        }
        assert_eq!(answer, expected);
        // A zero factor, a short seed or rows of another shape are refused.
        let mut zero_beta = seed.clone();
        zero_beta.beta = Fp3::ZERO;
        assert!(small.answer(&rows, &zero_beta).is_err());
        let mut short = seed.clone();
        short.z.truncate(3);
        assert!(small.answer(&rows, &short).is_err());
        assert!(small.answer(&rows[..1], &seed).is_err());
        // One test's opening list: the plan count, positions inside the tree,
        // no repeats, and the table's own read ledger.
        // The sample form reads the mask table row bits and the mask code
        // message bits.
        assert_eq!(plan.sample_bits().unwrap(), 2 + 4);
        // The stack source rows: the shown A0/B0 against the table sides with
        // equation weights and their own mask slots.
        let (left_row, right_row) = stack_source_rows(
            21,
            &[0, 2],
            &[Fp3::from_u64(3), Fp3::from_u64(5)],
            21,
            &[1, 3],
            &[Fp3::from_u64(7), Fp3::from_u64(11)],
            Fp3::from_u64(31),
            Fp3::from_u64(41),
            41,
            6,
            9,
            Fp3::from_u64(13),
        )
        .unwrap();
        assert_eq!(left_row.target, Fp3::from_u64(31));
        assert_eq!(left_row.units.len(), 3);
        assert_eq!(left_row.units[2].index, 6);
        assert_eq!(left_row.units[2].coefficient, Fp3::from_u64(13));
        assert_eq!(right_row.target, Fp3::from_u64(41));
        assert_eq!(right_row.units[1].index, 3);
        assert_eq!(right_row.units[2].index, 9);
        // A weights list of another length and a zero factor are refused.
        assert!(stack_source_rows(
            21,
            &[0],
            &[Fp3::ONE],
            21,
            &[1],
            &[Fp3::ONE],
            Fp3::ONE,
            Fp3::ONE,
            41,
            6,
            9,
            Fp3::ZERO,
        )
        .is_err());
        // The merge remaining row: the output row form lands on the table
        // through the same trusted slot list and kept owners.
        let output = crate::proof::psc_general::GeneralOutput {
            point: Vec::new(),
            hiding_form: vec![Fp3::from_u64(2)],
            claim_scale: Fp3::ONE,
            row_target: Fp3::from_u64(21),
            ends: Vec::new(),
            row_maps: vec![(0, Fp3::from_u64(7))],
            claim_masks: vec![crate::proof::psc_general::MaskTerm {
                slot: 0,
                coefficient: Fp3::from_u64(5),
            }],
        };
        let merge_row = merge_remaining_row(&plan, &output, &[1], &[(9, 2)]).unwrap();
        assert_eq!(merge_row.target, Fp3::from_u64(21));
        assert_eq!(merge_row.units.len(), 3);
        assert_eq!(merge_row.units[1].index, plan.entry_virtual_index(1).unwrap());
        assert!(merge_remaining_row(&plan, &output, &[], &[(9, 2)]).is_err());
        // A sub-proof row form lands on the table: hiding entries at the first
        // entries, the counted slot through the plan slot list, the kept value
        // on its own owner.
        let form = crate::proof::row_claim::RowForm {
            constant: Fp3::from_u64(21),
            hiding: vec![Fp3::from_u64(2)],
            slots: vec![crate::proof::row_claim::SlotTerm {
                slot: 0,
                coefficient: Fp3::from_u64(5),
            }],
            kept: vec![(0, Fp3::from_u64(7))],
        };
        let row = chain_mask_row_form(&plan, &form, &[1], &[(9, 2)]).unwrap();
        assert_eq!(row.target, Fp3::from_u64(21));
        assert_eq!(row.units.len(), 3);
        assert_eq!(row.units[0].oracle, plan.oracle);
        assert_eq!(row.units[1].index, plan.entry_virtual_index(1).unwrap());
        assert_eq!(row.units[2].oracle, 9);
        assert_eq!(row.units[2].index, 2);
        // A slot without a trusted entry is refused.
        assert!(chain_mask_row_form(&plan, &form, &[], &[(9, 2)]).is_err());
        // A committed mask-only tree opening verifies against its root; another
        // root is refused.
        use zkie_engine::pcs::request::runtime_tree::{RuntimePartRows, RuntimeTree};
        let short = 1usize << plan.tree.log_n_zk;
        let rows = vec![vec![Fp3::from_u64(7); short]; 4];
        let salts = vec![vec![9u8; plan.tree.salt_bytes]; plan.tree.columns() as usize];
        let parts = [RuntimePartRows {
            main: Vec::new(),
            mask: rows,
            main_deg: 3,
        }];
        let tree = RuntimeTree::commit(plan.tree.clone(), &parts, &salts).unwrap();
        assert!(tree.plan().fault().is_none());
        let opening = tree.open(&[0, 2]).unwrap();
        verify_chain_mask_opening(&plan, tree.root(), &[0, 2], &opening).unwrap();
        let mut wrong = *tree.root();
        wrong.0[0] ^= 1;
        assert!(verify_chain_mask_opening(&plan, &wrong, &[0, 2], &opening).is_err());
        // A broken table plan is refused before any opening is checked.
        let mut broken_plan = plan.clone();
        broken_plan.openings = 0;
        assert!(broken_plan.fault().is_some());
        assert!(verify_chain_mask_opening(&broken_plan, tree.root(), &[0, 2], &opening).is_err());
        let mut fresh = ChainMaskReads::default();
        plan.check_positions(&[0, 1, 2, 3], &mut fresh).unwrap();
        assert_eq!(fresh.used, 4);
        assert!(plan.check_positions(&[0, 1, 2], &mut fresh).is_err());
        assert!(plan.check_positions(&[0, 1, 2, 2], &mut fresh).is_err());
        assert!(plan
            .check_positions(&[0, 1, 2, plan.tree.columns()], &mut fresh)
            .is_err());
    }
}
