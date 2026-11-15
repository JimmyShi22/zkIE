//! The digit lookups of the range object.
//!
//! The low and the high digit of every combination cell must be a row of a
//! public table: `0..2^T2D0` for the low plane and `0..2^T2D1` for the high
//! plane. Each plane is one complete single-call lookup over the committed
//! digit message: the keys are the plane's cells at their message places, the
//! multiplicities are counted per table row and committed bit by bit, and the
//! padding keys take the public value zero (their count is added to row zero's
//! multiplicity, so the product trees hold no extra padding factor).
//!
//! The two calls are independent lookups with their own table identity,
//! messages and mask layout. The reference shares one forest across both
//! calls; this carrier does not, and its cost is reported as ours.

use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::{Fp, Fp3};

use crate::proof::hidden_norm_psc::lookup_mask_rows;
use crate::proof::hidden_send_session::WShape;
use crate::proof::lookup::{
    mask_count, LookupKeyPlan, LookupKeySource, LookupPlan, LookupTable, BYTE_TABLE_IDENTITY,
};
use crate::proof::lookup_messages::LookupCommitPlan;
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
use crate::proof::lookup_table::{
    content_digest, TableConfig, TableEncoding, TABLE_ENCODING_VERSION,
};
use crate::proof::psc_general::TableAt;
use crate::proof::request_chain::RangePlan;

/// The low digit's width `T2D0`.
pub const LOW_DIGIT_LOG: u32 = 10;
/// The high digit's width `T2D1`.
pub const HIGH_DIGIT_LOG: u32 = 8;
/// The multiplicity bit planes: 32 planes cover every count below `2^32`.
pub const DIGIT_MU: u32 = 5;
/// The smallest key count's logarithm: at least 1024 key slots per call.
pub const DIGIT_KEY_BITS_MIN: u32 = 10;
/// The largest key count's logarithm one call supports without the reference's
/// multi-tree layout.
pub const DIGIT_KEY_BITS_MAX: u32 = 22;
/// The trusted identity of the public low-digit table `0, 1, .., 1023`.
pub const LOW_DIGIT_IDENTITY: Digest = Digest([0x64; 32]);

/// The public low-digit table, as a multilinear extension over ten
/// coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct LowDigitTable;

impl LookupTable for LowDigitTable {
    fn columns(&self) -> usize {
        1
    }

    fn row_count(&self) -> usize {
        1usize << LOW_DIGIT_LOG
    }

    fn identity(&self) -> Digest {
        LOW_DIGIT_IDENTITY
    }

    fn column_at(&self, column: usize, point: &[Fp3]) -> Result<Fp3> {
        if column != 0 {
            return Err(Error::InvalidIndex);
        }
        self.at(point)
    }

    fn column_rows(&self, column: usize) -> Result<Vec<Fp3>> {
        if column != 0 {
            return Err(Error::InvalidIndex);
        }
        Ok((0..(1u64 << LOW_DIGIT_LOG)).map(Fp3::from_u64).collect())
    }

    fn config(&self) -> TableConfig {
        low_digit_config()
    }

    fn check_content(&self) -> Result<()> {
        let rows: Vec<i64> = (0..(1i64 << LOW_DIGIT_LOG)).collect();
        if content_digest(TableEncoding::I64Le, &rows)? != low_digit_config().content_root {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

impl TableAt for LowDigitTable {
    fn at(&self, point: &[Fp3]) -> Result<Fp3> {
        if point.len() != LOW_DIGIT_LOG as usize {
            return Err(Error::ShapeMismatch);
        }
        let values = (0..(1u64 << LOW_DIGIT_LOG))
            .map(Fp3::from_u64)
            .collect::<Vec<_>>();
        crate::proof::lookup::fold_adjacent(&values, point)
    }
}

/// The trusted configuration of the public low-digit table.
fn low_digit_config() -> TableConfig {
    let rows: Vec<i64> = (0..(1i64 << LOW_DIGIT_LOG)).collect();
    let content_root =
        content_digest(TableEncoding::I64Le, &rows).expect("the low digit table rows encode");
    TableConfig {
        identity: LOW_DIGIT_IDENTITY,
        content_root,
        real_len: 1usize << LOW_DIGIT_LOG,
        encoding: TableEncoding::I64Le,
        version: TABLE_ENCODING_VERSION,
    }
}

/// One digit lookup of the range object: its plane, its public table, the
/// key sources on the committed digit message and the plans of its two
/// messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigitLookupPlan {
    /// 0 for the low digit, 1 for the high digit.
    pub plane: u32,
    pub lookup: LookupPlan,
    pub commit: LookupCommitPlan,
    pub key: LookupKeyPlan,
}

/// The parent-plan values of one digit lookup: the codes, identities, group
/// and the trusted tree numbers, salt and configuration of its two messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DigitLookupParams {
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub group: u64,
    pub bits_oracle: u64,
    pub mask_oracle: u64,
    pub bits_tree: u64,
    pub mask_tree: u64,
    pub leaf_class: u32,
    pub salt_bytes: usize,
    pub run: Digest,
    pub configuration: Digest,
}

impl DigitLookupPlan {
    /// The number of real digit keys `D = combinations * cell_rows()`.
    pub fn real_keys(&self) -> usize {
        self.key.sources.len()
    }

    /// The number of padding keys `N - D`.
    pub fn padding_keys(&self) -> Result<usize> {
        let keys = self.lookup.key_count()?;
        keys.checked_sub(self.real_keys())
            .ok_or(Error::InvalidConfig)
    }

    pub fn validate(&self, range: &RangePlan) -> Result<()> {
        self.lookup.validate()?;
        let bits_columns = self
            .lookup
            .planes()?
            .checked_mul(self.lookup.table_rows()?)
            .ok_or(Error::ResourceLimit)?;
        self.commit
            .validate(bits_columns, mask_count(&self.lookup)?)?;
        if self.commit.group != range.group
            || self.commit.bits_oracle == 0
            || self.commit.mask_oracle == 0
            || self.commit.bits_oracle == self.commit.mask_oracle
            || self.commit.mask_oracle == range.digit_oracle
            || self.commit.bits_oracle == range.digit_oracle
            || self.commit.bits_shape.fault(bits_columns as u64, 1).is_some()
        {
            return Err(Error::InvalidConfig);
        }
        if self.plane > 1 {
            return Err(Error::InvalidConfig);
        }
        let table_log = if self.plane == 0 {
            LOW_DIGIT_LOG
        } else {
            HIGH_DIGIT_LOG
        };
        if self.lookup.table_log != table_log
            || self.lookup.columns != 1
            || self.lookup.mu != DIGIT_MU
            || self.lookup.key_bits < DIGIT_KEY_BITS_MIN
            || self.lookup.key_bits > DIGIT_KEY_BITS_MAX
        {
            return Err(Error::InvalidConfig);
        }
        let identity = if self.plane == 0 {
            LOW_DIGIT_IDENTITY
        } else {
            BYTE_TABLE_IDENTITY
        };
        if self.key.table_identity != identity {
            return Err(Error::InvalidConfig);
        }
        let expected_sources = digit_key_count(range)?;
        if self.real_keys() != expected_sources || self.lookup.key_count()? < expected_sources {
            return Err(Error::ShapeMismatch);
        }
        self.key.validate(&self.lookup)?;
        Ok(())
    }
}

impl DigitLookupPlan {
    /// Pads one key vector with the public padding value to the key count.
    /// The real keys keep their order; only the tail is padding.
    pub fn pad_keys(&self, mut keys: Vec<Fp3>) -> Result<Vec<Fp3>> {
        let count = self.lookup.key_count()?;
        if keys.len() > count || self.key.padding_values.len() != 1 {
            return Err(Error::ShapeMismatch);
        }
        keys.resize(count, self.key.padding_values[0]);
        Ok(keys)
    }
}

/// The number of real digit cells of one plane: `tau` combinations over
/// the plane's `cell_rows()` places.
pub fn digit_key_count(range: &RangePlan) -> Result<usize> {
    let places = usize::try_from(range.cell_rows()?).map_err(|_| Error::ResourceLimit)?;
    (range.combinations as usize)
        .checked_mul(places)
        .ok_or(Error::ResourceLimit)
}

/// Sizes one plane's lookup and enumerates its key sources. The sources
/// cover the plane's real cells in dense order, one per key; the tail of the
/// key vector is padding with the public value zero.
fn digit_lookup_sizing(range: &RangePlan, plane: u32) -> Result<(LookupPlan, LookupKeyPlan)> {
    if plane > 1 {
        return Err(Error::InvalidConfig);
    }
    let real = digit_key_count(range)?;
    let mut key_bits = DIGIT_KEY_BITS_MIN;
    while (1usize << key_bits) < real {
        key_bits = key_bits.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    if key_bits > DIGIT_KEY_BITS_MAX {
        return Err(Error::UnsupportedPrimitive);
    }
    let table_log = if plane == 0 {
        LOW_DIGIT_LOG
    } else {
        HIGH_DIGIT_LOG
    };
    let lookup = LookupPlan {
        key_bits,
        mu: DIGIT_MU,
        table_log,
        columns: 1,
    };
    lookup.validate()?;
    let combinations = range.combinations;
    let rows_cell = u32::try_from(range.cell_rows()?).map_err(|_| Error::ResourceLimit)?;
    // The plane's stored columns are `plane * rows_cell .. (plane + 1) *
    // rows_cell`; the dense digit index is `place * tau + combination`, so the
    // key vector follows the dense order and the trusted plan fixes the
    // coverage.
    let first_place = plane
        .checked_mul(rows_cell)
        .ok_or(Error::ResourceLimit)?;
    let mut sources = Vec::with_capacity(real);
    let digit_rows = range.digit_shape.main_rows;
    for place in first_place..first_place.checked_add(rows_cell).ok_or(Error::ResourceLimit)? {
        for combination in 0..combinations {
            // The source address is the dense entry read through the digit
            // message's own row count, the same convention as every value
            // message.
            let dense = (place as u64)
                .checked_mul(combinations as u64)
                .and_then(|value| value.checked_add(combination as u64))
                .ok_or(Error::ResourceLimit)?;
            let row = u32::try_from(dense % digit_rows).map_err(|_| Error::ResourceLimit)?;
            let column_place =
                u32::try_from(dense / digit_rows).map_err(|_| Error::ResourceLimit)?;
            sources.push(LookupKeySource {
                column: 0,
                oracle: range.digit_oracle,
                position: 0,
                row,
                place: column_place,
                coefficient: Fp3::ONE,
                offset: Fp3::ZERO,
            });
        }
    }
    if sources.len() != real {
        return Err(Error::ShapeMismatch);
    }
    let key = LookupKeyPlan {
        sources,
        padding_values: vec![Fp3::ZERO],
        table_identity: if plane == 0 {
            LOW_DIGIT_IDENTITY
        } else {
            BYTE_TABLE_IDENTITY
        },
    };
    Ok((lookup, key))
}

/// The multiplicity-bit message shape of one lookup: `planes * table_rows`
/// entries over the parent code.
fn lookup_shape(lookup: &LookupPlan, params: &DigitLookupParams) -> Result<WShape> {
    let columns = (lookup.planes()? * lookup.table_rows()?)
        .try_into()
        .map_err(|_| Error::ResourceLimit)?;
    let shape = WShape::new(
        columns,
        params.code,
        1,
        1,
        params.salt_bytes,
        params.bits_tree,
        params.leaf_class,
        params.run,
        params.configuration,
    )?;
    if shape.main_rows > (1u64 << params.code.log_n) {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

/// The mask message's tree over the parent mask code.
fn mask_tree(lookup: &LookupPlan, params: &DigitLookupParams) -> Result<RuntimeTreePlan> {
    let mask_rows = lookup_mask_rows(lookup, params.mask_code)?;
    let mask_symbols = mask_rows
        .checked_mul(zkie_engine::pcs::request::field::E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    Ok(RuntimeTreePlan {
        tree_number: params.mask_tree,
        leaf_class: params.leaf_class,
        log_n: params.mask_code.log_n,
        log_n_zk: params.mask_code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: 0,
            mask_symbols,
        }],
        salt_bytes: params.salt_bytes,
        run: params.run,
        configuration: params.configuration,
    })
}

/// Builds one plane's complete lookup plan over the range object's digit
/// message with the caller's message plans.
pub fn digit_lookup_plan(
    range: &RangePlan,
    plane: u32,
    params: &DigitLookupParams,
) -> Result<DigitLookupPlan> {
    let (lookup, key) = digit_lookup_sizing(range, plane)?;
    let bits_shape = lookup_shape(&lookup, params)?;
    let mask_rows = lookup_mask_rows(&lookup, params.mask_code)?;
    let tree = mask_tree(&lookup, params)?;
    let commit = LookupCommitPlan {
        bits_shape,
        bits_oracle: params.bits_oracle,
        group: params.group,
        bits_openings: 1,
        mask_oracle: params.mask_oracle,
        mask_code: params.mask_code,
        mask_rows,
        mask_openings: 1,
        mask_tree: tree,
    };
    let plan = DigitLookupPlan {
        plane,
        lookup,
        commit,
        key,
    };
    plan.validate(range)?;
    Ok(plan)
}

/// The test plan's parent values: the fixture's group code, a mask code and
/// distinct identities for the two messages.
#[cfg(test)]
fn test_params() -> DigitLookupParams {
    use zkie_engine::pcs::request::code::build_code_plan;
    DigitLookupParams {
        code: build_code_plan(12, 32, 4).unwrap(),
        mask_code: build_code_plan(12, 32, 4).unwrap(),
        group: 7,
        bits_oracle: 81,
        mask_oracle: 82,
        bits_tree: 921,
        mask_tree: 922,
        leaf_class: 4,
        salt_bytes: 40,
        run: Digest([0x83; 32]),
        configuration: Digest([0x84; 32]),
    }
}

/// The multiplicity bit planes of one plane's digit cells: the count of each
/// table row over the real keys plus the padding keys on row zero, bit
/// `plane + planes * row` of the table row's count. Every digit must be a
/// table row; a value outside the table is refused, never reduced.
pub fn digit_multiplicity_bits(
    plan: &DigitLookupPlan,
    digit_values: &[Fp3],
) -> Result<Vec<Fp3>> {
    let keys = plan.lookup.key_count()?;
    let planes = plan.lookup.planes()?;
    let rows = plan.lookup.table_rows()?;
    if plan.real_keys() > keys || digit_values.len() != plan.real_keys() {
        return Err(Error::ShapeMismatch);
    }
    let mut counts = vec![0u64; rows];
    for value in digit_values {
        let index = value.c[0].as_canonical_u64();
        if value.c[1] != Fp::ZERO || value.c[2] != Fp::ZERO
            || index >= rows as u64
        {
            return Err(Error::BoundViolation);
        }
        counts[index as usize] = counts[index as usize]
            .checked_add(1)
            .ok_or(Error::ResourceLimit)?;
    }
    // The padding keys take the public value zero and are counted on row zero
    // with the real zeros, so the product trees carry no extra factor.
    counts[0] = counts[0]
        .checked_add((keys - digit_values.len()) as u64)
        .ok_or(Error::ResourceLimit)?;
    if planes < 32 && counts.iter().any(|count| *count >= (1u64 << planes)) {
        return Err(Error::BoundViolation);
    }
    let mut bits = vec![Fp3::ZERO; planes.checked_mul(rows).ok_or(Error::ResourceLimit)?];
    for row in 0..rows {
        for plane in 0..planes {
            if (counts[row] >> plane) & 1 == 1 {
                bits[plane + planes * row] = Fp3::ONE;
            }
        }
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::hidden_send_session::WShape;
    use crate::proof::packed_matrix::{PackedMatrixPlan, RangeCombinationPlan};
    use crate::proof::request_chain::{RangeObject, RangePlan};
    use zkie_engine::pcs::request::code::build_code_plan;

    fn shape(columns: u64, log_n: u32, ell: u64, r: u64, tree_number: u64) -> WShape {
        WShape::new(
            columns,
            build_code_plan(log_n, ell, r).unwrap(),
            1,
            1,
            40,
            tree_number,
            4,
            Digest([0x81; 32]),
            Digest([0x82; 32]),
        )
        .unwrap()
    }

    fn range() -> RangePlan {
        RangePlan {
            objects: vec![RangeObject {
                matrix: PackedMatrixPlan::new(4, 2).unwrap(),
                array_oracle: 71,
                array_shape: shape(4, 3, 4, 2, 910),
            }],
            combination: RangeCombinationPlan::new(4, 2, 10, 0).unwrap(),
            combinations: 2,
            high_bits: 8,
            window: 1i128 << 17,
            group: 7,
            unit_oracle: 72,
            unit_shape: shape(4, 2, 4, 0, 911),
            digit_oracle: 73,
            digit_shape: shape(16, 4, 16, 0, 912),
        }
    }

    #[test]
    fn the_digit_lookup_enumerates_every_cell_once() {
        let range = range();
        for plane in 0..2u32 {
            let plan = digit_lookup_plan(&range, plane, &test_params()).unwrap();
            assert_eq!(plan.real_keys(), 8);
            assert_eq!(plan.padding_keys().unwrap(), 1024 - 8);
            assert_eq!(plan.lookup.key_bits, 10);
            let mut seen = std::collections::BTreeSet::new();
            for source in &plan.key.sources {
                assert_eq!(source.oracle, range.digit_oracle);
                assert_eq!(source.column, 0);
                assert_eq!(source.coefficient, Fp3::ONE);
                assert_eq!(source.offset, Fp3::ZERO);
                assert!(source.row < range.combinations);
                let first = (plane as u32) * (range.cell_rows().unwrap() as u32);
                assert!(source.place >= first && source.place < first + range.cell_rows().unwrap() as u32);
                assert!(seen.insert((source.row, source.place)));
            }
            assert_eq!(seen.len(), 8);
            // The message plans carry the multiplicity-bit values and the
            // lookup's mask rows under distinct identities in the group.
            let bits = 32usize * plan.lookup.table_rows().unwrap();
            assert_eq!(plan.commit.group, range.group);
            assert_eq!(plan.commit.bits_oracle, 81);
            assert_eq!(plan.commit.mask_oracle, 82);
            assert!(plan.commit.bits_shape.fault(bits as u64, 1).is_none());
            assert_eq!(
                plan.commit.mask_rows,
                crate::proof::hidden_norm_psc::lookup_mask_rows(
                    &plan.lookup,
                    plan.commit.mask_code
                )
                .unwrap()
            );
        }
        // A plan whose plane disagrees with its table identity is refused.
        let mut broken = digit_lookup_plan(&range, 0, &test_params()).unwrap();
        broken.plane = 1;
        assert!(broken.validate(&range).is_err());
    }

    #[test]
    fn the_digit_lookups_prove_and_verify_on_their_tables() {
        use crate::proof::lookup::{
            mask_count, prove_lookup_with_key_values, verify_lookup, ByteTable,
            LookupTable, LookupWitness,
        };
        use crate::proof::lookup_messages::{commit_lookup, LookupRandomSource, LookupRoots};
        use zkie_engine::pcs::request::chain::Chain;
        let range = range();
        for plane in 0..2u32 {
            let plan = digit_lookup_plan(&range, plane, &test_params()).unwrap();
            let rows = plan.lookup.table_rows().unwrap() as u64;
            let real: Vec<Fp3> = (0..plan.real_keys())
                .map(|index| Fp3::from_u64((index as u64 * 7 + plane as u64) % rows))
                .collect();
            let bits = digit_multiplicity_bits(&plan, &real).unwrap();
            let values = plan.pad_keys(real).unwrap();
            let masks = mask_count(&plan.lookup).unwrap();
            let mut source = LookupRandomSource::new(9 + plane as u64);
            let messages = commit_lookup(&plan.commit, &bits, masks, &mut source).unwrap();
            let table: &dyn LookupTable = if plane == 0 {
                &LowDigitTable
            } else {
                &ByteTable
            };
            let mut prover = Chain::begin(
                0x4C4B_5550,
                &Digest::ZERO,
                &[],
                &Digest::ZERO.0,
                None,
            )
            .unwrap();
            let (proof, proven) = prove_lookup_with_key_values(
                &plan.lookup,
                &plan.commit,
                &plan.key,
                &values,
                &LookupWitness { bits },
                &messages,
                table,
                &mut prover,
            )
            .unwrap();
            let mut verifier = Chain::begin(
                0x4C4B_5550,
                &Digest::ZERO,
                &[],
                &Digest::ZERO.0,
                None,
            )
            .unwrap();
            let verified = verify_lookup(
                &plan.lookup,
                &plan.commit,
                &plan.key,
                &LookupRoots {
                    bits: messages.bits_root(),
                    mask: messages.mask_root,
                },
                &proof,
                table,
                &mut verifier,
            )
            .unwrap();
            assert_eq!(proven, verified);
        }
        // A key outside the table cannot be proved: the product tie fails.
        let plan = digit_lookup_plan(&range, 0, &test_params()).unwrap();
        let rows = plan.lookup.table_rows().unwrap() as u64;
        let mut real: Vec<Fp3> = (0..plan.real_keys())
            .map(|index| Fp3::from_u64(index as u64 % rows))
            .collect();
        real[0] = Fp3::from_u64(rows);
        let mut values = plan.pad_keys(real).unwrap();
        values[0] = Fp3::from_u64(rows);
        // The multiplicity bits count the in-table keys only; the foreign key
        // stays in the key vector, so the product tie must fail.
        let planes = plan.lookup.planes().unwrap();
        let mut counts = vec![0u64; rows as usize];
        for value in &values {
            let index = value.c[0].as_canonical_u64();
            if index < rows {
                counts[index as usize] += 1;
            }
        }
        counts[0] += (plan.lookup.key_count().unwrap() - values.len()) as u64;
        let mut bits = vec![Fp3::ZERO; planes * rows as usize];
        for row in 0..rows as usize {
            for plane in 0..planes {
                if (counts[row] >> plane) & 1 == 1 {
                    bits[plane + planes * row] = Fp3::ONE;
                }
            }
        }
        let masks = mask_count(&plan.lookup).unwrap();
        let mut source = LookupRandomSource::new(21);
        let messages = commit_lookup(&plan.commit, &bits, masks, &mut source).unwrap();
        let mut prover = Chain::begin(
            0x4C4B_5550,
            &Digest::ZERO,
            &[],
            &Digest::ZERO.0,
            None,
        )
        .unwrap();
        assert!(prove_lookup_with_key_values(
            &plan.lookup,
            &plan.commit,
            &plan.key,
            &values,
            &LookupWitness { bits },
            &messages,
            &LowDigitTable,
            &mut prover,
        )
        .is_err());
    }

    #[test]
    fn the_multiplicity_bits_count_every_table_row_once() {
        let range = range();
        let plan = digit_lookup_plan(&range, 0, &test_params()).unwrap();
        let mut digits = vec![Fp3::from_u64(3); 8];
        digits[0] = Fp3::from_u64(0);
        digits[1] = Fp3::from_u64(0);
        let bits = digit_multiplicity_bits(&plan, &digits).unwrap();
        let planes = plan.lookup.planes().unwrap();
        let rows = plan.lookup.table_rows().unwrap();
        assert_eq!(bits.len(), planes * rows);
        let mut count0 = 0u64;
        let mut count3 = 0u64;
        for plane in 0..planes {
            if bits[plane + planes * 0] == Fp3::ONE {
                count0 |= 1 << plane;
            }
            if bits[plane + planes * 3] == Fp3::ONE {
                count3 |= 1 << plane;
            }
        }
        // Row zero's count includes the padding keys, so the product trees
        // carry no extra padding factor.
        assert_eq!(count0, 2 + (1024 - 8));
        assert_eq!(count3, 6);
        // A digit outside the table is refused by both planes.
        let mut outside = vec![Fp3::from_u64(3); 8];
        outside[0] = Fp3::from_u64(1024);
        assert!(digit_multiplicity_bits(&plan, &outside).is_err());
        let high = digit_lookup_plan(&range, 1, &test_params()).unwrap();
        let mut outside = vec![Fp3::from_u64(200); 8];
        outside[0] = Fp3::from_u64(256);
        assert!(digit_multiplicity_bits(&high, &outside).is_err());
        // Wrong key counts are refused before any count is taken.
        assert!(digit_multiplicity_bits(&plan, &digits[..7]).is_err());
    }
}
