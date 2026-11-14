//! Request-level weight layout and claim factors.
//!
//! A request's sub-proof reduces a weight matrix `M` with `2^lk` columns
//! (referred to by column bits) and `m0` rows. Weights are stored as digits
//! (one to a cell), so one weight is the sum over its digits' place values.
//! A claim is a sum over one unit's weights, each times a product of factors,
//! one for every index group of the unit. The layout splits each group's index
//! between the matrix's columns and its rows: a claim's coefficient on column
//! `c` of row `y` is `gamma[y] * eps[c]`.
//!
//! This module fixes the layout and the claim arithmetic: the column factor
//! at a point (`eps_at`), the row factor (`row_part`) and the per-row
//! coefficients (`gamma_add`). The merge that batches claims, the pending row
//! it leaves and the relation checks live in the caller.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

/// One weight is the sum over `j` of `SECOND_DIGIT^j` times its digit `j`.
pub const SECOND_DIGIT: u64 = 256;
/// The layout admits at most `2^ROWS_MOST_LOG2` rows.
pub const ROWS_MOST_LOG2: u32 = 22;

/// One index group of a unit's tensor.
///
/// Class `E`: `R = o * 2^z` with `o` odd; the index is `hi * 2^z + lo`, and
/// the lowest (or highest) `t` bits of `lo` are taken by the columns.
/// Class `A`: the whole index is taken (`t = ceil(log2 R)`) or none of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupClass {
    E,
    A,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightGroup {
    pub name: &'static str,
    pub class: GroupClass,
    pub r: u64,
    pub o: u64,
    /// The number of index values left to the rows.
    pub left: u64,
    pub high: bool,
    pub z: i32,
    pub t: i32,
    /// The group's lowest column bit.
    pub cb0: i32,
}

impl WeightGroup {
    pub fn e(name: &'static str, r: u64, high: bool) -> Result<Self> {
        if r == 0 {
            return Err(Error::InvalidConfig);
        }
        let mut o = r;
        let mut z = 0;
        while o & 1 == 0 {
            o >>= 1;
            z += 1;
        }
        Ok(Self {
            name,
            class: GroupClass::E,
            r,
            o,
            left: 0,
            high,
            z,
            t: 0,
            cb0: 0,
        })
    }

    pub fn a(name: &'static str, r: u64) -> Result<Self> {
        if r == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            name,
            class: GroupClass::A,
            r,
            o: r,
            left: 0,
            high: false,
            z: 0,
            t: 0,
            cb0: 0,
        })
    }

    fn taken_bit(&self, bit: i32) -> bool {
        if self.high {
            bit >= self.z - self.t
        } else {
            bit < self.t
        }
    }

    fn taken_at(&self, index: i32) -> i32 {
        if self.high {
            self.z - self.t + index
        } else {
            index
        }
    }

    fn a_taken(&self) -> bool {
        self.class == GroupClass::A && self.t > 0
    }
}

/// One unit's rows in the matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightUnit {
    pub kind: u32,
    pub layer: u32,
    /// Digits per weight.
    pub digits: usize,
    pub groups: Vec<WeightGroup>,
    pub rows: u64,
    pub off: u64,
    pub column_bits: i32,
}

/// The weight matrix's layout: `2^lk` columns and `m0` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightLayout {
    pub lk: i32,
    pub m0: u64,
    pub units: Vec<WeightUnit>,
}

impl WeightLayout {
    pub fn new(lk: i32) -> Result<Self> {
        if !(1..=20).contains(&lk) {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            lk,
            m0: 0,
            units: Vec::new(),
        })
    }

    /// Appends one unit from the current row offset on, assigning each group
    /// its taken bits, its lowest column bit and what is left to the rows.
    pub fn add(
        &mut self,
        kind: u32,
        layer: u32,
        digits: usize,
        groups: Vec<WeightGroup>,
    ) -> Result<usize> {
        if digits == 0 || groups.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let mut left = self.lk;
        let mut column_bit = 0;
        let mut rows: u128 = digits as u128;
        let mut groups = groups;
        for group in &mut groups {
            let taken = match group.class {
                GroupClass::E => group.z.min(left).max(0),
                GroupClass::A => {
                    let bits = ceil_log2(group.r) as i32;
                    if bits <= left {
                        bits
                    } else {
                        0
                    }
                }
            };
            group.t = taken;
            group.cb0 = column_bit;
            left -= taken;
            column_bit += taken;
            group.left = match group.class {
                GroupClass::E => group.r >> taken.max(0),
                GroupClass::A => {
                    if group.a_taken() {
                        1
                    } else {
                        group.r
                    }
                }
            };
            rows = rows
                .checked_mul(group.left as u128)
                .ok_or(Error::ResourceLimit)?;
            if rows > (1u128 << ROWS_MOST_LOG2) {
                return Err(Error::ResourceLimit);
            }
        }
        let rows = u64::try_from(rows).map_err(|_| Error::ResourceLimit)?;
        let unit = WeightUnit {
            kind,
            layer,
            digits,
            groups,
            rows,
            off: self.m0,
            column_bits: column_bit,
        };
        self.m0 = self
            .m0
            .checked_add(rows)
            .ok_or(Error::ResourceLimit)?;
        if self.m0 > (1u64 << ROWS_MOST_LOG2) {
            return Err(Error::ResourceLimit);
        }
        self.units.push(unit);
        Ok(self.units.len() - 1)
    }

    /// One claim's factor for every index group of its unit.
    #[allow(clippy::too_many_arguments)]
    pub fn claim_shape(&self, unit: usize, factors: &[ClaimFactor]) -> Result<()> {
        let unit = self.units.get(unit).ok_or(Error::InvalidIndex)?;
        if factors.len() != unit.groups.len() {
            return Err(Error::InvalidConfig);
        }
        for (group, factor) in unit.groups.iter().zip(factors) {
            match group.class {
                GroupClass::E => {
                    if factor.hv.len() as u64 != group.o
                        || factor.pt.len() as i32 != group.z
                        || !factor.av.is_empty()
                    {
                        return Err(Error::InvalidConfig);
                    }
                }
                GroupClass::A => {
                    if factor.av.len() as u64 != group.r
                        || !factor.hv.is_empty()
                        || !factor.pt.is_empty()
                    {
                        return Err(Error::InvalidConfig);
                    }
                }
            }
        }
        Ok(())
    }

    /// The claim's column factor at the point `rm`, whose coordinates are the
    /// highest column bit first.
    pub fn eps_at(&self, unit: usize, factors: &[ClaimFactor], rm: &[Fp3]) -> Result<Fp3> {
        if rm.len() != self.lk as usize {
            return Err(Error::ShapeMismatch);
        }
        let unit_ref = self.units.get(unit).ok_or(Error::InvalidIndex)?;
        let x = |bit: i32| -> Result<Fp3> {
            if bit < 0 || bit >= self.lk {
                return Err(Error::InvalidConfig);
            }
            Ok(rm[(self.lk - 1 - bit) as usize])
        };
        let mut value = Fp3::ONE;
        for (group, factor) in unit_ref.groups.iter().zip(factors) {
            match group.class {
                GroupClass::E => {
                    for index in 0..group.t {
                        let coordinate = factor.pt[(group.z - 1 - group.taken_at(index)) as usize];
                        value = value.mul(&eq1(coordinate, x(group.cb0 + index)?));
                    }
                }
                GroupClass::A => {
                    if group.a_taken() {
                        let mut point = Vec::with_capacity(group.t as usize);
                        for step in 0..group.t {
                            point.push(x(group.cb0 + group.t - 1 - step)?);
                        }
                        let table = eq_table(&point)?;
                        let mut sum = Fp3::ZERO;
                        for index in 0..group.r as usize {
                            sum = sum.add(&factor.av[index].mul(&table[index]));
                        }
                        value = value.mul(&sum);
                    }
                }
            }
        }
        for bit in unit_ref.column_bits..self.lk {
            value = value.mul(&Fp3::ONE.sub(&x(bit)?));
        }
        Ok(value)
    }

    /// What remains of the claim once the columns are fixed: its row factor,
    /// with the merge coefficient folded in.
    pub fn row_part(&self, unit: usize, factors: &[ClaimFactor], coefficient: Fp3) -> Result<RowClaim> {
        let unit_ref = self.units.get(unit).ok_or(Error::InvalidIndex)?;
        let mut row_factors = Vec::with_capacity(unit_ref.groups.len());
        for (group, factor) in unit_ref.groups.iter().zip(factors) {
            match group.class {
                GroupClass::E => {
                    let mut pl = Vec::new();
                    for bit in (0..group.z).rev() {
                        if !group.taken_bit(bit) {
                            pl.push(factor.pt[(group.z - 1 - bit) as usize]);
                        }
                    }
                    row_factors.push(RowFactor {
                        hv: factor.hv.clone(),
                        pl,
                        av: Vec::new(),
                    });
                }
                GroupClass::A => {
                    if group.a_taken() {
                        row_factors.push(RowFactor {
                            hv: Vec::new(),
                            pl: Vec::new(),
                            av: Vec::new(),
                        });
                    } else {
                        row_factors.push(RowFactor {
                            hv: Vec::new(),
                            pl: Vec::new(),
                            av: factor.av.clone(),
                        });
                    }
                }
            }
        }
        Ok(RowClaim {
            unit,
            coefficient,
            factors: row_factors,
        })
    }

    /// Adds `weight * coefficient * gamma[y]` to the row coefficients of the
    /// claim's unit. A row number is mixed-radix: the digit number most
    /// significant first, then the groups in reverse order, each with the
    /// part of its index left to the rows.
    pub fn gamma_add(&self, gamma: &mut [Fp3], claim: &RowClaim, weight: Fp3) -> Result<()> {
        let unit = self.units.get(claim.unit).ok_or(Error::InvalidIndex)?;
        if claim.factors.len() != unit.groups.len() {
            return Err(Error::InvalidConfig);
        }
        let mut values = Vec::with_capacity(unit.digits);
        let scale = weight.mul(&claim.coefficient);
        let mut factor = Fp3::ONE;
        for _ in 0..unit.digits {
            values.push(scale.mul(&factor));
            factor = factor.mul(&Fp3::from_u64(SECOND_DIGIT));
        }
        for (group, row) in unit.groups.iter().zip(&claim.factors).rev() {
            let left = usize::try_from(group.left).map_err(|_| Error::ResourceLimit)?;
            let mut rf = vec![Fp3::ZERO; left];
            match group.class {
                GroupClass::E => {
                    let untaken = (group.z - group.t).max(0);
                    if rf.len() != (group.o as usize) << untaken {
                        return Err(Error::InvalidConfig);
                    }
                    let table = eq_table(&row.pl)?;
                    for hi in 0..group.o as usize {
                        for lo in 0..(1usize << untaken) {
                            rf[(hi << untaken) + lo] = row.hv[hi].mul(&table[lo]);
                        }
                    }
                }
                GroupClass::A => {
                    if group.a_taken() {
                        if rf.len() != 1 {
                            return Err(Error::InvalidConfig);
                        }
                        rf[0] = Fp3::ONE;
                    } else {
                        if row.av.len() != rf.len() {
                            return Err(Error::InvalidConfig);
                        }
                        rf.copy_from_slice(&row.av);
                    }
                }
            }
            let mut next = vec![Fp3::ZERO; values.len() * rf.len()];
            for (i, value) in values.iter().enumerate() {
                for (q, factor) in rf.iter().enumerate() {
                    next[i * rf.len() + q] = value.mul(factor);
                }
            }
            values = next;
        }
        let rows = usize::try_from(unit.rows).map_err(|_| Error::ResourceLimit)?;
        if values.len() != rows {
            return Err(Error::InvalidConfig);
        }
        let off = usize::try_from(unit.off).map_err(|_| Error::ResourceLimit)?;
        let end = off.checked_add(rows).ok_or(Error::ResourceLimit)?;
        let target = gamma.get_mut(off..end).ok_or(Error::ShapeMismatch)?;
        for (slot, value) in target.iter_mut().zip(values) {
            *slot = slot.add(&value);
        }
        Ok(())
    }
}

/// One claim's value at the merge: the scalar the stack sub-proof sent as an
/// end value, the slot that value holds in the sub-proof's mask row and the
/// stack sumcheck's end factor. The factor is the stack's, not the merge's:
/// the merge's own masked target uses the request's index-mask entry and the
/// merge's end factor instead, and the two must not be mixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeValue {
    pub shown: Fp3,
    pub mask_slot: usize,
    pub mask_factor: Fp3,
}

impl MergeValue {
    /// One weight value as the stack sub-proof sent it: its shown end value,
    /// its slot in the sub-proof's mask row and the stack sumcheck's end
    /// factor.
    pub fn from_stack_end(shown: Fp3, mask_slot: usize, beta_stack: Fp3) -> Self {
        Self {
            shown,
            mask_slot,
            mask_factor: beta_stack,
        }
    }
}

/// What one merge produced: the masked sumcheck's proof, its output claim, the
/// claims' row coefficients and the accumulated row weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergeOutcome {
    pub proof: crate::proof::psc_general::GeneralProof,
    pub output: crate::proof::psc_general::GeneralOutput,
    pub rows: Vec<RowClaim>,
    pub gamma: Vec<Fp3>,
}

fn batch_powers(theta: Fp3, count: usize) -> Result<Vec<Fp3>> {
    let mut powers = Vec::with_capacity(count);
    let mut value = Fp3::ONE;
    for _ in 0..count {
        powers.push(value);
        value = value.mul(&theta);
    }
    Ok(powers)
}

fn merge_targets(
    values: &[MergeValue],
    rho: &[Fp3],
) -> Result<(Fp3, Vec<crate::proof::psc_general::MaskTerm>)> {
    if values.is_empty() || values.len() != rho.len() {
        return Err(Error::InvalidConfig);
    }
    let mut target = Fp3::ZERO;
    let mut masks = Vec::with_capacity(values.len());
    for (value, rho) in values.iter().zip(rho) {
        target = target.add(&rho.mul(&value.shown));
        masks.push(crate::proof::psc_general::MaskTerm {
            slot: value.mask_slot,
            coefficient: rho.mul(&value.mask_factor),
        });
    }
    Ok((target, masks))
}

/// Proves the merge of the weight claims. The batch challenge is drawn first;
/// its powers batch the claims' shown values into the sumcheck's target and
/// fold the claims' mask slots. The batched map is summed over the matrix's
/// column cube, its end value is masked by the sumcheck's last challenge, and
/// the claims' row coefficients and the accumulated row weights follow from
/// the sumcheck's point.
#[allow(clippy::too_many_arguments)]
pub fn prove_merge(
    layout: &WeightLayout,
    claims: &[WeightClaim],
    claim_maps: &[Vec<Fp3>],
    values: &[MergeValue],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
    masks: &crate::proof::psc_general::GeneralMaskEntries,
    challenges: &mut dyn crate::proof::psc_general::GeneralChallenges,
) -> Result<MergeOutcome> {
    if claims.is_empty()
        || claims.len() != claim_maps.len()
        || claims.len() != values.len()
        || values.len() > 4096
    {
        return Err(Error::InvalidConfig);
    }
    let cube = 1usize << layout.lk;
    if claim_maps.iter().any(|map| map.len() != cube) {
        return Err(Error::ShapeMismatch);
    }
    let theta = challenges.draw()?;
    let rho = batch_powers(theta, claims.len())?;
    let (target, claim_masks) = merge_targets(values, &rho)?;
    let mut map = vec![Fp3::ZERO; cube];
    for (claim_map, rho) in claim_maps.iter().zip(&rho) {
        for (slot, value) in map.iter_mut().zip(claim_map) {
            *slot = slot.add(&rho.mul(value));
        }
    }
    let variables = (0..layout.lk)
        .map(|_| crate::proof::psc::Weight::plain(2))
        .collect::<Result<Vec<_>>>()?;
    let mut inner = crate::proof::psc_general::MapInner::new(&map)?;
    let (proof, output) = crate::proof::psc_general::prove_general(
        &variables,
        &claim_masks,
        mask_value,
        target,
        &crate::proof::psc_general::Summand::Map,
        None,
        &mut inner,
        masks,
        challenges,
    )?;
    let (rows, gamma) = layout.merge_claims(claims, &rho, &output.point)?;
    Ok(MergeOutcome {
        proof,
        output,
        rows,
        gamma,
    })
}

/// Verifies one merge. The verifier draws the same batch challenge, rebuilds
/// the target and the claim's mask terms from the shown values, and checks the
/// sumcheck's messages; the row coefficients and row weights follow from the
/// verified point.
#[allow(clippy::too_many_arguments)]
pub fn verify_merge(
    layout: &WeightLayout,
    claims: &[WeightClaim],
    values: &[MergeValue],
    proof: &crate::proof::psc_general::GeneralProof,
    hiding_entries: usize,
    challenges: &mut dyn crate::proof::psc_general::GeneralChallenges,
) -> Result<(crate::proof::psc_general::GeneralOutput, Vec<RowClaim>, Vec<Fp3>)> {
    if claims.is_empty() || claims.len() != values.len() || values.len() > 4096 {
        return Err(Error::InvalidConfig);
    }
    let theta = challenges.draw()?;
    let rho = batch_powers(theta, claims.len())?;
    let (target, claim_masks) = merge_targets(values, &rho)?;
    let variables = (0..layout.lk)
        .map(|_| crate::proof::psc::Weight::plain(2))
        .collect::<Result<Vec<_>>>()?;
    let output = crate::proof::psc_general::verify_general(
        &variables,
        &claim_masks,
        target,
        &crate::proof::psc_general::Summand::Map,
        None,
        proof,
        hiding_entries,
        challenges,
    )?;
    let (rows, gamma) = layout.merge_claims(claims, &rho, &output.point)?;
    Ok((output, rows, gamma))
}

/// One claim about a unit's weights: a sum over the unit's rows with a
/// product of factors, one per index group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightClaim {
    pub unit: usize,
    pub factors: Vec<ClaimFactor>,
}

impl WeightLayout {
    /// Batches the claims at the merge: every claim's column factor at `rm`
    /// times its batch coefficient becomes the claim's row coefficient, and
    /// the row coefficients of all claims accumulate into one vector over the
    /// matrix's rows. The calls follow the merge: the claim values are fixed
    /// before the batch challenge, and `rm` comes from the sumcheck's rounds.
    pub fn merge_claims(
        &self,
        claims: &[WeightClaim],
        rho: &[Fp3],
        rm: &[Fp3],
    ) -> Result<(Vec<RowClaim>, Vec<Fp3>)> {
        if claims.is_empty() || claims.len() != rho.len() || rm.len() != self.lk as usize {
            return Err(Error::InvalidConfig);
        }
        let mut gamma = vec![Fp3::ZERO; usize::try_from(self.m0).map_err(|_| Error::ResourceLimit)?];
        let mut rows = Vec::with_capacity(claims.len());
        for (claim, rho) in claims.iter().zip(rho) {
            self.claim_shape(claim.unit, &claim.factors)?;
            let eps = self.eps_at(claim.unit, &claim.factors, rm)?;
            let coefficient = rho.mul(&eps);
            let row = self.row_part(claim.unit, &claim.factors, coefficient)?;
            self.gamma_add(&mut gamma, &row, Fp3::ONE)?;
            rows.push(row);
        }
        Ok((rows, gamma))
    }

    /// The pending claim the merge leaves: the accumulated row weights over
    /// the matrix's rows, the point over its columns consumed lowest column
    /// bit first, the masked target and its index-mask term.
    #[allow(clippy::too_many_arguments)]
    pub fn pending_row(
        &self,
        gamma: &[Fp3],
        rm: &[Fp3],
        beta: Fp3,
        target: Fp3,
        index_oracle: u64,
        mask_slot: u64,
    ) -> Result<crate::proof::hidden_member::PendingRow> {
        if gamma.len() as u64 != self.m0
            || rm.len() != self.lk as usize
            || !self.m0.is_power_of_two()
            || beta.is_zero()
        {
            return Err(Error::InvalidConfig);
        }
        let k_row = self.m0.trailing_zeros();
        let weights = crate::proof::hidden_member::stretch_weights(k_row, 0, gamma)?;
        let mut psi = rm.to_vec();
        psi.reverse();
        Ok(crate::proof::hidden_member::PendingRow {
            weights,
            psi,
            target,
            masks: vec![crate::proof::hidden_member::MaskTerm {
                oracle: index_oracle,
                slot: mask_slot,
                coeff: beta,
            }],
        })
    }
}

/// One claim's factor for one index group: `hv` and `pt` for class E, `av`
/// for class A.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimFactor {
    pub hv: Vec<Fp3>,
    pub pt: Vec<Fp3>,
    pub av: Vec<Fp3>,
}

impl ClaimFactor {
    pub fn e(hv: Vec<Fp3>, pt: Vec<Fp3>) -> Self {
        Self {
            hv,
            pt,
            av: Vec::new(),
        }
    }

    pub fn a(av: Vec<Fp3>) -> Self {
        Self {
            hv: Vec::new(),
            pt: Vec::new(),
            av,
        }
    }
}

/// The row part of one factor: `hv` and `pl` for class E, `av` for class A.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowFactor {
    pub hv: Vec<Fp3>,
    pub pl: Vec<Fp3>,
    pub av: Vec<Fp3>,
}

/// One claim after the columns are fixed at the merge's point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowClaim {
    pub unit: usize,
    pub coefficient: Fp3,
    pub factors: Vec<RowFactor>,
}

pub(crate) fn eq1(p: Fp3, x: Fp3) -> Fp3 {
    p.mul(&x).add(&Fp3::ONE.sub(&p).mul(&Fp3::ONE.sub(&x)))
}

fn ceil_log2(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// `eq(point, index)` for every index, with coordinate 0 the top bit.
pub(crate) fn eq_table(point: &[Fp3]) -> Result<Vec<Fp3>> {
    if point.len() >= 32 {
        return Err(Error::ResourceLimit);
    }
    let count = 1usize << point.len();
    let mut table = vec![Fp3::ONE; count];
    for (bit, coordinate) in point.iter().enumerate() {
        let shift = point.len() - 1 - bit;
        for (index, value) in table.iter_mut().enumerate() {
            let chosen = if (index >> shift) & 1 == 1 {
                *coordinate
            } else {
                Fp3::ONE.sub(coordinate)
            };
            *value = value.mul(&chosen);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    fn layout() -> WeightLayout {
        // Six column bits: one group low with R=8 takes its lowest three
        // bits, one group high with R=4 takes one of its two bits from the
        // top; two digits per weight.
        let mut layout = WeightLayout::new(6).unwrap();
        layout
            .add(
                0,
                0,
                2,
                vec![
                    WeightGroup::e("kk", 8, false).unwrap(),
                    WeightGroup::e("x", 4, true).unwrap(),
                ],
            )
            .unwrap();
        layout
    }

    #[test]
    fn column_bits_are_assigned_in_group_order() {
        let layout = layout();
        let unit = &layout.units[0];
        assert_eq!(unit.groups[0].z, 3);
        assert_eq!(unit.groups[0].t, 3);
        assert_eq!(unit.groups[0].cb0, 0);
        assert_eq!(unit.groups[0].left, 1);
        // The high group takes its two bits; one value is left to the rows.
        assert_eq!(unit.groups[1].z, 2);
        assert_eq!(unit.groups[1].t, 2);
        assert_eq!(unit.groups[1].cb0, 3);
        assert_eq!(unit.groups[1].left, 1);
        assert_eq!(unit.column_bits, 5);
        // Two digits and one value left to the rows.
        assert_eq!(unit.rows, 2);
        assert_eq!(layout.m0, 2);
    }

    #[test]
    fn eps_at_products_the_group_factors_and_the_untaken_bits() {
        let layout = layout();
        let factors = vec![
            ClaimFactor::e(vec![fp(1)], vec![fp(2), fp(3), fp(5)]),
            ClaimFactor::e(vec![fp(7)], vec![fp(13), fp(17)]),
        ];
        layout.claim_shape(0, &factors).unwrap();
        let rm = vec![fp(2), fp(3), fp(5), fp(7), fp(11), fp(13)];
        let x = |bit: i32| rm[(6 - 1 - bit) as usize];
        let mut expected = Fp3::ONE;
        // Group 0 low: bit 0 uses pt[2], bit 1 pt[1], bit 2 pt[0].
        expected = expected.mul(&eq1(fp(5), x(0)));
        expected = expected.mul(&eq1(fp(3), x(1)));
        expected = expected.mul(&eq1(fp(2), x(2)));
        // Group 1 high: taken bit 0 is coordinate pt[1], taken bit 1 is pt[0].
        expected = expected.mul(&eq1(fp(17), x(3)));
        expected = expected.mul(&eq1(fp(13), x(4)));
        // The one column bit no group takes multiplies (1 - x).
        expected = expected.mul(&Fp3::ONE.sub(&x(5)));
        assert_eq!(layout.eps_at(0, &factors, &rm).unwrap(), expected);
        assert!(layout.eps_at(0, &factors, &rm[..5]).is_err());
    }

    #[test]
    fn the_row_part_keeps_the_untaken_high_bits() {
        let layout = layout();
        let factors = vec![
            ClaimFactor::e(vec![fp(1)], vec![fp(2), fp(3), fp(5)]),
            ClaimFactor::e(vec![fp(7)], vec![fp(13), fp(17)]),
        ];
        let row = layout.row_part(0, &factors, fp(19)).unwrap();
        assert_eq!(row.coefficient, fp(19));
        // Both groups take all their bits; nothing is left to the rows.
        assert!(row.factors[0].pl.is_empty());
        assert!(row.factors[1].pl.is_empty());
        assert_eq!(row.factors[1].hv, vec![fp(7)]);
    }

    #[test]
    fn gamma_adds_the_digit_and_row_factors() {
        // One column bit, one low group of two values with one taken bit, two
        // digits: two rows.
        let mut layout = WeightLayout::new(1).unwrap();
        layout
            .add(0, 0, 2, vec![WeightGroup::e("x", 2, false).unwrap()])
            .unwrap();
        let factors = vec![ClaimFactor::e(vec![fp(3)], vec![fp(5)])];
        let row = layout.row_part(0, &factors, fp(7)).unwrap();
        let mut gamma = vec![Fp3::ZERO; 2];
        layout.gamma_add(&mut gamma, &row, fp(11)).unwrap();
        // Row 0 is the first digit; row 1 carries the second digit's place.
        assert_eq!(gamma[0], fp(11).mul(&fp(7)).mul(&fp(3)));
        assert_eq!(
            gamma[1],
            fp(11)
                .mul(&fp(7))
                .mul(&fp(3))
                .mul(&fp(SECOND_DIGIT))
        );
    }

    #[test]
    fn class_a_groups_are_taken_or_left_to_the_rows() {
        let mut layout = WeightLayout::new(2).unwrap();
        // A group of two values: its one bit is taken.
        layout
            .add(0, 0, 1, vec![WeightGroup::a("x", 2).unwrap()])
            .unwrap();
        let unit = &layout.units[0];
        assert_eq!(unit.groups[0].t, 1);
        assert_eq!(unit.groups[0].left, 1);
        let factors = vec![ClaimFactor::a(vec![fp(2), fp(3)])];
        layout.claim_shape(0, &factors).unwrap();
        let rm = vec![fp(5), fp(7)];
        let x = |bit: i32| rm[(2 - 1 - bit) as usize];
        // The taken bit's eq table sums av[0] (1-x) + av[1] x; the column bit
        // the unit does not take multiplies (1 - x).
        let expected = fp(2)
            .mul(&Fp3::ONE.sub(&x(0)))
            .add(&fp(3).mul(&x(0)))
            .mul(&Fp3::ONE.sub(&x(1)));
        assert_eq!(layout.eps_at(0, &factors, &rm).unwrap(), expected);
        let row = layout.row_part(0, &factors, fp(1)).unwrap();
        assert!(row.factors[0].av.is_empty());

        // A group of eight values cannot take its bits in two columns; all of
        // it is left to the rows.
        let mut layout = WeightLayout::new(2).unwrap();
        layout
            .add(0, 0, 1, vec![WeightGroup::a("x", 8).unwrap()])
            .unwrap();
        assert_eq!(layout.units[0].groups[0].t, 0);
        assert_eq!(layout.units[0].groups[0].left, 8);
        let av: Vec<Fp3> = (1..=8).map(fp).collect();
        let factors = vec![ClaimFactor::a(av.clone())];
        let expected = Fp3::ONE
            .sub(&x(0))
            .mul(&Fp3::ONE.sub(&x(1)));
        assert_eq!(layout.eps_at(0, &factors, &rm).unwrap(), expected);
        let row = layout.row_part(0, &factors, fp(1)).unwrap();
        assert_eq!(row.factors[0].av, av);
    }

    #[test]
    fn the_merge_accumulates_row_coefficients_and_the_pending_row() {
        // One column bit, one low group of two values, one digit: one row.
        let mut layout = WeightLayout::new(1).unwrap();
        layout
            .add(0, 0, 1, vec![WeightGroup::e("x", 2, false).unwrap()])
            .unwrap();
        let first = WeightClaim {
            unit: 0,
            factors: vec![ClaimFactor::e(vec![fp(3)], vec![fp(5)])],
        };
        let second = WeightClaim {
            unit: 0,
            factors: vec![ClaimFactor::e(vec![fp(7)], vec![fp(11)])],
        };
        let rm = vec![fp(13)];
        let rho = [fp(2), fp(3)];
        let (rows, gamma) = layout
            .merge_claims(&[first, second], &rho, &rm)
            .unwrap();
        // The claim's column factor at the single coordinate, times its batch
        // coefficient.
        let eps_first = eq1(fp(5), rm[0]);
        let eps_second = eq1(fp(11), rm[0]);
        assert_eq!(rows[0].coefficient, fp(2).mul(&eps_first));
        assert_eq!(rows[1].coefficient, fp(3).mul(&eps_second));
        // One row: the two row factors add.
        let expected = fp(2)
            .mul(&eps_first)
            .mul(&fp(3))
            .add(&fp(3).mul(&eps_second).mul(&fp(7)));
        assert_eq!(gamma, vec![expected]);

        let pending = layout
            .pending_row(&gamma, &rm, fp(17), fp(19), 60, 9)
            .unwrap();
        // The point's coordinates are consumed lowest column bit first.
        assert_eq!(pending.psi, vec![rm[0]]);
        assert_eq!(pending.target, fp(19));
        assert_eq!(pending.masks.len(), 1);
        assert_eq!(pending.masks[0].coeff, fp(17));
        assert_eq!(pending.row_bits().unwrap(), 0);
        assert!(layout
            .pending_row(&gamma, &rm, Fp3::ZERO, fp(19), 60, 9)
            .is_err());
        assert!(layout
            .pending_row(&gamma, &rm[..0], fp(1), fp(19), 60, 9)
            .is_err());
    }

    struct BatchSource(u64);

    impl crate::proof::psc_general::GeneralChallenges for BatchSource {
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
        fn send_round(&mut self, _index: usize, _round: &crate::proof::psc::PscRound) -> Result<()> {
            Ok(())
        }
        fn send_end_values(&mut self, _values: &[Fp3]) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn the_merge_takes_the_stack_group_values() {
        let mut layout = WeightLayout::new(2).unwrap();
        layout
            .add(0, 0, 1, vec![WeightGroup::e("x", 4, false).unwrap()])
            .unwrap();
        let claims = vec![
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![fp(2)], vec![fp(3), fp(5)])],
            },
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![fp(7)], vec![fp(11), fp(13)])],
            },
        ];
        // The maps sum to the group's hidden weight values.
        let maps = vec![
            vec![fp(1), fp(2), fp(3), fp(4)],
            vec![fp(4), fp(6), fp(4), fp(6)],
        ];
        let beta = fp(3);
        let mask = |slot: usize| Ok(fp(5 * (slot as u64 - 9)));
        let hidden = [fp(10), fp(20)];
        let sent = [
            hidden[0].add(&beta.mul(&mask(10).unwrap())),
            hidden[1].add(&beta.mul(&mask(11).unwrap())),
        ];
        let values = vec![
            MergeValue::from_stack_end(sent[0], 10, beta),
            MergeValue::from_stack_end(sent[1], 11, beta),
        ];
        let variables = [
            crate::proof::psc::Weight::plain(2).unwrap(),
            crate::proof::psc::Weight::plain(2).unwrap(),
        ];
        let hiding_count =
            crate::proof::psc::hp_entries(&crate::proof::psc_general::general_variables(
                &variables,
                &crate::proof::psc_general::Summand::Map,
            )
            .unwrap())
            .unwrap();
        let masks = crate::proof::psc_general::GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(900 + index as u64)).collect(),
            shown: vec![fp(999)],
        };
        let outcome = prove_merge(
            &layout,
            &claims,
            &maps,
            &values,
            &mask,
            &masks,
            &mut BatchSource(23),
        )
        .unwrap();
        let (output, _rows, _gamma) = verify_merge(
            &layout,
            &claims,
            &values,
            &outcome.proof,
            masks.hiding.len(),
            &mut BatchSource(23),
        )
        .unwrap();
        crate::proof::row_claim::RowForm::from_output(&output)
            .check(&masks.hiding, &mask, &[])
            .unwrap();
        // The merge's own end factor does not belong on the input values.
        let mixed = vec![
            MergeValue::from_stack_end(sent[0], 10, beta.add(&Fp3::ONE)),
            MergeValue::from_stack_end(sent[1], 11, beta.add(&Fp3::ONE)),
        ];
        assert!(prove_merge(
            &layout,
            &claims,
            &maps,
            &mixed,
            &mask,
            &masks,
            &mut BatchSource(23),
        )
        .is_err());
    }

    #[test]
    fn the_merge_proves_its_sumcheck_and_derives_the_row_weights() {
        let mut layout = WeightLayout::new(2).unwrap();
        layout
            .add(0, 0, 1, vec![WeightGroup::e("x", 4, false).unwrap()])
            .unwrap();
        let claims = vec![
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![fp(2)], vec![fp(3), fp(5)])],
            },
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![fp(7)], vec![fp(11), fp(13)])],
            },
        ];
        // The maps sum to the claims' hidden values.
        let maps = vec![
            vec![fp(1), fp(2), fp(3), fp(4)],
            vec![fp(4), fp(6), fp(4), fp(6)],
        ];
        let values = vec![
            MergeValue {
                shown: fp(10),
                mask_slot: 0,
                mask_factor: fp(1),
            },
            MergeValue {
                shown: fp(20),
                mask_slot: 1,
                mask_factor: fp(1),
            },
        ];
        let variables = [
            crate::proof::psc::Weight::plain(2).unwrap(),
            crate::proof::psc::Weight::plain(2).unwrap(),
        ];
        let hiding_entries = crate::proof::psc_general::general_masks(
            &crate::proof::psc_general::Summand::Map,
            &variables,
        )
        .unwrap();
        let hiding_count =
            crate::proof::psc::hp_entries(&crate::proof::psc_general::general_variables(
                &variables,
                &crate::proof::psc_general::Summand::Map,
            )
            .unwrap())
            .unwrap();
        assert_eq!(hiding_entries, hiding_count + 1);
        let masks = crate::proof::psc_general::GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(900 + index as u64)).collect(),
            shown: vec![fp(999)],
        };
        // The claims' mask entries are zero: their hidden values are the shown
        // ones, so the batched map sums to the target.
        let mask_value = |_slot: usize| Ok(Fp3::ZERO);
        let outcome = prove_merge(
            &layout,
            &claims,
            &maps,
            &values,
            &mask_value,
            &masks,
            &mut BatchSource(19),
        )
        .unwrap();
        assert_eq!(outcome.output.point.len(), 2);
        let (output, rows, gamma) = verify_merge(
            &layout,
            &claims,
            &values,
            &outcome.proof,
            masks.hiding.len(),
            &mut BatchSource(19),
        )
        .unwrap();
        assert_eq!(output, outcome.output);
        assert_eq!(rows, outcome.rows);
        assert_eq!(gamma, outcome.gamma);
        // The claim maps must fill the column cube.
        assert!(prove_merge(
            &layout,
            &claims,
            &[vec![fp(1)], vec![fp(2)]],
            &values,
            &mask_value,
            &masks,
            &mut BatchSource(19),
        )
        .is_err());
        // A changed round leaves another output claim.
        let mut changed = outcome.proof.clone();
        changed.rounds[0].coefficients[0] =
            changed.rounds[0].coefficients[0].add(&Fp3::ONE);
        let tampered = verify_merge(
            &layout,
            &claims,
            &values,
            &changed,
            masks.hiding.len(),
            &mut BatchSource(19),
        )
        .unwrap();
        assert_ne!(tampered.0, outcome.output);
    }

    #[test]
    fn claim_shapes_are_checked_against_the_groups() {
        let layout = layout();
        let bad = vec![ClaimFactor::e(vec![fp(1)], vec![fp(2)])];
        assert!(layout.claim_shape(0, &bad).is_err());
        assert!(layout.claim_shape(1, &bad).is_err());
        let wrong_class = vec![
            ClaimFactor::a(vec![fp(1)]),
            ClaimFactor::e(vec![fp(7)], vec![fp(13), fp(17)]),
        ];
        assert!(layout.claim_shape(0, &wrong_class).is_err());
    }
}
