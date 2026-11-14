//! General masked product-sumcheck foundation for the lookup argument.
//!
//! The product sumcheck in [`crate::proof::psc`] fixes the summand
//! `S(y_0, y_1) = y_0 y_1`, two shown maps and no kept maps. The lookup
//! argument needs the same masking and hiding machinery for:
//!
//! - the product summand over two halves of a tree level;
//! - the tie summands, which combine sent roots without a sum over X;
//! - the bit summand `y^2 - y`;
//! - the affine leaf summand `1 + b (alpha - 1 - tau)`, whose coefficient is a
//!   public table's multilinear extension.
//!
//! This module states the summand forms, the output claims (shown end values
//! and kept-map terms) and the general prover/verifier around a caller-supplied
//! inner sumcheck. The existing product, dot, closure and norm callers keep
//! their own message types and are unchanged.
//!
//! Scope: no new commitment, message or challenge order enters this module. It
//! is arithmetic and claim layout only. The leaf summand's inner sumcheck is
//! supplied by the lookup module, where the public table is fixed.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::psc::{
    fold_first, hp_entries, hp_form, product_round, HidingPolynomial, PointClaim,
    ProductSumcheckProver, PscRound, Weight,
};

/// A source of verifier challenges for a general sumcheck.
pub trait GeneralChallenges {
    fn draw(&mut self) -> Result<Fp3>;
    fn draw_nonzero(&mut self) -> Result<Fp3>;
    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()>;
    fn send_round(&mut self, index: usize, round: &PscRound) -> Result<()>;
    fn send_end_values(&mut self, values: &[Fp3]) -> Result<()>;
}

/// A public table's multilinear extension at a point.
pub trait TableAt {
    fn at(&self, point: &[Fp3]) -> Result<Fp3>;
}

/// One weight panel's end geometry: its offset and width in the stack cube and
/// its row and entry counts. The end coefficient of a row is the anchor's
/// `cB = sb * EI[i]`, computed at the end point from this geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelEnd {
    pub off: u64,
    pub w: u32,
    pub ni: u64,
    pub ny: u64,
}

fn panel_ceil_log2(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

fn panel_eq1(coordinate: Fp3, bit: Fp3) -> Fp3 {
    // (1 - coordinate)(1 - bit) + coordinate * bit
    Fp3::ONE
        .sub(&coordinate)
        .mul(&Fp3::ONE.sub(&bit))
        .add(&coordinate.mul(&bit))
}

fn panel_eq_low(point: &[Fp3], lo: usize, hi: usize) -> Vec<Fp3> {
    let count = 1usize << (hi - lo);
    let mut table = vec![Fp3::ONE; count];
    for j in 0..(hi - lo) {
        for (index, value) in table.iter_mut().enumerate() {
            let chosen = if (index >> j) & 1 == 1 {
                point[lo + j]
            } else {
                Fp3::ONE.sub(&point[lo + j])
            };
            *value = value.mul(&chosen);
        }
    }
    table
}

fn panel_eq_at(point: &[Fp3], lo: usize, hi: usize, index: u64) -> Fp3 {
    let mut value = Fp3::ONE;
    for j in 0..(hi - lo) {
        let bit = if (index >> j) & 1 == 1 {
            Fp3::ONE
        } else {
            Fp3::ZERO
        };
        value = value.mul(&panel_eq1(point[lo + j], bit));
    }
    value
}

/// The summand of a general masked sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Summand {
    /// `y_0 y_1`: two shown maps, no kept map, degree two.
    Product,
    /// `y_0 - a y_1^2 y_2`: three shown maps, degree three.
    TieFirst { a: Fp3 },
    /// `y_0^2 y_1`: two shown maps, degree three.
    TieNext,
    /// `y_0^2 - y_0`: one shown map, degree two.
    Bits,
    /// `1 + b_i (alpha - 1 - tau(point))`: no shown map, one kept map.
    Leaves { alpha: Fp3, mu: usize },
    /// `y_0`: one shown map, degree one. The map is summed over the cube.
    Map,
    /// `y_0 k_0`: one shown map and one kept map, degree one in the shown map.
    /// The kept map's value is not sent: its coefficient is the shown value as
    /// it was sent, and it stays a term of the row claim.
    ShownKept,
    /// `(A0 + sum ku_i u_i)(B0 + sum kv_a v_a)`: the whole end group is sent,
    /// in the order A0, the u values, B0, the v values, each with its own mask
    /// entry and the same factor. The last round in Y is
    /// `(A* + a1 Y)(B* + b1 Y)` with the masks folded in as the anchor states.
    StackProduct { ku: Vec<Fp3>, kv: Vec<Fp3> },
    /// `A0 (B0 + sum_panels sum_i cB_i v_i)`: the whole end group is sent as
    /// A0, B0 and then each weight panel's rows. `cB_i = sb * EI[i]` follows
    /// from the geometry at the end point, so the weight values stay the
    /// unpacked row values the anchor sends.
    PanelProduct { lam: u32, panels: Vec<PanelEnd> },
    /// `sum_t coeff_t * prod_{j in t} y_j`: a sum of products of shown maps.
    ProductSum { terms: Vec<(Fp3, Vec<usize>)> },
}

impl Summand {
    pub fn shown(&self) -> usize {
        match self {
            Self::Product => 2,
            Self::TieFirst { .. } => 3,
            Self::TieNext => 2,
            Self::Bits => 1,
            Self::Leaves { .. } => 0,
            Self::Map | Self::ShownKept => 1,
            Self::StackProduct { ku, kv } => 2 + ku.len() + kv.len(),
            Self::PanelProduct { panels, .. } => {
                2 + panels.iter().map(|panel| panel.ni as usize).sum::<usize>()
            }
            Self::ProductSum { terms } => terms
                .iter()
                .flat_map(|(_, indices)| indices.iter())
                .max()
                .map_or(0, |index| index + 1),
        }
    }

    pub fn kept(&self) -> usize {
        match self {
            Self::Leaves { .. } | Self::ShownKept => 1,
            _ => 0,
        }
    }

    pub fn degree(&self) -> usize {
        match self {
            Self::Product => 2,
            Self::TieFirst { .. } | Self::TieNext => 3,
            Self::Bits => 2,
            Self::Leaves { .. } => 0,
            Self::Map | Self::ShownKept => 1,
            Self::StackProduct { .. } | Self::PanelProduct { .. } => 2,
            Self::ProductSum { terms } => terms
                .iter()
                .map(|(_, indices)| indices.len())
                .max()
                .unwrap_or(0),
        }
    }

    /// The summand and the coefficients of its kept maps, the constant first.
    /// Kept-map values are not needed by the leaf summand; the caller may pass
    /// zeros for them.
    pub fn at(
        &self,
        point: &[Fp3],
        shown: &[Fp3],
        kept: &[Fp3],
        table: Option<&dyn TableAt>,
    ) -> Result<Vec<Fp3>> {
        if shown.len() != self.shown() || kept.len() != self.kept() {
            return Err(Error::ShapeMismatch);
        }
        match self {
            Self::Product => Ok(vec![shown[0].mul(&shown[1])]),
            Self::TieFirst { a } => Ok(vec![shown[0].sub(
                &a.mul(&shown[1]).mul(&shown[1]).mul(&shown[2]),
            )]),
            Self::TieNext => Ok(vec![shown[0].mul(&shown[0]).mul(&shown[1])]),
            Self::Bits => Ok(vec![shown[0].mul(&shown[0]).sub(&shown[0])]),
            Self::Leaves { alpha, mu } => {
                let table_point = point.get(*mu..).ok_or(Error::ShapeMismatch)?;
                let table_at = table
                    .ok_or(Error::UnsupportedPrimitive)?
                    .at(table_point)?;
                Ok(vec![Fp3::ONE, alpha.sub(&Fp3::ONE).sub(&table_at)])
            }
            Self::Map => Ok(vec![shown[0]]),
            Self::ShownKept => Ok(vec![Fp3::ZERO, shown[0]]),
            Self::PanelProduct { lam, panels } => {
                // The last round passes the point with the further variable
                // appended; the geometry uses the X coordinates only.
                if point.len() < *lam as usize {
                    return Err(Error::ShapeMismatch);
                }
                let mut bs = shown[1];
                let mut index = 2usize;
                for panel in panels {
                    let wy = panel_ceil_log2(panel.ny) as usize;
                    let w = panel.w as usize;
                    if w < wy || w > *lam as usize {
                        return Err(Error::InvalidConfig);
                    }
                    let sb = if panel.w == *lam {
                        Fp3::ONE
                    } else {
                        panel_eq_at(point, w, *lam as usize, panel.off >> panel.w)
                    };
                    let ei = panel_eq_low(point, wy, w);
                    if ei.len() as u64 != panel.ni {
                        return Err(Error::InvalidConfig);
                    }
                    for coefficient in ei.iter().take(panel.ni as usize) {
                        bs = bs.add(&sb.mul(coefficient).mul(&shown[index]));
                        index += 1;
                    }
                }
                if index != shown.len() {
                    return Err(Error::ShapeMismatch);
                }
                Ok(vec![shown[0].mul(&bs)])
            }
            Self::StackProduct { ku, kv } => {
                let mut left = shown[0];
                for (index, coefficient) in ku.iter().enumerate() {
                    left = left.add(&coefficient.mul(&shown[1 + index]));
                }
                let base = 1 + ku.len();
                let mut right = shown[base];
                for (index, coefficient) in kv.iter().enumerate() {
                    right = right.add(&coefficient.mul(&shown[base + 1 + index]));
                }
                Ok(vec![left.mul(&right)])
            }
            Self::ProductSum { terms } => {
                let mut sum = Fp3::ZERO;
                for (coefficient, indices) in terms {
                    let mut term = *coefficient;
                    for index in indices {
                        let value = shown.get(*index).ok_or(Error::ShapeMismatch)?;
                        term = term.mul(value);
                    }
                    sum = sum.add(&term);
                }
                Ok(vec![sum])
            }
        }
    }
}

/// The prover's inner unmasked sumcheck: round polynomials over X, the
/// challenge binding and the shown/kept map values at the final point.
pub trait InnerSumcheck {
    fn round(&self) -> Result<Vec<Fp3>>;
    fn bind(&mut self, challenge: Fp3) -> Result<()>;
    /// Shown map values first, then kept map values.
    fn end_values(&self) -> Result<Vec<Fp3>>;
}

/// The stack summand's inner: the aggregated maps' rounds and the whole end
/// group at the point, with the rest values the anchor sends.
pub struct StackInner {
    a0: Vec<Fp3>,
    u: Vec<Vec<Fp3>>,
    b0: Vec<Fp3>,
    v: Vec<Vec<Fp3>>,
    ku: Vec<Fp3>,
    kv: Vec<Fp3>,
}

impl StackInner {
    pub fn new(
        a0: &[Fp3],
        u: &[Vec<Fp3>],
        b0: &[Fp3],
        v: &[Vec<Fp3>],
        ku: Vec<Fp3>,
        kv: Vec<Fp3>,
    ) -> Result<Self> {
        if a0.len() < 2
            || !a0.len().is_power_of_two()
            || b0.len() != a0.len()
            || ku.len() != u.len()
            || kv.len() != v.len()
        {
            return Err(Error::ShapeMismatch);
        }
        if u.iter().chain(v.iter()).any(|map| map.len() != a0.len()) {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            a0: a0.to_vec(),
            u: u.to_vec(),
            b0: b0.to_vec(),
            v: v.to_vec(),
            ku,
            kv,
        })
    }

    fn left(&self) -> Vec<Fp3> {
        let mut left = self.a0.clone();
        for (coefficient, map) in self.ku.iter().zip(&self.u) {
            for (slot, value) in left.iter_mut().zip(map) {
                *slot = slot.add(&coefficient.mul(value));
            }
        }
        left
    }

    fn right(&self) -> Vec<Fp3> {
        let mut right = self.b0.clone();
        for (coefficient, map) in self.kv.iter().zip(&self.v) {
            for (slot, value) in right.iter_mut().zip(map) {
                *slot = slot.add(&coefficient.mul(value));
            }
        }
        right
    }
}

impl InnerSumcheck for StackInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        Ok(product_round(&self.left(), &self.right())?.to_vec())
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        fold_first(&mut self.a0, challenge)?;
        for map in &mut self.u {
            fold_first(map, challenge)?;
        }
        fold_first(&mut self.b0, challenge)?;
        for map in &mut self.v {
            fold_first(map, challenge)?;
        }
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.a0.len() != 1 || self.b0.len() != 1 || self.u.iter().chain(&self.v).any(|map| map.len() != 1) {
            return Err(Error::InvalidSessionState);
        }
        let mut values = Vec::with_capacity(2 + self.ku.len() + self.kv.len());
        let mut rest = self.left()[0];
        for (coefficient, map) in self.ku.iter().zip(&self.u) {
            rest = rest.sub(&coefficient.mul(&map[0]));
        }
        values.push(rest);
        for map in &self.u {
            values.push(map[0]);
        }
        let mut rest = self.right()[0];
        for (coefficient, map) in self.kv.iter().zip(&self.v) {
            rest = rest.sub(&coefficient.mul(&map[0]));
        }
        values.push(rest);
        for map in &self.v {
            values.push(map[0]);
        }
        Ok(values)
    }
}

/// The product summand's inner sumcheck over two maps.
pub struct ProductInner(ProductSumcheckProver);

impl ProductInner {
    pub fn new(x: &[Fp3], omega: &[Fp3]) -> Result<Self> {
        Ok(Self(ProductSumcheckProver::new(x, omega)?))
    }
}

impl InnerSumcheck for ProductInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        Ok(self.0.round()?.to_vec())
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        self.0.bind(challenge)
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        Ok(self.0.end_values()?.to_vec())
    }
}

/// The product summand's inner sumcheck under equality weights on the X
/// variables. The current round's polynomial carries the equality factors of
/// the variables after it; the variable being bound is weighted by the
/// verifier's .
pub struct WeightedProductInner {
    x: Vec<Fp3>,
    omega: Vec<Fp3>,
    z: Vec<Fp3>,
    variables: usize,
    bound: usize,
}

impl WeightedProductInner {
    pub fn new(x: &[Fp3], omega: &[Fp3], z: &[Fp3]) -> Result<Self> {
        if x.len() != omega.len() || x.len() < 2 || !x.len().is_power_of_two() {
            return Err(Error::ShapeMismatch);
        }
        let variables = x.len().trailing_zeros() as usize;
        if z.len() != variables {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            x: x.to_vec(),
            omega: omega.to_vec(),
            z: z.to_vec(),
            variables,
            bound: 0,
        })
    }
}

impl InnerSumcheck for WeightedProductInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        if self.bound >= self.variables || self.x.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let pairs = self.x.len() / 2;
        let future = self.variables - self.bound - 1;
        let mut round = vec![Fp3::ZERO; 3];
        for index in 0..pairs {
            let mut weight = Fp3::ONE;
            for bit in 0..future {
                let value = if (index >> bit) & 1 == 1 {
                    self.z[self.bound + 1 + bit]
                } else {
                    Fp3::ONE.sub(&self.z[self.bound + 1 + bit])
                };
                weight = weight.mul(&value);
            }
            let low_x = self.x[2 * index];
            let delta_x = self.x[2 * index + 1].sub(&low_x);
            let low_o = self.omega[2 * index];
            let delta_o = self.omega[2 * index + 1].sub(&low_o);
            round[0] = round[0].add(&weight.mul(&low_x).mul(&low_o));
            round[1] = round[1].add(
                &weight.mul(&low_x.mul(&delta_o).add(&delta_x.mul(&low_o))),
            );
            round[2] = round[2].add(&weight.mul(&delta_x).mul(&delta_o));
        }
        Ok(round)
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables || self.x.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let rows = self.x.len() / 2;
        for index in 0..rows {
            let low = self.x[2 * index];
            let high = self.x[2 * index + 1];
            self.x[index] = low.add(&challenge.mul(&high.sub(&low)));
            let low = self.omega[2 * index];
            let high = self.omega[2 * index + 1];
            self.omega[index] = low.add(&challenge.mul(&high.sub(&low)));
        }
        self.x.truncate(rows);
        self.omega.truncate(rows);
        self.bound += 1;
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.bound != self.variables || self.x.len() != 1 || self.omega.len() != 1 {
            return Err(Error::InvalidSessionState);
        }
        Ok(vec![self.x[0], self.omega[0]])
    }
}

/// The bit summand's inner sumcheck over one map.
pub struct BitsInner {
    values: Vec<Fp3>,
    variables: usize,
    bound: usize,
}

impl BitsInner {
    pub fn new(values: &[Fp3]) -> Result<Self> {
        if values.len() < 2 || !values.len().is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            values: values.to_vec(),
            variables: values.len().trailing_zeros() as usize,
            bound: 0,
        })
    }

    fn round_impl(&self) -> Result<Vec<Fp3>> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let mut round = vec![Fp3::ZERO; 3];
        for index in 0..self.values.len() / 2 {
            let low = self.values[2 * index];
            let delta = self.values[2 * index + 1].sub(&low);
            round[0] = round[0].add(&low.mul(&low).sub(&low));
            round[1] = round[1].add(
                &low
                    .mul(&delta)
                    .add(&delta.mul(&low))
                    .sub(&delta),
            );
            round[2] = round[2].add(&delta.mul(&delta));
        }
        Ok(round)
    }
}

impl InnerSumcheck for BitsInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        self.round_impl()
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let rows = self.values.len() / 2;
        for index in 0..rows {
            let low = self.values[2 * index];
            let high = self.values[2 * index + 1];
            self.values[index] = low.add(&challenge.mul(&high.sub(&low)));
        }
        self.values.truncate(rows);
        self.bound += 1;
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.bound != self.variables || self.values.len() != 1 {
            return Err(Error::InvalidSessionState);
        }
        Ok(vec![self.values[0]])
    }
}

/// The bit summand's inner sumcheck under equality weights.
pub struct WeightedBitsInner {
    values: Vec<Fp3>,
    z: Vec<Fp3>,
    variables: usize,
    bound: usize,
}

impl WeightedBitsInner {
    pub fn new(values: &[Fp3], z: &[Fp3]) -> Result<Self> {
        if values.len() < 2 || !values.len().is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        let variables = values.len().trailing_zeros() as usize;
        if z.len() != variables {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            values: values.to_vec(),
            z: z.to_vec(),
            variables,
            bound: 0,
        })
    }

    fn future_weight(&self, index: usize) -> Fp3 {
        let mut weight = Fp3::ONE;
        for bit in 0..(self.variables - self.bound - 1) {
            let coordinate = self.z[self.bound + 1 + bit];
            let value = if (index >> bit) & 1 == 1 {
                coordinate
            } else {
                Fp3::ONE.sub(&coordinate)
            };
            weight = weight.mul(&value);
        }
        weight
    }
}

impl InnerSumcheck for WeightedBitsInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let mut round = vec![Fp3::ZERO; 3];
        for index in 0..self.values.len() / 2 {
            let weight = self.future_weight(index);
            let low = self.values[2 * index];
            let high = self.values[2 * index + 1];
            let delta = high.sub(&low);
            let at_two = high.add(&delta);
            let v0 = low.mul(&low).sub(&low);
            let v1 = high.mul(&high).sub(&high);
            let v2 = at_two.mul(&at_two).sub(&at_two);
            let half = Fp3::from_u64(2).inverse()?;
            let quadratic = v2
                .sub(&v1.mul(&Fp3::from_u64(2)))
                .add(&v0)
                .mul(&half);
            round[0] = round[0].add(&weight.mul(&v0));
            round[1] = round[1].add(&weight.mul(&v1.sub(&v0).sub(&quadratic)));
            round[2] = round[2].add(&weight.mul(&quadratic));
        }
        Ok(round)
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let rows = self.values.len() / 2;
        for index in 0..rows {
            let low = self.values[2 * index];
            let high = self.values[2 * index + 1];
            self.values[index] = low.add(&challenge.mul(&high.sub(&low)));
        }
        self.values.truncate(rows);
        self.bound += 1;
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.bound != self.variables || self.values.len() != 1 {
            return Err(Error::InvalidSessionState);
        }
        Ok(vec![self.values[0]])
    }
}

/// The affine leaf summand's inner sumcheck under equality weights. The kept
/// map is `b`; the public table contributes `tau` on the same cube and the
/// summand is `1 + b (alpha - 1 - tau)`.
pub struct LeavesInner {
    values: Vec<Fp3>,
    table: Vec<Fp3>,
    z: Vec<Fp3>,
    alpha: Fp3,
    variables: usize,
    bound: usize,
}

impl LeavesInner {
    pub fn new(values: &[Fp3], table: &[Fp3], z: &[Fp3], alpha: Fp3) -> Result<Self> {
        if values.len() != table.len() || values.len() < 2 || !values.len().is_power_of_two() {
            return Err(Error::ShapeMismatch);
        }
        let variables = values.len().trailing_zeros() as usize;
        if z.len() != variables {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            values: values.to_vec(),
            table: table.to_vec(),
            z: z.to_vec(),
            alpha,
            variables,
            bound: 0,
        })
    }

    fn future_weight(&self, index: usize) -> Fp3 {
        let mut weight = Fp3::ONE;
        for bit in 0..(self.variables - self.bound - 1) {
            let coordinate = self.z[self.bound + 1 + bit];
            let value = if (index >> bit) & 1 == 1 {
                coordinate
            } else {
                Fp3::ONE.sub(&coordinate)
            };
            weight = weight.mul(&value);
        }
        weight
    }
}

impl InnerSumcheck for LeavesInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let alpha_less_one = self.alpha.sub(&Fp3::ONE);
        let mut round = vec![Fp3::ONE, Fp3::ZERO, Fp3::ZERO];
        for index in 0..self.values.len() / 2 {
            let weight = self.future_weight(index);
            let b_low = self.values[2 * index];
            let b_delta = self.values[2 * index + 1].sub(&b_low);
            let t_low = self.table[2 * index];
            let t_delta = self.table[2 * index + 1].sub(&t_low);
            round[0] = round[0].add(
                &weight
                    .mul(&alpha_less_one.mul(&b_low).sub(&b_low.mul(&t_low))),
            );
            round[1] = round[1].add(
                &weight.mul(
                    &alpha_less_one
                        .mul(&b_delta)
                        .sub(&b_low.mul(&t_delta).add(&b_delta.mul(&t_low))),
                ),
            );
            round[2] = round[2].sub(&weight.mul(&b_delta).mul(&t_delta));
        }
        Ok(round)
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let rows = self.values.len() / 2;
        for index in 0..rows {
            let low = self.values[2 * index];
            let high = self.values[2 * index + 1];
            self.values[index] = low.add(&challenge.mul(&high.sub(&low)));
            let low = self.table[2 * index];
            let high = self.table[2 * index + 1];
            self.table[index] = low.add(&challenge.mul(&high.sub(&low)));
        }
        self.values.truncate(rows);
        self.table.truncate(rows);
        self.bound += 1;
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.bound != self.variables || self.values.len() != 1 {
            return Err(Error::InvalidSessionState);
        }
        Ok(vec![self.values[0]])
    }
}

/// An inner sumcheck whose shown map values are supplied directly. Used by the
/// tie steps, which have no X variables.
pub struct ScalarInner {
    values: Vec<Fp3>,
    bound: bool,
}

impl ScalarInner {
    pub fn new(values: &[Fp3]) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            values: values.to_vec(),
            bound: false,
        })
    }
}

impl InnerSumcheck for ScalarInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        Err(Error::UnsupportedPrimitive)
    }

    fn bind(&mut self, _challenge: Fp3) -> Result<()> {
        self.bound = true;
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        Ok(self.values.clone())
    }
}

/// A single map's inner sumcheck over the Boolean cube: every variable is
/// summed plainly, so a round polynomial is linear and its two coefficients
/// are the round's sum at the two points of the variable.
pub struct MapInner {
    values: Vec<Fp3>,
    variables: usize,
    bound: usize,
}

impl MapInner {
    pub fn new(values: &[Fp3]) -> Result<Self> {
        if values.len() < 2 || !values.len().is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            values: values.to_vec(),
            variables: values.len().trailing_zeros() as usize,
            bound: 0,
        })
    }
}

impl InnerSumcheck for MapInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let mut low = Fp3::ZERO;
        let mut high = Fp3::ZERO;
        for pair in self.values.chunks_exact(2) {
            low = low.add(&pair[0]);
            high = high.add(&pair[1]);
        }
        Ok(vec![low, high.sub(&low)])
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let mut next = Vec::with_capacity(self.values.len() / 2);
        for pair in self.values.chunks_exact(2) {
            next.push(pair[0].add(&challenge.mul(&pair[1].sub(&pair[0]))));
        }
        self.values = next;
        self.bound += 1;
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.bound != self.variables || self.values.len() != 1 {
            return Err(Error::InvalidSessionState);
        }
        Ok(vec![self.values[0]])
    }
}

/// A single map's inner sumcheck under equality weights on the X variables.
/// The current round's polynomial carries the equality factors of the
/// variables after it; the variable being bound is weighted by the verifier.
pub struct WeightedMapInner {
    values: Vec<Fp3>,
    z: Vec<Fp3>,
    variables: usize,
    bound: usize,
}

impl WeightedMapInner {
    pub fn new(values: &[Fp3], z: &[Fp3]) -> Result<Self> {
        if values.len() < 2 || !values.len().is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        let variables = values.len().trailing_zeros() as usize;
        if z.len() != variables {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            values: values.to_vec(),
            z: z.to_vec(),
            variables,
            bound: 0,
        })
    }
}

impl InnerSumcheck for WeightedMapInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let future = self.variables - self.bound - 1;
        let mut low = Fp3::ZERO;
        let mut delta = Fp3::ZERO;
        for (index, pair) in self.values.chunks_exact(2).enumerate() {
            let mut weight = Fp3::ONE;
            for bit in 0..future {
                let value = if (index >> bit) & 1 == 1 {
                    self.z[self.bound + 1 + bit]
                } else {
                    Fp3::ONE.sub(&self.z[self.bound + 1 + bit])
                };
                weight = weight.mul(&value);
            }
            low = low.add(&weight.mul(&pair[0]));
            delta = delta.add(&weight.mul(&pair[1].sub(&pair[0])));
        }
        Ok(vec![low, delta])
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables || self.values.len() % 2 != 0 {
            return Err(Error::InvalidSessionState);
        }
        let mut next = Vec::with_capacity(self.values.len() / 2);
        for pair in self.values.chunks_exact(2) {
            next.push(pair[0].add(&challenge.mul(&pair[1].sub(&pair[0]))));
        }
        self.values = next;
        self.bound += 1;
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.bound != self.variables || self.values.len() != 1 {
            return Err(Error::InvalidSessionState);
        }
        Ok(vec![self.values[0]])
    }
}

/// The mask entries of one general sumcheck: hiding coefficients first, then
/// one entry for each shown map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneralMaskEntries {
    pub hiding: Vec<Fp3>,
    pub shown: Vec<Fp3>,
}

/// One claim mask term: a slot in the mask message and its coefficient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskTerm {
    pub slot: usize,
    pub coefficient: Fp3,
}

/// A claim on a multilinear map at a point, with a linear form in mask entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneralClaim {
    pub point: Vec<Fp3>,
    pub value: Fp3,
    pub masks: Vec<MaskTerm>,
}

/// LINE: reduces two claims on the two halves of a multilinear map at one point
/// to one claim on the whole map. The verifier draws the last coordinate.
pub fn line(
    challenges: &mut dyn GeneralChallenges,
    half0: GeneralClaim,
    half1: GeneralClaim,
) -> Result<GeneralClaim> {
    if half0.point != half1.point {
        return Err(Error::ShapeMismatch);
    }
    let rho = challenges.draw()?;
    let one_minus = Fp3::ONE.sub(&rho);
    let mut point = half0.point;
    point.push(rho);
    let mut masks = Vec::with_capacity(half0.masks.len() + half1.masks.len());
    for term in half0.masks {
        masks.push(MaskTerm {
            slot: term.slot,
            coefficient: one_minus.mul(&term.coefficient),
        });
    }
    for term in half1.masks {
        masks.push(MaskTerm {
            slot: term.slot,
            coefficient: rho.mul(&term.coefficient),
        });
    }
    Ok(GeneralClaim {
        point,
        value: one_minus.mul(&half0.value).add(&rho.mul(&half1.value)),
        masks,
    })
}

/// The prover's messages of one general sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneralProof {
    pub hiding_sum: Fp3,
    pub rounds: Vec<PscRound>,
    pub y_round: PscRound,
    pub end_values: Vec<Fp3>,
}

impl GeneralProof {
    /// Writes the proof: the hiding sum, the rounds, the last round and the
    /// end values, each with its count.
    pub fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_fp3(&self.hiding_sum);
        writer.put_u64(self.rounds.len() as u64);
        for round in &self.rounds {
            writer.put_u64(round.coefficients.len() as u64);
            for coefficient in &round.coefficients {
                writer.put_fp3(coefficient);
            }
        }
        writer.put_u64(self.y_round.coefficients.len() as u64);
        for coefficient in &self.y_round.coefficients {
            writer.put_fp3(coefficient);
        }
        writer.put_u64(self.end_values.len() as u64);
        for value in &self.end_values {
            writer.put_fp3(value);
        }
    }

    /// Reads a proof. `rounds` is the number of X rounds the claim expects and
    /// `end_values` the number of shown maps; both stored counts must match.
    pub fn decode(
        reader: &mut CanonicalReader,
        rounds: usize,
        end_values: usize,
    ) -> Result<Self> {
        let hiding_sum = reader.get_fp3()?;
        if usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)? != rounds {
            return Err(Error::InvalidEncoding);
        }
        let mut round_values = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            if count > 8 {
                return Err(Error::InvalidEncoding);
            }
            let mut coefficients = Vec::with_capacity(count);
            for _ in 0..count {
                coefficients.push(reader.get_fp3()?);
            }
            round_values.push(PscRound { coefficients });
        }
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count > 8 {
            return Err(Error::InvalidEncoding);
        }
        let mut y_round = Vec::with_capacity(count);
        for _ in 0..count {
            y_round.push(reader.get_fp3()?);
        }
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count != end_values {
            return Err(Error::InvalidEncoding);
        }
        let mut ends = Vec::with_capacity(count);
        for _ in 0..count {
            ends.push(reader.get_fp3()?);
        }
        Ok(Self {
            hiding_sum,
            rounds: round_values,
            y_round: PscRound {
                coefficients: y_round,
            },
            end_values: ends,
        })
    }
}

/// The output claims of one general sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneralOutput {
    /// The X challenge point.
    pub point: Vec<Fp3>,
    /// Coefficients on the hiding mask entries, in hiding-entry order.
    pub hiding_form: Vec<Fp3>,
    /// The product of the weights' `psi` values at the challenges.
    pub claim_scale: Fp3,
    /// The `(U)` row target.
    pub row_target: Fp3,
    /// One masked point claim per shown map.
    pub ends: Vec<PointClaim>,
    /// The kept-map terms of the `(U)` row: map index and coefficient.
    pub row_maps: Vec<(usize, Fp3)>,
    /// The claim's mask terms, scaled by the output claim scale.
    pub claim_masks: Vec<MaskTerm>,
}

/// The variables of the general sumcheck: the claim's X variables and, when a
/// map is shown, the last variable summed at zero with degree `degree + 1`.
pub fn general_variables(x_variables: &[Weight], summand: &Summand) -> Result<Vec<Weight>> {
    let mut variables = x_variables.to_vec();
    if summand.shown() > 0 {
        variables.push(Weight::at_zero(summand.degree() + 1)?);
    }
    if variables.is_empty() {
        return Err(Error::InvalidConfig);
    }
    Ok(variables)
}

pub fn general_masks(summand: &Summand, x_variables: &[Weight]) -> Result<usize> {
    Ok(hp_entries(&general_variables(x_variables, summand)?)? + summand.shown())
}

fn poly_at(coefficients: &[Fp3], x: Fp3) -> Fp3 {
    let mut value = Fp3::ZERO;
    for coefficient in coefficients.iter().rev() {
        value = value.mul(&x).add(coefficient);
    }
    value
}

fn interpolate(values: &[Fp3]) -> Result<Vec<Fp3>> {
    if values.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let count = values.len();
    let mut divided = values.to_vec();
    for order in 1..count {
        let step = Fp3::from_u64(order as u64).inverse()?;
        for index in (order..count).rev() {
            divided[index] = divided[index].sub(&divided[index - 1]).mul(&step);
        }
    }
    let mut polynomial = vec![Fp3::ZERO; count];
    for index in (0..count).rev() {
        let node = Fp3::from_u64(index as u64);
        for degree in (1..count).rev() {
            polynomial[degree] = polynomial[degree - 1].sub(&node.mul(&polynomial[degree]));
        }
        polynomial[0] = divided[index].sub(&node.mul(&polynomial[0]));
    }
    Ok(polynomial)
}

fn add_scaled(target: &mut [Fp3], source: &[Fp3], scale: Fp3) -> Result<()> {
    if source.len() > target.len() {
        if source[target.len()..].iter().any(|value| *value != Fp3::ZERO) {
            return Err(Error::ShapeMismatch);
        }
    }
    for (slot, value) in target.iter_mut().zip(source) {
        *slot = slot.add(&scale.mul(value));
    }
    Ok(())
}

fn add_weight(target: &mut [Fp3], weight: &Weight, scale: Fp3) -> Result<()> {
    if weight.psi().len() > target.len() {
        return Err(Error::ShapeMismatch);
    }
    for (index, coefficient) in weight.psi().iter().enumerate() {
        target[index] = target[index].add(&scale.mul(coefficient));
    }
    Ok(())
}

fn sent_coefficients(weight: &Weight, round: &[Fp3]) -> Result<Vec<Fp3>> {
    if round.len() != weight.degree() {
        return Err(Error::ShapeMismatch);
    }
    let mut sent = Vec::with_capacity(weight.degree() - 1);
    for (index, coefficient) in round.iter().enumerate() {
        if index != weight.fixed() {
            sent.push(*coefficient);
        }
    }
    Ok(sent)
}

fn complete_round(weight: &Weight, message: &PscRound, claim: Fp3) -> Result<Vec<Fp3>> {
    if message.coefficients.len() + 1 != weight.degree() {
        return Err(Error::InvalidProof);
    }
    let mut round = vec![Fp3::ZERO; weight.degree()];
    let mut sent = message.coefficients.iter();
    for index in 0..weight.degree() {
        if index != weight.fixed() {
            round[index] = *sent.next().ok_or(Error::InvalidProof)?;
        }
    }
    weight.complete(&mut round, claim)?;
    Ok(round)
}

fn output(
    variables: &[Weight],
    summand: &Summand,
    table: Option<&dyn TableAt>,
    eps: Fp3,
    challenges: &[Fp3],
    last: Fp3,
    end_values: &[Fp3],
    claim_masks: &[MaskTerm],
) -> Result<GeneralOutput> {
    let x_count = variables.len() - usize::from(summand.shown() > 0);
    let point = challenges[..x_count].to_vec();
    let form = hp_form(variables, challenges)?;
    let mut psis = eps;
    for (variable, challenge) in variables.iter().zip(challenges) {
        psis = psis.mul(&poly_at(variable.psi(), *challenge));
    }
    let shown = &end_values[..summand.shown()];
    let mut kept = vec![Fp3::ZERO; summand.kept()];
    if end_values.len() == summand.shown() + summand.kept() {
        kept.copy_from_slice(&end_values[summand.shown()..]);
    } else if end_values.len() != summand.shown() {
        return Err(Error::ShapeMismatch);
    }
    let coefficients = summand.at(&point, shown, &kept, table)?;
    if coefficients.len() != 1 + summand.kept() {
        return Err(Error::ShapeMismatch);
    }
    let beta = *challenges.last().ok_or(Error::ShapeMismatch)?;
    let mut ends = Vec::with_capacity(summand.shown());
    for value in shown {
        ends.push(PointClaim {
            point: point.clone(),
            mask_scale: beta,
            value: *value,
        });
    }
    let mut row_maps = Vec::with_capacity(summand.kept());
    for index in 0..summand.kept() {
        row_maps.push((index, eps.mul(&coefficients[1 + index])));
    }
    Ok(GeneralOutput {
        point,
        hiding_form: form,
        claim_scale: psis,
        row_target: last.sub(&eps.mul(&coefficients[0])),
        ends,
        row_maps,
        claim_masks: claim_masks
            .iter()
            .map(|term| MaskTerm {
                slot: term.slot,
                coefficient: psis.mul(&term.coefficient),
            })
            .collect(),
    })
}

/// Proves one general masked sumcheck. The claim has no mask form; its target is
/// the masked summand's sum. `inner` supplies the unmasked round polynomials.
pub fn prove_general(
    x_variables: &[Weight],
    claim_masks: &[MaskTerm],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
    target: Fp3,
    summand: &Summand,
    table: Option<&dyn TableAt>,
    inner: &mut dyn InnerSumcheck,
    masks: &GeneralMaskEntries,
    challenges: &mut dyn GeneralChallenges,
) -> Result<(GeneralProof, GeneralOutput)> {
    let variables = general_variables(x_variables, summand)?;
    if masks.hiding.len() != hp_entries(&variables)? || masks.shown.len() != summand.shown() {
        return Err(Error::ShapeMismatch);
    }
    let x_count = x_variables.len();
    let mut hiding = HidingPolynomial::new(&variables, &masks.hiding)?;
    let mut proof = GeneralProof {
        hiding_sum: hiding.sum(),
        rounds: Vec::with_capacity(x_count),
        y_round: PscRound {
            coefficients: Vec::new(),
        },
        end_values: Vec::new(),
    };
    let mut sigma = Fp3::ZERO;
    for term in claim_masks {
        sigma = sigma.add(&term.coefficient.mul(&mask_value(term.slot)?));
    }
    challenges.send_hiding_sum(proof.hiding_sum)?;
    let eps = challenges.draw()?;
    let mut running = proof.hiding_sum.add(&eps.mul(&target));
    let mut psis = Fp3::ONE;
    let mut drawn = Vec::with_capacity(variables.len());
    for variable in 0..x_count {
        let mut round = hiding.round()?;
        add_scaled(&mut round, &inner.round()?, eps)?;
        add_weight(&mut round, &variables[variable], eps.mul(&sigma).mul(&psis))?;
        if variables[variable].apply(&round)? != running {
            return Err(Error::InvalidProof);
        }
        proof.rounds.push(PscRound {
            coefficients: sent_coefficients(&variables[variable], &round)?,
        });
        challenges.send_round(
            variable,
            proof.rounds.last().ok_or(Error::InvalidProof)?,
        )?;
        let challenge = challenges.draw()?;
        drawn.push(challenge);
        running = poly_at(&round, challenge);
        hiding.bind(challenge)?;
        psis = psis.mul(&poly_at(variables[variable].psi(), challenge));
        inner.bind(challenge)?;
    }
    let map_values = inner.end_values()?;
    if map_values.len() != summand.shown() + summand.kept() {
        return Err(Error::ShapeMismatch);
    }
    if summand.shown() > 0 {
        let kept = &map_values[summand.shown()..];
        let mut values = Vec::with_capacity(summand.degree() + 1);
        for node in 0..=summand.degree() {
            let point = Fp3::from_u64(node as u64);
            let shown: Vec<Fp3> = map_values[..summand.shown()]
                .iter()
                .enumerate()
                .map(|(index, value)| value.add(&masks.shown[index].mul(&point)))
                .collect();
            let coefficients = summand.at(&drawn, &shown, kept, table)?;
            let mut value = coefficients[0];
            for (index, kept_value) in kept.iter().enumerate() {
                value = value.add(&coefficients[1 + index].mul(kept_value));
            }
            values.push(value);
        }
        let summand_round = interpolate(&values)?;
        let mut round = hiding.round()?;
        add_scaled(&mut round, &summand_round, eps)?;
        add_weight(
            &mut round,
            &variables[x_count],
            eps.mul(&sigma).mul(&psis),
        )?;
        if variables[x_count].apply(&round)? != running {
            return Err(Error::InvalidProof);
        }
        proof.y_round = PscRound {
            coefficients: sent_coefficients(&variables[x_count], &round)?,
        };
        challenges.send_round(x_count, &proof.y_round)?;
        let beta = challenges.draw_nonzero()?;
        drawn.push(beta);
        running = poly_at(&round, beta);
        hiding.bind(beta)?;
        for index in 0..summand.shown() {
            proof.end_values.push(map_values[index].add(&beta.mul(&masks.shown[index])));
        }
        challenges.send_end_values(&proof.end_values)?;
    }
    let mut all_values = proof.end_values.clone();
    all_values.extend_from_slice(&map_values[summand.shown()..]);
    let out = output(
        &variables,
        summand,
        table,
        eps,
        &drawn,
        running,
        &all_values,
        claim_masks,
    )?;
    Ok((proof, out))
}

/// Verifies one general masked sumcheck's messages and returns its output
/// claims. The shown end values are read from the proof; the caller binds every
/// output claim to committed messages.
pub fn verify_general(
    x_variables: &[Weight],
    claim_masks: &[MaskTerm],
    target: Fp3,
    summand: &Summand,
    table: Option<&dyn TableAt>,
    proof: &GeneralProof,
    hiding_entries: usize,
    challenges: &mut dyn GeneralChallenges,
) -> Result<GeneralOutput> {
    let variables = general_variables(x_variables, summand)?;
    if hiding_entries != hp_entries(&variables)? {
        return Err(Error::ShapeMismatch);
    }
    if proof.rounds.len() != x_variables.len()
        || proof.end_values.len() != summand.shown()
    {
        return Err(Error::InvalidProof);
    }
    challenges.send_hiding_sum(proof.hiding_sum)?;
    let eps = challenges.draw()?;
    let mut running = proof.hiding_sum.add(&eps.mul(&target));
    let mut drawn = Vec::with_capacity(variables.len());
    for variable in 0..x_variables.len() {
        let round = complete_round(&variables[variable], &proof.rounds[variable], running)?;
        challenges.send_round(variable, &proof.rounds[variable])?;
        let challenge = challenges.draw()?;
        drawn.push(challenge);
        running = poly_at(&round, challenge);
    }
    if summand.shown() > 0 {
        let round = complete_round(&variables[x_variables.len()], &proof.y_round, running)?;
        challenges.send_round(x_variables.len(), &proof.y_round)?;
        let beta = challenges.draw_nonzero()?;
        drawn.push(beta);
        running = poly_at(&round, beta);
    }
    if summand.shown() > 0 {
        challenges.send_end_values(&proof.end_values)?;
    }
    output(
        &variables,
        summand,
        table,
        eps,
        &drawn,
        running,
        &proof.end_values,
        claim_masks,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn f(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    fn eq_factor(z: Fp3, bit: bool) -> Fp3 {
        if bit {
            z
        } else {
            Fp3::ONE.sub(&z)
        }
    }

    #[test]
    fn product_with_equality_weights_proves_and_verifies() {
        let z = [f(2), f(3)];
        let x = [f(1), f(2), f(3), f(4)];
        let omega = [f(5), f(6), f(7), f(8)];
        let mut target = Fp3::ZERO;
        for index in 0..4 {
            let factor = eq_factor(z[0], index & 1 == 1)
                .mul(&eq_factor(z[1], index & 2 == 2));
            target = target.add(&factor.mul(&x[index]).mul(&omega[index]));
        }
        let variables = [
            Weight::equality(z[0], 3).unwrap(),
            Weight::equality(z[1], 3).unwrap(),
        ];
        let count = general_masks(&Summand::Product, &variables).unwrap();
        let hiding = (0..hp_entries(&general_variables(&variables, &Summand::Product).unwrap()).unwrap())
            .map(|index| f(100 + index as u64))
            .collect::<Vec<_>>();
        let masks = GeneralMaskEntries {
            hiding,
            shown: vec![f(200), f(201)],
        };
        assert_eq!(masks.hiding.len() + masks.shown.len(), count);
        let mut inner = WeightedProductInner::new(&x, &omega, &z).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &Summand::Product,
            None,
            &mut inner,
            &masks,
            &mut Source(7),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &[],
            target,
            &Summand::Product,
            None,
            &proof,
            masks.hiding.len(),
            &mut Source(7),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.ends.len(), 2);
    }

    #[test]
    fn a_single_map_proves_its_cube_sum() {
        let variables = [Weight::plain(2).unwrap(), Weight::plain(2).unwrap()];
        let values = [f(1), f(2), f(3), f(4)];
        let target = values.iter().fold(Fp3::ZERO, |sum, value| sum.add(value));
        let count = general_masks(&Summand::Map, &variables).unwrap();
        let hiding = (0..hp_entries(&general_variables(&variables, &Summand::Map).unwrap()).unwrap())
            .map(|index| f(300 + index as u64))
            .collect::<Vec<_>>();
        let masks = GeneralMaskEntries {
            hiding,
            shown: vec![f(400)],
        };
        assert_eq!(masks.hiding.len() + masks.shown.len(), count);
        let mut inner = MapInner::new(&values).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &Summand::Map,
            None,
            &mut inner,
            &masks,
            &mut Source(13),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &[],
            target,
            &Summand::Map,
            None,
            &proof,
            masks.hiding.len(),
            &mut Source(13),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.ends.len(), 1);
        // A changed round is rejected.
        // A changed round leaves another output claim; the caller binds the
        // claim to committed messages, so the check rejects it there.
        let mut changed = proof.clone();
        changed.rounds[0].coefficients[0] =
            changed.rounds[0].coefficients[0].add(&Fp3::ONE);
        let tampered = verify_general(
            &variables,
            &[],
            target,
            &Summand::Map,
            None,
            &changed,
            masks.hiding.len(),
            &mut Source(13),
        )
        .unwrap();
        assert_ne!(tampered, verified);
    }

    #[test]
    fn the_product_sum_form_combines_shown_maps() {
        let summand = Summand::ProductSum {
            terms: vec![(f(2), vec![0, 1]), (f(3), vec![1])],
        };
        assert_eq!(summand.shown(), 2);
        assert_eq!(summand.degree(), 2);
        let shown = [f(5), f(7)];
        assert_eq!(
            summand.at(&[], &shown, &[], None).unwrap(),
            vec![f(2).mul(&f(5)).mul(&f(7)).add(&f(3).mul(&f(7)))]
        );
        // An index outside the shown maps is refused.
        let bad = Summand::ProductSum {
            terms: vec![(f(1), vec![2])],
        };
        assert!(bad.at(&[], &shown, &[], None).is_err());
    }

    #[test]
    fn bits_proves_booleanity_and_rejects_a_non_boolean_map() {
        let variables = [Weight::plain(3).unwrap(), Weight::plain(3).unwrap()];
        let values = [f(0), f(1), f(1), f(0)];
        let masks = GeneralMaskEntries {
            hiding: (0..7).map(|index| f(10 + index as u64)).collect(),
            shown: vec![f(90)],
        };
        let mut inner = BitsInner::new(&values).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            Fp3::ZERO,
            &Summand::Bits,
            None,
            &mut inner,
            &masks,
            &mut Source(11),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &[],
            Fp3::ZERO,
            &Summand::Bits,
            None,
            &proof,
            masks.hiding.len(),
            &mut Source(11),
        )
        .unwrap();
        assert_eq!(proven, verified);

        let bad = [f(2), f(0), f(0), f(0)];
        let mut inner = BitsInner::new(&bad).unwrap();
        assert!(prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            Fp3::ZERO,
            &Summand::Bits,
            None,
            &mut inner,
            &masks,
            &mut Source(11),
        )
        .is_err());
    }

    #[test]
    fn claim_mask_terms_enter_the_output() {
        let variables = [Weight::plain(3).unwrap(), Weight::plain(3).unwrap()];
        let values = [f(0), f(1), f(1), f(0)];
        let masks = GeneralMaskEntries {
            hiding: (0..7).map(|index| f(60 + index as u64)).collect(),
            shown: vec![f(600)],
        };
        let claim_masks = [MaskTerm {
            slot: 3,
            coefficient: f(5),
        }];
        let provider = |slot: usize| -> Result<Fp3> {
            if slot == 3 {
                Ok(f(7))
            } else {
                Err(Error::InvalidIndex)
            }
        };
        let mut inner = BitsInner::new(&values).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &claim_masks,
            &provider,
            f(35),
            &Summand::Bits,
            None,
            &mut inner,
            &masks,
            &mut Source(29),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &claim_masks,
            f(35),
            &Summand::Bits,
            None,
            &proof,
            masks.hiding.len(),
            &mut Source(29),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.claim_masks.len(), 1);
        assert_eq!(proven.claim_masks[0].slot, 3);
    }

    #[test]
    fn line_combines_two_halves() {
        let half0 = GeneralClaim {
            point: vec![f(2)],
            value: f(10),
            masks: vec![MaskTerm {
                slot: 1,
                coefficient: f(3),
            }],
        };
        let half1 = GeneralClaim {
            point: vec![f(2)],
            value: f(20),
            masks: vec![MaskTerm {
                slot: 2,
                coefficient: f(4),
            }],
        };
        let combined = line(&mut Source(31), half0, half1).unwrap();
        assert_eq!(combined.point.len(), 2);
        assert_eq!(combined.point[0], f(2));
        let rho = combined.point[1];
        assert_eq!(
            combined.value,
            Fp3::ONE.sub(&rho).mul(&f(10)).add(&rho.mul(&f(20)))
        );
        assert_eq!(combined.masks.len(), 2);
        assert_eq!(combined.masks[0].slot, 1);
        assert_eq!(combined.masks[1].slot, 2);
    }

    #[test]
    fn tie_combines_scalars_without_x_variables() {
        let variables: [Weight; 0] = [];
        let masks = GeneralMaskEntries {
            hiding: (0..hp_entries(&[Weight::at_zero(4).unwrap()]).unwrap())
                .map(|index| f(30 + index as u64))
                .collect(),
            shown: vec![f(301), f(302), f(303)],
        };
        let mut inner = ScalarInner::new(&[f(2), f(1), f(1)]).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            Fp3::ZERO,
            &Summand::TieFirst { a: f(2) },
            None,
            &mut inner,
            &masks,
            &mut Source(13),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &[],
            Fp3::ZERO,
            &Summand::TieFirst { a: f(2) },
            None,
            &proof,
            masks.hiding.len(),
            &mut Source(13),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.ends.len(), 3);
    }

    #[test]
    fn weighted_bits_prove_booleanity() {
        let z = [f(2), f(3)];
        let values = [f(0), f(1), f(1), f(0)];
        let variables = [
            Weight::equality(z[0], 3).unwrap(),
            Weight::equality(z[1], 3).unwrap(),
        ];
        let masks = GeneralMaskEntries {
            hiding: (0..7).map(|index| f(50 + index as u64)).collect(),
            shown: vec![f(500)],
        };
        let mut inner = WeightedBitsInner::new(&values, &z).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            Fp3::ZERO,
            &Summand::Bits,
            None,
            &mut inner,
            &masks,
            &mut Source(19),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &[],
            Fp3::ZERO,
            &Summand::Bits,
            None,
            &proof,
            masks.hiding.len(),
            &mut Source(19),
        )
        .unwrap();
        assert_eq!(proven, verified);
    }

    #[test]
    fn leaves_inner_matches_the_public_table() {
        struct PointTable;
        impl TableAt for PointTable {
            fn at(&self, point: &[Fp3]) -> Result<Fp3> {
                if point.len() != 1 {
                    return Err(Error::ShapeMismatch);
                }
                Ok(Fp3::ONE.sub(&point[0]).mul(&f(3)).add(&point[0].mul(&f(5))))
            }
        }
        let z = [f(2)];
        let table = [f(3), f(5)];
        let values = [f(1), f(0)];
        let alpha = f(9);
        let leaves = Summand::Leaves { alpha, mu: 0 };
        let variables = [Weight::equality(z[0], 3).unwrap()];
        let mut target = Fp3::ZERO;
        for index in 0..2 {
            let weight = if index == 1 { z[0] } else { Fp3::ONE.sub(&z[0]) };
            let coefficient = alpha.sub(&Fp3::ONE).sub(&table[index]);
            target = target.add(&weight.mul(&Fp3::ONE.add(&values[index].mul(&coefficient))));
        }
        let masks = GeneralMaskEntries {
            hiding: (0..hp_entries(&general_variables(&variables, &leaves).unwrap()).unwrap())
                .map(|index| f(70 + index as u64))
                .collect(),
            shown: Vec::new(),
        };
        let table_impl = PointTable;
        let mut inner = LeavesInner::new(&values, &table, &z, alpha).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &leaves,
            Some(&table_impl),
            &mut inner,
            &masks,
            &mut Source(23),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &[],
            target,
            &leaves,
            Some(&table_impl),
            &proof,
            masks.hiding.len(),
            &mut Source(23),
        )
        .unwrap();
        assert_eq!(proven, verified);
    }

    #[test]
    fn leaf_summand_reads_a_public_table() {
        struct TwoPointTable;
        impl TableAt for TwoPointTable {
            fn at(&self, point: &[Fp3]) -> Result<Fp3> {
                if point.len() != 1 {
                    return Err(Error::ShapeMismatch);
                }
                Ok(Fp3::ONE.sub(&point[0]).mul(&f(3)).add(&point[0].mul(&f(5))))
            }
        }
        let table = TwoPointTable;
        let coefficients = Summand::Leaves { alpha: f(9), mu: 0 }
            .at(&[f(1)], &[], &[Fp3::ZERO], Some(&table))
            .unwrap();
        assert_eq!(coefficients.len(), 2);
        assert_eq!(coefficients[0], Fp3::ONE);
        assert_eq!(coefficients[1], f(9).sub(&Fp3::ONE).sub(&f(5)));
        assert!(Summand::Leaves { alpha: f(9), mu: 0 }
            .at(&[f(1)], &[], &[Fp3::ZERO], None)
            .is_err());
    }

    use zkie_core::common::codec::DecodeLimits;

    fn aggregate(base: &[Fp3], maps: &[Vec<Fp3>], coefficients: &[Fp3]) -> Vec<Fp3> {
        let mut values = base.to_vec();
        for (coefficient, map) in coefficients.iter().zip(maps) {
            for (slot, value) in values.iter_mut().zip(map) {
                *slot = slot.add(&coefficient.mul(value));
            }
        }
        values
    }

    #[test]
    fn the_panel_product_weights_the_group_rows() {
        let summand = Summand::PanelProduct {
            lam: 3,
            panels: vec![PanelEnd {
                off: 4,
                w: 2,
                ni: 2,
                ny: 2,
            }],
        };
        assert_eq!(summand.shown(), 4);
        assert_eq!(summand.degree(), 2);
        let point = [f(3), f(4), f(5)];
        // sb: the panel sits at offset four, so its offset bits are one and
        // pair with the point's last coordinate.
        let sb = f(5);
        let ey = [Fp3::ONE.sub(&f(4)), f(4)];
        let shown = [f(2), f(7), f(11), f(13)];
        let expected = f(2).mul(
            &f(7)
                .add(&sb.mul(&ey[0]).mul(&f(11)))
                .add(&sb.mul(&ey[1]).mul(&f(13))),
        );
        assert_eq!(
            summand
                .at(&point, &shown, &[], None)
                .unwrap(),
            vec![expected]
        );
        // The point's width, the group's size and the panel's geometry are
        // checked.
        assert!(summand.at(&point[..2], &shown, &[], None).is_err());
        assert!(summand.at(&point, &shown[..3], &[], None).is_err());
        let bad = Summand::PanelProduct {
            lam: 3,
            panels: vec![PanelEnd {
                off: 0,
                w: 0,
                ni: 1,
                ny: 2,
            }],
        };
        assert!(bad.at(&point, &[f(1), f(2), f(3)], &[], None).is_err());
    }

    #[test]
    fn the_stack_end_group_proves_and_verifies() {
        let a0 = [f(1), f(2), f(3), f(4)];
        let u = [vec![f(5), f(6), f(7), f(8)]];
        let b0 = [f(2), f(3), f(4), f(5)];
        let v = [
            vec![f(1), f(1), f(1), f(1)],
            vec![f(0), f(1), f(0), f(1)],
        ];
        let ku = vec![f(3)];
        let kv = vec![f(5), f(7)];
        let summand = Summand::StackProduct {
            ku: ku.clone(),
            kv: kv.clone(),
        };
        assert_eq!(summand.shown(), 5);
        assert_eq!(summand.degree(), 2);
        // The claim is the sum of the aggregated product over the cube.
        let left = aggregate(&a0, &u, &ku);
        let right = aggregate(&b0, &v, &kv);
        let mut target = Fp3::ZERO;
        for (left, right) in left.iter().zip(&right) {
            target = target.add(&left.mul(right));
        }
        let variables = [
            crate::proof::psc::Weight::plain(3).unwrap(),
            crate::proof::psc::Weight::plain(3).unwrap(),
        ];
        let hiding_count = crate::proof::psc::hp_entries(
            &general_variables(&variables, &summand).unwrap(),
        )
        .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| f(700 + index as u64)).collect(),
            shown: (0..5).map(|index| f(800 + index as u64)).collect(),
        };
        let mut inner = StackInner::new(&a0, &u, &b0, &v, ku.clone(), kv.clone()).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &summand,
            None,
            &mut inner,
            &masks,
            &mut Source(67),
        )
        .unwrap();
        let verified = verify_general(
            &variables,
            &[],
            target,
            &summand,
            None,
            &proof,
            masks.hiding.len(),
            &mut Source(67),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.ends.len(), 5);
        // A changed last-round coefficient leaves another output claim.
        let mut changed = proof.clone();
        changed.y_round.coefficients[0] = changed.y_round.coefficients[0].add(&Fp3::ONE);
        let tampered = verify_general(
            &variables,
            &[],
            target,
            &summand,
            None,
            &changed,
            masks.hiding.len(),
            &mut Source(67),
        )
        .unwrap();
        assert_ne!(tampered, verified);
        // The end group's size is fixed by the layout.
        let mut short = masks.clone();
        short.shown.pop();
        assert!(prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &summand,
            None,
            &mut StackInner::new(&a0, &u, &b0, &v, ku.clone(), kv.clone()).unwrap(),
            &short,
            &mut Source(67),
        )
        .is_err());
        // A changed v map does not sum to the claim.
        let mut changed_v = v.clone();
        changed_v[1][1] = changed_v[1][1].add(&Fp3::ONE);
        assert!(prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &summand,
            None,
            &mut StackInner::new(&a0, &u, &b0, &changed_v, ku, kv).unwrap(),
            &masks,
            &mut Source(67),
        )
        .is_err());
    }

    #[test]
    fn the_general_proof_round_trips_through_the_canonical_encoding() {
        let proof = GeneralProof {
            hiding_sum: f(1),
            rounds: vec![
                PscRound {
                    coefficients: vec![f(2), f(3)],
                },
                PscRound {
                    coefficients: vec![f(4)],
                },
            ],
            y_round: PscRound {
                coefficients: vec![f(5)],
            },
            end_values: vec![f(6), f(7)],
        };
        let mut writer = CanonicalWriter::new();
        proof.encode_into(&mut writer);
        let bytes = writer.into_bytes();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert_eq!(GeneralProof::decode(&mut reader, 2, 2).unwrap(), proof);
        // Another round or end-value count does not decode.
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(GeneralProof::decode(&mut reader, 1, 2).is_err());
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(GeneralProof::decode(&mut reader, 2, 1).is_err());
        // A round with too many coefficients does not decode.
        let wide = GeneralProof {
            hiding_sum: f(1),
            rounds: vec![PscRound {
                coefficients: vec![f(0); 9],
            }],
            y_round: PscRound {
                coefficients: vec![f(0)],
            },
            end_values: vec![f(0); 2],
        };
        let mut writer = CanonicalWriter::new();
        wide.encode_into(&mut writer);
        let bytes = writer.into_bytes();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(GeneralProof::decode(&mut reader, 1, 2).is_err());
    }
}
