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

use crate::proof::psc::{
    hp_entries, hp_form, HidingPolynomial, PointClaim, ProductSumcheckProver, PscRound, Weight,
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
}

impl Summand {
    pub fn shown(&self) -> usize {
        match self {
            Self::Product => 2,
            Self::TieFirst { .. } => 3,
            Self::TieNext => 2,
            Self::Bits => 1,
            Self::Leaves { .. } => 0,
        }
    }

    pub fn kept(&self) -> usize {
        match self {
            Self::Leaves { .. } => 1,
            _ => 0,
        }
    }

    pub fn degree(&self) -> usize {
        match self {
            Self::Product => 2,
            Self::TieFirst { .. } | Self::TieNext => 3,
            Self::Bits => 2,
            Self::Leaves { .. } => 0,
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
            values.push(summand.at(&drawn, &shown, kept, table)?[0]);
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
}
