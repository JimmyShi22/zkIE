//! Masked product sumcheck for the hidden dot relation.
//!
//! The relation is the plain Boolean-cube sum
//!
//! ```text
//! sum_{u in {0,1}^k} X(u) Omega(u) = <x, omega>
//! ```
//!
//! of the multilinear extensions of two vectors of length `2^k`. Each
//! variable is summed by `phi(g) = g(0) + g(1)`, so the cube sum of the two
//! extensions is exactly the inner product of the vectors. There is no
//! `2^-k` normalization: `phi(1) = 2`, and a cube sum is not an average.
//!
//! The sumcheck runs on `H = U + eps (S + T)`, where:
//!
//! * `U` is the hiding polynomial
//!
//!   ```text
//!   U(X_1, .., X_k) = sum_l w_l(X_l) prod_{j>l} psi_j(X_j)
//!   ```
//!
//!   with `w_l` of degree below variable `l`'s degree bound and random
//!   coefficients, and `psi_j` the polynomial of the variable's weight with
//!   `phi_j(psi_j) = 1`. A constant term of `w_l` for `l < k` is redundant and
//!   fixed to zero, so the coefficient count is `1 + sum_j (d_j - 1)`.
//! * `S(y_X, y_Omega) = y_X y_Omega` is the product summand, affine in the two
//!   shown map values and in no kept map.
//! * A last variable `Y`, summed by `g -> g(0)`, turns each shown value
//!   `F(gamma)` into the masked value `F(gamma) + beta c` that the prover
//!   sends; the end-value masks `c` are used once each and `beta != 0`.
//! * `T = sigma prod_t psi_t(X_t)` masks the claim's mask form
//!   `sigma = sum_t coeff_t value_t`; it sums to `sigma` because
//!   `phi_t(psi_t) = 1`.
//!
//! The prover sends the hiding sum `mu'` before the verifier draws `eps`; the
//! round polynomials after `eps` are uniform among those that pass the checks.
//! The final output is a linear claim on the hiding coefficients, a linear
//! claim on the claim masks, and one point claim for each shown map.
//!
//! This module is arithmetic only. It proves and verifies over plaintext
//! vectors and plaintext mask entries: the vectors stand in for the committed
//! oracles and the mask entries stand in for the committed mask message whose
//! linear claims a production caller checks. Commitments, samples, rows and
//! the request wiring are not part of this module. The challenge source is
//! also a stand-in: a production source must absorb every message sent so far
//! into the transcript before it yields a challenge.

use std::fmt;

use serde::{Deserialize, Serialize};
use zkie_core::common::field::XorShift64;
use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

/// Number of coefficients of one plain sumcheck variable (degree below 3).
pub const PLAIN_DEGREE: usize = 3;
/// Degree of the product summand in the last variable.
pub const PRODUCT_DEGREE: usize = 2;
/// Number of shown maps of the product summand.
pub const SHOWN_MAPS: usize = 2;
/// Number of variables of the hidden dot sumcheck for 1024-entry rows.
pub const DOT_VARIABLES: usize = 10;

// ---------------------------------------------------------------------------
// One variable of the sumcheck.

/// How one variable is summed: a linear functional `phi` on polynomials with
/// `degree` coefficients, and a polynomial `psi` with `phi(psi) = 1`.
///
/// `moment[t]` is `phi(X^t)`. The verifier's check `phi(q) = claim` fixes one
/// coefficient of a round polynomial; `fixed` is its index and `fixed_inv` is
/// `1 / moment[fixed]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Weight {
    moment: Vec<Fp3>,
    psi: Vec<Fp3>,
    fixed: usize,
    fixed_inv: Fp3,
}

impl Weight {
    /// `phi(g) = g(0) + g(1)`, a plain sum over the two Boolean points, with
    /// `psi = X`. Requires at least two coefficients.
    pub fn plain(degree: usize) -> Result<Self> {
        if degree < 2 {
            return Err(Error::InvalidConfig);
        }
        let mut moment = vec![Fp3::ONE; degree];
        moment[0] = Fp3::ONE.add(&Fp3::ONE);
        let fixed = 1;
        let fixed_inv = moment[fixed].inverse()?;
        Ok(Self {
            moment,
            psi: vec![Fp3::ZERO, Fp3::ONE],
            fixed,
            fixed_inv,
        })
    }

    /// `phi(g) = (1 - z) g(0) + z g(1)`: a sum over the two Boolean points
    /// weighted by the equality polynomial's factor at `z`, with `psi = 1`.
    pub fn equality(z: Fp3, degree: usize) -> Result<Self> {
        if degree < 2 {
            return Err(Error::InvalidConfig);
        }
        let mut moment = vec![z; degree];
        moment[0] = Fp3::ONE;
        let fixed = 0;
        let fixed_inv = moment[fixed].inverse()?;
        Ok(Self {
            moment,
            psi: vec![Fp3::ONE],
            fixed,
            fixed_inv,
        })
    }

    /// `phi(g) = g(0)`, used for a further variable that does not change the
    /// sum, with `psi = 1`.
    pub fn at_zero(degree: usize) -> Result<Self> {
        Self::equality(Fp3::ZERO, degree)
    }

    /// Number of coefficients of a polynomial in this variable.
    pub fn degree(&self) -> usize {
        self.moment.len()
    }

    /// The index of the coefficient fixed by the round check.
    pub fn fixed(&self) -> usize {
        self.fixed
    }

    /// The coefficients of `psi`, lowest degree first.
    pub fn psi(&self) -> &[Fp3] {
        &self.psi
    }

    /// `phi(g)`. A polynomial with fewer than `degree` coefficients is
    /// accepted; more are rejected.
    pub fn apply(&self, g: &[Fp3]) -> Result<Fp3> {
        if g.len() > self.moment.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut value = Fp3::ZERO;
        for (index, coefficient) in g.iter().enumerate() {
            value = value.add(&self.moment[index].mul(coefficient));
        }
        Ok(value)
    }

    /// `phi(1)`.
    pub fn of_one(&self) -> Fp3 {
        self.moment[0]
    }

    /// Sets `g[fixed]` to the one coefficient for which `phi(g) = claim`; the
    /// old value of `g[fixed]` is ignored. `g` must have exactly `degree`
    /// coefficients.
    pub fn complete(&self, g: &mut [Fp3], claim: Fp3) -> Result<()> {
        if g.len() != self.moment.len() || self.fixed >= g.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut rest = Fp3::ZERO;
        for index in 0..g.len() {
            if index != self.fixed {
                rest = rest.add(&self.moment[index].mul(&g[index]));
            }
        }
        g[self.fixed] = claim.sub(&rest).mul(&self.fixed_inv);
        Ok(())
    }
}

/// The variables of the hidden dot sumcheck: `k` plain variables of degree
/// below [`PLAIN_DEGREE`] and one `at_zero` variable of degree below
/// `PRODUCT_DEGREE + 1` for `Y`.
pub fn psc_variables(k: usize) -> Result<Vec<Weight>> {
    if k == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut variables = Vec::with_capacity(k + 1);
    for _ in 0..k {
        variables.push(Weight::plain(PLAIN_DEGREE)?);
    }
    variables.push(Weight::at_zero(PRODUCT_DEGREE + 1)?);
    Ok(variables)
}

/// Number of hiding coefficients: `1 + sum_j (d_j - 1)`.
pub fn hp_entries(variables: &[Weight]) -> Result<usize> {
    if variables.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut entries = 1usize;
    for weight in variables {
        if weight.degree() < 2 {
            return Err(Error::InvalidConfig);
        }
        entries = entries
            .checked_add(weight.degree() - 1)
            .ok_or(Error::ResourceLimit)?;
    }
    Ok(entries)
}

/// Total one-time mask entries of the sumcheck: the hiding coefficients and
/// the two shown-value masks.
pub fn psc_mask_entries(k: usize) -> Result<usize> {
    Ok(hp_entries(&psc_variables(k)?)?
        .checked_add(SHOWN_MAPS)
        .ok_or(Error::ResourceLimit)?)
}

/// The position of the coefficient of `X_l^t` in the hiding coefficient
/// vector: ordered by variable, then by rising degree, with the constant of a
/// non-last variable omitted.
pub fn hp_place(variables: &[Weight], variable: usize, power: usize) -> Result<usize> {
    if variable >= variables.len() {
        return Err(Error::InvalidIndex);
    }
    let last = variable + 1 == variables.len();
    if power >= variables[variable].degree() || (!last && power == 0) {
        return Err(Error::InvalidIndex);
    }
    let mut at = 0usize;
    for weight in &variables[..variable] {
        at = at
            .checked_add(weight.degree() - 1)
            .ok_or(Error::ResourceLimit)?;
    }
    Ok(at + if last { power } else { power - 1 })
}

/// `U` at the challenges as a linear form in the hiding coefficients:
/// `out[hp_place(l, t)] = gamma_l^t prod_{j>l} psi_j(gamma_j)`.
pub fn hp_form(variables: &[Weight], challenges: &[Fp3]) -> Result<Vec<Fp3>> {
    if variables.is_empty() || challenges.len() != variables.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut form = vec![Fp3::ZERO; hp_entries(variables)?];
    let mut tail = Fp3::ONE;
    for variable in (0..variables.len()).rev() {
        let last = variable + 1 == variables.len();
        let mut power = if last {
            tail
        } else {
            tail.mul(&challenges[variable])
        };
        let first = if last { 0 } else { 1 };
        for exponent in first..variables[variable].degree() {
            form[hp_place(variables, variable, exponent)?] = power;
            power = power.mul(&challenges[variable]);
        }
        tail = tail.mul(&poly_at(
            variables[variable].psi(),
            challenges[variable],
        ));
    }
    Ok(form)
}

// ---------------------------------------------------------------------------
// The hiding polynomial and its sumcheck.

/// The hiding polynomial `U(X_1, .., X_k) = sum_l w_l(X_l) prod_{j>l}
/// psi_j(X_j)` with the coefficients supplied in [`hp_place`] order.
///
/// Call [`HidingPolynomial::sum`] first, then [`HidingPolynomial::round`] and
/// [`HidingPolynomial::bind`] for each variable, then
/// [`HidingPolynomial::value`].
pub struct HidingPolynomial {
    variables: Vec<Weight>,
    coefficients: Vec<Vec<Fp3>>,
    after: Vec<Fp3>,
    sum: Fp3,
    before: Fp3,
    bound: usize,
}

impl HidingPolynomial {
    pub fn new(variables: &[Weight], entries: &[Fp3]) -> Result<Self> {
        let expected = hp_entries(variables)?;
        if entries.len() != expected {
            return Err(Error::ShapeMismatch);
        }
        let k = variables.len();
        let mut coefficients = Vec::with_capacity(k);
        for (variable, weight) in variables.iter().enumerate() {
            let mut w = vec![Fp3::ZERO; weight.degree()];
            let first = if variable + 1 == k { 0 } else { 1 };
            for exponent in first..weight.degree() {
                w[exponent] = entries[hp_place(variables, variable, exponent)?];
            }
            coefficients.push(w);
        }
        // after[i] is the constant that the variables after i contribute to
        // round i: sum_{l>i} phi_l(w_l) prod_{i<j<l} phi_j(1). One more step
        // of the same recursion gives the sum of U.
        let mut after = vec![Fp3::ZERO; k];
        for index in (0..k - 1).rev() {
            after[index] = variables[index + 1]
                .apply(&coefficients[index + 1])?
                .add(&variables[index + 1].of_one().mul(&after[index + 1]));
        }
        let sum = variables[0]
            .apply(&coefficients[0])?
            .add(&variables[0].of_one().mul(&after[0]));
        Ok(Self {
            variables: variables.to_vec(),
            coefficients,
            after,
            sum,
            before: Fp3::ZERO,
            bound: 0,
        })
    }

    /// `(phi_1 x .. x phi_k)(U)`.
    pub fn sum(&self) -> Fp3 {
        self.sum
    }

    /// The round polynomial of the next variable: `w_i + before psi_i +
    /// after[i]`.
    pub fn round(&self) -> Result<Vec<Fp3>> {
        if self.bound >= self.variables.len() {
            return Err(Error::InvalidSessionState);
        }
        let weight = &self.variables[self.bound];
        let mut polynomial = self.coefficients[self.bound].clone();
        for (index, coefficient) in weight.psi().iter().enumerate() {
            polynomial[index] = polynomial[index].add(&self.before.mul(coefficient));
        }
        polynomial[0] = polynomial[0].add(&self.after[self.bound]);
        Ok(polynomial)
    }

    /// Binds the next variable to `challenge`.
    pub fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables.len() {
            return Err(Error::InvalidSessionState);
        }
        let weight = &self.variables[self.bound];
        self.before = self
            .before
            .mul(&poly_at(weight.psi(), challenge))
            .add(&poly_at(&self.coefficients[self.bound], challenge));
        self.bound += 1;
        Ok(())
    }

    /// `U` at the challenges; all variables must be bound.
    pub fn value(&self) -> Result<Fp3> {
        if !self.is_complete() {
            return Err(Error::InvalidSessionState);
        }
        Ok(self.before)
    }

    /// Whether every variable has been bound.
    pub fn is_complete(&self) -> bool {
        self.bound == self.variables.len()
    }
}

// ---------------------------------------------------------------------------
// The product summand and its unmasked round polynomials.

/// The public product summand `S(y_X, y_Omega) = y_X y_Omega`: two shown
/// maps, no kept maps, degree two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductSummand;

impl ProductSummand {
    /// Number of shown maps.
    pub const SHOWN: usize = SHOWN_MAPS;
    /// Number of kept maps.
    pub const KEPT: usize = 0;
    /// Degree in the shown values.
    pub const DEGREE: usize = PRODUCT_DEGREE;

    /// The summand at a point, as an affine form in the kept maps: one
    /// coefficient, `y_X y_Omega`. The point does not enter the product.
    pub fn at(_point: &[Fp3], y: &[Fp3]) -> Result<[Fp3; 1]> {
        if y.len() != Self::SHOWN {
            return Err(Error::ShapeMismatch);
        }
        Ok([y[0].mul(&y[1])])
    }
}

/// The prover's unmasked sumcheck for `S = X_chi Omega` over `k` variables.
///
/// The vector index is read as bits from low to high, the lowest bit being
/// the first variable. Binding a variable replaces each adjacent pair by its
/// linear interpolation at the challenge, which is the multilinear extension
/// evaluated at that challenge.
pub struct ProductSumcheckProver {
    x: Vec<Fp3>,
    omega: Vec<Fp3>,
    variables: usize,
    bound: usize,
}

impl ProductSumcheckProver {
    pub fn new(x: &[Fp3], omega: &[Fp3]) -> Result<Self> {
        if x.len() != omega.len() {
            return Err(Error::ShapeMismatch);
        }
        let variables = power_of_two_exponent(x.len())?;
        if variables == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            x: x.to_vec(),
            omega: omega.to_vec(),
            variables,
            bound: 0,
        })
    }

    /// Number of variables of the sumcheck.
    pub fn variables(&self) -> usize {
        self.variables
    }

    /// The round polynomial of the next variable: for each adjacent pair,
    /// `(x_lo + X (x_hi - x_lo)) (omega_lo + X (omega_hi - omega_lo))`,
    /// summed over all pairs.
    pub fn round(&self) -> Result<[Fp3; 3]> {
        if self.bound >= self.variables {
            return Err(Error::InvalidSessionState);
        }
        product_round(&self.x, &self.omega)
    }

    /// Binds the next variable to `challenge` and folds both vectors.
    pub fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.bound >= self.variables {
            return Err(Error::InvalidSessionState);
        }
        fold_first(&mut self.x, challenge)?;
        fold_first(&mut self.omega, challenge)?;
        self.bound += 1;
        Ok(())
    }

    /// The two map values at the challenges; all variables must be bound.
    pub fn end_values(&self) -> Result<[Fp3; SHOWN_MAPS]> {
        if self.bound != self.variables || self.x.len() != 1 || self.omega.len() != 1 {
            return Err(Error::InvalidSessionState);
        }
        Ok([self.x[0], self.omega[0]])
    }
}

fn product_round(x: &[Fp3], omega: &[Fp3]) -> Result<[Fp3; 3]> {
    if x.len() != omega.len() || x.len() % 2 != 0 {
        return Err(Error::ShapeMismatch);
    }
    let mut round = [Fp3::ZERO; 3];
    for index in 0..x.len() / 2 {
        let x_low = x[2 * index];
        let x_delta = x[2 * index + 1].sub(&x_low);
        let omega_low = omega[2 * index];
        let omega_delta = omega[2 * index + 1].sub(&omega_low);
        round[0] = round[0].add(&x_low.mul(&omega_low));
        round[1] = round[1].add(&x_low.mul(&omega_delta).add(&x_delta.mul(&omega_low)));
        round[2] = round[2].add(&x_delta.mul(&omega_delta));
    }
    Ok(round)
}

fn fold_first(values: &mut Vec<Fp3>, challenge: Fp3) -> Result<()> {
    if values.len() % 2 != 0 {
        return Err(Error::ShapeMismatch);
    }
    let rows = values.len() / 2;
    for index in 0..rows {
        let low = values[2 * index];
        let high = values[2 * index + 1];
        values[index] = low.add(&challenge.mul(&high.sub(&low)));
    }
    values.truncate(rows);
    Ok(())
}

// ---------------------------------------------------------------------------
// Claim, masks, messages and output.

/// One claim mask: a coefficient and the one-time mask entry it multiplies.
/// The entry is private; only its linear claim is public.
#[derive(Clone, PartialEq, Eq)]
pub struct ClaimMask {
    pub coefficient: Fp3,
    pub value: Fp3,
}

impl fmt::Debug for ClaimMask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ClaimMask")
            .field("coefficient", &self.coefficient)
            .field("value", &"<mask entry>")
            .finish()
    }
}

/// The public claim: the mask coefficients, and the target that the masked
/// sum must equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotClaim {
    pub masks: Vec<ClaimMask>,
    pub target: Fp3,
}

impl DotClaim {
    /// The value of the claim's mask form, `sigma = sum_t coeff_t value_t`.
    pub fn mask_sum(&self) -> Result<Fp3> {
        let mut sum = Fp3::ZERO;
        for mask in &self.masks {
            sum = sum.add(&mask.coefficient.mul(&mask.value));
        }
        Ok(sum)
    }
}

/// The prover's one-time mask entries: the hiding coefficients in
/// [`hp_place`] order and the two shown-value masks.
#[derive(Clone, PartialEq, Eq)]
pub struct DotMaskEntries {
    pub hiding: Vec<Fp3>,
    pub shown: [Fp3; SHOWN_MAPS],
}

impl fmt::Debug for DotMaskEntries {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DotMaskEntries")
            .field("hiding_entries", &self.hiding.len())
            .field("shown_entries", &self.shown.len())
            .finish()
    }
}

/// One round message: the coefficients in increasing degree order with the
/// coefficient fixed by the round check omitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PscRound {
    pub coefficients: Vec<Fp3>,
}

/// The prover's messages of the masked sumcheck.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DotProof {
    /// `mu'`, the sum of the hiding polynomial.
    pub hiding_sum: Fp3,
    /// One round for each `X` variable.
    pub rounds: Vec<PscRound>,
    /// The round of the last (`Y`) variable.
    pub y_round: PscRound,
    /// The masked end values of the two shown maps.
    pub end_values: [Fp3; SHOWN_MAPS],
}

/// A point claim on one shown map: `map(point) + mask_scale value = value`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointClaim {
    pub point: Vec<Fp3>,
    pub mask_scale: Fp3,
    pub value: Fp3,
}

/// The linear claims that the masked sumcheck leaves for its caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotOutput {
    /// The `X` challenge point `gamma`, of length `k`.
    pub point: Vec<Fp3>,
    /// The coefficients of the hiding polynomial's linear form at
    /// `(gamma, beta)`.
    pub hiding_form: Vec<Fp3>,
    /// `eps prod_t psi_t(challenge_t)`: the coefficient that multiplies the
    /// claim's mask form.
    pub claim_scale: Fp3,
    /// The `(U)` row target: the last round value minus
    /// `eps (end_X end_Omega)`.
    pub row_target: Fp3,
    /// One claim for each shown map, at `point`.
    pub ends: [PointClaim; SHOWN_MAPS],
}

/// A source of verifier challenges.
///
/// The source must absorb every message sent so far before it yields the next
/// challenge. The hooks below are called at the points where the prover sends
/// a value: the hiding sum before the first challenge, each round message
/// before its challenge, and the masked end values after the last challenge.
/// A production source absorbs them into its transcript; the default
/// implementations do nothing, so the seedable source in this module's tests
/// only exercises the arithmetic.
pub trait PscChallenges {
    fn draw(&mut self) -> Result<Fp3>;
    fn draw_nonzero(&mut self) -> Result<Fp3>;

    /// Absorbs `mu'`, the hiding polynomial's sum, before the first challenge.
    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()> {
        let _ = value;
        Ok(())
    }

    /// Absorbs the round message of variable `index` before its challenge.
    fn send_round(&mut self, index: usize, round: &PscRound) -> Result<()> {
        let _ = (index, round);
        Ok(())
    }

    /// Absorbs the masked end values after the last challenge.
    fn send_end_values(&mut self, values: &[Fp3; SHOWN_MAPS]) -> Result<()> {
        let _ = values;
        Ok(())
    }
}

impl PscChallenges for XorShift64 {
    fn draw(&mut self) -> Result<Fp3> {
        Ok(Fp3::from_base(self.field()))
    }

    fn draw_nonzero(&mut self) -> Result<Fp3> {
        for _ in 0..64 {
            let value = self.draw()?;
            if !value.is_zero() {
                return Ok(value);
            }
        }
        Err(Error::EntropyUnavailable)
    }
}

// ---------------------------------------------------------------------------
// Prover and verifier.

/// Proves the masked product-sumcheck claim over plaintext vectors.
///
/// The prover needs both vectors, the claim (masks and target), and its
/// one-time mask entries. The target is the public value
/// `sum_t coeff_t (tilde_t)` of the caller's relation; an honest prover
/// refuses a target that does not equal the masked sum.
pub fn prove_dot_sumcheck(
    x_chi: &[Fp3],
    omega: &[Fp3],
    claim: &DotClaim,
    masks: &DotMaskEntries,
    challenges: &mut dyn PscChallenges,
) -> Result<(DotProof, DotOutput)> {
    let _profile_psc = zkie_engine::profile::Scope::new("psc.prove_dot_sumcheck");
    if x_chi.len() != omega.len() {
        return Err(Error::ShapeMismatch);
    }
    let k = power_of_two_exponent(x_chi.len())?;
    if k == 0 {
        return Err(Error::InvalidConfig);
    }
    let variables = psc_variables(k)?;
    if masks.hiding.len() != hp_entries(&variables)? {
        return Err(Error::ShapeMismatch);
    }
    let sigma = claim.mask_sum()?;

    let _profile_product_new = zkie_engine::profile::Scope::new("psc.product_new");
    let mut inner = ProductSumcheckProver::new(x_chi, omega)?;
    drop(_profile_product_new);
    let _profile_hiding_new = zkie_engine::profile::Scope::light("psc.hiding_new");
    let mut hiding = HidingPolynomial::new(&variables, &masks.hiding)?;
    drop(_profile_hiding_new);

    let mut proof = DotProof {
        hiding_sum: hiding.sum(),
        rounds: Vec::with_capacity(k),
        y_round: PscRound {
            coefficients: Vec::new(),
        },
        end_values: [Fp3::ZERO; SHOWN_MAPS],
    };
    challenges.send_hiding_sum(proof.hiding_sum)?;
    let eps = challenges.draw()?;
    let mut running = proof.hiding_sum.add(&eps.mul(&claim.target));
    let mut psis = Fp3::ONE;
    let mut drawn = Vec::with_capacity(k + 1);

    for variable in 0..k {
        let _profile_product_round = zkie_engine::profile::Scope::light("psc.product_round");
        let summand_round = inner.round()?;
        drop(_profile_product_round);
        let _profile_hiding_round = zkie_engine::profile::Scope::light("psc.hiding_round");
        let mut round = hiding.round()?;
        drop(_profile_hiding_round);
        add_scaled(&mut round, &summand_round, eps)?;
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
        let _profile_bind = zkie_engine::profile::Scope::light("psc.bind");
        hiding.bind(challenge)?;
        psis = psis.mul(&poly_at(variables[variable].psi(), challenge));
        inner.bind(challenge)?;
        drop(_profile_bind);
    }

    // The last variable masks the shown end values. Its round polynomial is
    // the product summand in Y, read from its values at 0, 1 and 2.
    let _profile_end = zkie_engine::profile::Scope::new("psc.end");
    let ends = inner.end_values()?;
    let mut values = Vec::with_capacity(PRODUCT_DEGREE + 1);
    for node in 0..=PRODUCT_DEGREE {
        let point = Fp3::from_u64(node as u64);
        let shown = [
            ends[0].add(&masks.shown[0].mul(&point)),
            ends[1].add(&masks.shown[1].mul(&point)),
        ];
        values.push(ProductSummand::at(&[], &shown)?[0]);
    }
    let summand_round = poly_from_values_at_0_1_2(&values)?;
    let mut round = hiding.round()?;
    add_scaled(&mut round, &summand_round, eps)?;
    add_weight(
        &mut round,
        &variables[k],
        eps.mul(&sigma).mul(&psis),
    )?;
    if variables[k].apply(&round)? != running {
        return Err(Error::InvalidProof);
    }
    proof.y_round = PscRound {
        coefficients: sent_coefficients(&variables[k], &round)?,
    };
    challenges.send_round(k, &proof.y_round)?;
    let beta = challenges.draw_nonzero()?;
    drawn.push(beta);
    running = poly_at(&round, beta);
    hiding.bind(beta)?;

    proof.end_values = [
        ends[0].add(&beta.mul(&masks.shown[0])),
        ends[1].add(&beta.mul(&masks.shown[1])),
    ];
    challenges.send_end_values(&proof.end_values)?;
    let output = build_output(&variables, eps, &drawn, running, &proof.end_values)?;
    Ok((proof, output))
}

/// Verifies the prover messages and returns the linear output claims.
///
/// This is the commitment-free part: it recomputes every challenge from its
/// own source, completes each round polynomial with the fixed coefficient,
/// and checks the final round value. The caller must check the output claims
/// against the committed vectors and mask message; see
/// [`check_product_claims`] for the plaintext form.
pub fn verify_psc_transcript(
    claim: &DotClaim,
    masks: &DotMaskEntries,
    proof: &DotProof,
    k: usize,
    challenges: &mut dyn PscChallenges,
) -> Result<DotOutput> {
    let variables = psc_variables(k)?;
    if masks.hiding.len() != hp_entries(&variables)? {
        return Err(Error::ShapeMismatch);
    }
    if proof.rounds.len() != k {
        return Err(Error::InvalidProof);
    }
    challenges.send_hiding_sum(proof.hiding_sum)?;
    let eps = challenges.draw()?;
    let mut running = proof.hiding_sum.add(&eps.mul(&claim.target));
    let mut drawn = Vec::with_capacity(k + 1);
    for variable in 0..k {
        let round = complete_round(&variables[variable], &proof.rounds[variable], running)?;
        challenges.send_round(variable, &proof.rounds[variable])?;
        let challenge = challenges.draw()?;
        drawn.push(challenge);
        running = poly_at(&round, challenge);
    }
    let round = complete_round(&variables[k], &proof.y_round, running)?;
    challenges.send_round(k, &proof.y_round)?;
    let beta = challenges.draw_nonzero()?;
    drawn.push(beta);
    running = poly_at(&round, beta);
    challenges.send_end_values(&proof.end_values)?;
    build_output(&variables, eps, &drawn, running, &proof.end_values)
}

/// Verifies the masked product-sumcheck claim over plaintext vectors and
/// checks the output claims against them.
pub fn verify_dot_sumcheck(
    x_chi: &[Fp3],
    omega: &[Fp3],
    claim: &DotClaim,
    masks: &DotMaskEntries,
    proof: &DotProof,
    challenges: &mut dyn PscChallenges,
) -> Result<DotOutput> {
    if x_chi.len() != omega.len() {
        return Err(Error::ShapeMismatch);
    }
    let k = power_of_two_exponent(x_chi.len())?;
    let output = verify_psc_transcript(claim, masks, proof, k, challenges)?;
    check_product_claims(&output, claim, masks, x_chi, omega)?;
    Ok(output)
}

/// Checks the output claims against plaintext vectors and mask entries:
///
/// * the `(U)` row
///   `<hiding_form, entries> + claim_scale sigma = row_target`;
/// * each shown-map claim
///   `value = map(point) + mask_scale entry`.
///
/// A caller with commitments checks the same equations as linear claims on
/// the committed oracles instead of evaluating plaintext vectors.
pub fn check_product_claims(
    output: &DotOutput,
    claim: &DotClaim,
    masks: &DotMaskEntries,
    x_chi: &[Fp3],
    omega: &[Fp3],
) -> Result<()> {
    if x_chi.len() != omega.len() {
        return Err(Error::ShapeMismatch);
    }
    let k = power_of_two_exponent(x_chi.len())?;
    if output.point.len() != k || output.hiding_form.len() != masks.hiding.len() {
        return Err(Error::InvalidProof);
    }
    let sigma = claim.mask_sum()?;
    let mut row = Fp3::ZERO;
    for (coefficient, entry) in output.hiding_form.iter().zip(&masks.hiding) {
        row = row.add(&coefficient.mul(entry));
    }
    row = row.add(&output.claim_scale.mul(&sigma));
    if row != output.row_target {
        return Err(Error::InvalidProof);
    }
    let maps = [
        multilinear_eval(x_chi, &output.point)?,
        multilinear_eval(omega, &output.point)?,
    ];
    for (index, end) in output.ends.iter().enumerate() {
        if end.point != output.point {
            return Err(Error::InvalidProof);
        }
        let expected = maps[index].add(&end.mask_scale.mul(&masks.shown[index]));
        if end.value != expected {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

fn sent_coefficients(weight: &Weight, round: &[Fp3]) -> Result<Vec<Fp3>> {
    if round.len() != weight.degree() {
        return Err(Error::ShapeMismatch);
    }
    let mut sent = Vec::with_capacity(round.len() - 1);
    for (index, coefficient) in round.iter().enumerate() {
        if index != weight.fixed() {
            sent.push(*coefficient);
        }
    }
    Ok(sent)
}

fn complete_round(weight: &Weight, message: &PscRound, claim: Fp3) -> Result<Vec<Fp3>> {
    if message.coefficients.len() != weight.degree() - 1 {
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

fn build_output(
    variables: &[Weight],
    eps: Fp3,
    challenges: &[Fp3],
    last: Fp3,
    end_values: &[Fp3; SHOWN_MAPS],
) -> Result<DotOutput> {
    if challenges.len() != variables.len() {
        return Err(Error::ShapeMismatch);
    }
    let point = challenges[..variables.len() - 1].to_vec();
    let hiding_form = hp_form(variables, challenges)?;
    let mut claim_scale = eps;
    for (weight, challenge) in variables.iter().zip(challenges) {
        claim_scale = claim_scale.mul(&poly_at(weight.psi(), *challenge));
    }
    let product = end_values[0].mul(&end_values[1]);
    let row_target = last.sub(&eps.mul(&product));
    let beta = challenges[variables.len() - 1];
    let ends = [
        PointClaim {
            point: point.clone(),
            mask_scale: beta,
            value: end_values[0],
        },
        PointClaim {
            point: point.clone(),
            mask_scale: beta,
            value: end_values[1],
        },
    ];
    Ok(DotOutput {
        point,
        hiding_form,
        claim_scale,
        row_target,
        ends,
    })
}

// ---------------------------------------------------------------------------
// Shared arithmetic.

fn poly_at(coefficients: &[Fp3], x: Fp3) -> Fp3 {
    let mut value = Fp3::ZERO;
    for coefficient in coefficients.iter().rev() {
        value = value.mul(&x).add(coefficient);
    }
    value
}

/// The polynomial of degree below `values.len()` that takes `values[i]` at
/// the node `i` for `i = 0, 1, ..` (Newton's divided differences).
fn poly_from_values_at_0_1_2(values: &[Fp3]) -> Result<Vec<Fp3>> {
    let count = values.len();
    if count == 0 {
        return Err(Error::InvalidConfig);
    }
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
        return Err(Error::ShapeMismatch);
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

fn power_of_two_exponent(length: usize) -> Result<usize> {
    if length < 2 || !length.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    Ok(length.trailing_zeros() as usize)
}

/// The direct Boolean-cube sum of the two vectors, equal to their inner
/// product.
pub fn direct_product_sum(x: &[Fp3], omega: &[Fp3]) -> Result<Fp3> {
    if x.len() != omega.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (left, right) in x.iter().zip(omega) {
        sum = sum.add(&left.mul(right));
    }
    Ok(sum)
}

/// The multilinear extension of `values` at `point`, by folding one variable
/// per coordinate.
pub fn multilinear_eval(values: &[Fp3], point: &[Fp3]) -> Result<Fp3> {
    let exponent = u32::try_from(point.len()).map_err(|_| Error::InvalidConfig)?;
    let expected = 1usize
        .checked_shl(exponent)
        .ok_or(Error::InvalidConfig)?;
    if values.len() != expected {
        return Err(Error::ShapeMismatch);
    }
    let mut current = values.to_vec();
    for challenge in point {
        let rows = current.len() / 2;
        for index in 0..rows {
            let low = current[2 * index];
            let high = current[2 * index + 1];
            current[index] = low.add(&challenge.mul(&high.sub(&low)));
        }
        current.truncate(rows);
    }
    Ok(current[0])
}

/// The powers `chi^0, chi^1, ..` used to batch the positions of a claim.
pub fn chi_powers(chi: Fp3, positions: usize) -> Result<Vec<Fp3>> {
    if positions == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut powers = Vec::with_capacity(positions);
    let mut power = Fp3::ONE;
    for _ in 0..positions {
        powers.push(power);
        power = power.mul(&chi);
    }
    Ok(powers)
}

/// The coefficient-weighted sum of rows, `sum_t coefficients[t] rows[t]`.
pub fn batch_chi_rows(rows: &[Vec<Fp3>], coefficients: &[Fp3]) -> Result<Vec<Fp3>> {
    if rows.is_empty() || rows.len() != coefficients.len() {
        return Err(Error::ShapeMismatch);
    }
    let width = rows[0].len();
    if width == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut batch = vec![Fp3::ZERO; width];
    for (row, coefficient) in rows.iter().zip(coefficients) {
        if row.len() != width {
            return Err(Error::ShapeMismatch);
        }
        for (slot, value) in batch.iter_mut().zip(row) {
            *slot = slot.add(&coefficient.mul(value));
        }
    }
    Ok(batch)
}

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod tests {
    use super::*;

    fn f(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    fn neg(value: u64) -> Fp3 {
        Fp3::ZERO.sub(&f(value))
    }

    fn field_vector(values: &[i64]) -> Vec<Fp3> {
        values
            .iter()
            .map(|&value| {
                if value >= 0 {
                    f(value as u64)
                } else {
                    neg(value.unsigned_abs())
                }
            })
            .collect()
    }

    fn dot(left: &[Fp3], right: &[Fp3]) -> Fp3 {
        assert_eq!(left.len(), right.len());
        let mut sum = Fp3::ZERO;
        for (a, b) in left.iter().zip(right) {
            sum = sum.add(&a.mul(b));
        }
        sum
    }

    /// Independent multilinear extension: enumerate the Boolean cube and
    /// weight each vertex by the Lagrange factor of `point`.
    fn mle_by_enumeration(values: &[Fp3], point: &[Fp3]) -> Fp3 {
        let k = point.len();
        assert_eq!(values.len(), 1usize << k);
        let mut total = Fp3::ZERO;
        for vertex in 0..(1usize << k) {
            let mut weight = Fp3::ONE;
            for (variable, challenge) in point.iter().enumerate() {
                let bit = (vertex >> variable) & 1;
                let factor = if bit == 1 {
                    *challenge
                } else {
                    Fp3::ONE.sub(challenge)
                };
                weight = weight.mul(&factor);
            }
            total = total.add(&weight.mul(&values[vertex]));
        }
        total
    }

    /// Independent round-polynomial evaluation: fix the previous challenges,
    /// set the current variable to `at`, and sum the product of the two
    /// multilinear extensions over the remaining Boolean variables.
    fn round_value_by_enumeration(
        x: &[Fp3],
        omega: &[Fp3],
        previous: &[Fp3],
        at: Fp3,
    ) -> Fp3 {
        let k = x.len().trailing_zeros() as usize;
        assert_eq!(omega.len(), x.len());
        assert!(previous.len() < k);
        let remaining = k - previous.len() - 1;
        let mut total = Fp3::ZERO;
        for tail in 0..(1usize << remaining) {
            let mut point = previous.to_vec();
            point.push(at);
            for variable in 0..remaining {
                let bit = (tail >> variable) & 1;
                point.push(if bit == 1 { Fp3::ONE } else { Fp3::ZERO });
            }
            let product =
                mle_by_enumeration(x, &point).mul(&mle_by_enumeration(omega, &point));
            total = total.add(&product);
        }
        total
    }

    /// U evaluated from the polynomial definition with the entry layout
    /// written out directly: for the first `k` variables the entries are
    /// `(w_1, w_2)` and the constant is zero; the last variable keeps
    /// `(w_0, w_1, w_2)`.
    fn independent_hiding_value(entries: &[Fp3], point: &[Fp3]) -> Fp3 {
        let k = point.len() - 1;
        assert_eq!(entries.len(), 2 * k + 3);
        let mut total = Fp3::ZERO;
        for variable in 0..=k {
            let w = if variable < k {
                [Fp3::ZERO, entries[2 * variable], entries[2 * variable + 1]]
            } else {
                [
                    entries[2 * k],
                    entries[2 * k + 1],
                    entries[2 * k + 2],
                ]
            };
            let mut term = w[0]
                .add(&w[1].mul(&point[variable]))
                .add(&w[2].mul(&point[variable]).mul(&point[variable]));
            for later in (variable + 1)..=k {
                let psi = if later == k {
                    Fp3::ONE
                } else {
                    point[later]
                };
                term = term.mul(&psi);
            }
            total = total.add(&term);
        }
        total
    }

    /// The verifier-side form written out directly.
    fn independent_hiding_form(point: &[Fp3]) -> Vec<Fp3> {
        let k = point.len() - 1;
        let mut form = Vec::with_capacity(2 * k + 3);
        for variable in 0..k {
            let mut tail = Fp3::ONE;
            for later in (variable + 1)..k {
                tail = tail.mul(&point[later]);
            }
            form.push(point[variable].mul(&tail));
            form.push(point[variable].mul(&point[variable]).mul(&tail));
        }
        form.push(Fp3::ONE);
        form.push(point[k]);
        form.push(point[k].mul(&point[k]));
        form
    }

    fn test_masks(k: usize) -> DotMaskEntries {
        let entries = hp_entries(&psc_variables(k).unwrap()).unwrap();
        DotMaskEntries {
            hiding: (0..entries).map(|index| f(200 + index as u64)).collect(),
            shown: [f(13), f(17)],
        }
    }

    fn test_claim(
        x: &[Fp3],
        omega: &[Fp3],
        chi: Fp3,
        positions: &[u64],
    ) -> DotClaim {
        let masks = chi_powers(chi, positions.len())
            .unwrap()
            .into_iter()
            .zip(positions)
            .map(|(coefficient, &value)| ClaimMask {
                coefficient,
                value: f(value),
            })
            .collect();
        let claim = DotClaim {
            masks,
            target: Fp3::ZERO,
        };
        let target = dot(x, omega).add(&claim.mask_sum().unwrap());
        DotClaim { target, ..claim }
    }

    fn proving_source(seed: u64) -> XorShift64 {
        XorShift64::new(seed)
    }

    #[test]
    fn weights_follow_the_plain_and_at_zero_functionals() {
        let plain = Weight::plain(3).unwrap();
        assert_eq!(plain.apply(&[f(5), f(7), f(11)]).unwrap(), f(10 + 7 + 11));
        assert_eq!(plain.of_one(), f(2));
        assert_eq!(plain.fixed(), 1);
        let mut round = vec![f(5), f(0), f(11)];
        plain.complete(&mut round, f(70)).unwrap();
        assert_eq!(round[1], f(49));
        assert_eq!(plain.apply(&round).unwrap(), f(70));

        let at_zero = Weight::at_zero(3).unwrap();
        assert_eq!(at_zero.apply(&[f(5), f(7), f(11)]).unwrap(), f(5));
        assert_eq!(at_zero.of_one(), f(1));
        assert_eq!(at_zero.fixed(), 0);
        let mut round = vec![f(0), f(7), f(11)];
        at_zero.complete(&mut round, f(42)).unwrap();
        assert_eq!(round[0], f(42));

        let variables = psc_variables(DOT_VARIABLES).unwrap();
        assert_eq!(variables.len(), DOT_VARIABLES + 1);
        assert_eq!(hp_entries(&variables).unwrap(), 23);
        assert_eq!(psc_mask_entries(DOT_VARIABLES).unwrap(), 25);
        assert_eq!(hp_entries(&psc_variables(2).unwrap()).unwrap(), 7);
        assert_eq!(psc_mask_entries(2).unwrap(), 9);
    }

    #[test]
    fn hand_k2_matches_independent_evaluation() {
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[5, 6, 7, 8]);

        assert_eq!(dot(&x, &omega), f(70));
        assert_eq!(direct_product_sum(&x, &omega).unwrap(), f(70));
        assert_eq!(mle_by_enumeration(&x, &field_vector(&[2, 3])), f(9));
        assert_eq!(mle_by_enumeration(&omega, &field_vector(&[2, 3])), f(13));
        assert_eq!(
            multilinear_eval(&x, &field_vector(&[2, 3])).unwrap(),
            f(9)
        );
        assert_eq!(
            multilinear_eval(&omega, &field_vector(&[2, 3])).unwrap(),
            f(13)
        );

        let mut prover = ProductSumcheckProver::new(&x, &omega).unwrap();
        assert_eq!(prover.round().unwrap(), [f(26), f(16), f(2)]);
        assert_eq!(poly_at(&prover.round().unwrap(), f(5)), f(156));
        assert_eq!(
            round_value_by_enumeration(&x, &omega, &[], f(5)),
            f(156)
        );
        prover.bind(f(2)).unwrap();
        assert_eq!(prover.round().unwrap(), [f(21), f(20), f(4)]);
        assert_eq!(poly_at(&prover.round().unwrap(), f(7)), f(357));
        assert_eq!(
            round_value_by_enumeration(&x, &omega, &[f(2)], f(7)),
            f(357)
        );
        prover.bind(f(3)).unwrap();
        assert_eq!(prover.end_values().unwrap(), [f(9), f(13)]);

        // The v1 normalization is not the plain cube sum.
        let scaled = f(70).mul(&f(4).inverse().unwrap());
        assert_ne!(scaled, f(70));
    }

    #[test]
    fn hand_k3_matches_independent_evaluation() {
        let x = field_vector(&[1, 0, 2, 0, 3, 0, 4, 0]);
        let omega = field_vector(&[2, 1, 3, 1, 4, 1, 5, 1]);

        assert_eq!(dot(&x, &omega), f(40));
        assert_eq!(direct_product_sum(&x, &omega).unwrap(), f(40));

        let mut prover = ProductSumcheckProver::new(&x, &omega).unwrap();
        assert_eq!(prover.round().unwrap(), [f(40), neg(70), f(30)]);
        assert_eq!(
            round_value_by_enumeration(&x, &omega, &[], f(2)),
            poly_at(&prover.round().unwrap(), f(2))
        );
        prover.bind(f(2)).unwrap();
        assert_eq!(prover.round().unwrap(), [f(6), f(6), f(2)]);
        assert_eq!(
            round_value_by_enumeration(&x, &omega, &[f(2)], f(3)),
            poly_at(&prover.round().unwrap(), f(3))
        );
        prover.bind(f(3)).unwrap();
        assert_eq!(prover.round().unwrap(), [f(12), f(14), f(4)]);
        assert_eq!(
            round_value_by_enumeration(&x, &omega, &[f(2), f(3)], f(5)),
            poly_at(&prover.round().unwrap(), f(5))
        );
        prover.bind(f(5)).unwrap();
        let ends = prover.end_values().unwrap();
        assert_eq!(ends, [neg(14), neg(13)]);
        let point = field_vector(&[2, 3, 5]);
        assert_eq!(mle_by_enumeration(&x, &point), ends[0]);
        assert_eq!(mle_by_enumeration(&omega, &point), ends[1]);
        assert_eq!(ends[0].mul(&ends[1]), f(182));
    }

    #[test]
    fn hiding_polynomial_matches_its_definition() {
        for k in [2usize, 3usize] {
            let variables = psc_variables(k).unwrap();
            let entries: Vec<Fp3> = (0..hp_entries(&variables).unwrap())
                .map(|index| f(100 + index as u64))
                .collect();
            let hiding = HidingPolynomial::new(&variables, &entries).unwrap();

            // The sum is the plain cube sum of U at Boolean vertices, with the
            // last variable read at zero.
            let mut expected_sum = Fp3::ZERO;
            for vertex in 0..(1usize << k) {
                let mut point: Vec<Fp3> = (0..k)
                    .map(|variable| {
                        if (vertex >> variable) & 1 == 1 {
                            Fp3::ONE
                        } else {
                            Fp3::ZERO
                        }
                    })
                    .collect();
                point.push(Fp3::ZERO);
                expected_sum = expected_sum.add(&independent_hiding_value(&entries, &point));
            }
            assert_eq!(hiding.sum(), expected_sum);

            let point: Vec<Fp3> = (0..=k).map(|index| f(2 + index as u64)).collect();
            let mut hiding = hiding;
            for challenge in &point {
                hiding.bind(*challenge).unwrap();
            }
            let expected_value = independent_hiding_value(&entries, &point);
            assert_eq!(hiding.value().unwrap(), expected_value);

            let form = hp_form(&variables, &point).unwrap();
            assert_eq!(form, independent_hiding_form(&point));
            assert_eq!(dot(&form, &entries), expected_value);
        }
    }

    fn rank_over_fp3(matrix: &[Vec<Fp3>]) -> usize {
        let rows = matrix.len();
        let cols = matrix.first().map_or(0, Vec::len);
        let mut work = matrix.to_vec();
        let mut rank = 0usize;
        for column in 0..cols {
            let pivot = (rank..rows).find(|&row| work[row][column] != Fp3::ZERO);
            let Some(pivot) = pivot else {
                continue;
            };
            work.swap(rank, pivot);
            let inverse = work[rank][column].inverse().unwrap();
            for value in work[rank].iter_mut().skip(column) {
                *value = value.mul(&inverse);
            }
            for row in 0..rows {
                if row == rank || work[row][column] == Fp3::ZERO {
                    continue;
                }
                let factor = work[row][column];
                for column_after in column..cols {
                    work[row][column_after] =
                        work[row][column_after].sub(&factor.mul(&work[rank][column_after]));
                }
            }
            rank += 1;
        }
        rank
    }

    #[test]
    fn production_hiding_coefficient_map_is_full_rank() {
        // The 23 hiding coefficients randomize the 23 sent scalars: the
        // hiding sum plus the 22 non-fixed round coefficients. The sigma
        // direction is one of the coefficient directions (`w_0 = sigma psi_0`),
        // so it adds no rank beyond the hiding map. A full-rank map is the
        // per-session one-to-one property the hiding argument uses.
        let variables = psc_variables(DOT_VARIABLES).unwrap();
        let entries = hp_entries(&variables).unwrap();
        assert_eq!(entries, 23);
        let challenges: Vec<Fp3> = (0..variables.len())
            .map(|index| f(0x100 + index as u64))
            .collect();
        let mut matrix = vec![vec![Fp3::ZERO; entries]; entries];
        for (basis, row) in matrix.iter_mut().enumerate() {
            let mut unit = vec![Fp3::ZERO; entries];
            unit[basis] = Fp3::ONE;
            let mut hiding = HidingPolynomial::new(&variables, &unit).unwrap();
            let mut values = vec![hiding.sum()];
            for (variable, weight) in variables.iter().enumerate() {
                let round = hiding.round().unwrap();
                for (index, coefficient) in round.iter().enumerate() {
                    if index != weight.fixed() {
                        values.push(*coefficient);
                    }
                }
                hiding.bind(challenges[variable]).unwrap();
            }
            assert_eq!(values.len(), entries);
            for (sent, value) in values.into_iter().enumerate() {
                row[sent] = value;
            }
        }
        assert_eq!(rank_over_fp3(&matrix), entries);
        // The sigma direction is the degree-one coefficient of the first
        // variable, which is the first sent hiding entry.
        assert_eq!(hp_place(&variables, 0, 1).unwrap(), 0);
    }

    #[test]
    fn interpolation_matches_the_three_nodes() {
        let values = [f(5), f(11), f(23)];
        let polynomial = poly_from_values_at_0_1_2(&values).unwrap();
        assert_eq!(polynomial, [f(5), f(3), f(3)]);
        for (node, expected) in values.iter().enumerate() {
            assert_eq!(poly_at(&polynomial, f(node as u64)), *expected);
        }
    }

    #[test]
    fn psc_accepts_honest_proof_k2() {
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[5, 6, 7, 8]);
        let claim = test_claim(&x, &omega, f(3), &[7, 11]);
        assert_eq!(claim.mask_sum().unwrap(), f(7 + 33));
        assert_eq!(claim.target, f(110));
        let masks = test_masks(2);

        let (proof, output) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut proving_source(9)).unwrap();
        assert_eq!(proof.rounds.len(), 2);
        assert_eq!(proof.y_round.coefficients.len(), 2);
        assert_eq!(output.point.len(), 2);
        assert_eq!(output.hiding_form.len(), 7);

        let verified = verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(9),
        )
        .unwrap();
        assert_eq!(verified, output);
        check_product_claims(&verified, &claim, &masks, &x, &omega).unwrap();

        // Replay the challenge order independently: eps, gamma_0, gamma_1,
        // beta. The verifier's point and claim scale follow that order.
        let mut replay = proving_source(9);
        let eps = replay.draw().unwrap();
        let gamma_0 = replay.draw().unwrap();
        let gamma_1 = replay.draw().unwrap();
        let beta = replay.draw_nonzero().unwrap();
        assert_eq!(output.point, vec![gamma_0, gamma_1]);
        assert_eq!(output.ends[0].mask_scale, beta);
        assert_eq!(output.ends[1].mask_scale, beta);
        assert_eq!(output.claim_scale, eps.mul(&gamma_0).mul(&gamma_1));

        // A different challenge sequence is a different transcript.
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(10),
        )
        .is_err());
    }

    #[test]
    fn psc_accepts_chi_batched_positions() {
        let k = 3usize;
        let rows = vec![
            field_vector(&[1, 2, 3, 4, 5, 6, 7, 8]),
            field_vector(&[2, 0, 1, 0, 3, 0, 1, 0]),
            field_vector(&[5, 4, 3, 2, 1, 0, 1, 2]),
            vec![Fp3::ONE; 8],
        ];
        let omega = field_vector(&[4, 1, 5, 9, 2, 6, 5, 3]);
        let chi = f(2);
        let powers = chi_powers(chi, rows.len()).unwrap();
        assert_eq!(powers, [f(1), f(2), f(4), f(8)]);
        let batch = batch_chi_rows(&rows, &powers).unwrap();
        let mut expected = Fp3::ZERO;
        for (row, coefficient) in rows.iter().zip(&powers) {
            expected = expected.add(&coefficient.mul(&dot(row, &omega)));
        }
        assert_eq!(dot(&batch, &omega), expected);

        let claim = test_claim(&batch, &omega, chi, &[3, 5, 7, 9]);
        let masks = test_masks(k);
        let (proof, _) =
            prove_dot_sumcheck(&batch, &omega, &claim, &masks, &mut proving_source(21)).unwrap();
        assert!(verify_dot_sumcheck(
            &batch,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(21),
        )
        .is_ok());
    }

    #[test]
    fn control_1024_matches_direct_inner_product() {
        let x: Vec<Fp3> = (0..1024)
            .map(|index| match index % 3 {
                0 => neg(1),
                1 => Fp3::ZERO,
                _ => Fp3::ONE,
            })
            .collect();
        let omega: Vec<Fp3> = (0..1024).map(|index| f(index as u64 + 1)).collect();
        assert_eq!(dot(&x, &omega), neg(342));
        assert_eq!(direct_product_sum(&x, &omega).unwrap(), neg(342));

        let claim = test_claim(&x, &omega, f(5), &[2, 4]);
        let masks = test_masks(10);
        let (proof, _) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut proving_source(31)).unwrap();
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(31),
        )
        .is_ok());

        let ones = vec![Fp3::ONE; 1024];
        assert_eq!(direct_product_sum(&ones, &ones).unwrap(), f(1024));
        let claim = test_claim(&ones, &ones, f(7), &[1, 3]);
        let (proof, _) =
            prove_dot_sumcheck(&ones, &ones, &claim, &masks, &mut proving_source(32)).unwrap();
        assert!(verify_dot_sumcheck(
            &ones,
            &ones,
            &claim,
            &masks,
            &proof,
            &mut proving_source(32),
        )
        .is_ok());

        // The failed normalization: 1024/1024 = 1 instead of 1024.
        let scaled = DotClaim {
            masks: Vec::new(),
            target: Fp3::ONE,
        };
        assert!(prove_dot_sumcheck(
            &ones,
            &ones,
            &scaled,
            &masks,
            &mut proving_source(32),
        )
        .is_err());
    }

    struct SequenceChallenges {
        values: Vec<Fp3>,
        next: usize,
    }

    impl SequenceChallenges {
        fn new(values: Vec<Fp3>) -> Self {
            Self { values, next: 0 }
        }
    }

    impl PscChallenges for SequenceChallenges {
        fn draw(&mut self) -> Result<Fp3> {
            let value = self
                .values
                .get(self.next)
                .copied()
                .ok_or(Error::InvalidProof)?;
            self.next += 1;
            Ok(value)
        }

        fn draw_nonzero(&mut self) -> Result<Fp3> {
            let value = self.draw()?;
            if value.is_zero() {
                return Err(Error::InvalidProof);
            }
            Ok(value)
        }
    }

    struct RecordingChallenges {
        inner: XorShift64,
        events: Vec<String>,
    }

    impl RecordingChallenges {
        fn new(seed: u64) -> Self {
            Self {
                inner: XorShift64::new(seed),
                events: Vec::new(),
            }
        }
    }

    impl PscChallenges for RecordingChallenges {
        fn draw(&mut self) -> Result<Fp3> {
            self.events.push("draw".to_string());
            self.inner.draw()
        }

        fn draw_nonzero(&mut self) -> Result<Fp3> {
            self.events.push("draw_nonzero".to_string());
            self.inner.draw_nonzero()
        }

        fn send_hiding_sum(&mut self, _value: Fp3) -> Result<()> {
            self.events.push("hiding_sum".to_string());
            Ok(())
        }

        fn send_round(&mut self, index: usize, _round: &PscRound) -> Result<()> {
            self.events.push(format!("round_{index}"));
            Ok(())
        }

        fn send_end_values(&mut self, _values: &[Fp3; SHOWN_MAPS]) -> Result<()> {
            self.events.push("end_values".to_string());
            Ok(())
        }
    }

    #[test]
    fn every_sent_value_is_absorbed_before_its_challenge() {
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[5, 6, 7, 8]);
        let claim = test_claim(&x, &omega, f(3), &[7, 11]);
        let masks = test_masks(2);
        let expected = [
            "hiding_sum",
            "draw",
            "round_0",
            "draw",
            "round_1",
            "draw",
            "round_2",
            "draw_nonzero",
            "end_values",
        ];

        let mut prover = RecordingChallenges::new(9);
        let (proof, _) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut prover).unwrap();
        assert_eq!(prover.events, expected);

        let mut verifier = RecordingChallenges::new(9);
        verify_dot_sumcheck(&x, &omega, &claim, &masks, &proof, &mut verifier).unwrap();
        assert_eq!(verifier.events, expected);
    }

    fn honest_k2() -> (Vec<Fp3>, Vec<Fp3>, DotClaim, DotMaskEntries, DotProof) {
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[5, 6, 7, 8]);
        let claim = test_claim(&x, &omega, f(3), &[7, 11]);
        let masks = test_masks(2);
        let (proof, _) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut proving_source(5)).unwrap();
        (x, omega, claim, masks, proof)
    }

    #[test]
    fn rejects_changed_witness_vectors() {
        let (x, omega, claim, masks, proof) = honest_k2();
        let mut changed_x = x.clone();
        changed_x[0] = changed_x[0].add(&Fp3::ONE);
        assert!(verify_dot_sumcheck(
            &changed_x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());

        let mut changed_omega = omega.clone();
        changed_omega[3] = changed_omega[3].add(&Fp3::ONE);
        assert!(verify_dot_sumcheck(
            &x,
            &changed_omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());
    }

    #[test]
    fn rejects_changed_messages() {
        let (x, omega, claim, masks, mut proof) = honest_k2();
        proof.rounds[0].coefficients[0] =
            proof.rounds[0].coefficients[0].add(&Fp3::ONE);
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());

        let (_, _, claim, masks, mut proof) = honest_k2();
        proof.hiding_sum = proof.hiding_sum.add(&Fp3::ONE);
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());

        let (_, _, claim, masks, mut proof) = honest_k2();
        proof.end_values[1] = proof.end_values[1].add(&Fp3::ONE);
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());
    }

    #[test]
    fn rejects_malformed_round_lists() {
        let (x, omega, claim, masks, mut proof) = honest_k2();
        proof.rounds.pop();
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());

        let (_, _, claim, masks, mut proof) = honest_k2();
        proof.y_round.coefficients.push(Fp3::ONE);
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());
    }

    #[test]
    fn rejects_wrong_challenge_order() {
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[5, 6, 7, 8]);
        let claim = test_claim(&x, &omega, f(3), &[7, 11]);
        let masks = test_masks(2);

        // eps, gamma_0, gamma_1, beta.
        let honest = vec![f(2), f(3), f(5), f(7)];
        let (proof, _) = prove_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &mut SequenceChallenges::new(honest.clone()),
        )
        .unwrap();
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut SequenceChallenges::new(honest),
        )
        .is_ok());

        let swapped = vec![f(2), f(5), f(3), f(7)];
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut SequenceChallenges::new(swapped),
        )
        .is_err());

        let zero_beta = vec![f(2), f(3), f(5), Fp3::ZERO];
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut SequenceChallenges::new(zero_beta),
        )
        .is_err());
    }

    #[test]
    fn rejects_wrong_target_and_scaled_target() {
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[5, 6, 7, 8]);
        let claim = test_claim(&x, &omega, f(3), &[7, 11]);
        let masks = test_masks(2);

        // An honest prover refuses a target that is not the masked sum.
        let wrong = DotClaim {
            masks: claim.masks.clone(),
            target: claim.target.add(&Fp3::ONE),
        };
        assert!(prove_dot_sumcheck(
            &x,
            &omega,
            &wrong,
            &masks,
            &mut proving_source(5),
        )
        .is_err());

        // A verified-proof transcript with a wrong public target is rejected.
        let (proof, _) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut proving_source(5)).unwrap();
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &wrong,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());

        // The failed normalization: the target scaled by 2^-2.
        let scaled_target = claim
            .target
            .mul(&f(4).inverse().unwrap());
        let scaled = DotClaim {
            masks: claim.masks.clone(),
            target: scaled_target,
        };
        assert_ne!(scaled_target, claim.target);
        assert!(prove_dot_sumcheck(
            &x,
            &omega,
            &scaled,
            &masks,
            &mut proving_source(5),
        )
        .is_err());
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &scaled,
            &masks,
            &proof,
            &mut proving_source(5),
        )
        .is_err());
    }

    #[test]
    fn accepts_cancellation_and_zero_sums() {
        // The two vectors are orthogonal: every term cancels.
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[4, 3, -2, -1]);
        assert_eq!(dot(&x, &omega), Fp3::ZERO);
        let claim = test_claim(&x, &omega, f(2), &[5, 9]);
        let masks = test_masks(2);
        let (proof, _) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut proving_source(41)).unwrap();
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(41),
        )
        .is_ok());

        // Partial cancellation with a non-zero sum.
        let omega = field_vector(&[2, -1, 2, -1]);
        assert_eq!(dot(&x, &omega), f(2));
        let claim = test_claim(&x, &omega, f(2), &[5, 9]);
        let (proof, _) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut proving_source(42)).unwrap();
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &masks,
            &proof,
            &mut proving_source(42),
        )
        .is_ok());
    }

    #[test]
    fn rejects_short_mask_entries() {
        let x = field_vector(&[1, 2, 3, 4]);
        let omega = field_vector(&[5, 6, 7, 8]);
        let claim = test_claim(&x, &omega, f(3), &[7, 11]);
        let masks = test_masks(2);
        let short = DotMaskEntries {
            hiding: masks.hiding[..masks.hiding.len() - 1].to_vec(),
            shown: masks.shown,
        };
        assert!(prove_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &short,
            &mut proving_source(5),
        )
        .is_err());
        let (proof, _) =
            prove_dot_sumcheck(&x, &omega, &claim, &masks, &mut proving_source(5)).unwrap();
        assert!(verify_dot_sumcheck(
            &x,
            &omega,
            &claim,
            &short,
            &proof,
            &mut proving_source(5),
        )
        .is_err());
    }

    #[test]
    fn equality_weight_sums_the_two_boolean_points() {
        let z = f(3);
        let weight = Weight::equality(z, 3).unwrap();
        let g = [f(5), f(7), f(11)];
        assert_eq!(weight.apply(&g).unwrap(), Fp3::ONE.sub(&z).mul(&f(5)).add(&z.mul(&f(23))));
        assert_eq!(weight.psi(), &[Fp3::ONE]);
        assert_eq!(weight.fixed(), 0);
        assert_eq!(weight.of_one(), Fp3::ONE);
        assert!(Weight::equality(z, 1).is_err());
        assert_eq!(
            Weight::at_zero(3).unwrap().apply(&g).unwrap(),
            Weight::equality(Fp3::ZERO, 3).unwrap().apply(&g).unwrap()
        );
    }

    #[test]
    fn summand_constants_match_the_specification() {
        assert_eq!(ProductSummand::SHOWN, 2);
        assert_eq!(ProductSummand::KEPT, 0);
        assert_eq!(ProductSummand::DEGREE, 2);
        assert_eq!(
            ProductSummand::at(&[f(1)], &[f(3), f(5)]).unwrap(),
            [f(15)]
        );
        assert!(ProductSummand::at(&[], &[f(3)]).is_err());
    }
}
