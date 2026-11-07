//! FOLD: a hiding sumcheck that reduces a linear claim on a group of rows to a
//! claim on the folded rows.
//!
//! The prover sends the fresh mask's commitment and the mask polynomial's sum,
//! receives `eps`, then sends one quadratic round polynomial per variable. Each
//! challenge is drawn from the chain after the round's message. The verifier
//! checks `q(0) + q(1)` against its running claim and the terminal equation on
//! the folded rows.

use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::code::evaluate_ood_fp3;
use crate::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use crate::pcs::request::field::Fp3;
use crate::pcs::request::merkle::{build_column_tree, ColumnTree, ColumnTreeSpec};
use crate::pcs::request::short_code::fold_mask_columns;
use crate::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

fn zero() -> Fp3 {
    Fp3::ZERO
}

fn one() -> Fp3 {
    Fp3::ONE
}

fn two() -> Fp3 {
    Fp3::from_u64(2)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldShape {
    pub k: u32,
    pub h: u32,
    pub len_zk: usize,
}

impl FoldShape {
    pub fn new(k: u32, h: u32, len_zk: usize) -> Result<Self> {
        if k < 1 || h > k {
            return Err(Error::InvalidConfig);
        }
        if (len_zk as u64) < fold_mask_columns(k, h)? {
            return Err(Error::InvalidConfig);
        }
        Ok(Self { k, h, len_zk })
    }

    pub fn main_rows(&self) -> usize {
        1usize << self.k
    }

    pub fn mask_rows(&self) -> usize {
        1usize << self.h
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldRound {
    pub polynomial: [Fp3; 3],
    pub challenge: Fp3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldOutput {
    pub main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub fresh: Vec<Fp3>,
    pub coefficients_main: Vec<Fp3>,
    pub coefficients_mask: Vec<Fp3>,
    pub kept: Fp3,
    pub target: Fp3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldReceipt {
    pub shape: FoldShape,
    pub fresh_root: Digest,
    pub mask_sum: Fp3,
    pub rounds: Vec<FoldRound>,
    pub output: FoldOutput,
}

fn fold_low_once(values: &mut Vec<Fp3>, challenge: Fp3) -> Result<()> {
    if values.len() % 2 != 0 {
        return Err(Error::ShapeMismatch);
    }
    let half = values.len() / 2;
    for index in 0..half {
        let low = values[2 * index];
        let high = values[2 * index + 1];
        values[index] = low.add(&challenge.mul(&high.sub(&low)));
    }
    values.truncate(half);
    Ok(())
}

fn pair_products(a: &[Fp3], m: &[Fp3]) -> Result<[Fp3; 3]> {
    if a.len() != m.len() || a.len() % 2 != 0 {
        return Err(Error::ShapeMismatch);
    }
    let mut q = [zero(); 3];
    for index in (0..a.len()).step_by(2) {
        let a0 = a[index];
        let a1 = a[index + 1].sub(&a0);
        let m0 = m[index];
        let m1 = m[index + 1].sub(&m0);
        q[0] = q[0].add(&a0.mul(&m0));
        q[1] = q[1].add(&a0.mul(&m1).add(&a1.mul(&m0)));
        q[2] = q[2].add(&a1.mul(&m1));
    }
    Ok(q)
}

fn inner(a: &[Fp3], b: &[Fp3]) -> Result<Fp3> {
    if a.len() != b.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = zero();
    for (left, right) in a.iter().zip(b) {
        sum = sum.add(&left.mul(right));
    }
    Ok(sum)
}

fn polynomial_at(coefficients: &[Fp3; 3], point: Fp3) -> Fp3 {
    coefficients[0]
        .add(&coefficients[1].mul(&point))
        .add(&coefficients[2].mul(&point).mul(&point))
}

/// Place the 2k+1 random coefficients: variable `l` gets degrees 1 and 2, the
/// last variable also degree 0.
fn mask_coefficients(k: u32, coins: &[Fp3]) -> Result<Vec<[Fp3; 3]>> {
    if coins.len() != 2 * k as usize + 1 {
        return Err(Error::InvalidConfig);
    }
    let mut w = vec![[zero(); 3]; k as usize];
    let mut at = 0usize;
    for index in 0..k as usize {
        let last = index + 1 == k as usize;
        let first_degree = if last { 0 } else { 1 };
        for degree in first_degree..3 {
            w[index][degree] = coins[at];
            at += 1;
        }
    }
    Ok(w)
}

struct HidingRounds {
    w: Vec<[Fp3; 3]>,
    after: Vec<Fp3>,
    before: Fp3,
    index: u32,
    sum: Fp3,
}

impl HidingRounds {
    fn new(k: u32, coins: &[Fp3]) -> Result<Self> {
        let w = mask_coefficients(k, coins)?;
        let k = k as usize;
        let mut after = vec![zero(); k];
        for index in (0..k.saturating_sub(1)).rev() {
            let next = index + 1;
            let apply = two()
                .mul(&w[next][0])
                .add(&w[next][1])
                .add(&w[next][2]);
            after[index] = apply.add(&two().mul(&after[next]));
        }
        let first = two()
            .mul(&w[0][0])
            .add(&w[0][1])
            .add(&w[0][2]);
        let sum = first.add(&two().mul(&after[0]));
        Ok(Self {
            w,
            after,
            before: zero(),
            index: 0,
            sum,
        })
    }

    fn round(&self) -> Result<[Fp3; 3]> {
        if self.index as usize >= self.w.len() {
            return Err(Error::InvalidSessionState);
        }
        let mut q = self.w[self.index as usize];
        q[0] = q[0].add(&self.after[self.index as usize]);
        q[1] = q[1].add(&self.before);
        Ok(q)
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        if self.index as usize >= self.w.len() {
            return Err(Error::InvalidSessionState);
        }
        let w = self.w[self.index as usize];
        let value = w[0].add(&w[1].mul(&challenge)).add(&w[2].mul(&challenge).mul(&challenge));
        self.before = self.before.mul(&challenge).add(&value);
        self.index += 1;
        Ok(())
    }
}

/// The fresh mask's rows, in the layout `[row + 2^h * column]`.
pub fn fold_mask_table(shape: &FoldShape, coins: &[Fp3]) -> Result<Vec<Fp3>> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Fold);
    let k = shape.k as usize;
    let h = shape.h as usize;
    let rows = shape.mask_rows();
    let len = shape.len_zk;
    let w = mask_coefficients(shape.k, coins)?;
    let coeff = |l: usize, t: usize| w[l - 1][t];
    let mut table = vec![zero(); rows * len];
    for row in 0..rows {
        let at = |column: usize| row + rows * column;
        table[at(0)] = if h < k { coeff(h + 1, 1) } else { coeff(k, 0) };
        for l in (1..=h).rev() {
            if (row >> (l - 1)) & 1 == 0 {
                break;
            }
            table[at(0)] = table[at(0)].add(&coeff(l, 1));
            table[at(l)] = coeff(l, 2);
        }
        let mut column = h + 1;
        for l in (h + 1)..=k {
            let start = if l == h + 1 { 2 } else { 1 };
            for t in start..=2 {
                table[at(column)] = coeff(l, t);
                column += 1;
            }
        }
        if h < k {
            table[at(column)] = coeff(k, 0);
        }
    }
    Ok(table)
}

/// The public weights of the folded fresh mask: `U(g) = < u(g), fresh(g) >`.
pub fn fold_mask_weights(shape: &FoldShape, challenges: &[Fp3]) -> Result<Vec<Fp3>> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Fold);
    let k = shape.k as usize;
    let h = shape.h as usize;
    if challenges.len() != k {
        return Err(Error::ShapeMismatch);
    }
    let len = shape.len_zk;
    let mut after = vec![one(); k + 1];
    for l in (0..k).rev() {
        after[l] = after[l + 1].mul(&challenges[l]);
    }
    let mut u = vec![zero(); len];
    u[0] = after[h];
    for l in 1..=h {
        u[l] = challenges[l - 1].mul(&after[h]);
    }
    let mut column = h + 1;
    for l in (h + 1)..=k {
        let start = if l == h + 1 { 2 } else { 1 };
        for t in start..=2 {
            let power = if t == 1 {
                challenges[l - 1]
            } else {
                challenges[l - 1].mul(&challenges[l - 1])
            };
            u[column] = power.mul(&after[l]);
            column += 1;
        }
    }
    if h < k {
        u[column] = one();
    }
    Ok(u)
}

fn derive_salt(coins: &[Fp3], index: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(coins.len() * 24 + 8);
    for coin in coins {
        data.extend_from_slice(&coin.encode());
    }
    data.extend_from_slice(&index.to_le_bytes());
    let first = hash_labeled(tag_word(STATEMENT, 0), index, None, &data).unwrap();
    let second = hash_labeled(tag_word(STATEMENT, 1), index, None, &data).unwrap();
    let mut salt = Vec::with_capacity(40);
    salt.extend_from_slice(&first.0);
    salt.extend_from_slice(&second.0[..8]);
    salt
}

/// The committed fresh mask. The verifier binds its root in the first message
/// and reads it later only through openings.
pub fn fresh_mask_tree(shape: &FoldShape, coins: &[Fp3]) -> Result<ColumnTree> {
    let table = fold_mask_table(shape, coins)?;
    let rows = shape.mask_rows();
    let mut columns = Vec::with_capacity(shape.len_zk);
    for column in 0..shape.len_zk {
        let mut symbols = Vec::with_capacity(rows * 3);
        for row in 0..rows {
            let value = table[row + rows * column];
            for coefficient in value.c {
                symbols.push(coefficient.as_canonical_u64());
            }
        }
        columns.push((symbols, derive_salt(coins, column as u64)));
    }
    let identity = hash_labeled(
        tag_word(STATEMENT, 2),
        0,
        None,
        format!("fresh-mask-{}-{}-{}", shape.k, shape.h, shape.len_zk).as_bytes(),
    )?;
    let parameters = hash_labeled(tag_word(STATEMENT, 3), 0, None, &identity.0)?;
    let spec = ColumnTreeSpec::new(identity, parameters, 1, rows * 3, 40)?;
    build_column_tree(&spec, &columns)
}

#[derive(Clone, Debug)]
pub struct FoldClaim {
    pub main_rows: usize,
    pub main_cols: usize,
    pub main: Vec<Fp3>,
    pub coefficients_main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub coefficients_mask: Vec<Fp3>,
    pub kept: Fp3,
    pub target: Fp3,
}

impl FoldClaim {
    pub fn validate(&self, shape: &FoldShape) -> Result<()> {
        if self.main_rows != shape.main_rows()
            || self.mask.len() != shape.mask_rows() * shape.len_zk
            || self.coefficients_mask.len() != self.mask.len()
        {
            return Err(Error::ShapeMismatch);
        }
        if self.main.len() != self.main_rows * self.main_cols
            || self.coefficients_main.len() != self.main.len()
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

pub fn prove_fold(
    shape: &FoldShape,
    claim: &FoldClaim,
    coins: &[Fp3],
    chain: &mut Chain,
) -> Result<FoldReceipt> {
    claim.validate(shape)?;
    let fresh_tree = fresh_mask_tree(shape, coins)?;
    let fresh_root = fresh_tree.root;
    let mut hiding = HidingRounds::new(shape.k, coins)?;
    let mask_sum = hiding.sum;

    let mut first = chain.message();
    for word in fresh_root.0.chunks_exact(8) {
        first.push_u64(u64::from_le_bytes(word.try_into().map_err(|_| Error::InvalidEncoding)?));
    }
    for word in mask_sum.encode().chunks_exact(8) {
        first.push_u64(u64::from_le_bytes(word.try_into().map_err(|_| Error::InvalidEncoding)?));
    }
    let prefix = first.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let eps = chain.draw_e()?;

    let mut running = mask_sum.add(&eps.mul(&claim.target));
    let mut main = claim.main.clone();
    let mut coefficients_main = claim.coefficients_main.clone();
    let mut mask = claim.mask.clone();
    let mut coefficients_mask = claim.coefficients_mask.clone();
    let mut fresh = fold_mask_table(shape, coins)?;
    let mut kept = claim.kept;
    let mut later = one();
    let mut rounds = Vec::with_capacity(shape.k as usize);

    for variable in 0..shape.k {
        let main_part = pair_products(&coefficients_main, &main)?;
        let mask_part = if variable < shape.h {
            pair_products(&coefficients_mask, &mask)?
        } else {
            let value = inner(&coefficients_mask, &mask)?;
            [zero(), value.mul(&later), zero()]
        };
        let mut q = hiding.round()?;
        for degree in 0..3 {
            q[degree] = q[degree].add(&eps.mul(&main_part[degree].add(&mask_part[degree])));
        }
        q[0] = q[0].add(&eps.mul(&kept));
        q[1] = q[1].sub(&eps.mul(&kept));
        if polynomial_at(&q, zero()).add(&polynomial_at(&q, one())) != running {
            return Err(Error::InvalidProof);
        }

        let mut message = chain.message();
        for coefficient in q {
            for word in coefficient.encode().chunks_exact(8) {
                message.push_u64(u64::from_le_bytes(
                    word.try_into().map_err(|_| Error::InvalidEncoding)?,
                ));
            }
        }
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let challenge = chain.draw_e()?;
        rounds.push(FoldRound {
            polynomial: q,
            challenge,
        });
        running = polynomial_at(&q, challenge);

        fold_low_once(&mut main, challenge)?;
        fold_low_once(&mut coefficients_main, challenge)?;
        if variable < shape.h {
            fold_low_once(&mut mask, challenge)?;
            fold_low_once(&mut coefficients_mask, challenge)?;
            fold_low_once(&mut fresh, challenge)?;
        } else {
            later = later.mul(&challenge);
        }
        kept = kept.mul(&one().sub(&challenge));
        hiding.bind(challenge)?;
    }

    let later_scale = eps.mul(&later);
    for coefficient in &mut coefficients_mask {
        *coefficient = coefficient.mul(&later_scale);
    }
    for coefficient in &mut coefficients_main {
        *coefficient = coefficient.mul(&eps);
    }
    kept = kept.mul(&eps);

    Ok(FoldReceipt {
        shape: *shape,
        fresh_root,
        mask_sum,
        rounds,
        output: FoldOutput {
            main,
            mask,
            fresh,
            coefficients_main,
            coefficients_mask,
            kept,
            target: running,
        },
    })
}

/// Verify a receipt against its public target. The chain is replayed from the
/// receipt's messages; a changed message changes the challenges.
pub fn verify_fold(
    shape: &FoldShape,
    target: Fp3,
    receipt: &FoldReceipt,
    chain: &mut Chain,
) -> Result<()> {
    if receipt.shape != *shape || receipt.rounds.len() != shape.k as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut first = chain.message();
    for word in receipt.fresh_root.0.chunks_exact(8) {
        first.push_u64(u64::from_le_bytes(word.try_into().map_err(|_| Error::InvalidEncoding)?));
    }
    for word in receipt.mask_sum.encode().chunks_exact(8) {
        first.push_u64(u64::from_le_bytes(word.try_into().map_err(|_| Error::InvalidEncoding)?));
    }
    let prefix = first.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let eps = chain.draw_e()?;
    let mut running = receipt.mask_sum.add(&eps.mul(&target));

    let mut challenges = Vec::with_capacity(shape.k as usize);
    for round in &receipt.rounds {
        if polynomial_at(&round.polynomial, zero())
            .add(&polynomial_at(&round.polynomial, one()))
            != running
        {
            return Err(Error::InvalidProof);
        }
        let mut message = chain.message();
        for coefficient in round.polynomial {
            for word in coefficient.encode().chunks_exact(8) {
                message.push_u64(u64::from_le_bytes(
                    word.try_into().map_err(|_| Error::InvalidEncoding)?,
                ));
            }
        }
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let challenge = chain.draw_e()?;
        if challenge != round.challenge {
            return Err(Error::InvalidProof);
        }
        running = polynomial_at(&round.polynomial, challenge);
        challenges.push(challenge);
    }
    if receipt.output.target != running {
        return Err(Error::InvalidProof);
    }
    let weights = fold_mask_weights(shape, &challenges)?;
    let terminal = inner(&receipt.output.coefficients_main, &receipt.output.main)?
        .add(&inner(&receipt.output.coefficients_mask, &receipt.output.mask)?)
        .add(&inner(&weights, &receipt.output.fresh)?)
        .add(&receipt.output.kept);
    if terminal != running {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// OOD answer for a folded row message, used by the round that follows FOLD.
pub fn fold_ood_answer(
    plan: &crate::pcs::request::code::CodeParams,
    message: &[Fp3],
    tail: &[Fp3],
    point: Fp3,
    mask: Fp3,
) -> Result<Fp3> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Fold);
    Ok(evaluate_ood_fp3(plan, message, tail, point)?.add(&mask))
}
