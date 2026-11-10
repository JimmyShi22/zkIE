//! CS/BASE local algebraic component: local BASE algebra.
//!
//! The prover sends target* and the plaintext M'/R' rows. The verifier checks
//! the linear relation and the sampled symbol relations against authenticated
//! openings. Commitments, SEND and opening verification belong to the isolated
//! session; this module never treats a public target or a fixture row as
//! authenticated.
//!
//! This is a local algebraic component, not a complete BASE, IOPP or proof
//! acceptance. The fixture is not claimed to satisfy iopp_fault parameters, and
//! the real SEND sample declaration and its follow-up checks remain unclosed.

use zkie_core::error::{Error, Result};

use crate::pcs::request::field::Fp3;
use crate::pcs::request::proximity::SingleRows;

/// The prover's BASE message and plaintext rows. M' and R' are deliberately
/// not committed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseRows {
    pub target_star: Fp3,
    pub lambda: Fp3,
    pub main_prime: Vec<Fp3>,
    pub main_tail_prime: Vec<Fp3>,
    pub mask_prime: Vec<Fp3>,
    pub mask_tail_prime: Vec<Fp3>,
}

fn inner(left: &[Fp3], right: &[Fp3]) -> Result<Fp3> {
    if left.len() != right.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (a, b) in left.iter().zip(right) {
        sum = sum.add(&a.mul(b));
    }
    Ok(sum)
}

fn combine(star: &[Fp3], lambda: Fp3, row: &[Fp3]) -> Result<Vec<Fp3>> {
    if star.len() != row.len() {
        return Err(Error::ShapeMismatch);
    }
    Ok(star
        .iter()
        .zip(row)
        .map(|(star, row)| star.add(&lambda.mul(row)))
        .collect())
}

/// Prover-side: builds target* and M'/R' from the final plaintext rows. The
/// prover must commit the star row and send target*/its commitment before the
/// verifier draws lambda; this function only performs algebra.
pub fn prepare(
    rows: &SingleRows,
    star_main: &[Fp3],
    star_tail: &[Fp3],
    star_mask: &[Fp3],
    star_mask_tail: &[Fp3],
    lambda: Fp3,
) -> Result<BaseRows> {
    rows.base_ready()?;
    if rows.masks.len() != 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    if star_main.len() != rows.main.len()
        || star_tail.len() != rows.main_tail.len()
        || star_mask.len() != rows.masks[0].len()
        || star_mask_tail.len() != rows.mask_tails[0].len()
    {
        return Err(Error::ShapeMismatch);
    }
    let target_star = inner(&rows.claim, star_main)?
        .add(&inner(&rows.mask_claims[0], star_mask)?);
    Ok(BaseRows {
        target_star,
        lambda,
        main_prime: combine(star_main, lambda, &rows.main)?,
        main_tail_prime: combine(star_tail, lambda, &rows.main_tail)?,
        mask_prime: combine(star_mask, lambda, &rows.masks[0])?,
        mask_tail_prime: combine(star_mask_tail, lambda, &rows.mask_tails[0])?,
    })
}

/// Verifier-side: the final linear check, using the verifier's own combined
/// claim and target, not the prover's claim values.
pub fn linear_holds(
    claim: &[Fp3],
    mask_claim: &[Fp3],
    target: Fp3,
    prepared: &BaseRows,
) -> Result<bool> {
    if claim.len() != prepared.main_prime.len() || mask_claim.len() != prepared.mask_prime.len() {
        return Err(Error::ShapeMismatch);
    }
    let main = inner(claim, &prepared.main_prime)?;
    let mask = inner(mask_claim, &prepared.mask_prime)?;
    Ok(main.add(&mask) == prepared.target_star.add(&prepared.lambda.mul(&target)))
}

/// The symbol check at one position. `encoded_prime` is the symbol of
/// Enc(M',R') and `star_symbol`/`old_symbol` are the authenticated symbols of
/// the masking and old rows.
pub fn symbol_holds(
    encoded_prime: Fp3,
    star_symbol: Fp3,
    lambda: Fp3,
    old_symbol: Fp3,
) -> bool {
    encoded_prime == star_symbol.add(&lambda.mul(&old_symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::code::encode_row_fp3;

    fn rows() -> SingleRows {
        let code = build_code_plan(3, 4, 1).unwrap();
        SingleRows {
            code,
            mask_code: code,
            main: vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3), Fp3::from_u64(4)],
            main_tail: vec![Fp3::from_u64(5)],
            masks: vec![vec![Fp3::from_u64(6), Fp3::from_u64(7), Fp3::from_u64(8), Fp3::from_u64(9)]],
            mask_tails: vec![vec![Fp3::from_u64(10)]],
            claim: vec![Fp3::from_u64(1), Fp3::ZERO, Fp3::ZERO, Fp3::ZERO],
            mask_claims: vec![vec![Fp3::ZERO, Fp3::from_u64(1), Fp3::ZERO, Fp3::ZERO]],
            target: Fp3::from_u64(8),
        }
    }

    #[test]
    fn base_linear_and_symbol_checks() {
        let rows = rows();
        let star_main = vec![Fp3::from_u64(11); 4];
        let star_tail = vec![Fp3::from_u64(12)];
        let star_mask = vec![Fp3::from_u64(13); 4];
        let star_mask_tail = vec![Fp3::from_u64(14)];
        let lambda = Fp3::from_u64(3);
        let prepared = prepare(
            &rows,
            &star_main,
            &star_tail,
            &star_mask,
            &star_mask_tail,
            lambda,
        )
        .unwrap();
        assert!(linear_holds(&rows.claim, &rows.mask_claims[0], rows.target, &prepared).unwrap());
        let encoded = encode_row_fp3(&rows.code, &prepared.main_prime, &prepared.main_tail_prime)
            .unwrap();
        let old = encode_row_fp3(&rows.code, &rows.main, &rows.main_tail).unwrap();
        let star = encode_row_fp3(&rows.code, &star_main, &star_tail).unwrap();
        assert!(symbol_holds(encoded[0], star[0], lambda, old[0]));
        let mut bad_target = prepared.clone();
        bad_target.target_star = bad_target.target_star.add(&Fp3::ONE);
        assert!(!linear_holds(&rows.claim, &rows.mask_claims[0], rows.target, &bad_target).unwrap());
        assert!(!symbol_holds(encoded[0].add(&Fp3::ONE), star[0], lambda, old[0]));
        let mut wrong_claim = rows.claim.clone();
        wrong_claim[0] = wrong_claim[0].add(&Fp3::ONE);
        assert!(!linear_holds(&wrong_claim, &rows.mask_claims[0], rows.target, &prepared).unwrap());
    }
}
