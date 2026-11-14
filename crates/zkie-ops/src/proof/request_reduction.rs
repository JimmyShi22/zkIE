//! The token sumcheck of a request's relation reduction.
//!
//! A request's relation is reduced over its token positions first: the claim
//! `sum_t eq(P, t) s(t)` runs over the token cube at the point `P`, with `s`
//! the token map. The map's end value at the challenged tokens is sent masked;
//! the caller's later rounds and end know what the end value is.
//!
//! Token bit `j` is the bit of weight `j` (`(t >> j) & 1`), and its equality
//! factor is `(1 - P[j])` or `P[j]`. The sumcheck's variables carry exactly
//! those equality weights, in bit order.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::psc::Weight;
use crate::proof::psc_general::{
    prove_general, verify_general, GeneralChallenges, GeneralMaskEntries, GeneralOutput,
    GeneralProof, MaskTerm, Summand, WeightedMapInner,
};

/// The token claim: the equality-weighted sum of the token map.
pub fn token_claim(point: &[Fp3], values: &[Fp3]) -> Result<Fp3> {
    if point.len() >= 32 || values.len() != (1usize << point.len()) {
        return Err(Error::InvalidConfig);
    }
    let mut sum = Fp3::ZERO;
    for (index, value) in values.iter().enumerate() {
        let mut factor = Fp3::ONE;
        for (bit, coordinate) in point.iter().enumerate() {
            let chosen = if (index >> bit) & 1 == 1 {
                *coordinate
            } else {
                Fp3::ONE.sub(coordinate)
            };
            factor = factor.mul(&chosen);
        }
        sum = sum.add(&factor.mul(value));
    }
    Ok(sum)
}

/// The inner-dimension claim: the equality-weighted product of two maps at an
/// inner point.
pub fn inner_claim(point: &[Fp3], left: &[Fp3], right: &[Fp3]) -> Result<Fp3> {
    if point.len() >= 32
        || left.len() != (1usize << point.len())
        || right.len() != left.len()
    {
        return Err(Error::InvalidConfig);
    }
    let mut sum = Fp3::ZERO;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        let mut factor = Fp3::ONE;
        for (bit, coordinate) in point.iter().enumerate() {
            let chosen = if (index >> bit) & 1 == 1 {
                *coordinate
            } else {
                Fp3::ONE.sub(coordinate)
            };
            factor = factor.mul(&chosen);
        }
        sum = sum.add(&factor.mul(left).mul(right));
    }
    Ok(sum)
}

/// The inner sumcheck's variables: one equality weight per inner bit, in bit
/// order.
pub fn inner_variables(point: &[Fp3]) -> Result<Vec<Weight>> {
    if point.is_empty() || point.len() >= 32 {
        return Err(Error::InvalidConfig);
    }
    point
        .iter()
        .map(|coordinate| Weight::equality(*coordinate, 3))
        .collect()
}

/// Proves the inner-dimension claim over the two maps.
#[allow(clippy::too_many_arguments)]
pub fn prove_inner(
    point: &[Fp3],
    left: &[Fp3],
    right: &[Fp3],
    claim_masks: &[MaskTerm],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
    shown: Fp3,
    masks: &GeneralMaskEntries,
    challenges: &mut dyn GeneralChallenges,
) -> Result<(GeneralProof, GeneralOutput)> {
    let variables = inner_variables(point)?;
    let mut inner = crate::proof::psc_general::WeightedProductInner::new(left, right, point)?;
    prove_general(
        &variables,
        claim_masks,
        mask_value,
        shown,
        &Summand::Product,
        None,
        &mut inner,
        masks,
        challenges,
    )
}

/// Verifies the inner-dimension claim and returns its output claim.
pub fn verify_inner(
    point: &[Fp3],
    claim_masks: &[MaskTerm],
    shown: Fp3,
    proof: &GeneralProof,
    hiding_entries: usize,
    challenges: &mut dyn GeneralChallenges,
) -> Result<GeneralOutput> {
    let variables = inner_variables(point)?;
    verify_general(
        &variables,
        claim_masks,
        shown,
        &Summand::Product,
        None,
        proof,
        hiding_entries,
        challenges,
    )
}

/// The token sumcheck's variables: one equality weight per token bit, in bit
/// order.
pub fn token_variables(point: &[Fp3]) -> Result<Vec<Weight>> {
    if point.is_empty() || point.len() >= 32 {
        return Err(Error::InvalidConfig);
    }
    point
        .iter()
        .map(|coordinate| Weight::equality(*coordinate, 3))
        .collect()
}

/// Proves the token claim over the token map. `shown` is the claim as it was
/// sent (the hidden claim plus its mask entries); `claim_masks` names those
/// entries, and the sumcheck's end value is masked by its last challenge.
pub fn prove_token(
    point: &[Fp3],
    values: &[Fp3],
    claim_masks: &[MaskTerm],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
    shown: Fp3,
    masks: &GeneralMaskEntries,
    challenges: &mut dyn GeneralChallenges,
) -> Result<(GeneralProof, GeneralOutput)> {
    let variables = token_variables(point)?;
    let mut inner = WeightedMapInner::new(values, point)?;
    prove_general(
        &variables,
        claim_masks,
        mask_value,
        shown,
        &Summand::Map,
        None,
        &mut inner,
        masks,
        challenges,
    )
}

/// Verifies the token claim and returns its output claim.
#[allow(clippy::too_many_arguments)]
pub fn verify_token(
    point: &[Fp3],
    claim_masks: &[MaskTerm],
    shown: Fp3,
    proof: &GeneralProof,
    hiding_entries: usize,
    challenges: &mut dyn GeneralChallenges,
) -> Result<GeneralOutput> {
    let variables = token_variables(point)?;
    verify_general(
        &variables,
        claim_masks,
        shown,
        &Summand::Map,
        None,
        proof,
        hiding_entries,
        challenges,
    )
}

/// The kept command's claim: the equality-weighted product of the shown map
/// and the kept map.
pub fn token_kept_claim(point: &[Fp3], shown: &[Fp3], kept: &[Fp3]) -> Result<Fp3> {
    inner_claim(point, shown, kept)
}

/// Proves the kept token claim. The kept map's value is not sent: the output
/// leaves it with the coefficient of the shown end value as it was sent.
#[allow(clippy::too_many_arguments)]
pub fn prove_token_kept(
    point: &[Fp3],
    shown: &[Fp3],
    kept: &[Fp3],
    claim_masks: &[MaskTerm],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
    shown_claim: Fp3,
    masks: &GeneralMaskEntries,
    challenges: &mut dyn GeneralChallenges,
) -> Result<(GeneralProof, GeneralOutput)> {
    let variables = token_variables(point)?;
    let mut inner = crate::proof::psc_general::WeightedProductInner::new(shown, kept, point)?;
    prove_general(
        &variables,
        claim_masks,
        mask_value,
        shown_claim,
        &Summand::ShownKept,
        None,
        &mut inner,
        masks,
        challenges,
    )
}

/// Verifies the kept token claim and returns its output claims.
pub fn verify_token_kept(
    point: &[Fp3],
    claim_masks: &[MaskTerm],
    shown_claim: Fp3,
    proof: &GeneralProof,
    hiding_entries: usize,
    challenges: &mut dyn GeneralChallenges,
) -> Result<GeneralOutput> {
    let variables = token_variables(point)?;
    verify_general(
        &variables,
        claim_masks,
        shown_claim,
        &Summand::ShownKept,
        None,
        proof,
        hiding_entries,
        challenges,
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
        fn send_round(
            &mut self,
            _index: usize,
            _round: &crate::proof::psc::PscRound,
        ) -> Result<()> {
            Ok(())
        }
        fn send_end_values(&mut self, _values: &[Fp3]) -> Result<()> {
            Ok(())
        }
    }

    fn fp(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    fn eq_factor(coordinate: Fp3, bit: u64) -> Fp3 {
        if bit == 1 {
            coordinate
        } else {
            Fp3::ONE.sub(&coordinate)
        }
    }

    #[test]
    fn the_token_claim_is_the_equality_weighted_sum() {
        let point = [fp(2), fp(3)];
        let values = [fp(1), fp(2), fp(3), fp(4)];
        let mut expected = Fp3::ZERO;
        for (index, value) in values.iter().enumerate() {
            expected = expected.add(
                &eq_factor(point[0], index as u64 & 1)
                    .mul(&eq_factor(point[1], (index as u64 >> 1) & 1))
                    .mul(value),
            );
        }
        assert_eq!(token_claim(&point, &values).unwrap(), expected);
        assert!(token_claim(&point, &values[..3]).is_err());
        assert!(token_variables(&[]).is_err());
    }

    #[test]
    fn the_token_sumcheck_proves_and_verifies_the_claim() {
        let point = [fp(2), fp(3)];
        let values = [fp(1), fp(2), fp(3), fp(4)];
        let claim = token_claim(&point, &values).unwrap();
        let variables = token_variables(&point).unwrap();
        let hiding_count =
            crate::proof::psc::hp_entries(&crate::proof::psc_general::general_variables(
                &variables,
                &Summand::Map,
            )
            .unwrap())
            .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(700 + index as u64)).collect(),
            shown: vec![fp(800)],
        };
        let (proof, proven) = prove_token(
            &point,
            &values,
            &[],
            &|_| Ok(Fp3::ZERO),
            claim,
            &masks,
            &mut Source(23),
        )
        .unwrap();
        let verified = verify_token(
            &point,
            &[],
            claim,
            &proof,
            masks.hiding.len(),
            &mut Source(23),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.point.len(), point.len());
        // A changed round leaves another output claim.
        let mut changed = proof.clone();
        changed.rounds[0].coefficients[0] =
            changed.rounds[0].coefficients[0].add(&Fp3::ONE);
        let tampered = verify_token(
            &point,
            &[],
            claim,
            &changed,
            masks.hiding.len(),
            &mut Source(23),
        )
        .unwrap();
        assert_ne!(tampered, verified);
    }

    #[test]
    fn the_inner_claim_is_the_equality_weighted_product() {
        let point = [fp(2), fp(3)];
        let left = [fp(1), fp(2), fp(3), fp(4)];
        let right = [fp(5), fp(6), fp(7), fp(8)];
        let mut expected = Fp3::ZERO;
        for index in 0..4 {
            expected = expected.add(
                &eq_factor(point[0], index as u64 & 1)
                    .mul(&eq_factor(point[1], (index as u64 >> 1) & 1))
                    .mul(&left[index])
                    .mul(&right[index]),
            );
        }
        assert_eq!(inner_claim(&point, &left, &right).unwrap(), expected);
        assert!(inner_claim(&point, &left, &right[..3]).is_err());
        let claim = expected;
        let variables = inner_variables(&point).unwrap();
        let hiding_count =
            crate::proof::psc::hp_entries(&crate::proof::psc_general::general_variables(
                &variables,
                &Summand::Product,
            )
            .unwrap())
            .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(850 + index as u64)).collect(),
            shown: vec![fp(940), fp(941)],
        };
        let (proof, proven) = prove_inner(
            &point,
            &left,
            &right,
            &[],
            &|_| Ok(Fp3::ZERO),
            claim,
            &masks,
            &mut Source(31),
        )
        .unwrap();
        let verified = verify_inner(
            &point,
            &[],
            claim,
            &proof,
            masks.hiding.len(),
            &mut Source(31),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.ends.len(), 2);
        // A pair whose product sum does not match the claim is refused.
        assert!(prove_inner(
            &point,
            &left,
            &[fp(9), fp(8), fp(7), fp(6)],
            &[],
            &|_| Ok(Fp3::ZERO),
            claim,
            &masks,
            &mut Source(31),
        )
        .is_err());
    }

    #[test]
    fn the_claim_of_another_map_does_not_match() {
        let point = [fp(5)];
        let values = [fp(1), fp(2)];
        let claim = token_claim(&point, &values).unwrap();
        let variables = token_variables(&point).unwrap();
        let hiding_count =
            crate::proof::psc::hp_entries(&crate::proof::psc_general::general_variables(
                &variables,
                &Summand::Map,
            )
            .unwrap())
            .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(900 + index as u64)).collect(),
            shown: vec![fp(901)],
        };
        // The prover runs on another map; its first round does not match the
        // claim, so the sumcheck refuses before any message is accepted.
        assert!(prove_token(
            &point,
            &[fp(3), fp(4)],
            &[],
            &|_| Ok(Fp3::ZERO),
            claim,
            &masks,
            &mut Source(29),
        )
        .is_err());
    }

    #[test]
    fn the_kept_token_sumcheck_leaves_the_kept_coefficient() {
        let point = [fp(2), fp(3)];
        let shown = [fp(1), fp(2), fp(3), fp(4)];
        let kept = [fp(5), fp(6), fp(7), fp(8)];
        let claim = token_kept_claim(&point, &shown, &kept).unwrap();
        let variables = token_variables(&point).unwrap();
        let hiding_count =
            crate::proof::psc::hp_entries(&crate::proof::psc_general::general_variables(
                &variables,
                &Summand::ShownKept,
            )
            .unwrap())
            .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(700 + index as u64)).collect(),
            shown: vec![fp(800)],
        };
        let (proof, proven) = prove_token_kept(
            &point,
            &shown,
            &kept,
            &[],
            &|_| Ok(Fp3::ZERO),
            claim,
            &masks,
            &mut Source(43),
        )
        .unwrap();
        let verified = verify_token_kept(
            &point,
            &[],
            claim,
            &proof,
            masks.hiding.len(),
            &mut Source(43),
        )
        .unwrap();
        assert_eq!(proven, verified);
        // Only the shown end value is sent; the kept map stays a coefficient.
        assert_eq!(proven.ends.len(), 1);
        assert_eq!(proven.row_maps.len(), 1);
        assert_eq!(proven.row_maps[0].0, 0);
        // The coefficient is eps times the shown end value as it was sent.
        let mut source = Source(43);
        let eps = source.draw().unwrap();
        assert_eq!(
            proven.row_maps[0].1,
            eps.mul(&proven.ends[0].value)
        );
        // A changed round leaves another output claim.
        let mut changed = proof.clone();
        changed.rounds[0].coefficients[0] =
            changed.rounds[0].coefficients[0].add(&Fp3::ONE);
        let tampered = verify_token_kept(
            &point,
            &[],
            claim,
            &changed,
            masks.hiding.len(),
            &mut Source(43),
        )
        .unwrap();
        assert_ne!(tampered, verified);
        // Another kept map does not sum to the claim.
        assert!(prove_token_kept(
            &point,
            &shown,
            &[fp(9), fp(8), fp(7), fp(6)],
            &[],
            &|_| Ok(Fp3::ZERO),
            claim,
            &masks,
            &mut Source(43),
        )
        .is_err());
    }
}
