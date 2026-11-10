//! Isolated code-switch claim registration.
//!
//! This module implements the algebraic core of the isolated CS stage: the
//! old single rows are placed in new-code messages/tails, the old symbols are
//! read, and the original and generator-position claims are registered. It
//! does not prove that the new message is a correct re-encoding; that is the
//! later proximity test's task. No kept-row path is exercised in this subset.

use zkie_core::error::{Error, Result};

use crate::pcs::request::code::{encode_row_fp3, CodeParams};
use crate::pcs::request::field::Fp3;
use crate::pcs::request::proximity::SingleRows;

/// One registered linear claim on the new-code main/mask messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchClaim {
    pub main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub target: Fp3,
}

/// The switched messages and registered claims. This is not a proof that the
/// new encoding is correct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchedRows {
    pub main: Vec<Fp3>,
    pub main_tail: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub mask_tail: Vec<Fp3>,
    pub positions: Vec<u64>,
    pub old_symbols: Vec<Fp3>,
    pub claims: Vec<SwitchClaim>,
    pub theta: Fp3,
    pub combined: SwitchClaim,
}

fn coeff_on_message(code: &CodeParams, position: u64, length: usize) -> Result<Vec<Fp3>> {
    let mut out = vec![Fp3::ZERO; length];
    let zeros = vec![Fp3::ZERO; code.r as usize];
    for index in 0..length {
        let mut message = vec![Fp3::ZERO; length];
        message[index] = Fp3::ONE;
        let encoded = encode_row_fp3(code, &message, &zeros)?;
        out[index] = *encoded.get(position as usize).ok_or(Error::InvalidIndex)?;
    }
    Ok(out)
}

fn coeff_on_tail(code: &CodeParams, position: u64) -> Result<Vec<Fp3>> {
    let message = vec![Fp3::ZERO; code.ell as usize];
    let mut out = vec![Fp3::ZERO; code.r as usize];
    for index in 0..code.r as usize {
        let mut tail = vec![Fp3::ZERO; code.r as usize];
        tail[index] = Fp3::ONE;
        let encoded = encode_row_fp3(code, &message, &tail)?;
        out[index] = *encoded.get(position as usize).ok_or(Error::InvalidIndex)?;
    }
    Ok(out)
}

fn add_scaled(target: &mut [Fp3], source: &[Fp3], scale: Fp3) -> Result<()> {
    if target.len() != source.len() {
        return Err(Error::ShapeMismatch);
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target = target.add(&scale.mul(source));
    }
    Ok(())
}

/// Registers the original claim and one generator-position claim per old
/// position, then combines them with the supplied theta. The returned claims
/// are pending: registration is not verification, and a false claim is for
/// the later proximity test to catch.
pub fn register_switch(
    rows: &SingleRows,
    new_code: CodeParams,
    new_mask_code: CodeParams,
    positions: &[u64],
    theta: Fp3,
) -> Result<SwitchedRows> {
    rows.base_ready()?;
    if rows.masks.len() != 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    if new_code.ell < rows.code.ell + rows.code.r
        || new_mask_code.ell < rows.mask_code.ell + rows.mask_code.r
    {
        return Err(Error::ShapeMismatch);
    }
    let old_main = encode_row_fp3(&rows.code, &rows.main, &rows.main_tail)?;
    let old_mask = encode_row_fp3(
        &rows.mask_code,
        &rows.masks[0],
        &rows.mask_tails[0],
    )?;
    let mut main = vec![Fp3::ZERO; new_code.ell as usize];
    let main_tail = vec![Fp3::ZERO; new_code.r as usize];
    let mut mask = vec![Fp3::ZERO; new_mask_code.ell as usize];
    let mask_tail = vec![Fp3::ZERO; new_mask_code.r as usize];
    for (to, from) in main.iter_mut().zip(&rows.main) {
        *to = *from;
    }
    for (to, from) in main.iter_mut().skip(rows.code.ell as usize).zip(&rows.main_tail) {
        *to = *from;
    }
    for (to, from) in mask.iter_mut().zip(&rows.masks[0]) {
        *to = *from;
    }
    for (to, from) in mask.iter_mut().skip(rows.mask_code.ell as usize).zip(&rows.mask_tails[0]) {
        *to = *from;
    }

    let mut claims = Vec::new();
    let mut original_main = vec![Fp3::ZERO; new_code.ell as usize];
    let mut original_mask = vec![Fp3::ZERO; new_mask_code.ell as usize];
    for (to, from) in original_main.iter_mut().zip(&rows.claim) {
        *to = *from;
    }
    for (to, from) in original_mask.iter_mut().zip(&rows.mask_claims[0]) {
        *to = *from;
    }
    claims.push(SwitchClaim {
        main: original_main,
        mask: original_mask,
        target: rows.target,
    });

    let mut old_symbols = Vec::with_capacity(positions.len());
    for position in positions {
        let main_position = usize::try_from(*position).map_err(|_| Error::ResourceLimit)?;
        let symbol = *old_main.get(main_position).ok_or(Error::InvalidIndex)?;
        old_symbols.push(symbol);
        let mask_position = usize::try_from(rows.mask_position(*position))
            .map_err(|_| Error::ResourceLimit)?;
        let mask_symbol = *old_mask.get(mask_position).ok_or(Error::InvalidIndex)?;
        let mut main_coeff = vec![Fp3::ZERO; new_code.ell as usize];
        let mut mask_coeff = vec![Fp3::ZERO; new_mask_code.ell as usize];
        let main_coeffs = coeff_on_message(&rows.code, *position, rows.code.ell as usize)?;
        let main_tail_coeffs = coeff_on_tail(&rows.code, *position)?;
        for (to, from) in main_coeff.iter_mut().zip(&main_coeffs) {
            *to = *from;
        }
        for (to, from) in main_coeff
            .iter_mut()
            .skip(rows.code.ell as usize)
            .zip(&main_tail_coeffs)
        {
            *to = *from;
        }
        let mask_coeffs = coeff_on_message(
            &rows.mask_code,
            rows.mask_position(*position),
            rows.mask_code.ell as usize,
        )?;
        let mask_tail_coeffs = coeff_on_tail(&rows.mask_code, rows.mask_position(*position))?;
        for (to, from) in mask_coeff.iter_mut().zip(&mask_coeffs) {
            *to = *from;
        }
        for (to, from) in mask_coeff
            .iter_mut()
            .skip(rows.mask_code.ell as usize)
            .zip(&mask_tail_coeffs)
        {
            *to = *from;
        }
        claims.push(SwitchClaim {
            main: main_coeff,
            mask: mask_coeff,
            target: symbol.add(&mask_symbol),
        });
    }

    let mut combined_main = vec![Fp3::ZERO; new_code.ell as usize];
    let mut combined_mask = vec![Fp3::ZERO; new_mask_code.ell as usize];
    let mut combined_target = Fp3::ZERO;
    let mut power = Fp3::ONE;
    for claim in &claims {
        add_scaled(&mut combined_main, &claim.main, power)?;
        add_scaled(&mut combined_mask, &claim.mask, power)?;
        combined_target = combined_target.add(&power.mul(&claim.target));
        power = power.mul(&theta);
    }
    Ok(SwitchedRows {
        main,
        main_tail,
        mask,
        mask_tail,
        positions: positions.to_vec(),
        old_symbols,
        claims,
        theta,
        combined: SwitchClaim {
            main: combined_main,
            mask: combined_mask,
            target: combined_target,
        },
    })
}

/// The last-stage handoff. Only the no-kept-row subset is represented; the
/// handoff is explicit and checked by `SingleRows::base_ready`.
pub fn switched_to_single_rows(
    switched: &SwitchedRows,
    code: CodeParams,
    mask_code: CodeParams,
) -> Result<SingleRows> {
    let rows = SingleRows {
        code,
        mask_code,
        main: switched.main.clone(),
        main_tail: switched.main_tail.clone(),
        masks: vec![switched.mask.clone()],
        mask_tails: vec![switched.mask_tail.clone()],
        claim: switched.combined.main.clone(),
        mask_claims: vec![switched.combined.mask.clone()],
        target: switched.combined.target,
    };
    rows.base_ready()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::proximity::SingleRows;

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
            target: Fp3::from_u64(1),
        }
    }

    #[test]
    fn registration_is_deterministic_and_pending() {
        let rows = rows();
        let new_code = build_code_plan(3, 6, 1).unwrap();
        let first = register_switch(&rows, new_code, new_code, &[1, 2], Fp3::from_u64(7)).unwrap();
        let second = register_switch(&rows, new_code, new_code, &[1, 2], Fp3::from_u64(7)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.claims.len(), 3);
        assert_eq!(first.old_symbols.len(), 2);
        let final_rows = switched_to_single_rows(&first, new_code, new_code).unwrap();
        assert!(final_rows.base_ready().is_ok());
    }

    #[test]
    fn wrong_code_capacity_or_shape_is_rejected() {
        let rows = rows();
        let small = build_code_plan(2, 2, 1).unwrap();
        assert!(register_switch(&rows, small, rows.mask_code, &[1], Fp3::ONE).is_err());
        let mut bad = rows;
        bad.main.pop();
        assert!(bad.base_ready().is_err());
    }
}
