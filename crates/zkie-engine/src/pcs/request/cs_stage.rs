//! Isolated code-switch stage.
//!
//! The prover side builds the switched message. The verifier side registers
//! the original claim and one generator-position claim per authenticated old
//! opening, then combines them by powers of theta. Registration is not a proof
//! that the switched message is a correct re-encoding; that is the later
//! proximity test's task. No kept-row path is exercised in this subset.

use zkie_core::error::{Error, Result};

use crate::pcs::request::code::{encode_row_fp3, CodeParams};
use crate::pcs::request::field::Fp3;
use crate::pcs::request::proximity::{ProximityStatement, SingleRows};

/// One registered linear claim on the new-code main/mask messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchClaim {
    pub main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub target: Fp3,
}

/// One authenticated old-row opening at a drawn position. The symbol values
/// come from `verify_opening`; the coordinate is the drawn position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthenticatedSymbols {
    pub position: u64,
    pub main: Fp3,
    pub mask: Fp3,
}

/// The prover's switched message. It is prover material; the verifier never
/// receives these values, only the commitment to their encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchedMessage {
    pub main: Vec<Fp3>,
    pub main_tail: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub mask_tail: Vec<Fp3>,
}

/// The verifier-side registration output. There is no plaintext message here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchedClaims {
    pub positions: Vec<u64>,
    pub old_symbols: Vec<Fp3>,
    pub claims: Vec<SwitchClaim>,
    pub theta: Fp3,
    pub combined: SwitchClaim,
    pub main_len: u64,
    pub main_tail_len: u64,
    pub mask_len: u64,
    pub mask_tail_len: u64,
}

impl SwitchedClaims {
    pub fn check(&self, code: CodeParams, mask_code: CodeParams) -> Result<()> {
        if self.main_len != code.ell
            || self.main_tail_len != code.r
            || self.mask_len != mask_code.ell
            || self.mask_tail_len != mask_code.r
            || self.combined.main.len() as u64 != self.main_len
            || self.combined.mask.len() as u64 != self.mask_len
        {
            return Err(Error::ShapeMismatch);
        }
        if self.positions.len() != self.old_symbols.len() {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    /// The isolated no-kept-row handoff is ready when the register shapes fit
    /// the final codes. The second mask row and kept rows are unsupported.
    pub fn base_ready(&self, code: CodeParams, mask_code: CodeParams) -> Result<()> {
        self.check(code, mask_code)
    }
}

fn switched_lengths(
    rows: &SingleRows,
    new_code: CodeParams,
    new_mask_code: CodeParams,
) -> Result<(usize, usize, usize, usize)> {
    if rows.masks.len() != 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    if new_code.ell < rows.code.ell + rows.code.r
        || new_mask_code.ell < rows.mask_code.ell + rows.mask_code.r
    {
        return Err(Error::ShapeMismatch);
    }
    Ok((
        new_code.ell as usize,
        new_code.r as usize,
        new_mask_code.ell as usize,
        new_mask_code.r as usize,
    ))
}

/// Builds the switched message for the prover. The old messages and their
/// random tails occupy the front of the new message; the remaining entries are
/// zero in this subset.
pub fn switched_message(
    rows: &SingleRows,
    new_code: CodeParams,
    new_mask_code: CodeParams,
) -> Result<SwitchedMessage> {
    rows.base_ready()?;
    let (main_len, main_tail_len, mask_len, mask_tail_len) =
        switched_lengths(rows, new_code, new_mask_code)?;
    let mut main = vec![Fp3::ZERO; main_len];
    let mut mask = vec![Fp3::ZERO; mask_len];
    main[..rows.main.len()].copy_from_slice(&rows.main);
    main[rows.main.len()..rows.main.len() + rows.main_tail.len()]
        .copy_from_slice(&rows.main_tail);
    mask[..rows.masks[0].len()].copy_from_slice(&rows.masks[0]);
    mask[rows.masks[0].len()..rows.masks[0].len() + rows.mask_tails[0].len()]
        .copy_from_slice(&rows.mask_tails[0]);
    Ok(SwitchedMessage {
        main,
        main_tail: vec![Fp3::ZERO; main_tail_len],
        mask,
        mask_tail: vec![Fp3::ZERO; mask_tail_len],
    })
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

/// Registers the original claim and one generator-position claim per
/// authenticated old position, then combines them with the supplied theta.
/// The returned claims are pending: registration is not verification, and a
/// false claim is for the later proximity test to catch.
pub fn register_switch(
    statement: &ProximityStatement,
    new_code: CodeParams,
    new_mask_code: CodeParams,
    authenticated: &[AuthenticatedSymbols],
    theta: Fp3,
) -> Result<SwitchedClaims> {
    statement.check()?;
    if new_code.ell < statement.main_len + statement.main_tail_len
        || new_mask_code.ell < statement.mask_len + statement.mask_tail_len
    {
        return Err(Error::ShapeMismatch);
    }
    let main_len = new_code.ell as usize;
    let mask_len = new_mask_code.ell as usize;

    let mut claims = Vec::new();
    let mut original_main = vec![Fp3::ZERO; main_len];
    let mut original_mask = vec![Fp3::ZERO; mask_len];
    original_main[..statement.claim.len()].copy_from_slice(&statement.claim);
    original_mask[..statement.mask_claims[0].len()].copy_from_slice(&statement.mask_claims[0]);
    claims.push(SwitchClaim {
        main: original_main,
        mask: original_mask,
        target: statement.target,
    });

    let mut old_symbols = Vec::with_capacity(authenticated.len());
    for auth in authenticated {
        if auth.position >= statement.code.n() {
            return Err(Error::InvalidIndex);
        }
        old_symbols.push(auth.main);
        let main_coeffs = coeff_on_message(&statement.code, auth.position, statement.code.ell as usize)?;
        let main_tail_coeffs = coeff_on_tail(&statement.code, auth.position)?;
        let mut main_coeff = vec![Fp3::ZERO; main_len];
        main_coeff[..statement.code.ell as usize].copy_from_slice(&main_coeffs);
        main_coeff[statement.code.ell as usize..statement.code.ell as usize + statement.code.r as usize]
            .copy_from_slice(&main_tail_coeffs);

        let mask_position = statement.mask_position(auth.position);
        let mask_coeffs = coeff_on_message(
            &statement.mask_code,
            mask_position,
            statement.mask_code.ell as usize,
        )?;
        let mask_tail_coeffs = coeff_on_tail(&statement.mask_code, mask_position)?;
        let mut mask_coeff = vec![Fp3::ZERO; mask_len];
        mask_coeff[..statement.mask_code.ell as usize].copy_from_slice(&mask_coeffs);
        mask_coeff[statement.mask_code.ell as usize
            ..statement.mask_code.ell as usize + statement.mask_code.r as usize]
            .copy_from_slice(&mask_tail_coeffs);

        // Keep the main and mask equations separate: summing them into one
        // target can cancel a wrong relation.
        claims.push(SwitchClaim {
            main: main_coeff,
            mask: vec![Fp3::ZERO; mask_len],
            target: auth.main,
        });
        claims.push(SwitchClaim {
            main: vec![Fp3::ZERO; main_len],
            mask: mask_coeff,
            target: auth.mask,
        });
    }

    let mut combined_main = vec![Fp3::ZERO; main_len];
    let mut combined_mask = vec![Fp3::ZERO; mask_len];
    let mut combined_target = Fp3::ZERO;
    let mut power = Fp3::ONE;
    for claim in &claims {
        add_scaled(&mut combined_main, &claim.main, power)?;
        add_scaled(&mut combined_mask, &claim.mask, power)?;
        combined_target = combined_target.add(&power.mul(&claim.target));
        power = power.mul(&theta);
    }
    Ok(SwitchedClaims {
        positions: authenticated.iter().map(|auth| auth.position).collect(),
        old_symbols,
        claims,
        theta,
        combined: SwitchClaim {
            main: combined_main,
            mask: combined_mask,
            target: combined_target,
        },
        main_len: new_code.ell,
        main_tail_len: new_code.r,
        mask_len: new_mask_code.ell,
        mask_tail_len: new_mask_code.r,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::proximity::{ProximityStatement, SingleRows};

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
        let statement = ProximityStatement::from_rows(&rows).unwrap();
        let message = switched_message(&rows, build_code_plan(3, 6, 1).unwrap(), build_code_plan(3, 6, 1).unwrap()).unwrap();
        assert_eq!(message.main.len(), 6);
        assert_eq!(message.main_tail.len(), 1);
        let new_code = build_code_plan(3, 6, 1).unwrap();
        let auth = [
            AuthenticatedSymbols { position: 1, main: Fp3::from_u64(111), mask: Fp3::from_u64(222) },
            AuthenticatedSymbols { position: 2, main: Fp3::from_u64(333), mask: Fp3::from_u64(444) },
        ];
        let first = register_switch(&statement, new_code, new_code, &auth, Fp3::from_u64(7)).unwrap();
        let second = register_switch(&statement, new_code, new_code, &auth, Fp3::from_u64(7)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.claims.len(), 5);
        assert_eq!(first.old_symbols.len(), 2);
        assert!(first.base_ready(new_code, new_code).is_ok());
    }

    #[test]
    fn wrong_code_capacity_or_shape_is_rejected() {
        let rows = rows();
        let statement = ProximityStatement::from_rows(&rows).unwrap();
        let small = build_code_plan(2, 2, 1).unwrap();
        let auth = [AuthenticatedSymbols { position: 1, main: Fp3::ONE, mask: Fp3::ONE }];
        assert!(register_switch(&statement, small, statement.mask_code, &auth, Fp3::ONE).is_err());
        assert!(switched_message(&rows, small, rows.mask_code).is_err());
        let mut bad = rows;
        bad.main.pop();
        assert!(bad.base_ready().is_err());
    }
}
