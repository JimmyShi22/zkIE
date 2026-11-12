//! CS/BASE local algebraic component: code-switch claim registration.
//!
//! The prover side builds the switched message. The verifier side registers
//! the original claim and one generator-position claim per authenticated old
//! opening, then combines them by powers of theta. Registration is not a proof
//! that the switched message is a correct re-encoding; that is the later
//! proximity test's task. No kept-row path is exercised in this subset.
//!
//! This is an implemented algebraic subset. It is not an IOPP, not a complete
//! BASE result and not an iopp_fault-equivalent parameter check. The k = 0
//! fixture is algebraic only; real FOLD output/witness, all applicable plan
//! constraints, and a real SEND sample declaration with follow-up checks
//! remain unclosed.

use serde::{Deserialize, Serialize};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::code::{build_code_plan, encode_row_fp3, CodeParams};
use crate::pcs::request::field::{Fp3, E_DEGREE};
use crate::pcs::request::proximity::{ProximityStatement, SingleRows};
use crate::pcs::request::runtime_tree::{
    RuntimePartRows, RuntimeTree, RuntimeTreePart, RuntimeTreePlan,
};

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
    if length != code.ell as usize {
        return Err(Error::ShapeMismatch);
    }
    let form = crate::pcs::request::code::coefficients_at_fp3(code, position)?;
    Ok(form[code.r as usize..].to_vec())
}

fn coeff_on_tail(code: &CodeParams, position: u64) -> Result<Vec<Fp3>> {
    let form = crate::pcs::request::code::coefficients_at_fp3(code, position)?;
    Ok(form[..code.r as usize].to_vec())
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
/// Data flow: the original statement claim/target enter at theta^0; each
/// authenticated main symbol at position p contributes a main-message
/// coefficient claim with target equal to that symbol; each authenticated mask
/// symbol at p mod mask_n contributes a separate mask-message coefficient claim
/// with target equal to that symbol. The claims are combined as
/// sum(theta^i * claim_i) and are pending, not proved.
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


/// The part of the switched message's table a cut fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CutPart {
    Main,
    Mask,
}

/// An aligned block of `2^bits` rows of one table of the switched message. A
/// vector fills it row first: entry `j` is in row `first_row + (j mod 2^bits)`
/// at place `j div 2^bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cut {
    pub part: CutPart,
    pub bits: u32,
    pub first_row: u64,
}

/// The public shape of one switch stage: the next stage's code, mask code,
/// row counts and answer count, and where the old main message and the rest
/// go. The current stage's openings and samples use the current shape; this
/// target is built from the next stage of the same bound plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwitchTarget {
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub answers: u32,
    pub main: Cut,
    pub rest: Cut,
}

impl SwitchTarget {
    pub fn from_stage(
        code: CodeParams,
        mask_code: CodeParams,
        k: u32,
        h: u32,
        answers: u32,
    ) -> Result<Self> {
        build_code_plan(code.log_n, code.ell, code.r)?;
        build_code_plan(mask_code.log_n, mask_code.ell, mask_code.r)?;
        if k >= 32 || h >= 32 || answers < 1 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            code,
            mask_code,
            main_rows: 1u64 << k,
            mask_rows: 1u64 << h,
            answers,
            main: Cut {
                part: CutPart::Main,
                bits: k,
                first_row: 0,
            },
            rest: Cut {
                part: CutPart::Mask,
                bits: h,
                first_row: 0,
            },
        })
    }

    /// The index of entry `j` of a vector filling `cut` in a table whose row
    /// bits are `table_row_bits`.
    pub fn cut_index(cut: &Cut, table_row_bits: u32, j: u64) -> Result<u64> {
        if table_row_bits >= 64 || cut.bits >= 64 || cut.bits > table_row_bits {
            return Err(Error::InvalidConfig);
        }
        let block = 1u64
            .checked_shl(cut.bits)
            .ok_or(Error::ResourceLimit)?;
        let row = cut
            .first_row
            .checked_add(j & (block - 1))
            .ok_or(Error::ResourceLimit)?;
        let place = j >> cut.bits;
        row.checked_add(place.checked_shl(table_row_bits).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)
    }

    /// The sample pad entries of the new oracle message, in table-index order.
    /// Pads live in the mask table when it has rows, otherwise in the main
    /// table; with `2^rows` real rows the pad of answer `a` is in the last row
    /// at place `ell - answers + a`.
    pub fn pads(&self) -> Result<Vec<(CutPart, u64)>> {
        let mut pads = Vec::with_capacity(self.answers as usize);
        if self.mask_rows > 0 {
            let row_bits = self.mask_rows.trailing_zeros();
            for answer in 0..self.answers as u64 {
                let place = self
                    .mask_code
                    .ell
                    .checked_sub(self.answers as u64)
                    .and_then(|base| base.checked_add(answer))
                    .ok_or(Error::ShapeMismatch)?;
                let index = (self.mask_rows - 1)
                    .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
                    .ok_or(Error::ResourceLimit)?;
                pads.push((CutPart::Mask, index));
            }
        } else {
            let row_bits = self.main_rows.trailing_zeros();
            for answer in 0..self.answers as u64 {
                let place = self
                    .code
                    .ell
                    .checked_sub(self.answers as u64)
                    .and_then(|base| base.checked_add(answer))
                    .ok_or(Error::ShapeMismatch)?;
                let index = (self.main_rows - 1)
                    .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
                    .ok_or(Error::ResourceLimit)?;
                pads.push((CutPart::Main, index));
            }
        }
        Ok(pads)
    }

    /// The table indices of the rest's entries, in fill order, with the pad
    /// entries skipped. The list is small enough for this subset (the rest
    /// block has `2^h * mask ell` entries).
    pub fn rest_places(&self) -> Result<Vec<u64>> {
        if self.rest.part != CutPart::Mask {
            return Err(Error::UnsupportedPrimitive);
        }
        let room = self
            .mask_code
            .ell
            .checked_mul(1u64 << self.rest.bits)
            .ok_or(Error::ResourceLimit)?;
        let row_bits = self.rest.bits;
        let mut skip = vec![false; usize::try_from(room).map_err(|_| Error::ResourceLimit)?];
        for (part, index) in self.pads()? {
            if part == CutPart::Mask {
                let fill = (index & ((1u64 << row_bits) - 1)) - self.rest.first_row
                    + ((index >> row_bits) << row_bits);
                let at = usize::try_from(fill).map_err(|_| Error::ResourceLimit)?;
                if at >= skip.len() {
                    return Err(Error::ShapeMismatch);
                }
                skip[at] = true;
            }
        }
        let mut places = Vec::with_capacity(skip.len());
        let mut fill = 0u64;
        for (at, skipped) in skip.iter().enumerate() {
            if *skipped {
                continue;
            }
            let _ = at;
            places.push(Self::cut_index(&self.rest, row_bits, fill)?);
            fill = fill.checked_add(1).ok_or(Error::ResourceLimit)?;
        }
        Ok(places)
    }

    /// The number of entries the rest contributes.
    pub fn rest_length(rows: &SingleRows) -> usize {
        let masks = rows.masks.len();
        masks * rows.mask_code.ell as usize
            + rows.main_tail.len()
            + masks * rows.mask_code.r as usize
    }

    /// Checks the public configuration. The old main message must be a power
    /// of two and fit the main block; the rest must fit its block after pads.
    pub fn check(&self, rows: &SingleRows) -> Result<()> {
        if rows.masks.len() != 2
            || rows.mask_tails.len() != 2
            || rows.mask_claims.len() != 2
            || rows.main.len() as u64 != rows.code.ell
            || rows.main_tail.len() as u64 != rows.code.r
            || rows.claim.len() as u64 != rows.code.ell
            || rows.masks.iter().any(|row| row.len() as u64 != rows.mask_code.ell)
            || rows.mask_tails.iter().any(|row| row.len() as u64 != rows.mask_code.r)
            || rows.mask_claims.iter().any(|row| row.len() as u64 != rows.mask_code.ell)
        {
            return Err(Error::ShapeMismatch);
        }
        if !rows.code.ell.is_power_of_two() {
            return Err(Error::UnsupportedPrimitive);
        }
        let main_block = 1u64 << self.main.bits;
        if rows.code.ell < main_block
            || rows.code.ell > self.code.ell << self.main.bits
            || self.main.part != CutPart::Main
            || self.main.first_row != 0
            || main_block != self.main_rows
        {
            return Err(Error::ShapeMismatch);
        }
        if self.rest.part != CutPart::Mask
            || self.rest.first_row != 0
            || (1u64 << self.rest.bits) != self.mask_rows
        {
            return Err(Error::UnsupportedPrimitive);
        }
        if Self::rest_length(rows) > self.rest_places()?.len() {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// Builds the multi-row switched message: the old main message fills the
/// target's main block through its cut; the rest fills the target's mask block
/// in the reference order (mask messages, main tail, mask tails), skipping the
/// sample pads; the pads take the caller's private pad values; the new
/// encoding tails are the caller's private values.
pub fn switched_message_multi(
    rows: &SingleRows,
    target: &SwitchTarget,
    main_tail: Vec<Fp3>,
    mask_tail: Vec<Fp3>,
    pad_value: Fp3,
) -> Result<SwitchedMessage> {
    target.check(rows)?;
    if main_tail.len() as u64 != target.code.r || mask_tail.len() as u64 != target.mask_code.r {
        return Err(Error::ShapeMismatch);
    }
    let main_len = usize::try_from(target.main_rows.checked_mul(target.code.ell).ok_or(Error::ResourceLimit)?).map_err(|_| Error::ResourceLimit)?;
    let mask_len = usize::try_from(target.mask_rows.checked_mul(target.mask_code.ell).ok_or(Error::ResourceLimit)?).map_err(|_| Error::ResourceLimit)?;
    let mut main = vec![Fp3::ZERO; main_len];
    let main_row_bits = target.main.bits;
    for (j, value) in rows.main.iter().enumerate() {
        let index = SwitchTarget::cut_index(&target.main, main_row_bits, j as u64)?;
        *main.get_mut(index as usize).ok_or(Error::ShapeMismatch)? = *value;
    }
    let mut rest = Vec::with_capacity(SwitchTarget::rest_length(rows));
    for row in &rows.masks {
        rest.extend_from_slice(row);
    }
    rest.extend_from_slice(&rows.main_tail);
    for tail in &rows.mask_tails {
        rest.extend_from_slice(tail);
    }
    let places = target.rest_places()?;
    if rest.len() > places.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut mask = vec![Fp3::ZERO; mask_len];
    for (value, index) in rest.iter().zip(&places) {
        *mask.get_mut(*index as usize).ok_or(Error::ShapeMismatch)? = *value;
    }
    for (part, index) in target.pads()? {
        if part == CutPart::Mask {
            *mask.get_mut(index as usize).ok_or(Error::ShapeMismatch)? = pad_value;
        }
    }
    Ok(SwitchedMessage {
        main,
        main_tail,
        mask,
        mask_tail,
    })
}

/// One authenticated old-row opening for the multi-row switch: the main
/// symbol and one mask symbol per mask row of the single rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthenticatedRows {
    pub position: u64,
    pub main: Fp3,
    pub masks: [Fp3; 2],
}

fn place_rest(
    out: &mut [Fp3],
    places: &[u64],
    first: usize,
    coefficients: &[Fp3],
) -> Result<()> {
    for (offset, value) in coefficients.iter().enumerate() {
        let at = places
            .get(first + offset)
            .ok_or(Error::ShapeMismatch)?;
        *out.get_mut(*at as usize).ok_or(Error::ShapeMismatch)? = *value;
    }
    Ok(())
}

/// Registers the multi-row switch claims: the original claim, then for every
/// authenticated old position one main-row claim and one claim per mask row.
/// The old messages and tails are lifted through the target's cut and rest
/// layout; the claims are combined by powers of theta and remain pending.
pub fn register_switch_multi(
    rows: &SingleRows,
    target: &SwitchTarget,
    authenticated: &[AuthenticatedRows],
    theta: Fp3,
) -> Result<SwitchedClaims> {
    target.check(rows)?;
    let main_len = usize::try_from(
        target
            .main_rows
            .checked_mul(target.code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mask_len = usize::try_from(
        target
            .mask_rows
            .checked_mul(target.mask_code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let places = target.rest_places()?;
    let mask_ell = rows.mask_code.ell as usize;
    let mask_r = rows.mask_code.r as usize;
    let main_r = rows.code.r as usize;
    let base_mask = [0usize, mask_ell];
    let base_main_tail = 2 * mask_ell;
    let base_mask_tail = [
        base_main_tail + main_r,
        base_main_tail + main_r + mask_r,
    ];

    let mut claims = Vec::new();
    let mut original_main = vec![Fp3::ZERO; main_len];
    for (j, value) in rows.claim.iter().enumerate() {
        let at = SwitchTarget::cut_index(&target.main, target.main.bits, j as u64)?;
        original_main[at as usize] = *value;
    }
    let mut original_mask = vec![Fp3::ZERO; mask_len];
    for (r, claim) in rows.mask_claims.iter().enumerate().take(2) {
        place_rest(&mut original_mask, &places, base_mask[r], claim)?;
    }
    claims.push(SwitchClaim {
        main: original_main,
        mask: original_mask,
        target: rows.target,
    });

    for auth in authenticated {
        if auth.position >= rows.code.n() {
            return Err(Error::InvalidIndex);
        }
        let mut main_claim = vec![Fp3::ZERO; main_len];
        let main_message = coeff_on_message(&rows.code, auth.position, rows.code.ell as usize)?;
        for (j, value) in main_message.iter().enumerate() {
            let at = SwitchTarget::cut_index(&target.main, target.main.bits, j as u64)?;
            main_claim[at as usize] = main_claim[at as usize].add(value);
        }
        let mut mask_claim = vec![Fp3::ZERO; mask_len];
        let main_tail = coeff_on_tail(&rows.code, auth.position)?;
        place_rest(&mut mask_claim, &places, base_main_tail, &main_tail)?;
        claims.push(SwitchClaim {
            main: main_claim,
            mask: mask_claim,
            target: auth.main,
        });

        let mask_position = rows.mask_position(auth.position);
        let mask_message =
            coeff_on_message(&rows.mask_code, mask_position, rows.mask_code.ell as usize)?;
        let mask_tail = coeff_on_tail(&rows.mask_code, mask_position)?;
        for r in 0..2 {
            let mut claim_main = vec![Fp3::ZERO; main_len];
            let mut claim_mask = vec![Fp3::ZERO; mask_len];
            place_rest(&mut claim_mask, &places, base_mask[r], &mask_message)?;
            place_rest(&mut claim_mask, &places, base_mask_tail[r], &mask_tail)?;
            let _ = &mut claim_main;
            claims.push(SwitchClaim {
                main: claim_main,
                mask: claim_mask,
                target: auth.masks[r],
            });
        }
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
        old_symbols: authenticated.iter().map(|auth| auth.main).collect(),
        claims,
        theta,
        combined: SwitchClaim {
            main: combined_main,
            mask: combined_mask,
            target: combined_target,
        },
        main_len: target.main_rows * target.code.ell,
        main_tail_len: 0,
        mask_len: target.mask_rows * target.mask_code.ell,
        mask_tail_len: 0,
    })
}

/// The multi-row switched message with one encoding tail per new row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchedRows {
    pub main: Vec<Fp3>,
    pub main_tails: Vec<Vec<Fp3>>,
    pub mask: Vec<Fp3>,
    pub mask_tails: Vec<Vec<Fp3>>,
}

/// Builds the multi-row switched message with a private encoding tail for
/// every main and mask row. The rest (old messages and old tails) is laid out
/// exactly as `switched_message_multi`; the new tails are not part of it.
pub fn switched_rows_multi(
    rows: &SingleRows,
    target: &SwitchTarget,
    main_tails: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
    pad_value: Fp3,
) -> Result<SwitchedRows> {
    target.check(rows)?;
    if main_tails.len() as u64 != target.main_rows
        || mask_tails.len() as u64 != target.mask_rows
        || main_tails.iter().any(|tail| tail.len() as u64 != target.code.r)
        || mask_tails
            .iter()
            .any(|tail| tail.len() as u64 != target.mask_code.r)
    {
        return Err(Error::ShapeMismatch);
    }
    let main_len = usize::try_from(
        target
            .main_rows
            .checked_mul(target.code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mask_len = usize::try_from(
        target
            .mask_rows
            .checked_mul(target.mask_code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mut main = vec![Fp3::ZERO; main_len];
    for (j, value) in rows.main.iter().enumerate() {
        let index = SwitchTarget::cut_index(&target.main, target.main.bits, j as u64)?;
        *main.get_mut(index as usize).ok_or(Error::ShapeMismatch)? = *value;
    }
    let mut rest = Vec::with_capacity(SwitchTarget::rest_length(rows));
    for row in &rows.masks {
        rest.extend_from_slice(row);
    }
    rest.extend_from_slice(&rows.main_tail);
    for tail in &rows.mask_tails {
        rest.extend_from_slice(tail);
    }
    let places = target.rest_places()?;
    if rest.len() > places.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut mask = vec![Fp3::ZERO; mask_len];
    for (value, index) in rest.iter().zip(&places) {
        *mask.get_mut(*index as usize).ok_or(Error::ShapeMismatch)? = *value;
    }
    for (part, index) in target.pads()? {
        if part == CutPart::Mask {
            *mask.get_mut(index as usize).ok_or(Error::ShapeMismatch)? = pad_value;
        }
    }
    Ok(SwitchedRows {
        main,
        main_tails,
        mask,
        mask_tails,
    })
}

/// The runtime tree plan of the multi-row switched oracle message: one part
/// with `2^k` main rows and `2^h` mask rows over the target's codes.
/// The leaf class of a switched oracle's message tree. The last stage's
/// independent masking oracle uses a different class so the two trees never
/// share leaves.
pub const SWITCH_LEAF_CLASS: u32 = 7;

pub fn switch_tree_plan_multi(
    tag: u64,
    target: &SwitchTarget,
    salt_bytes: usize,
    run: Digest,
    configuration: Digest,
) -> Result<RuntimeTreePlan> {
    switch_tree_plan_multi_at(
        0x300u64.wrapping_add(tag & 0xff),
        target,
        salt_bytes,
        run,
        configuration,
    )
}

/// Builds the same switched oracle tree with a tree number allocated by the
/// caller's trusted plan. Two stage trees may not share a number with any
/// other request tree, so the number is an input instead of a tag formula.
pub fn switch_tree_plan_multi_at(
    tree_number: u64,
    target: &SwitchTarget,
    salt_bytes: usize,
    run: Digest,
    configuration: Digest,
) -> Result<RuntimeTreePlan> {
    let plan = RuntimeTreePlan {
        tree_number,
        leaf_class: SWITCH_LEAF_CLASS,
        log_n: target.code.log_n,
        log_n_zk: target.mask_code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: target
                .main_rows
                .checked_mul(E_DEGREE as u64)
                .ok_or(Error::ResourceLimit)?,
            mask_symbols: target
                .mask_rows
                .checked_mul(E_DEGREE as u64)
                .ok_or(Error::ResourceLimit)?,
        }],
        salt_bytes,
        run,
        configuration,
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(plan)
}

/// Encodes every new main and mask row with its own tail and commits the
/// runtime tree of the switched oracle message.
pub fn commit_switch(
    target: &SwitchTarget,
    rows: &SwitchedRows,
    plan: &RuntimeTreePlan,
    salts: &[Vec<u8>],
) -> Result<RuntimeTree> {
    let main_rows = usize::try_from(target.main_rows).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(target.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(target.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_ell = usize::try_from(target.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    if rows.main_tails.len() != main_rows || rows.mask_tails.len() != mask_rows {
        return Err(Error::ShapeMismatch);
    }
    let mut main = Vec::with_capacity(main_rows);
    for row in 0..main_rows {
        let mut message = Vec::with_capacity(ell);
        for place in 0..ell {
            let index = row
                .checked_add(place << target.main.bits)
                .ok_or(Error::ResourceLimit)?;
            message.push(*rows.main.get(index).ok_or(Error::ShapeMismatch)?);
        }
        main.push(encode_row_fp3(
            &target.code,
            &message,
            &rows.main_tails[row],
        )?);
    }
    let mut mask = Vec::with_capacity(mask_rows);
    for row in 0..mask_rows {
        let mut message = Vec::with_capacity(mask_ell);
        for place in 0..mask_ell {
            let index = row
                .checked_add(place << target.rest.bits)
                .ok_or(Error::ResourceLimit)?;
            message.push(*rows.mask.get(index).ok_or(Error::ShapeMismatch)?);
        }
        mask.push(encode_row_fp3(
            &target.mask_code,
            &message,
            &rows.mask_tails[row],
        )?);
    }
    RuntimeTree::commit(
        plan.clone(),
        &[RuntimePartRows {
            main,
            mask,
            main_deg: E_DEGREE as u32,
        }],
        salts,
    )
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

    #[test]
    fn the_multi_row_switch_fills_the_next_stage_blocks() {
        use crate::pcs::request::code::build_code_plan;
        let old_code = build_code_plan(17, 4096, 94).unwrap();
        let old_mask = build_code_plan(14, 64, 47).unwrap();
        let rows = SingleRows {
            code: old_code,
            mask_code: old_mask,
            main: (0..4096).map(|index| Fp3::from_u64(index + 1)).collect(),
            main_tail: (0..94).map(|index| Fp3::from_u64(0x1000 + index)).collect(),
            masks: vec![
                (0..64).map(|index| Fp3::from_u64(0x2000 + index)).collect(),
                (0..64).map(|index| Fp3::from_u64(0x3000 + index)).collect(),
            ],
            mask_tails: vec![
                (0..47).map(|index| Fp3::from_u64(0x4000 + index)).collect(),
                (0..47).map(|index| Fp3::from_u64(0x5000 + index)).collect(),
            ],
            claim: vec![Fp3::ZERO; 4096],
            mask_claims: vec![vec![Fp3::ZERO; 64], vec![Fp3::ZERO; 64]],
            target: Fp3::ZERO,
        };
        let next_code = build_code_plan(12, 256, 64).unwrap();
        let next_mask = build_code_plan(12, 64, 64).unwrap();
        let target = SwitchTarget::from_stage(next_code, next_mask, 4, 3, 1).unwrap();
        assert_eq!(SwitchTarget::rest_length(&rows), 316);
        target.check(&rows).unwrap();
        let message = switched_message_multi(
            &rows,
            &target,
            vec![Fp3::from_u64(7); 64],
            vec![Fp3::from_u64(8); 64],
            Fp3::from_u64(9),
        )
        .unwrap();
        assert_eq!(message.main.len(), 16 * 256);
        assert_eq!(message.mask.len(), 8 * 64);
        assert_eq!(message.main[0], rows.main[0]);
        assert_eq!(message.main[1], rows.main[1]);
        assert_eq!(message.main[16], rows.main[16]);
        let places = target.rest_places().unwrap();
        assert_eq!(message.mask[places[0] as usize], rows.masks[0][0]);
        assert_eq!(message.mask[places[64] as usize], rows.masks[1][0]);
        assert_eq!(message.mask[places[128] as usize], rows.main_tail[0]);
        assert_eq!(message.mask[places[222] as usize], rows.mask_tails[0][0]);
        // The pad entry is not in the rest and carries the private pad value.
        assert_eq!(message.mask[511], Fp3::from_u64(9));
    }


    fn evaluate_claims(message: &SwitchedMessage, claims: &SwitchedClaims) -> Fp3 {
        let mut value = Fp3::ZERO;
        for (coefficient, symbol) in claims.combined.main.iter().zip(&message.main) {
            value = value.add(&coefficient.mul(symbol));
        }
        for (coefficient, symbol) in claims.combined.mask.iter().zip(&message.mask) {
            value = value.add(&coefficient.mul(symbol));
        }
        value
    }

    #[test]
    fn the_multi_row_switch_claims_hold_on_the_honest_message() {
        use crate::pcs::request::code::{build_code_plan, encode_row_fp3};
        let old_code = build_code_plan(17, 4096, 94).unwrap();
        let old_mask = build_code_plan(14, 64, 47).unwrap();
        let rows = SingleRows {
            code: old_code,
            mask_code: old_mask,
            main: (0..4096).map(|index| Fp3::from_u64(index + 1)).collect(),
            main_tail: (0..94).map(|index| Fp3::from_u64(0x1000 + index)).collect(),
            masks: vec![
                (0..64).map(|index| Fp3::from_u64(0x2000 + index)).collect(),
                (0..64).map(|index| Fp3::from_u64(0x3000 + index)).collect(),
            ],
            mask_tails: vec![
                (0..47).map(|index| Fp3::from_u64(0x4000 + index)).collect(),
                (0..47).map(|index| Fp3::from_u64(0x5000 + index)).collect(),
            ],
            claim: vec![Fp3::ZERO; 4096],
            mask_claims: vec![vec![Fp3::ZERO; 64], vec![Fp3::ZERO; 64]],
            target: Fp3::ZERO,
        };
        let we = encode_row_fp3(&rows.code, &rows.main, &rows.main_tail).unwrap();
        let wm = [
            encode_row_fp3(&rows.mask_code, &rows.masks[0], &rows.mask_tails[0]).unwrap(),
            encode_row_fp3(&rows.mask_code, &rows.masks[1], &rows.mask_tails[1]).unwrap(),
        ];
        let positions = [3u64, 100u64];
        let auth: Vec<AuthenticatedRows> = positions
            .iter()
            .map(|position| {
                let mask_position = rows.mask_position(*position) as usize;
                AuthenticatedRows {
                    position: *position,
                    main: we[*position as usize],
                    masks: [wm[0][mask_position], wm[1][mask_position]],
                }
            })
            .collect();
        let next_code = build_code_plan(12, 256, 64).unwrap();
        let next_mask = build_code_plan(12, 64, 64).unwrap();
        let target = SwitchTarget::from_stage(next_code, next_mask, 4, 3, 1).unwrap();
        let message = switched_message_multi(
            &rows,
            &target,
            vec![Fp3::from_u64(7); 64],
            vec![Fp3::from_u64(8); 64],
            Fp3::from_u64(9),
        )
        .unwrap();
        let theta = Fp3::from_u64(11);
        let switched = register_switch_multi(&rows, &target, &auth, theta).unwrap();
        assert_eq!(switched.claims.len(), 1 + 2 * 3);
        let mut value = Fp3::ZERO;
        for (coefficient, symbol) in switched.combined.main.iter().zip(&message.main) {
            value = value.add(&coefficient.mul(symbol));
        }
        for (coefficient, symbol) in switched.combined.mask.iter().zip(&message.mask) {
            value = value.add(&coefficient.mul(symbol));
        }
        assert_eq!(value, switched.combined.target);
        let mut tampered = message.clone();
        tampered.main[0] = tampered.main[0].add(&Fp3::ONE);
        assert_ne!(evaluate_claims(&tampered, &switched), switched.combined.target);
        let mut tampered = message.clone();
        let first_rest = target.rest_places().unwrap()[0] as usize;
        tampered.mask[first_rest] = tampered.mask[first_rest].add(&Fp3::ONE);
        assert_ne!(evaluate_claims(&tampered, &switched), switched.combined.target);
        let mut bad_auth = auth.clone();
        bad_auth[0].main = bad_auth[0].main.add(&Fp3::ONE);
        let bad = register_switch_multi(&rows, &target, &bad_auth, theta).unwrap();
        assert_ne!(evaluate_claims(&message, &bad), bad.combined.target);
    }

    fn evaluate_rows_claims(message: &SwitchedRows, claims: &SwitchedClaims) -> Fp3 {
        let mut value = Fp3::ZERO;
        for (coefficient, symbol) in claims.combined.main.iter().zip(&message.main) {
            value = value.add(&coefficient.mul(symbol));
        }
        for (coefficient, symbol) in claims.combined.mask.iter().zip(&message.mask) {
            value = value.add(&coefficient.mul(symbol));
        }
        value
    }

    #[test]
    fn the_multi_row_switch_accepts_a_zero_target_claim() {
        use crate::pcs::request::code::build_code_plan;
        let code = build_code_plan(3, 4, 1).unwrap();
        let mask_code = build_code_plan(3, 4, 1).unwrap();
        let rows = SingleRows {
            code,
            mask_code,
            main: vec![Fp3::ZERO; 4],
            main_tail: vec![Fp3::ZERO; 1],
            masks: vec![vec![Fp3::ZERO; 4], vec![Fp3::ZERO; 4]],
            mask_tails: vec![vec![Fp3::ZERO; 1], vec![Fp3::ZERO; 1]],
            claim: vec![Fp3::ZERO; 4],
            mask_claims: vec![vec![Fp3::ZERO; 4], vec![Fp3::ZERO; 4]],
            target: Fp3::ZERO,
        };
        let target = SwitchTarget::from_stage(
            build_code_plan(4, 2, 1).unwrap(),
            build_code_plan(4, 2, 1).unwrap(),
            2,
            3,
            1,
        )
        .unwrap();
        let message = switched_rows_multi(
            &rows,
            &target,
            vec![vec![Fp3::ZERO; 1]; 4],
            vec![vec![Fp3::ZERO; 1]; 8],
            Fp3::ZERO,
        )
        .unwrap();
        let auth = [AuthenticatedRows {
            position: 1,
            main: Fp3::ZERO,
            masks: [Fp3::ZERO, Fp3::ZERO],
        }];
        let switched =
            register_switch_multi(&rows, &target, &auth, Fp3::from_u64(7)).unwrap();
        assert_eq!(switched.combined.target, Fp3::ZERO);
        assert_eq!(evaluate_rows_claims(&message, &switched), Fp3::ZERO);
    }


    #[test]
    fn the_multi_row_switch_commits_one_codeword_per_new_row() {
        use crate::pcs::request::code::build_code_plan;
        use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
        let old_code = build_code_plan(17, 4096, 94).unwrap();
        let old_mask = build_code_plan(14, 64, 47).unwrap();
        let rows = SingleRows {
            code: old_code,
            mask_code: old_mask,
            main: (0..4096).map(|index| Fp3::from_u64(index + 1)).collect(),
            main_tail: (0..94).map(|index| Fp3::from_u64(0x1000 + index)).collect(),
            masks: vec![
                (0..64).map(|index| Fp3::from_u64(0x2000 + index)).collect(),
                (0..64).map(|index| Fp3::from_u64(0x3000 + index)).collect(),
            ],
            mask_tails: vec![
                (0..47).map(|index| Fp3::from_u64(0x4000 + index)).collect(),
                (0..47).map(|index| Fp3::from_u64(0x5000 + index)).collect(),
            ],
            claim: vec![Fp3::ZERO; 4096],
            mask_claims: vec![vec![Fp3::ZERO; 64], vec![Fp3::ZERO; 64]],
            target: Fp3::ZERO,
        };
        let next_code = build_code_plan(12, 256, 64).unwrap();
        let next_mask = build_code_plan(12, 64, 64).unwrap();
        let target = SwitchTarget::from_stage(next_code, next_mask, 4, 3, 1).unwrap();
        let main_tails: Vec<Vec<Fp3>> = (0..16)
            .map(|row| (0..64).map(|entry| Fp3::from_u64(0x6000 + row * 0x100 + entry)).collect())
            .collect();
        let mask_tails: Vec<Vec<Fp3>> = (0..8)
            .map(|row| (0..64).map(|entry| Fp3::from_u64(0x7000 + row * 0x100 + entry)).collect())
            .collect();
        let rows = switched_rows_multi(
            &rows,
            &target,
            main_tails,
            mask_tails,
            Fp3::from_u64(0x99),
        )
        .unwrap();
        let plan = switch_tree_plan_multi(
            1,
            &target,
            LEAF_SALT_MIN_BYTES,
            Digest([0xc1; 32]),
            Digest([0xc2; 32]),
        )
        .unwrap();
        let salts: Vec<Vec<u8>> = (0..plan.columns())
            .map(|position| vec![(position as u8) ^ 0x77; plan.salt_bytes])
            .collect();
        let tree = commit_switch(&target, &rows, &plan, &salts).unwrap();
        assert_ne!(tree.root(), &Digest::ZERO);
        let again = commit_switch(&target, &rows, &plan, &salts).unwrap();
        assert_eq!(tree.root(), again.root());
        let mut changed = rows.clone();
        changed.main_tails[0][0] = Fp3::from_u64(0xdead);
        let other = commit_switch(&target, &changed, &plan, &salts).unwrap();
        assert_ne!(tree.root(), other.root());
    }

}
