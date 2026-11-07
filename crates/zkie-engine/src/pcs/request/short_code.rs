//! Short codes: a mask row is read at the long position modulo its length.

use zkie_core::error::{Error, Result};

use crate::pcs::request::code::CodeParams;

/// The short position that corresponds to a long position.
pub fn short_position(log_n: u32, log_short: u32, position: u64) -> Result<u64> {
    if log_short > log_n || log_short >= 64 {
        return Err(Error::InvalidConfig);
    }
    if position >> log_n != 0 {
        return Err(Error::InvalidIndex);
    }
    Ok(position & ((1u64 << log_short) - 1))
}

/// The number of mask columns needed by FOLD's hiding polynomial.
pub fn fold_mask_columns(k: u32, h: u32) -> Result<u64> {
    if k < 1 || h > k {
        return Err(Error::InvalidConfig);
    }
    Ok(2 * k as u64 + 1 - h as u64)
}

/// A mask code fits inside the main code and has enough columns.
pub fn short_code_params(main: &CodeParams, log_short: u32) -> Result<CodeParams> {
    if log_short < 1 || log_short > main.log_n {
        return Err(Error::InvalidConfig);
    }
    crate::pcs::request::code::build_code_plan(log_short, main.ell, main.r)
}
