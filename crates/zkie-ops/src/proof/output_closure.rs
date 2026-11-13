//! Exact integer range and comparison closure of the hidden output-head row.
//!
//! The hidden LIN request proves `<X_t, Omega> = W_t(psi*)` over the committed
//! reduced rows. Its trusted output relation is bound by a digest only. This
//! module adds the verifier-checked relation between the same committed rows
//! and the public first-maximum output:
//!
//! ```text
//! A  = 2^D - 1
//! u_j = Y_j + A,   v_j = A - Y_j                 (D+1 bits each)
//! S_j = Z_a - Z_j - [j < a],  0 <= S_j < 2^b     (b bits)
//! Z_j = Y_j + (j >= V ? M : 0),  a = winner,  M <= -2^D
//! ```
//!
//! The prover commits three new messages per run:
//!
//! * the slack message `S`, one entry per column and position, with the same
//!   entry map as the reduced rows;
//! * one packed bit message per position: the `D+1` bits of `u`, the `D+1`
//!   bits of `v` and the `b` bits of `S` for every column, padded to a
//!   power-of-two vector;
//! * the committed product message `D` above, one entry per packed bit;
//! * one mask-only message holding the product-sumcheck hiding coefficients,
//!   the two shown-value masks and the three relation masks of every
//!   position.
//!
//! The verifier checks the relation through the existing own-group
//! BAT/FOLD/CS/BASE test:
//!
//! * two random-point reconstruction rows per position bind the packed bits to
//!   the committed reduced row (`u = Y + A`, `v = A - Y`) and one binds the
//!   slack bits to the committed slack message;
//! * one random-point comparison row per position binds `S` to the same
//!   reduced row, the public mask values and the public winner;
//! * a committed product message `D(u) = eq(r,u) B(u)` is tied to the bit
//!   message by two public linear functionals at a fresh point `q` plus one
//!   equality row: `sum_u eq(q,u) D(u) = sum_u eq(q,u) eq(r,u) B(u)`;
//! * one masked product-sumcheck instance per position proves
//!   `sum_u D(u) (1 - B(u)) = 0`. Its first map is the committed `D`; its
//!   second map `1 - B` is affine in the committed bit map, so the end claim
//!   is bound by one linear row on the bit point value and the closure mask.
//!   With `D = eq(r,.) B`, the proved sum is
//!   `sum_u eq(r,u) B(u) (1 - B(u))`, the random-point booleanity check.
//!   The `(U)` row binds the one-time hiding coefficients.
//!
//! Every probabilistic check is a random-point check over the extension field;
//! the module states its error terms in the documentation but claims no formal
//! soundness proof and no zero-knowledge proof. The committed messages are
//! hiding through the existing product-form encoding with fresh tails, salts
//! and non-zero sample pads.

use rayon::prelude::*;
use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::fold::MaskCoins;
use zkie_engine::pcs::request::code::{encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp3, MODULUS, E_DEGREE};
use zkie_engine::pcs::request::runtime_tree::{
    RuntimePartRows, RuntimeTree, RuntimeTreeOpening, RuntimeTreePlan,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::psc::{
    hp_entries, prove_dot_sumcheck, psc_variables, verify_psc_transcript, DotClaim, DotMaskEntries,
    DotOutput, DotProof, SHOWN_MAPS,
};
pub use crate::proof::committed_psc::CommittedGroupMessage;
use crate::proof::committed_psc::{
    absorb_point, get_psc_transcript, get_tree_plan, put_psc_transcript, put_tree_plan,
    CommittedPscGroup, CommittedPscGroupRef, PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::hidden_send_session::{
    HiddenSendPlan, SendGroup, SendGroupEntry, SendSamples,
};
use crate::proof::column_session::fp_from_signed_word;

/// Version of the output-closure relation. Version 2 fixes the exact integer
/// decomposition of the revised output relation; a different convention is a
/// different relation and uses a new version.
pub const CLOSURE_RELATION_VERSION: u64 = 2;
/// The number of shown-value end masks of one product-sumcheck instance.
pub const CLOSURE_SHOWN: usize = SHOWN_MAPS;
/// The number of one-time shown masks per position: the two product-sumcheck
/// end masks, the committed-bit point mask and the two relation-function
/// masks.
pub const CLOSURE_MASK_SHOWN: usize = 5;
/// Upper bound on the packed bit-message width.
const BIT_COLUMNS_BOUND: u64 = 1 << 30;
/// Schema version of the closure mask message.
const CLOSURE_MASK_SCHEMA_VERSION: u64 = 1;
/// Transcript tags of the closure product-sumcheck. They are distinct from the
/// hidden Dot tags 16, 17 and 18 and carry the position as a domain word.
const CLOSURE_HIDING_SUM_TAG: u64 = 20;
const CLOSURE_ROUND_TAG: u64 = 21;
const CLOSURE_END_TAG: u64 = 22;
/// The transcript tags of the closure product-sumcheck.
const CLOSURE_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: CLOSURE_MASK_SCHEMA_VERSION,
    hiding_sum: CLOSURE_HIDING_SUM_TAG,
    round: CLOSURE_ROUND_TAG,
    end: CLOSURE_END_TAG,
    claim: 23,
};

/// The trusted plan of the mask-only closure message. Its rows carry the
/// one-time product-sumcheck hiding coefficients and the two shown-value end
/// masks of every position; the SEND pad is the last entry of the last real
/// row and no allocated entry collides with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureMaskPlan {
    pub oracle: u64,
    pub group: u64,
    pub mask_code: CodeParams,
    pub mask_rows: u64,
    pub answers: u32,
    pub openings: usize,
    pub tree: RuntimeTreePlan,
    pub positions: u32,
    /// Product-sumcheck hiding coefficients per position.
    pub hiding: usize,
    /// Shown-value end masks per position.
    pub shown: usize,
}

impl ClosureMaskPlan {
    /// The number of allocated entries of one position.
    pub fn per_position(&self) -> Result<usize> {
        self.hiding
            .checked_add(self.shown)
            .ok_or(Error::ResourceLimit)
    }

    /// The total number of allocated entries.
    pub fn total_entries(&self) -> Result<usize> {
        (self.positions as usize)
            .checked_mul(self.per_position()?)
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one hiding entry of a position.
    pub fn hiding_slot(&self, position: u32, index: usize) -> Result<usize> {
        if position >= self.positions || index >= self.hiding {
            return Err(Error::InvalidIndex);
        }
        (position as usize)
            .checked_mul(self.per_position()?)
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one shown-value mask of a position.
    pub fn shown_slot(&self, position: u32, index: usize) -> Result<usize> {
        if position >= self.positions || index >= self.shown {
            return Err(Error::InvalidIndex);
        }
        (position as usize)
            .checked_mul(self.per_position()?)
            .and_then(|base| base.checked_add(self.hiding))
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The number of seed coordinates the mask sample form reads.
    pub fn sample_bits(&self) -> Result<u32> {
        ceil_log2_u64(self.mask_rows)
            .checked_add(self.mask_code.log_ell())
            .ok_or(Error::ResourceLimit)
    }

    /// The number of rows of the fresh fold-mask table: the next power of two
    /// at or above the group's mask rows.
    pub fn tree_rows(&self) -> Result<u64> {
        let shift = ceil_log2_u64(self.mask_rows);
        1u64.checked_shl(shift).ok_or(Error::ResourceLimit)
    }

    /// The free entries of the mask rows, excluding the SEND pad.
    pub fn free_entries(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let answers = self.answers as usize;
        if answers == 0 || answers > ell {
            return Err(Error::InvalidConfig);
        }
        let per_row = ell.checked_sub(answers).ok_or(Error::ResourceLimit)?;
        rows.checked_mul(per_row)
            .and_then(|value| value.checked_add(rows.saturating_sub(1)))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact row-major offset of an allocated entry.
    pub fn entry_offset(&self, slot: usize) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let length = rows.checked_mul(ell).ok_or(Error::ResourceLimit)?;
        if slot >= length {
            return Err(Error::InvalidIndex);
        }
        Ok(slot)
    }

    /// The BAT virtual table index of an allocated entry.
    pub fn entry_virtual_index(&self, slot: usize) -> Result<u64> {
        let offset = self.entry_offset(slot)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let row = offset / ell;
        let place = offset % ell;
        let row_bits = ceil_log2_u64(self.mask_rows);
        (row as u64)
            .checked_add((place as u64) << row_bits)
            .ok_or(Error::ResourceLimit)
    }

    /// The compact offset of the SEND pad.
    pub fn pad_offset(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        rows.checked_mul(ell)
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::ShapeMismatch)
    }

    /// The structural checks of the mask-only message plan.
    pub fn fault(&self, variables: usize) -> Option<&'static str> {
        if self.oracle == 0 || self.group == 0 {
            return Some("the closure-mask message has no identity");
        }
        if build_code(self.mask_code).is_err() {
            return Some("the closure-mask code is not a code");
        }
        if self.mask_rows == 0 {
            return Some("the closure-mask message has no mask row");
        }
        if self.answers != 1 {
            return Some("the closure-mask message does not have one answer");
        }
        if self.openings == 0 || self.openings as u64 > self.mask_code.r {
            return Some("the closure-mask opening count exceeds its tail");
        }
        if self.tree.fault().is_some() {
            return Some("the closure-mask tree is not a runtime tree");
        }
        if self.tree.log_n != self.mask_code.log_n
            || self.tree.log_n_zk != self.mask_code.log_n
            || self.tree.parts.len() != 1
        {
            return Some("the closure-mask tree does not match its code");
        }
        let mask_symbols = match self.mask_rows.checked_mul(E_DEGREE as u64) {
            Some(value) => value,
            None => return Some("the closure-mask row width overflows"),
        };
        if self.tree.parts[0].main_symbols != 0 || self.tree.parts[0].mask_symbols != mask_symbols {
            return Some("the closure-mask tree does not carry its mask rows");
        }
        let variables = match psc_variables(variables) {
            Ok(variables) => variables,
            Err(_) => return Some("the booleanity instance has no variable set"),
        };
        match hp_entries(&variables) {
            Ok(entries) if entries == self.hiding => {}
            _ => return Some("the closure-mask hiding count does not match its booleanity instance"),
        }
        if self.shown != CLOSURE_MASK_SHOWN {
            return Some("the closure-mask shown count is not five");
        }
        match self.free_entries() {
            Ok(free) if free >= self.total_entries().unwrap_or(usize::MAX) => {}
            _ => return Some("the closure-mask rows are shorter than the entry allocation"),
        }
        None
    }

    /// Encodes the public plan fields. The values are plan values, never proof
    /// bytes.
    pub fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        writer.put_u64(self.mask_code.log_n as u64);
        writer.put_u64(self.mask_code.ell);
        writer.put_u64(self.mask_code.r);
        writer.put_u64(self.mask_rows);
        writer.put_u64(self.answers as u64);
        writer.put_u64(self.openings as u64);
        writer.put_u64(self.tree.tree_number);
        writer.put_u64(self.tree.leaf_class as u64);
        writer.put_u64(self.tree.log_n as u64);
        writer.put_u64(self.tree.log_n_zk as u64);
        writer.put_u64(self.tree.parts.len() as u64);
        for part in &self.tree.parts {
            writer.put_u64(part.main_symbols);
            writer.put_u64(part.mask_symbols);
        }
        writer.put_u64(self.tree.salt_bytes as u64);
        writer.put_digest(&self.tree.run);
        writer.put_digest(&self.tree.configuration);
        writer.put_u64(self.positions as u64);
        writer.put_u64(self.hiding as u64);
        writer.put_u64(self.shown as u64);
    }
}

fn build_code(code: CodeParams) -> Result<()> {
    zkie_engine::pcs::request::code::build_code_plan(code.log_n, code.ell, code.r)?;
    Ok(())
}

fn next_power_of_two(value: u64) -> Option<u64> {
    if value == 0 || value > (1u64 << 62) {
        return None;
    }
    Some(value.next_power_of_two())
}

fn ceil_log2_u64(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// The trusted plan of the output closure: the public integer parameters, the
/// public winner and the shapes of the three new messages. The values are plan
/// values; a proof cannot substitute any of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosurePlan {
    /// The integer bound `D` of `|Y_j|`: `A = 2^D - 1`.
    pub range_bits: u32,
    /// The slack bound exponent `b`: `B = 2^b`, `0 <= S_j < B`.
    pub slack_bits: u32,
    /// The valid vocabulary boundary `V`.
    pub valid: usize,
    /// The complete padded column domain `K`.
    pub vocab: usize,
    /// The public mask constant `M < 0`.
    pub mask_value: i64,
    /// The trusted first-maximum winner, one index per position.
    pub winner: Vec<u32>,
    /// The committed slack message shape.
    pub slack: crate::proof::hidden_send_session::WShape,
    pub slack_oracle_base: u64,
    /// The number of drawn openings of one slack message.
    pub slack_openings: usize,
    /// The packed bit message shape.
    pub bits: crate::proof::hidden_send_session::WShape,
    pub bits_oracle_base: u64,
    /// The number of drawn openings of one bit message.
    pub bits_openings: usize,
    /// The committed product message shape `D(u) = eq(r,u) B(u)`.
    pub map: crate::proof::hidden_send_session::WShape,
    pub map_oracle_base: u64,
    /// The number of drawn openings of one map message.
    pub map_openings: usize,
    /// The logical packed bit vector width (`K * bits_per_cell`, padded).
    pub bits_columns: u64,
    /// Bits stored per column cell: `2*(D+1) + b`.
    pub bits_per_cell: u32,
    /// The number of variables of the booleanity instance.
    pub boolean_variables: u32,
    /// The mask-only closure message.
    pub mask: ClosureMaskPlan,
}

impl ClosurePlan {
    /// The integer bound `A = 2^D - 1`.
    pub fn offset(&self) -> Result<u64> {
        1u64.checked_shl(self.range_bits)
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::ResourceLimit)
    }

    /// The slack bound `B = 2^b`.
    pub fn slack_bound(&self) -> Result<u64> {
        1u64.checked_shl(self.slack_bits).ok_or(Error::ResourceLimit)
    }

    /// The per-cell bit layout: `u` bits at `0..D+1`, `v` bits at
    /// `D+1..2(D+1)`, `S` bits at `2(D+1)..2(D+1)+b`.
    pub fn bit_offsets(&self) -> Result<(usize, usize, usize)> {
        let width = self.range_bits as usize + 1;
        let u = 0usize;
        let v = width;
        let s = width.checked_mul(2).ok_or(Error::ResourceLimit)?;
        Ok((u, v, s))
    }

    /// The compact entry index of one cell bit in the packed message.
    pub fn bit_entry(&self, column: usize, bit: usize) -> Result<usize> {
        if column >= self.vocab || bit >= self.bits_per_cell as usize {
            return Err(Error::InvalidIndex);
        }
        column
            .checked_mul(self.bits_per_cell as usize)
            .and_then(|base| base.checked_add(bit))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact message index of one logical column entry.
    pub fn column_entry(shape: &crate::proof::hidden_send_session::WShape, column: usize) -> Result<usize> {
        let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
        let row = column % rows;
        let place = column / rows;
        if place >= ell {
            return Err(Error::ShapeMismatch);
        }
        row.checked_mul(ell)
            .and_then(|base| base.checked_add(place))
            .ok_or(Error::ResourceLimit)
    }

    /// The structural checks against the send plan and the proximity mask
    /// code. Oracle/tree collision checks against the rest of the request are
    /// done by the merged plan, which sees every member.
    pub fn fault(&self, send: &HiddenSendPlan) -> Option<&'static str> {
        if self.range_bits == 0
            || self.range_bits > 62
            || self.slack_bits == 0
            || self.slack_bits > 62
        {
            return Some("the closure integer bounds do not fit");
        }
        let width = match (self.range_bits as u64 + 1).checked_mul(2) {
            Some(width) => width,
            None => return Some("the closure bit layout overflows"),
        };
        let width = match width.checked_add(self.slack_bits as u64) {
            Some(width) => width,
            None => return Some("the closure bit layout overflows"),
        };
        if width != self.bits_per_cell as u64 {
            return Some("the closure bit layout does not match its bounds");
        }
        if self.valid == 0
            || self.valid >= self.vocab
            || self.vocab as u64 != send.columns
            || !self.vocab.is_power_of_two()
        {
            return Some("the closure vocabulary boundary is not a padded valid range");
        }
        if self.mask_value >= 0 || self.winner.len() != send.positions() as usize {
            return Some("the closure public winner or mask value is not a plan value");
        }
        if self.winner.iter().any(|index| *index as usize >= self.valid) {
            return Some("the closure winner is outside the valid vocabulary");
        }
        let offset = match self.offset() {
            Ok(offset) => offset,
            Err(_) => return Some("the closure offset overflows"),
        };
        if (self.mask_value as i128) > -((offset as i128) + 1) {
            return Some("the closure mask does not separate invalid columns");
        }
        let bound = match self.slack_bound() {
            Ok(bound) => bound,
            Err(_) => return Some("the closure slack bound overflows"),
        };
        // B + 2A + |M| < p, with A = 2^D - 1. The check is in u128 so the
        // integer expression cannot wrap.
        let no_wrap = (bound as u128)
            .checked_add(2u128.checked_mul(offset as u128).unwrap_or(u128::MAX))
            .and_then(|value| {
                value.checked_add((self.mask_value as i128).unsigned_abs())
            });
        match no_wrap {
            Some(value) if value < MODULUS as u128 => {}
            _ => return Some("the closure no-wrap bound does not hold"),
        }
        let columns = match (self.vocab as u64).checked_mul(self.bits_per_cell as u64) {
            Some(columns) if columns > 0 && columns <= BIT_COLUMNS_BOUND => columns,
            _ => return Some("the packed bit message width overflows"),
        };
        let padded = match next_power_of_two(columns) {
            Some(value) => value,
            None => return Some("the packed bit message width overflows"),
        };
        if padded != self.bits_columns || !self.bits_columns.is_power_of_two() {
            return Some("the packed bit message is not padded to a power of two");
        }
        if self.boolean_variables != self.bits_columns.trailing_zeros() {
            return Some("the booleanity variable count does not match the bit message");
        }
        if self.slack.fault(self.vocab as u64, send.positions()).is_some() {
            return Some("the slack message shape is not a message shape");
        }
        if self.slack_openings == 0 || self.slack_openings as u64 > self.slack.code.r {
            return Some("the slack opening count exceeds its tail");
        }
        if self.slack.code != send.w.code || self.slack.answers != 1 {
            return Some("the slack message does not share the own-group main code");
        }
        if self.bits.fault(self.bits_columns, send.positions()).is_some() {
            return Some("the bit message shape is not a message shape");
        }
        if self.bits_openings == 0 || self.bits_openings as u64 > self.bits.code.r {
            return Some("the bit opening count exceeds its tail");
        }
        if self.bits.code != send.w.code || self.bits.answers != 1 {
            return Some("the bit message does not share the own-group main code");
        }
        if self.bits.main_rows.checked_mul(self.bits.code.ell) < Some(self.bits_columns) {
            return Some("the bit message is shorter than its packed width");
        }
        if self.map.fault(self.bits_columns, send.positions()).is_some() {
            return Some("the map message shape is not a message shape");
        }
        if self.map.code != send.w.code || self.map.answers != 1 {
            return Some("the map message does not share the own-group main code");
        }
        if self.map_openings == 0 || self.map_openings as u64 > self.map.code.r {
            return Some("the map opening count exceeds its tail");
        }
        if self.map.main_rows.checked_mul(self.map.code.ell) < Some(self.bits_columns) {
            return Some("the map message is shorter than the bit vector width");
        }
        if self.mask.fault(self.boolean_variables as usize).is_some() {
            return Some("the closure-mask message is not a mask-only message");
        }
        if build_code(self.mask.mask_code).is_err() {
            return Some("the closure-mask code is not a code");
        }
        if self.mask.group != send.group {
            return Some("the closure-mask message does not match the own group");
        }
        if self.mask.positions != send.positions() {
            return Some("the closure-mask message does not carry one position set");
        }
        // Oracle and tree ranges must be disjoint from the reduced rows and
        // from each other. The index-mask and shown-map ranges are checked by
        // the merged plan.
        let positions = send.positions() as u64;
        let w_end = match send.oracle_base.checked_add(positions) {
            Some(end) => end,
            None => return Some("the reduced-row oracle range overflows"),
        };
        let slack_end = match self.slack_oracle_base.checked_add(positions) {
            Some(end) => end,
            None => return Some("the slack oracle range overflows"),
        };
        let bits_end = match self.bits_oracle_base.checked_add(positions) {
            Some(end) => end,
            None => return Some("the bit oracle range overflows"),
        };
        let map_end = match self.map_oracle_base.checked_add(positions) {
            Some(end) => end,
            None => return Some("the map oracle range overflows"),
        };
        let ranges = [
            (send.oracle_base, w_end),
            (self.slack_oracle_base, slack_end),
            (self.bits_oracle_base, bits_end),
            (self.map_oracle_base, map_end),
        ];
        for first in 0..ranges.len() {
            for second in first + 1..ranges.len() {
                let (a_first, a_end) = ranges[first];
                let (b_first, b_end) = ranges[second];
                if a_first < b_end && b_first < a_end {
                    return Some("two closure oracle ranges overlap");
                }
            }
        }
        if self.mask.oracle >= send.oracle_base && self.mask.oracle < w_end {
            return Some("the closure-mask oracle collides with a reduced-row oracle");
        }
        if self.mask.oracle >= self.slack_oracle_base && self.mask.oracle < slack_end {
            return Some("the closure-mask oracle collides with a slack oracle");
        }
        if self.mask.oracle >= self.bits_oracle_base && self.mask.oracle < bits_end {
            return Some("the closure-mask oracle collides with a bit oracle");
        }
        if self.mask.oracle >= self.map_oracle_base && self.mask.oracle < map_end {
            return Some("the closure-mask oracle collides with a map oracle");
        }
        let w_trees = match send.w.tree_number.checked_add(positions) {
            Some(end) => end,
            None => return Some("the reduced-row tree range overflows"),
        };
        let slack_trees = match self.slack.tree_number.checked_add(positions) {
            Some(end) => end,
            None => return Some("the slack tree range overflows"),
        };
        let bits_trees = match self.bits.tree_number.checked_add(positions) {
            Some(end) => end,
            None => return Some("the bit tree range overflows"),
        };
        let map_trees = match self.map.tree_number.checked_add(positions) {
            Some(end) => end,
            None => return Some("the map tree range overflows"),
        };
        let tree_ranges = [
            (send.w.tree_number, w_trees),
            (self.slack.tree_number, slack_trees),
            (self.bits.tree_number, bits_trees),
            (self.map.tree_number, map_trees),
        ];
        for first in 0..tree_ranges.len() {
            for second in first + 1..tree_ranges.len() {
                let (a_first, a_end) = tree_ranges[first];
                let (b_first, b_end) = tree_ranges[second];
                if a_first < b_end && b_first < a_end {
                    return Some("two closure tree ranges overlap");
                }
            }
        }
        let mask_tree = self.mask.tree.tree_number;
        if tree_ranges
            .iter()
            .any(|(first, end)| mask_tree >= *first && mask_tree < *end)
        {
            return Some("the closure-mask tree collides with a message tree");
        }
        None
    }

    /// Encodes the complete public plan declaration. The first statement and
    /// the parameters digest absorb it before any prover message.
    pub fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(CLOSURE_RELATION_VERSION);
        writer.put_u64(self.range_bits as u64);
        writer.put_u64(self.slack_bits as u64);
        writer.put_u64(self.valid as u64);
        writer.put_u64(self.vocab as u64);
        writer.put_u64(self.mask_value as u64);
        writer.put_u64(self.winner.len() as u64);
        for value in &self.winner {
            writer.put_u64(*value as u64);
        }
        self.slack.encode_into(writer);
        writer.put_u64(self.slack_oracle_base);
        writer.put_u64(self.slack_openings as u64);
        self.bits.encode_into(writer);
        writer.put_u64(self.bits_oracle_base);
        writer.put_u64(self.bits_openings as u64);
        self.map.encode_into(writer);
        writer.put_u64(self.map_oracle_base);
        writer.put_u64(self.map_openings as u64);
        writer.put_u64(self.bits_columns);
        writer.put_u64(self.bits_per_cell as u64);
        writer.put_u64(self.boolean_variables as u64);
        self.mask.encode_into(writer);
    }

    pub fn parameters_digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        self.encode_into(&mut writer);
        zkie_engine::primitives::transcript::hash_labeled(
            zkie_engine::primitives::transcript::tag_word(
                zkie_engine::primitives::transcript::PARAMETERS,
                4,
            ),
            0,
            None,
            &writer.into_bytes(),
        )
        .unwrap_or(Digest::ZERO)
    }
}

// ---------------------------------------------------------------------------
// Private randomness and the committed messages
// ---------------------------------------------------------------------------

/// The private randomness of the closure messages: encoding tails, salts and
/// non-zero sample pads, drawn from the run's private stream after the existing
/// message randomness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureRandomness {
    pub slack_tails: Vec<Vec<Fp3>>,
    pub slack_salts: Vec<Vec<Vec<u8>>>,
    pub slack_pads: Vec<Fp3>,
    pub bits_tails: Vec<Vec<Fp3>>,
    pub bits_salts: Vec<Vec<Vec<u8>>>,
    pub bits_pads: Vec<Fp3>,
    pub map_tails: Vec<Vec<Fp3>>,
    pub map_salts: Vec<Vec<Vec<u8>>>,
    pub map_pads: Vec<Fp3>,
    pub mask_rows: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub mask_salts: Vec<Vec<u8>>,
}

impl ClosureRandomness {
    /// Draws the closure randomness in a fixed order from the shared mask-coin
    /// ledger.
    pub fn from_coins(
        coins: &mut zkie_engine::pcs::request::fold::ByteMaskCoins,
        plan: &ClosurePlan,
        send: &HiddenSendPlan,
    ) -> Result<Self> {
        if plan.fault(send).is_some() {
            return Err(Error::InvalidConfig);
        }
        let positions = send.positions() as usize;
        let slack_tails = draw_shape_tails(coins, &plan.slack, positions)?;
        let slack_pads = draw_shape_pads(coins, &plan.slack, positions)?;
        let slack_salts = draw_shape_salts(coins, &plan.slack, positions)?;
        let bits_tails = draw_shape_tails(coins, &plan.bits, positions)?;
        let bits_pads = draw_shape_pads(coins, &plan.bits, positions)?;
        let bits_salts = draw_shape_salts(coins, &plan.bits, positions)?;
        let map_tails = draw_shape_tails(coins, &plan.map, positions)?;
        let map_pads = draw_shape_pads(coins, &plan.map, positions)?;
        let map_salts = draw_shape_salts(coins, &plan.map, positions)?;
        let mask_row_count =
            usize::try_from(plan.mask.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(plan.mask.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let mut rows = Vec::with_capacity(mask_row_count);
        for _ in 0..mask_row_count {
            let mut row = Vec::with_capacity(ell);
            for _ in 0..ell {
                row.push(coins.draw()?.0);
            }
            rows.push(row);
        }
        let pad_row = mask_row_count.checked_sub(1).ok_or(Error::ShapeMismatch)?;
        let pad_place = ell.checked_sub(1).ok_or(Error::ShapeMismatch)?;
        for _ in 0..64 {
            if rows[pad_row][pad_place] != Fp3::ZERO {
                break;
            }
            rows[pad_row][pad_place] = coins.draw()?.0;
        }
        if rows[pad_row][pad_place] == Fp3::ZERO {
            return Err(Error::EntropyUnavailable);
        }
        let mut mask_tails = Vec::with_capacity(mask_row_count);
        let tail_len = usize::try_from(plan.mask.mask_code.r).map_err(|_| Error::ResourceLimit)?;
        for _ in 0..mask_row_count {
            let mut tail = Vec::with_capacity(tail_len);
            for _ in 0..tail_len {
                tail.push(coins.draw()?.0);
            }
            mask_tails.push(tail);
        }
        let columns = usize::try_from(plan.mask.tree.columns()).map_err(|_| Error::ResourceLimit)?;
        let mask_salts = draw_salt_bytes(coins, columns, plan.mask.tree.salt_bytes)?;
        let randomness = Self {
            slack_tails,
            slack_salts,
            slack_pads,
            bits_tails,
            bits_salts,
            bits_pads,
            map_tails,
            map_salts,
            map_pads,
            mask_rows: rows,
            mask_tails,
            mask_salts,
        };
        randomness.check(plan, send)?;
        Ok(randomness)
    }

    fn check(&self, plan: &ClosurePlan, send: &HiddenSendPlan) -> Result<()> {
        if plan.fault(send).is_some() {
            return Err(Error::InvalidConfig);
        }
        check_shape_randomness(
            &plan.slack,
            send.positions(),
            &self.slack_tails,
            &self.slack_salts,
            &self.slack_pads,
        )?;
        check_shape_randomness(
            &plan.bits,
            send.positions(),
            &self.bits_tails,
            &self.bits_salts,
            &self.bits_pads,
        )?;
        check_shape_randomness(
            &plan.map,
            send.positions(),
            &self.map_tails,
            &self.map_salts,
            &self.map_pads,
        )?;
        let rows = usize::try_from(plan.mask.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(plan.mask.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let tail = usize::try_from(plan.mask.mask_code.r).map_err(|_| Error::ResourceLimit)?;
        let columns = usize::try_from(plan.mask.tree.columns()).map_err(|_| Error::ResourceLimit)?;
        if self.mask_rows.len() != rows
            || self.mask_rows.iter().any(|row| row.len() != ell)
            || self.mask_tails.len() != rows
            || self.mask_tails.iter().any(|row| row.len() != tail)
            || self.mask_salts.len() != columns
            || self.mask_salts.iter().any(|salt| salt.len() != plan.mask.tree.salt_bytes)
            || self.mask_rows[rows - 1][ell - 1] == Fp3::ZERO
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    /// The compact mask entry at one slot.
    pub fn mask_entry(&self, plan: &ClosurePlan, slot: usize) -> Result<Fp3> {
        let offset = plan.mask.entry_offset(slot)?;
        let ell = usize::try_from(plan.mask.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        self.mask_rows
            .get(offset / ell)
            .and_then(|row| row.get(offset % ell))
            .copied()
            .ok_or(Error::ShapeMismatch)
    }
}

fn draw_shape_tails(
    coins: &mut zkie_engine::pcs::request::fold::ByteMaskCoins,
    shape: &crate::proof::hidden_send_session::WShape,
    positions: usize,
) -> Result<Vec<Vec<Fp3>>> {
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let tail = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let mut tails = Vec::with_capacity(positions);
    for _ in 0..positions {
        let mut values = Vec::with_capacity(rows * tail);
        for _ in 0..rows * tail {
            values.push(coins.draw()?.0);
        }
        tails.push(values);
    }
    Ok(tails)
}

fn draw_shape_pads(
    coins: &mut zkie_engine::pcs::request::fold::ByteMaskCoins,
    shape: &crate::proof::hidden_send_session::WShape,
    positions: usize,
) -> Result<Vec<Fp3>> {
    let answers = shape.answers as usize;
    let mut pads = Vec::with_capacity(positions);
    for _ in 0..positions {
        for _ in 0..answers {
            let mut pad = Fp3::ZERO;
            for _ in 0..64 {
                pad = coins.draw()?.0;
                if pad != Fp3::ZERO {
                    break;
                }
            }
            if pad == Fp3::ZERO {
                return Err(Error::EntropyUnavailable);
            }
            pads.push(pad);
        }
    }
    Ok(pads)
}

fn draw_shape_salts(
    coins: &mut zkie_engine::pcs::request::fold::ByteMaskCoins,
    shape: &crate::proof::hidden_send_session::WShape,
    positions: usize,
) -> Result<Vec<Vec<Vec<u8>>>> {
    let columns = usize::try_from(shape.code.n()).map_err(|_| Error::ResourceLimit)?;
    let mut salts = Vec::with_capacity(positions);
    for _ in 0..positions {
        salts.push(draw_salt_bytes(coins, columns, shape.salt_bytes)?);
    }
    Ok(salts)
}

fn draw_salt_bytes(
    coins: &mut zkie_engine::pcs::request::fold::ByteMaskCoins,
    count: usize,
    salt_bytes: usize,
) -> Result<Vec<Vec<u8>>> {
    let mut salts = Vec::with_capacity(count);
    let words = salt_bytes.div_ceil(24);
    for _ in 0..count {
        let mut bytes = Vec::with_capacity(salt_bytes);
        for _ in 0..words {
            let (value, _) = coins.draw()?;
            for coefficient in value.c {
                bytes.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
            }
        }
        bytes.truncate(salt_bytes);
        salts.push(bytes);
    }
    Ok(salts)
}

fn check_shape_randomness(
    shape: &crate::proof::hidden_send_session::WShape,
    positions: u32,
    tails: &[Vec<Fp3>],
    salts: &[Vec<Vec<u8>>],
    pads: &[Fp3],
) -> Result<()> {
    let count = positions as usize;
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let tail = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(shape.code.n()).map_err(|_| Error::ResourceLimit)?;
    let answers = shape.answers as usize;
    if tails.len() != count
        || tails.iter().any(|values| values.len() != rows * tail)
        || salts.len() != count
        || salts.iter().any(|values| {
            values.len() != columns || values.iter().any(|salt| salt.len() != shape.salt_bytes)
        })
        || pads.len() != count * answers
        || pads.iter().any(|pad| *pad == Fp3::ZERO)
    {
        return Err(Error::ShapeMismatch);
    }
    Ok(())
}

/// The commit trees retained from one value-group commit, in position order.
/// A tree is derived data: equality compares the tree plan and root rather
/// than the materialized columns.
#[derive(Clone, Debug, Default)]
pub struct RetainedTrees(pub Vec<std::sync::Arc<RuntimeTree>>);

impl RetainedTrees {
    pub fn get(&self, position: usize) -> Option<&std::sync::Arc<RuntimeTree>> {
        self.0.get(position)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl PartialEq for RetainedTrees {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(left, right)| left.plan() == right.plan() && left.root() == right.root())
    }
}

impl Eq for RetainedTrees {}

/// One committed value message group (the slack rows or the packed bit rows).
#[derive(Clone, Debug)]
pub struct CommittedValueGroup {
    pub group: SendGroup,
    /// `messages[position][row][place]`, the unencoded message rows.
    pub messages: Vec<Vec<Vec<Fp3>>>,
    pub roots: Vec<Digest>,
    /// The trees built by the same commit, reused by the witness and the
    /// drawn opening steps instead of rebuilding from the same inputs.
    pub trees: RetainedTrees,
}

impl PartialEq for CommittedValueGroup {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group && self.messages == other.messages && self.roots == other.roots
    }
}

impl Eq for CommittedValueGroup {}

/// Commits one value message per position with the product-form encoding.
#[allow(clippy::too_many_arguments)]
pub fn commit_value_group(
    shape: &crate::proof::hidden_send_session::WShape,
    columns: u64,
    oracle_base: u64,
    group: u64,
    values: &[Vec<Fp3>],
    pads: &[Fp3],
    tails: &[Vec<Fp3>],
    salts: &[Vec<Vec<u8>>],
    positions: u32,
) -> Result<CommittedValueGroup> {
    let _profile_commit = zkie_engine::profile::Scope::new("value_group.commit");
    if values.len() != positions as usize
        || values.iter().any(|row| row.len() as u64 != columns)
        || pads.len() != positions as usize * shape.answers as usize
        || tails.len() != positions as usize
        || salts.len() != positions as usize
    {
        return Err(Error::ShapeMismatch);
    }
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let answers = shape.answers as usize;
    let mut entries = Vec::with_capacity(positions as usize);
    let mut messages = Vec::with_capacity(positions as usize);
    let mut roots = Vec::with_capacity(positions as usize);
    let mut trees = Vec::with_capacity(positions as usize);
    for (position, values) in values.iter().enumerate() {
        let _profile_layout = zkie_engine::profile::Scope::light("value_group.layout");
        let mut message = vec![vec![Fp3::ZERO; ell]; rows];
        for (column, value) in values.iter().enumerate() {
            let row = column % rows;
            let place = column / rows;
            let entry = message
                .get_mut(row)
                .and_then(|row| row.get_mut(place))
                .ok_or(Error::ShapeMismatch)?;
            *entry = *value;
        }
        for answer in 0..answers {
            let pad = *pads
                .get(position * answers + answer)
                .ok_or(Error::ShapeMismatch)?;
            let place = ell
                .checked_sub(answers)
                .and_then(|first| first.checked_add(answer))
                .ok_or(Error::ResourceLimit)?;
            let entry = message
                .get_mut(rows - 1)
                .and_then(|row| row.get_mut(place))
                .ok_or(Error::ShapeMismatch)?;
            if !entry.is_zero() {
                return Err(Error::StatementMismatch);
            }
            *entry = pad;
        }
        drop(_profile_layout);
        let tail = tails
            .get(position)
            .ok_or(Error::ShapeMismatch)?;
        let _profile_encode = zkie_engine::profile::Scope::light("value_group.encode");
        let mut codewords = Vec::with_capacity(rows);
        for row in 0..rows {
            let message_row = message.get(row).ok_or(Error::ShapeMismatch)?;
            let row_tail = tail
                .get(row * tail_len..(row + 1) * tail_len)
                .ok_or(Error::ShapeMismatch)?;
            codewords.push(encode_row_fp3(&shape.code, message_row, row_tail)?);
        }
        drop(_profile_encode);
        let salt = salts.get(position).ok_or(Error::ShapeMismatch)?;
        let tree_plan = shape.tree_plan(position as u32)?;
        let tree = RuntimeTree::commit(
            tree_plan.clone(),
            &[RuntimePartRows {
                main: codewords,
                mask: Vec::new(),
                main_deg: E_DEGREE as u32,
            }],
            salt,
        )?;
        let root = *tree.root();
        entries.push(SendGroupEntry {
            position: position as u32,
            oracle: oracle_base
                .checked_add(position as u64)
                .ok_or(Error::ResourceLimit)?,
            shape: shape.oracle_shape(),
            tree: tree_plan,
            root,
        });
        roots.push(root);
        trees.push(std::sync::Arc::new(tree));
        messages.push(message);
    }
    let group = SendGroup { group, entries };
    if group.fault(positions).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(CommittedValueGroup {
        group,
        messages,
        roots,
        trees: RetainedTrees(trees),
    })
}

/// The committed mask-only closure message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedClosureMask {
    pub root: Digest,
    pub rows: Vec<Vec<Fp3>>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
}

impl CommittedClosureMask {
    pub fn tree(&self, plan: &ClosureMaskPlan) -> Result<RuntimeTree> {
        let mut codewords = Vec::with_capacity(self.rows.len());
        for (row, tail) in self.rows.iter().zip(&self.tails) {
            codewords.push(encode_row_fp3(&plan.mask_code, row, tail)?);
        }
        RuntimeTree::commit(
            plan.tree.clone(),
            &[RuntimePartRows {
                main: Vec::new(),
                mask: codewords,
                main_deg: 1,
            }],
            &self.salts,
        )
    }

    /// The compact mask entry at one slot.
    pub fn entry(&self, plan: &ClosureMaskPlan, slot: usize) -> Result<Fp3> {
        let offset = plan.entry_offset(slot)?;
        let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        self.rows
            .get(offset / ell)
            .and_then(|row| row.get(offset % ell))
            .copied()
            .ok_or(Error::ShapeMismatch)
    }
}

/// Commits the mask-only closure message from its private randomness.
pub fn commit_closure_mask(
    plan: &ClosureMaskPlan,
    randomness: &ClosureRandomness,
) -> Result<CommittedClosureMask> {
    let committed = CommittedClosureMask {
        root: Digest::ZERO,
        rows: randomness.mask_rows.clone(),
        tails: randomness.mask_tails.clone(),
        salts: randomness.mask_salts.clone(),
    };
    let root = *committed.tree(plan)?.root();
    Ok(CommittedClosureMask { root, ..committed })
}

/// The public first-stage closure messages: the three roots, the SEND samples
/// and the drawn opening positions of the new messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureFirstMessages {
    pub slack: CommittedGroupMessage,
    pub slack_sample: SendSamples,
    pub slack_positions: Vec<Vec<u64>>,
    pub slack_openings: Vec<RuntimeTreeOpening>,
    pub bits: CommittedGroupMessage,
    pub bits_sample: SendSamples,
    pub bits_positions: Vec<Vec<u64>>,
    pub bits_openings: Vec<RuntimeTreeOpening>,
    pub map: CommittedGroupMessage,
    pub map_sample: SendSamples,
    pub map_positions: Vec<Vec<u64>>,
    pub map_openings: Vec<RuntimeTreeOpening>,
    pub mask: ClosureMaskMessage,
}

/// The public mask-only closure message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureMaskMessage {
    pub tree: RuntimeTreePlan,
    pub root: Digest,
    pub sample: SendSamples,
    pub positions: Vec<u64>,
    pub opening: RuntimeTreeOpening,
}

impl ClosureFirstMessages {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CLOSURE_MASK_SCHEMA_VERSION);
        CommittedPscGroupRef {
            message: &self.slack,
            sample: &self.slack_sample,
            positions: &self.slack_positions,
            openings: &self.slack_openings,
        }
        .encode_into(&mut writer)?;
        CommittedPscGroupRef {
            message: &self.bits,
            sample: &self.bits_sample,
            positions: &self.bits_positions,
            openings: &self.bits_openings,
        }
        .encode_into(&mut writer)?;
        CommittedPscGroupRef {
            message: &self.map,
            sample: &self.map_sample,
            positions: &self.map_positions,
            openings: &self.map_openings,
        }
        .encode_into(&mut writer)?;
        put_tree_plan(&mut writer, &self.mask.tree);
        writer.put_digest(&self.mask.root);
        writer.put_bytes(&self.mask.sample.encode());
        writer.put_u64(self.mask.positions.len() as u64);
        for position in &self.mask.positions {
            writer.put_u64(*position);
        }
        writer.put_bytes(&zkie_engine::pcs::request::runtime_tree::encode_runtime_opening(
            &self.mask.tree,
            &self.mask.opening,
        )?);
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &ClosurePlan, positions: u32, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, zkie_core::common::codec::DecodeLimits::unlimited());
        if reader.get_u64()? != CLOSURE_MASK_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let slack = CommittedPscGroup::decode(&mut reader, &plan.slack, positions)?;
        let bits = CommittedPscGroup::decode(&mut reader, &plan.bits, positions)?;
        let map = CommittedPscGroup::decode(&mut reader, &plan.map, positions)?;
        let tree = get_tree_plan(&mut reader)?;
        if tree != plan.mask.tree {
            return Err(Error::StatementMismatch);
        }
        let root = reader.get_digest()?;
        if root == Digest::ZERO {
            return Err(Error::InvalidEncoding);
        }
        let sample = SendSamples::decode(reader.get_bytes()?)?;
        if sample.seed.len() != plan.mask.answers as usize
            || sample.answers.len() != plan.mask.answers as usize
        {
            return Err(Error::ShapeMismatch);
        }
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count != plan.mask.openings {
            return Err(Error::ShapeMismatch);
        }
        let mut mask_positions = Vec::with_capacity(count);
        for _ in 0..count {
            mask_positions.push(reader.get_u64()?);
        }
        let opening = zkie_engine::pcs::request::runtime_tree::decode_runtime_opening(
            &tree,
            &mask_positions,
            reader.get_bytes()?,
        )?;
        reader.finish()?;
        Ok(Self {
            slack: slack.message,
            slack_sample: slack.sample,
            slack_positions: slack.positions,
            slack_openings: slack.openings,
            bits: bits.message,
            bits_sample: bits.sample,
            bits_positions: bits.positions,
            bits_openings: bits.openings,
            map: map.message,
            map_sample: map.sample,
            map_positions: map.positions,
            map_openings: map.openings,
            mask: ClosureMaskMessage {
                tree,
                root,
                sample,
                positions: mask_positions,
                opening,
            },
        })
    }
}

/// The prover's private closure material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureWitness {
    pub slack: CommittedValueGroup,
    pub slack_tails: Vec<Vec<Fp3>>,
    pub slack_salts: Vec<Vec<Vec<u8>>>,
    pub slack_pads: Vec<Fp3>,
    pub bits: CommittedValueGroup,
    pub bits_tails: Vec<Vec<Fp3>>,
    pub bits_salts: Vec<Vec<Vec<u8>>>,
    pub bits_pads: Vec<Fp3>,
    pub map: CommittedValueGroup,
    pub map_tails: Vec<Vec<Fp3>>,
    pub map_salts: Vec<Vec<Vec<u8>>>,
    pub map_pads: Vec<Fp3>,
    /// The packed bit values of every position, in PSC order.
    pub bit_values: Vec<Vec<Fp3>>,
    /// The committed product values `D(u) = eq(r,u) B(u)`, in PSC order.
    pub map_values: Vec<Vec<Fp3>>,
    /// The first-stage booleanity point `r`.
    pub r_bool: Vec<Fp3>,
    pub mask: CommittedClosureMask,
    /// `mask_entries[position] = hiding.. ..shown`, compact slot order.
    pub mask_entries: Vec<Vec<Fp3>>,
}

impl ClosureWitness {
    pub fn check(&self, plan: &ClosurePlan, send: &HiddenSendPlan) -> Result<()> {
        if plan.fault(send).is_some() {
            return Err(Error::InvalidConfig);
        }
        let positions = send.positions() as usize;
        let rows = usize::try_from(plan.slack.main_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(plan.slack.code.ell).map_err(|_| Error::ResourceLimit)?;
        if self.slack.messages.len() != positions
            || self.slack.messages.iter().any(|message| {
                message.len() != rows || message.iter().any(|row| row.len() != ell)
            })
            || self.slack_tails.len() != positions
            || self.slack_salts.len() != positions
            || self.slack_pads.len() != positions
        {
            return Err(Error::ShapeMismatch);
        }
        let bit_rows = usize::try_from(plan.bits.main_rows).map_err(|_| Error::ResourceLimit)?;
        let bit_ell = usize::try_from(plan.bits.code.ell).map_err(|_| Error::ResourceLimit)?;
        if self.bits.messages.len() != positions
            || self.bits.messages.iter().any(|message| {
                message.len() != bit_rows || message.iter().any(|row| row.len() != bit_ell)
            })
            || self.bits_tails.len() != positions
            || self.bits_salts.len() != positions
            || self.bits_pads.len() != positions
            || self.bit_values.len() != positions
            || self
                .bit_values
                .iter()
                .any(|values| values.len() as u64 != plan.bits_columns)
        {
            return Err(Error::ShapeMismatch);
        }
        let map_rows = usize::try_from(plan.map.main_rows).map_err(|_| Error::ResourceLimit)?;
        let map_ell = usize::try_from(plan.map.code.ell).map_err(|_| Error::ResourceLimit)?;
        if self.map.messages.len() != positions
            || self.map.messages.iter().any(|message| {
                message.len() != map_rows || message.iter().any(|row| row.len() != map_ell)
            })
            || self.map_tails.len() != positions
            || self.map_salts.len() != positions
            || self.map_pads.len() != positions
            || self.map_values.len() != positions
            || self
                .map_values
                .iter()
                .any(|values| values.len() as u64 != plan.bits_columns)
            || self.r_bool.len() != plan.boolean_variables as usize
        {
            return Err(Error::ShapeMismatch);
        }
        if self.mask_entries.len() != positions
            || self.mask_entries.iter().any(|entries| {
                entries.len() != plan.mask.hiding + plan.mask.shown
            })
        {
            return Err(Error::ShapeMismatch);
        }
        if *self.mask.tree(&plan.mask)?.root() != self.mask.root {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// The public second-stage closure messages: one product-sumcheck transcript
/// per position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureSecondMessages {
    pub psc: Vec<DotProof>,
    /// The masked fresh-point evaluations `(B(pt), D(q), weighted B(q))` per
    /// position.
    pub point_values: Vec<[Fp3; 3]>,
}

impl ClosureSecondMessages {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        put_psc_transcript(
            &mut writer,
            CLOSURE_MASK_SCHEMA_VERSION,
            &self.psc,
            &self.point_values,
        )?;
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &ClosurePlan, positions: u32, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, zkie_core::common::codec::DecodeLimits::unlimited());
        let (psc, point_values) = get_psc_transcript::<3>(
            &mut reader,
            CLOSURE_MASK_SCHEMA_VERSION,
            positions,
            plan.boolean_variables as usize,
        )?;
        reader.finish()?;
        Ok(Self { psc, point_values })
    }
}

/// One position's checked product-sumcheck binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosurePositionBinding {
    pub r_bool: Vec<Fp3>,
    pub r_diff: Vec<Fp3>,
    pub r_recon: Vec<Fp3>,
    /// The fresh point of the committed-product relation.
    pub q: Vec<Fp3>,
    /// The masked evaluations at the booleanity point and at `q`.
    pub b_point: Fp3,
    pub d_value: Fp3,
    pub bq_value: Fp3,
    pub output: DotOutput,
}

/// The checked bindings of every position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureRowBinding {
    pub positions: Vec<ClosurePositionBinding>,
}

fn multilinear(values: &[Fp3], point: &[Fp3]) -> Result<Fp3> {
    if values.len() != (1usize << point.len()) {
        return Err(Error::ShapeMismatch);
    }
    let mut current = values.to_vec();
    for challenge in point {
        let half = current.len() / 2;
        for index in 0..half {
            let low = current[2 * index];
            let high = current[2 * index + 1];
            current[index] = low.add(&challenge.mul(&high.sub(&low)));
        }
        current.truncate(half);
    }
    current.first().copied().ok_or(Error::ShapeMismatch)
}

fn absorb_closure_point(chain: &mut Chain, position: u64, values: &[Fp3]) -> Result<()> {
    absorb_point(chain, CLOSURE_MASK_SCHEMA_VERSION, 38, position, values)
}

fn closure_claim() -> DotClaim {
    DotClaim {
        masks: Vec::new(),
        target: Fp3::ZERO,
    }
}

fn closure_mask_entries(
    plan: &ClosurePlan,
    mask_entries: &[Fp3],
) -> Result<DotMaskEntries> {
    if mask_entries.len() != plan.mask.hiding + plan.mask.shown {
        return Err(Error::ShapeMismatch);
    }
    let hiding = mask_entries[..plan.mask.hiding].to_vec();
    let shown = [
        *mask_entries
            .get(plan.mask.hiding)
            .ok_or(Error::ShapeMismatch)?,
        *mask_entries
            .get(plan.mask.hiding + 1)
            .ok_or(Error::ShapeMismatch)?,
    ];
    Ok(DotMaskEntries { hiding, shown })
}

fn empty_mask_entries(plan: &ClosurePlan) -> DotMaskEntries {
    DotMaskEntries {
        hiding: vec![Fp3::ZERO; plan.mask.hiding],
        shown: [Fp3::ZERO; CLOSURE_SHOWN],
    }
}


/// The committed product vector `D(u) = eq(r,u) B(u)` over the cube with the
/// shared `eq(r, .)` vector supplied by the caller, so every position reuses
/// one draw of the boolean point.
pub fn closure_map_values_with_eq(bits: &[Fp3], eq: &[Fp3]) -> Result<Vec<Fp3>> {
    closure_map_values_with_eq_min(bits, eq, 1 << 14)
}

fn closure_map_values_with_eq_min(
    bits: &[Fp3],
    eq: &[Fp3],
    parallel_min: usize,
) -> Result<Vec<Fp3>> {
    if bits.len() != eq.len() {
        return Err(Error::ShapeMismatch);
    }
    if bits.len() >= parallel_min {
        Ok(eq
            .par_iter()
            .zip(bits.par_iter())
            .map(|(eq, bit)| eq.mul(bit))
            .collect())
    } else {
        Ok(eq.iter().zip(bits).map(|(eq, bit)| eq.mul(bit)).collect())
    }
}

/// The committed product vector `D(u) = eq(r,u) B(u)` over the cube.
pub fn closure_map_values(bits: &[Fp3], r: &[Fp3]) -> Result<Vec<Fp3>> {
    let eq = eq_vector(r)?;
    closure_map_values_with_eq(bits, &eq)
}

/// The complete `eq(point, .)` vector over the Boolean cube in `O(2^k)`.
pub fn eq_vector(point: &[Fp3]) -> Result<Vec<Fp3>> {
    eq_vector_with_min(point, 1 << 14)
}

fn eq_vector_with_min(point: &[Fp3], parallel_min: usize) -> Result<Vec<Fp3>> {
    if point.len() >= 30 {
        return Err(Error::ResourceLimit);
    }
    let mut values = vec![Fp3::ONE];
    for coordinate in point.iter().rev() {
        let next_len = values.len() * 2;
        let mut next = vec![Fp3::ZERO; next_len];
        if next_len >= parallel_min {
            next.par_chunks_mut(2)
                .zip(values.par_iter())
                .for_each(|(pair, value)| {
                    pair[0] = value.mul(&Fp3::ONE.sub(coordinate));
                    pair[1] = value.mul(coordinate);
                });
        } else {
            for (index, value) in values.iter().enumerate() {
                next[2 * index] = value.mul(&Fp3::ONE.sub(coordinate));
                next[2 * index + 1] = value.mul(coordinate);
            }
        }
        values = next;
    }
    Ok(values)
}

fn inner_product(a: &[Fp3], b: &[Fp3]) -> Result<Fp3> {
    if a.len() != b.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (x, y) in a.iter().zip(b) {
        sum = sum.add(&x.mul(y));
    }
    Ok(sum)
}

/// Prover side of the closure second stage: runs one masked product-sumcheck
/// per position with the product map and the affine complement of the bit
/// map, sends the fresh-point functionals, then draws the comparison and
/// reconstruction points.
pub fn prove_closure_second(
    chain: &mut Chain,
    plan: &ClosurePlan,
    bit_values: &[Vec<Fp3>],
    map_values: &[Vec<Fp3>],
    mask_entries: &[Vec<Fp3>],
    r_bool: &[Fp3],
) -> Result<(ClosureSecondMessages, ClosureRowBinding)> {
    let _profile_psc = zkie_engine::profile::Scope::new("second.closure_psc");
    let positions = plan.winner.len();
    if bit_values.len() != positions
        || map_values.len() != positions
        || mask_entries.len() != positions
        || r_bool.len() != plan.boolean_variables as usize
    {
        return Err(Error::ShapeMismatch);
    }
    let k = plan.boolean_variables as usize;
    let k_col = plan.vocab.trailing_zeros() as usize;
    let eq_r = eq_vector(r_bool)?;
    let mut proofs = Vec::with_capacity(positions);
    let mut point_values = Vec::with_capacity(positions);
    let mut bindings = Vec::with_capacity(positions);
    for position in 0..positions {
        let _profile_position =
            zkie_engine::profile::Scope::new("second.closure_psc_position");
        let omega: Vec<Fp3> = bit_values[position]
            .iter()
            .map(|value| Fp3::ONE.sub(value))
            .collect();
        let claim = closure_claim();
        let entries = closure_mask_entries(plan, &mask_entries[position])?;
        let (proof, output) = {
            let mut challenges =
                PositionPscChallenges::new(chain, CLOSURE_PSC_TAGS, position as u64);
            prove_dot_sumcheck(
                &map_values[position],
                &omega,
                &claim,
                &entries,
                &mut challenges,
            )?
        };
        let mut q = Vec::with_capacity(k);
        for _ in 0..k {
            q.push(chain.draw_e()?);
        }
        let b_point = multilinear(&bit_values[position], &output.point)?;
        let eq_q = eq_vector(&q)?;
        let d_value_masked = inner_product(&eq_q, &map_values[position])?
            .add(&mask_entries[position][plan.mask.hiding + 3]);
        let weighted: Vec<Fp3> = eq_q
            .iter()
            .zip(&eq_r)
            .map(|(q, r)| q.mul(r))
            .collect();
        let bq_value_masked = inner_product(&weighted, &bit_values[position])?
            .add(&mask_entries[position][plan.mask.hiding + 4]);
        let b_point_masked = b_point.add(&mask_entries[position][plan.mask.hiding + 2]);
        let point_value = [b_point_masked, d_value_masked, bq_value_masked];
        absorb_closure_point(chain, position as u64, &point_value)?;
        let mut r_diff = Vec::with_capacity(k_col);
        for _ in 0..k_col {
            r_diff.push(chain.draw_e()?);
        }
        let mut r_recon = Vec::with_capacity(k_col);
        for _ in 0..k_col {
            r_recon.push(chain.draw_e()?);
        }
        proofs.push(proof);
        point_values.push(point_value);
        bindings.push(ClosurePositionBinding {
            r_bool: r_bool.to_vec(),
            r_diff,
            r_recon,
            q,
            b_point: point_value[0],
            d_value: point_value[1],
            bq_value: point_value[2],
            output,
        });
    }
    Ok((
        ClosureSecondMessages {
            psc: proofs,
            point_values,
        },
        ClosureRowBinding { positions: bindings },
    ))
}

/// Verifier side of the closure second stage: derives the same points, checks
/// the product-sumcheck transcripts and returns the row bindings. The mask
/// entry values are committed and checked by the rows; the transcript only
/// needs their counts.
pub fn verify_closure_second(
    chain: &mut Chain,
    plan: &ClosurePlan,
    messages: &ClosureSecondMessages,
    r_bool: &[Fp3],
) -> Result<ClosureRowBinding> {
    let positions = plan.winner.len();
    if messages.psc.len() != positions || r_bool.len() != plan.boolean_variables as usize {
        return Err(Error::ShapeMismatch);
    }
    let k = plan.boolean_variables as usize;
    let k_col = plan.vocab.trailing_zeros() as usize;
    let claim = closure_claim();
    let entries = empty_mask_entries(plan);
    let mut bindings = Vec::with_capacity(positions);
    for (position, proof) in messages.psc.iter().enumerate() {
        let output = {
            let mut challenges =
                PositionPscChallenges::new(chain, CLOSURE_PSC_TAGS, position as u64);
            verify_psc_transcript(&claim, &entries, proof, k, &mut challenges)?
        };
        let mut q = Vec::with_capacity(k);
        for _ in 0..k {
            q.push(chain.draw_e()?);
        }
        let point = messages
            .point_values
            .get(position)
            .ok_or(Error::ShapeMismatch)?;
        absorb_closure_point(chain, position as u64, point)?;
        let mut r_diff = Vec::with_capacity(k_col);
        for _ in 0..k_col {
            r_diff.push(chain.draw_e()?);
        }
        let mut r_recon = Vec::with_capacity(k_col);
        for _ in 0..k_col {
            r_recon.push(chain.draw_e()?);
        }
        bindings.push(ClosurePositionBinding {
            r_bool: r_bool.to_vec(),
            r_diff,
            r_recon,
            q,
            b_point: point[0],
            d_value: point[1],
            bq_value: point[2],
            output,
        });
    }
    Ok(ClosureRowBinding { positions: bindings })
}

/// Computes the slack and packed bit values of every position from the reduced
/// rows. The prover fails closed when a row, a slack or a masked value is
/// outside the public bounds.
pub fn closure_values_from_rows(
    plan: &ClosurePlan,
    w_rows: &[Vec<Fp3>],
) -> Result<(Vec<Vec<Fp3>>, Vec<Vec<Fp3>>)> {
    let _profile_values = zkie_engine::profile::Scope::new("closure.values_from_rows");
    if w_rows.len() != plan.winner.len() {
        return Err(Error::ShapeMismatch);
    }
    let offset = plan.offset()? as i128;
    let bound = plan.slack_bound()? as i128;
    let mask = plan.mask_value as i128;
    let (u_offset, v_offset, s_offset) = plan.bit_offsets()?;
    let mut slack_values = Vec::with_capacity(w_rows.len());
    let mut bit_values = Vec::with_capacity(w_rows.len());
    for (position, row) in w_rows.iter().enumerate() {
        if row.len() != plan.vocab {
            return Err(Error::ShapeMismatch);
        }
        let winner = plan.winner[position] as usize;
        let winner_value = signed_i128(&row[winner])?;
        let mut slack_row = Vec::with_capacity(row.len());
        let mut bits = vec![Fp3::ZERO; plan.bits_columns as usize];
        for (column, value) in row.iter().enumerate() {
            let value = signed_i128(value)?;
            let masked = value
                .checked_add(if column >= plan.valid { mask } else { 0 })
                .ok_or(Error::BoundViolation)?;
            let slack = winner_value
                .checked_sub(masked)
                .and_then(|value| {
                    value.checked_sub(if column < winner { 1 } else { 0 })
                })
                .ok_or(Error::BoundViolation)?;
            if slack < 0 || slack >= bound {
                return Err(Error::BoundViolation);
            }
            slack_row.push(Fp3::from_u64(slack as u64));
            let u = value.checked_add(offset).ok_or(Error::BoundViolation)?;
            let v = offset.checked_sub(value).ok_or(Error::BoundViolation)?;
            if u < 0 || u > 2 * offset || v < 0 || v > 2 * offset {
                return Err(Error::BoundViolation);
            }
            for bit in 0..plan.range_bits as usize + 1 {
                if (u >> bit) & 1 == 1 {
                    let entry = plan.bit_entry(column, u_offset + bit)?;
                    bits[entry] = Fp3::ONE;
                }
                if (v >> bit) & 1 == 1 {
                    let entry = plan.bit_entry(column, v_offset + bit)?;
                    bits[entry] = Fp3::ONE;
                }
            }
            for bit in 0..plan.slack_bits as usize {
                if (slack >> bit) & 1 == 1 {
                    let entry = plan.bit_entry(column, s_offset + bit)?;
                    bits[entry] = Fp3::ONE;
                }
            }
        }
        slack_values.push(slack_row);
        bit_values.push(bits);
    }
    Ok((slack_values, bit_values))
}

fn signed_i128(value: &Fp3) -> Result<i128> {
    if value.c[1].as_canonical_u64() != 0 || value.c[2].as_canonical_u64() != 0 {
        return Err(Error::BoundViolation);
    }
    let canonical = value.c[0].as_canonical_u64();
    if canonical <= i64::MAX as u64 {
        Ok(canonical as i128)
    } else {
        let magnitude = MODULUS
            .checked_sub(canonical)
            .ok_or(Error::BoundViolation)?;
        if magnitude > (1u64 << 63) {
            return Err(Error::BoundViolation);
        }
        Ok(-(magnitude as i128))
    }
}

/// The public `Fp3` image of a signed public integer.
pub fn fp3_signed(value: i128) -> Result<Fp3> {
    if value < i64::MIN as i128 || value > i64::MAX as i128 {
        return Err(Error::BoundViolation);
    }
    Ok(Fp3::from_base(fp_from_signed_word(value as i64 as u64)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::fold::ByteMaskCoins;
    use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
    use crate::proof::hidden_send_session::WShape;

    const SALT: usize = 40;

    fn code() -> CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn shape(columns: u64, tree: u64) -> WShape {
        WShape::new(
            columns,
            code(),
            1,
            1,
            SALT,
            tree,
            1,
            Digest([0x41; 32]),
            Digest([0x51; 32]),
        )
        .unwrap()
    }

    fn plan() -> ClosurePlan {
        // D = 2, b = 3, K = 4, V = 3; one position.
        let width = 2 * 3 + 3;
        let bits_columns = (4u64 * width as u64).next_power_of_two();
        let variables = bits_columns.trailing_zeros() as usize;
        let mask = ClosureMaskPlan {
            oracle: 9,
            group: 1,
            mask_code: code(),
            mask_rows: 1,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: 9,
                leaf_class: 1,
                log_n: 4,
                log_n_zk: 4,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 3,
                }],
                salt_bytes: SALT,
                run: Digest([0x61; 32]),
                configuration: Digest([0x62; 32]),
            },
            positions: 1,
            hiding: hp_entries(&psc_variables(variables).unwrap()).unwrap(),
            shown: CLOSURE_MASK_SHOWN,
        };
        ClosurePlan {
            range_bits: 2,
            slack_bits: 3,
            valid: 3,
            vocab: 4,
            mask_value: -4,
            winner: vec![0],
            slack: shape(4, 1),
            slack_oracle_base: 20,
            slack_openings: 1,
            bits: shape(bits_columns, 2),
            bits_oracle_base: 30,
            bits_openings: 1,
            map: shape(bits_columns, 3),
            map_oracle_base: 40,
            map_openings: 1,
            bits_columns,
            bits_per_cell: width as u32,
            boolean_variables: bits_columns.trailing_zeros(),
            mask,
        }
    }

    fn signed(value: i64) -> Fp3 {
        crate::proof::output_closure::fp3_signed(value as i128).unwrap()
    }

    #[test]
    fn closure_values_decompose_an_honest_row() {
        let plan = plan();
        let row = vec![signed(3), signed(-3), signed(0), signed(0)];
        let (slack, bits) = closure_values_from_rows(&plan, &[row]).unwrap();
        // Z = [3, -3, 0, -4]; the winner is column 0.
        assert_eq!(slack[0], vec![signed(0), signed(6), signed(3), signed(7)]);
        // u = Y + 3 and v = 3 - Y, both in [0, 6].
        let (u_off, v_off, s_off) = plan.bit_offsets().unwrap();
        let mut expected = vec![Fp3::ZERO; plan.bits_columns as usize];
        for (column, value) in [3i64, -3, 0, 0].iter().enumerate() {
            let u = value + 3;
            let v = 3 - value;
            for bit in 0..3 {
                if (u >> bit) & 1 == 1 {
                    expected[plan.bit_entry(column, u_off + bit).unwrap()] = Fp3::ONE;
                }
                if (v >> bit) & 1 == 1 {
                    expected[plan.bit_entry(column, v_off + bit).unwrap()] = Fp3::ONE;
                }
            }
        }
        for (column, value) in [0i64, 6, 3, 7].iter().enumerate() {
            for bit in 0..3 {
                if (value >> bit) & 1 == 1 {
                    expected[plan.bit_entry(column, s_off + bit).unwrap()] = Fp3::ONE;
                }
            }
        }
        assert_eq!(bits[0], expected);
    }

    #[test]
    fn closure_values_reject_out_of_range_cells() {
        let plan = plan();
        // Y = 4 needs a D+1 = 3 bit value 7 > 2A = 6.
        let row = vec![signed(4), signed(0), signed(0), signed(0)];
        assert_eq!(
            closure_values_from_rows(&plan, &[row]).unwrap_err(),
            Error::BoundViolation
        );
        // An invalid column far below the winner makes the slack exceed B.
        let row = vec![signed(0), signed(0), signed(0), signed(-4)];
        assert_eq!(
            closure_values_from_rows(&plan, &[row]).unwrap_err(),
            Error::BoundViolation
        );
    }

    #[test]
    fn map_values_are_eq_times_bits_and_booleanity_sum_holds() {
        let r = [Fp3::from_u64(2), Fp3::from_u64(3), Fp3::from_u64(5)];
        let mut bits = vec![Fp3::ZERO; 8];
        bits[1] = Fp3::ONE;
        bits[6] = Fp3::ONE;
        let map = closure_map_values(&bits, &r).unwrap();
        let eq = eq_vector(&r).unwrap();
        for index in 0..8 {
            assert_eq!(map[index], eq[index].mul(&bits[index]));
        }
        let sum: Fp3 = map
            .iter()
            .zip(&bits)
            .fold(Fp3::ZERO, |acc, (d, b)| acc.add(&d.mul(&Fp3::ONE.sub(b))));
        assert_eq!(sum, Fp3::ZERO);

        // A non-boolean cell makes the booleanity sum nonzero for this r.
        let mut bad = bits;
        bad[3] = Fp3::from_u64(2);
        let map = closure_map_values(&bad, &r).unwrap();
        let sum: Fp3 = map
            .iter()
            .zip(&bad)
            .fold(Fp3::ZERO, |acc, (d, b)| acc.add(&d.mul(&Fp3::ONE.sub(b))));
        assert_ne!(sum, Fp3::ZERO);
    }

    #[test]
    fn multilinear_evaluation_matches_the_eq_dot_product() {
        let point = [Fp3::from_u64(3), Fp3::from_u64(5), Fp3::from_u64(7)];
        let values: Vec<Fp3> = (0..8).map(|index| Fp3::from_u64(index + 1)).collect();
        let expected = inner_product(&eq_vector(&point).unwrap(), &values).unwrap();
        assert_eq!(multilinear(&values, &point).unwrap(), expected);
    }

    #[test]
    fn eq_vector_matches_the_product_formula() {
        let point = [Fp3::from_u64(7), Fp3::from_u64(11)];
        let eq = eq_vector(&point).unwrap();
        for index in 0..4 {
            let mut expected = Fp3::ONE;
            for (bit, coordinate) in point.iter().enumerate() {
                let factor = if (index >> bit) & 1 == 1 {
                    *coordinate
                } else {
                    Fp3::ONE.sub(coordinate)
                };
                expected = expected.mul(&factor);
            }
            assert_eq!(eq[index], expected);
        }
        // A consumed coin stream is not needed here; the source type is only
        // named so the module compiles the same imports in every test build.
        let _ = ByteMaskCoins::new(Digest([0x72; 32])).unwrap();
    }

    fn eq_serial(point: &[Fp3]) -> Vec<Fp3> {
        let mut values = vec![Fp3::ONE];
        for coordinate in point.iter().rev() {
            let mut next = vec![Fp3::ZERO; values.len() * 2];
            for (index, value) in values.iter().enumerate() {
                next[2 * index] = value.mul(&Fp3::ONE.sub(coordinate));
                next[2 * index + 1] = value.mul(coordinate);
            }
            values = next;
        }
        values
    }

    #[test]
    fn eq_vector_parallel_matches_serial_across_threshold_and_threads() {
        for k in [0usize, 1, 13, 14, 15] {
            let point = (0..k)
                .map(|index| Fp3::from_u64(index as u64 + 2))
                .collect::<Vec<_>>();
            let expected = eq_serial(&point);
            assert_eq!(eq_vector_with_min(&point, usize::MAX).unwrap(), expected);
            for threads in [1usize, 2, 64] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let got = pool
                    .install(|| eq_vector_with_min(&point, 0).unwrap());
                assert_eq!(got, expected, "k {k} threads {threads}");
            }
            assert_eq!(eq_vector(&point).unwrap(), expected);
        }
    }

    #[test]
    fn closure_map_parallel_matches_serial_across_threshold_and_threads() {
        for len in [(1usize << 14) - 1, 1usize << 14] {
            let eq = (0..len)
                .map(|index| Fp3::from_u64(index as u64 + 3))
                .collect::<Vec<_>>();
            let bits = (0..len)
                .map(|index| Fp3::from_u64((index as u64 % 2) + 1))
                .collect::<Vec<_>>();
            let expected: Vec<Fp3> = eq
                .iter()
                .zip(&bits)
                .map(|(left, right)| left.mul(right))
                .collect();
            assert_eq!(
                closure_map_values_with_eq_min(&bits, &eq, usize::MAX).unwrap(),
                expected
            );
            for threads in [1usize, 2, 64] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let got = pool.install(|| {
                    closure_map_values_with_eq_min(&bits, &eq, 0).unwrap()
                });
                assert_eq!(got, expected, "len {len} threads {threads}");
            }
            assert_eq!(closure_map_values_with_eq(&bits, &eq).unwrap(), expected);
        }
    }
}
