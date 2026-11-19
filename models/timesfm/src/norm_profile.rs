//! Frozen arithmetic profile of the first per-layer RMS norm (L0 attention).
//!
//! The profile fixes the trusted integer layout of the L0 pre-attention RMS
//! norm: padded hidden width, valid column count, output shift, input bound,
//! rounding digit widths and offsets, the no-op clamp interval used as a width
//! bound, and the output-tail digit widths. The values come from the same
//! profile derivation recorded for the registered case; the measured maxima
//! are below the fixed bounds and the plan-level no-wrap check runs over the
//! fixed bounds.
//!
//! The rstd lookup reads the existing frozen public rsqrt table as its view.

use zkie_core::artifact::digest_from_hex;
use zkie_core::error::Result;
use zkie_core::types::Digest;
use zkie_ops::proof::hidden_norm_psc::{
    NormNoWrap, NormOutputTailWidths, NormRoundLayout, NormRoundWidths,
};

/// Padded hidden width of the L0 norm (real columns 1280, padding zero).
pub const HIDDEN_PAD: u32 = 2048;
/// Valid (non-padded) columns of the L0 norm input.
pub const VALID: u32 = 1280;
/// Output shift of the norm rounding.
pub const OUTPUT_SHIFT: u32 = 32;
/// Trusted bound of one input cell, `|x| <= 2^17`.
pub const INPUT_BOUND: u64 = 1 << 17;
/// Fixed offset of the sum digit block; larger than the measured `|sum|` max.
pub const SUM_OFFSET: u64 = 1 << 28;
/// Fixed offset of the mean digit block; larger than the measured `|mean|` max.
pub const MEAN_OFFSET: u64 = 1 << 18;
/// Trusted maximum of the gain entries used by the registered case.
pub const GAIN_MAX: i64 = 1 << 17;
/// Observed largest absolute input cell, for the record.
pub const INPUT_OBSERVED: i64 = 115_420;
/// Observed largest absolute centered input cell, for the record.
pub const CENTERED_OBSERVED: i64 = 140_127;
/// The used rstd table index range and its largest entry, for the record.
pub const RSTD_INDEX_MIN: u64 = 3_772;
pub const RSTD_INDEX_MAX: u64 = 6_240;
pub const RSTD_OBSERVED: i64 = 136_585;
/// Trusted maximum of the rstd entries used by the registered case.
pub const RSTD_VALUE_BOUND: u64 = 1 << 18;
/// Real length of the frozen public rstd table.
pub const RSTD_TABLE_REAL_LEN: u64 = 7_340_032;
/// Row-count bit width of the frozen public rstd table view.
pub const RSTD_TABLE_LOG: u32 = 23;
/// The observed largest absolute pre-clamp value, for the record.
pub const PRE_CLAMP_OBSERVED: i64 = 246_852;
/// The no-op clamp interval is the signed-offset domain of the four-byte
/// pre-clamp cell. Every reachable pre-clamp value lies strictly inside it, so
/// the clamp selectors are forced to zero by the chain's own sign relations and
/// the output equals the pre-clamp value in-chain.
pub const CLAMP_LOW: i64 = -(1 << 31);
pub const CLAMP_HIGH: i64 = (1 << 31) - 1;

/// The public content root of the frozen rstd table, the same digest the model
/// plan already freezes for the shared rsqrt table.
const RSTD_CONTENT_ROOT: &str =
    "7cba6a7c88053b99112c1271a2d593e1147a4b3b4a92fe1091c148e2e33573ed";

/// The rounding digit widths of the L0 norm profile.
pub fn round_widths() -> NormRoundWidths {
    NormRoundWidths {
        sum: 6,
        squares: 6,
        variance: 6,
        mean: 5,
        offset: 2,
    }
}

/// The output-tail digit widths of the L0 norm profile with the output shift
/// fixed to `2^32`.
pub fn tail_widths() -> NormOutputTailWidths {
    NormOutputTailWidths {
        quotient: 4,
        remainder: 4,
        doubled_remainder: 5,
        tie_slack: 4,
        shifted: 4,
        pre_clamp: 4,
    }
}

/// The planned rounding layout with the two fixed signed-value offsets.
pub fn round_layout() -> Result<NormRoundLayout> {
    NormRoundLayout::new(round_widths(), VALID, SUM_OFFSET, MEAN_OFFSET)
}

/// The plan-level no-wrap verdict over the trusted input bound.
pub fn no_wrap() -> Result<NormNoWrap> {
    zkie_ops::proof::hidden_norm_psc::norm_no_wrap_mode(
        INPUT_BOUND,
        VALID,
        &round_widths(),
        SUM_OFFSET,
        MEAN_OFFSET,
        zkie_ops::proof::hidden_norm::NormMode::RmsNorm,
    )
}

/// The content root of the frozen rstd table view.
pub fn rstd_content_root() -> Digest {
    digest_from_hex(RSTD_CONTENT_ROOT).expect("the fixed rstd content root")
}

/// The plan identity of the frozen rstd table view.
pub fn rstd_table_identity() -> Digest {
    rstd_content_root()
}

/// The trusted frozen-table configuration of the rstd lookup view.
pub fn rstd_table_config() -> zkie_ops::proof::lookup_table::TableConfig {
    zkie_ops::proof::lookup_table::TableConfig {
        identity: rstd_table_identity(),
        content_root: rstd_content_root(),
        real_len: RSTD_TABLE_REAL_LEN as usize,
        encoding: zkie_ops::proof::lookup_table::TableEncoding::I32Le,
        version: zkie_ops::proof::lookup_table::TABLE_ENCODING_VERSION,
    }
}

/// The trusted segment profile of one request shape.
pub fn segment_profile(
    positions: u32,
    x_seed: Digest,
) -> Result<zkie_ops::proof::norm_segment::NormSegmentProfile> {
    Ok(zkie_ops::proof::norm_segment::NormSegmentProfile {
        positions,
        hidden: HIDDEN_PAD,
        valid: VALID,
        columns: u64::from(HIDDEN_PAD),
        mode: zkie_ops::proof::hidden_norm::NormMode::RmsNorm,
        shift: OUTPUT_SHIFT,
        clamp: Some((CLAMP_LOW, CLAMP_HIGH)),
        clamp_widths: Some(zkie_ops::proof::hidden_norm_psc::NormClampWidths {
            low: 5,
            high: 5,
            low_strict: 4,
            low_nonstrict: 4,
            high_strict: 4,
            high_nonstrict: 4,
        }),
        no_op_clamp: true,
        input_bound: INPUT_BOUND,
        sum_offset: SUM_OFFSET,
        mean_offset: MEAN_OFFSET,
        round_widths: round_widths(),
        tail_widths: tail_widths(),
        table_real_len: RSTD_TABLE_REAL_LEN,
        x_seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::field::MODULUS;

    #[test]
    fn the_round_layout_is_admitted() {
        round_layout().expect("the fixed rounding layout");
    }

    #[test]
    fn the_tail_widths_cover_the_output_shift() {
        assert!(tail_widths().fault(OUTPUT_SHIFT).is_none());
    }

    #[test]
    fn the_no_wrap_bounds_stay_below_the_modulus() {
        let bounds = no_wrap().expect("the fixed bounds");
        for value in [
            bounds.relation_sum,
            bounds.relation_squares,
            bounds.relation_mean,
            bounds.relation_variance,
            bounds.reconstruction,
            bounds.offsets,
            bounds.remainder,
            bounds.rsqrt,
        ] {
            assert!(value < MODULUS as u128);
        }
    }

    #[test]
    fn the_clamp_interval_is_the_pre_cell_domain() {
        assert!(PRE_CLAMP_OBSERVED > CLAMP_LOW && PRE_CLAMP_OBSERVED <= CLAMP_HIGH);
        let profile = segment_profile(16, Digest([0x91; 32])).unwrap();
        assert!(profile.no_op_clamp);
        assert_eq!(profile.clamp, Some((CLAMP_LOW, CLAMP_HIGH)));
        let widths = profile.clamp_widths.expect("the clamp rows");
        assert_eq!(profile.tail_widths.pre_clamp, 4);
        assert_eq!((widths.low, widths.high), (5, 5));
    }

    #[test]
    fn the_segment_profile_and_rstd_view_are_consistent() {
        let profile = segment_profile(16, Digest([0x91; 32])).unwrap();
        profile.validate().unwrap();
        let table = rstd_table_config();
        assert_eq!(table.real_len as u64, RSTD_TABLE_REAL_LEN);
        assert_eq!(table.content_root, rstd_content_root());
        assert_eq!(table.identity, rstd_table_identity());
    }

    #[test]
    fn the_observed_bounds_fit_the_declared_bounds() {
        assert!(INPUT_OBSERVED.unsigned_abs() <= INPUT_BOUND);
        assert!(CENTERED_OBSERVED.unsigned_abs() <= 2 * INPUT_BOUND);
        assert!(RSTD_OBSERVED.unsigned_abs() as u64 <= RSTD_VALUE_BOUND);
        assert!(GAIN_MAX >= 80_546);
        assert!(RSTD_INDEX_MIN < RSTD_INDEX_MAX);
        assert!(RSTD_INDEX_MAX < RSTD_TABLE_REAL_LEN);
    }
}
