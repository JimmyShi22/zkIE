//! Frozen arithmetic profile of the gpt2 final norm for the concise segment.
//!
//! The profile fixes the trusted integer layout of the final layer norm
//! (`ln_f`) used by the production concise chain: the padded hidden width, the
//! valid column count, the output shift, the trusted input bound, the rounding
//! digit widths and offsets, and the output-tail digit widths. The values come
//! from the same-profile derivation recorded for the final norm: the measured
//! maxima of the registered case are below the fixed bounds below, and the
//! plan-level no-wrap check runs over the fixed bounds.
//!
//! The rstd lookup reads the existing frozen public rsqrt table as its view.
//! Its content root equals the public table digest already frozen for the
//! model, so no second table with the same content is created.

use zkie_core::artifact::digest_from_hex;
use zkie_core::error::Result;
use zkie_core::types::Digest;
use zkie_ops::proof::hidden_norm_psc::{
    norm_no_wrap, NormNoWrap, NormOutputTailWidths, NormRoundLayout, NormRoundWidths,
};

/// Padded hidden width of the final norm (real columns 768, padding zero).
pub const HIDDEN_PAD: u32 = 1024;
/// Valid (non-padded) columns of the final norm input.
pub const VALID: u32 = 768;
/// Output shift of the final norm's rounding.
pub const OUTPUT_SHIFT: u32 = 32;
/// Trusted bound of one input cell, `|x| <= 2^25`.
pub const INPUT_BOUND: u64 = 1 << 25;
/// Fixed offset of the sum digit block; larger than the measured `|sum|` max.
pub const SUM_OFFSET: u64 = 1 << 25;
/// Fixed offset of the mean digit block; larger than the measured `|mean|` max.
pub const MEAN_OFFSET: u64 = 1 << 16;
/// Real length of the frozen public rstd table.
pub const RSTD_TABLE_REAL_LEN: u64 = 7_340_032;
/// Row-count bit width of the frozen public rstd table view.
pub const RSTD_TABLE_LOG: u32 = 23;

/// The trusted maximum of the rstd table entries used by the registered
/// case. Measured evidence: the used table index range is 1,055,139 through
/// 1,070,937 and the largest entry in that range is 5,078; the plan bound is
/// the next power of two, 8,192. It is a trusted plan constant, not a proof
/// value.
pub const RSTD_VALUE_BOUND: u64 = 8_192;
/// The first table row value, the lookup padding pair's value entry.
pub const RSTD_FIRST_VALUE: i64 = 65_536_000;
/// The largest absolute gain entry of the registered final norm asset.
pub const GAIN_MAX: i64 = 1_141_592;
/// The observed largest absolute centered input row entry, for the record.
pub const CENTERED_OBSERVED: i64 = 22_823_876;
/// The trusted no-op clamp interval of the segment: the registered model has
/// no clamp, and the chain proves the clamp rows over the symmetric interval
/// `[-CLAMP_BOUND, CLAMP_BOUND]`. The observed largest absolute pre-clamp
/// value is 14,562,155, below the bound, so the clamp is the identity on the
/// honest range; the equivalence is carried by the trusted input contract.
pub const CLAMP_BOUND: i64 = 1 << 24;
/// The observed largest absolute pre-clamp value, for the record.
pub const PRE_CLAMP_OBSERVED: i64 = 14_562_155;

/// The public content root of the frozen rstd table, the same digest the
/// model plan already freezes for the shared rsqrt table.
const RSTD_CONTENT_ROOT: &str =
    "7cba6a7c88053b99112c1271a2d593e1147a4b3b4a92fe1091c148e2e33573ed";

/// The rounding digit widths of the final norm profile.
pub fn round_widths() -> NormRoundWidths {
    NormRoundWidths {
        sum: 5,
        squares: 7,
        variance: 6,
        mean: 3,
        offset: 2,
    }
}

/// The output-tail digit widths of the final norm profile with the output
/// shift fixed to `2^32`.
pub fn tail_widths() -> NormOutputTailWidths {
    NormOutputTailWidths {
        quotient: 3,
        remainder: 4,
        doubled_remainder: 5,
        tie_slack: 4,
        shifted: 3,
        pre_clamp: 3,
    }
}

/// The planned rounding layout with the two fixed signed-value offsets.
pub fn round_layout() -> Result<NormRoundLayout> {
    NormRoundLayout::new(round_widths(), VALID, SUM_OFFSET, MEAN_OFFSET)
}

/// The plan-level no-wrap verdict over the trusted input bound.
pub fn no_wrap() -> Result<NormNoWrap> {
    norm_no_wrap(INPUT_BOUND, VALID, &round_widths(), SUM_OFFSET, MEAN_OFFSET)
}

/// The content root of the frozen rstd table view.
pub fn rstd_content_root() -> Digest {
    digest_from_hex(RSTD_CONTENT_ROOT).expect("the fixed rstd content root")
}

/// The plan identity of the frozen rstd table view. The content root is the
/// binding value; the identity is fixed with it in the trusted plan.
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
) -> zkie_core::error::Result<zkie_ops::proof::norm_segment::NormSegmentProfile> {
    Ok(zkie_ops::proof::norm_segment::NormSegmentProfile {
        positions,
        hidden: HIDDEN_PAD,
        valid: VALID,
        columns: u64::from(HIDDEN_PAD),
        shift: OUTPUT_SHIFT,
        clamp: Some((-CLAMP_BOUND, CLAMP_BOUND)),
        clamp_widths: Some(zkie_ops::proof::hidden_norm_psc::NormClampWidths {
            low: 4,
            high: 4,
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

    #[test]
    fn the_round_layout_is_admitted() {
        round_layout().expect("the fixed rounding layout");
    }

    #[test]
    fn the_tail_widths_cover_the_output_shift() {
        assert!(tail_widths().fault(OUTPUT_SHIFT).is_none());
    }

    #[test]
    fn the_no_wrap_bounds_stay_below_half_the_modulus() {
        let bounds = no_wrap().expect("the fixed bounds");
        let half = (u128::from(u64::MAX) / 2) as u128;
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
            assert!(value < half);
        }
    }

    #[test]
    fn the_no_op_clamp_bound_covers_the_honest_range() {
        assert!(PRE_CLAMP_OBSERVED.abs() < CLAMP_BOUND);
        let profile = segment_profile(16, zkie_core::types::Digest([0x91; 32])).unwrap();
        assert!(profile.no_op_clamp);
        assert_eq!(profile.clamp, Some((-CLAMP_BOUND, CLAMP_BOUND)));
        assert!(profile.clamp_widths.is_some());
    }

    #[test]
    fn the_segment_profile_and_rstd_view_are_consistent() {
        let profile = segment_profile(16, zkie_core::types::Digest([0x91; 32])).unwrap();
        profile.validate().unwrap();
        let table = rstd_table_config();
        assert_eq!(table.real_len as u64, RSTD_TABLE_REAL_LEN);
        assert_eq!(table.content_root, rstd_content_root());
        assert_eq!(table.identity, rstd_table_identity());
    }

    #[test]
    fn the_rstd_view_matches_the_public_table_digest() {
        let root = rstd_content_root();
        assert_eq!(
            root,
            digest_from_hex(
                "7cba6a7c88053b99112c1271a2d593e1147a4b3b4a92fe1091c148e2e33573ed"
            )
            .unwrap()
        );
        assert_eq!(RSTD_TABLE_REAL_LEN, 7_340_032);
        assert_eq!(1u64 << RSTD_TABLE_LOG, 8_388_608);
        assert!(RSTD_TABLE_REAL_LEN < (1u64 << RSTD_TABLE_LOG));
    }
}
