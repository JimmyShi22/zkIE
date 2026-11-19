//! Frozen arithmetic profile of the Gemma 3 final RMS norm for the concise
//! segment.
//!
//! The profile fixes the trusted integer layout of the final RMS norm used by
//! the production concise chain: the padded hidden width, the valid column
//! count, the output shift, the trusted input bound, the mean-square bound of
//! the authenticated row check, the correlated pre-clamp bound, the rounding
//! digit widths and the output-tail digit widths.
//!
//! The RMS statistic is the rounded mean square of the valid cells. Its sum
//! of squares can reach `640 * (2^33)^2 < 2^75.4`, so it exists only as
//! base-256 digits and carry chains; the exact arithmetic is in
//! [`zkie_ops::proof::rms_limbs`]. The rounded mean square itself is below
//! `2^62` under the authenticated-data check, so it enters the rsqrt relations
//! as an integer field element.
//!
//! The rstd lookup reads the existing frozen public rsqrt table as its base
//! and the frozen public extension as its continuation: rows below
//! `RSTD_BASE_LEN` come from the base file, the rest from the extension. The
//! concatenated content root is frozen with the identity.

use zkie_core::artifact::{digest_from_hex, digest_bytes};
use zkie_core::error::Result;
use zkie_core::types::Digest;
use zkie_ops::proof::hidden_norm::NormMode;
use zkie_ops::proof::hidden_norm_psc::{
    NormClampWidths, NormOutputTailWidths, NormRoundWidths,
};
use zkie_ops::proof::lookup_table::{
    TableConfig, TableEncoding, TABLE_ENCODING_VERSION,
};

/// Padded hidden width of the final norm (real columns 640, padding zero).
pub const HIDDEN_PAD: u32 = 1024;
/// Valid (non-padded) columns of the final norm input.
pub const VALID: u32 = 640;
/// Output shift of the final norm's rounding.
pub const OUTPUT_SHIFT: u32 = 32;
/// Trusted bound of one input cell, `|x| <= 2^33`. The measured maximum of the
/// registered case is 7,859,748,442, below this bound.
pub const INPUT_BOUND: u64 = 1 << 33;
/// Trusted bound of the rounded mean square. The measured maximum of the
/// registered case is 97,721,861,469,415,282 (2^56.44); the bound keeps a wide
/// margin and makes the two-half reconstruction of the mean stay below the
/// field half range.
pub const MEAN_BOUND: u64 = 1 << 61;
/// The largest absolute gain entry of the registered final norm asset.
pub const GAIN_MAX: i64 = 19_726_336;
/// Real length of the combined frozen rstd table.
pub const RSTD_TABLE_REAL_LEN: u64 = 22_189_295;
/// Row-count bit width of the combined frozen rstd table view.
pub const RSTD_TABLE_LOG: u32 = 25;
/// Base table length of the combined view.
pub const RSTD_BASE_LEN: u64 = 7_340_032;
/// The trusted maximum of the rstd entries reachable by the registered case.
pub const RSTD_VALUE_BOUND: u64 = 64;
/// The frozen table value at the index of `INPUT_BOUND^2 / VALID`, used for
/// the correlated pre-clamp bound. Evidence: index 11,937,366 reads 13.
pub const RSTD_AT_INPUT_BOUND: i64 = 13;
/// The correlated pre-clamp bound: `2 * X * GainMax * rstd(index(X^2/valid))
/// + 1`. The factor two is the recorded safety margin for the sampled peak of
/// `M * rstd(M^2/valid)`; the bound stays below `2^63`.
pub const PRE_BOUND: u64 = 4_405_646_335_475_187_713;
/// The clamp interval of the segment is the signed-offset domain of the
/// four-byte pre-clamp cell. Every reachable pre-clamp value lies strictly
/// inside it, so the clamp selectors are forced to zero by the chain's own
/// sign relations and the output equals the pre-clamp value in-chain.
pub const CLAMP_LOW: i64 = -(1 << 31);
pub const CLAMP_HIGH: i64 = (1 << 31) - 1;

/// The public content root of the combined rstd table, over the base file
/// followed by the extension file in encoded row order.
const RSTD_CONTENT_ROOT: &str =
    "15f2a688cc4c695e50d518a88c67c76113ae92eccbd6a745c319e1e021411f10";

/// The rounding digit widths of the final norm profile. The sum, squares,
/// variance and mean blocks are the shared digit slots; the multi-limb RMS
/// rows carry their own digit and carry slots.
pub fn round_widths() -> NormRoundWidths {
    NormRoundWidths {
        sum: 7,
        squares: 7,
        variance: 7,
        mean: 7,
        offset: 3,
    }
}

/// The output-tail digit widths of the final norm profile with the output
/// shift fixed to `2^32`.
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

/// The content root of the combined frozen rstd table view.
pub fn rstd_content_root() -> Digest {
    digest_from_hex(RSTD_CONTENT_ROOT).expect("the fixed rstd content root")
}

/// The plan identity of the combined frozen rstd table view.
pub fn rstd_table_identity() -> Digest {
    rstd_content_root()
}

/// The trusted frozen-table configuration of the rstd lookup view.
pub fn rstd_table_config() -> TableConfig {
    TableConfig {
        identity: rstd_table_identity(),
        content_root: rstd_content_root(),
        real_len: RSTD_TABLE_REAL_LEN as usize,
        encoding: TableEncoding::I32Le,
        version: TABLE_ENCODING_VERSION,
    }
}

/// The multi-limb widths of the sum of squares and the mean square.
pub fn rms_widths() -> Result<zkie_ops::proof::rms_limbs::RmsLimbWidths> {
    zkie_ops::proof::rms_limbs::RmsLimbWidths::derive(INPUT_BOUND, VALID)
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
        mode: NormMode::RmsNorm,
        shift: OUTPUT_SHIFT,
        clamp: Some((CLAMP_LOW, CLAMP_HIGH)),
        clamp_widths: Some(NormClampWidths {
            low: 5,
            high: 5,
            low_strict: 4,
            low_nonstrict: 4,
            high_strict: 4,
            high_nonstrict: 4,
        }),
        no_op_clamp: true,
        input_bound: INPUT_BOUND,
        sum_offset: 0,
        mean_offset: 0,
        round_widths: round_widths(),
        tail_widths: tail_widths(),
        table_real_len: RSTD_TABLE_REAL_LEN,
        mean_bound: MEAN_BOUND,
        pre_bound: PRE_BOUND,
        x_seed,
    })
}

/// The digest of the plan-level bounds that the segment absorbs. The value is
/// recorded with its derivation in NOTES.
pub fn bounds_digest() -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&INPUT_BOUND.to_le_bytes());
    bytes.extend_from_slice(&MEAN_BOUND.to_le_bytes());
    bytes.extend_from_slice(&(GAIN_MAX as u64).to_le_bytes());
    bytes.extend_from_slice(&RSTD_VALUE_BOUND.to_le_bytes());
    bytes.extend_from_slice(&(RSTD_AT_INPUT_BOUND as u64).to_le_bytes());
    bytes.extend_from_slice(&PRE_BOUND.to_le_bytes());
    digest_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_round_layout_is_admitted() {
        let widths = round_widths();
        assert!(widths.fault(VALID).is_none());
        assert!(tail_widths().fault(OUTPUT_SHIFT).is_none());
        let profile = segment_profile(16, Digest([0x91; 32])).unwrap();
        assert_eq!(profile.mode, NormMode::RmsNorm);
        assert!(profile.no_op_clamp);
        assert_eq!(profile.clamp, Some((CLAMP_LOW, CLAMP_HIGH)));
    }

    #[test]
    fn the_correlated_bounds_are_consistent() {
        let x = INPUT_BOUND as u128;
        let gain = GAIN_MAX as u128;
        let bound = 2 * x * gain * RSTD_AT_INPUT_BOUND as u128 + 1;
        assert_eq!(bound, PRE_BOUND as u128);
        // The 752 window cover: pre bound plus the reconstruction window.
        let reconstruction = ((1u128 << 31) - 1) * (1u128 << 32) + ((1u128 << 32) - 1);
        assert!(bound + reconstruction < (1u128 << 64) - (1u128 << 32) + 1);
    }

    #[test]
    fn the_rstd_view_matches_the_combined_root_and_length() {
        let table = rstd_table_config();
        assert_eq!(table.real_len as u64, RSTD_TABLE_REAL_LEN);
        assert_eq!(table.content_root, rstd_content_root());
        assert_eq!(table.identity, rstd_table_identity());
        assert_eq!(
            RSTD_CONTENT_ROOT,
            "15f2a688cc4c695e50d518a88c67c76113ae92eccbd6a745c319e1e021411f10"
        );
        assert!(RSTD_TABLE_REAL_LEN < (1u64 << RSTD_TABLE_LOG));
        assert!(RSTD_BASE_LEN < RSTD_TABLE_REAL_LEN);
    }

    #[test]
    fn the_real_multi_limb_bounds_are_admitted() {
        zkie_ops::proof::hidden_norm_psc::norm_rms_no_wrap(
            INPUT_BOUND,
            VALID,
            &rms_widths().unwrap(),
            MEAN_BOUND,
        )
        .expect("the real RMS no-wrap bounds");
    }

    #[test]
    fn the_rms_widths_cover_the_bound() {
        let widths = rms_widths().unwrap();
        assert_eq!(widths.s_ll, 5);
        assert_eq!(widths.s_lh, 6);
        assert_eq!(widths.s_hh, 6);
        assert_eq!(widths.v, 10);
    }
}
