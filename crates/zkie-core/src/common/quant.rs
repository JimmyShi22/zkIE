//! Decoding tables for the low-precision formats used by DeepSeek-V4.1-Flash.
//!
//! All decoders return values scaled by [`FIXED_SCALE`] (2^16). The field
//! element representation used elsewhere in zkIE is integer-valued, so a
//! fractional quantized value `v` becomes `round(v * 2^16)`.

pub const FIXED_SCALE: i64 = 1 << 16;

pub const E2M1_FN_VALUES: [i64; 16] = [
    0,
    32768,
    65536,
    98304,
    131072,
    196608,
    262144,
    393216,
    0,
    -32768,
    -65536,
    -98304,
    -131072,
    -196608,
    -262144,
    -393216,
];

pub fn e2m1_to_i64(code: u8) -> i64 {
    E2M1_FN_VALUES[(code & 0x0F) as usize]
}

pub fn e8m0_to_i64(code: u8) -> i64 {
    // E8M0 is a pure power-of-two format: value = 2^(code - 127). The
    // fixed-point result is round(2^exponent * 2^16). Negative exponents are
    // fractions, and exponents past i64's range saturate.
    let exponent = (code as i32) - 127;
    if exponent >= 47 {
        return i64::MAX;
    }
    if exponent >= 0 {
        1i64 << (FIXED_SCALE.trailing_zeros() as i32 + exponent)
    } else if exponent <= -18 {
        0
    } else {
        round_div(1i64 << FIXED_SCALE.trailing_zeros(), 1i64 << (-exponent))
    }
}

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

pub fn e4m3_to_i64(code: u8) -> i64 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = ((code >> 3) & 0x0F) as i32;
    let mantissa = (code & 0x07) as f64;
    let value = if exponent == 0 {
        mantissa * 2f64.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    };
    (sign * value * FIXED_SCALE as f64).round() as i64
}

pub fn unpack_e2m1(code: u8) -> (i64, i64) {
    // DeepSeek's `convert.py` packs two E2M1 values per byte and expands them
    // as `[low_nibble, high_nibble]`:
    //   low = x & 0x0F
    //   high = (x >> 4) & 0x0F
    //   torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1)
    // The official order is therefore low first, high second.
    let low = e2m1_to_i64(code & 0x0F);
    let high = e2m1_to_i64(code >> 4);
    (low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_decodes_official_table() {
        assert_eq!(e2m1_to_i64(0), 0);
        assert_eq!(e2m1_to_i64(1), 32768);
        assert_eq!(e2m1_to_i64(7), 393216);
        assert_eq!(e2m1_to_i64(15), -393216);
    }

    #[test]
    fn e8m0_decodes_powers_of_two() {
        assert_eq!(e8m0_to_i64(127), 65536);
        assert_eq!(e8m0_to_i64(128), 131072);
        assert_eq!(e8m0_to_i64(126), 32768);
        assert_eq!(e8m0_to_i64(125), 16384);
        assert_eq!(e8m0_to_i64(130), 524288);
        assert_eq!(e8m0_to_i64(0), 0);
    }

    #[test]
    fn e4m3_decodes_common_values() {
        assert_eq!(e4m3_to_i64(0x38), 65536);
        assert_eq!(e4m3_to_i64(0x3C), 98304);
        assert_eq!(e4m3_to_i64(0x40), 131072);
        assert_eq!(e4m3_to_i64(0x7E), 29_360_128);
        assert_eq!(e4m3_to_i64(0xB8), -65536);
    }

    #[test]
    fn unpacked_e2m1_pairs_use_low_then_high_nibble() {
        assert_eq!(unpack_e2m1(0x1F), (-393216, 32768));
        assert_eq!(unpack_e2m1(0x10), (0, 32768));
    }
}
