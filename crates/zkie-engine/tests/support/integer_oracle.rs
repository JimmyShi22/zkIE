//! Independent exact-integer oracle used by the primitive tests.
//!
//! This module deliberately does not call the implementation under test.

/// `rq(a, k) = floor((a + 2^(k-1)) / 2^k)` through Euclidean division.
pub fn rq_oracle(a: i128, k: u32) -> i128 {
    if k == 0 {
        return a;
    }
    let divisor = 1i128 << k;
    (a + divisor / 2).div_euclid(divisor)
}

pub fn euclid_oracle(a: i128, b: i128) -> (i128, i128) {
    assert_ne!(b, 0);
    (a.div_euclid(b), a.rem_euclid(b))
}

pub fn add_oracle(a: u128, b: u128) -> Option<u128> {
    a.checked_add(b)
}

pub fn mul_oracle(a: u128, b: u128) -> Option<u128> {
    a.checked_mul(b)
}

pub fn div_oracle(a: u128, b: u128) -> Option<(u128, u128)> {
    if b == 0 {
        None
    } else {
        Some((a / b, a % b))
    }
}
