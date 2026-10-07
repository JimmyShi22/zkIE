//! Shared deterministic integer-table generator used by BOTH PCS schemes.
//!
//! `values(n, table_index)` returns `2^n` integers in 0..99 derived from a
//! xorshift64 seeded by `(n, table_index)`. Both the KZG (BN254) and WHIR
//! (Goldilocks) sides embed these integers into their own fields; the integer
//! patterns are identical, the field elements are not.

pub fn values(n: usize, table_index: usize) -> Vec<u64> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64
        .wrapping_mul((n as u64).wrapping_add(1))
        .wrapping_add((table_index as u64).wrapping_add(1));
    (0..(1usize << n))
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % 100
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_bounded() {
        assert_eq!(values(6, 0), values(6, 0));
        assert_ne!(values(6, 0), values(6, 1));
        assert!(values(6, 0).iter().all(|&v| v < 100));
        assert_eq!(values(6, 0).len(), 64);
    }
}
