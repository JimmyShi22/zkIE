//! Kernel-versus-oracle checks and determinism under different schedules.

use std::sync::Arc;

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::Error;
use zkie_core::types::Digest;

use zkie_engine::compute::cpu::{
    batched_matmul_fp, broadcast_row_major, dft_fp, dft_root, div_rem_floor, field_matmul_fp,
    field_matmul_fp3, field_sum_fp, field_sum_fp3, fixed_matmul, fixed_matmul_rows, gather_matmul, integer_max,
    integer_sum, isqrt_u128, leaf_tag, limb_add, limb_decompose, limb_recompose,
    lookup_accumulate_counts, lookup_accumulate_fp, merkle_build, merkle_leaf, merkle_node,
    merkle_verify, mle_fold_fp, mle_fold_fp3, parallel_fixed_matmul, reshape_row_major,
    transpose_2d,
};
use zkie_engine::parallel::ThreadPool;
use zkie_engine::pcs::request::field::{fp_from_u64, fp_pow, fp_to_u64, Fp, Fp3};
use zkie_engine::primitives::transcript::LEAF_MAIN;

fn fp(value: u64) -> Fp {
    fp_from_u64(value)
}

fn hex_bytes(value: &str) -> Vec<u8> {
    (0..value.len() / 2)
        .map(|index| u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap())
        .collect()
}

#[test]
fn fixed_matmul_is_exact_and_checked() {
    let a = vec![1i64, -2, 3, 4, 5, -6];
    let b = vec![7i64, 8, -9, 10, 11, -12];
    let out = fixed_matmul(&a, &b, 2, 2, 3).unwrap();
    assert_eq!(out, vec![58, -48, -83, 154]);

    assert!(matches!(
        fixed_matmul(&[i64::MAX; 3], &[i64::MAX; 3], 1, 1, 3),
        Err(Error::BoundViolation)
    ));
    assert!(matches!(fixed_matmul(&[1], &[1], 2, 2, 2), Err(Error::ShapeMismatch)));
}

#[test]
fn field_matmul_matches_the_scalar_oracle() {
    let a: Vec<Fp> = (1..=6).map(fp).collect();
    let b: Vec<Fp> = (7..=12).map(|value| fp(value + 5)).collect();
    let out = field_matmul_fp(&a, &b, 2, 2, 3).unwrap();
    let mut expected = vec![Fp::ZERO; 4];
    for row in 0..2 {
        for column in 0..2 {
            let mut sum = Fp::ZERO;
            for inner in 0..3 {
                sum = sum + a[row * 3 + inner] * b[inner * 2 + column];
            }
            expected[row * 2 + column] = sum;
        }
    }
    assert_eq!(out, expected);

    let a3 = vec![
        Fp3::new(fp(1), fp(2), fp(3)),
        Fp3::new(fp(4), fp(5), fp(6)),
    ];
    let b3 = vec![
        Fp3::new(fp(7), fp(8), fp(9)),
        Fp3::new(fp(10), fp(11), fp(12)),
    ];
    let out3 = field_matmul_fp3(&a3, &b3, 1, 1, 2).unwrap();
    assert_eq!(out3[0], a3[0].mul(&b3[0]).add(&a3[1].mul(&b3[1])));
}

#[test]
fn batched_matmul_equals_per_batch_calls() {
    let a: Vec<Fp> = vec![1, 2, 3, 4, 5, 6, 7, 8].into_iter().map(fp).collect();
    let b: Vec<Fp> = vec![2, 3, 4, 5, 6, 7, 8, 9].into_iter().map(fp).collect();
    let batched = batched_matmul_fp(&a, &b, 2, 2, 2, 2).unwrap();
    let mut expected = Vec::new();
    for batch in 0..2 {
        expected.extend(field_matmul_fp(&a[batch * 4..batch * 4 + 4], &b[batch * 4..batch * 4 + 4], 2, 2, 2).unwrap());
    }
    assert_eq!(batched, expected);
}

#[test]
fn gather_uses_only_the_complete_table_and_rejects_bad_indices() {
    let table = vec![1i64, 2, 3, 4, 5, 6];
    let out = gather_matmul(&table, 3, 2, &[2, 0]).unwrap();
    assert_eq!(out, vec![5, 6, 1, 2]);
    assert!(matches!(gather_matmul(&table, 3, 2, &[3]), Err(Error::InvalidIndex)));
}

#[test]
fn mle_fold_matches_the_direct_expansion() {
    let values = vec![fp(1), fp(2), fp(3), fp(4)];
    let r0 = fp(5);
    let r1 = fp(7);
    let folded = mle_fold_fp(&values, &[r0, r1]).unwrap();
    let one = Fp::ONE;
    let expected = (one - r0) * (one - r1) * values[0]
        + r0 * (one - r1) * values[1]
        + (one - r0) * r1 * values[2]
        + r0 * r1 * values[3];
    assert_eq!(folded, expected);
    assert!(matches!(mle_fold_fp(&values, &[r0]), Err(Error::ShapeMismatch)));

    let values3 = vec![
        Fp3::new(fp(1), fp(2), fp(3)),
        Fp3::new(fp(4), fp(5), fp(6)),
    ];
    let challenge = Fp3::new(fp(9), fp(10), fp(11));
    let folded3 = mle_fold_fp3(&values3, &[challenge]).unwrap();
    let expected3 = values3[0].add(&challenge.mul(&values3[1].sub(&values3[0])));
    assert_eq!(folded3, expected3);
}

#[test]
fn reductions_are_exact_and_deterministic() {
    assert_eq!(integer_sum(&[1, -2, 3]).unwrap(), 2);
    assert_eq!(integer_max(&[3, 1, 3, -4]), Some((3, 0)));
    assert_eq!(integer_max(&[]), None);
    assert_eq!(fp_to_u64(field_sum_fp(&[fp(1), fp(2), fp(3)])), 6);
    let values = [Fp3::ONE, Fp3::new(fp(2), fp(3), fp(4))];
    assert_eq!(field_sum_fp3(&values), Fp3::new(fp(3), fp(3), fp(4)));
}

#[test]
fn dft_matches_the_naive_transform() {
    let values: Vec<Fp> = (1..=8).map(fp).collect();
    let transformed = dft_fp(&values).unwrap();
    let root = dft_root(3).unwrap();
    for index in 0..8 {
        let mut expected = Fp::ZERO;
        for (position, value) in values.iter().enumerate() {
            expected = expected + *value * fp_pow(root, (position * index) as u64);
        }
        assert_eq!(transformed[index], expected);
    }
    assert!(matches!(dft_fp(&[fp(1), fp(2), fp(3)]), Err(Error::InvalidConfig)));
    assert!(matches!(dft_fp(&[]), Err(Error::InvalidConfig)));
}

#[test]
fn merkle_hash_matches_independent_layout_vectors() {
    let start = Digest((0..32).collect::<Vec<_>>().try_into().unwrap());
    let tag = leaf_tag(LEAF_MAIN, 0, 4, 40).unwrap();
    assert_eq!(tag, 360_287_987_369_509_028);
    let symbols = [1u64, 2, 3, 4];
    let salt: Vec<u8> = (40..80).collect();
    let leaf = merkle_leaf(tag, 7, &start, &symbols, &salt).unwrap();
    assert_eq!(
        leaf.0.to_vec(),
        hex_bytes("212ba08070528f7adaa4bc5e716753a97f9850d35e593bbf423202ada16c34d0")
    );
    assert!(matches!(
        merkle_leaf(tag, 7, &start, &symbols[..3], &salt),
        Err(Error::InvalidEncoding)
    ));

    let left = Digest([1u8; 32]);
    let right = Digest([2u8; 32]);
    let node = merkle_node(166, 0, &start, &left, &right).unwrap();
    assert_eq!(
        node.0.to_vec(),
        hex_bytes("9e72277a6734af014240b798d1c61621d7f0e708764faec7efca275583e3c39d")
    );
}

#[test]
fn merkle_tree_is_perfect_and_rejects_non_power_of_two() {
    let start = Digest([3u8; 32]);
    let leaves: Vec<Digest> = (0..4).map(|index| Digest([index as u8 + 1; 32])).collect();
    let tree = merkle_build(&start, &leaves).unwrap();
    assert_eq!(tree.leaf_count, 4);
    assert_eq!(tree.padded_count, 4);
    for index in 0..4 {
        let path = tree.open(index).unwrap();
        assert!(merkle_verify(&start, &tree.root, index, &leaves[index], &path).unwrap());
        let mut tampered = path.clone();
        tampered[0] = Digest([99u8; 32]);
        assert!(!merkle_verify(&start, &tree.root, index, &leaves[index], &tampered).unwrap());
    }
    assert!(matches!(tree.open(4), Err(Error::InvalidIndex)));
    assert!(matches!(merkle_build(&start, &[]), Err(Error::InvalidConfig)));
    for length in [1usize, 2, 8] {
        let leaves: Vec<Digest> = (0..length).map(|index| Digest([index as u8 + 1; 32])).collect();
        let tree = merkle_build(&start, &leaves).unwrap();
        assert_eq!(tree.leaf_count, length);
    }
    for length in [3usize, 5, 6, 7] {
        let leaves: Vec<Digest> = (0..length).map(|index| Digest([index as u8 + 1; 32])).collect();
        assert!(matches!(merkle_build(&start, &leaves), Err(Error::InvalidConfig)));
    }
}

#[test]
fn lookup_counts_and_field_accumulation_are_separate() {
    let mut counts = vec![0u64; 4];
    lookup_accumulate_counts(&mut counts, &[1, 1, 3]).unwrap();
    assert_eq!(counts, vec![0, 2, 0, 1]);
    assert!(matches!(
        lookup_accumulate_counts(&mut counts, &[9]),
        Err(Error::InvalidIndex)
    ));
    let mut saturated = vec![u64::MAX];
    assert!(matches!(
        lookup_accumulate_counts(&mut saturated, &[0]),
        Err(Error::ResourceLimit)
    ));
    let mut table = vec![Fp::ZERO; 3];
    lookup_accumulate_fp(&mut table, &[0, 2, 2]).unwrap();
    assert_eq!(fp_to_u64(table[0]), 1);
    assert_eq!(fp_to_u64(table[2]), 2);
}

#[test]
fn integer_relations_are_exact() {
    for (a, d) in [(7i128, 3i128), (-7, 3), (0, 5), (i128::MAX - 1, 7)] {
        let (quotient, remainder) = div_rem_floor(a, d).unwrap();
        assert_eq!(quotient * d + remainder, a);
        assert!(remainder >= 0 && remainder < d);
    }
    assert!(matches!(div_rem_floor(1, 0), Err(Error::DivisionByZero)));
    assert!(matches!(div_rem_floor(1, -1), Err(Error::InvalidConfig)));

    for value in [0u128, 1, 2, 3, 4, 15, 16, 17, u64::MAX as u128] {
        let root = isqrt_u128(value);
        assert!(root * root <= value);
        assert!((root + 1).checked_mul(root + 1).map_or(true, |square| square > value));
    }
    let large = (1u128 << 100) + 12345;
    let root = isqrt_u128(large);
    assert!(root * root <= large && (root + 1) * (root + 1) > large);

    for base_bits in [8u32, 16, 32] {
        let value = 0x0123_4567_89ab_cdefi128;
        let limbs = limb_decompose(value, base_bits).unwrap();
        assert_eq!(limb_recompose(&limbs, base_bits).unwrap(), value);
    }
    assert!(matches!(limb_decompose(-1, 16), Err(Error::BoundViolation)));
    assert!(matches!(limb_recompose(&[1 << 16], 16), Err(Error::InvalidEncoding)));

    let (sum, carry) = limb_add(&[0xffff, 0xffff], &[1], 16).unwrap();
    assert_eq!(carry, 1);
    assert_eq!(sum, vec![0, 0]);
}

#[test]
fn layout_is_index_safe_and_deterministic() {
    let input = vec![1, 2, 3, 4, 5, 6];
    assert_eq!(transpose_2d(&input, 2, 3).unwrap(), vec![1, 4, 2, 5, 3, 6]);
    assert_eq!(transpose_2d::<i32>(&[], 0, 3).unwrap(), Vec::<i32>::new());
    assert!(matches!(transpose_2d(&input, 4, 2), Err(Error::ShapeMismatch)));

    let broadcast = broadcast_row_major(&[10, 20], &[2, 1], &[2, 3]).unwrap();
    assert_eq!(broadcast, vec![10, 10, 10, 20, 20, 20]);
    assert!(matches!(
        broadcast_row_major(&[1, 2, 3, 4], &[2, 2], &[3, 2]),
        Err(Error::ShapeMismatch)
    ));
    assert!(matches!(
        broadcast_row_major(&[1, 2, 3], &[3], &[2, 3]),
        Err(Error::ShapeMismatch)
    ));

    let reshaped = reshape_row_major(&input, &[2, 3], &[3, 2]).unwrap();
    assert_eq!(reshaped, input);
    assert!(matches!(
        reshape_row_major(&input, &[2, 3], &[4, 2]),
        Err(Error::ShapeMismatch)
    ));
}

#[test]
fn worker_count_and_tile_order_do_not_change_results() {
    let m = 8usize;
    let n = 8usize;
    let k = 8usize;
    let a: Vec<i64> = (0..(m * k) as i64).map(|value| value - 17).collect();
    let b: Vec<i64> = (0..(k * n) as i64).map(|value| (value % 13) - 6).collect();
    let sequential = fixed_matmul(&a, &b, m, n, k).unwrap();

    for workers in [1usize, 4] {
        let pool = ThreadPool::new(workers).unwrap();
        let parallel = parallel_fixed_matmul(&pool, a.clone(), b.clone(), m, n, k, 3).unwrap();
        assert_eq!(parallel, sequential);
    }

    // Assemble row blocks in reverse completion order: the key (row start)
    // decides the final position, not the order the blocks finished.
    let mut blocks: Vec<(usize, Vec<i128>)> = Vec::new();
    for start in (0..m).step_by(3) {
        let end = core::cmp::min(start + 3, m);
        blocks.push((start, fixed_matmul_rows(&a, &b, m, n, k, start, end).unwrap()));
    }
    blocks.reverse();
    blocks.sort_by_key(|(start, _)| *start);
    let mut assembled = Vec::new();
    for (_, block) in blocks {
        assembled.extend(block);
    }
    assert_eq!(assembled, sequential);
}

#[test]
fn kernels_are_reentrant_from_several_threads() {
    let a: Arc<Vec<i64>> = Arc::new((0..64).map(|value| value - 32).collect());
    let b: Arc<Vec<i64>> = Arc::new((0..64).map(|value| (value % 7) - 3).collect());
    let expected = fixed_matmul(&a, &b, 8, 8, 8).unwrap();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let a = Arc::clone(&a);
            let b = Arc::clone(&b);
            std::thread::spawn(move || fixed_matmul(&a, &b, 8, 8, 8).unwrap())
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}
