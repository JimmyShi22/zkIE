//! Nonlinear operators: softmax, norms, rotation, top-k, cache and patches.

use zkie_ops::argmax::ArgmaxOp;
use zkie_ops::int_sqrt::IntSqrtOp;
use zkie_ops::kv_cache::KvCacheOp;
use zkie_ops::layer_norm::LayerNormOp;
use zkie_ops::patchify::PatchifyOp;
use zkie_ops::registry::{isqrt_u128, PureForward};
use zkie_ops::regression_head::RegressionHeadOp;
use zkie_ops::rms_norm::RmsNormOp;
use zkie_ops::rope_projection::RopeProjectionOp;
use zkie_ops::softmax::SoftmaxOp;
use zkie_ops::topk::TopKOp;

fn output(op: &dyn PureForward, inputs: &[Vec<i64>]) -> Vec<i64> {
    op.forward_values(inputs).unwrap().outputs.remove(0)
}

#[test]
fn softmax_uses_max_table_and_one_division() {
    let logits = vec![5, 3, 1];
    let table = vec![100, 50, 10, 5, 1];
    let record = SoftmaxOp { shift: 4 }
        .forward_values(&[logits.clone(), table.clone()])
        .unwrap();
    let max = 5i128;
    let u: Vec<i128> = logits
        .iter()
        .map(|value| {
            let index = (table.len() as i128) - (max - *value as i128);
            table[index.clamp(0, table.len() as i128 - 1) as usize] as i128
        })
        .collect();
    let denominator: i128 = u.iter().sum();
    let expected: Vec<i64> = u
        .iter()
        .map(|entry| ((entry * 16).div_euclid(denominator)) as i64)
        .collect();
    assert_eq!(record.outputs[0], expected);
    assert_eq!(record.auxiliaries[2][0], denominator as i64);
    let clamped = SoftmaxOp { shift: 0 }
        .forward_values(&[vec![0, 10], vec![1]])
        .unwrap();
    assert_eq!(clamped.outputs[0], vec![1, 1]);
}

#[test]
fn norms_use_isqrt_and_an_explicit_clamp() {
    let layer = LayerNormOp {
        rows: 1,
        cols: 4,
        valid_cols: 4,
        shift: 2,
        clamp: None,
    };
    let record = layer
        .forward_values(&[vec![4, 0, 0, 0], vec![1, 1, 1, 1]])
        .unwrap();
    let mean = 1i64;
    let variance = 3u128;
    let rstd = isqrt_u128(16 / variance);
    let expected: Vec<i64> = [4i64, 0, 0, 0]
        .iter()
        .map(|value| ((((value - mean) as i128 * rstd as i128 + 2).div_euclid(4)) as i64))
        .collect();
    assert_eq!(record.outputs[0], expected);

    let clamped = LayerNormOp {
        clamp: Some((0, 0)),
        ..layer
    };
    assert_eq!(
        clamped
            .forward_values(&[vec![4, 0, 0, 0], vec![1, 1, 1, 1]])
            .unwrap()
            .outputs[0],
        vec![0, 0, 0, 0]
    );

    let rms = RmsNormOp {
        rows: 1,
        cols: 2,
        valid_cols: 2,
        shift: 1,
        clamp: None,
    };
    assert_eq!(
        rms.forward_values(&[vec![2, 0], vec![1, 1]]).unwrap().outputs[0],
        vec![1, 0]
    );
    assert!(RmsNormOp { rows: 1, cols: 2, valid_cols: 3, shift: 0, clamp: None }
        .forward_values(&[vec![1, 2], vec![1, 1]])
        .is_err());
}

#[test]
fn rotation_projection_is_one_rounding() {
    assert_eq!(
        output(&RopeProjectionOp { shift: 0 }, &[vec![2, 3], vec![1], vec![0]]),
        vec![2, 3]
    );
    assert_eq!(
        output(&RopeProjectionOp { shift: 0 }, &[vec![2, 3], vec![0], vec![1]]),
        vec![-3, 2]
    );
    assert!(RopeProjectionOp { shift: 0 }
        .forward_values(&[vec![1, 2], vec![1, 1], vec![1]])
        .is_err());
}

#[test]
fn topk_and_argmax_have_deterministic_ties() {
    assert_eq!(output(&TopKOp { k: 2 }, &[vec![5, 3, 5, 1]]), vec![0, 2, 5, 5]);
    assert!(TopKOp { k: 0 }.forward_values(&[vec![1]]).is_err());
    assert!(TopKOp { k: 3 }.forward_values(&[vec![1, 2]]).is_err());

    assert_eq!(output(&ArgmaxOp, &[vec![1, 5, 5, 2]]), vec![1]);
    assert_eq!(output(&ArgmaxOp, &[vec![-1, -5]]), vec![0]);
}

#[test]
fn kv_cache_and_patchify_bounds() {
    let cache = KvCacheOp {
        capacity: 2,
        width: 2,
        position: 1,
    };
    let record = cache
        .forward_values(&[vec![1, 2, 0, 0], vec![3, 4, 0, 0], vec![7, 8], vec![9, 10]])
        .unwrap();
    assert_eq!(record.outputs[0], vec![1, 2, 7, 8]);
    assert_eq!(record.outputs[1], vec![3, 4, 9, 10]);
    assert!(KvCacheOp { capacity: 2, width: 2, position: 2 }
        .forward_values(&[vec![0; 4], vec![0; 4], vec![1, 2], vec![3, 4]])
        .is_err());

    assert_eq!(output(&PatchifyOp { patch: 2 }, &[vec![1, 2, 3, 4]]), vec![1, 2, 3, 4]);
    assert!(PatchifyOp { patch: 2 }.forward_values(&[vec![1, 2, 3]]).is_err());
}

#[test]
fn regression_head_shares_the_exact_linear_path() {
    let record = RegressionHeadOp {
        m: 1,
        n: 2,
        k: 2,
        shift: 1,
        bias: Some(vec![1, -1]),
    }
    .forward_values(&[vec![2, 3], vec![1, 0, 0, 1]])
    .unwrap();
    assert_eq!(record.raw_accumulators[0], vec![3, 2]);
    assert_eq!(record.outputs[0], vec![2, 1]);
    let _ = IntSqrtOp; // shared sqrt path is covered by the linear suite
}

#[test]
fn profile_exp_index_subtracts_distance_from_table_length() {
    use zkie_ops::registry::exp_profile_value;
    let table = vec![10, 20, 30, 40, 50];
    assert_eq!(exp_profile_value(&table, 0).unwrap(), 50);
    assert_eq!(exp_profile_value(&table, 1).unwrap(), 50);
    assert_eq!(exp_profile_value(&table, 4).unwrap(), 20);
    assert_eq!(exp_profile_value(&table, 5).unwrap(), 10);
    assert_eq!(exp_profile_value(&table, 99).unwrap(), 10);
    assert!(exp_profile_value(&table, -1).is_err());
}
