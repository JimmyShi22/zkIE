//! Linear/structural operators against independent arithmetic.

use zkie_core::types::{DType, TensorSpec};


use zkie_ops::add::AddOp;
use zkie_ops::affine_scale::AffineScaleOp;
use zkie_ops::batched_matmul::BatchedMatMulOp;
use zkie_ops::div_round::DivRoundOp;
use zkie_ops::gather::GatherOp;
use zkie_ops::gather_matmul::GatherMatMulOp;
use zkie_ops::hadamard::HadamardOp;
use zkie_ops::index_compose::IndexComposeOp;
use zkie_ops::int_sqrt::IntSqrtOp;
use zkie_ops::layout::LayoutOp;
use zkie_ops::matmul::MatMulOp;
use zkie_ops::pad::PadOp;
use zkie_ops::projection::ProjectionOp;
use zkie_ops::range::RangeOp;
use zkie_ops::registry::{isqrt_u128, PureForward};
use zkie_ops::relu::ReluOp;
use zkie_ops::requantize::RequantizeOp;
use zkie_ops::scatter::ScatterOp;
use zkie_ops::table_lookup::TableLookupOp;

fn output(op: &dyn PureForward, inputs: &[Vec<i64>]) -> Vec<i64> {
    op.forward_values(inputs).unwrap().outputs.remove(0)
}

fn wide(record: &zkie_ops::registry::KernelRecord, index: usize) -> Vec<i128> {
    record.raw_accumulators[index].clone()
}

#[test]
fn layout_reshape_and_transpose() {
    let values = vec![1, 2, 3, 4, 5, 6];
    assert_eq!(
        output(&LayoutOp::reshape(vec![2, 3], vec![3, 2]), &[values.clone()]),
        values
    );
    assert_eq!(
        output(&LayoutOp::transpose(2, 3), &[values]),
        vec![1, 4, 2, 5, 3, 6]
    );
    assert!(LayoutOp::reshape(vec![2, 3], vec![4, 2])
        .forward_values(&[vec![1, 2, 3, 4, 5, 6]])
        .is_err());
}

#[test]
fn rounding_ops_use_the_exact_rules() {
    assert_eq!(output(&RequantizeOp { shift: 1 }, &[vec![-3, -1, 3]]), vec![-1, 0, 2]);
    assert_eq!(output(&RequantizeOp { shift: 0 }, &[vec![-3, 3]]), vec![-3, 3]);

    let affine = AffineScaleOp {
        multiplier: 3,
        addend: 1,
        shift: 1,
        clamp: Some((-2, 4)),
    };
    let record = affine.forward_values(&[vec![1, 2, 5]]).unwrap();
    assert_eq!(record.outputs[0], vec![2, 4, 4]);
    assert_eq!(record.raw_accumulators[0], vec![4, 7, 16]);

    assert_eq!(output(&ReluOp, &[vec![-3, 0, 5]]), vec![0, 0, 5]);
}

#[test]
fn matmul_projection_and_add_are_exact() {
    let a = vec![1, -2, 3, 4, 5, -6];
    let b = vec![7, 8, -9, 10, 11, -12];
    let record = MatMulOp { m: 2, n: 2, k: 3, shift: 1 }
        .forward_values(&[a.clone(), b.clone()])
        .unwrap();
    assert_eq!(wide(&record, 0), vec![58, -48, -83, 154]);
    assert_eq!(record.outputs[0], vec![29, -24, -41, 77]);

    let projection = ProjectionOp {
        m: 2,
        n: 2,
        k: 3,
        shift: 1,
        bias: Some(vec![1, 1, 1, 1]),
    };
    let record = projection.forward_values(&[a.clone(), b]).unwrap();
    assert_eq!(record.raw_accumulators[0], vec![59, -47, -82, 155]);
    assert_eq!(record.outputs[0], vec![30, -23, -41, 78]);

    assert_eq!(output(&AddOp, &[vec![1, -2], vec![3, 4]]), vec![4, 2]);

    let mismatch_a = TensorSpec::new(DType::I64, vec![2]);
    let mut mismatch_a = mismatch_a;
    mismatch_a.fixed = Some(zkie_core::types::FixedSpec::new(1, 1 << 20, zkie_core::types::RoundRule::Floor));
    let mut mismatch_b = TensorSpec::new(DType::I64, vec![2]);
    mismatch_b.fixed = Some(zkie_core::types::FixedSpec::new(2, 1 << 20, zkie_core::types::RoundRule::Floor));
    assert!(AddOp.infer_specs(&[mismatch_a, mismatch_b]).is_err());
}

#[test]
fn lookup_and_index_ops_are_checked() {
    assert_eq!(
        output(&TableLookupOp, &[vec![10, 20, 30], vec![2, 0]]),
        vec![30, 10]
    );
    assert!(TableLookupOp.forward_values(&[vec![10], vec![1]]).is_err());

    let table = vec![1, 2, 3, 4, 5, 6];
    assert_eq!(output(&GatherOp { cols: 2 }, &[table.clone(), vec![2, 0]]), vec![5, 6, 1, 2]);
    assert!(GatherOp { cols: 2 }.forward_values(&[table, vec![3]]).is_err());

    assert_eq!(
        output(&IndexComposeOp, &[vec![2, 0], vec![10, 20, 30]]),
        vec![30, 10]
    );
    assert!(IndexComposeOp.forward_values(&[vec![3], vec![1, 2]]).is_err());
}

#[test]
fn batched_and_gather_matmul_match_scalar_reference() {
    let a: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let b: Vec<i64> = vec![2, 3, 4, 5, 6, 7, 8, 9];
    let record = BatchedMatMulOp { batch: 2, m: 2, n: 2, k: 2, shift: 0 }
        .forward_values(&[a.clone(), b.clone()])
        .unwrap();
    let mut expected = Vec::new();
    for batch in 0..2 {
        for row in 0..2 {
            for column in 0..2 {
                let mut sum = 0i128;
                for inner in 0..2 {
                    sum += a[batch * 4 + row * 2 + inner] as i128 * b[batch * 4 + inner * 2 + column] as i128;
                }
                expected.push(sum);
            }
        }
    }
    assert_eq!(wide(&record, 0), expected);

    let table = vec![1, 2, 3, 4, 5, 6];
    let weights = vec![1, 0, 0, 1];
    let record = GatherMatMulOp { cols: 2, n: 2, shift: 0 }
        .forward_values(&[table, vec![2, 0], weights])
        .unwrap();
    assert_eq!(record.outputs[0], vec![5, 6, 1, 2]);
    assert!(GatherMatMulOp { cols: 2, n: 2, shift: 0 }
        .forward_values(&[vec![1, 2], vec![1], vec![1, 0, 0, 1]])
        .is_err());
}

#[test]
fn structural_range_div_and_sqrt_ops() {
    assert_eq!(output(&PadOp { valid_len: 2, value: -1 }, &[vec![1, 2, 3, 4]]), vec![1, 2, -1, -1]);
    assert!(PadOp { valid_len: 5, value: 0 }.forward_values(&[vec![1]]).is_err());

    assert_eq!(
        output(&HadamardOp { shift: 1 }, &[vec![3, -3], vec![3, 3]]),
        vec![5, -4]
    );

    assert_eq!(output(&RangeOp { low: -2, high: 3 }, &[vec![-3, -2, 0, 3, 4]]), vec![0, 1, 1, 1, 0]);
    assert!(RangeOp { low: 4, high: 3 }.forward_values(&[vec![0]]).is_err());

    let record = DivRoundOp.forward_values(&[vec![-7, 7], vec![3, 3]]).unwrap();
    assert_eq!(record.outputs[0], vec![-3, 2, 2, 1]);
    assert!(DivRoundOp.forward_values(&[vec![1], vec![0]]).is_err());
    assert!(DivRoundOp.forward_values(&[vec![1], vec![-1]]).is_err());

    assert_eq!(output(&IntSqrtOp, &[vec![0, 1, 2, 3, 4, 15, 16]]), vec![0, 1, 1, 1, 2, 3, 4]);
    assert_eq!(isqrt_u128(1 << 100), 1u128 << 50);
    assert!(IntSqrtOp.forward_values(&[vec![-1]]).is_err());
}

#[test]
fn scatter_is_deterministic_and_rejects_duplicates() {
    assert_eq!(output(&ScatterOp { size: 4 }, &[vec![1, 3], vec![9, 7]]), vec![0, 9, 0, 7]);
    assert!(ScatterOp { size: 4 }.forward_values(&[vec![1, 1], vec![9, 7]]).is_err());
    assert!(ScatterOp { size: 2 }.forward_values(&[vec![2], vec![9]]).is_err());
}

#[test]
fn all_linear_ops_expose_infer_and_workspace() {
    fn check<O: PureForward>(op: &O, inputs: &[TensorSpec]) {
        let outputs = op.infer_specs(inputs).unwrap();
        assert!(!outputs.is_empty());
        let bytes: u64 = inputs.iter().map(|spec| spec.byte_len().unwrap()).sum::<u64>()
            + outputs.iter().map(|spec| spec.byte_len().unwrap()).sum::<u64>();
        assert!(bytes > 0);
    }
    check(&MatMulOp { m: 1, n: 1, k: 1, shift: 0 }, &[
        TensorSpec::new(DType::I64, vec![1]),
        TensorSpec::new(DType::I64, vec![1]),
    ]);
    check(&ReluOp, &[TensorSpec::new(DType::I64, vec![3])]);
    check(&ScatterOp { size: 2 }, &[
        TensorSpec::new(DType::I64, vec![1]),
        TensorSpec::new(DType::I64, vec![1]),
    ]);
}
