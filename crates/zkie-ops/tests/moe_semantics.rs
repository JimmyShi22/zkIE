//! Router gating, gated FFN and the fused MoE down path.

use zkie_ops::gated_ffn::GatedFfnOp;
use zkie_ops::moe_down::MoeDownOp;
use zkie_ops::registry::PureForward;
use zkie_ops::route_gate::RouteGateOp;

fn output(op: &dyn PureForward, inputs: &[Vec<i64>]) -> Vec<i64> {
    op.forward_values(inputs).unwrap().outputs.remove(0)
}

#[test]
fn route_gate_uses_topk_threshold_and_index_ties() {
    let gate = RouteGateOp {
        top_k: 2,
        threshold: 10,
    };
    assert_eq!(
        output(&gate, &[vec![10, 20, 20, 5]]),
        vec![2, 1, 2, 20, 20]
    );
    let above = RouteGateOp {
        top_k: 2,
        threshold: 15,
    };
    assert_eq!(output(&above, &[vec![10, 20, 20, 5]]), vec![2, 1, 2, 20, 20]);
    let none = RouteGateOp {
        top_k: 2,
        threshold: 25,
    };
    assert_eq!(output(&none, &[vec![10, 20, 20, 5]]), vec![0, -1, -1, 0, 0]);
    assert!(RouteGateOp { top_k: 5, threshold: 0 }
        .forward_values(&[vec![1, 2]])
        .is_err());
}

#[test]
fn gated_ffn_has_one_final_rounding() {
    assert_eq!(output(&GatedFfnOp { shift: 1 }, &[vec![3], vec![5]]), vec![8]);
    let record = GatedFfnOp { shift: 1 }
        .forward_values(&[vec![3, -3], vec![5, 5]])
        .unwrap();
    assert_eq!(record.raw_accumulators[0], vec![15, -15]);
    assert_eq!(record.outputs[0], vec![8, -7]);
}

#[test]
fn moe_down_adds_raw_paths_before_rounding() {
    let op = MoeDownOp {
        shared_weights: vec![1],
        routed_weights: vec![1],
        width: 1,
        out_cols: 1,
        shift: 1,
    };
    let record = op.forward_values(&[vec![1], vec![1]]).unwrap();
    assert_eq!(record.raw_accumulators[0], vec![1]);
    assert_eq!(record.raw_accumulators[1], vec![1]);
    assert_eq!(record.raw_accumulators[2], vec![2]);
    assert_eq!(record.outputs[0], vec![1]);

    // The shared path reads m, the routed path reads xr.
    let gated = op.forward_values(&[vec![2], vec![1]]).unwrap();
    assert_eq!(gated.raw_accumulators[0], vec![2]);
    assert_eq!(gated.raw_accumulators[1], vec![1]);
    assert_eq!(gated.outputs[0], vec![2]);

    assert!(MoeDownOp {
        shared_weights: vec![1, 2],
        routed_weights: vec![1],
        width: 1,
        out_cols: 1,
        shift: 0,
    }
    .forward_values(&[vec![1], vec![1]])
    .is_err());
}

#[test]
fn moe_down_is_batched_over_output_columns() {
    let op = MoeDownOp {
        shared_weights: vec![1, 0, 0, 1],
        routed_weights: vec![2, 0, 0, 2],
        width: 2,
        out_cols: 2,
        shift: 1,
    };
    let record = op
        .forward_values(&[vec![1, 2], vec![3, 4]])
        .unwrap();
    assert_eq!(record.raw_accumulators[0], vec![1, 2]);
    assert_eq!(record.raw_accumulators[1], vec![6, 8]);
    assert_eq!(record.raw_accumulators[2], vec![7, 10]);
    assert_eq!(record.outputs[0], vec![4, 5]);
}
