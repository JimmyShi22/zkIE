use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_ops::softmax_scaled::{prove_softmax_bias, softmax_bias_forward, verify_softmax_bias};
use zkie_core::common::sumcheck::verify_virtual;
use zkie_core::common::mle;

#[test]
fn softmax_bias_roundtrip() {
    let mut rng = XorShift64::new(0x55AA);
    let m = 2usize;
    let n = 2usize;
    let e = vec![from_i64(65536), from_i64(131072), from_i64(196608), from_i64(262144)];
    let bias = vec![from_i64(32768), from_i64(65536)];
    let (out, sum) = softmax_bias_forward(&e, &bias, m, n);
    println!("out {:?}", out.iter().map(|&v| zkie_core::common::fixed_point::to_i64(v)).collect::<Vec<_>>());
    println!("sum {:?}", sum.iter().map(|&v| zkie_core::common::fixed_point::to_i64(v)).collect::<Vec<_>>());
    let proof = prove_softmax_bias(&e, &bias, &out, &sum, m, n, &mut rng);
    println!("claimed {:?}", proof.claimed);
    println!("actual {:?}", mle::eval(&sum, &proof.r_sum));
    println!("final evals {:?}", proof.row_sum.final_evals);
    let one = Goldilocks::ONE;
    let row_ok = verify_virtual(
        &proof.row_sum,
        &[(one, vec![0usize,1usize])],
        proof.claimed,
        &proof.sum_ch,
        &proof.row_sum.final_evals,
    );
    println!("row_ok={row_ok}");
    let add_ok = verify_virtual(
        &proof.add,
        &[(one, vec![0usize]), (one, vec![1usize]), (Goldilocks::ZERO-Goldilocks::ONE, vec![2usize])],
        Goldilocks::ZERO,
        &proof.r_sum,
        &proof.add.final_evals,
    );
    println!("add_ok={add_ok}");
    let two16 = Goldilocks::from_u64(1u64<<16);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let res_ok = verify_virtual(
        &proof.rescale,
        &[(one, vec![0usize,2usize]), (neg*two16, vec![1usize]), (one, vec![3usize])],
        Goldilocks::ZERO,
        &proof.r_scale,
        &proof.rescale.final_evals,
    );
    println!("res_ok={res_ok}");
    assert!(verify_softmax_bias(&proof));
}
