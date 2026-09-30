//! Minimal elementwise layer compiler: translate a small op list (add / mul /
//! linear affine) into `prove_layer_circuit` constraints. This is the first,
//! arithmetic-only slice of the "Op -> one g per layer" compiler; the GPT-2
//! rounding/lookup/matmul ops build on the same constraint-folding pattern.
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::from_i64;
use crate::layer_circuit::{prove_layer_circuit, verify_layer_circuit, LayerCircuitProof};
#[derive(Clone, Debug)]
pub enum ElemOp {
    Add { a: usize, b: usize, c: usize },
    Mul { a: usize, b: usize, c: usize },
    Affine { x: usize, y: usize, scale: Goldilocks, bias: usize },
}
pub fn compile(ops: &[ElemOp]) -> Vec<Vec<(Goldilocks, Vec<usize>)>> {
    let mut out = Vec::new();
    for op in ops {
        match *op {
            ElemOp::Add { a, b, c } => out.push(vec![
                (Goldilocks::from_u64(1), vec![c]),
                (from_i64(-1), vec![a]),
                (from_i64(-1), vec![b]),
            ]),
            ElemOp::Mul { a, b, c } => out.push(vec![
                (Goldilocks::from_u64(1), vec![c]),
                (from_i64(-1), vec![a, b]),
            ]),
            ElemOp::Affine { x, y, scale, bias } => out.push(vec![
                (Goldilocks::from_u64(1), vec![y]),
                (from_i64(-1) * scale, vec![x]),
                (from_i64(-1), vec![bias]),
            ]),
        }
    }
    out
}
pub fn prove_layer(
    tensors: &[&[Goldilocks]],
    ops: &[ElemOp],
    r: &[Goldilocks],
    rng: &mut XorShift64,
) -> LayerCircuitProof {
    prove_layer_circuit(tensors, &compile(ops), r, rng)
}
pub fn verify_layer(
    proof: &LayerCircuitProof,
    tensors: &[&[Goldilocks]],
    ops: &[ElemOp],
    r: &[Goldilocks],
) -> bool {
    verify_layer_circuit(proof, tensors, &compile(ops), r)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compile_add_mul_affine() {
        let mut rng = XorShift64::new(0xF00D);
        let n = 1usize << 5;
        let x: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let w: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let scale = rng.field();
        let y: Vec<Goldilocks> = (0..n).map(|i| x[i] * scale + b[i]).collect();
        let z: Vec<Goldilocks> = (0..n).map(|i| y[i] * w[i]).collect();
        let t = n.trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let ops = vec![
            ElemOp::Affine { x: 0, y: 3, scale, bias: 1 },
            ElemOp::Mul { a: 3, b: 2, c: 4 },
        ];
        let tensors: Vec<&[Goldilocks]> = vec![&x, &b, &w, &y, &z];
        let proof = prove_layer(&tensors, &ops, &r, &mut rng);
        assert!(verify_layer(&proof, &tensors, &ops, &r));
        let mut bad_z = z.clone();
        bad_z[0] = bad_z[0] + Goldilocks::from_u64(1);
        let tensors_bad: Vec<&[Goldilocks]> = vec![&x, &b, &w, &y, &bad_z];
        assert!(!verify_layer(&proof, &tensors_bad, &ops, &r));
    }
}
