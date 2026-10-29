//! Reusable projection-block prover: out = round((x @ W) / 2^shift) + bias, with
//! the matmul output h = x@W never committed (chained via its GKR claim), the
//! affine rounding as an arithmetic constraint, and the range check as a logUp
//! lookup. This is the first reusable integration building block for the full
//! GPT-2 layer circuit.
use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::{from_i64, to_i32};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, verify_lookup_fractional, verify_lookup_fractional_chain,
    FractionalProof,
};
use zkie_core::common::matmul::{prove as matmul_prove, verify as matmul_verify, MatmulProof};
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ProjectionProof {
    pub matmul: MatmulProof,
    pub affine: VirtualProof,
    pub frac: FractionalProof,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub pt: Vec<Goldilocks>,
    pub u: Vec<Goldilocks>,
    pub v: Vec<Goldilocks>,
    pub ch: Vec<Goldilocks>,
}
fn matmul_full(
    x: &[Goldilocks],
    w: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    crate::par::mm_par_fixed(x, w, m, k, n)
}
fn transpose(x: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut wt = vec![Goldilocks::from_u64(0); k * m];
    for kk in 0..k {
        for i in 0..m {
            wt[kk * m + i] = x[i * k + kk];
        }
    }
    wt
}
pub fn prove_projection(
    x: &[Goldilocks],
    w: &[Goldilocks],
    bias: &[Goldilocks],
    out: &[Goldilocks],
    rem_off: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
    transcript: &mut Transcript,
) -> ProjectionProof {
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let h = matmul_full(x, w, m, k, n);
    let wt = transpose(x, m, k);

    // S1 ordering: the affine sumcheck runs FIRST. Its eq point is squeezed
    // upfront (the eq table needs it), its challenge vector r_aff is derived
    // per-round from its own messages, and r_aff becomes the matmul's claim
    // point (u = high bits, v = low bits). That way the matmul's claimed
    // eval of the virtual `h` sits exactly at the affine's final-check point.
    let arity_mn = (m * n).trailing_zeros() as usize;
    let q_aff: Vec<Goldilocks> = transcript.field_vec(arity_mn);
    let eq = zkie_core::common::mle::eq_evals(&q_aff);
    let ones = vec![Goldilocks::from_u64(1); m * n];
    let neg = from_i64(-1);
    let terms = vec![
        (Goldilocks::from_u64(1), vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (two_shift, vec![0usize, 3usize]),
        (neg * two_shift, vec![0usize, 4usize]),
        (neg * half, vec![0usize, 5usize]),
    ];
    let mles: Vec<&[Goldilocks]> = vec![&eq, &rem_off, &h, &out, &bias, &ones];
    let (affine, r_aff) = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), transcript);

    let n_log = n.trailing_zeros() as usize;
    let v: Vec<Goldilocks> = r_aff[..n_log].to_vec();
    let u: Vec<Goldilocks> = r_aff[n_log..].to_vec();
    let (matmul, ch) = matmul_prove(&wt, w, &h, m, k, n, &u, &v, transcript);
    let mut pt = v.clone();
    pt.extend_from_slice(&u);

    let table: Vec<Goldilocks> = (0..(1usize << shift))
        .map(|j| Goldilocks::from_u64(j as u64))
        .collect();
    let idx: Vec<u32> = rem_off.iter().map(|&vv| to_i32(vv) as u32).collect();
    let alpha = transcript.field();
    let beta = transcript.field();
    let frac = prove_lookup_fractional(&idx, rem_off, &table, alpha, beta, transcript);
    ProjectionProof {
        matmul,
        affine,
        frac,
        alpha,
        beta,
        pt,
        u,
        v,
        ch,
    }
}

pub fn verify_projection(
    proof: &ProjectionProof,
    shift: u32,
    m: usize,
    k: usize,
    n: usize,
    transcript: &mut Transcript,
) -> bool {
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let neg = from_i64(-1);
    let terms = vec![
        (Goldilocks::from_u64(1), vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (two_shift, vec![0usize, 3usize]),
        (neg * two_shift, vec![0usize, 4usize]),
        (neg * half, vec![0usize, 5usize]),
    ];
    let arity_mn = (m * n).trailing_zeros() as usize;
    let _q_aff: Vec<Goldilocks> = transcript.field_vec(arity_mn); // advance state
    let (ok, r_aff, fc) = verify_virtual(
        &proof.affine,
        &terms,
        Goldilocks::from_u64(0),
        arity_mn,
        transcript,
    );
    if !ok || r_aff != proof.pt {
        return false;
    }
    let n_log = n.trailing_zeros() as usize;
    let v: Vec<Goldilocks> = r_aff[..n_log].to_vec();
    let u: Vec<Goldilocks> = r_aff[n_log..].to_vec();
    if u != proof.u || v != proof.v {
        return false;
    }
    let (ok, ch) = matmul_verify(&proof.matmul, k.trailing_zeros() as usize, transcript);
    if !ok || ch != proof.ch {
        return false;
    }
    let mut fe = proof.affine.final_evals.clone();
    // h is virtual: its eval at the affine's final-check point IS the
    // matmul's claim (the matmul was proven at u/v = r_aff split).
    fe[2] = proof.matmul.claimed;
    if fc != virtual_eval(&terms, &fe) {
        return false;
    }
    if transcript.field() != proof.alpha || transcript.field() != proof.beta {
        return false;
    }
    // S3: the fractional depth comes from the statement: m*n lookup entries
    // plus the 2^shift range table.
    let frac_t = (m * n + (1usize << shift))
        .next_power_of_two()
        .trailing_zeros() as usize;
    verify_lookup_fractional_chain(&proof.frac, frac_t, transcript)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_block_roundtrip() {
        let mut rng = XorShift64::new(0xABCD);
        let (m, k, n) = (2usize, 2usize, 2usize);
        let shift = 4u32;
        let half = Goldilocks::from_u64(1u64 << (shift - 1));
        let two_shift = Goldilocks::from_u64(1u64 << shift);
        let div_round = |a: i64, b: i64| -> i64 {
            let q = a.div_euclid(b);
            let rr = a.rem_euclid(b);
            if rr * 2 >= b {
                q + 1
            } else {
                q
            }
        };
        let x: Vec<Goldilocks> = (0..m * k)
            .map(|_| from_i64((rng.next_u64() % 100) as i64))
            .collect();
        let w: Vec<Goldilocks> = (0..k * n)
            .map(|_| from_i64((rng.next_u64() % 100) as i64))
            .collect();
        let bias: Vec<Goldilocks> = (0..m * n)
            .map(|_| from_i64((rng.next_u64() % 20) as i64 - 10))
            .collect();
        let h = matmul_full(&x, &w, m, k, n);
        let out: Vec<Goldilocks> = (0..m * n)
            .map(|ij| {
                from_i64(
                    div_round(zkie_core::common::fixed_point::to_i64(h[ij]), 1i64 << shift)
                        + zkie_core::common::fixed_point::to_i64(bias[ij]),
                )
            })
            .collect();
        let rem_off: Vec<Goldilocks> = (0..m * n)
            .map(|ij| {
                let h_i = zkie_core::common::fixed_point::to_i64(h[ij]);
                let o_i = zkie_core::common::fixed_point::to_i64(out[ij]);
                let b_i = zkie_core::common::fixed_point::to_i64(bias[ij]);
                from_i64(h_i - (o_i - b_i) * (1i64 << shift) + (1i64 << (shift - 1)))
            })
            .collect();
        for &vv in &rem_off {
            assert!(to_i32(vv) >= 0 && to_i32(vv) < (1i32 << shift));
        }
        let _ = (half, two_shift);
        let mut tp = Transcript::new(b"zkie-projection-test");
        tp.absorb_u64(0xABCD);
        let mut proof = prove_projection(&x, &w, &bias, &out, &rem_off, m, k, n, shift, &mut tp);
        let mut tv = Transcript::new(b"zkie-projection-test");
        tv.absorb_u64(0xABCD);
        assert!(verify_projection(&proof, shift, m, k, n, &mut tv));
        proof.frac.layers[0].proof.final_evals[3] += Goldilocks::from_u64(1);
        let mut tv2 = Transcript::new(b"zkie-projection-test");
        tv2.absorb_u64(0xABCD);
        assert!(!verify_projection(&proof, shift, m, k, n, &mut tv2));
    }
    #[test]
    fn committed_projection_roundtrip() {
        use zkie_core::common::matmul::{prove as mm_prove, verify as mm_verify};
        use zkie_core::pcs::committed::commit;
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0xDD22);
        let (m, k, n) = (8usize, 8usize, 8usize);
        let shift = 4u32;
        let half = Goldilocks::from_u64(1u64 << (shift - 1));
        let two_shift = Goldilocks::from_u64(1u64 << shift);
        let div_round = |a: i64, b: i64| -> i64 {
            let q = a.div_euclid(b);
            let rr = a.rem_euclid(b);
            if rr * 2 >= b {
                q + 1
            } else {
                q
            }
        };
        let x: Vec<Goldilocks> = (0..m * k)
            .map(|_| from_i64((rng.next_u64() % 100) as i64))
            .collect();
        let w: Vec<Goldilocks> = (0..k * n)
            .map(|_| from_i64((rng.next_u64() % 100) as i64))
            .collect();
        let bias: Vec<Goldilocks> = (0..m * n)
            .map(|_| from_i64((rng.next_u64() % 20) as i64 - 10))
            .collect();
        let h = matmul_full(&x, &w, m, k, n);
        let out: Vec<Goldilocks> = (0..m * n)
            .map(|ij| {
                from_i64(
                    div_round(zkie_core::common::fixed_point::to_i64(h[ij]), 1i64 << shift)
                        + zkie_core::common::fixed_point::to_i64(bias[ij]),
                )
            })
            .collect();
        let rem_off: Vec<Goldilocks> = (0..m * n)
            .map(|ij| {
                let h_i = zkie_core::common::fixed_point::to_i64(h[ij]);
                let o_i = zkie_core::common::fixed_point::to_i64(out[ij]);
                let b_i = zkie_core::common::fixed_point::to_i64(bias[ij]);
                from_i64(h_i - (o_i - b_i) * (1i64 << shift) + (1i64 << (shift - 1)))
            })
            .collect();
        let mut at = vec![Goldilocks::from_u64(0); k * m];
        for kk in 0..k {
            for i in 0..m {
                at[kk * m + i] = x[i * k + kk];
            }
        }
        let mut tp0 = Transcript::new(b"zkie-projection-committed-test");
        tp0.absorb_u64(0xDD22);
        let mut tp = tp0.clone();
        // S1 order: the affine sumcheck runs first; its per-round-derived
        // challenge vector r_aff becomes the matmul claim point.
        let arity_mn = (m * n).trailing_zeros() as usize;
        let q_aff: Vec<Goldilocks> = tp.field_vec(arity_mn);
        let eq = zkie_core::common::mle::eq_evals(&q_aff);
        let ones = vec![Goldilocks::from_u64(1); m * n];
        let neg = from_i64(-1);
        let terms = vec![
            (Goldilocks::from_u64(1), vec![0usize, 1usize]),
            (neg, vec![0usize, 2usize]),
            (two_shift, vec![0usize, 3usize]),
            (neg * two_shift, vec![0usize, 4usize]),
            (neg * half, vec![0usize, 5usize]),
        ];
        let mles: Vec<&[Goldilocks]> = vec![&eq, &rem_off, &h, &out, &bias, &ones];
        let (affine_proof, r_aff) =
            zkie_core::common::sumcheck::prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &mut tp);
        let n_log = n.trailing_zeros() as usize;
        let v: Vec<Goldilocks> = r_aff[..n_log].to_vec();
        let u: Vec<Goldilocks> = r_aff[n_log..].to_vec();
        let (mm, ch) = mm_prove(&at, &w, &h, m, k, n, &u, &v, &mut tp);
        let whir_km = Whir::new_testing((k * m).trailing_zeros() as usize);
        let whir_kn = Whir::new_testing((k * n).trailing_zeros() as usize);
        let whir_mn = Whir::new_testing((m * n).trailing_zeros() as usize);
        let c_at = commit(&whir_km, &x);
        let c_w = commit(&whir_kn, &w);
        let c_bias = commit(&whir_mn, &bias);
        let c_out = commit(&whir_mn, &out);
        let c_rem = commit(&whir_mn, &rem_off);
        let mut ap = ch.clone();
        ap.extend_from_slice(&u);
        let (a_open, a_ev) = whir_km.open(c_at.prover_data.clone(), &c_at.protocol, &ap);
        assert_eq!(
            whir_km
                .verify(&c_at.commitment, &a_open, &c_at.protocol, &ap)
                .unwrap(),
            a_ev
        );
        assert_eq!(a_ev, zkie_core::common::mle::eval(&x, &ap));
        let mut bp = v.clone();
        bp.extend_from_slice(&ch);
        let (b_open, b_ev) = whir_kn.open(c_w.prover_data.clone(), &c_w.protocol, &bp);
        assert_eq!(
            whir_kn
                .verify(&c_w.commitment, &b_open, &c_w.protocol, &bp)
                .unwrap(),
            b_ev
        );
        assert_eq!(b_ev, zkie_core::common::mle::eval(&w, &bp));
        let mut tv = tp0.clone();
        let _q: Vec<Goldilocks> = tv.field_vec(arity_mn);
        // Replay the affine sumcheck: its per-round-derived challenge vector is
        // the point where the committed mles must be opened for the final check
        // and where the matmul claim on the virtual h sits.
        let (ok, r_aff_v, fc) = zkie_core::common::sumcheck::verify_virtual(
            &affine_proof,
            &terms,
            Goldilocks::from_u64(0),
            arity_mn,
            &mut tv,
        );
        assert!(ok);
        assert_eq!(r_aff_v, r_aff);
        let (ok, ch_v) = mm_verify(&mm, k.trailing_zeros() as usize, &mut tv);
        assert!(ok);
        assert_eq!(ch_v, ch);
        assert_eq!(r_aff[..n_log].to_vec(), v);
        assert_eq!(r_aff[n_log..].to_vec(), u);
        let rem_ev = {
            let (o, e) = whir_mn.open(c_rem.prover_data.clone(), &c_rem.protocol, &r_aff);
            assert_eq!(
                whir_mn
                    .verify(&c_rem.commitment, &o, &c_rem.protocol, &r_aff)
                    .unwrap(),
                e
            );
            e
        };
        let out_ev = {
            let (o, e) = whir_mn.open(c_out.prover_data.clone(), &c_out.protocol, &r_aff);
            assert_eq!(
                whir_mn
                    .verify(&c_out.commitment, &o, &c_out.protocol, &r_aff)
                    .unwrap(),
                e
            );
            e
        };
        let bias_ev = {
            let (o, e) = whir_mn.open(c_bias.prover_data.clone(), &c_bias.protocol, &r_aff);
            assert_eq!(
                whir_mn
                    .verify(&c_bias.commitment, &o, &c_bias.protocol, &r_aff)
                    .unwrap(),
                e
            );
            e
        };
        let fe = vec![
            zkie_core::common::mle::eval(&eq, &r_aff),
            rem_ev,
            mm.claimed,
            out_ev,
            bias_ev,
            zkie_core::common::mle::eval(&ones, &r_aff),
        ];
        assert_eq!(fc, zkie_core::common::sumcheck::virtual_eval(&terms, &fe));
    }
}
