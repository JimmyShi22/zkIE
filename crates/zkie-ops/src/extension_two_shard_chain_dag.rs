//! Two-shard chain DAG: TWO existing `extension_chain` GKR chains composed
//! head-to-tail — the downstream chain's input `X` IS the upstream chain's
//! output `Y` (the SAME raw base buffer, committed once per chain — the
//! deterministic commit yields the identical root).
//!
//! Boundary contract (checked by the root-only verifier, in order):
//! 1. dimensions: `downstream.m == upstream.m` and
//!    `downstream.d == upstream.n`;
//! 2. commitment: `upstream.root_y == downstream.root_x` (byte-equal — the
//!    shared boundary tensor, not a prover-supplied pairing);
//! 3. each of the two existing chain verifiers.
//!
//! Verifier = (`TwoShardChainDagWhir`, statement, proof) only: no witness,
//! no recomputation, no `open` calls. The prover calls the existing
//! `extension_chain::prove` twice (upstream with the shared buffer as `y`,
//! downstream with the SAME buffer as `x`) and rejects unless the boundary
//! roots are byte-equal. No generic traits; the wrappers are explicit.

use crate::extension_chain::{self, ChainStatement, ChainWhir, ExtensionChainProof};
use zkie_core::common::field::Goldilocks;

/// The two chained WHIR instance sets: `upstream` produces `Y`, `downstream`
/// consumes it as its `X`.
pub struct TwoShardChainDagWhir {
    pub upstream: ChainWhir,
    pub downstream: ChainWhir,
}

impl TwoShardChainDagWhir {
    /// Build the two chain instance sets. Upstream dims `[m, d, k, n]`;
    /// downstream dims `[m, n, k2, n2]` (the boundary forces downstream's
    /// `m` and `d = n`). Returns `None` for malformed dims or infeasible
    /// WHIR configurations.
    pub fn new(
        m: usize,
        d: usize,
        k: usize,
        n: usize,
        k2: usize,
        n2: usize,
        security_level: usize,
        pow_budget: usize,
    ) -> Option<Self> {
        Some(TwoShardChainDagWhir {
            upstream: ChainWhir::new(m, d, k, n, security_level, pow_budget)?,
            downstream: ChainWhir::new(m, n, k2, n2, security_level, pow_budget)?,
        })
    }
}

/// Public statement: the two chain statements; the boundary root appears as
/// `upstream.root_y` and `downstream.root_x`.
#[derive(Clone, Debug)]
pub struct TwoShardChainDagStatement {
    pub upstream: ChainStatement,
    pub downstream: ChainStatement,
}

/// The transported proof: the two chain proofs.
#[derive(Clone)]
pub struct TwoShardChainDagProof {
    pub upstream: ExtensionChainProof,
    pub downstream: ExtensionChainProof,
}

/// Prove the composed DAG. The upstream witness is `(x, w1, h, w2, y)` with
/// dims `[m, d, k, n]`; the downstream witness is `(w1_2, h_2, w2_2, y_2)`
/// with dims `[m, n, k2, n2]` and its input `x` IS the upstream `y` buffer
/// (passed once). Calls the existing `extension_chain::prove` twice and
/// rejects unless the boundary roots are byte-equal. Returns `None` for
/// malformed inputs (never panics).
pub fn prove(
    dag: &TwoShardChainDagWhir,
    x: &[Goldilocks],
    w1: &[Goldilocks],
    h: &[Goldilocks],
    w2: &[Goldilocks],
    y: &[Goldilocks],
    m: usize,
    d: usize,
    k: usize,
    n: usize,
    w1_2: &[Goldilocks],
    h_2: &[Goldilocks],
    w2_2: &[Goldilocks],
    y_2: &[Goldilocks],
    k2: usize,
    n2: usize,
) -> Option<(TwoShardChainDagStatement, TwoShardChainDagProof)> {
    let (up_stmt, up_proof) =
        extension_chain::prove(&dag.upstream, x, w1, h, w2, y, m, d, k, n)?;
    // Downstream X is the SAME raw buffer as upstream Y.
    let (down_stmt, down_proof) =
        extension_chain::prove(&dag.downstream, y, w1_2, h_2, w2_2, y_2, m, n, k2, n2)?;
    // The shared boundary commitment must be byte-identical.
    if up_stmt.root_y != down_stmt.root_x {
        return None;
    }
    Some((
        TwoShardChainDagStatement {
            upstream: up_stmt,
            downstream: down_stmt,
        },
        TwoShardChainDagProof {
            upstream: up_proof,
            downstream: down_proof,
        },
    ))
}

/// Root-only verify: boundary dimensions first, then the boundary root
/// equality, then the two existing chain verifiers. Statement and proof
/// only — no witness, no recomputation, no `open` calls. Malformed shapes
/// return `false` (never panic).
pub fn verify(
    dag: &TwoShardChainDagWhir,
    stmt: &TwoShardChainDagStatement,
    proof: &TwoShardChainDagProof,
) -> bool {
    // Boundary dimensions: downstream consumes upstream's output shape.
    if stmt.downstream.m != stmt.upstream.m || stmt.downstream.d != stmt.upstream.n {
        return false;
    }
    // Boundary commitment: byte-equal shared root.
    if stmt.upstream.root_y != stmt.downstream.root_x {
        return false;
    }
    extension_chain::verify(&dag.upstream, &stmt.upstream, &proof.upstream)
        && extension_chain::verify(&dag.downstream, &stmt.downstream, &proof.downstream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::{PrimeCharacteristicRing, XorShift64};

    // Smallest dims with every chain WHIR arity >= the folding factor.
    const M: usize = 8;
    const D: usize = 4;
    const K: usize = 8;
    const N: usize = 8;
    const K2: usize = 4;
    const N2: usize = 8;

    fn matmul(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
        let mut out = vec![Goldilocks::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for t in 0..k {
                    acc = acc + a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    /// One honest chain witness `(x, w1, h, w2, y)`; when `x_in` is given it
    /// becomes the chain's input (the boundary buffer).
    fn chain_witness(
        rng: &mut XorShift64,
        m: usize,
        d: usize,
        k: usize,
        n: usize,
        x_in: Option<&[Goldilocks]>,
    ) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
        let x = match x_in {
            Some(v) => v.to_vec(),
            None => (0..m * d).map(|_| rng.field()).collect(),
        };
        let w1: Vec<Goldilocks> = (0..d * k).map(|_| rng.field()).collect();
        let h = matmul(&x, &w1, m, d, k);
        let w2: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let y = matmul(&h, &w2, m, k, n);
        (x, w1, h, w2, y)
    }

    fn fixture() -> (
        TwoShardChainDagWhir,
        TwoShardChainDagStatement,
        TwoShardChainDagProof,
        Vec<Vec<Goldilocks>>,
    ) {
        let dag = TwoShardChainDagWhir::new(M, D, K, N, K2, N2, 90, 0).expect("valid dims");
        let mut rng = XorShift64::new(0xD46);
        let (x, w1, h, w2, y) = chain_witness(&mut rng, M, D, K, N, None);
        let (_, w1_2, h_2, w2_2, y_2) = chain_witness(&mut rng, M, N, K2, N2, Some(&y));
        let (s, p) = prove(
            &dag, &x, &w1, &h, &w2, &y, M, D, K, N, &w1_2, &h_2, &w2_2, &y_2, K2, N2,
        )
        .expect("honest prove");
        (dag, s, p, vec![x, w1, h, w2, y, w1_2, h_2, w2_2, y_2])
    }

    fn all_whirs(dag: &TwoShardChainDagWhir) -> [&zkie_core::pcs::whir::Whir; 10] {
        [
            &dag.upstream.x,
            &dag.upstream.w1,
            &dag.upstream.h,
            &dag.upstream.w2,
            &dag.upstream.y,
            &dag.downstream.x,
            &dag.downstream.w1,
            &dag.downstream.h,
            &dag.downstream.w2,
            &dag.downstream.y,
        ]
    }

    /// Honest root-only roundtrip: after ALL raw witness buffers are
    /// dropped, the verifier accepts from statement + proof only, and none
    /// of the ten WHIR instances ever opens.
    #[test]
    fn honest_roundtrip_after_witness_dropped() {
        let (dag, s, p, witnesses) = fixture();
        drop(witnesses);
        let before: Vec<_> = all_whirs(&dag).iter().map(|w| w.open_stats()).collect();
        assert!(verify(&dag, &s, &p));
        let after: Vec<_> = all_whirs(&dag).iter().map(|w| w.open_stats()).collect();
        assert_eq!(before, after, "verifier must never open");
    }

    /// Splice: two INDEPENDENTLY valid DAG artifacts (distinct witnesses ->
    /// distinct boundary roots). Each verifies standalone; a composed
    /// artifact pairing upstream of A with downstream of B must be rejected
    /// by the boundary-root check.
    #[test]
    fn spliced_mismatched_boundary_rejected() {
        let dag = TwoShardChainDagWhir::new(M, D, K, N, K2, N2, 90, 0).expect("valid dims");
        let mut rng_a = XorShift64::new(0xD46);
        let (xa, w1a, ha, w2a, ya) = chain_witness(&mut rng_a, M, D, K, N, None);
        let (_, w1a_2, ha_2, w2a_2, ya_2) = chain_witness(&mut rng_a, M, N, K2, N2, Some(&ya));
        let (sa, pa) = prove(
            &dag, &xa, &w1a, &ha, &w2a, &ya, M, D, K, N, &w1a_2, &ha_2, &w2a_2, &ya_2, K2, N2,
        )
        .expect("artifact A");

        let mut rng_b = XorShift64::new(0xBAD);
        let (xb, w1b, hb, w2b, yb) = chain_witness(&mut rng_b, M, D, K, N, None);
        let (_, w1b_2, hb_2, w2b_2, yb_2) = chain_witness(&mut rng_b, M, N, K2, N2, Some(&yb));
        let (sb, pb) = prove(
            &dag, &xb, &w1b, &hb, &w2b, &yb, M, D, K, N, &w1b_2, &hb_2, &w2b_2, &yb_2, K2, N2,
        )
        .expect("artifact B");

        // Each standalone DAG verifies.
        assert!(verify(&dag, &sa, &pa));
        assert!(verify(&dag, &sb, &pb));
        // The boundary roots differ across the two artifacts.
        assert_ne!(sa.upstream.root_y, sb.downstream.root_x);

        // Composed: upstream from A, downstream from B -> boundary mismatch.
        let spliced_stmt = TwoShardChainDagStatement {
            upstream: sa.upstream.clone(),
            downstream: sb.downstream.clone(),
        };
        let spliced_proof = TwoShardChainDagProof {
            upstream: pa.upstream.clone(),
            downstream: pb.downstream.clone(),
        };
        assert!(!verify(&dag, &spliced_stmt, &spliced_proof));
    }

    /// Malformed boundary metadata: wrong dimensions and a tampered
    /// boundary root are each rejected before the chain verifiers run.
    #[test]
    fn malformed_boundary_metadata_rejected() {
        let (dag, s, p, _) = fixture();
        assert!(verify(&dag, &s, &p));

        // Wrong boundary dimensions: downstream.d must equal upstream.n.
        let mut bad = s.clone();
        bad.downstream.d = 4; // != upstream.n (8)
        assert!(!verify(&dag, &bad, &p));
        // Downstream.m must equal upstream.m.
        let mut bad = s.clone();
        bad.downstream.m = 4;
        assert!(!verify(&dag, &bad, &p));
        // Tampered boundary root.
        let mut bad = s.clone();
        let mut rng = XorShift64::new(0xE47);
        let fake: Vec<Goldilocks> = (0..M * N).map(|_| rng.field()).collect();
        let (fake_root, _, _) = dag.downstream.x.commit(&fake);
        bad.downstream.root_x = fake_root;
        assert!(!verify(&dag, &bad, &p));
    }
}
