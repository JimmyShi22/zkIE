//! Non-recursive, MODEL-BOUND two-shard aggregation artifact: a public
//! statement anchoring the committed I/O roots AND the complete model
//! metadata (dimensions + weight roots of BOTH chains) to the enclosed
//! `extension_two_shard_chain_dag` artifact.
//!
//! Anchor contract (checked by the root-only verifier, in order):
//! 1. I/O: `input_root == enclosed.upstream.root_x` and
//!    `output_root == enclosed.downstream.root_y`;
//! 2. model binding: for EACH chain, the public dimensions `m, d, k, n` and
//!    the weight roots `root_w1`/`root_w2` byte-equal the enclosed chain
//!    statements (no prover-supplied relabeling of the model either);
//! 3. the existing two-shard DAG verifier (delegated verbatim — the inner
//!    verification is NEVER duplicated here).
//!
//! The envelope is explicitly NON-recursive and NOT succinct: it retains the
//! complete `TwoShardChainDagStatement` AND `TwoShardChainDagProof` of the
//! enclosed artifact, unchanged. This module only anchors and delegates.

use crate::extension_two_shard_chain_dag::{
    self, TwoShardChainDagProof, TwoShardChainDagStatement, TwoShardChainDagWhir,
};
use zkie_core::pcs::whir::Commitment;

/// Model-binding metadata for ONE chain: the dimensions and the two weight
/// roots. Together with the I/O roots, this fixes the complete public
/// statement of one chain artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainModelBinding {
    pub m: usize,
    pub d: usize,
    pub k: usize,
    pub n: usize,
    pub root_w1: Commitment,
    pub root_w2: Commitment,
}

/// Public I/O + model statement: the committed input/output roots AND the
/// complete model binding of BOTH enclosed chains.
#[derive(Clone, Debug)]
pub struct AggregationStatement {
    pub input_root: Commitment,
    pub output_root: Commitment,
    pub upstream: ChainModelBinding,
    pub downstream: ChainModelBinding,
}

impl AggregationStatement {
    /// Build the complete public statement FROM a DAG statement — the exact
    /// anchored fields (I/O roots and both model bindings).
    pub fn from_dag_statement(ds: &TwoShardChainDagStatement) -> Self {
        AggregationStatement {
            input_root: ds.upstream.root_x.clone(),
            output_root: ds.downstream.root_y.clone(),
            upstream: ChainModelBinding {
                m: ds.upstream.m,
                d: ds.upstream.d,
                k: ds.upstream.k,
                n: ds.upstream.n,
                root_w1: ds.upstream.root_w1.clone(),
                root_w2: ds.upstream.root_w2.clone(),
            },
            downstream: ChainModelBinding {
                m: ds.downstream.m,
                d: ds.downstream.d,
                k: ds.downstream.k,
                n: ds.downstream.n,
                root_w1: ds.downstream.root_w1.clone(),
                root_w2: ds.downstream.root_w2.clone(),
            },
        }
    }
}

/// One chain's public binding against its enclosed chain statement: the
/// dimensions and both weight roots byte-equal.
fn binding_matches(stmt: &crate::extension_chain::ChainStatement, b: &ChainModelBinding) -> bool {
    stmt.m == b.m
        && stmt.d == b.d
        && stmt.k == b.k
        && stmt.n == b.n
        && stmt.root_w1 == b.root_w1
        && stmt.root_w2 == b.root_w2
}

/// All public fields (I/O anchors AND both model bindings) against the
/// enclosed DAG statement.
fn anchors_match(io: &AggregationStatement, ds: &TwoShardChainDagStatement) -> bool {
    io.input_root == ds.upstream.root_x
        && io.output_root == ds.downstream.root_y
        && binding_matches(&ds.upstream, &io.upstream)
        && binding_matches(&ds.downstream, &io.downstream)
}

/// The envelope: the COMPLETE enclosed DAG artifact (statement + proof,
/// retained verbatim). Non-recursive and not succinct by construction.
#[derive(Clone)]
pub struct AggregationEnvelope {
    pub dag_statement: TwoShardChainDagStatement,
    pub dag_proof: TwoShardChainDagProof,
}

/// Wrap a proven DAG artifact into an envelope. Rejects unless ALL public
/// fields (I/O anchors AND both model bindings) equal the enclosed DAG
/// statement's fields — checked at prove time too, so a caller can never
/// mint an envelope whose public statement disagrees with its content.
pub fn prove(
    _dag: &TwoShardChainDagWhir,
    io: &AggregationStatement,
    dag_statement: TwoShardChainDagStatement,
    dag_proof: TwoShardChainDagProof,
) -> Option<AggregationEnvelope> {
    if !anchors_match(io, &dag_statement) {
        return None;
    }
    Some(AggregationEnvelope {
        dag_statement,
        dag_proof,
    })
}

/// Root-only verify: ALL public binding checks (I/O anchors and both model
/// bindings) against the enclosed DAG statement first, then the existing
/// two-shard DAG verifier. Statement and proof only — no witness, no
/// recomputation, no `open` calls; malformed shapes return `false` (never
/// panic).
pub fn verify(
    dag: &TwoShardChainDagWhir,
    io: &AggregationStatement,
    envelope: &AggregationEnvelope,
) -> bool {
    if !anchors_match(io, &envelope.dag_statement) {
        return false;
    }
    extension_two_shard_chain_dag::verify(dag, &envelope.dag_statement, &envelope.dag_proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};

    // Same smallest dims as the DAG module's tests (every WHIR arity above
    // the folding floor).
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

    /// Full artifact: dag whir, public I/O statement, envelope, and all raw
    /// witness buffers (for the drop test).
    fn fixture() -> (
        TwoShardChainDagWhir,
        AggregationStatement,
        AggregationEnvelope,
        Vec<Vec<Goldilocks>>,
    ) {
        let dag = TwoShardChainDagWhir::new(M, D, K, N, K2, N2, 90, 0).expect("valid dims");
        let mut rng = XorShift64::new(0xA66);
        let (x, w1, h, w2, y) = chain_witness(&mut rng, M, D, K, N, None);
        let (_, w1_2, h_2, w2_2, y_2) = chain_witness(&mut rng, M, N, K2, N2, Some(&y));
        let (dag_stmt, dag_proof) = extension_two_shard_chain_dag::prove(
            &dag, &x, &w1, &h, &w2, &y, M, D, K, N, &w1_2, &h_2, &w2_2, &y_2, K2, N2,
        )
        .expect("honest dag prove");
        let io = AggregationStatement::from_dag_statement(&dag_stmt);
        let envelope = prove(&dag, &io, dag_stmt, dag_proof).expect("anchor ok");
        (dag, io, envelope, vec![x, w1, h, w2, y, w1_2, h_2, w2_2, y_2])
    }

    /// Honest roundtrip: after ALL raw witness buffers are dropped, the
    /// root-only verifier accepts from the public statement + envelope only
    /// and never opens. Also asserts the envelope RETAINS the full child DAG
    /// proof (per-shard round counts match the dims).
    #[test]
    fn honest_after_witnesses_dropped() {
        let (dag, io, envelope, witnesses) = fixture();
        // The envelope retains the complete child DAG proof: upstream shard
        // rounds = (log2 k, log2 d) = (3, 2); downstream = (log2 d2, log2 k2)
        // = (3, 2).
        assert_eq!(envelope.dag_proof.upstream.rounds2.len(), 3);
        assert_eq!(envelope.dag_proof.upstream.rounds1.len(), 2);
        assert_eq!(envelope.dag_proof.downstream.rounds1.len(), 3);
        assert_eq!(envelope.dag_proof.downstream.rounds2.len(), 2);
        drop(witnesses);
        let whirs: Vec<&zkie_core::pcs::whir::Whir> = vec![
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
        ];
        let before: Vec<_> = whirs.iter().map(|w| w.open_stats()).collect();
        assert!(verify(&dag, &io, &envelope));
        let after: Vec<_> = whirs.iter().map(|w| w.open_stats()).collect();
        assert_eq!(before, after, "verifier must never open");
    }

    /// Swapped public anchors must be rejected at prove time AND at verify
    /// time (the public I/O roots are the DAG's boundary roots, in order).
    #[test]
    fn swapped_anchors_rejected() {
        let (dag, io, envelope, _) = fixture();
        assert!(verify(&dag, &io, &envelope));
        let swapped = AggregationStatement {
            input_root: io.output_root.clone(),
            output_root: io.input_root.clone(),
            ..io.clone()
        };
        assert!(!verify(&dag, &swapped, &envelope));
        // prove refuses to mint an envelope with mismatched anchors.
        assert!(prove(
            &dag,
            &swapped,
            envelope.dag_statement.clone(),
            envelope.dag_proof.clone()
        )
        .is_none());
    }

    /// Tampered public I/O roots must be rejected by the anchor check.
    #[test]
    fn tampered_io_roots_rejected() {
        let (dag, io, envelope, _) = fixture();
        let mut rng = XorShift64::new(0x0F7);
        let fake: Vec<Goldilocks> = (0..M * D).map(|_| rng.field()).collect();
        let (fake_root, _, _) = dag.upstream.x.commit(&fake);
        let mut bad = io.clone();
        bad.input_root = fake_root.clone();
        assert!(!verify(&dag, &bad, &envelope));
        let mut bad = io.clone();
        bad.output_root = fake_root;
        assert!(!verify(&dag, &bad, &envelope));
    }

    /// Tampered public MODEL binding (weight root or dimension) must be
    /// rejected by the public-binding gate while the envelope itself stays
    /// valid — the rejection is isolated to the public statement, not the
    /// enclosed artifact.
    #[test]
    fn tampered_model_binding_rejected() {
        let (dag, io, envelope, _) = fixture();
        // The envelope verifies with the honest public statement.
        assert!(verify(&dag, &io, &envelope));
        // Tampered public weight root: the envelope is unchanged.
        let mut rng = XorShift64::new(0x0F8);
        let fake: Vec<Goldilocks> = (0..D * K).map(|_| rng.field()).collect();
        let (fake_root, _, _) = dag.upstream.w1.commit(&fake);
        let mut bad = io.clone();
        bad.upstream.root_w1 = fake_root.clone();
        assert!(!verify(&dag, &bad, &envelope));
        assert!(prove(
            &dag,
            &bad,
            envelope.dag_statement.clone(),
            envelope.dag_proof.clone()
        )
        .is_none());
        // Tampered public dimension (downstream k): the envelope is
        // unchanged.
        let mut bad = io.clone();
        bad.downstream.k = 8; // honest value is 4
        assert!(!verify(&dag, &bad, &envelope));
        assert!(prove(
            &dag,
            &bad,
            envelope.dag_statement.clone(),
            envelope.dag_proof.clone()
        )
        .is_none());
        // Sanity: the honest statement still verifies after the tampering
        // attempts.
        assert!(verify(&dag, &io, &envelope));
    }
}
