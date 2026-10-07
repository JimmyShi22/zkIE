//! Committed N=8 LogUp multiset primitive.
//!
//! q/t/m roots are fixed before beta. iq/it roots are derived after beta.
//! Inverse identities are eq-weighted at fresh transcript points after those
//! roots, so errors cannot cancel across entries. The logical N=8 tensors are
//! padded to arity 7; tail q/t/m values are separately eq-weighted to zero.
//! one is verifier-known, never a prover commitment. This still excludes
//! index-to-output wiring, batching, and a full 90-bit end-to-end analysis.

use crate::extension_lookup_fractional::{gen, to_p3_point};
use zkie_core::common::field::{BasedVectorSpace, Field, Goldilocks, PrimeCharacteristicRing, EF};
use zkie_core::common::sumcheck::{interpolate_f, virtual_round_pvals_f};
use zkie_core::common::transcript::TwoPhaseTranscript;
use zkie_core::pcs::whir::{Commitment, OpeningProtocol, Proof, ProverData, Whir};

pub const N: usize = 8;
pub const ARITY: usize = 7;
const PAD: usize = 1 << ARITY;
const ROUNDS: usize = ARITY;
const PROTOCOL: &str = "zkie/ext-logup-multiset/v2";
const INV_Q: usize = 0;
const INV_T: usize = 1;
const MULTI: usize = 2;
const TAIL_Q: usize = 3;
const TAIL_T: usize = 4;
const TAIL_M: usize = 5;

#[derive(Clone, Debug)]
pub struct TensorCommitment {
    pub re: Commitment,
    pub im: Commitment,
}
#[derive(Clone, Debug)]
pub struct Statement {
    pub q: TensorCommitment,
    pub t: TensorCommitment,
    pub m: TensorCommitment,
    pub iq: TensorCommitment,
    pub it: TensorCommitment,
}
impl Statement {
    fn initial(&self) -> [&Commitment; 6] {
        [
            &self.q.re, &self.q.im, &self.t.re, &self.t.im, &self.m.re, &self.m.im,
        ]
    }
    fn derived(&self) -> [&Commitment; 4] {
        [&self.iq.re, &self.iq.im, &self.it.re, &self.it.im]
    }
    fn all(&self) -> [&TensorCommitment; 5] {
        [&self.q, &self.t, &self.m, &self.iq, &self.it]
    }
}
#[derive(Clone)]
pub struct TensorOpen {
    pub re: (Proof, EF),
    pub im: (Proof, EF),
}
#[derive(Clone)]
pub struct RelationProof {
    pub r: Vec<EF>,
    pub rounds: Vec<Vec<EF>>,
    pub z: Vec<EF>,
    pub opens: Vec<TensorOpen>,
}
#[derive(Clone)]
pub struct MultisetProof {
    pub relations: Vec<RelationProof>,
}
struct Data {
    root: TensorCommitment,
    re_pd: ProverData,
    im_pd: ProverData,
    protocol: OpeningProtocol,
    values: Vec<EF>,
}

fn commit(w: &Whir, values: Vec<EF>) -> Data {
    let (mut re, mut im) = (Vec::with_capacity(PAD), Vec::with_capacity(PAD));
    for x in &values {
        let c = x.as_basis_coefficients_slice();
        re.push(c[0]);
        im.push(c[1]);
    }
    let (rr, re_pd, protocol) = w.commit(&re);
    let (ri, im_pd, _) = w.commit(&im);
    Data {
        root: TensorCommitment { re: rr, im: ri },
        re_pd,
        im_pd,
        protocol,
        values,
    }
}
fn open(w: &Whir, d: &Data, z: &[EF]) -> TensorOpen {
    let p = to_p3_point(z);
    TensorOpen {
        re: w.open_ef(&d.root.re, d.re_pd.clone(), &d.protocol, &p),
        im: w.open_ef(&d.root.im, d.im_pd.clone(), &d.protocol, &p),
    }
}
fn opened(x: &TensorOpen) -> EF {
    x.re.1 + gen() * x.im.1
}
fn verify_open(w: &Whir, root: &TensorCommitment, x: &TensorOpen, z: &[EF]) -> Option<EF> {
    let p = to_p3_point(z);
    let proto = w.opening_protocol(ARITY, 1);
    let re = w.verify_ef(&root.re, &x.re.0, &proto, &p).ok()?;
    let im = w.verify_ef(&root.im, &x.im.0, &proto, &p).ok()?;
    (re == x.re.1 && im == x.im.1).then(|| opened(x))
}
fn eq_values(r: &[EF]) -> Vec<EF> {
    let mut out = vec![EF::ONE; PAD];
    for (j, &x) in r.iter().enumerate() {
        for (i, v) in out.iter_mut().enumerate() {
            *v = *v * if (i >> j) & 1 == 1 { x } else { EF::ONE - x };
        }
    }
    out
}
fn eval_eq(r: &[EF], z: &[EF]) -> EF {
    r.iter().zip(z).fold(EF::ONE, |a, (&ri, &zi)| {
        a * (zi * ri + (EF::ONE - zi) * (EF::ONE - ri))
    })
}
fn head() -> Vec<EF> {
    let mut x = vec![EF::ZERO; PAD];
    for v in x.iter_mut().take(N) {
        *v = EF::ONE;
    }
    x
}
fn tail() -> Vec<EF> {
    head().into_iter().map(|x| EF::ONE - x).collect()
}
fn has_r(k: usize) -> bool {
    k != MULTI
}
fn terms(k: usize, beta: EF) -> Vec<(EF, Vec<usize>)> {
    match k {
        INV_Q => vec![
            (beta, vec![3, 5]),
            (EF::ONE, vec![0, 3, 5]),
            (EF::NEG_ONE, vec![5]),
        ],
        INV_T => vec![
            (beta, vec![4, 5]),
            (EF::ONE, vec![1, 4, 5]),
            (EF::NEG_ONE, vec![5]),
        ],
        MULTI => vec![(EF::ONE, vec![5, 3]), (EF::NEG_ONE, vec![5, 2, 4])],
        TAIL_Q => vec![(EF::ONE, vec![5, 6, 0])],
        TAIL_T => vec![(EF::ONE, vec![5, 6, 1])],
        TAIL_M => vec![(EF::ONE, vec![5, 6, 2])],
        _ => unreachable!(),
    }
}
fn terminal(k: usize, beta: EF, r: &[EF], z: &[EF], v: &[EF]) -> EF {
    match k {
        INV_Q => eval_eq(r, z) * ((beta + v[0]) * v[3] - EF::ONE),
        INV_T => eval_eq(r, z) * ((beta + v[1]) * v[4] - EF::ONE),
        MULTI => fold(&head(), z) * (v[3] - v[2] * v[4]),
        TAIL_Q => eval_eq(r, z) * fold(&tail(), z) * v[0],
        TAIL_T => eval_eq(r, z) * fold(&tail(), z) * v[1],
        TAIL_M => eval_eq(r, z) * fold(&tail(), z) * v[2],
        _ => EF::ZERO,
    }
}
fn fold(values: &[EF], z: &[EF]) -> EF {
    let mut b = values.to_vec();
    let mut n = b.len();
    for &r in z {
        let h = n / 2;
        for i in 0..h {
            b[i] = b[2 * i] + r * (b[2 * i + 1] - b[2 * i]);
        }
        n = h;
    }
    b[0]
}
fn prove_relation(
    w: &Whir,
    all: &[&Data; 5],
    beta: EF,
    k: usize,
    tx: &mut TwoPhaseTranscript,
) -> RelationProof {
    let r = if has_r(k) {
        tx.sample_vec(ARITY)
    } else {
        vec![]
    };
    let mut bufs: Vec<Vec<EF>> = all.iter().map(|d| d.values.clone()).collect();
    if k == MULTI {
        bufs.push(head());
    } else {
        bufs.push(eq_values(&r));
        if k >= TAIL_Q {
            bufs.push(tail());
        }
    }
    let mut rounds = Vec::with_capacity(ROUNDS);
    let mut z = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let p = virtual_round_pvals_f(&bufs, &terms(k, beta), 3);
        for &x in &p {
            tx.absorb(x);
        }
        let c = tx.sample();
        let h = bufs[0].len() / 2;
        for b in &mut bufs {
            for i in 0..h {
                b[i] = b[2 * i] + c * (b[2 * i + 1] - b[2 * i]);
            }
            b.truncate(h);
        }
        rounds.push(p);
        z.push(c);
    }
    RelationProof {
        r,
        rounds,
        z: z.clone(),
        opens: all.iter().map(|d| open(w, d, &z)).collect(),
    }
}
fn verify_relation(
    w: &Whir,
    all: &[&TensorCommitment; 5],
    beta: EF,
    k: usize,
    tx: &mut TwoPhaseTranscript,
    p: &RelationProof,
) -> bool {
    let r = if has_r(k) {
        let r = tx.sample_vec(ARITY);
        if p.r != r {
            return false;
        }
        r
    } else {
        if !p.r.is_empty() {
            return false;
        }
        vec![]
    };
    if p.rounds.len() != ROUNDS || p.z.len() != ROUNDS || p.opens.len() != 5 {
        return false;
    }
    let mut previous = EF::ZERO;
    for (round, &claimed) in p.rounds.iter().zip(&p.z) {
        if round.len() != 4 || round[0] + round[1] != previous {
            return false;
        }
        for &x in round {
            tx.absorb(x);
        }
        let z = tx.sample();
        if z != claimed {
            return false;
        }
        previous = interpolate_f(round, z, 3);
    }
    let mut v = Vec::with_capacity(5);
    for (root, x) in all.iter().zip(&p.opens) {
        let Some(y) = verify_open(w, root, x, &p.z) else {
            return false;
        };
        v.push(y);
    }
    previous == terminal(k, beta, &r, &p.z, &v)
}
fn default_witness() -> (Vec<EF>, Vec<EF>, Vec<EF>) {
    let mut t = vec![EF::ZERO; PAD];
    for (i, x) in t.iter_mut().take(N).enumerate() {
        *x = EF::from(Goldilocks::from_u64((i + 3) as u64));
    }
    let mut q = vec![EF::ZERO; PAD];
    let p = [3, 0, 7, 2, 5, 1, 6, 4];
    for (i, j) in p.into_iter().enumerate() {
        q[i] = t[j];
    }
    let mut m = vec![EF::ZERO; PAD];
    for v in m.iter_mut().take(N) {
        *v = EF::ONE;
    }
    (q, t, m)
}
enum InverseMode {
    Honest,
    AggregateConstant,
}
fn prove_from(
    w: &Whir,
    q: Vec<EF>,
    t: Vec<EF>,
    m: Vec<EF>,
    mode: InverseMode,
) -> Option<(Statement, MultisetProof)> {
    if w.num_variables() != ARITY || q.len() != PAD || t.len() != PAD || m.len() != PAD {
        return None;
    }
    let q = commit(w, q);
    let t = commit(w, t);
    let m = commit(w, m);
    let initial = [
        &q.root.re, &q.root.im, &t.root.re, &t.root.im, &m.root.re, &m.root.im,
    ];
    let mut tx = TwoPhaseTranscript::new(PROTOCOL, &[N], &initial);
    let (_, beta) = tx.sample_derived_challenges();
    let mut iq: Vec<EF> = q.values.iter().map(|&x| (beta + x).inverse()).collect();
    let mut it: Vec<EF> = t.values.iter().map(|&x| (beta + x).inverse()).collect();
    match mode {
        InverseMode::Honest => {}
        InverseMode::AggregateConstant => {
            let sum = q.values.iter().fold(EF::ZERO, |a, &x| a + x);
            assert_eq!(sum, t.values.iter().fold(EF::ZERO, |a, &x| a + x));
            let domain = EF::from(Goldilocks::from_u64(PAD as u64));
            let a = domain * (domain * beta + sum).inverse();
            iq = vec![a; PAD];
            it = vec![a; PAD];
        }
    }
    let iq = commit(w, iq);
    let it = commit(w, it);
    let s = Statement {
        q: q.root.clone(),
        t: t.root.clone(),
        m: m.root.clone(),
        iq: iq.root.clone(),
        it: it.root.clone(),
    };
    tx.absorb_derived_roots(&s.derived());
    let all = [&q, &t, &m, &iq, &it];
    Some((
        s,
        MultisetProof {
            relations: (0..6)
                .map(|k| prove_relation(w, &all, beta, k, &mut tx))
                .collect(),
        },
    ))
}
pub fn prove(w: &Whir) -> Option<(Statement, MultisetProof)> {
    let (q, t, m) = default_witness();
    prove_from(w, q, t, m, InverseMode::Honest)
}
pub fn verify(w: &Whir, s: &Statement, p: &MultisetProof) -> bool {
    if w.num_variables() != ARITY || p.relations.len() != 6 {
        return false;
    }
    let mut tx = TwoPhaseTranscript::new(PROTOCOL, &[N], &s.initial());
    let (_, beta) = tx.sample_derived_challenges();
    tx.absorb_derived_roots(&s.derived());
    let all = s.all();
    p.relations
        .iter()
        .enumerate()
        .all(|(k, p)| verify_relation(w, &all, beta, k, &mut tx, p))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Whir, Statement, MultisetProof) {
        let w = Whir::new_target(ARITY, 90, 0).expect("arity");
        let (s, p) = prove(&w).expect("proof");
        (w, s, p)
    }
    #[test]
    fn honest_statement_only() {
        let (w, s, p) = fixture();
        let n = w.open_stats();
        assert!(verify(&w, &s, &p));
        assert_eq!(n, w.open_stats());
    }
    #[test]
    fn roots_and_proof_tamper() {
        let (w, mut s, p) = fixture();
        let (r, _, _) = w.commit(&vec![Goldilocks::from_u64(9); PAD]);
        s.q.re = r;
        assert!(!verify(&w, &s, &p));
        let (w, s, mut p) = fixture();
        p.relations[0].rounds[0][0] = p.relations[0].rounds[0][0] + EF::ONE;
        assert!(!verify(&w, &s, &p));
    }
    fn beta(s: &Statement) -> EF {
        let mut tx = TwoPhaseTranscript::new(PROTOCOL, &[N], &s.initial());
        tx.sample_derived_challenges().1
    }
    fn sum(xs: &[EF]) -> EF {
        xs.iter().fold(EF::ZERO, |a, &x| a + x)
    }
    #[test]
    fn balanced_inverse_sum_old_bug_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("arity");
        let mut t = vec![EF::ZERO; PAD];
        for (i, x) in t.iter_mut().take(N).enumerate() {
            *x = EF::from(Goldilocks::from_u64((i + 3) as u64));
        }
        let mut q = vec![EF::ZERO; PAD];
        for (i, x) in [2u64, 3, 4, 5, 6, 7, 8, 17].into_iter().enumerate() {
            q[i] = EF::from(Goldilocks::from_u64(x));
        }
        assert_eq!(
            sum(&q),
            sum(&t),
            "unequal multisets deliberately retain the same sum"
        );
        let m = vec![EF::ONE; PAD];
        let (s, p) = prove_from(
            &w,
            q.clone(),
            t.clone(),
            m.clone(),
            InverseMode::AggregateConstant,
        )
        .expect("malicious proof");
        let b = beta(&s);
        let domain = EF::from(Goldilocks::from_u64(PAD as u64));
        let a = domain * (domain * b + sum(&q)).inverse();
        assert_eq!(
            sum(&q.iter().map(|&x| (b + x) * a - EF::ONE).collect::<Vec<_>>()),
            EF::ZERO
        );
        assert_eq!(
            sum(&t.iter().map(|&x| (b + x) * a - EF::ONE).collect::<Vec<_>>()),
            EF::ZERO
        );
        assert_eq!(
            sum(&m.iter().map(|&x| a - x * a).collect::<Vec<_>>()),
            EF::ZERO
        );
        assert!(
            !verify(&w, &s, &p),
            "old aggregate equations hold but new eq checks reject"
        );
    }
    #[test]
    fn prefix_padding_swap_old_full_domain_relation_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("arity");
        let (mut q, mut t, _) = default_witness();
        q.swap(0, N);
        t.swap(0, N);
        let m = vec![EF::ONE; PAD];
        let (s, p) = prove_from(&w, q.clone(), t.clone(), m.clone(), InverseMode::Honest)
            .expect("swapped proof");
        let b = beta(&s);
        let iq: Vec<EF> = q.iter().map(|&x| (b + x).inverse()).collect();
        let it: Vec<EF> = t.iter().map(|&x| (b + x).inverse()).collect();
        assert_eq!(
            sum(&q
                .iter()
                .zip(&iq)
                .map(|(&x, &i)| (b + x) * i - EF::ONE)
                .collect::<Vec<_>>()),
            EF::ZERO
        );
        assert_eq!(
            sum(&t
                .iter()
                .zip(&it)
                .map(|(&x, &i)| (b + x) * i - EF::ONE)
                .collect::<Vec<_>>()),
            EF::ZERO
        );
        assert_eq!(
            sum(&iq
                .iter()
                .zip(m.iter().zip(&it))
                .map(|(&i, (&mm, &j))| i - mm * j)
                .collect::<Vec<_>>()),
            EF::ZERO
        );
        assert!(
            !verify(&w, &s, &p),
            "fixed prefix/tail semantics reject a full-domain-preserving swap"
        );
    }
    #[test]
    fn no_prover_one_commitment() {
        let (_, s, _) = fixture();
        let _ = s.q; /* Statement intentionally has no one field. */
    }
}
