//! Generic root-bound fractional LogUp lookup (idx/out/table/multiset), with
//! TWO independent rational relations sharing the SAME committed
//! multiplicity tensor `m`:
//!
//! 1. index relation: `sum_i 1/(gamma + idx_i) = sum_j m_j/(gamma + j)`;
//! 2. pair relation:
//!    `sum_i 1/(alpha + idx_i + beta*out_i) =
//!     sum_j m_j/(alpha + j + beta*table_j)`.
//!
//! Shape: arbitrary power-of-two `P` rows and `T` table entries (`>= 2`);
//! each domain is committed at arity `max(5, log2 size)` — the WHIR folding
//! floor is 5 — so small domains carry a padded tail. The padded tail
//! entries are MASKED/IGNORED by the logical head sums (the head indicator
//! restricts every sum and the equality to the logical domain); they are
//! NOT canonical-zero constrained and carry no lookup semantics. No hidden
//! padding: the prover validates `idx.len() == out.len() == P`,
//! `table.len() == m.len() == T`, every `idx[i] < T`.
//!
//! Transcript schedule (roots before challenges, never `m` after the
//! initial roots):
//! 1. initial roots `[idx, out, table, m]` (EF pairs) at transcript
//!    construction;
//! 2. `(u, alpha) = sample_derived_challenges()`; `gamma = u`,
//!    `beta = u + alpha` (deterministic transcript-derived challenges);
//! 3. derived inverse commitments `iq_idx`, `it_idx`, `iq_pair`, `it_pair`
//!    committed from the challenges and absorbed via `absorb_derived_roots`;
//! 4. the four claimed sums absorbed;
//! 5. per relation: fresh eq point `r` (inverse relations), then interactive
//!    absorb-every-coefficient rounds with fresh fold challenges `z`,
//!    terminal WHIR openings at `z`.
//!
//! Relation soundness (each relation is an independent virtual sumcheck
//! over degree <= 3 products, full EF throughout — authenticated values are
//! never downcast):
//! - inverse relations are eq-weighted POINTWISE identities
//!   `inv * (denom) - 1 == 0` (claimed final value 0);
//! - sum relations are head-weighted plain sums with the CLAIMED scalar as
//!   the initial round claim (standard sumcheck: the first round binds the
//!   sum, the final round binds the committed values via the openings at
//!   `z`);
//! - `s_idx_row == s_idx_entry` and `s_pair_row == s_pair_entry` close the
//!   two rational equalities (Schwartz–Zippel over gamma / (alpha, beta)).
//!
//! Verifier = `(lookup_whir, statement, proof)` ONLY: no witness, no Store,
//! no raw tensors, no `open` calls. A generic actual-IR adapter
//! (`prove_op_lookup`) recognizes `Op::Lookup` and materializes the three
//! logical tensors from a real `Store` (validated, mmap-safe). The direct
//! bridge in `extension_direct_lookup` and all legacy modules remain
//! untouched.

use crate::compose::{Op, Store, TensorData};
use crate::extension_lookup_fractional::{fold_ef, gen, to_p3_point};
use zkie_core::common::field::{
    BasedVectorSpace, EF, Field, Goldilocks, PrimeCharacteristicRing, PrimeField64,
};
use zkie_core::common::sumcheck::{interpolate_f, virtual_round_pvals_f};
use zkie_core::common::transcript::TwoPhaseTranscript;
use zkie_core::pcs::whir::{Commitment, Proof, ProverData, Whir};

pub const PROTOCOL: &str = "zkie/ext-lookup-logup/v1";

// Relation ids, in transcript order.
const INV_IDX_ROW: usize = 0;
const INV_IDX_ENTRY: usize = 1;
const INV_PAIR_ROW: usize = 2;
const INV_PAIR_ENTRY: usize = 3;
const SUM_IDX_ROW: usize = 4;
const SUM_IDX_ENTRY: usize = 5;
const SUM_PAIR_ROW: usize = 6;
const SUM_PAIR_ENTRY: usize = 7;
const RELATIONS: usize = 8;

/// One `Whir` instance per domain (a WHIR instance is bound to one arity).
pub struct LookupWhir {
    pub rows: Whir,
    pub entries: Whir,
}

impl LookupWhir {
    /// Build the two instances for `P` rows and `T` table entries (both
    /// powers of two, `>= 2`). Returns `None` for malformed dims or
    /// infeasible WHIR configurations.
    pub fn new(
        p: usize,
        t: usize,
        security_level: usize,
        pow_budget: usize,
    ) -> Option<Self> {
        if !p.is_power_of_two() || !t.is_power_of_two() || p < 2 || t < 2 {
            return None;
        }
        let ar_p = (p.trailing_zeros() as usize).max(5);
        let ar_t = (t.trailing_zeros() as usize).max(5);
        Some(LookupWhir {
            rows: Whir::new_target(ar_p, security_level, pow_budget)?,
            entries: Whir::new_target(ar_t, security_level, pow_budget)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TensorCommitment {
    pub re: Commitment,
    pub im: Commitment,
}

/// Statement: `P`, `T` and the roots of the four initial tensors and the
/// four transcript-derived inverse tensors (all EF pairs). Root-only.
#[derive(Clone, Debug)]
pub struct LookupStatement {
    pub p: usize,
    pub t: usize,
    pub idx: TensorCommitment,
    pub out: TensorCommitment,
    pub table: TensorCommitment,
    pub m: TensorCommitment,
    pub iq_idx: TensorCommitment,
    pub it_idx: TensorCommitment,
    pub iq_pair: TensorCommitment,
    pub it_pair: TensorCommitment,
}

impl LookupStatement {
    fn initial(&self) -> [&Commitment; 8] {
        [
            &self.idx.re,
            &self.idx.im,
            &self.out.re,
            &self.out.im,
            &self.table.re,
            &self.table.im,
            &self.m.re,
            &self.m.im,
        ]
    }
    fn derived(&self) -> [&Commitment; 8] {
        [
            &self.iq_idx.re,
            &self.iq_idx.im,
            &self.it_idx.re,
            &self.it_idx.im,
            &self.iq_pair.re,
            &self.iq_pair.im,
            &self.it_pair.re,
            &self.it_pair.im,
        ]
    }
}

/// One opening pair (re/im) at one point.
#[derive(Clone)]
pub struct TensorOpen {
    pub re: (Proof, EF),
    pub im: (Proof, EF),
}

/// One relation's transcript: optional eq point `r` (inverse relations
/// only), the absorbed round polynomials, the fold challenges `z`, and the
/// terminal openings (per-relation fixed order).
#[derive(Clone)]
pub struct RelationProof {
    pub r: Vec<EF>,
    pub rounds: Vec<Vec<EF>>,
    pub z: Vec<EF>,
    pub opens: Vec<TensorOpen>,
}

/// Prover-side phase timings (diagnostics only — the root-only verifier
/// never reads them). `derived_inverse_commit_secs` covers the four inverse
/// tensor COMMITS only; the inverse ARITHMETIC is not included. Likewise
/// `terminal_open_secs` is the opening time inside the relations (subtracted
/// from `relation_prove_secs`).
#[derive(Clone, Copy, Debug, Default)]
pub struct ProveTimings {
    pub initial_commit_secs: f64,
    pub derived_inverse_commit_secs: f64,
    pub relation_prove_secs: f64,
    pub terminal_open_secs: f64,
}

/// The transported proof: the four claimed sums (absorbed before any
/// relation challenge), the eight relation transcripts, and the prover-side
/// phase timings (ignored by `verify`).
#[derive(Clone)]
pub struct LookupProof {
    pub s_idx_row: EF,
    pub s_idx_entry: EF,
    pub s_pair_row: EF,
    pub s_pair_entry: EF,
    pub relations: Vec<RelationProof>,
    pub timings: ProveTimings,
}

struct Data {
    root: TensorCommitment,
    re_pd: ProverData,
    im_pd: ProverData,
    values: Vec<EF>,
    arity: usize,
}

fn commit_ef_pair(w: &Whir, values: &[EF], arity: usize) -> Data {
    let size = 1usize << arity;
    assert_eq!(values.len(), size);
    let mut re = Vec::with_capacity(size);
    let mut im = Vec::with_capacity(size);
    for x in values {
        let c: &[Goldilocks] = x.as_basis_coefficients_slice();
        re.push(c[0]);
        im.push(c[1]);
    }
    let (re_root, re_pd, _) = w.commit(&re);
    let (im_root, im_pd, _) = w.commit(&im);
    Data {
        root: TensorCommitment { re: re_root, im: im_root },
        re_pd,
        im_pd,
        values: values.to_vec(),
        arity,
    }
}

/// Reuse scratch buffer `idx` (preserving its allocation): refill it from
/// `src`. `clear` keeps the capacity; the caller's later `truncate` never
/// shrinks the allocation either — so one prove reuses the same buffers for
/// every relation.
fn refill_scratch(scratch: &mut Vec<Vec<EF>>, src: &[EF], idx: usize) {
    if idx >= scratch.len() {
        scratch.push(Vec::with_capacity(src.len()));
    }
    let b = &mut scratch[idx];
    b.clear();
    b.extend_from_slice(src);
}

fn open_at(w: &Whir, d: &Data, z: &[EF]) -> TensorOpen {
    let p3 = to_p3_point(z);
    let proto = w.opening_protocol(d.arity, 1);
    TensorOpen {
        re: w.open_ef(&d.root.re, d.re_pd.clone(), &proto, &p3),
        im: w.open_ef(&d.root.im, d.im_pd.clone(), &proto, &p3),
    }
}

fn verify_open(
    w: &Whir,
    root: &TensorCommitment,
    x: &TensorOpen,
    z: &[EF],
    arity: usize,
) -> Option<EF> {
    let p3 = to_p3_point(z);
    let proto = w.opening_protocol(arity, 1);
    let re = w.verify_ef(&root.re, &x.re.0, &proto, &p3).ok()?;
    let im = w.verify_ef(&root.im, &x.im.0, &proto, &p3).ok()?;
    (re == x.re.1 && im == x.im.1).then(|| x.re.1 + gen() * x.im.1)
}

fn eq_values(r: &[EF], arity: usize) -> Vec<EF> {
    let n = 1usize << arity;
    let mut out = vec![EF::ONE; n];
    for (j, &x) in r.iter().enumerate() {
        let one_minus = EF::ONE - x;
        for (i, v) in out.iter_mut().enumerate() {
            *v = *v * if (i >> j) & 1 == 1 { x } else { one_minus };
        }
    }
    out
}

fn eq_point(r: &[EF], z: &[EF]) -> EF {
    r.iter().zip(z).fold(EF::ONE, |acc, (&ri, &zi)| {
        acc * (zi * ri + (EF::ONE - zi) * (EF::ONE - ri))
    })
}

/// Public head indicator: 1 on the first `len` flat entries.
fn head(len: usize, arity: usize) -> Vec<EF> {
    let mut v = vec![EF::ZERO; 1usize << arity];
    for x in v.iter_mut().take(len) {
        *x = EF::ONE;
    }
    v
}

/// Public index-value buffer: `J(k) = k` (flat index as a field element).
/// Materialized only for the PROVER-side virtual polynomials.
fn j_values(arity: usize) -> Vec<EF> {
    (0..(1usize << arity))
        .map(|k| EF::from(Goldilocks::from_u64(k as u64)))
        .collect()
}

/// Closed-form MLE evaluation of `J(k) = k` (LSB-first project bit order,
/// arity <= 32): `J(z) = sum_b 2^b * z_b`. Allocation-free — used by the
/// VERIFIER terminal path.
fn j_eval(z: &[EF]) -> EF {
    z.iter().enumerate().fold(EF::ZERO, |acc, (b, &zb)| {
        acc + EF::from(Goldilocks::from_u64((1usize << b) as u64)) * zb
    })
}

/// Closed-form MLE evaluation of the head indicator (1 on the first
/// `len = 2^m` flat entries, LSB-first): entries `< len` have their high
/// bits `m..arity` equal to zero, so
/// `head(z) = prod_{b=m}^{arity-1} (1 - z_b)` (empty product = 1 for the
/// full domain). Allocation-free — used by the VERIFIER terminal path.
fn head_eval(len: usize, z: &[EF]) -> EF {
    let m = len.trailing_zeros() as usize;
    z[m..].iter().fold(EF::ONE, |acc, &zb| acc * (EF::ONE - zb))
}

/// Per-relation public metadata: which domain, whether it carries an eq
/// point, and which tensor slots get terminal openings (in order).
fn relation_meta(id: usize) -> (bool, bool, Vec<usize>) {
    match id {
        INV_IDX_ROW => (true, true, vec![0, 4]),
        INV_IDX_ENTRY => (false, true, vec![1]),
        INV_PAIR_ROW => (true, true, vec![2, 4, 5]),
        INV_PAIR_ENTRY => (false, true, vec![3, 6]),
        SUM_IDX_ROW => (true, false, vec![0]),
        SUM_IDX_ENTRY => (false, false, vec![7, 1]),
        SUM_PAIR_ROW => (true, false, vec![2]),
        SUM_PAIR_ENTRY => (false, false, vec![7, 3]),
        _ => unreachable!(),
    }
}

/// Term marker for the public affine index buffer `J` (never materialized
/// on the prover side — see `virtual_round_pvals_affine_j`).
const J_MARK: usize = usize::MAX;

/// Which tensor slots a relation reads (buffer order), and whether its
/// virtual polynomial uses the public affine J source.
/// Slots: 0 = iq_idx, 1 = it_idx, 2 = iq_pair, 3 = it_pair, 4 = idx,
/// 5 = out, 6 = table, 7 = m.
fn relation_srcs(id: usize) -> (Vec<usize>, bool) {
    match id {
        INV_IDX_ROW => (vec![0, 4], false),
        INV_IDX_ENTRY => (vec![1], true),
        INV_PAIR_ROW => (vec![2, 4, 5], false),
        INV_PAIR_ENTRY => (vec![3, 6], true),
        SUM_IDX_ROW => (vec![0], false),
        SUM_IDX_ENTRY => (vec![7, 1], false),
        SUM_PAIR_ROW => (vec![2], false),
        SUM_PAIR_ENTRY => (vec![7, 3], false),
        _ => unreachable!(),
    }
}

/// The head mask length for a sum relation, or `None` when the head covers
/// the FULL domain (constant one — the mask buffer is omitted entirely).
fn head_len(id: usize, p: usize, t: usize, arity: usize) -> Option<usize> {
    let len = match id {
        SUM_IDX_ROW | SUM_PAIR_ROW => p,
        SUM_IDX_ENTRY | SUM_PAIR_ENTRY => t,
        _ => return None,
    };
    (len < (1usize << arity)).then_some(len)
}

/// Per-relation virtual-polynomial terms. Buffer layout: the slots from
/// `relation_srcs`, then the head mask (if `has_head`), then the eq buffer
/// (inverse relations). `J_MARK` denotes the affine J factor. The term
/// SUMS are identical to the previously materialized-J/head layout
/// (equivalence-tested), so the absorbed round polynomials are bit-identical.
fn relation_terms(
    id: usize,
    gamma: EF,
    alpha: EF,
    beta: EF,
    has_head: bool,
) -> Vec<(EF, Vec<usize>)> {
    let neg = EF::ZERO - EF::ONE;
    match id {
        INV_IDX_ROW => vec![(gamma, vec![2, 0]), (EF::ONE, vec![2, 0, 1]), (neg, vec![2])],
        // New entry-inverse layout [inv(0), eq(1)] with J affine: the J
        // factor appears ONLY in the mixed term; the coefficient terms keep
        // eq (same products as the old materialized layout).
        INV_IDX_ENTRY => {
            vec![(gamma, vec![1, 0]), (EF::ONE, vec![1, 0, J_MARK]), (neg, vec![1])]
        }
        INV_PAIR_ROW => vec![
            (alpha, vec![3, 0]),
            (EF::ONE, vec![3, 0, 1]),
            (beta, vec![3, 0, 2]),
            (neg, vec![3]),
        ],
        // New entry-inverse layout [inv(0), table(1), eq(2)] with J affine.
        INV_PAIR_ENTRY => vec![
            (alpha, vec![2, 0]),
            (EF::ONE, vec![2, 0, J_MARK]),
            (beta, vec![2, 0, 1]),
            (neg, vec![2]),
        ],
        SUM_IDX_ROW if has_head => vec![(EF::ONE, vec![1, 0])],
        SUM_IDX_ROW => vec![(EF::ONE, vec![0])],
        SUM_IDX_ENTRY if has_head => vec![(EF::ONE, vec![2, 0, 1])],
        SUM_IDX_ENTRY => vec![(EF::ONE, vec![0, 1])],
        SUM_PAIR_ROW if has_head => vec![(EF::ONE, vec![1, 0])],
        SUM_PAIR_ROW => vec![(EF::ONE, vec![0])],
        SUM_PAIR_ENTRY if has_head => vec![(EF::ONE, vec![2, 0, 1])],
        SUM_PAIR_ENTRY => vec![(EF::ONE, vec![0, 1])],
        _ => unreachable!(),
    }
}

/// Round-polynomial values at `0..=max_deg` for a virtual sumcheck whose
/// buffers EXCLUDE the public affine index buffer `J(k) = k` (LSB-first
/// flat index). Terms may use `J_MARK` as a factor: after `round` folds at
/// challenges `z_0..z_{round-1}`, the J factor's linear extension over the
/// current pair `(2s, 2s+1)` is
/// `fk(J) = j_base + s * 2^{round+1} + X * 2^round`
/// where `j_base = sum_{b<round} 2^b z_b` and `X` is the current variable
/// (the pair's original indices differ exactly in bit `round`). This is
/// BIT-IDENTICAL to materializing J as a buffer (equivalence-tested) and
/// never allocates the 2^arity J vector.
fn virtual_round_pvals_affine_j(
    bufs: &[Vec<EF>],
    terms: &[(EF, Vec<usize>)],
    max_deg: usize,
    round: usize,
    j_base: EF,
) -> Vec<EF> {
    let half = bufs[0].len() / 2;
    let mut pvals = vec![EF::ZERO; max_deg + 1];
    let two_j = EF::from(Goldilocks::from_u64((1usize << round) as u64));
    let two_j1 = EF::from(Goldilocks::from_u64((1usize << (round + 1)) as u64));
    for k in 0..=max_deg {
        let xk = EF::from(Goldilocks::from_u64(k as u64));
        let one_minus = EF::ONE - xk;
        for s in 0..half {
            let j_fk = j_base + EF::from(Goldilocks::from_u64(s as u64)) * two_j1 + xk * two_j;
            for (coeff, idxs) in terms {
                let mut prod = *coeff;
                for &j in idxs {
                    let fk = if j == J_MARK {
                        j_fk
                    } else {
                        // k*f1 - (k-1)*f0 == xk*f1 + (1-xk)*f0 (same as the
                        // generic virtual_round_pvals_f).
                        xk * bufs[j][2 * s + 1] + one_minus * bufs[j][2 * s]
                    };
                    prod = prod * fk;
                }
                pvals[k] = pvals[k] + prod;
            }
        }
    }
    pvals
}

/// Verifier-side terminal expression for relation `id` from the opened
/// values (in `relation_meta`'s slot order). Allocation-free closed forms
/// only: `j_eval(z)` for the public index buffer and `head_eval(len, z)`
/// for the head masks — the materialized `j_values`/`head` vectors are
/// never built on the verifier path.
fn terminal_expr(
    id: usize,
    p: usize,
    t: usize,
    gamma: EF,
    alpha: EF,
    beta: EF,
    r: &[EF],
    z: &[EF],
    v: &[EF],
) -> EF {
    let jz = j_eval(z);
    match id {
        INV_IDX_ROW => eq_point(r, z) * (v[0] * (gamma + v[1]) - EF::ONE),
        INV_IDX_ENTRY => eq_point(r, z) * (v[0] * (gamma + jz) - EF::ONE),
        INV_PAIR_ROW => eq_point(r, z) * (v[0] * (alpha + v[1] + beta * v[2]) - EF::ONE),
        INV_PAIR_ENTRY => eq_point(r, z) * (v[0] * (alpha + jz + beta * v[1]) - EF::ONE),
        SUM_IDX_ROW => head_eval(p, z) * v[0],
        SUM_IDX_ENTRY => head_eval(t, z) * v[0] * v[1],
        SUM_PAIR_ROW => head_eval(p, z) * v[0],
        SUM_PAIR_ENTRY => head_eval(t, z) * v[0] * v[1],
        _ => unreachable!(),
    }
}

fn prove_relation(
    lw: &LookupWhir,
    p: usize,
    t: usize,
    gamma: EF,
    alpha: EF,
    beta: EF,
    d: &[Data; 8],
    id: usize,
    tx: &mut TwoPhaseTranscript,
    scratch: &mut Vec<Vec<EF>>,
    open_secs: &mut f64,
) -> RelationProof {
    let (is_rows, has_r, open_ids) = relation_meta(id);
    let arity = if is_rows { lw.rows.num_variables() } else { lw.entries.num_variables() };
    let r = if has_r { tx.sample_vec(arity) } else { vec![] };
    let (slots, uses_j) = relation_srcs(id);
    let mask = head_len(id, p, t, arity);
    let terms = relation_terms(id, gamma, alpha, beta, mask.is_some());
    // Reusable scratch: the committed `Data.values` are only READ; the fold
    // working set lives in scratch buffers whose ALLOCATIONS are preserved
    // across relations (`clear` keeps capacity, the fold's `truncate` never
    // shrinks the allocation, and `refill_scratch` reuses the same buffer).
    // The public J source is affine (never materialized) and a full-domain
    // head mask is constant one (omitted).
    let mut used = 0usize;
    for &s in &slots {
        refill_scratch(scratch, &d[s].values, used);
        used += 1;
    }
    if let Some(len) = mask {
        let h = head(len, arity);
        refill_scratch(scratch, &h, used);
        used += 1;
    }
    if has_r {
        let eq = eq_values(&r, arity);
        refill_scratch(scratch, &eq, used);
        used += 1;
    }
    let mut rounds = Vec::with_capacity(arity);
    let mut z = Vec::with_capacity(arity);
    let mut j_base = EF::ZERO;
    for round in 0..arity {
        let pvals = if uses_j {
            virtual_round_pvals_affine_j(&scratch[..used], &terms, 3, round, j_base)
        } else {
            virtual_round_pvals_f(&scratch[..used], &terms, 3)
        };
        for &x in &pvals {
            tx.absorb(x);
        }
        let c = tx.sample();
        let half = scratch[0].len() / 2;
        for b in scratch[..used].iter_mut() {
            for i in 0..half {
                b[i] = b[2 * i] + c * (b[2 * i + 1] - b[2 * i]);
            }
            b.truncate(half);
        }
        j_base = j_base + EF::from(Goldilocks::from_u64((1usize << round) as u64)) * c;
        rounds.push(pvals);
        z.push(c);
    }
    let w = if is_rows { &lw.rows } else { &lw.entries };
    let t0 = std::time::Instant::now();
    let opens: Vec<TensorOpen> = open_ids.iter().map(|&i| open_at(w, &d[i], &z)).collect();
    *open_secs += t0.elapsed().as_secs_f64();
    RelationProof { r, rounds, z, opens }
}

fn verify_relation(
    lw: &LookupWhir,
    stmt: &LookupStatement,
    p: usize,
    t: usize,
    gamma: EF,
    alpha: EF,
    beta: EF,
    claimed0: EF,
    id: usize,
    tx: &mut TwoPhaseTranscript,
    pr: &RelationProof,
) -> bool {
    let (is_rows, has_r, open_ids) = relation_meta(id);
    let arity = if is_rows { lw.rows.num_variables() } else { lw.entries.num_variables() };
    let r = if has_r {
        let r = tx.sample_vec(arity);
        if pr.r != r {
            return false;
        }
        r
    } else {
        if !pr.r.is_empty() {
            return false;
        }
        vec![]
    };
    if pr.rounds.len() != arity || pr.z.len() != arity || pr.opens.len() != open_ids.len() {
        return false;
    }
    let mut previous = claimed0;
    for (round, &zc) in pr.rounds.iter().zip(&pr.z) {
        if round.len() != 4 || round[0] + round[1] != previous {
            return false;
        }
        for &x in round {
            tx.absorb(x);
        }
        let z = tx.sample();
        if z != zc {
            return false;
        }
        previous = interpolate_f(round, z, 3);
    }
    let w = if is_rows { &lw.rows } else { &lw.entries };
    // Map opened slots to statement roots.
    let roots: Vec<&TensorCommitment> = match id {
        _ => {
            let all = [
                &stmt.iq_idx,
                &stmt.it_idx,
                &stmt.iq_pair,
                &stmt.it_pair,
                &stmt.idx,
                &stmt.out,
                &stmt.table,
                &stmt.m,
            ];
            open_ids.iter().map(|&i| all[i]).collect()
        }
    };
    let mut v = Vec::with_capacity(open_ids.len());
    for (root, x) in roots.iter().zip(&pr.opens) {
        let Some(y) = verify_open(w, root, x, &pr.z, arity) else {
            return false;
        };
        v.push(y);
    }
    previous == terminal_expr(id, p, t, gamma, alpha, beta, &r, &pr.z, &v)
}

/// Prove with an EXPLICIT multiplicity witness (validated for shape only —
/// the histogram property is what the proof itself enforces). The
/// multiplicity tensor is committed BEFORE any transcript challenge and is
/// never accepted afterwards.
pub fn prove_with_m(
    lw: &LookupWhir,
    idx: &[u32],
    out: &[u64],
    table: &[u64],
    m: &[u64],
) -> Option<(LookupStatement, LookupProof)> {
    let p = idx.len();
    let t = table.len();
    let ar_r = lw.rows.num_variables();
    let ar_e = lw.entries.num_variables();
    if !p.is_power_of_two()
        || !t.is_power_of_two()
        || p < 2
        || t < 2
        || out.len() != p
        || m.len() != t
        || ar_r < p.trailing_zeros() as usize
        || ar_e < t.trailing_zeros() as usize
        || idx.iter().any(|&i| i as usize >= t)
    {
        return None;
    }
    let row_size = 1usize << ar_r;
    let ent_size = 1usize << ar_e;
    let to_ef = |v: u64| EF::from(Goldilocks::from_u64(v));
    let mut idx_vals = vec![EF::ZERO; row_size];
    let mut out_vals = vec![EF::ZERO; row_size];
    for i in 0..p {
        idx_vals[i] = to_ef(idx[i] as u64);
        out_vals[i] = to_ef(out[i]);
    }
    let mut table_vals = vec![EF::ZERO; ent_size];
    let mut m_vals = vec![EF::ZERO; ent_size];
    for j in 0..t {
        table_vals[j] = to_ef(table[j]);
        m_vals[j] = to_ef(m[j]);
    }
    // Initial commitments (roots BEFORE the challenges).
    let t0 = std::time::Instant::now();
    let d_idx = commit_ef_pair(&lw.rows, &idx_vals, ar_r);
    let d_out = commit_ef_pair(&lw.rows, &out_vals, ar_r);
    let d_table = commit_ef_pair(&lw.entries, &table_vals, ar_e);
    let d_m = commit_ef_pair(&lw.entries, &m_vals, ar_e);
    let initial_commit_secs = t0.elapsed().as_secs_f64();

    // Initial roots (BEFORE the challenges) — the transcript is built from
    // them directly.
    let initial_roots: Vec<&Commitment> = vec![
        &d_idx.root.re,
        &d_idx.root.im,
        &d_out.root.re,
        &d_out.root.im,
        &d_table.root.re,
        &d_table.root.im,
        &d_m.root.re,
        &d_m.root.im,
    ];
    let mut tx = TwoPhaseTranscript::new(PROTOCOL, &[p, t], &initial_roots);
    let (u, alpha) = tx.sample_derived_challenges();
    // Deterministic transcript-derived challenges: gamma = u, beta = u + alpha.
    let gamma = u;
    let beta = u + alpha;

    // Derived inverse tensors (after the challenges, before the relations).
    let mut iq_idx_vals = vec![EF::ZERO; row_size];
    let mut iq_pair_vals = vec![EF::ZERO; row_size];
    for i in 0..p {
        iq_idx_vals[i] = (gamma + idx_vals[i]).inverse();
        iq_pair_vals[i] = (alpha + idx_vals[i] + beta * out_vals[i]).inverse();
    }
    for i in p..row_size {
        iq_idx_vals[i] = (gamma + EF::ZERO).inverse();
        iq_pair_vals[i] = (alpha + EF::ZERO + beta * EF::ZERO).inverse();
    }
    let jv = j_values(ar_e);
    let mut it_idx_vals = vec![EF::ZERO; ent_size];
    let mut it_pair_vals = vec![EF::ZERO; ent_size];
    for j in 0..ent_size {
        it_idx_vals[j] = (gamma + jv[j]).inverse();
        it_pair_vals[j] = (alpha + jv[j] + beta * table_vals[j]).inverse();
    }
    let t0 = std::time::Instant::now();
    let d_iq_idx = commit_ef_pair(&lw.rows, &iq_idx_vals, ar_r);
    let d_it_idx = commit_ef_pair(&lw.entries, &it_idx_vals, ar_e);
    let d_iq_pair = commit_ef_pair(&lw.rows, &iq_pair_vals, ar_r);
    let d_it_pair = commit_ef_pair(&lw.entries, &it_pair_vals, ar_e);
    let derived_inverse_commit_secs = t0.elapsed().as_secs_f64();

    let stmt = LookupStatement {
        p,
        t,
        idx: d_idx.root.clone(),
        out: d_out.root.clone(),
        table: d_table.root.clone(),
        m: d_m.root.clone(),
        iq_idx: d_iq_idx.root.clone(),
        it_idx: d_it_idx.root.clone(),
        iq_pair: d_iq_pair.root.clone(),
        it_pair: d_it_pair.root.clone(),
    };
    tx.absorb_derived_roots(&stmt.derived());

    // Claimed sums (bound before any relation challenge).
    let s_idx_row = iq_idx_vals.iter().take(p).fold(EF::ZERO, |a, &x| a + x);
    let s_idx_entry = (0..t).fold(EF::ZERO, |a, j| a + m_vals[j] * it_idx_vals[j]);
    let s_pair_row = iq_pair_vals.iter().take(p).fold(EF::ZERO, |a, &x| a + x);
    let s_pair_entry = (0..t).fold(EF::ZERO, |a, j| a + m_vals[j] * it_pair_vals[j]);
    tx.absorb(s_idx_row);
    tx.absorb(s_idx_entry);
    tx.absorb(s_pair_row);
    tx.absorb(s_pair_entry);

    let d = [
        d_iq_idx, d_it_idx, d_iq_pair, d_it_pair, d_idx, d_out, d_table, d_m,
    ];
    let mut scratch: Vec<Vec<EF>> = Vec::new();
    let mut open_secs = 0.0f64;
    let t_rel = std::time::Instant::now();
    let mut relations = Vec::with_capacity(RELATIONS);
    for id in 0..RELATIONS {
        relations.push(prove_relation(
            lw, p, t, gamma, alpha, beta, &d, id, &mut tx, &mut scratch, &mut open_secs,
        ));
    }
    let relation_prove_secs = t_rel.elapsed().as_secs_f64() - open_secs;
    Some((
        stmt,
        LookupProof {
            s_idx_row,
            s_idx_entry,
            s_pair_row,
            s_pair_entry,
            relations,
            timings: ProveTimings {
                initial_commit_secs,
                derived_inverse_commit_secs,
                relation_prove_secs,
                terminal_open_secs: open_secs,
            },
        },
    ))
}

/// Prove with the EXACT histogram `m[j] = #{i : idx[i] = j}`.
pub fn prove(
    lw: &LookupWhir,
    idx: &[u32],
    out: &[u64],
    table: &[u64],
) -> Option<(LookupStatement, LookupProof)> {
    let t = table.len();
    if idx.len() == 0 || t == 0 || idx.iter().any(|&i| i as usize >= t) {
        return None;
    }
    let mut m = vec![0u64; t];
    for &i in idx {
        m[i as usize] += 1;
    }
    prove_with_m(lw, idx, out, table, &m)
}

/// Pre-materialize metadata check for one tensor id (caller has already
/// bounds-checked the id): the logical length must be exactly `expected`,
/// and for mmap-backed tensors the claimed `[start, start + len)` range
/// must exist inside the mapping via checked arithmetic — validated BEFORE
/// any read.
fn tensor_metadata_ok(store: &Store, id: usize, expected: usize) -> bool {
    match &store.v[id] {
        TensorData::Owned(v) => v.len() == expected,
        TensorData::Mmap { mmap, start, len } => {
            *len == expected && start.checked_add(*len).is_some_and(|end| end <= mmap.len())
        }
    }
}

/// Actual-IR adapter: recognize `Op::Lookup { idx, out, table }` and
/// materialize the three logical tensors from a REAL `Store`. Validates
/// every logical shape BEFORE reads: `P = idx.len() == out.len()` and
/// `T = table.len()` are exact powers of two (`>= 2`), every `idx[i] < T`,
/// and the output/table tensor METADATA (Owned length / mmap range with
/// checked arithmetic) is sound. The exact histogram `m` is derived only
/// AFTER the idx domain validation, then the existing root-bound `prove`
/// path runs (root-only `verify` unchanged). Materializes ONLY the three
/// logical tensors — no production-size claim.
pub fn prove_op_lookup(
    lw: &LookupWhir,
    op: &Op,
    store: &Store,
) -> Option<(LookupStatement, LookupProof)> {
    let Op::Lookup { idx, out, table } = op else {
        return None;
    };
    if *idx >= store.idx.len() || *out >= store.v.len() || *table >= store.v.len() {
        return None;
    }
    let idxs = &store.idx[*idx];
    let p = idxs.len();
    if !p.is_power_of_two() || p < 2 {
        return None;
    }
    // The table's declared length (from metadata, before any read).
    let t = match &store.v[*table] {
        TensorData::Owned(v) => v.len(),
        TensorData::Mmap { len, .. } => *len,
    };
    if !t.is_power_of_two() || t < 2 {
        return None;
    }
    // Cheap config check BEFORE any materialization or histogram work: the
    // two WHIR domains must be able to hold the logical sizes.
    if lw.rows.num_variables() < p.trailing_zeros() as usize
        || lw.entries.num_variables() < t.trailing_zeros() as usize
    {
        return None;
    }
    // Idx domain validation BEFORE the histogram.
    if idxs.iter().any(|&i| i as usize >= t) {
        return None;
    }
    // Metadata BEFORE materialize (both output and table).
    if !tensor_metadata_ok(store, *out, p) || !tensor_metadata_ok(store, *table, t) {
        return None;
    }
    let out_t = store.materialize(*out);
    let tbl_t = store.materialize(*table);
    if out_t.len() != p || tbl_t.len() != t {
        return None;
    }
    let idx_u32: Vec<u32> = idxs.clone();
    let out_u64: Vec<u64> = out_t.iter().map(|g| g.as_canonical_u64()).collect();
    let table_u64: Vec<u64> = tbl_t.iter().map(|g| g.as_canonical_u64()).collect();
    let mut m = vec![0u64; t];
    for &i in &idx_u32 {
        m[i as usize] += 1;
    }
    prove_with_m(lw, &idx_u32, &out_u64, &table_u64, &m)
}

/// Verify the generic LogUp lookup from `(lookup_whir, statement, proof)`
/// ONLY: validates dims/arity/domain, replays the transcript (roots,
/// challenges, claimed sums), checks the eight relations, and closes the
/// two rational equalities. No witness, no Store, no raw tensors, no `open`
/// calls; malformed shapes return `false` (never panic).
pub fn verify(lw: &LookupWhir, stmt: &LookupStatement, proof: &LookupProof) -> bool {
    let p = stmt.p;
    let t = stmt.t;
    let ar_r = lw.rows.num_variables();
    let ar_e = lw.entries.num_variables();
    if !p.is_power_of_two()
        || !t.is_power_of_two()
        || p < 2
        || t < 2
        || ar_r < p.trailing_zeros() as usize
        || ar_e < t.trailing_zeros() as usize
        || proof.relations.len() != RELATIONS
    {
        return false;
    }
    let mut tx = TwoPhaseTranscript::new(PROTOCOL, &[p, t], &stmt.initial());
    let (u, alpha) = tx.sample_derived_challenges();
    let gamma = u;
    let beta = u + alpha;
    tx.absorb_derived_roots(&stmt.derived());
    tx.absorb(proof.s_idx_row);
    tx.absorb(proof.s_idx_entry);
    tx.absorb(proof.s_pair_row);
    tx.absorb(proof.s_pair_entry);
    let claimed = [
        EF::ZERO,
        EF::ZERO,
        EF::ZERO,
        EF::ZERO,
        proof.s_idx_row,
        proof.s_idx_entry,
        proof.s_pair_row,
        proof.s_pair_entry,
    ];
    for (id, pr) in proof.relations.iter().enumerate() {
        if !verify_relation(lw, stmt, p, t, gamma, alpha, beta, claimed[id], id, &mut tx, pr) {
            return false;
        }
    }
    // Close the two rational equalities on the SAME committed m.
    proof.s_idx_row == proof.s_idx_entry && proof.s_pair_row == proof.s_pair_entry
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::XorShift64;

    const P: usize = 16;
    const T: usize = 32;

    fn witness(rng: &mut XorShift64) -> (Vec<u32>, Vec<u64>, Vec<u64>, Vec<u64>) {
        let idx: Vec<u32> = (0..P)
            .map(|_| (rng.field().as_canonical_u64() % T as u64) as u32)
            .collect();
        let table: Vec<u64> = (0..T).map(|_| rng.field().as_canonical_u64() % 100).collect();
        let out: Vec<u64> = idx.iter().map(|&i| table[i as usize]).collect();
        let mut m = vec![0u64; T];
        for &i in &idx {
            m[i as usize] += 1;
        }
        (idx, out, table, m)
    }

    fn fixture() -> (LookupWhir, LookupStatement, LookupProof, Vec<u32>, Vec<u64>, Vec<u64>, Vec<u64>) {
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let mut rng = XorShift64::new(0x10E);
        let (idx, out, table, m) = witness(&mut rng);
        let (s, p) = prove_with_m(&lw, &idx, &out, &table, &m).expect("honest prove");
        (lw, s, p, idx, out, table, m)
    }

    fn all_whirs(lw: &LookupWhir) -> [&Whir; 2] {
        [&lw.rows, &lw.entries]
    }

    /// Honest root-only roundtrip: after the raw witness is dropped, the
    /// verifier accepts from statement + proof only and never opens.
    #[test]
    fn honest_roundtrip_after_witness_dropped() {
        let (lw, s, p, idx, out, table, m) = fixture();
        drop(idx);
        drop(out);
        drop(table);
        drop(m);
        let before: Vec<_> = all_whirs(&lw).iter().map(|w| w.open_stats()).collect();
        assert!(verify(&lw, &s, &p));
        let after: Vec<_> = all_whirs(&lw).iter().map(|w| w.open_stats()).collect();
        assert_eq!(before, after, "verifier must never open");
    }

    /// An AUTHENTIC bad output (honestly committed, all openings valid):
    /// the pair relation's rational equality fails and the verifier rejects.
    #[test]
    fn authentic_bad_output_rejected() {
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let mut rng = XorShift64::new(0x10E);
        let (idx, mut out, table, m) = witness(&mut rng);
        out[0] = (out[0] + 1) % 100;
        let (s, p) = prove_with_m(&lw, &idx, &out, &table, &m).expect("authentic artifacts");
        assert!(!verify(&lw, &s, &p));
    }

    /// An AUTHENTIC bad multiplicity (a different committed m, with a fully
    /// valid protocol proof): the index relation's equality fails and the
    /// verifier rejects. The index/out/table tensors are unchanged.
    #[test]
    fn authentic_bad_m_rejected() {
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let mut rng = XorShift64::new(0x10E);
        let (idx, out, table, mut m) = witness(&mut rng);
        m[0] += 1;
        m[1] -= 1.min(m[1]);
        let (s, p) = prove_with_m(&lw, &idx, &out, &table, &m).expect("authentic artifacts");
        assert!(!verify(&lw, &s, &p));
    }

    /// Cross-splice: two INDIVIDUALLY valid artifacts; mixing the statement
    /// roots or the proofs across them must be rejected.
    #[test]
    fn cross_splice_rejected() {
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let mut rng_a = XorShift64::new(0x10E);
        let (idx_a, out_a, table_a, m_a) = witness(&mut rng_a);
        let (sa, pa) = prove_with_m(&lw, &idx_a, &out_a, &table_a, &m_a).expect("artifact A");
        let mut rng_b = XorShift64::new(0xB0E);
        let (idx_b, out_b, table_b, m_b) = witness(&mut rng_b);
        let (sb, pb) = prove_with_m(&lw, &idx_b, &out_b, &table_b, &m_b).expect("artifact B");
        assert!(verify(&lw, &sa, &pa));
        assert!(verify(&lw, &sb, &pb));
        // Proof A against statement B.
        assert!(!verify(&lw, &sb, &pa));
        // Mixed statement roots: A's idx/out, B's table/m (and B's derived
        // inverses) against proof A.
        let mut mixed = sa.clone();
        mixed.table = sb.table.clone();
        mixed.m = sb.m.clone();
        assert!(!verify(&lw, &mixed, &pa));
    }

    /// Malformed shapes and domains are rejected by the prover and by
    /// `LookupWhir::new` (never panic).
    #[test]
    fn malformed_shapes_rejected() {
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let mut rng = XorShift64::new(0x10E);
        let (idx, out, table, m) = witness(&mut rng);
        assert!(prove_with_m(&lw, &idx, &out, &table, &m).is_some());
        // idx out of range.
        let mut bad_idx = idx.clone();
        bad_idx[0] = T as u32;
        assert!(prove_with_m(&lw, &bad_idx, &out, &table, &m).is_none());
        // wrong lengths.
        assert!(prove_with_m(&lw, &idx[..P - 1], &out, &table, &m).is_none());
        assert!(prove_with_m(&lw, &idx, &out[..P - 1], &table, &m).is_none());
        assert!(prove_with_m(&lw, &idx, &out, &table, &m[..T - 1]).is_none());
        // non-power-of-two dims at construction.
        assert!(LookupWhir::new(12, 32, 90, 0).is_none());
        assert!(LookupWhir::new(16, 24, 90, 0).is_none());
        // Verifier rejects a tampered relation shape.
        let (lw, s, mut p, ..) = fixture();
        p.relations.pop();
        assert!(!verify(&lw, &s, &p));
        let (lw, s, mut p, ..) = fixture();
        p.s_idx_row = p.s_idx_row + EF::ONE;
        assert!(!verify(&lw, &s, &p));
    }

    /// The closed-form verifier-terminal helpers (`j_eval`, `head_eval`)
    /// match the materialized reference folds (`fold_ef` over `j_values` /
    /// `head`) across arities and power-of-two prefixes, with genuine
    /// extension coordinates — and the verifier's `terminal_expr` uses ONLY
    /// the closed forms (no `j_values`/`head` allocation on the verifier
    /// path).
    #[test]
    fn closed_form_terminals_match_materialized_reference() {
        let mut rng = XorShift64::new(0xC10);
        let rand_ef = |rng: &mut XorShift64| EF::from(rng.field()) + gen() * EF::from(rng.field());
        for arity in 5..=8 {
            let z: Vec<EF> = (0..arity).map(|_| rand_ef(&mut rng)).collect();
            // J(z): closed form vs materialized fold.
            assert_eq!(j_eval(&z), fold_ef(&j_values(arity), &z));
            // head(len, ·)(z): closed form vs materialized fold, for every
            // power-of-two prefix length including the full domain.
            for m in 0..=arity {
                let len = 1usize << m;
                assert_eq!(head_eval(len, &z), fold_ef(&head(len, arity), &z));
            }
            // Terminal spot check: the entry-side inverse terminal (the J
            // consumer) and a head-masked sum terminal agree with the
            // materialized reference computation.
            let r: Vec<EF> = (0..arity).map(|_| rand_ef(&mut rng)).collect();
            let (gamma, alpha, beta) = (rand_ef(&mut rng), rand_ef(&mut rng), rand_ef(&mut rng));
            let v = [rand_ef(&mut rng), rand_ef(&mut rng)];
            let expr = terminal_expr(INV_IDX_ENTRY, 16, 32, gamma, alpha, beta, &r, &z, &v);
            let reference = eq_point(&r, &z)
                * (v[0] * (gamma + fold_ef(&j_values(arity), &z)) - EF::ONE);
            assert_eq!(expr, reference);
            let expr = terminal_expr(SUM_IDX_ENTRY, 16, 32, gamma, alpha, beta, &r, &z, &v);
            let reference = fold_ef(&head(32, arity), &z) * v[0] * v[1];
            assert_eq!(expr, reference);
        }
    }

    /// The affine-J round polynomials are BIT-IDENTICAL to materializing J
    /// as a buffer: at every round, across arities, with the J buffer folded
    /// alongside the other buffers in the reference and `j_base` tracked
    /// from the same fold challenges.
    #[test]
    fn affine_j_pvals_match_materialized_reference() {
        let mut rng = XorShift64::new(0xA11);
        let rand_ef = |rng: &mut XorShift64| EF::from(rng.field()) + gen() * EF::from(rng.field());
        let (alpha, beta) = (rand_ef(&mut rng), rand_ef(&mut rng));
        for arity in 5..=8 {
            let n = 1usize << arity;
            // INV_PAIR_ENTRY layout: new bufs [it(0), table(1), eq(2)] with
            // J_MARK terms; old bufs [it(0), J(1), table(2), eq(3)].
            let terms_new = vec![
                (alpha, vec![2, 0]),
                (EF::ONE, vec![2, 0, J_MARK]),
                (beta, vec![2, 0, 1]),
                (EF::ZERO - EF::ONE, vec![2]),
            ];
            let terms_old = vec![
                (alpha, vec![3, 0]),
                (EF::ONE, vec![3, 0, 1]),
                (beta, vec![3, 0, 2]),
                (EF::ZERO - EF::ONE, vec![3]),
            ];
            let mut it: Vec<EF> = (0..n).map(|_| rand_ef(&mut rng)).collect();
            let mut table: Vec<EF> = (0..n).map(|_| rand_ef(&mut rng)).collect();
            let mut eq: Vec<EF> = (0..n).map(|_| rand_ef(&mut rng)).collect();
            let mut jmat = j_values(arity);
            let mut j_base = EF::ZERO;
            for round in 0..arity {
                let p_new = virtual_round_pvals_affine_j(
                    &[it.clone(), table.clone(), eq.clone()],
                    &terms_new,
                    3,
                    round,
                    j_base,
                );
                let p_old = virtual_round_pvals_f(
                    &[it.clone(), jmat.clone(), table.clone(), eq.clone()],
                    &terms_old,
                    3,
                );
                assert_eq!(p_new, p_old, "round {round} arity {arity}");
                // Fold everything (J included) at a fresh challenge.
                let zc = rand_ef(&mut rng);
                let fold_in_place = |b: &mut Vec<EF>| {
                    let half = b.len() / 2;
                    for i in 0..half {
                        b[i] = b[2 * i] + zc * (b[2 * i + 1] - b[2 * i]);
                    }
                    b.truncate(half);
                };
                fold_in_place(&mut it);
                fold_in_place(&mut table);
                fold_in_place(&mut eq);
                fold_in_place(&mut jmat);
                j_base = j_base
                    + EF::from(Goldilocks::from_u64((1usize << round) as u64)) * zc;
            }
        }
    }

    /// A FULL-domain head mask is omitted (constant one): the sum-relation
    /// round polynomials match the old materialized all-ones head buffer.
    #[test]
    fn full_domain_head_omission_matches_reference() {
        let mut rng = XorShift64::new(0xA22);
        let rand_ef = |rng: &mut XorShift64| EF::from(rng.field()) + gen() * EF::from(rng.field());
        for arity in 5..=8 {
            let n = 1usize << arity;
            let m: Vec<EF> = (0..n).map(|_| rand_ef(&mut rng)).collect();
            let it: Vec<EF> = (0..n).map(|_| rand_ef(&mut rng)).collect();
            let head_ones = vec![EF::ONE; n];
            let p_new = virtual_round_pvals_f(&[m.clone(), it.clone()], &[(EF::ONE, vec![0, 1])], 3);
            let p_old = virtual_round_pvals_f(
                &[m.clone(), it.clone(), head_ones],
                &[(EF::ONE, vec![2, 0, 1])],
                3,
            );
            assert_eq!(p_new, p_old, "arity {arity}");
        }
    }

    /// The scratch refill preserves inner allocations across relations:
    /// after the first fill, repeated refill + fold (truncate) + refill
    /// cycles never reallocate (capacity invariant — no allocator hooks
    /// needed) and the contents always match the source.
    #[test]
    fn scratch_refill_preserves_allocations() {
        let mut rng = XorShift64::new(0x5C4);
        let n = 1usize << 6;
        let src_a: Vec<EF> = (0..n).map(|_| EF::from(rng.field())).collect();
        let src_b: Vec<EF> = (0..n).map(|_| EF::from(rng.field())).collect();
        let mut scratch: Vec<Vec<EF>> = Vec::new();
        refill_scratch(&mut scratch, &src_a, 0);
        let cap = scratch[0].capacity();
        assert!(cap >= n);
        assert_eq!(&scratch[0][..], &src_a[..]);
        // Fold in place like the relation rounds, then refill from another
        // source: the allocation must survive.
        for _ in 0..6 {
            let half = scratch[0].len() / 2;
            for i in 0..half {
                scratch[0][i] = scratch[0][2 * i] + scratch[0][2 * i + 1];
            }
            scratch[0].truncate(half);
        }
        refill_scratch(&mut scratch, &src_b, 0);
        assert_eq!(scratch[0].capacity(), cap, "inner allocation reused, no realloc");
        assert_eq!(&scratch[0][..], &src_b[..]);
    }

    /// A REAL store + `Op::Lookup` fixture (P = 16, T = 32): idx tensor,
    /// table tensor, materialized output `out[i] = table[idx[i]]`.
    fn op_fixture(out_wrong: bool) -> (Store, Op) {
        let mut store = Store::new();
        let table: Vec<Goldilocks> = (0..T).map(|j| Goldilocks::from_u64((j % 100) as u64)).collect();
        let idxs: Vec<u32> = (0..P).map(|i| ((i * 7 + 3) % T as usize) as u32).collect();
        let mut out_vals: Vec<Goldilocks> = idxs.iter().map(|&i| table[i as usize]).collect();
        if out_wrong {
            out_vals[0] = out_vals[0] + Goldilocks::ONE;
        }
        let idx_id = store.push_idx(idxs);
        let tbl_id = store.push(table);
        let out_id = store.push(out_vals);
        (store, Op::Lookup { idx: idx_id, out: out_id, table: tbl_id })
    }

    /// End-to-end through the real IR: `Op::Lookup` + Store -> adapter ->
    /// Store/op dropped -> root-only verifier accepts and never opens.
    #[test]
    fn op_adapter_roundtrip_after_ir_dropped() {
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let (store, op) = op_fixture(false);
        let (s, p) = prove_op_lookup(&lw, &op, &store).expect("adapter prove");
        drop(store);
        drop(op);
        let before: Vec<_> = all_whirs(&lw).iter().map(|w| w.open_stats()).collect();
        assert!(verify(&lw, &s, &p));
        let after: Vec<_> = all_whirs(&lw).iter().map(|w| w.open_stats()).collect();
        assert_eq!(before, after, "verifier must never open");
    }

    /// An IR fixture with a WRONG output tensor still emits an artifact
    /// (the adapter is honest about the store), but the verifier rejects it
    /// via the pair relation.
    #[test]
    fn op_adapter_wrong_output_rejected() {
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let (store, op) = op_fixture(true);
        let (s, p) = prove_op_lookup(&lw, &op, &store).expect("adapter prove");
        drop(store);
        drop(op);
        assert!(!verify(&lw, &s, &p));
    }

    /// RAII temp file (unique name per construction, create_new, removed on
    /// drop) — local copy of the safe pattern.
    struct TempFile(std::path::PathBuf);
    static TMP_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    impl TempFile {
        fn write_i32(vals: &[i32]) -> Self {
            use std::io::Write as _;
            use std::sync::atomic::Ordering;
            let mut bytes = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            for _ in 0..16 {
                let n = TMP_N.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "zkie_logup_mmap_{}_{}_{}.bin",
                    std::process::id(),
                    n,
                    vals.len()
                ));
                let mut f = match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                {
                    Ok(f) => f,
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("temp file creation failed: {e}"),
                };
                f.write_all(&bytes).expect("write temp file");
                return TempFile(path);
            }
            panic!("could not create a unique temp file after 16 attempts");
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Malformed IR shapes and mmap metadata are rejected by the adapter
    /// (`None`), never a panic and never a read past the declared range:
    /// non-Lookup ops, non-power-of-two logical sizes, an out-of-range
    /// index, a declared mmap len over the backing data, and an
    /// overflowing mmap start.
    #[test]
    fn op_adapter_malformed_rejected() {
        use std::sync::Arc;
        use zkie_core::common::weights_io::WeightMmap;
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let (store, _) = op_fixture(false);
        // A non-Lookup op.
        let add = Op::Add { a: 0, b: 1, c: 2 };
        assert!(prove_op_lookup(&lw, &add, &store).is_none());
        // Non-power-of-two idx length.
        let mut store_bad = Store::new();
        let table: Vec<Goldilocks> = (0..T).map(|j| Goldilocks::from_u64((j % 100) as u64)).collect();
        let idx12 = store_bad.push_idx(vec![0; 12]);
        let tbl = store_bad.push(table);
        let out12 = store_bad.push(vec![Goldilocks::ZERO; 12]);
        let bad = Op::Lookup { idx: idx12, out: out12, table: tbl };
        assert!(prove_op_lookup(&lw, &bad, &store_bad).is_none());
        // An index >= T.
        let mut store_bad = Store::new();
        let table: Vec<Goldilocks> = (0..T).map(|j| Goldilocks::from_u64((j % 100) as u64)).collect();
        let mut idxs: Vec<u32> = (0..P).map(|i| (i % T as usize) as u32).collect();
        idxs[0] = T as u32;
        let idx_id = store_bad.push_idx(idxs);
        let tbl_id = store_bad.push(table);
        let out_id = store_bad.push(vec![Goldilocks::ZERO; P]);
        let bad = Op::Lookup { idx: idx_id, out: out_id, table: tbl_id };
        assert!(prove_op_lookup(&lw, &bad, &store_bad).is_none());

        // Mmap-backed table with a POWER-OF-TWO declared len (32) over the
        // backing data (16) -> the checked bound rejects before any read.
        let backing: Vec<i32> = (0..16).map(|j| j + 3).collect();
        let tmp = TempFile::write_i32(&backing);
        let mmap = Arc::new(WeightMmap::open(&tmp.0).expect("open mmap"));
        let mut store_m = Store::new();
        let idx_m = store_m.push_idx(vec![0; P]);
        let tbl_m = store_m.push_mmap(mmap.clone(), 0, T); // declared 32, backed by 16
        let out_m = store_m.push(vec![Goldilocks::ZERO; P]);
        let bad = Op::Lookup { idx: idx_m, out: out_m, table: tbl_m };
        assert!(prove_op_lookup(&lw, &bad, &store_m).is_none());
        // Malformed OUTPUT mmap metadata: declared P = 16 backed by 8.
        let backing8: Vec<i32> = (0..8).map(|j| j + 3).collect();
        let tmp8 = TempFile::write_i32(&backing8);
        let mmap8 = Arc::new(WeightMmap::open(&tmp8.0).expect("open mmap"));
        let mut store_m = Store::new();
        let idx_m = store_m.push_idx(vec![0; P]);
        let table_m: Vec<Goldilocks> =
            (0..T).map(|j| Goldilocks::from_u64((j % 100) as u64)).collect();
        let tbl_m = store_m.push(table_m);
        let out_m = store_m.push_mmap(mmap8, 0, P); // declared 16, backed by 8
        let bad = Op::Lookup { idx: idx_m, out: out_m, table: tbl_m };
        assert!(prove_op_lookup(&lw, &bad, &store_m).is_none());
        // Overflowing mmap start.
        let mut store_m = Store::new();
        let idx_m = store_m.push_idx(vec![0; P]);
        let tbl_m = store_m.push_mmap(mmap.clone(), usize::MAX, T);
        let out_m = store_m.push(vec![Goldilocks::ZERO; P]);
        let bad = Op::Lookup { idx: idx_m, out: out_m, table: tbl_m };
        assert!(prove_op_lookup(&lw, &bad, &store_m).is_none());
    }

    /// A VALID mmap-backed IR fixture (BOTH output and table from mmap,
    /// exact power-of-two declared lengths over exact backing data)
    /// round-trips through the adapter and verifies.
    #[test]
    fn op_adapter_mmap_roundtrip() {
        use std::sync::Arc;
        use zkie_core::common::weights_io::WeightMmap;
        let lw = LookupWhir::new(P, T, 90, 0).expect("valid dims");
        let idxs: Vec<u32> = (0..P).map(|i| ((i * 7 + 3) % T as usize) as u32).collect();
        let table_vals: Vec<i32> = (0..T).map(|j| (j % 100) as i32).collect();
        let out_vals: Vec<i32> = idxs.iter().map(|&i| table_vals[i as usize]).collect();
        let tmp_t = TempFile::write_i32(&table_vals);
        let tmp_o = TempFile::write_i32(&out_vals);
        let mmap_t = Arc::new(WeightMmap::open(&tmp_t.0).expect("open mmap"));
        let mmap_o = Arc::new(WeightMmap::open(&tmp_o.0).expect("open mmap"));
        let mut store = Store::new();
        let idx_id = store.push_idx(idxs);
        let tbl_id = store.push_mmap(mmap_t, 0, T);
        let out_id = store.push_mmap(mmap_o, 0, P);
        let op = Op::Lookup { idx: idx_id, out: out_id, table: tbl_id };
        let (s, p) = prove_op_lookup(&lw, &op, &store).expect("adapter prove");
        drop(store);
        drop(op);
        assert!(verify(&lw, &s, &p));
    }
}
