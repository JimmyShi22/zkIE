//! logUp GKR: prove sum_i num[i]/den[i] == C via the fraction-addition tree
//! with eq-weighted virtual sumchecks (DeepProve "fractional sumcheck").
//!
//! S1 challenge derivation (Fiat-Shamir, layered top-down claim propagation):
//! claims flow from the fraction-tree root (final_num/final_den) toward the
//! leaves. For each layer k (t down to 1):
//! - `gamma_k` is squeezed before the layer (binding the previous layer's
//!   messages and the incoming claims), batching the numerator and
//!   denominator identities into one eq-weighted virtual sumcheck whose claim
//!   is 0. The incoming claim points `q_num`/`q_den` (the previous layer's
//!   same_poly merged points) are known before the layer starts.
//! - The layer's challenge vector `s_k` is derived PER-ROUND from the layer's
//!   own round messages; successive layers do NOT share nested challenge
//!   prefixes.
//! - The layer's final evals at `s_k` are claims on the child tensors at
//!   `(s_k, 0)` and `(s_k, 1)`; two same_poly merges (sound
//!   claim-combination) reduce them to single claims
//!   `(q_num', a')`/`(q_den', b')` that feed the next layer.
//! - The leaf claims on the committed num_0/den_0 (at q_num_0/q_den_0) are
//!   authenticated through the existing batch-opening path (per-table
//!   points). The root binding is final_num = C*final_den (checked by the
//!   caller). S3 placeholder: zero-denominator rejection is tracked
//!   separately (S3), not implemented here.

use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use crate::common::fixed_point::from_i64;
use crate::common::same_poly::{prove_same_poly, verify_same_poly, SamePolyProof};
use crate::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use crate::common::transcript::Transcript;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FractionalLayer {
    pub gamma: Goldilocks,
    pub proof: VirtualProof,
    pub s: Vec<Goldilocks>,
    pub num_merge: SamePolyProof,
    pub den_merge: SamePolyProof,
}

/// The layered fractional proof: one γ-batched identity sumcheck per tree
/// level, plus the two same_poly merges feeding the next level's claims.
/// S3: NO root fields — the tree root (final_num, final_den) is DERIVED by
/// the verifier from the top layer's child evals (N0*D1 + N1*D0, D0*D1),
/// which the lower chain authenticates; a prover-supplied root is exactly the
/// gap that let the LogUp final ratio be steered off the reduction.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FractionalProof {
    pub layers: Vec<FractionalLayer>,
}

/// The four leaf claims (num point + eval, den point + eval) produced by the
/// final layer's same_poly merges. The VERIFIER only ever uses leaves derived
/// from its own replay (see [`verify_fractional_chain_core`]); the prover-side
/// helper [`fractional_leaf`] reads the same values back out of a freshly
/// proven chain.
pub type FractionalLeaf = (Vec<Goldilocks>, Goldilocks, Vec<Goldilocks>, Goldilocks);

/// Everything a fractional-chain replay derives (S2/S3): the leaf claims
/// (which the openings authenticate) AND the tree roots (which the LogUp
/// binding uses). Both are replay-derived — the proof stores neither.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionalChainClaims {
    pub q_num: Vec<Goldilocks>,
    pub num_leaf: Goldilocks,
    pub q_den: Vec<Goldilocks>,
    pub den_leaf: Goldilocks,
    pub final_num: Goldilocks,
    pub final_den: Goldilocks,
}

fn build_terms(gamma: Goldilocks, a: Goldilocks, b: Goldilocks) -> Vec<(Goldilocks, Vec<usize>)> {
    // G(x) = eq0(x) * [N0*D1 + N1*D0 - a] + gamma * eq1(x) * [D0*D1 - b],
    // mles = [eq0, eq1, N0, N1, D0, D1]. Claim sum G = 0 iff both tree
    // identities hold at the incoming claim points (q_num, q_den).
    let neg_a = Goldilocks::ZERO - a;
    let neg_gb = Goldilocks::ZERO - gamma * b;
    // mles = [eq0, eq1, N0, N1, D0, D1]; the tree identity is
    // num_k = N0*D1 + N1*D0 and den_k = D0*D1.
    vec![
        (Goldilocks::ONE, vec![0usize, 2, 5]),
        (Goldilocks::ONE, vec![0, 3, 4]),
        (neg_a, vec![0]),
        (gamma, vec![1, 4, 5]),
        (neg_gb, vec![1]),
    ]
}

/// Fiat-Shamir prove (S1): layered top-down claim propagation, see the module
/// doc for the exact ordering. Every layer's challenge vector is derived
/// per-round from that layer's own round messages.
pub fn prove_fractional(
    num: Vec<Goldilocks>,
    den: Vec<Goldilocks>,
    transcript: &mut Transcript,
) -> FractionalProof {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    // S3: a single-entry domain (t = 0) has no layer to derive the root
    // from; the protocol explicitly rejects it. All production columns are
    // padded to at least 2^5.
    assert!(num.len() >= 2, "fractional domain must have at least 2 entries");
    let t = num.len().trailing_zeros() as usize;

    // Build the whole fraction tree first: tn[i] = num_i (arity t-i).
    let mut tn = vec![num];
    let mut td = vec![den];
    while tn.last().unwrap().len() > 1 {
        let half = tn.last().unwrap().len() / 2;
        let (nl, nh) = tn.last().unwrap().split_at(half);
        let (dl, dh) = td.last().unwrap().split_at(half);
        let next_n: Vec<Goldilocks> = (0..half).map(|i| nl[i] * dh[i] + nh[i] * dl[i]).collect();
        let next_d: Vec<Goldilocks> = (0..half).map(|i| dl[i] * dh[i]).collect();
        tn.push(next_n);
        td.push(next_d);
    }
    let final_num = tn[t][0];
    let final_den = td[t][0];

    // S3 placeholder: den values are assumed non-zero at every claimed point;
    // the zero-denominator rejection is tracked separately (S3).
    let mut a = final_num; // root claim on num_t
    let mut b = final_den; // root claim on den_t
    let mut q_num: Vec<Goldilocks> = Vec::new(); // root point (arity 0)
    let mut q_den: Vec<Goldilocks> = Vec::new();
    let mut layers = Vec::with_capacity(t);

    // Layer k processes the claims (a, b) on (num_k, den_k) at (q_num, q_den)
    // and produces claims on (num_{k-1}, den_{k-1}) via the identity sumcheck
    // plus two same_poly merges. Layer k's sumcheck runs over arity t-k.
    for k in (1..=t).rev() {
        let gamma = transcript.field();
        let eq_num = crate::common::mle::eq_evals(&q_num);
        let eq_den = crate::common::mle::eq_evals(&q_den);
        let half = 1usize << (t - k);
        let num_k1 = &tn[k - 1];
        let den_k1 = &td[k - 1];
        let mles: Vec<&[Goldilocks]> = vec![
            &eq_num,
            &eq_den,
            &num_k1[..half],
            &num_k1[half..],
            &den_k1[..half],
            &den_k1[half..],
        ];
        let terms = build_terms(gamma, a, b);
        let (proof, s) = prove_virtual(&mles, &terms, Goldilocks::ZERO, transcript);
        let fe = &proof.final_evals;
        let mut p0 = s.clone();
        p0.push(Goldilocks::ZERO);
        let mut p1 = s.clone();
        p1.push(Goldilocks::ONE);
        let num_merge =
            prove_same_poly(num_k1, &[(p0.clone(), fe[2]), (p1.clone(), fe[3])], transcript);
        let den_merge = prove_same_poly(den_k1, &[(p0, fe[4]), (p1, fe[5])], transcript);
        a = num_merge.merged_eval;
        b = den_merge.merged_eval;
        q_num = num_merge.merged_point.clone();
        q_den = den_merge.merged_point.clone();
        layers.push(FractionalLayer {
            gamma,
            proof,
            s,
            num_merge,
            den_merge,
        });
    }
    FractionalProof { layers }
}

pub fn verify_fractional(
    proof: &FractionalProof,
    num: Vec<Goldilocks>,
    den: Vec<Goldilocks>,
    claimed: Goldilocks,
    transcript: &mut Transcript,
) -> bool {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    let t = num.len().trailing_zeros() as usize;

    // Rebuild the tree, then replay the same layered top-down chain as the
    // prover, computing every final eval from the rebuilt tensors at the
    // derived points.
    let mut tn = vec![num];
    let mut td = vec![den];
    while tn.last().unwrap().len() > 1 {
        let half = tn.last().unwrap().len() / 2;
        let (nl, nh) = tn.last().unwrap().split_at(half);
        let (dl, dh) = td.last().unwrap().split_at(half);
        let next_n: Vec<Goldilocks> = (0..half).map(|i| nl[i] * dh[i] + nh[i] * dl[i]).collect();
        let next_d: Vec<Goldilocks> = (0..half).map(|i| dl[i] * dh[i]).collect();
        tn.push(next_n);
        td.push(next_d);
    }

    // S3: the chain must span the column-derived depth exactly (layer li has
    // arity li); a zero-layer domain is explicitly invalid.
    if t == 0 || proof.layers.len() != t {
        return false;
    }

    let mut a = Goldilocks::ZERO;
    let mut b = Goldilocks::ZERO;
    let mut q_num: Vec<Goldilocks> = Vec::new();
    let mut q_den: Vec<Goldilocks> = Vec::new();
    let mut root: Option<(Goldilocks, Goldilocks)> = None;
    for (k, layer) in proof.layers.iter().enumerate() {
        let gamma = transcript.field();
        if gamma != layer.gamma {
            return false;
        }
        let arity = k;
        if layer.s.len() != arity || layer.proof.final_evals.len() != 6 {
            return false;
        }
        let half = 1usize << arity;
        let num_k1 = &tn[t - 1 - k];
        let den_k1 = &td[t - 1 - k];
        let eq_num = crate::common::mle::eq_evals(&q_num);
        let eq_den = crate::common::mle::eq_evals(&q_den);
        let fe = vec![
            crate::common::mle::eval(&eq_num, &layer.s),
            crate::common::mle::eval(&eq_den, &layer.s),
            crate::common::mle::eval(&num_k1[..half], &layer.s),
            crate::common::mle::eval(&num_k1[half..], &layer.s),
            crate::common::mle::eval(&den_k1[..half], &layer.s),
            crate::common::mle::eval(&den_k1[half..], &layer.s),
        ];
        if k == 0 {
            // S3: the tree roots are DERIVED from the (here fully rebuilt and
            // tree-authenticated) top-layer child evals.
            a = fe[2] * fe[5] + fe[3] * fe[4];
            b = fe[4] * fe[5];
            root = Some((a, b));
        }
        let terms = build_terms(gamma, a, b);
        let (ok, s, final_claim) =
            verify_virtual(&layer.proof, &terms, Goldilocks::ZERO, arity, transcript);
        if !ok || s != layer.s {
            return false;
        }
        if final_claim != virtual_eval(&terms, &fe) {
            return false;
        }
        let mut p0 = s.clone();
        p0.push(Goldilocks::ZERO);
        let mut p1 = s.clone();
        p1.push(Goldilocks::ONE);
        let num_claims = vec![(p0.clone(), fe[2]), (p1.clone(), fe[3])];
        let den_claims = vec![(p0, fe[4]), (p1, fe[5])];
        if verify_same_poly(&layer.num_merge, &num_claims, transcript).is_none() {
            return false;
        }
        if verify_same_poly(&layer.den_merge, &den_claims, transcript).is_none() {
            return false;
        }
        a = layer.num_merge.merged_eval;
        b = layer.den_merge.merged_eval;
        q_num = layer.num_merge.merged_point.clone();
        q_den = layer.den_merge.merged_point.clone();
    }
    let (final_num, final_den) = root.unwrap();
    // S3: a zero tree denominator is a protocol violation (the ratio would
    // be undefined and cross-multiplication would hold vacuously) — reject.
    if final_den == Goldilocks::ZERO {
        return false;
    }
    final_num == claimed * final_den
}

/// Store-free per-layer transcript replay (S3): squeezes and checks every
/// gamma; derives each layer's challenge vector per-round from the proof's
/// round messages (S1); REQUIRES the layer count, every point dimension and
/// the final-eval column count to match the column-derived depth `t`; checks
/// the stored child evals with the eq columns RECOMPUTED in closed form
/// (never proof-supplied); verifies the two same_poly merges; and finally
/// DERIVES the tree roots from the top layer's child evals.
///
/// Returns the derived leaf claims AND the derived roots on success; `None`
/// rejects. The proof carries no root or leaf fields.
fn replay_fractional_storefree(
    proof: &FractionalProof,
    t: usize,
    transcript: &mut Transcript,
) -> Option<FractionalChainClaims> {
    // S3 explicit rules: a zero-layer domain is invalid; the chain must span
    // the column depth exactly.
    if t == 0 || proof.layers.len() != t {
        return None;
    }
    let mut a = Goldilocks::ZERO;
    let mut b = Goldilocks::ZERO;
    let mut q_num: Vec<Goldilocks> = Vec::new();
    let mut q_den: Vec<Goldilocks> = Vec::new();
    let mut root: Option<(Goldilocks, Goldilocks)> = None;
    for (li, layer) in proof.layers.iter().enumerate() {
        let gamma = transcript.field();
        if gamma != layer.gamma {
            return None;
        }
        // Layer li (top-down) covers arity li; the round count and the
        // final-eval column count are fixed by the protocol.
        let arity = li;
        let fe = &layer.proof.final_evals;
        if layer.s.len() != arity || layer.proof.rounds.len() != arity || fe.len() != 6 {
            return None;
        }
        // The eq columns are closed-form recomputed from the derived claim
        // points, never accepted from the proof (S3).
        let eq0 = crate::common::mle::eq_poly(&q_num, &layer.s);
        let eq1 = crate::common::mle::eq_poly(&q_den, &layer.s);
        if fe[0] != eq0 || fe[1] != eq1 {
            return None;
        }
        if li == 0 {
            // S3: the tree roots come from the top layer's child evals
            // (N0*D1 + N1*D0, D0*D1) — authenticated by the lower chain.
            a = fe[2] * fe[5] + fe[3] * fe[4];
            b = fe[4] * fe[5];
            root = Some((a, b));
        }
        let terms = build_terms(gamma, a, b);
        let (ok, s, final_claim) =
            verify_virtual(&layer.proof, &terms, Goldilocks::ZERO, arity, transcript);
        if !ok || s != layer.s {
            return None;
        }
        if final_claim != virtual_eval(&terms, &[eq0, eq1, fe[2], fe[3], fe[4], fe[5]]) {
            return None;
        }
        let mut p0 = s.clone();
        p0.push(Goldilocks::ZERO);
        let mut p1 = s.clone();
        p1.push(Goldilocks::ONE);
        let num_claims = vec![(p0.clone(), fe[2]), (p1.clone(), fe[3])];
        let den_claims = vec![(p0, fe[4]), (p1, fe[5])];
        if verify_same_poly(&layer.num_merge, &num_claims, transcript).is_none() {
            return None;
        }
        if verify_same_poly(&layer.den_merge, &den_claims, transcript).is_none() {
            return None;
        }
        a = layer.num_merge.merged_eval;
        b = layer.den_merge.merged_eval;
        q_num = layer.num_merge.merged_point.clone();
        q_den = layer.den_merge.merged_point.clone();
    }
    let (final_num, final_den) = root.unwrap();
    Some(FractionalChainClaims {
        q_num,
        num_leaf: a,
        q_den,
        den_leaf: b,
        final_num,
        final_den,
    })
}

/// Chained store-free verify: the full S1 layered replay with the S3
/// structural and root-derivation rules. `t` is the column depth (log2 of
/// the padded column length), supplied by the caller from the column /
/// statement — never from the proof. Does not check the
/// final_num == claimed*final_den relation, so split lookup/table proofs can
/// share it and enforce the link separately. Returns the derived leaf AND
/// root claims on success; `None` rejects.
pub fn verify_fractional_chain_core(
    proof: &FractionalProof,
    t: usize,
    transcript: &mut Transcript,
) -> Option<FractionalChainClaims> {
    replay_fractional_storefree(proof, t, transcript)
}

pub fn verify_fractional_chain(
    proof: &FractionalProof,
    claimed: Goldilocks,
    t: usize,
    transcript: &mut Transcript,
) -> bool {
    match verify_fractional_chain_core(proof, t, transcript) {
        Some(c) => c.final_den != Goldilocks::ZERO && c.final_num == claimed * c.final_den,
        None => false,
    }
}

// Store-free verify: use the final_evals already stored in each layer
// VirtualProof, without rebuilding the num/den fraction tree.
// Leaf authentication (opening) restores soundness in P2/P3.
pub fn verify_fractional_storefree(
    proof: &FractionalProof,
    claimed: Goldilocks,
    t: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_fractional_chain(proof, claimed, t, transcript)
}

/// PROVER-SIDE / post-replay helper only: read the leaf claims out of a
/// freshly proven chain (the final layer's same_poly merge outputs). The
/// verifier must never use this on an untrusted proof — it derives the leaves
/// from its own replay via [`verify_fractional_chain_core`], which checks the
/// merges first.
pub fn fractional_leaf(
    proof: &FractionalProof,
) -> (Vec<Goldilocks>, Goldilocks, Vec<Goldilocks>, Goldilocks) {
    match proof.layers.last() {
        Some(layer) => (
            layer.num_merge.merged_point.clone(),
            layer.num_merge.merged_eval,
            layer.den_merge.merged_point.clone(),
            layer.den_merge.merged_eval,
        ),
        // Unreachable for valid proofs (S3: the domain has >= 2 entries).
        None => (Vec::new(), Goldilocks::ZERO, Vec::new(), Goldilocks::ZERO),
    }
}

/// Verify the split lookup: each side is chained-GKR sound over its own
/// column depth (`t_witness` / `t_table`, statement/column-derived), the two
/// sides agree on the fractional sum (cross-multiplied LogUp identity), and
/// neither tree denominator may be zero (S3: 0/0 must not hold vacuously).
/// The witness side is replayed first, then the table side, matching the
/// prover's order.

pub fn prove_lookup_table(
    m: &[u64],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> FractionalProof {
    let (num, den) = build_lookup_table_num_den(m, table, alpha, beta);
    prove_fractional(num, den, transcript)
}

pub fn verify_lookup_split(
    witness: &FractionalProof,
    table: &FractionalProof,
    t_witness: usize,
    t_table: usize,
    transcript: &mut Transcript,
) -> bool {
    let wc = match verify_fractional_chain_core(witness, t_witness, transcript) {
        Some(c) => c,
        None => return false,
    };
    if wc.final_den == Goldilocks::ZERO {
        return false;
    }
    let tc = match verify_fractional_chain_core(table, t_table, transcript) {
        Some(c) => c,
        None => return false,
    };
    if tc.final_den == Goldilocks::ZERO {
        return false;
    }
    wc.final_num * tc.final_den == tc.final_num * wc.final_den
}

/// Build the combined lookup num/den columns: witness entries 1/(alpha + x
/// + beta*y) in the first `n` slots, table multiplicities
/// -m/(alpha + j + beta*table[j]) in the following `t` slots, padded to the
/// next power of two.
pub fn build_lookup_num_den(
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let n = x.len();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    for i in 0..n {
        num[i] = Goldilocks::from_u64(1);
        den[i] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
    }
    let mut m = vec![0u64; t];
    for &i in x {
        m[i as usize] += 1;
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    (num, den)
}


// Split lookup into a witness side (x/y) and a table side (table/multiplicity),
// matching DeepProve: the table side is committed once globally (small PoW),
// while the witness side is committed per lookup. The two sides are linked by
// the LogUp identity: sum_i 1/(alpha+x_i+beta*y_i) == sum_j m_j/(alpha+j+beta*table_j).
pub fn lookup_multiplicity(x: &[u32], t: usize) -> Vec<u64> {
    let mut m = vec![0u64; t];
    for &i in x {
        m[i as usize] += 1;
    }
    m
}

// Merge multiplicities from several lookups into one table-side multiplicity
// column, for committing a table + global multiplicity once (DeepProve table side).
pub fn merge_multiplicity(xs: &[&[u32]], t: usize) -> Vec<u64> {
    let mut m = vec![0u64; t];
    for x in xs {
        for &i in *x {
            m[i as usize] += 1;
        }
    }
    m
}

pub fn build_lookup_witness_num_den(
    x: &[u32],
    y: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let n = x.len();
    let total = n.next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    for i in 0..n {
        num[i] = Goldilocks::from_u64(1);
        den[i] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
    }
    (num, den)
}

pub fn pad_table(table: &[Goldilocks]) -> Vec<Goldilocks> {
    let total = table.len().next_power_of_two();
    let mut out = vec![Goldilocks::ZERO; total];
    out[..table.len()].copy_from_slice(table);
    out
}

pub fn build_lookup_table_num_den(
    m: &[u64],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let t = table.len();
    let total = t.next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(0); total];
    for j in 0..total {
        num[j] = if j < t { Goldilocks::from_u64(m[j]) } else { Goldilocks::ZERO };
        let tj = if j < t { table[j] } else { Goldilocks::ZERO };
        den[j] = alpha + Goldilocks::from_u64(j as u64) + beta * tj;
    }
    (num, den)
}

/// Fiat-Shamir prove of the witness-side fractional sumcheck. `alpha`/`beta`
/// are squeezed by the caller before this call.
pub fn prove_lookup_witness(
    x: &[u32],
    y: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> FractionalProof {
    let (num, den) = build_lookup_witness_num_den(x, y, alpha, beta);
    prove_fractional(num, den, transcript)
}

// Batch several lookups (same table) into one witness-side fractional sumcheck.
// The witness side is committed once per batch; the table side is committed once
// globally. Linked together this is the DeepProve-style split LogUp.
pub fn prove_lookup_witness_batch(
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> FractionalProof {
    assert_eq!(xs.len(), ys.len());
    let n: usize = xs.iter().map(|x| x.len()).sum();
    let total = n.next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    let mut pos = 0usize;
    for (x, y) in xs.iter().zip(ys.iter()) {
        for i in 0..x.len() {
            num[pos] = Goldilocks::from_u64(1);
            den[pos] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
            pos += 1;
        }
    }
    prove_fractional(num, den, transcript)
}

pub fn prove_lookup_witness_batch_columns(
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> (FractionalProof, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
    assert_eq!(xs.len(), ys.len());
    let n: usize = xs.iter().map(|x| x.len()).sum();
    // WHIR folding factor 5 needs arity >= 5; pad tiny witnesses to 2^5.
    let total = n.next_power_of_two().max(32);
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut x_flat = vec![Goldilocks::from_u64(0); total];
    let mut y_flat = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(0); total];
    let mut pos = 0usize;
    for (x, y) in xs.iter().zip(ys.iter()) {
        for i in 0..x.len() {
            num[pos] = Goldilocks::from_u64(1);
            x_flat[pos] = Goldilocks::from_u64(x[i] as u64);
            y_flat[pos] = y[i];
            pos += 1;
        }
    }
    for j in 0..total {
        den[j] = alpha + x_flat[j] + beta * y_flat[j];
    }
    let proof = prove_fractional(num.clone(), den.clone(), transcript);
    (proof, x_flat, y_flat, num, den)
}

pub fn prove_lookup_fractional(
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> FractionalProof {
    assert_eq!(x.len(), y.len());
    let (num, den) = build_lookup_num_den(x, y, table, alpha, beta);
    prove_fractional(num, den, transcript)
}

// Prove a lookup and also return the merged num_0/den_0 columns, so the
// committed path can batch-commit them and open at the leaf point.
pub fn prove_lookup_fractional_columns(
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> (FractionalProof, Vec<Goldilocks>, Vec<Goldilocks>) {
    assert_eq!(x.len(), y.len());
    let (num, den) = build_lookup_num_den(x, y, table, alpha, beta);
    let proof = prove_fractional(num.clone(), den.clone(), transcript);
    (proof, num, den)
}

/// Fiat-Shamir verify (store-based): rebuilds num/den and replays the
/// fractional transcript. `alpha`/`beta` are replayed by the caller before.
pub fn verify_lookup_fractional(
    proof: &FractionalProof,
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> bool {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    for i in 0..n {
        num[i] = Goldilocks::from_u64(1);
        den[i] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
    }
    let mut m = vec![0u64; t];
    for &i in x {
        m[i as usize] += 1;
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    verify_fractional(proof, num, den, Goldilocks::from_u64(0), transcript)
}

// Chained store-free verify for a lookup: no x/y/table rebuild, no full
// witness columns. Soundness is restored by opening num_0/den_0 at the leaf claim points
// (see fractional_leaf) against their commitments in the committed path.
// `t` is the column depth (S3), derived by the caller from the statement
// column sizes.
pub fn verify_lookup_fractional_chain(
    proof: &FractionalProof,
    t: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_fractional_chain(proof, Goldilocks::from_u64(0), t, transcript)
}

/// Batch several lookups into the SAME `table` into one fractional-sumcheck
/// proof. All sample lists share one random challenge `(alpha, beta)`, so the
/// sumcheck count drops from `xs.len()` to one and the table multiplicity is
/// built only once.
pub fn prove_lookup_fractional_batch(
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> FractionalProof {
    assert_eq!(xs.len(), ys.len());
    let n: usize = xs.iter().map(|x| x.len()).sum();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    let mut pos = 0usize;
    for (x, y) in xs.iter().zip(ys.iter()) {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            num[pos] = Goldilocks::from_u64(1);
            den[pos] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
            pos += 1;
        }
    }
    let mut m = vec![0u64; t];
    for x in xs {
        for &i in *x {
            m[i as usize] += 1;
        }
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    prove_fractional(num, den, transcript)
}

pub fn verify_lookup_fractional_batch(
    proof: &FractionalProof,
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    transcript: &mut Transcript,
) -> bool {
    assert_eq!(xs.len(), ys.len());
    let n: usize = xs.iter().map(|x| x.len()).sum();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    let mut pos = 0usize;
    for (x, y) in xs.iter().zip(ys.iter()) {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            num[pos] = Goldilocks::from_u64(1);
            den[pos] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
            pos += 1;
        }
    }
    let mut m = vec![0u64; t];
    for x in xs {
        for &i in *x {
            m[i as usize] += 1;
        }
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    verify_fractional(proof, num, den, Goldilocks::from_u64(0), transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh transcript pair for roundtrip tests: prover and verifier start
    /// from the same statement-absorbed state.
    fn replay_pair(seed: u64) -> (Transcript, Transcript) {
        let mut tp = Transcript::new(b"zkie-logup-test");
        tp.absorb_u64(seed);
        let tv = tp.clone();
        (tp, tv)
    }

    #[test]
    fn fractional_transcript_roundtrip_and_binding() {
        let mut rng = XorShift64::new(0xDEC0 + 100);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });

        let (mut tp, tv) = replay_pair(0xF5);
        let proof = prove_fractional(num.clone(), den.clone(), &mut tp);
        let mut tv = tv;
        assert!(verify_fractional(&proof, num.clone(), den.clone(), claimed, &mut tv));
        let (mut _tp2, tv2) = replay_pair(0xF5);
        let mut tv2 = tv2;
        assert!(verify_fractional_chain(&proof, claimed, 6, &mut tv2));
        let (mut _tp3, tv3) = replay_pair(0xF5);
        let mut tv3 = tv3;
        assert!(verify_fractional_storefree(&proof, claimed, 6, &mut tv3));

        // Tamper the first layer's first round message: the store-free replay
        // derives different challenges and must reject.
        let mut tampered = FractionalProof { layers: Vec::new() };
        for (i, layer) in proof.layers.iter().enumerate() {
            let mut lp = FractionalLayer {
                gamma: layer.gamma,
                s: layer.s.clone(),
                num_merge: layer.num_merge.clone(),
                den_merge: layer.den_merge.clone(),
                proof: layer.proof.clone(),
            };
            if i == 1 {
                // layer 1 is just below the root and has a non-empty sumcheck
                lp.proof.rounds[0][0] = lp.proof.rounds[0][0] + Goldilocks::from_u64(1);
            }
            tampered.layers.push(lp);
        }
        let (mut _tp4, tv4) = replay_pair(0xF5);
        let mut tv4 = tv4;
        assert!(!verify_fractional_chain(&tampered, claimed, 6, &mut tv4));
    }

    #[test]
    fn fractional_roundtrip_storefree() {
        let mut rng = XorShift64::new(0xDEC1);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });
        let (mut tp, tv) = replay_pair(0xF5F);
        let mut proof = prove_fractional(num.clone(), den.clone(), &mut tp);
        let mut tv = tv;
        assert!(verify_fractional_storefree(&proof, claimed, 6, &mut tv));
        // Tamper a child eval of the root layer (N0): the root identity
        // check must reject. (Tampering fe[0] alone is invisible: it scales
        // a bracket that is already zero.)
        proof.layers[0].proof.final_evals[2] += Goldilocks::from_u64(1);
        let (mut _tp2, tv2) = replay_pair(0xF5F);
        let mut tv2 = tv2;
        assert!(!verify_fractional_storefree(&proof, claimed, 6, &mut tv2));
    }

    #[test]
    fn fractional_roundtrip_chain() {
        let mut rng = XorShift64::new(0xDEC2);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });
        let (mut tp, tv) = replay_pair(0xCA17);
        let mut proof = prove_fractional(num.clone(), den.clone(), &mut tp);
        let mut tv = tv;
        assert!(verify_fractional_chain(&proof, claimed, 6, &mut tv));
        proof.layers[0].proof.final_evals[3] += Goldilocks::from_u64(1);
        let (mut _tp2, tv2) = replay_pair(0xCA17);
        let mut tv2 = tv2;
        assert!(!verify_fractional_chain(&proof, claimed, 6, &mut tv2));
    }

    #[test]
    fn lookup_fractional_roundtrip() {
        let mut rng = XorShift64::new(0x10AD);
        let n = 64usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let (mut tp, tv) = replay_pair(0x10A0);
        let alpha = tp.field();
        let beta = tp.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut tp);
        let mut tv = tv;
        assert_eq!(tv.field(), alpha);
        assert_eq!(tv.field(), beta);
        assert!(verify_lookup_fractional(
            &proof, &x, &y, &table, alpha, beta, &mut tv
        ));
        let mut bad_y = y.clone();
        bad_y[0] = bad_y[0] + Goldilocks::from_u64(1);
        let (mut tp2, tv2) = replay_pair(0x10A0);
        let alpha2 = tp2.field();
        let beta2 = tp2.field();
        let proof_bad = prove_lookup_fractional(&x, &bad_y, &table, alpha2, beta2, &mut tp2);
        let mut tv2 = tv2;
        assert_eq!(tv2.field(), alpha2);
        assert_eq!(tv2.field(), beta2);
        assert!(!verify_lookup_fractional(
            &proof_bad, &x, &bad_y, &table, alpha2, beta2, &mut tv2
        ));
    }

    #[test]
    fn fractional_leaf_matches_direct_eval() {
        let mut rng = XorShift64::new(0x1EAF);
        let n = 64usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let (mut tp, _tv) = replay_pair(0x1EA0);
        let alpha = tp.field();
        let beta = tp.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut tp);
        let total = (n + t).next_power_of_two();
        let mut num = vec![Goldilocks::from_u64(0); total];
        let mut den = vec![Goldilocks::from_u64(1); total];
        for i in 0..n {
            num[i] = Goldilocks::from_u64(1);
            den[i] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
        }
        let mut m = vec![0u64; t];
        for &i in &x {
            m[i as usize] += 1;
        }
        for j in 0..t {
            num[n + j] = from_i64(-(m[j] as i64));
            den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
        }
        let (q_num, num_leaf, q_den, den_leaf) = fractional_leaf(&proof);
        assert_eq!(num_leaf, crate::common::mle::eval(&num, &q_num));
        assert_eq!(den_leaf, crate::common::mle::eval(&den, &q_den));
    }

    #[test]
    fn lookup_columns_commit_open_matches_leaf() {
        use crate::pcs::committed::commit;
        use crate::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x0BAD);
        let n = 32usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let (mut tp, _tv) = replay_pair(0x0BA0);
        let alpha = tp.field();
        let beta = tp.field();
        let (proof, num, den) =
            prove_lookup_fractional_columns(&x, &y, &table, alpha, beta, &mut tp);
        let (q_num, num_leaf, q_den, den_leaf) = fractional_leaf(&proof);
        let arity = num.len().trailing_zeros() as usize;
        let whir = Whir::new_testing(arity);
        let c_num = commit(&whir, &num);
        let (open_num, ev_num) = whir.open(c_num.prover_data.clone(), &c_num.protocol, &q_num);
        assert_eq!(
            whir.verify(&c_num.commitment, &open_num, &c_num.protocol, &q_num)
                .unwrap(),
            ev_num
        );
        assert_eq!(ev_num, num_leaf);
        let c_den = commit(&whir, &den);
        let (open_den, ev_den) = whir.open(c_den.prover_data.clone(), &c_den.protocol, &q_den);
        assert_eq!(
            whir.verify(&c_den.commitment, &open_den, &c_den.protocol, &q_den)
                .unwrap(),
            ev_den
        );
        assert_eq!(ev_den, den_leaf);
    }

    #[test]
    fn lookup_fractional_chain_roundtrip() {
        let mut rng = XorShift64::new(0x11AD);
        let n = 64usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let (mut tp, tv) = replay_pair(0x11A0);
        let alpha = tp.field();
        let beta = tp.field();
        let mut proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut tp);
        let mut tv = tv;
        assert_eq!(tv.field(), alpha);
        assert_eq!(tv.field(), beta);
        assert!(verify_lookup_fractional_chain(&proof, 7, &mut tv));
        proof.layers[0].proof.final_evals[3] += Goldilocks::from_u64(1);
        let (mut _tp2, tv2) = replay_pair(0x11A0);
        let mut tv2 = tv2;
        assert_eq!(tv2.field(), alpha);
        assert_eq!(tv2.field(), beta);
        assert!(!verify_lookup_fractional_chain(&proof, 7, &mut tv2));
    }

    #[test]
    fn lookup_split_matches_merged() {
        let mut rng = XorShift64::new(0x5E17);
        let n = 64usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let (mut tp, tv) = replay_pair(0x5E10);
        let alpha = tp.field();
        let beta = tp.field();
        let merged = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut tp);
        let mut tv = tv;
        assert_eq!(tv.field(), alpha);
        assert_eq!(tv.field(), beta);
        assert!(verify_lookup_fractional(&merged, &x, &y, &table, alpha, beta, &mut tv));
        let (mut tp2, tv2) = replay_pair(0x5E10);
        let alpha2 = tp2.field();
        let beta2 = tp2.field();
        let m = lookup_multiplicity(&x, t);
        let witness = prove_lookup_witness(&x, &y, alpha2, beta2, &mut tp2);
        let table_proof = prove_lookup_table(&m, &table, alpha2, beta2, &mut tp2);
        let mut tv2 = tv2;
        assert_eq!(tv2.field(), alpha2);
        assert_eq!(tv2.field(), beta2);
        assert!(verify_lookup_split(&witness, &table_proof, 6, 4, &mut tv2));
    }

    #[test]
    fn split_batch_matches_merged_batch() {
        let mut rng = XorShift64::new(0xB47C4);
        let t = 64usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let xs: Vec<Vec<u32>> = (0..4)
            .map(|_| (0..16).map(|_| (rng.next_u64() % t as u64) as u32).collect())
            .collect();
        let ys: Vec<Vec<Goldilocks>> = xs
            .iter()
            .map(|x| x.iter().map(|&i| table[i as usize]).collect())
            .collect();
        let (mut tp, tv) = replay_pair(0xB47C);
        let alpha = tp.field();
        let beta = tp.field();
        let xrefs: Vec<&[u32]> = xs.iter().map(|v| v.as_slice()).collect();
        let yrefs: Vec<&[Goldilocks]> = ys.iter().map(|v| v.as_slice()).collect();
        let merged = prove_lookup_fractional_batch(&xrefs, &yrefs, &table, alpha, beta, &mut tp);
        let mut tv = tv;
        assert_eq!(tv.field(), alpha);
        assert_eq!(tv.field(), beta);
        assert!(verify_lookup_fractional_batch(
            &merged, &xrefs, &yrefs, &table, alpha, beta, &mut tv
        ));
        let (mut tp2, tv2) = replay_pair(0xB47C);
        let alpha2 = tp2.field();
        let beta2 = tp2.field();
        let m = merge_multiplicity(&xrefs, t);
        let witness = prove_lookup_witness_batch(&xrefs, &yrefs, alpha2, beta2, &mut tp2);
        let table_proof = prove_lookup_table(&m, &table, alpha2, beta2, &mut tp2);
        let mut tv2 = tv2;
        assert_eq!(tv2.field(), alpha2);
        assert_eq!(tv2.field(), beta2);
        assert!(verify_lookup_split(&witness, &table_proof, 6, 6, &mut tv2));
    }

    #[test]
    fn merge_multiplicity_accumulates() {
        let t = 8usize;
        let xs: Vec<Vec<u32>> = vec![vec![0, 1, 0], vec![1, 0, 2], vec![0]];
        let refs: Vec<&[u32]> = xs.iter().map(|v| v.as_slice()).collect();
        let m = merge_multiplicity(&refs, t);
        assert_eq!(m[0], 4);
        assert_eq!(m[1], 2);
        assert_eq!(m[2], 1);
        for j in 3..t {
            assert_eq!(m[j], 0);
        }
    }

    #[test]
    fn lookup_table_commit_open_matches_leaf() {
        use crate::pcs::committed::commit;
        use crate::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x7AB1E);
        let t = 64usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let m: Vec<u64> = (0..t).map(|_| rng.next_u64() % 5).collect();
        let (mut tp, _tv) = replay_pair(0x7AB1);
        let alpha = tp.field();
        let beta = tp.field();
        let table_proof = prove_lookup_table(&m, &table, alpha, beta, &mut tp);
        let (q_num, num_leaf, q_den, den_leaf) = fractional_leaf(&table_proof);
        let total = t.next_power_of_two();
        let arity = total.trailing_zeros() as usize;
        let mut m_pad = vec![Goldilocks::from_u64(0); total];
        let mut table_pad = vec![Goldilocks::from_u64(0); total];
        for j in 0..t {
            m_pad[j] = Goldilocks::from_u64(m[j]);
            table_pad[j] = table[j];
        }
        let whir = Whir::new_testing(arity);
        let c_m = commit(&whir, &m_pad);
        let (open_m, ev_m) = whir.open(c_m.prover_data.clone(), &c_m.protocol, &q_num);
        assert_eq!(
            whir.verify(&c_m.commitment, &open_m, &c_m.protocol, &q_num)
                .unwrap(),
            ev_m
        );
        assert_eq!(ev_m, num_leaf);
        let c_t = commit(&whir, &table_pad);
        let (open_t, ev_t) = whir.open(c_t.prover_data.clone(), &c_t.protocol, &q_den);
        assert_eq!(
            whir.verify(&c_t.commitment, &open_t, &c_t.protocol, &q_den)
                .unwrap(),
            ev_t
        );
        let j_mle: Vec<Goldilocks> = (0..total).map(|j| Goldilocks::from_u64(j as u64)).collect();
        let expected_den = alpha + crate::common::mle::eval(&j_mle, &q_den) + beta * ev_t;
        assert_eq!(den_leaf, expected_den);
    }

    #[test]
    fn committed_lookup_roundtrip() {
        use crate::pcs::committed::commit;
        use crate::pcs::whir::Whir;
        let mut rng = XorShift64::new(0xEE33);
        let n = 32usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let (mut tp, tv) = replay_pair(0xEE30);
        let alpha = tp.field();
        let beta = tp.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut tp);
        let mut tv = tv;
        assert_eq!(tv.field(), alpha);
        assert_eq!(tv.field(), beta);
        assert!(verify_lookup_fractional(&proof, &x, &y, &table, alpha, beta, &mut tv));
        let whir = Whir::new_testing(n.trailing_zeros() as usize);
        let c_y = commit(&whir, &y);
        let r: Vec<Goldilocks> = (0..n.trailing_zeros() as usize)
            .map(|_| rng.field())
            .collect();
        let (open, ev) = whir.open(c_y.prover_data.clone(), &c_y.protocol, &r);
        assert_eq!(
            whir.verify(&c_y.commitment, &open, &c_y.protocol, &r)
                .unwrap(),
            ev
        );
        assert_eq!(ev, crate::common::mle::eval(&y, &r));
    }

    /// S1 regression (PR #45 review): inner fractional layers must not be
    /// steerable with advance-known challenges. The forged layer below steers
    /// its rounds assuming the honest challenge vector (the pre-fix attack);
    /// with the layered design the verifier derives the layer's challenges
    /// from the forged rounds themselves, so the stored s mismatches and the
    /// proof is REJECTED. (Pre-fix, with the nested-prefix design, the
    /// analogous forgery was ACCEPTED.)
    #[test]
    fn s1_inner_fractional_layer_steer_rejected() {
        let mut rng = XorShift64::new(0x51_51_51);
        let n = 1usize << 6; // depth 6: several layers with non-empty sumchecks
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });

        let (mut tp, tv) = replay_pair(0x1A1A);
        let honest = prove_fractional(num.clone(), den.clone(), &mut tp);
        let mut tv = tv;
        assert!(verify_fractional_chain(&honest, claimed, 6, &mut tv));

        // Pick an inner layer with a non-empty sumcheck: layers are pushed
        // root-first, so layers[1] is the layer just below the root.
        let layer_idx = 1;
        let l = &honest.layers[layer_idx];
        assert!(!l.s.is_empty(), "target layer must have a non-empty sumcheck");
        let arity = l.s.len();
        assert_eq!(l.proof.rounds.len(), arity);

        // The attacker's assumed challenge vector (the honest one) and a
        // steered final target.
        let r_assumed: Vec<Goldilocks> = l.s.clone();
        let mut honest_final = Goldilocks::ZERO;
        for (rp, &r) in l.proof.rounds.iter().zip(r_assumed.iter()) {
            assert_eq!(rp[0] + rp[1], honest_final);
            honest_final = lagrange4(rp, r);
        }
        let d_target = honest_final + Goldilocks::from_u64(1);

        // Steer the rounds under the assumed (advance-known) challenges:
        // keep the chain at prev = 0, drive the last round to d_target.
        let mut forged_rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(arity);
        let mut cur = Goldilocks::ZERO;
        for j in 0..arity {
            let target = if j + 1 < arity { cur } else { d_target };
            let p = steer4(cur, target, r_assumed[j]);
            cur = lagrange4(&p, r_assumed[j]);
            forged_rounds.push(p);
        }
        assert_eq!(cur, d_target);

        // Build the forged layer: honest gamma/s/merges, steered rounds and
        // untouched evals (the attack relies on the verifier accepting the
        // stored evals at the attacker-chosen rounds).
        let mut forged_layers = Vec::with_capacity(honest.layers.len());
        for (i, layer) in honest.layers.iter().enumerate() {
            if i == layer_idx {
                forged_layers.push(FractionalLayer {
                    gamma: l.gamma,
                    s: l.s.clone(),
                    num_merge: l.num_merge.clone(),
                    den_merge: l.den_merge.clone(),
                    proof: VirtualProof {
                        rounds: forged_rounds.clone(),
                        final_evals: l.proof.final_evals.clone(),
                    },
                });
            } else {
                forged_layers.push(FractionalLayer {
                    gamma: layer.gamma,
                    s: layer.s.clone(),
                    num_merge: layer.num_merge.clone(),
                    den_merge: layer.den_merge.clone(),
                    proof: layer.proof.clone(),
                });
            }
        }
        let forged = FractionalProof {
            layers: forged_layers,
        };

        // S1: the verifier derives the layer's challenges from the forged
        // rounds; the stored s no longer matches and the proof is REJECTED.
        let (mut _tp2, tv2) = replay_pair(0x1A1A);
        let mut tv2 = tv2;
        assert!(!verify_fractional_chain(&forged, claimed, 6, &mut tv2));
    }

    /// S1: changing an inner layer's round message changes that layer's
    /// derived challenge vector (and the proof is rejected).
    #[test]
    fn s1_inner_fractional_round_binds_challenge() {
        let mut rng = XorShift64::new(0x52_52_52);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });

        let (mut tp, tv) = replay_pair(0x1B1B);
        let proof = prove_fractional(num.clone(), den.clone(), &mut tp);
        let mut tv = tv;
        assert!(verify_fractional_chain(&proof, claimed, 6, &mut tv));

        // Tamper an inner layer's first round message: its stored s no longer
        // matches the derived vector, so verification fails. Earlier layers
        // are unaffected (their stored s's still match).
        let layer_idx = 1;
        assert!(!proof.layers[layer_idx].s.is_empty());
        let mut tampered = FractionalProof { layers: Vec::new() };
        for (i, layer) in proof.layers.iter().enumerate() {
            let mut lp = FractionalLayer {
                gamma: layer.gamma,
                s: layer.s.clone(),
                num_merge: layer.num_merge.clone(),
                den_merge: layer.den_merge.clone(),
                proof: layer.proof.clone(),
            };
            if i == layer_idx {
                lp.proof.rounds[0][0] = lp.proof.rounds[0][0] + Goldilocks::from_u64(1);
            }
            tampered.layers.push(lp);
        }
        let (mut _tp2, tv2) = replay_pair(0x1B1B);
        let mut tv2 = tv2;
        assert!(!verify_fractional_chain(&tampered, claimed, 6, &mut tv2));
    }

    // ------------------------------------------------------------------
    // S3 (LogUp top binding) test-first regressions. The forgery cases are
    // expected to be ACCEPTED pre-fix (tests fail) and REJECTED post-fix.
    // ------------------------------------------------------------------

    /// Honest split pair helper: prove a witness/table fraction pair on one
    /// transcript and return them, plus the WITNESS-side claimed ratio
    /// (sum num_w/den_w), used by the S3 forgery tests.
    fn honest_split_pair(
        seed: u64,
    ) -> (FractionalProof, FractionalProof, Transcript, Goldilocks) {
        let mut rng = XorShift64::new(seed);
        let n = 1usize << 6;
        // Identical sides: the split cross relation (equal fraction ratios)
        // holds honestly, like a real lookup pair whose multiplicities agree.
        let num_w: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den_w: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let num_t: Vec<Goldilocks> = num_w.clone();
        let den_t: Vec<Goldilocks> = den_w.clone();
        let claimed = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num_w[i] * den_w[i].inverse()
        });
        let mut start = Transcript::new(b"zkie-logup-test");
        start.absorb_u64(seed);
        let mut tp = start.clone();
        let w = prove_fractional(num_w, den_w, &mut tp);
        let t = prove_fractional(num_t, den_t, &mut tp);
        // Return the PRE-prove transcript state: verifiers replay from the
        // statement state, never from the post-prove transcript.
        (w, t, start, claimed)
    }

    /// S3: honest multi-layer chain + store verification still pass.
    #[test]
    fn s3_honest_multi_layer_roundtrip() {
        let (w, t, tp, claimed) = honest_split_pair(0x5301);
        let mut tv = tp.clone();
        assert!(verify_lookup_split(&w, &t, 6, 6, &mut tv));
        let mut tv2 = tp.clone();
        assert!(verify_fractional_chain(&w, claimed, 6, &mut tv2));
        // Store version on the witness side (rebuilds the tree).
        let mut rng = XorShift64::new(0x5301);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed_w: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });
        let (mut tp3, mut tv3) = replay_pair(0x5301);
        let w2 = prove_fractional(num.clone(), den.clone(), &mut tp3);
        assert!(verify_fractional(&w2, num, den, claimed_w, &mut tv3));
    }

    /// S3: honest single-leaf domain (t=1, two entries) still passes.
    #[test]
    fn s3_honest_single_leaf_roundtrip() {
        let mut rng = XorShift64::new(0x5302);
        let num: Vec<Goldilocks> = vec![rng.field(), rng.field()];
        let den: Vec<Goldilocks> = vec![
            from_i64(3),
            from_i64(5),
        ];
        let claimed = num[0] * den[0].inverse() + num[1] * den[1].inverse();
        let (mut tp, tv) = replay_pair(0x5302);
        let proof = prove_fractional(num.clone(), den.clone(), &mut tp);
        let mut tv = tv;
        assert!(verify_fractional_chain(&proof, claimed, 1, &mut tv));
        let (mut tp2, mut tv2) = replay_pair(0x5302);
        let proof2 = prove_fractional(num.clone(), den.clone(), &mut tp2);
        assert!(verify_fractional(&proof2, num, den, claimed, &mut tv2));
    }

    /// S3: a proof with zero layers must be rejected. (Pre-fix the empty
    /// chain returned the stored root fields as the leaves and was ACCEPTED;
    /// the split-level check only rejected it incidentally via the transcript
    /// shift.) Post-fix the layer count must equal the column depth.
    #[test]
    fn s3_zero_layer_rejected() {
        let (_w, _t, tp, claimed) = honest_split_pair(0x5303);
        let empty = FractionalProof { layers: Vec::new() };
        let mut tv = tp.clone();
        assert!(!verify_fractional_chain(&empty, claimed, 6, &mut tv));
    }

    /// S3: steering the root pair along the top layer's constraint line
    /// (a' = a + gamma*delta, b' = b - delta) leaves every layer identity
    /// unchanged, so pre-fix the split check accepts it even though the LogUp
    /// final ratio is rewritten. Post-fix the roots are DERIVED from the top
    /// layer's child evals, so the pre-fix steering attack no longer exists;
    /// its closest structural form — forging the top-layer child evals the
    /// roots derive from — must be rejected.
    #[test]
    fn s3_root_steer_rejected() {
        let (w, _t, tp, claimed) = honest_split_pair(0x5304);
        let mut forged = FractionalProof { layers: Vec::new() };
        for (i, layer) in w.layers.iter().enumerate() {
            let mut lp = FractionalLayer {
                gamma: layer.gamma,
                s: layer.s.clone(),
                num_merge: layer.num_merge.clone(),
                den_merge: layer.den_merge.clone(),
                proof: layer.proof.clone(),
            };
            if i == 0 {
                lp.proof.final_evals[2] = lp.proof.final_evals[2] + Goldilocks::ONE;
                lp.proof.final_evals[5] = lp.proof.final_evals[5] + Goldilocks::ONE;
            }
            forged.layers.push(lp);
        }
        let mut tv = tp.clone();
        assert!(!verify_fractional_chain(&forged, claimed, 6, &mut tv));
    }

    /// S3: the eq columns must be recomputed in closed form, never taken from
    /// the proof. At the TOP layer (arity 0) the eq points are empty, so the
    /// true eq evals are BOTH ONE and the parenthesized tree identities are
    /// zero — setting the stored eq evals (final_evals[0]/[1]) to ZERO leaves
    /// every pre-fix check untouched and is ACCEPTED. Post-fix the verifier
    /// computes eq0 == eq1 == 1 and rejects the forged columns.
    #[test]
    fn s3_eq_evals_recomputed_rejected() {
        let (w, _t, tp, claimed) = honest_split_pair(0x5305);
        let mut forged = FractionalProof { layers: Vec::new() };
        for (i, layer) in w.layers.iter().enumerate() {
            let mut lp = FractionalLayer {
                gamma: layer.gamma,
                s: layer.s.clone(),
                num_merge: layer.num_merge.clone(),
                den_merge: layer.den_merge.clone(),
                proof: layer.proof.clone(),
            };
            if i == 0 {
                // Top layer: both parenthesized parts are zero (the roots are
                // the honest tree roots), so the eq values vanish from the
                // pre-fix virtual-eval check — any stored eq passes.
                lp.proof.final_evals[0] = Goldilocks::ZERO;
                lp.proof.final_evals[1] = Goldilocks::ZERO;
            }
            forged.layers.push(lp);
        }
        let mut tv = tp.clone();
        // S3: must be REJECTED (pre-fix it is ACCEPTED).
        assert!(!verify_fractional_chain(&forged, claimed, 6, &mut tv));
    }

    /// S3: zero denominators must be handled by the protocol — 0/0 must not
    /// make the cross-multiplied relation hold vacuously.
    #[test]
    fn s3_zero_denominator_rejected() {
        let n = 1usize << 4;
        let num: Vec<Goldilocks> = vec![Goldilocks::ONE; n];
        let mut den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 5) as i64 + 1)).collect();
        den[3] = Goldilocks::ZERO; // one zero entry -> tree den root == 0
        let mut start = Transcript::new(b"zkie-logup-test");
        start.absorb_u64(0x5306);
        let mut tp = start.clone();
        let w = prove_fractional(num.clone(), den.clone(), &mut tp);
        let t = prove_fractional(num.clone(), den.clone(), &mut tp);
        // Pre-fix the split check was ACCEPTED: both chains pass and the
        // cross relation 0 == 0 holds vacuously. S3: zero denominators must
        // be rejected explicitly.
        let mut tv = start.clone();
        assert!(!verify_lookup_split(&w, &t, 4, 4, &mut tv));
        // Chain level: an all-zero numerator hides the zero denominator
        // pre-fix (final_num == 0 == 0*final_den). S3: rejected explicitly.
        let num0: Vec<Goldilocks> = vec![Goldilocks::ZERO; n];
        let mut tp2 = start.clone();
        let w0 = prove_fractional(num0.clone(), den.clone(), &mut tp2);
        let mut tv2 = start.clone();
        assert!(!verify_fractional_chain(&w0, Goldilocks::ZERO, 4, &mut tv2));
    }

    /// S3: the layer count must equal the column-derived tree depth. Dropping
    /// the BOTTOM layer leaves every remaining gamma in place, so pre-fix the
    /// chain replays fine with leaves one level up and the split check
    /// ACCEPTS it; post-fix layers.len() != t rejects it.
    #[test]
    fn s3_truncated_chain_rejected() {
        let (w, _t, tp, claimed) = honest_split_pair(0x5307);
        let truncated = FractionalProof {
            layers: w.layers[..w.layers.len() - 1]
                .iter()
                .map(|layer| FractionalLayer {
                    gamma: layer.gamma,
                    s: layer.s.clone(),
                    num_merge: layer.num_merge.clone(),
                    den_merge: layer.den_merge.clone(),
                    proof: layer.proof.clone(),
                })
                .collect(),
        };
        // Pre-fix this was ACCEPTED at the chain level (transcript-aligned,
        // self-consistent leaves one level up; the split check only rejects
        // it incidentally via the transcript shift). S3: layers.len() must
        // equal the column depth.
        let mut tv = tp.clone();
        assert!(!verify_fractional_chain(&truncated, claimed, 6, &mut tv));
    }

    /// S3: malformed layer structure is rejected, never a panic. A truncated
    /// final_evals (5 entries instead of 6) PANICS the pre-fix verifier; the
    /// fixed verifier rejects it as malformed.
    #[test]
    fn s3_malformed_structure_rejected() {
        let (w, _t, tp, claimed) = honest_split_pair(0x5308);
        let mut forged = FractionalProof { layers: Vec::new() };
        for (i, layer) in w.layers.iter().enumerate() {
            let mut lp = FractionalLayer {
                gamma: layer.gamma,
                s: layer.s.clone(),
                num_merge: layer.num_merge.clone(),
                den_merge: layer.den_merge.clone(),
                proof: layer.proof.clone(),
            };
            if i == 1 {
                lp.proof.final_evals.truncate(5);
            }
            forged.layers.push(lp);
        }
        let mut tv = tp.clone();
        // S3: must be rejected WITHOUT panicking (pre-fix panics on the
        // out-of-bounds final-eval index).
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_fractional_chain(&forged, claimed, 6, &mut tv)
        }));
        assert!(matches!(outcome, Ok(false)), "malformed fe must reject, not panic");
    }

    /// S3 guard: tampering a top-layer child eval breaks the top layer's
    /// same_poly merge claims and is rejected (kept green pre- and post-fix).
    #[test]
    fn s3_top_layer_child_eval_tamper_rejected() {
        let (w, _t, tp, claimed) = honest_split_pair(0x5309);
        let mut forged = FractionalProof { layers: Vec::new() };
        for (i, layer) in w.layers.iter().enumerate() {
            let mut lp = FractionalLayer {
                gamma: layer.gamma,
                s: layer.s.clone(),
                num_merge: layer.num_merge.clone(),
                den_merge: layer.den_merge.clone(),
                proof: layer.proof.clone(),
            };
            if i == 0 {
                lp.proof.final_evals[2] = lp.proof.final_evals[2] + Goldilocks::ONE;
            }
            forged.layers.push(lp);
        }
        let mut tv = tp.clone();
        assert!(!verify_fractional_chain(&forged, claimed, 6, &mut tv));
    }

    /// S3: the claimed root binding final_num == claimed*final_den must hold
    /// for the DERIVED roots; an empty layer list plus matching root fields
    /// passes pre-fix even though nothing was reduced. S3: the layer count
    /// must equal the column depth, and the roots only exist as derived
    /// values.
    #[test]
    fn s3_empty_layer_list_with_claimed_rejected() {
        let (_w, _t, tp, claimed) = honest_split_pair(0x530A);
        let empty = FractionalProof { layers: Vec::new() };
        let mut tv = tp.clone();
        assert!(!verify_fractional_chain(&empty, claimed, 6, &mut tv));
    }

    /// Manual timing harness (run with `cargo test -p zkie-core --release
    /// frac_timing -- --ignored --nocapture`): measures the layered fractional
    /// prove/verify on a real-sized 2^17-entry instance.
    #[test]
    #[ignore]
    fn frac_timing() {
        let mut rng = XorShift64::new(0x71_71);
        let n = 1usize << 17;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });
        let (mut tp, tv) = replay_pair(0x71);
        let t0 = std::time::Instant::now();
        let proof = prove_fractional(num.clone(), den.clone(), &mut tp);
        let prove_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut tv = tv;
        let ok = verify_fractional_chain(&proof, claimed, 17, &mut tv);
        let verify_s = t1.elapsed().as_secs_f64();
        eprintln!(
            "frac n={} layers={} prove={:.3}s verify={:.3}s ok={}",
            n,
            proof.layers.len(),
            prove_s,
            verify_s,
            ok
        );
    }

    /// Lagrange interpolation through a round's stored values at 0,1,2,3.
    fn lagrange4(v: &[Goldilocks], r: Goldilocks) -> Goldilocks {
        let one = Goldilocks::ONE;
        let two = Goldilocks::from_u64(2);
        let three = Goldilocks::from_u64(3);
        let six_inv = Goldilocks::from_u64(6).inverse();
        let two_inv = two.inverse();
        let l0 = -(r - one) * (r - two) * (r - three) * six_inv;
        let l1 = r * (r - two) * (r - three) * two_inv;
        let l2 = -r * (r - one) * (r - three) * two_inv;
        let l3 = r * (r - one) * (r - two) * six_inv;
        v[0] * l0 + v[1] * l1 + v[2] * l2 + v[3] * l3
    }

    /// Steer a round (stored values at 0,1,2,3) to satisfy p(0)+p(1) == prev
    /// and p(r) == target under an advance-known challenge r.
    fn steer4(prev: Goldilocks, target: Goldilocks, r: Goldilocks) -> Vec<Goldilocks> {
        let zero = Goldilocks::ZERO;
        let one = Goldilocks::ONE;
        let two = Goldilocks::from_u64(2);
        let three = Goldilocks::from_u64(3);
        let six_inv = Goldilocks::from_u64(6).inverse();
        let two_inv = two.inverse();
        let l0 = -(r - one) * (r - two) * (r - three) * six_inv;
        let l1 = r * (r - two) * (r - three) * two_inv;
        let l2 = -r * (r - one) * (r - three) * two_inv;
        let l3 = r * (r - one) * (r - two) * six_inv;
        if l2 != zero {
            return vec![prev, zero, (target - prev * l0) * l2.inverse(), zero];
        }
        if l3 != zero {
            return vec![prev, zero, zero, (target - prev * l0) * l3.inverse()];
        }
        assert_ne!(l3, zero, "fixed seed must avoid fully degenerate r");
        vec![zero, prev, zero, (target - prev * l1) * l3.inverse()]
    }
}

/// LogUp permutation argument: prove two same-length multisets `a` and `b` are
/// equal as multisets (i.e. `b` is a permutation of `a`) via the fractional
/// identity `sum 1/(alpha - a_i) == sum 1/(alpha - b_j)`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PermutationProof {
    pub a_side: FractionalProof,
    pub b_side: FractionalProof,
    pub alpha: Goldilocks,
    pub a: Vec<Goldilocks>,
    pub b: Vec<Goldilocks>,
}

/// num = indicator, den = alpha - value (padded with value 0, so padding den = alpha).
pub fn build_perm_num_den(
    value: &[Goldilocks],
    alpha: Goldilocks,
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let total = value.len().next_power_of_two().max(32);
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![alpha; total];
    for i in 0..value.len() {
        num[i] = Goldilocks::from_u64(1);
        den[i] = alpha - value[i];
    }
    (num, den)
}

pub fn prove_permutation(
    a: &[Goldilocks],
    b: &[Goldilocks],
    transcript: &mut Transcript,
) -> PermutationProof {
    assert_eq!(a.len(), b.len());
    let alpha = transcript.field();
    let (num_a, den_a) = build_perm_num_den(a, alpha);
    let (num_b, den_b) = build_perm_num_den(b, alpha);
    let a_side = prove_fractional(num_a, den_a, transcript);
    let b_side = prove_fractional(num_b, den_b, transcript);
    PermutationProof {
        a_side,
        b_side,
        alpha,
        a: a.to_vec(),
        b: b.to_vec(),
    }
}

/// `t` is the shared column depth of both sides (the permutation domain,
/// padded to at least 2^5 — S3).
pub fn verify_permutation(
    proof: &PermutationProof,
    t: usize,
    transcript: &mut Transcript,
) -> bool {
    let alpha = transcript.field();
    if alpha != proof.alpha {
        return false;
    }
    verify_lookup_split(&proof.a_side, &proof.b_side, t, t, transcript)
}

#[cfg(test)]
mod permutation_tests {
    use super::*;

    #[test]
    fn permutation_roundtrip() {
        let mut rng = XorShift64::new(0xABCD);
        let a: Vec<Goldilocks> = vec![1, 2, 3, 4]
            .into_iter()
            .map(|v| Goldilocks::from_u64(v))
            .collect();
        // b is a permutation of a
        let b: Vec<Goldilocks> = vec![3, 1, 4, 2]
            .into_iter()
            .map(|v| Goldilocks::from_u64(v))
            .collect();
        let mut tp = Transcript::new(b"zkie-perm-test");
        tp.absorb_u64(0xAB00);
        let proof = prove_permutation(&a, &b, &mut tp);
        let mut tv = Transcript::new(b"zkie-perm-test");
        tv.absorb_u64(0xAB00);
        assert!(verify_permutation(&proof, 5, &mut tv));
        // Tamper a layer child eval: the chained verify must reject.
        let mut bad_proof = PermutationProof {
            a_side: FractionalProof { layers: Vec::new() },
            b_side: proof.b_side,
            alpha: proof.alpha,
            a: proof.a,
            b: proof.b,
        };
        for layer in &proof.a_side.layers {
            let mut lp = FractionalLayer {
                gamma: layer.gamma,
                s: layer.s.clone(),
                num_merge: layer.num_merge.clone(),
                den_merge: layer.den_merge.clone(),
                proof: layer.proof.clone(),
            };
            lp.proof.final_evals[3] = lp.proof.final_evals[3] + Goldilocks::ONE;
            bad_proof.a_side.layers.push(lp);
        }
        let mut tv2 = Transcript::new(b"zkie-perm-test");
        tv2.absorb_u64(0xAB00);
        assert!(!verify_permutation(&bad_proof, 5, &mut tv2));
    }
}
