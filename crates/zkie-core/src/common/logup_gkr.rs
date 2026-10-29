//! logUp GKR: prove sum_i num[i]/den[i] == C via the fraction-addition tree
//! with eq-weighted virtual sumchecks (DeepProve "fractional sumcheck").
//!
//! S1 challenge derivation (Fiat-Shamir, per-layer interleaved):
//! - `c_k` is squeezed from the transcript before layer k's terms are built,
//!   binding the statement (k = 0) and every previous layer's round messages.
//! - Layer 0's eq point `q_0` is squeezed before its sumcheck (the eq table
//!   needs the point upfront), and layer 0's sumcheck challenge vector `r_0`
//!   is derived per-round from layer 0's own round messages.
//! - Inner layer k's challenge vector is the nested restriction
//!   `r_k = r_{k-1}[..-1]`, which keeps the inter-layer chain checks
//!   well-defined; its round messages are still absorbed so `c_{k+1}` binds
//!   them. The verifier replays the same interleaving against the proof's
//!   stored layers and checks every stored `c`/`r` against the derived values.
//! - The peeled top variable `full_r[t-1]` is squeezed after the last layer's
//!   messages and binds the whole tree.
//!
//! The `*_with_challenges` / `*_rng` variants keep the old pre-derived
//! challenge signatures and are TEST ONLY; production paths must use the
//! transcript versions.

use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use crate::common::fixed_point::from_i64;
use crate::common::sumcheck::{
    absorb_virtual_rounds, prove_virtual, prove_virtual_with_challenges, verify_virtual,
    verify_virtual_with_challenges, virtual_eval, VirtualProof,
};
use crate::common::transcript::Transcript;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FractionalLayer {
    pub r: Vec<Goldilocks>,
    pub c: Goldilocks,
    pub proof: VirtualProof,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FractionalProof {
    pub layers: Vec<FractionalLayer>,
    pub final_num: Goldilocks,
    pub final_den: Goldilocks,
    pub full_r: Vec<Goldilocks>,
}

fn build_terms(c: Goldilocks) -> Vec<(Goldilocks, Vec<usize>)> {
    let neg = from_i64(-1);
    vec![
        (Goldilocks::from_u64(1), vec![0usize, 1]),
        (neg, vec![0, 3, 6]),
        (neg, vec![0, 4, 5]),
        (c, vec![0, 2]),
        (neg * c, vec![0, 5, 6]),
    ]
}

/// TEST ONLY: prove with pre-derived `full_r`/`cs` (the pre-S1 construction).
pub fn prove_fractional_with_challenges(
    mut num: Vec<Goldilocks>,
    mut den: Vec<Goldilocks>,
    full_r: &[Goldilocks],
    cs: &[Goldilocks],
) -> FractionalProof {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    let total_vars = num.len().trailing_zeros() as usize;
    assert_eq!(full_r.len(), total_vars);
    assert_eq!(cs.len(), total_vars);
    let mut layers = Vec::new();
    let mut layer_idx = 0usize;
    while num.len() > 1 {
        let half = num.len() / 2;
        let (num_low, num_high) = num.split_at(half);
        let (den_low, den_high) = den.split_at(half);
        let next_num: Vec<Goldilocks> = (0..half)
            .map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s])
            .collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();
        let r: Vec<Goldilocks> = full_r[..total_vars - layer_idx - 1].to_vec();
        let c = cs[layer_idx];
        let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&r);
        let mles: Vec<&[Goldilocks]> = vec![
            &eq, &next_num, &next_den, num_low, num_high, den_low, den_high,
        ];
        let terms = build_terms(c);
        let proof = prove_virtual_with_challenges(&mles, &terms, Goldilocks::from_u64(0), &r);
        layers.push(FractionalLayer { r, c, proof });
        num = next_num;
        den = next_den;
        layer_idx += 1;
    }
    FractionalProof {
        layers,
        final_num: num[0],
        final_den: den[0],
        full_r: full_r.to_vec(),
    }
}

/// Fiat-Shamir prove (S1): per-layer interleaved transcript derivation, see the
/// module doc for the exact ordering.
pub fn prove_fractional(
    mut num: Vec<Goldilocks>,
    mut den: Vec<Goldilocks>,
    transcript: &mut Transcript,
) -> FractionalProof {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    let mut layers = Vec::new();
    let mut prev_r: Option<Vec<Goldilocks>> = None;
    let mut r0: Option<Vec<Goldilocks>> = None;
    while num.len() > 1 {
        let half = num.len() / 2;
        let (num_low, num_high) = num.split_at(half);
        let (den_low, den_high) = den.split_at(half);
        let next_num: Vec<Goldilocks> = (0..half)
            .map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s])
            .collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();

        // c_k binds the previous layer's messages (or the statement for k = 0).
        let c = transcript.field();
        let terms = build_terms(c);
        let arity = half.trailing_zeros() as usize;
        let (proof, r) = match &prev_r {
            None => {
                // Layer 0: eq point squeezed upfront (the eq table needs it),
                // sumcheck challenges derived per-round from its own rounds.
                let q0: Vec<Goldilocks> = transcript.field_vec(arity);
                let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&q0);
                let mles: Vec<&[Goldilocks]> = vec![
                    &eq, &next_num, &next_den, num_low, num_high, den_low, den_high,
                ];
                let (proof, r) = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), transcript);
                r0 = Some(r.clone());
                (proof, r)
            }
            Some(prev) => {
                // Inner layer: nested restriction of the previous layer's
                // transcript-derived vector; rounds absorbed for c_{k+1}.
                let r: Vec<Goldilocks> = prev[..prev.len() - 1].to_vec();
                let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&r);
                let mles: Vec<&[Goldilocks]> = vec![
                    &eq, &next_num, &next_den, num_low, num_high, den_low, den_high,
                ];
                let proof =
                    prove_virtual_with_challenges(&mles, &terms, Goldilocks::from_u64(0), &r);
                absorb_virtual_rounds(transcript, &proof.rounds);
                (proof, r)
            }
        };
        layers.push(FractionalLayer { r: r.clone(), c, proof });
        prev_r = Some(r);
        num = next_num;
        den = next_den;
    }
    // The peeled top variable, squeezed after the last layer's messages. The
    // full opening point is layer 0's challenge vector plus the top variable.
    let top = transcript.field();
    let mut full_r = r0.expect("num.len() > 1 guarantees at least one layer");
    full_r.push(top);
    FractionalProof {
        layers,
        final_num: num[0],
        final_den: den[0],
        full_r,
    }
}

/// TEST ONLY: prove with an arbitrary challenge source (samples `full_r`/`cs`
/// upfront, then calls the with-challenges construction).
pub fn prove_fractional_rng(
    num: Vec<Goldilocks>,
    den: Vec<Goldilocks>,
    rng: &mut impl ChallengeSource,
) -> FractionalProof {
    assert!(num.len().is_power_of_two());
    let total_vars = num.len().trailing_zeros() as usize;
    let full_r: Vec<Goldilocks> = (0..total_vars).map(|_| rng.field()).collect();
    let cs: Vec<Goldilocks> = (0..total_vars).map(|_| rng.field()).collect();
    prove_fractional_with_challenges(num, den, &full_r, &cs)
}

/// Fiat-Shamir verify (S1, store-based): rebuilds the num/den tree and replays
/// the per-layer transcript interleaving against the proof.
pub fn verify_fractional(
    proof: &FractionalProof,
    mut num: Vec<Goldilocks>,
    mut den: Vec<Goldilocks>,
    claimed: Goldilocks,
    transcript: &mut Transcript,
) -> bool {
    let mut prev_r: Option<Vec<Goldilocks>> = None;
    let mut r0: Option<Vec<Goldilocks>> = None;
    for layer in &proof.layers {
        let half = num.len() / 2;
        if half == 0 {
            return false; // more layers than the statement's tree depth
        }
        let num_low = &num[..half];
        let num_high = &num[half..];
        let den_low = &den[..half];
        let den_high = &den[half..];
        let next_num: Vec<Goldilocks> = (0..half)
            .map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s])
            .collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();

        let c = transcript.field();
        if c != layer.c {
            return false;
        }
        let terms = build_terms(c);
        let arity = half.trailing_zeros() as usize;
        match &prev_r {
            None => {
                let q0: Vec<Goldilocks> = transcript.field_vec(arity); // advance state
                let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&q0);
                let (ok, r, final_claim) = verify_virtual(
                    &layer.proof,
                    &terms,
                    Goldilocks::from_u64(0),
                    arity,
                    transcript,
                );
                if !ok || r != layer.r {
                    return false;
                }
                let fe = vec![
                    crate::common::mle::eval(&eq, &r),
                    crate::common::mle::eval(&next_num, &r),
                    crate::common::mle::eval(&next_den, &r),
                    crate::common::mle::eval(num_low, &r),
                    crate::common::mle::eval(num_high, &r),
                    crate::common::mle::eval(den_low, &r),
                    crate::common::mle::eval(den_high, &r),
                ];
                if final_claim != virtual_eval(&terms, &fe) {
                    return false;
                }
                r0 = Some(r.clone());
                prev_r = Some(r);
            }
            Some(prev) => {
                let r: Vec<Goldilocks> = prev[..prev.len() - 1].to_vec();
                if r != layer.r {
                    return false;
                }
                let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&r);
                let fe = vec![
                    crate::common::mle::eval(&eq, &r),
                    crate::common::mle::eval(&next_num, &r),
                    crate::common::mle::eval(&next_den, &r),
                    crate::common::mle::eval(num_low, &r),
                    crate::common::mle::eval(num_high, &r),
                    crate::common::mle::eval(den_low, &r),
                    crate::common::mle::eval(den_high, &r),
                ];
                if !verify_virtual_with_challenges(
                    &layer.proof,
                    &terms,
                    Goldilocks::from_u64(0),
                    &r,
                    &fe,
                ) {
                    return false;
                }
                absorb_virtual_rounds(transcript, &layer.proof.rounds);
                prev_r = Some(r);
            }
        }
        num = next_num;
        den = next_den;
    }
    if num.len() != 1 {
        return false; // fewer layers than the statement's tree depth
    }
    let top = transcript.field();
    let mut full_r = r0.expect("non-empty layers");
    full_r.push(top);
    if full_r != proof.full_r {
        return false;
    }
    proof.final_num == claimed * proof.final_den
}

/// TEST ONLY: verify with the proof's stored (pre-derived) challenges.
pub fn verify_fractional_with_challenges(
    proof: &FractionalProof,
    mut num: Vec<Goldilocks>,
    mut den: Vec<Goldilocks>,
    claimed: Goldilocks,
) -> bool {
    for layer in &proof.layers {
        let half = num.len() / 2;
        let num_low = &num[..half];
        let num_high = &num[half..];
        let den_low = &den[..half];
        let den_high = &den[half..];
        let next_num: Vec<Goldilocks> = (0..half)
            .map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s])
            .collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();
        let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&layer.r);
        let _mles: Vec<&[Goldilocks]> = vec![
            &eq, &next_num, &next_den, num_low, num_high, den_low, den_high,
        ];
        let terms = build_terms(layer.c);
        let final_evals = vec![
            crate::common::mle::eval(&eq, &layer.r),
            crate::common::mle::eval(&next_num, &layer.r),
            crate::common::mle::eval(&next_den, &layer.r),
            crate::common::mle::eval(num_low, &layer.r),
            crate::common::mle::eval(num_high, &layer.r),
            crate::common::mle::eval(den_low, &layer.r),
            crate::common::mle::eval(den_high, &layer.r),
        ];
        if !verify_virtual_with_challenges(
            &layer.proof,
            &terms,
            Goldilocks::from_u64(0),
            &layer.r,
            &final_evals,
        ) {
            return false;
        }
        num = next_num;
        den = next_den;
    }
    proof.final_num == claimed * proof.final_den
}

/// Store-free per-layer transcript replay shared by the chain-core and
/// store-free verifies: squeezes and checks every `c`, replays layer 0's rounds
/// per-round (deriving `r_0`), verifies the inner layers at the nested
/// restrictions, and checks the stored `full_r` against `r_0 || top`.
fn replay_fractional_storefree(proof: &FractionalProof, transcript: &mut Transcript) -> bool {
    if proof.layers.is_empty() || proof.full_r.is_empty() {
        return false;
    }
    let mut prev_r: Option<Vec<Goldilocks>> = None;
    let mut r0: Option<Vec<Goldilocks>> = None;
    for layer in &proof.layers {
        let c = transcript.field();
        if c != layer.c {
            return false;
        }
        let terms = build_terms(c);
        match &prev_r {
            None => {
                // Layer 0: arity from the statement-bound full_r length.
                let arity = proof.full_r.len() - 1;
                let _q0: Vec<Goldilocks> = transcript.field_vec(arity); // advance state
                let (ok, r, final_claim) = verify_virtual(
                    &layer.proof,
                    &terms,
                    Goldilocks::from_u64(0),
                    arity,
                    transcript,
                );
                if !ok || r != layer.r {
                    return false;
                }
                if final_claim != virtual_eval(&terms, &layer.proof.final_evals) {
                    return false;
                }
                r0 = Some(r.clone());
                prev_r = Some(r);
            }
            Some(prev) => {
                let r: Vec<Goldilocks> = prev[..prev.len() - 1].to_vec();
                if r != layer.r {
                    return false;
                }
                if !verify_virtual_with_challenges(
                    &layer.proof,
                    &terms,
                    Goldilocks::from_u64(0),
                    &r,
                    &layer.proof.final_evals,
                ) {
                    return false;
                }
                absorb_virtual_rounds(transcript, &layer.proof.rounds);
                prev_r = Some(r);
            }
        }
    }
    let top = transcript.field();
    let mut full_r = r0.expect("non-empty layers");
    full_r.push(top);
    full_r == proof.full_r
}

/// Chained store-free verify: per-layer sumcheck self-consistency + inter-layer
/// MLE folding, with the full S1 transcript replay. Does not check the
/// final_num/final_den == claimed relation, so split lookup/table proofs can
/// share it and enforce the link separately.
pub fn verify_fractional_chain_core(proof: &FractionalProof, transcript: &mut Transcript) -> bool {
    let one = Goldilocks::from_u64(1);
    if !replay_fractional_storefree(proof, transcript) {
        return false;
    }
    for w in proof.layers.windows(2) {
        let prev = &w[0];
        let curr = &w[1];
        let rk = prev.r[prev.r.len() - 1];
        let fold_num = (one - rk) * curr.proof.final_evals[3] + rk * curr.proof.final_evals[4];
        if prev.proof.final_evals[1] != fold_num {
            return false;
        }
        let fold_den = (one - rk) * curr.proof.final_evals[5] + rk * curr.proof.final_evals[6];
        if prev.proof.final_evals[2] != fold_den {
            return false;
        }
    }
    true
}

pub fn verify_fractional_chain(
    proof: &FractionalProof,
    claimed: Goldilocks,
    transcript: &mut Transcript,
) -> bool {
    verify_fractional_chain_core(proof, transcript) && proof.final_num == claimed * proof.final_den
}

// Store-free verify: use the final_evals already stored in each layer
// VirtualProof, without rebuilding the num/den fraction tree.
// Leaf authentication (opening) restores soundness in P2/P3.
pub fn verify_fractional_storefree(
    proof: &FractionalProof,
    claimed: Goldilocks,
    transcript: &mut Transcript,
) -> bool {
    replay_fractional_storefree(proof, transcript) && proof.final_num == claimed * proof.final_den
}

// Leaf of the chained GKR: recover eval(num_0, full_r) and eval(den_0, full_r)
// from the first layer final_evals by folding the peeled top variable. These are
// the only two values that must be opened against the committed num_0/den_0
// columns in P2/P3.
pub fn fractional_leaf(proof: &FractionalProof) -> (Vec<Goldilocks>, Goldilocks, Goldilocks) {
    let one = Goldilocks::from_u64(1);
    let top = *proof.full_r.last().unwrap();
    let num_leaf = (one - top) * proof.layers[0].proof.final_evals[3]
        + top * proof.layers[0].proof.final_evals[4];
    let den_leaf = (one - top) * proof.layers[0].proof.final_evals[5]
        + top * proof.layers[0].proof.final_evals[6];
    (proof.full_r.clone(), num_leaf, den_leaf)
}

/// Prove `y_i == table[x_i]` for every `i` via the fractional-sumcheck logUp:
/// `sum_i 1/(alpha+x_i+beta*y_i) = sum_j m_j/(alpha+j+beta*table[j])`, encoded as
/// a combined fraction list whose sum is 0 and proven with `prove_fractional`.
/// This replaces the standalone grand-product form (`prove_product`).
// Build the merged LogUp columns num_0/den_0 from the lookup witness x/y and
// the table. These are the only columns the chained GKR must commit and open
// (equivalent to committing x/y + table/multiplicity).
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

// Verify the split lookup: each side is chained-GKR sound, and the two sides
// agree on the fractional sum (cross-multiplied LogUp identity). The witness
// side is replayed first, then the table side, matching the prover's order.
pub fn verify_lookup_split(
    witness: &FractionalProof,
    table: &FractionalProof,
    transcript: &mut Transcript,
) -> bool {
    verify_fractional_chain_core(witness, transcript)
        && verify_fractional_chain_core(table, transcript)
        && witness.final_num * table.final_den == table.final_num * witness.final_den
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
// witness columns. Soundness is restored by opening num_0/den_0 at full_r
// (see fractional_leaf) against their commitments in the committed path.
pub fn verify_lookup_fractional_chain(
    proof: &FractionalProof,
    transcript: &mut Transcript,
) -> bool {
    verify_fractional_chain(proof, Goldilocks::from_u64(0), transcript)
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
    fn fractional_roundtrip() {
        let mut rng = XorShift64::new(0xDEC0);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });
        let proof = prove_fractional_rng(num.clone(), den.clone(), &mut rng);
        assert!(verify_fractional_with_challenges(
            &proof,
            num.clone(),
            den.clone(),
            claimed
        ));
        let tampered = FractionalProof {
            final_num: proof.final_num + Goldilocks::from_u64(1),
            final_den: proof.final_den,
            layers: proof.layers,
            full_r: proof.full_r,
        };
        assert!(!verify_fractional_with_challenges(
            &tampered,
            num.clone(),
            den.clone(),
            claimed
        ));
    }

    /// S1: transcript prove/verify roundtrip, honest replay, and round-message
    /// binding (tampering layer 0's first round breaks the derived challenges).
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

        // Tamper layer 0's first round message: the store-free replay derives
        // different challenges and must reject.
        let mut tampered = FractionalProof {
            final_num: proof.final_num,
            final_den: proof.final_den,
            layers: Vec::new(),
            full_r: proof.full_r.clone(),
        };
        for (i, layer) in proof.layers.iter().enumerate() {
            let mut lp = FractionalLayer {
                r: layer.r.clone(),
                c: layer.c,
                proof: layer.proof.clone(),
            };
            if i == 0 {
                lp.proof.rounds[0][0] = lp.proof.rounds[0][0] + Goldilocks::from_u64(1);
            }
            tampered.layers.push(lp);
        }
        let (mut _tp2, tv2) = replay_pair(0xF5);
        let mut tv2 = tv2;
        assert!(!verify_fractional(&tampered, num.clone(), den.clone(), claimed, &mut tv2));
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
        assert!(verify_fractional_storefree(&proof, claimed, &mut tv));
        proof.layers[0].proof.final_evals[0] += Goldilocks::from_u64(1);
        let (mut _tp2, tv2) = replay_pair(0xF5F);
        let mut tv2 = tv2;
        assert!(!verify_fractional_storefree(&proof, claimed, &mut tv2));
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
        assert!(verify_fractional_chain(&proof, claimed, &mut tv));
        proof.layers[0].proof.final_evals[3] += Goldilocks::from_u64(1);
        let (mut _tp2, tv2) = replay_pair(0xCA17);
        let mut tv2 = tv2;
        assert!(!verify_fractional_chain(&proof, claimed, &mut tv2));
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
        // Replay the alpha/beta preamble, then the fractional transcript.
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
        let (full_r, num_leaf, den_leaf) = fractional_leaf(&proof);
        assert_eq!(num_leaf, crate::common::mle::eval(&num, &full_r));
        assert_eq!(den_leaf, crate::common::mle::eval(&den, &full_r));
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
        let (full_r, num_leaf, den_leaf) = fractional_leaf(&proof);
        let arity = num.len().trailing_zeros() as usize;
        let whir = Whir::new_testing(arity);
        let c_num = commit(&whir, &num);
        let (open_num, ev_num) = whir.open(c_num.prover_data.clone(), &c_num.protocol, &full_r);
        assert_eq!(
            whir.verify(&c_num.commitment, &open_num, &c_num.protocol, &full_r)
                .unwrap(),
            ev_num
        );
        assert_eq!(ev_num, num_leaf);
        let c_den = commit(&whir, &den);
        let (open_den, ev_den) = whir.open(c_den.prover_data.clone(), &c_den.protocol, &full_r);
        assert_eq!(
            whir.verify(&c_den.commitment, &open_den, &c_den.protocol, &full_r)
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
        assert!(verify_lookup_fractional_chain(&proof, &mut tv));
        proof.layers[0].proof.final_evals[3] += Goldilocks::from_u64(1);
        let (mut _tp2, tv2) = replay_pair(0x11A0);
        let mut tv2 = tv2;
        assert_eq!(tv2.field(), alpha);
        assert_eq!(tv2.field(), beta);
        assert!(!verify_lookup_fractional_chain(&proof, &mut tv2));
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
        assert!(verify_lookup_split(&witness, &table_proof, &mut tv2));
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
        assert!(verify_lookup_split(&witness, &table_proof, &mut tv2));
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
        let (full_r, num_leaf, den_leaf) = fractional_leaf(&table_proof);
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
        let (open_m, ev_m) = whir.open(c_m.prover_data.clone(), &c_m.protocol, &full_r);
        assert_eq!(
            whir.verify(&c_m.commitment, &open_m, &c_m.protocol, &full_r)
                .unwrap(),
            ev_m
        );
        assert_eq!(ev_m, num_leaf);
        let c_t = commit(&whir, &table_pad);
        let (open_t, ev_t) = whir.open(c_t.prover_data.clone(), &c_t.protocol, &full_r);
        assert_eq!(
            whir.verify(&c_t.commitment, &open_t, &c_t.protocol, &full_r)
                .unwrap(),
            ev_t
        );
        let j_mle: Vec<Goldilocks> = (0..total).map(|j| Goldilocks::from_u64(j as u64)).collect();
        let expected_den = alpha + crate::common::mle::eval(&j_mle, &full_r) + beta * ev_t;
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

pub fn verify_permutation(proof: &PermutationProof, transcript: &mut Transcript) -> bool {
    let alpha = transcript.field();
    if alpha != proof.alpha {
        return false;
    }
    verify_lookup_split(&proof.a_side, &proof.b_side, transcript)
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
        assert!(verify_permutation(&proof, &mut tv));
        // Tamper a leaf eval: the chained verify must reject.
        let mut bad_proof = PermutationProof {
            a_side: FractionalProof {
                final_num: proof.a_side.final_num,
                final_den: proof.a_side.final_den,
                layers: Vec::new(),
                full_r: proof.a_side.full_r.clone(),
            },
            b_side: proof.b_side,
            alpha: proof.alpha,
            a: proof.a,
            b: proof.b,
        };
        for layer in &proof.a_side.layers {
            let mut lp = FractionalLayer {
                r: layer.r.clone(),
                c: layer.c,
                proof: layer.proof.clone(),
            };
            lp.proof.final_evals[3] = lp.proof.final_evals[3] + Goldilocks::ONE;
            bad_proof.a_side.layers.push(lp);
        }
        let mut tv2 = Transcript::new(b"zkie-perm-test");
        tv2.absorb_u64(0xAB00);
        assert!(!verify_permutation(&bad_proof, &mut tv2));
    }
}
