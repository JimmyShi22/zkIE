//! logUp GKR: prove sum_i num[i]/den[i] == C via the fraction-addition tree
//! with eq-weighted virtual sumchecks (DeepProve "fractional sumcheck").
use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use crate::common::fixed_point::from_i64;
use crate::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
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
        let proof = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &r);
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

pub fn prove_fractional(
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

pub fn verify_fractional(
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
        let mles: Vec<&[Goldilocks]> = vec![
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
        if !verify_virtual(
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

// Chained store-free verify: check each layer sumcheck against final_evals
// already stored in its VirtualProof (no num/den rebuild), then enforce the
// inter-layer MLE folding relation so next_num_{k-1}(r[k..]) agrees with the
// low/high evals of num_k at r[k+1..]. Leaf authentication (opening num_0/den_0
// at the full challenge point) restores soundness in P2/P3.
// Core chained-GKR check: per-layer sumcheck self-consistency + inter-layer
// MLE folding. Does not check the final_num/final_den == claimed relation, so
// split lookup/table proofs can share it and enforce the link separately.
pub fn verify_fractional_chain_core(proof: &FractionalProof) -> bool {
    let one = Goldilocks::from_u64(1);
    for layer in &proof.layers {
        let terms = build_terms(layer.c);
        if !verify_virtual(
            &layer.proof,
            &terms,
            Goldilocks::from_u64(0),
            &layer.r,
            &layer.proof.final_evals,
        ) {
            return false;
        }
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

pub fn verify_fractional_chain(proof: &FractionalProof, claimed: Goldilocks) -> bool {
    verify_fractional_chain_core(proof) && proof.final_num == claimed * proof.final_den
}

// Store-free verify: use the final_evals already stored in each layer
// VirtualProof, without rebuilding the num/den fraction tree.
// Leaf authentication (opening) restores soundness in P2/P3.
pub fn verify_fractional_storefree(proof: &FractionalProof, claimed: Goldilocks) -> bool {
    for layer in &proof.layers {
        let terms = build_terms(layer.c);
        if !verify_virtual(
            &layer.proof,
            &terms,
            Goldilocks::from_u64(0),
            &layer.r,
            &layer.proof.final_evals,
        ) {
            return false;
        }
    }
    proof.final_num == claimed * proof.final_den
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

pub fn prove_lookup_witness(
    x: &[u32],
    y: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut impl ChallengeSource,
) -> FractionalProof {
    let (num, den) = build_lookup_witness_num_den(x, y, alpha, beta);
    prove_fractional(num, den, rng)
}

// Batch several lookups (same table) into one witness-side fractional sumcheck.
// The witness side is committed once per batch; the table side is committed once
// globally. Linked together this is the DeepProve-style split LogUp.
pub fn prove_lookup_witness_batch(
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut impl ChallengeSource,
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
    prove_fractional(num, den, rng)
}

pub fn prove_lookup_witness_batch_columns(
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut impl ChallengeSource,
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
    let proof = prove_fractional(num.clone(), den.clone(), rng);
    (proof, x_flat, y_flat, num, den)
}

pub fn prove_lookup_table(
    m: &[u64],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut impl ChallengeSource,
) -> FractionalProof {
    let (num, den) = build_lookup_table_num_den(m, table, alpha, beta);
    prove_fractional(num, den, rng)
}

// Verify the split lookup: each side is chained-GKR sound, and the two sides
// agree on the fractional sum (cross-multiplied LogUp identity).
pub fn verify_lookup_split(witness: &FractionalProof, table: &FractionalProof) -> bool {
    verify_fractional_chain_core(witness)
        && verify_fractional_chain_core(table)
        && witness.final_num * table.final_den == table.final_num * witness.final_den
}

pub fn prove_lookup_fractional(
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut impl ChallengeSource,
) -> FractionalProof {
    assert_eq!(x.len(), y.len());
    let (num, den) = build_lookup_num_den(x, y, table, alpha, beta);
    prove_fractional(num, den, rng)
}

// Prove a lookup and also return the merged num_0/den_0 columns, so the
// committed path can batch-commit them and open at the leaf point.
pub fn prove_lookup_fractional_columns(
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut impl ChallengeSource,
) -> (FractionalProof, Vec<Goldilocks>, Vec<Goldilocks>) {
    assert_eq!(x.len(), y.len());
    let (num, den) = build_lookup_num_den(x, y, table, alpha, beta);
    let proof = prove_fractional(num.clone(), den.clone(), rng);
    (proof, num, den)
}
pub fn verify_lookup_fractional(
    proof: &FractionalProof,
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
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
    verify_fractional(proof, num, den, Goldilocks::from_u64(0))
}

// Chained store-free verify for a lookup: no x/y/table rebuild, no full
// witness columns. Soundness is restored by opening num_0/den_0 at full_r
// (see fractional_leaf) against their commitments in the committed path.
pub fn verify_lookup_fractional_chain(proof: &FractionalProof) -> bool {
    verify_fractional_chain(proof, Goldilocks::from_u64(0))
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
    rng: &mut impl ChallengeSource,
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
    prove_fractional(num, den, rng)
}

pub fn verify_lookup_fractional_batch(
    proof: &FractionalProof,
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
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
    verify_fractional(proof, num, den, Goldilocks::from_u64(0))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_roundtrip() {
        let mut rng = XorShift64::new(0xDEC0);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });
        let proof = prove_fractional(num.clone(), den.clone(), &mut rng);
        assert!(verify_fractional(&proof, num.clone(), den.clone(), claimed));
        let tampered = FractionalProof {
            final_num: proof.final_num + Goldilocks::from_u64(1),
            final_den: proof.final_den,
            layers: proof.layers,
            full_r: proof.full_r,
        };
        assert!(!verify_fractional(
            &tampered,
            num.clone(),
            den.clone(),
            claimed
        ));
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
        let mut proof = prove_fractional(num.clone(), den.clone(), &mut rng);
        assert!(verify_fractional(&proof, num.clone(), den.clone(), claimed));
        assert!(verify_fractional_storefree(&proof, claimed));
        proof.layers[0].proof.final_evals[0] += Goldilocks::from_u64(1);
        assert!(!verify_fractional_storefree(&proof, claimed));
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
        let mut proof = prove_fractional(num.clone(), den.clone(), &mut rng);
        assert!(verify_fractional(&proof, num.clone(), den.clone(), claimed));
        assert!(verify_fractional_chain(&proof, claimed));
        proof.layers[0].proof.final_evals[3] += Goldilocks::from_u64(1);
        assert!(!verify_fractional_chain(&proof, claimed));
    }
    #[test]
    fn lookup_fractional_roundtrip() {
        let mut rng = XorShift64::new(0x10AD);
        let n = 64usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let alpha = rng.field();
        let beta = rng.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_fractional(
            &proof, &x, &y, &table, alpha, beta
        ));
        let mut bad_y = y.clone();
        bad_y[0] = bad_y[0] + Goldilocks::from_u64(1);
        let proof_bad = prove_lookup_fractional(&x, &bad_y, &table, alpha, beta, &mut rng);
        assert!(!verify_lookup_fractional(
            &proof_bad, &x, &bad_y, &table, alpha, beta
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
        let alpha = rng.field();
        let beta = rng.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
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
        let alpha = rng.field();
        let beta = rng.field();
        let (proof, num, den) = prove_lookup_fractional_columns(&x, &y, &table, alpha, beta, &mut rng);
        let (full_r, num_leaf, den_leaf) = fractional_leaf(&proof);
        let arity = num.len().trailing_zeros() as usize;
        let whir = Whir::new_testing(arity);
        let c_num = commit(&whir, &num);
        let (open_num, ev_num) = whir.open(c_num.prover_data.clone(), &c_num.protocol, &full_r);
        assert_eq!(whir.verify(&c_num.commitment, &open_num, &c_num.protocol, &full_r).unwrap(), ev_num);
        assert_eq!(ev_num, num_leaf);
        let c_den = commit(&whir, &den);
        let (open_den, ev_den) = whir.open(c_den.prover_data.clone(), &c_den.protocol, &full_r);
        assert_eq!(whir.verify(&c_den.commitment, &open_den, &c_den.protocol, &full_r).unwrap(), ev_den);
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
        let alpha = rng.field();
        let beta = rng.field();
        let mut proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_fractional(&proof, &x, &y, &table, alpha, beta));
        assert!(verify_lookup_fractional_chain(&proof));
        proof.layers[0].proof.final_evals[3] += Goldilocks::from_u64(1);
        assert!(!verify_lookup_fractional_chain(&proof));
    }
    #[test]
    fn lookup_split_matches_merged() {
        let mut rng = XorShift64::new(0x5E17);
        let n = 64usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let alpha = rng.field();
        let beta = rng.field();
        let merged = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_fractional(&merged, &x, &y, &table, alpha, beta));
        let m = lookup_multiplicity(&x, t);
        let witness = prove_lookup_witness(&x, &y, alpha, beta, &mut rng);
        let table_proof = prove_lookup_table(&m, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_split(&witness, &table_proof));
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
        let alpha = rng.field();
        let beta = rng.field();
        let xrefs: Vec<&[u32]> = xs.iter().map(|v| v.as_slice()).collect();
        let yrefs: Vec<&[Goldilocks]> = ys.iter().map(|v| v.as_slice()).collect();
        let merged = prove_lookup_fractional_batch(&xrefs, &yrefs, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_fractional_batch(&merged, &xrefs, &yrefs, &table, alpha, beta));
        let m = merge_multiplicity(&xrefs, t);
        let witness = prove_lookup_witness_batch(&xrefs, &yrefs, alpha, beta, &mut rng);
        let table_proof = prove_lookup_table(&m, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_split(&witness, &table_proof));
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
        let alpha = rng.field();
        let beta = rng.field();
        let table_proof = prove_lookup_table(&m, &table, alpha, beta, &mut rng);
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
        assert_eq!(whir.verify(&c_m.commitment, &open_m, &c_m.protocol, &full_r).unwrap(), ev_m);
        assert_eq!(ev_m, num_leaf);
        let c_t = commit(&whir, &table_pad);
        let (open_t, ev_t) = whir.open(c_t.prover_data.clone(), &c_t.protocol, &full_r);
        assert_eq!(whir.verify(&c_t.commitment, &open_t, &c_t.protocol, &full_r).unwrap(), ev_t);
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
        let alpha = rng.field();
        let beta = rng.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_fractional(
            &proof, &x, &y, &table, alpha, beta
        ));
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
    rng: &mut impl ChallengeSource,
) -> PermutationProof {
    assert_eq!(a.len(), b.len());
    let alpha = rng.field();
    let (num_a, den_a) = build_perm_num_den(a, alpha);
    let (num_b, den_b) = build_perm_num_den(b, alpha);
    let a_side = prove_fractional(num_a, den_a, rng);
    let b_side = prove_fractional(num_b, den_b, rng);
    PermutationProof {
        a_side,
        b_side,
        alpha,
        a: a.to_vec(),
        b: b.to_vec(),
    }
}

pub fn verify_permutation(proof: &PermutationProof) -> bool {
    verify_lookup_split(&proof.a_side, &proof.b_side)
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
        let proof = prove_permutation(&a, &b, &mut rng);
        assert!(verify_permutation(&proof));
        // a non-permutation must reject
        let bad: Vec<Goldilocks> = vec![3, 1, 4, 5]
            .into_iter()
            .map(|v| Goldilocks::from_u64(v))
            .collect();
        let bad_proof = prove_permutation(&a, &bad, &mut rng);
        assert!(!verify_permutation(&bad_proof));
    }
}
