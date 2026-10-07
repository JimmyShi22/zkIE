//! Direct (deliberately NON-succinct) indexed lookup bridge, fixed logical
//! N = 8 / WHIR arity 7 (tensors padded to 128).
//!
//! Relation: for every logical row `i` in `0..8`, the committed witness index
//! `idx[i]` (represented as three committed bit vectors `b0/b1/b2`, padded
//! with zeros) selects a row of the committed static table, and the committed
//! output `out[i]` equals `table[idx[i]]`.
//!
//! Proof shape: for every logical `i`, five WHIR opening pairs (re/im) at
//! Boolean vertices —
//! - `b0`, `b1`, `b2`, `out` opened at the vertex of flat index `i`
//!   (LSB-first point `[bit0(i), bit1(i), bit2(i), 0, 0, 0, 0]`, converted to
//!   the PCS order by the shared `to_p3_point` reversal);
//! - `table` opened at `[b0_i, b1_i, b2_i, 0, 0, 0, 0]` built from the
//!   AUTHENTICATED bit values, so the table row is selected by the committed
//!   index, not by anything the prover can relabel.
//!
//! Verifier = statement (five root pairs) + proof only: no witness vectors,
//! no host recomputation, no `open` calls. It checks, per row: both opening
//! components verify against the right roots at the right points; the opened
//! bits are exactly 0/1; the table value at the bit-selected vertex equals
//! the opened output value.
//!
//! Soundness statement: this bridge is sound as a pointwise relation proof
//! under WHIR binding — the relation checked is exactly "for each of the
//! eight logical positions, `out` at that position equals `table` at the
//! vertex selected by the committed bits, which are 0/1". There is NO
//! sumcheck and NO Fiat–Shamir challenge here; all points are fixed Boolean
//! vertices. The cost is O(N) openings (8 rows x 5 tensors x re+im = 80 WHIR
//! openings) and the logical size is hard-wired to N = 8. This is a
//! bridge to `Op::Lookup`, not a scalable lookup argument.

use crate::extension_lookup_fractional::{gen, to_p3_point};
use zkie_core::common::field::{EF, Goldilocks, PrimeCharacteristicRing};
use zkie_core::pcs::whir::{Commitment, OpeningProtocol, Point, Proof, ProverData, Whir};

/// Logical table/query size (fixed).
pub const N: usize = 8;
/// WHIR arity: 3 index bits + 4 padding bits.
pub const ARITY: usize = 7;
const PAD: usize = 1 << ARITY;

#[derive(Clone, Debug)]
pub struct TensorCommitment {
    pub re: Commitment,
    pub im: Commitment,
}

/// Statement: the five committed padded tensors (roots only).
#[derive(Clone, Debug)]
pub struct Statement {
    pub b0: TensorCommitment,
    pub b1: TensorCommitment,
    pub b2: TensorCommitment,
    pub out: TensorCommitment,
    pub tbl: TensorCommitment,
}

/// One opening pair (re/im) at one point.
#[derive(Clone)]
pub struct LeafOpen {
    pub re: (Proof, EF),
    pub im: (Proof, EF),
}

/// One logical row: the five opening pairs described in the module header.
#[derive(Clone)]
pub struct RowProof {
    pub opens_b0: LeafOpen,
    pub opens_b1: LeafOpen,
    pub opens_b2: LeafOpen,
    pub opens_out: LeafOpen,
    pub opens_tbl: LeafOpen,
}

/// The transported proof: one `RowProof` per logical row, in row order
/// (row position `k` fixes the Boolean vertex, so rows cannot be reordered
/// or omitted without breaking the WHIR openings).
#[derive(Clone)]
pub struct LookupProof {
    pub rows: Vec<RowProof>,
}

struct Data {
    root: TensorCommitment,
    re_pd: ProverData,
    im_pd: ProverData,
    proto: OpeningProtocol,
}

fn commit(w: &Whir, vals: Vec<Goldilocks>) -> Data {
    let (re_root, re_pd, proto) = w.commit(&vals);
    let (im_root, im_pd, _) = w.commit(&vec![Goldilocks::ZERO; PAD]);
    Data {
        root: TensorCommitment { re: re_root, im: im_root },
        re_pd,
        im_pd,
        proto,
    }
}

fn field_bits(bits: &[u64]) -> Vec<EF> {
    bits.iter()
        .map(|&b| EF::from(Goldilocks::from_u64(b)))
        .collect()
}

/// Our-convention (LSB-first) Boolean vertex of flat index `k`: the low
/// three coordinates carry the bits of `k`, the high four are zero.
fn vertex_of(k: usize) -> Vec<EF> {
    let mut v = field_bits(&[(k & 1) as u64, ((k >> 1) & 1) as u64, ((k >> 2) & 1) as u64]);
    v.extend(vec![EF::ZERO; ARITY - 3]);
    v
}

fn open_pair(w: &Whir, d: &Data, pt: &[EF]) -> LeafOpen {
    let p3 = to_p3_point(pt);
    LeafOpen {
        re: w.open_ef(&d.root.re, d.re_pd.clone(), &d.proto, &p3),
        im: w.open_ef(&d.root.im, d.im_pd.clone(), &d.proto, &p3),
    }
}

/// Verify both components of an opening at `pt` against `root` and return
/// the recombined EF value, or `None` on any mismatch.
fn verify_pair(
    w: &Whir,
    root: &TensorCommitment,
    leaf: &LeafOpen,
    pt: &[EF],
    proto: &OpeningProtocol,
) -> Option<EF> {
    let p3 = to_p3_point(pt);
    let re = w.verify_ef(&root.re, &leaf.re.0, proto, &p3).ok()?;
    let im = w.verify_ef(&root.im, &leaf.im.0, proto, &p3).ok()?;
    (re == leaf.re.1 && im == leaf.im.1).then(|| leaf.re.1 + gen() * leaf.im.1)
}

/// Prove the fixed N = 8 indexed lookup. Returns `(statement, proof)` or
/// `None` for malformed inputs (wrong arity, wrong logical sizes, an index
/// out of range). The caller supplies the witness; the verifier never sees
/// it.
pub fn prove(
    w: &Whir,
    indices: &[u8],
    out: &[u64],
    table: &[u64],
) -> Option<(Statement, LookupProof)> {
    if w.num_variables() != ARITY
        || indices.len() != N
        || out.len() != N
        || table.len() != N
        || indices.iter().any(|&i| i as usize >= N)
    {
        return None;
    }
    let mut b0 = vec![Goldilocks::ZERO; PAD];
    let mut b1 = vec![Goldilocks::ZERO; PAD];
    let mut b2 = vec![Goldilocks::ZERO; PAD];
    let mut out_pad = vec![Goldilocks::ZERO; PAD];
    for i in 0..N {
        let j = indices[i] as usize;
        b0[i] = Goldilocks::from_u64((j & 1) as u64);
        b1[i] = Goldilocks::from_u64(((j >> 1) & 1) as u64);
        b2[i] = Goldilocks::from_u64(((j >> 2) & 1) as u64);
        out_pad[i] = Goldilocks::from_u64(out[i]);
    }
    let mut tbl_pad = vec![Goldilocks::ZERO; PAD];
    for j in 0..N {
        tbl_pad[j] = Goldilocks::from_u64(table[j]);
    }
    let d_b0 = commit(w, b0);
    let d_b1 = commit(w, b1);
    let d_b2 = commit(w, b2);
    let d_out = commit(w, out_pad);
    let d_tbl = commit(w, tbl_pad);
    let stmt = Statement {
        b0: d_b0.root.clone(),
        b1: d_b1.root.clone(),
        b2: d_b2.root.clone(),
        out: d_out.root.clone(),
        tbl: d_tbl.root.clone(),
    };
    let mut rows = Vec::with_capacity(N);
    for k in 0..N {
        let vertex = vertex_of(k);
        // The table point is the vertex of the WITNESS index: the prover
        // knows `indices[k]`; the verifier rebuilds the same point from the
        // authenticated bit openings.
        let tbl_pt = vertex_of(indices[k] as usize);
        rows.push(RowProof {
            opens_b0: open_pair(w, &d_b0, &vertex),
            opens_b1: open_pair(w, &d_b1, &vertex),
            opens_b2: open_pair(w, &d_b2, &vertex),
            opens_out: open_pair(w, &d_out, &vertex),
            opens_tbl: open_pair(w, &d_tbl, &tbl_pt),
        });
    }
    Some((stmt, LookupProof { rows }))
}

/// Verify the fixed N = 8 indexed lookup from statement and proof only.
/// Malformed shapes return `false` (never panic).
pub fn verify(w: &Whir, stmt: &Statement, proof: &LookupProof) -> bool {
    if w.num_variables() != ARITY || proof.rows.len() != N {
        return false;
    }
    let proto = w.opening_protocol(ARITY, 1);
    for (k, row) in proof.rows.iter().enumerate() {
        let vertex = vertex_of(k);
        let Some(vb0) = verify_pair(w, &stmt.b0, &row.opens_b0, &vertex, &proto) else {
            return false;
        };
        let Some(vb1) = verify_pair(w, &stmt.b1, &row.opens_b1, &vertex, &proto) else {
            return false;
        };
        let Some(vb2) = verify_pair(w, &stmt.b2, &row.opens_b2, &vertex, &proto) else {
            return false;
        };
        if vb0 != EF::ZERO && vb0 != EF::ONE {
            return false;
        }
        if vb1 != EF::ZERO && vb1 != EF::ONE {
            return false;
        }
        if vb2 != EF::ZERO && vb2 != EF::ONE {
            return false;
        }
        let Some(vout) = verify_pair(w, &stmt.out, &row.opens_out, &vertex, &proto) else {
            return false;
        };
        let tbl_pt = vec![vb0, vb1, vb2, EF::ZERO, EF::ZERO, EF::ZERO, EF::ZERO];
        let Some(vtbl) = verify_pair(w, &stmt.tbl, &row.opens_tbl, &tbl_pt, &proto) else {
            return false;
        };
        if vtbl != vout {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness() -> (Vec<u8>, Vec<u64>, Vec<u64>) {
        let indices = vec![3, 0, 7, 2, 5, 1, 6, 4];
        let table: Vec<u64> = (0..N).map(|j| (j + 3) as u64).collect();
        let out: Vec<u64> = indices.iter().map(|&i| table[i as usize]).collect();
        (indices, out, table)
    }

    fn fixture() -> (Whir, Statement, LookupProof) {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (indices, out, table) = witness();
        let (s, p) = prove(&w, &indices, &out, &table).expect("honest prove");
        (w, s, p)
    }

    #[test]
    fn honest_roundtrip() {
        let (w, s, p) = fixture();
        assert!(verify(&w, &s, &p));
    }

    /// Verifier never opens: statement + proof only.
    #[test]
    fn verify_never_opens() {
        let (w, s, p) = fixture();
        let before = w.open_stats();
        assert!(verify(&w, &s, &p));
        assert_eq!(w.open_stats(), before, "verifier must never open");
    }

    /// Build a statement/proof pair that MIRRORS `prove` exactly, except that
    /// the committed `b0[0]` value is `bad` and row 0's table opening point
    /// is computed from the b0 = 0 bit (so every PCS opening is authentic,
    /// even when the committed bit violates the relation).
    fn proof_with_b0_zero(w: &Whir, bad: Goldilocks) -> (Statement, LookupProof) {
        let (indices, out, table) = witness();
        let pad = |vals: &[u64]| -> Vec<Goldilocks> {
            let mut v = vec![Goldilocks::ZERO; PAD];
            for (i, &x) in vals.iter().enumerate() {
                v[i] = Goldilocks::from_u64(x);
            }
            v
        };
        let bits = |sh: u32| -> Vec<u64> {
            indices.iter().map(|&i| ((i as u64) >> sh) & 1).collect()
        };
        let mut b0 = pad(&bits(0));
        let b1 = pad(&bits(1));
        let b2 = pad(&bits(2));
        b0[0] = bad;
        let d_b0 = commit(w, b0);
        let d_b1 = commit(w, b1);
        let d_b2 = commit(w, b2);
        let d_out = commit(w, pad(&out));
        let d_tbl = commit(w, pad(&table));
        let stmt = Statement {
            b0: d_b0.root.clone(),
            b1: d_b1.root.clone(),
            b2: d_b2.root.clone(),
            out: d_out.root.clone(),
            tbl: d_tbl.root.clone(),
        };
        let mut rows = Vec::with_capacity(N);
        for k in 0..N {
            let vertex = vertex_of(k);
            // Row 0's table point uses the b0 = 0 bit (the point stays a
            // Boolean vertex); all other rows use the witness bits.
            let j0 = if k == 0 { 0 + 2 * bits(1)[0] + 4 * bits(2)[0] } else { indices[k] as u64 };
            rows.push(RowProof {
                opens_b0: open_pair(w, &d_b0, &vertex),
                opens_b1: open_pair(w, &d_b1, &vertex),
                opens_b2: open_pair(w, &d_b2, &vertex),
                opens_out: open_pair(w, &d_out, &vertex),
                opens_tbl: open_pair(w, &d_tbl, &vertex_of(j0 as usize)),
            });
        }
        (stmt, LookupProof { rows })
    }

    /// An HONESTLY committed non-Boolean bit (value 2) with authentic WHIR
    /// openings: the opening layer accepts it (verify_pair returns 2), but
    /// the relation's Boolean gate rejects it.
    #[test]
    fn committed_nonboolean_bit_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (s, p) = proof_with_b0_zero(&w, Goldilocks::from_u64(2));
        // The opening IS authentic: it verifies and yields the committed 2.
        let proto = w.opening_protocol(ARITY, 1);
        let vb0 = verify_pair(&w, &s.b0, &p.rows[0].opens_b0, &vertex_of(0), &proto)
            .expect("authentic opening of the committed value");
        assert_eq!(vb0, EF::from(Goldilocks::from_u64(2)));
        // ...but the relation's Boolean gate rejects the whole proof.
        assert!(!verify(&w, &s, &p));
    }

    /// An HONESTLY committed wrong index bit (valid 0/1, but flipped) with
    /// authentic openings: the table is opened at the vertex the b0 = 0 bit
    /// selects, every opening verifies, and the equality gate rejects.
    #[test]
    fn committed_wrong_index_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (s, p) = proof_with_b0_zero(&w, Goldilocks::ZERO);
        assert!(!verify(&w, &s, &p));
    }

    /// An HONESTLY committed wrong output (out[0] = table[indices[0]] + 1)
    /// with fully authentic openings: the equality gate rejects.
    #[test]
    fn committed_wrong_output_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (indices, mut out, table) = witness();
        out[0] += 1;
        let (s, p) = prove(&w, &indices, &out, &table).expect("authentic openings");
        assert!(!verify(&w, &s, &p));
    }

    /// Malformed proof shape: a wrong row count must be rejected.
    #[test]
    fn malformed_row_count_rejected() {
        let (w, s, mut p) = fixture();
        p.rows.pop();
        assert!(!verify(&w, &s, &p));
        let (w, s, mut p) = fixture();
        p.rows.push(p.rows[0].clone());
        assert!(!verify(&w, &s, &p));
    }

    /// A table opening substituted from another row is bound to the wrong
    /// point: WHIR rejects it before any value comparison.
    #[test]
    fn substituted_table_opening_rejected() {
        let (w, s, mut p) = fixture();
        p.rows[0].opens_tbl = p.rows[1].opens_tbl.clone();
        assert!(!verify(&w, &s, &p));
    }

    /// A tampered statement root must be rejected.
    #[test]
    fn tampered_root_rejected() {
        let (w, s, p) = fixture();
        let mut rng = zkie_core::common::field::XorShift64::new(0xD17);
        let fake: Vec<Goldilocks> = (0..PAD).map(|_| rng.field()).collect();
        let (r, _, _) = w.commit(&fake);
        let mut bad = s.clone();
        bad.tbl.re = r;
        assert!(!verify(&w, &bad, &p));
    }

    /// Malformed inputs are rejected by `prove` (never panic).
    #[test]
    fn malformed_inputs_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (indices, out, table) = witness();
        assert!(prove(&w, &indices, &out, &table).is_some());
        assert!(prove(&w, &indices[..7], &out, &table).is_none());
        assert!(prove(&w, &indices, &out[..7], &table).is_none());
        assert!(prove(&w, &indices, &out, &table[..7]).is_none());
        let bad_indices = vec![3, 0, 9, 2, 5, 1, 6, 4]; // 9 out of range
        assert!(prove(&w, &bad_indices, &out, &table).is_none());
        let w2 = Whir::new_target(6, 90, 0).expect("valid arity");
        assert!(prove(&w2, &indices, &out, &table).is_none());
    }

    /// The honest proof's table points equal the witness index vertices
    /// (spot reference: row k's table opening equals table[indices[k]]).
    #[test]
    fn honest_table_points_match_witness() {
        let (w, s, p) = fixture();
        assert!(verify(&w, &s, &p));
        let (indices, out, table) = witness();
        let proto = w.opening_protocol(ARITY, 1);
        for (k, row) in p.rows.iter().enumerate() {
            let j = indices[k] as usize;
            let vtbl = verify_pair(&w, &s.tbl, &row.opens_tbl, &vertex_of(j), &proto)
                .expect("honest table opening verifies at the witness vertex");
            assert_eq!(vtbl, EF::from(Goldilocks::from_u64(table[j])));
            assert_eq!(vtbl, EF::from(Goldilocks::from_u64(out[k])));
        }
    }
}
