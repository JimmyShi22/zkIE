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
//!
//! Dispatch seam: `LookupWitnessN8` / `CommittedLookupStatement` /
//! `CommittedLookupProof` / `dispatch_prove` / `LookupProofDispatch` form the
//! integration surface for a future `Op::Lookup` / compose path. That path
//! emits a typed committed artifact (statement + proof, no witness) and the
//! verifier dispatches `verify_with_whir` with WHIR only. The seam does NOT
//! change the fixed N = 8, non-succinct scope of this module.

use crate::compose::{Op, Store, TensorData};
use crate::extension_lookup_fractional::{gen, to_p3_point};
use zkie_core::common::field::{EF, Goldilocks, PrimeCharacteristicRing, PrimeField64};
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

// ==== dispatch seam (Op::Lookup / compose integration point) ====
//
// The seam below is the integration surface for future compose paths. A
// future succinct variant extends the two enums; the verifier call site
// (`verify_with_whir`) does not change.

/// Fixed-N=8 logical witness for the direct lookup bridge (arrays keep the
/// logical size exactly 8). Values must be below the Goldilocks modulus.
#[derive(Clone, Debug)]
pub struct LookupWitnessN8 {
    pub indices: [u8; N],
    pub out: [u64; N],
    pub table: [u64; N],
}

impl LookupWitnessN8 {
    /// Build from slices of logical length exactly 8; any other length or an
    /// index >= 8 yields `None` (logical N stays exactly 8).
    pub fn from_slices(indices: &[u8], out: &[u64], table: &[u64]) -> Option<Self> {
        if indices.len() != N || out.len() != N || table.len() != N {
            return None;
        }
        if indices.iter().any(|&i| i as usize >= N) {
            return None;
        }
        Some(LookupWitnessN8 {
            indices: indices.try_into().expect("length checked"),
            out: out.try_into().expect("length checked"),
            table: table.try_into().expect("length checked"),
        })
    }

    /// Prover-side extraction from a REAL `Op::Lookup` IR node. Matches only
    /// `Op::Lookup { idx, out, table }`; validates the tensor ids, the exact
    /// logical length N = 8, every index < 8, and the tensor METADATA
    /// (Owned length / mmap range with checked arithmetic) BEFORE
    /// materializing; then materializes the output/table tensors
    /// (mmap-safe) and converts Goldilocks values to their canonical u64
    /// representatives. Any other op or any out-of-contract shape yields
    /// `None`. This only READS tensors on prover extraction; the legacy
    /// compose paths are untouched.
    pub fn from_op_lookup(op: &Op, store: &Store) -> Option<Self> {
        let Op::Lookup { idx, out, table } = op else {
            return None;
        };
        if *idx >= store.idx.len() || *out >= store.v.len() || *table >= store.v.len() {
            return None;
        }
        let idxs = &store.idx[*idx];
        if idxs.len() != N || idxs.iter().any(|&i| i as usize >= N) {
            return None;
        }
        // Validate metadata BEFORE `materialize`: a lying or overflowing
        // mmap range must never reach the read. Applied to BOTH output and
        // table.
        if !tensor_metadata_ok(store, *out) || !tensor_metadata_ok(store, *table) {
            return None;
        }
        let out_t = store.materialize(*out);
        let tbl_t = store.materialize(*table);
        if out_t.len() != N || tbl_t.len() != N {
            return None;
        }
        let mut indices = [0u8; N];
        let mut out_arr = [0u64; N];
        let mut table_arr = [0u64; N];
        for i in 0..N {
            indices[i] = idxs[i] as u8;
            out_arr[i] = out_t[i].as_canonical_u64();
            table_arr[i] = tbl_t[i].as_canonical_u64();
        }
        Some(LookupWitnessN8 {
            indices,
            out: out_arr,
            table: table_arr,
        })
    }
}

/// Pre-materialize metadata check for one tensor id (caller has already
/// bounds-checked the id): the logical length must be exactly N = 8, and
/// for mmap-backed tensors the claimed `[start, start + len)` range must
/// exist inside the mapping via checked arithmetic.
fn tensor_metadata_ok(store: &Store, id: usize) -> bool {
    match &store.v[id] {
        TensorData::Owned(v) => v.len() == N,
        TensorData::Mmap { mmap, start, len } => {
            *len == N && start.checked_add(*len).is_some_and(|end| end <= mmap.len())
        }
    }
}

/// Typed committed lookup statement. One variant per proof system; the
/// current variant wraps the direct fixed-N=8 statement.
#[derive(Clone, Debug)]
pub enum CommittedLookupStatement {
    DirectN8(Statement),
}

/// Typed committed lookup proof (statement + proof only — no witness).
#[derive(Clone)]
pub enum CommittedLookupProof {
    DirectN8(LookupProof),
}

/// Produce a typed committed lookup artifact pair from the fixed-N=8
/// witness. `None` for a malformed witness (index out of range).
pub fn dispatch_prove(
    w: &Whir,
    wit: &LookupWitnessN8,
) -> Option<(CommittedLookupStatement, CommittedLookupProof)> {
    let (s, p) = prove(w, &wit.indices, &wit.out, &wit.table)?;
    Some((CommittedLookupStatement::DirectN8(s), CommittedLookupProof::DirectN8(p)))
}

/// Prover-side dispatch from a real `Op::Lookup` IR node: extract the fixed
/// N = 8 witness from the store (validated, mmap-safe) and emit the typed
/// committed artifact. `None` for a non-Lookup op or out-of-contract shapes.
pub fn dispatch_op_lookup_n8(
    w: &Whir,
    op: &Op,
    store: &Store,
) -> Option<(CommittedLookupStatement, CommittedLookupProof)> {
    let wit = LookupWitnessN8::from_op_lookup(op, store)?;
    dispatch_prove(w, &wit)
}

/// Verifier-side dispatch: verify a committed lookup proof against its
/// statement using WHIR only — no witness, no host recomputation, no `open`
/// calls. Variant/pairing mismatches return `false`.
pub trait LookupProofDispatch {
    fn verify_with_whir(&self, w: &Whir, stmt: &CommittedLookupStatement) -> bool;
}

impl LookupProofDispatch for CommittedLookupProof {
    fn verify_with_whir(&self, w: &Whir, stmt: &CommittedLookupStatement) -> bool {
        match (self, stmt) {
            (CommittedLookupProof::DirectN8(p), CommittedLookupStatement::DirectN8(s)) => {
                verify(w, s, p)
            }
            // Future variants must pair identically; anything else rejects.
            _ => false,
        }
    }
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

    /// The dispatch seam end-to-end: witness -> dispatched artifact -> witness
    /// DROPPED -> the verifier accepts from statement + proof only, and never
    /// opens.
    #[test]
    fn dispatch_roundtrip_after_witness_dropped() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (indices, out, table) = witness();
        let wit = LookupWitnessN8::from_slices(&indices, &out, &table).expect("valid witness");
        let (stmt, proof) = dispatch_prove(&w, &wit).expect("dispatch prove");
        drop(wit); // the verifier holds statement + proof only
        let before = w.open_stats();
        assert!(proof.verify_with_whir(&w, &stmt));
        assert_eq!(w.open_stats(), before, "verifier must never open");
    }

    /// The witness constructor rejects anything that is not exactly N = 8 or
    /// contains an out-of-range index.
    #[test]
    fn dispatch_witness_slices_validated() {
        let (indices, out, table) = witness();
        assert!(LookupWitnessN8::from_slices(&indices, &out, &table).is_some());
        assert!(LookupWitnessN8::from_slices(&indices[..7], &out, &table).is_none());
        assert!(LookupWitnessN8::from_slices(&indices, &out[..7], &table).is_none());
        assert!(LookupWitnessN8::from_slices(&indices, &out, &table[..7]).is_none());
        let bad = vec![3, 0, 9, 2, 5, 1, 6, 4]; // index 9 out of range
        assert!(LookupWitnessN8::from_slices(&bad, &out, &table).is_none());
    }

    /// Cross-witness pairing (statement from witness A, proof from witness
    /// B) must be rejected: the opening roots no longer match.
    #[test]
    fn dispatch_cross_witness_pairing_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (i0, o0, t0) = witness();
        let wit0 = LookupWitnessN8::from_slices(&i0, &o0, &t0).expect("valid witness");
        let i1 = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let o1: Vec<u64> = i1.iter().map(|&i| t0[i as usize]).collect();
        let wit1 = LookupWitnessN8::from_slices(&i1, &o1, &t0).expect("valid witness");
        let (s0, _) = dispatch_prove(&w, &wit0).expect("dispatch prove");
        let (_, p1) = dispatch_prove(&w, &wit1).expect("dispatch prove");
        assert!(!p1.verify_with_whir(&w, &s0));
    }

    /// A malformed inner proof is rejected through the dispatch (delegation
    /// to the legacy `verify`).
    #[test]
    fn dispatch_malformed_proof_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (indices, out, table) = witness();
        let wit = LookupWitnessN8::from_slices(&indices, &out, &table).expect("valid witness");
        let (stmt, proof) = dispatch_prove(&w, &wit).expect("dispatch prove");
        let CommittedLookupProof::DirectN8(mut inner) = proof else {
            unreachable!("single variant")
        };
        inner.rows.pop();
        assert!(!CommittedLookupProof::DirectN8(inner).verify_with_whir(&w, &stmt));
    }

    /// A REAL store + `Op::Lookup` fixture: idx tensor, table tensor, and
    /// the materialized output `out[i] = table[idx[i]]`.
    fn op_fixture(out_wrong: bool) -> (Store, Op) {
        let mut store = Store::new();
        let table: Vec<Goldilocks> = (0..N).map(|j| Goldilocks::from_u64((j + 3) as u64)).collect();
        let idxs: Vec<u32> = vec![3, 0, 7, 2, 5, 1, 6, 4];
        let mut out_vals: Vec<Goldilocks> = idxs
            .iter()
            .map(|&i| table[i as usize])
            .collect();
        if out_wrong {
            out_vals[0] = out_vals[0] + Goldilocks::ONE;
        }
        let idx_id = store.push_idx(idxs);
        let tbl_id = store.push(table);
        let out_id = store.push(out_vals);
        (store, Op::Lookup { idx: idx_id, out: out_id, table: tbl_id })
    }

    /// End-to-end through the real IR: `Op::Lookup` + Store -> dispatch ->
    /// Store/op/witness dropped -> verifier accepts from statement + proof
    /// only and never opens.
    #[test]
    fn dispatch_op_lookup_n8_roundtrip_after_ir_dropped() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (store, op) = op_fixture(false);
        // The extraction reads exactly the store's lookup triple.
        let wit = LookupWitnessN8::from_op_lookup(&op, &store).expect("extract witness");
        assert_eq!(wit.indices, [3, 0, 7, 2, 5, 1, 6, 4]);
        assert_eq!(wit.out[0], 6); // table[3]
        let (stmt, proof) = dispatch_op_lookup_n8(&w, &op, &store).expect("dispatch");
        drop(store);
        drop(op);
        drop(wit);
        let before = w.open_stats();
        assert!(proof.verify_with_whir(&w, &stmt));
        assert_eq!(w.open_stats(), before, "verifier must never open");
    }

    /// A real IR fixture whose output tensor is WRONG (out[0] !=
    /// table[idx[0]]) still dispatches (prover-side extraction is honest
    /// about the store), but the committed relation fails verification.
    #[test]
    fn dispatch_op_lookup_n8_wrong_output_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (store, op) = op_fixture(true);
        let (stmt, proof) = dispatch_op_lookup_n8(&w, &op, &store).expect("dispatch");
        drop(store);
        drop(op);
        assert!(!proof.verify_with_whir(&w, &stmt));
    }

    /// Non-Lookup ops and out-of-contract shapes are rejected by the
    /// extraction (`None`), never a panic.
    #[test]
    fn dispatch_op_lookup_n8_bad_shapes_rejected() {
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let (store, _) = op_fixture(false);
        // A non-Lookup op.
        let add = Op::Add { a: 0, b: 1, c: 2 };
        assert!(LookupWitnessN8::from_op_lookup(&add, &store).is_none());
        assert!(dispatch_op_lookup_n8(&w, &add, &store).is_none());
        // Out-of-range tensor ids.
        let bad_ids = Op::Lookup { idx: 7, out: 0, table: 0 };
        assert!(LookupWitnessN8::from_op_lookup(&bad_ids, &store).is_none());
        let bad_ids = Op::Lookup { idx: 0, out: 99, table: 0 };
        assert!(LookupWitnessN8::from_op_lookup(&bad_ids, &store).is_none());
        // Wrong logical length (idx tensor of length 7).
        let mut store7 = Store::new();
        let table7: Vec<Goldilocks> =
            (0..N).map(|j| Goldilocks::from_u64((j + 3) as u64)).collect();
        let idx7 = store7.push_idx(vec![3, 0, 7, 2, 5, 1, 6]);
        let tbl7 = store7.push(table7);
        let out7 = store7.push(vec![Goldilocks::ZERO; N]);
        let short = Op::Lookup { idx: idx7, out: out7, table: tbl7 };
        assert!(LookupWitnessN8::from_op_lookup(&short, &store7).is_none());
        assert!(dispatch_op_lookup_n8(&w, &short, &store7).is_none());
        // An index >= 8.
        let mut store_bad = Store::new();
        let table_bad: Vec<Goldilocks> =
            (0..N).map(|j| Goldilocks::from_u64((j + 3) as u64)).collect();
        let idx_bad = store_bad.push_idx(vec![3, 0, 9, 2, 5, 1, 6, 4]);
        let tbl_bad = store_bad.push(table_bad);
        let out_bad = store_bad.push(vec![Goldilocks::ZERO; N]);
        let bad_idx = Op::Lookup { idx: idx_bad, out: out_bad, table: tbl_bad };
        assert!(LookupWitnessN8::from_op_lookup(&bad_idx, &store_bad).is_none());
        // A table tensor of length 7.
        let mut store_t7 = Store::new();
        let table_t7: Vec<Goldilocks> =
            (0..N - 1).map(|j| Goldilocks::from_u64((j + 3) as u64)).collect();
        let idx_t7 = store_t7.push_idx(vec![3, 0, 7, 2, 5, 1, 6, 4]);
        let tbl_t7 = store_t7.push(table_t7);
        let out_t7 = store_t7.push(vec![Goldilocks::ZERO; N]);
        let short_tbl = Op::Lookup { idx: idx_t7, out: out_t7, table: tbl_t7 };
        assert!(LookupWitnessN8::from_op_lookup(&short_tbl, &store_t7).is_none());
    }

    /// RAII temp file (unique name per construction), removed on drop.
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
            // Bounded retry over fresh atomic-counter paths: create_new(true)
            // fails with AlreadyExists if anything (including a symlink) is
            // already at the path, so this never follows or truncates a
            // pre-existing file, and the guard only ever removes a path it
            // created.
            for _ in 0..16 {
                let n = TMP_N.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "zkie_direct_lookup_mmap_{}_{}_{}.bin",
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

    /// Mmap-backed output/table metadata is validated BEFORE materialize:
    /// a declared len 9 backed by 8 i32 values is rejected, an overflowing
    /// start is rejected, and an exact start 0 / len 8 range on BOTH output
    /// and table succeeds through the real dispatch and verifies.
    #[test]
    fn mmap_metadata_validated_before_materialize() {
        use std::sync::Arc;
        use zkie_core::common::weights_io::WeightMmap;
        let w = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let idxs: Vec<u32> = vec![3, 0, 7, 2, 5, 1, 6, 4];
        // Eight i32 LE values on disk (table 3..=10).
        let table_vals: Vec<i32> = (0..N as i32).map(|j| j + 3).collect();
        let tmp = TempFile::write_i32(&table_vals);
        let mmap = Arc::new(WeightMmap::open(&tmp.0).expect("open mmap"));

        // Declared len 9 backed by 8 -> rejected before materialize.
        let mut store = Store::new();
        let idx_id = store.push_idx(idxs.clone());
        let tbl_id = store.push_mmap(mmap.clone(), 0, 9);
        let out_id = store.push(vec![Goldilocks::ZERO; N]);
        let op = Op::Lookup { idx: idx_id, out: out_id, table: tbl_id };
        assert!(LookupWitnessN8::from_op_lookup(&op, &store).is_none());

        // start = usize::MAX -> checked_add overflow -> rejected.
        let mut store = Store::new();
        let idx_id = store.push_idx(idxs.clone());
        let tbl_id = store.push_mmap(mmap.clone(), usize::MAX, N);
        let out_id = store.push(vec![Goldilocks::ZERO; N]);
        let op = Op::Lookup { idx: idx_id, out: out_id, table: tbl_id };
        assert!(LookupWitnessN8::from_op_lookup(&op, &store).is_none());

        // The same metadata check applies to the OUTPUT tensor slot.
        let mut store = Store::new();
        let idx_id = store.push_idx(idxs.clone());
        let tbl_id = store.push_mmap(mmap.clone(), 0, N);
        let out_id = store.push_mmap(mmap.clone(), 0, 9);
        let op = Op::Lookup { idx: idx_id, out: out_id, table: tbl_id };
        assert!(LookupWitnessN8::from_op_lookup(&op, &store).is_none());

        // Valid exact start 0 / len 8 for BOTH output and table succeeds
        // through the real Op dispatch and verifies.
        let out_vals: Vec<i32> = idxs.iter().map(|&i| table_vals[i as usize]).collect();
        let tmp_out = TempFile::write_i32(&out_vals);
        let mmap_out = Arc::new(WeightMmap::open(&tmp_out.0).expect("open mmap"));
        let mut store = Store::new();
        let idx_id = store.push_idx(idxs);
        let tbl_id = store.push_mmap(mmap.clone(), 0, N);
        let out_id = store.push_mmap(mmap_out, 0, N);
        let op = Op::Lookup { idx: idx_id, out: out_id, table: tbl_id };
        let (stmt, proof) = dispatch_op_lookup_n8(&w, &op, &store).expect("dispatch");
        drop(store);
        drop(op);
        assert!(proof.verify_with_whir(&w, &stmt));
    }
}
