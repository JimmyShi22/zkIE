//! The LIN connection rows: (Eval), (Fold) and (Dot).
//!
//! LIN checks no proof by itself: it leaves the rows that the table's fold and
//! the proximity tests check. (Eval) ties the request's W to its claim's point
//! and masks; (Fold) ties Omega to the folded row; (Dot) ties the row weights
//! against Omega to W at the shared point. This module keeps those three
//! relations executable, with the row cut at the matrix's rows: a form that ran
//! on would read entries that mean nothing.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

fn eq1(coordinate: Fp3, bit: bool) -> Fp3 {
    if bit {
        coordinate
    } else {
        Fp3::ONE.sub(&coordinate)
    }
}

/// The multilinear extension of a vector at a point: bit `j` of the index
/// pairs with `point[j]`.
pub fn lin_eval(values: &[Fp3], point: &[Fp3]) -> Result<Fp3> {
    if point.len() >= 32 || values.len() != (1usize << point.len()) {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (index, value) in values.iter().enumerate() {
        let mut factor = Fp3::ONE;
        for (bit, coordinate) in point.iter().enumerate() {
            factor = factor.mul(&eq1(*coordinate, (index >> bit) & 1 == 1));
        }
        sum = sum.add(&factor.mul(value));
    }
    Ok(sum)
}

/// The equality table of a point over the rows below `matrix_rows`.
fn eq_below(point: &[Fp3], matrix_rows: u64) -> Result<Vec<Fp3>> {
    let size = 1usize << point.len();
    let mut table = vec![Fp3::ZERO; size];
    for (index, slot) in table.iter_mut().enumerate() {
        if (index as u64) >= matrix_rows {
            break;
        }
        let mut factor = Fp3::ONE;
        for (bit, coordinate) in point.iter().enumerate() {
            factor = factor.mul(&eq1(*coordinate, (index >> bit) & 1 == 1));
        }
        *slot = factor;
    }
    Ok(table)
}

/// Checks (Dot): the row weights against Omega, cut at the matrix's rows,
/// equal W at the shared point.
pub fn lin_dot(
    weights: &[Fp3],
    omega: &[Fp3],
    matrix_rows: u64,
    psi_star: &[Fp3],
    w: &[Fp3],
) -> Result<()> {
    if weights.len() != omega.len() || weights.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let cut = usize::try_from(matrix_rows.min(weights.len() as u64))
        .map_err(|_| Error::ResourceLimit)?;
    if cut == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut dot = Fp3::ZERO;
    for index in 0..cut {
        dot = dot.add(&weights[index].mul(&omega[index]));
    }
    if dot != lin_eval(w, psi_star)? {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

/// Checks (Fold): Omega at `zeta` equals the folded row's inner product with
/// the two equality tables, cut at the matrix's rows. `u` is the folded row,
/// flat by row bits then the low column bits.
pub fn lin_fold(
    psi_lo: &[Fp3],
    zeta: &[Fp3],
    u: &[Fp3],
    omega: &[Fp3],
    matrix_rows: u64,
) -> Result<()> {
    if psi_lo.is_empty() || zeta.is_empty() || psi_lo.len() + zeta.len() >= 32 {
        return Err(Error::InvalidConfig);
    }
    let rows = 1usize << zeta.len();
    let low = 1usize << psi_lo.len();
    if omega.len() != rows || u.len() != rows.checked_mul(low).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    let cut = usize::try_from(matrix_rows.min(rows as u64)).map_err(|_| Error::ResourceLimit)?;
    let eq_zeta = eq_below(zeta, matrix_rows)?;
    let eq_psi = eq_below(psi_lo, low as u64)?;
    let mut folded = Fp3::ZERO;
    for row in 0..cut {
        let mut inner = Fp3::ZERO;
        for low_index in 0..low {
            inner = inner.add(&eq_psi[low_index].mul(&u[row * low + low_index]));
        }
        folded = folded.add(&eq_zeta[row].mul(&inner));
    }
    let mut evaluated = Fp3::ZERO;
    for row in 0..cut {
        evaluated = evaluated.add(&eq_zeta[row].mul(&omega[row]));
    }
    if evaluated != folded {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::projection_claim::PlaceRows;

    fn f(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    #[test]
    fn the_evaluation_pairs_bits_low_first() {
        let w = [f(1), f(2), f(3), f(4)];
        assert_eq!(lin_eval(&w, &[f(0), f(0)]).unwrap(), f(1));
        assert_eq!(lin_eval(&w, &[f(1), f(0)]).unwrap(), f(2));
        assert_eq!(lin_eval(&w, &[f(0), f(1)]).unwrap(), f(3));
        assert_eq!(lin_eval(&w, &[f(1), f(1)]).unwrap(), f(4));
        assert!(lin_eval(&w, &[f(0)]).is_err());
    }

    #[test]
    fn the_dot_row_binds_the_weights_omega_and_the_shared_point() {
        let weights = [f(2), f(3)];
        let psi_star = [f(4), f(5)];
        let w = [f(1), f(2), f(3), f(4)];
        let value = lin_eval(&w, &psi_star).unwrap();
        // One row of Omega carries the value; the cut keeps the other out.
        let omega = [value.mul(&f(2).inverse().unwrap()), f(0)];
        lin_dot(&weights, &omega, 1, &psi_star, &w).unwrap();
        let mut changed = omega;
        changed[0] = changed[0].add(&Fp3::ONE);
        assert!(lin_dot(&weights, &changed, 1, &psi_star, &w).is_err());
        // Without the cut the second row would be read.
        assert!(lin_dot(&weights, &[f(1), f(1)], 2, &psi_star, &w).is_err());
    }

    #[test]
    fn the_eval_row_matches_the_pending_claim() {
        use crate::proof::hidden_member::{
            pending_value, stretch_weights, MaskTerm, PendingRow,
        };
        // Two matrix rows of four columns; the row weights combine them into
        // W = g^T M, and the pending claim reads the same combination at the
        // column point.
        let matrix = vec![
            vec![f(1), f(2), f(3), f(4)],
            vec![f(5), f(6), f(7), f(8)],
        ];
        let g = [f(2), f(3)];
        let psi = [f(4), f(5)];
        let mut w = vec![Fp3::ZERO; 4];
        for (row, weight) in g.iter().enumerate() {
            for (slot, value) in w.iter_mut().zip(&matrix[row]) {
                *slot = slot.add(&weight.mul(value));
            }
        }
        let eval = lin_eval(&w, &psi).unwrap();
        let pending = PendingRow {
            weights: stretch_weights(1, 0, &g).unwrap(),
            psi: psi.to_vec(),
            target: eval,
            masks: Vec::<MaskTerm>::new(),
        };
        assert_eq!(
            pending_value(&pending, &PlaceRows::new(&matrix), &|_, _| Ok(Fp3::ZERO)).unwrap(),
            eval
        );
        // Another weight or matrix row moves both sides together only when
        // the pending target is recomputed; the fixed target refuses it.
        let mut changed = matrix.clone();
        changed[0][0] = changed[0][0].add(&Fp3::ONE);
        assert_ne!(
            pending_value(&pending, &PlaceRows::new(&changed), &|_, _| Ok(Fp3::ZERO)).unwrap(),
            eval
        );
    }

    #[test]
    fn the_fold_row_binds_omega_to_the_folded_row() {
        let psi_lo = [f(2)];
        let zeta = [f(3)];
        let u = [f(1), f(2), f(3), f(4)];
        let eq_psi = [Fp3::ONE.sub(&f(2)), f(2)];
        let eq_zeta = [Fp3::ONE.sub(&f(3)), f(3)];
        let row0 = eq_psi[0].mul(&u[0]).add(&eq_psi[1].mul(&u[1]));
        let row1 = eq_psi[0].mul(&u[2]).add(&eq_psi[1].mul(&u[3]));
        let omega = [row0, row1];
        lin_fold(&psi_lo, &zeta, &u, &omega, 2).unwrap();
        // With the cut at one row, only the first row is read.
        lin_fold(&psi_lo, &zeta, &u, &omega, 1).unwrap();
        let mut changed = omega;
        changed[1] = changed[1].add(&Fp3::ONE);
        assert!(lin_fold(&psi_lo, &zeta, &u, &changed, 2).is_err());
        let _ = eq_zeta;
    }
}
