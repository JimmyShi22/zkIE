//! pcs-comparison: standalone benchmark of two PCS constructions on the same
//! integer-valued MLE tables. See README.md for scope and limitations.
pub mod kzg;
pub mod tables;
pub mod whir;

#[cfg(test)]
mod tests {
    /// Both schemes MUST feed on the identical shared integer generator, so
    /// their operation counts are comparable (fields still differ).
    #[test]
    fn shared_int_tables_match_between_schemes() {
        for n in [6usize, 12] {
            for i in 0..2 {
                let k = crate::kzg::table_values(n, i);
                let w = crate::whir::table_values(n, i);
                let t = crate::tables::values(n, i);
                assert_eq!(k, t, "kzg table diverged from shared generator");
                assert_eq!(w, t, "whir table diverged from shared generator");
                assert!(t.iter().all(|&v| v < 100));
                assert_eq!(t.len(), 1usize << n);
            }
        }
    }
}
