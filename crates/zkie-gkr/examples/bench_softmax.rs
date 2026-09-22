//! Benchmarks the logUp-based exp lookup + softmax normalization.

use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::softmax;

fn main() {
    // 1024-entry table, matching the current Halo2 non-linear lookup tables.
    let table_size = 1024usize;
    let n = 4096usize;
    let mut rng = XorShift64::new(0xbeef);

    let table: Vec<Goldilocks> = (0..table_size).map(|_| rng.field()).collect();
    let indices: Vec<u32> = (0..n).map(|_| (rng.next_u64() % table_size as u64) as u32).collect();

    let t0 = std::time::Instant::now();
    let (y, proof) = softmax::softmax(&indices, &table, rng.field(), rng.field());
    let secs = t0.elapsed().as_secs_f64();
    assert!(softmax::verify(&indices, &table, &y, &proof));

    // Dominant cost: one field inversion per lookup (Fermat pow). Batch
    // inversion turns that into 1 inversion + O(n) multiplications.
    println!("softmax n={n} table_size={table_size}");
    println!("lookup_terms={n} (one inversion per term, batchable)");
    println!("wall_secs={secs:.4}");
}
