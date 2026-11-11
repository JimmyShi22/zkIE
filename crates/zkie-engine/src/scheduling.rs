//! Order-preserving bounded Rayon scheduling over independent work units.
//!
//! Only the scheduling changes: every unit runs the same exact field
//! arithmetic as the serial path, results are collected in input order, and a
//! batch width bounds how many units hold their working set at the same time
//! so peak memory does not grow with the worker count.

use rayon::prelude::*;
use zkie_core::error::Result;

/// The default bound on independent units that may hold their working set at
/// the same time.
pub const DEFAULT_UNITS_IN_FLIGHT: usize = 4;

/// Applies `f` to every item and returns the results in input order. Items are
/// processed in batches of at most `in_flight`, so at most that many units are
/// in flight and the result order never depends on the worker count.
pub fn try_map_bounded<T, R, F>(items: &[T], in_flight: usize, f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R> + Sync + Send,
{
    let width = in_flight.max(1).min(items.len().max(1));
    if width == 1 || items.len() <= 1 {
        return items.iter().map(f).collect();
    }
    let mut out = Vec::with_capacity(items.len());
    for batch in items.chunks(width) {
        let part: Result<Vec<R>> = batch.par_iter().map(&f).collect();
        out.extend(part?);
    }
    Ok(out)
}

/// Applies `f` to every index in `0..len` and returns the results in index
/// order under the same bounded in-flight rule as [`try_map_bounded`].
pub fn try_map_range_bounded<R, F>(len: usize, in_flight: usize, f: F) -> Result<Vec<R>>
where
    R: Send,
    F: Fn(usize) -> Result<R> + Sync + Send,
{
    let width = in_flight.max(1).min(len.max(1));
    if width == 1 || len <= 1 {
        return (0..len).map(f).collect();
    }
    let mut out = Vec::with_capacity(len);
    let mut start = 0usize;
    while start < len {
        let end = (start + width).min(len);
        let part: Result<Vec<R>> = (start..end).into_par_iter().map(&f).collect();
        out.extend(part?);
        start = end;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_map_preserves_order_and_errors_across_widths() {
        let items: Vec<u64> = (0..37).collect();
        let expected: Vec<u64> = items.iter().map(|item| item * 3 + 1).collect();
        for threads in [1usize, 2, 64] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for in_flight in [1usize, 2, 4, 64] {
                let got = pool
                    .install(|| try_map_bounded(&items, in_flight, |item| Ok(item * 3 + 1)).unwrap());
                assert_eq!(got, expected, "threads {threads} width {in_flight}");
            }
        }
        let failed = try_map_bounded(&items, 2, |item| {
            if *item == 19 {
                Err(zkie_core::error::Error::ShapeMismatch)
            } else {
                Ok(*item)
            }
        });
        assert!(failed.is_err());
        assert!(try_map_range_bounded(0, 4, |index| Ok(index)).unwrap().is_empty());
    }

    #[test]
    fn range_map_preserves_order_and_errors_across_widths() {
        let expected: Vec<usize> = (0..37).map(|index| index * 5 + 2).collect();
        for threads in [1usize, 2, 64] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for in_flight in [1usize, 2, 4, 64] {
                let got = pool.install(|| {
                    try_map_range_bounded(37, in_flight, |index| Ok(index * 5 + 2)).unwrap()
                });
                assert_eq!(got, expected, "threads {threads} width {in_flight}");
            }
        }
        let failed = try_map_range_bounded(37, 4, |index| {
            if index == 19 {
                Err(zkie_core::error::Error::ShapeMismatch)
            } else {
                Ok(index)
            }
        });
        assert!(failed.is_err());
        assert_eq!(try_map_range_bounded(1, 64, |index| Ok(index)).unwrap(), vec![0]);
    }
}
