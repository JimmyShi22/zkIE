//! Environment-gated timing and byte counters for the formal prove path.
//!
//! The instrumentation is inert unless `ZKIE_PROFILE_HIDDEN` is present and
//! not `0`. A scope records wall time, the table bytes read through the asset
//! row source during the scope, the explicit bytes attached to the scope, and
//! the process resident-set/thread counts observed at the scope end. Nested
//! scopes overlap on purpose: a parent covers its children.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the hidden-path profile is enabled for this process.
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| match std::env::var_os("ZKIE_PROFILE_HIDDEN") {
        Some(value) => value != "0",
        None => false,
    })
}

static TABLE_READ_BYTES: AtomicU64 = AtomicU64::new(0);

/// Adds asset bytes read by one table row source call.
pub fn add_table_bytes(bytes: u64) {
    if enabled() {
        TABLE_READ_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

fn table_read_bytes() -> u64 {
    TABLE_READ_BYTES.load(Ordering::Relaxed)
}

#[derive(Clone, Debug)]
struct Record {
    name: &'static str,
    wall_ns: u128,
    calls: u64,
    bytes: u64,
    table_bytes: u64,
    rss_start_kb: u64,
    rss_end_kb: u64,
    hwm_kb: u64,
    threads: u64,
}

static RECORDS: OnceLock<Mutex<Vec<Record>>> = OnceLock::new();

fn records() -> &'static Mutex<Vec<Record>> {
    RECORDS.get_or_init(|| Mutex::new(Vec::new()))
}

struct Inner {
    name: &'static str,
    start: Instant,
    table_start: u64,
    bytes: u64,
    rss_start_kb: u64,
    with_status: bool,
}

/// One measured scope; records on drop when the profile is enabled.
pub struct Scope {
    inner: Option<Inner>,
}

impl Scope {
    /// Starts a scope that samples the process resident set at both ends.
    pub fn new(name: &'static str) -> Self {
        if !enabled() {
            return Self { inner: None };
        }
        let (rss_start_kb, _, _) = vm_status();
        Self {
            inner: Some(Inner {
                name,
                start: Instant::now(),
                table_start: table_read_bytes(),
                bytes: 0,
                rss_start_kb,
                with_status: true,
            }),
        }
    }

    /// Starts a scope without the `/proc` resident-set sampling. Intended for
    /// hot, short-lived call sites where the status read would be measurable.
    pub fn light(name: &'static str) -> Self {
        if !enabled() {
            return Self { inner: None };
        }
        Self {
            inner: Some(Inner {
                name,
                start: Instant::now(),
                table_start: table_read_bytes(),
                bytes: 0,
                rss_start_kb: 0,
                with_status: false,
            }),
        }
    }

    /// Attaches a byte count (for example the encoded proof size) to the scope.
    pub fn add_bytes(&mut self, bytes: u64) {
        if let Some(inner) = &mut self.inner {
            inner.bytes = inner.bytes.saturating_add(bytes);
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let wall_ns = inner.start.elapsed().as_nanos();
        let table_bytes = table_read_bytes().saturating_sub(inner.table_start);
        let (rss_end_kb, hwm_kb, threads) = if inner.with_status {
            let (rss, hwm, threads) = vm_status();
            (rss, hwm, threads)
        } else {
            (0, 0, 0)
        };
        let record = Record {
            name: inner.name,
            wall_ns,
            calls: 1,
            bytes: inner.bytes,
            table_bytes,
            rss_start_kb: inner.rss_start_kb,
            rss_end_kb,
            hwm_kb,
            threads,
        };
        if let Ok(mut guard) = records().lock() {
            guard.push(record);
        }
    }
}

#[cfg(target_os = "linux")]
fn vm_status() -> (u64, u64, u64) {
    let mut rss = 0u64;
    let mut hwm = 0u64;
    let mut threads = 0u64;
    if let Ok(text) = std::fs::read_to_string("/proc/self/status") {
        for line in text.lines() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("VmRSS:") => {
                    rss = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                Some("VmHWM:") => {
                    hwm = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                Some("Threads:") => {
                    threads = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                _ => {}
            }
        }
    }
    (rss, hwm, threads)
}

#[cfg(not(target_os = "linux"))]
fn vm_status() -> (u64, u64, u64) {
    (0, 0, 0)
}

/// Drops every recorded scope and resets the table-read counter. Inert when
/// the profile is disabled.
pub fn reset() {
    if !enabled() {
        return;
    }
    if let Ok(mut guard) = records().lock() {
        guard.clear();
    }
    TABLE_READ_BYTES.store(0, Ordering::Relaxed);
}

/// Prints every aggregated scope. Inert when the profile is disabled.
pub fn dump() {
    if !enabled() {
        return;
    }
    let Ok(guard) = records().lock() else {
        return;
    };
    let mut aggregated: Vec<Record> = Vec::new();
    for record in guard.iter() {
        if let Some(slot) = aggregated.iter_mut().find(|slot| slot.name == record.name) {
            slot.wall_ns = slot.wall_ns.saturating_add(record.wall_ns);
            slot.calls = slot.calls.saturating_add(record.calls);
            slot.bytes = slot.bytes.saturating_add(record.bytes);
            slot.table_bytes = slot.table_bytes.saturating_add(record.table_bytes);
            slot.rss_start_kb = slot.rss_start_kb.min(record.rss_start_kb);
            slot.rss_end_kb = slot.rss_end_kb.max(record.rss_end_kb);
            slot.hwm_kb = slot.hwm_kb.max(record.hwm_kb);
            slot.threads = slot.threads.max(record.threads);
        } else {
            aggregated.push(record.clone());
        }
    }
    println!("hidden_profile_begin");
    for slot in &aggregated {
        println!(
            "hidden_profile_segment name={} wall_ms={:.3} calls={} bytes={} table_bytes={} rss_start_kb={} rss_end_kb={} hwm_kb={} threads={}",
            slot.name,
            slot.wall_ns as f64 / 1_000_000.0,
            slot.calls,
            slot.bytes,
            slot.table_bytes,
            slot.rss_start_kb,
            slot.rss_end_kb,
            slot.hwm_kb,
            slot.threads,
        );
    }
    println!("hidden_profile_end");
}
