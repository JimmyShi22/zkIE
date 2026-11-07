//! Process-level measurements.

use std::fs;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct MonotonicTimer {
    start: Instant,
}

impl MonotonicTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    pub fn elapsed_us(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX)
    }
}

/// Process peak resident set size from Linux `/proc/self/status`.
pub fn vm_hwm_kib() -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let value = rest.trim().trim_end_matches("kB").trim();
            return value.parse().ok();
        }
    }
    None
}

pub fn vm_hwm_bytes() -> Option<u64> {
    vm_hwm_kib().and_then(|kib| kib.checked_mul(1024))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcessMetrics {
    pub stage_wall_ms: u64,
    pub protocol_wall_ms: u64,
    pub vmhwm_kib: Option<u64>,
}
