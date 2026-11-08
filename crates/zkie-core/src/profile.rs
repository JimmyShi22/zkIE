//! Opt-in operator and kernel timing counters.
//!
//! Profiling is disabled unless `ZKIE_PROFILE=1` is set before the process
//! starts. When disabled, `Scope` construction only checks a cached flag.
//! Counters live per process; stages are separate processes, so no reset is
//! needed between stages.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

pub const OP_SLOTS: usize = 64;
pub const KERNEL_SLOTS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Kernel {
    Sha256 = 0,
    Transcript = 1,
    Merkle = 2,
    Matmul = 3,
    Lookup = 5,
    Rs = 6,
    Dft = 7,
    Fold = 8,
}

impl Kernel {
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
            Self::Transcript => "transcript",
            Self::Merkle => "merkle",
            Self::Matmul => "matmul",
            Self::Lookup => "lookup",
            Self::Rs => "rs",
            Self::Dft => "dft",
            Self::Fold => "fold",
        }
    }
}

static ENABLED: OnceLock<bool> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("ZKIE_PROFILE")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

fn op_nanos() -> &'static [AtomicU64; OP_SLOTS] {
    static SLOTS: OnceLock<[AtomicU64; OP_SLOTS]> = OnceLock::new();
    SLOTS.get_or_init(|| std::array::from_fn(|_| AtomicU64::new(0)))
}

fn op_calls() -> &'static [AtomicU64; OP_SLOTS] {
    static SLOTS: OnceLock<[AtomicU64; OP_SLOTS]> = OnceLock::new();
    SLOTS.get_or_init(|| std::array::from_fn(|_| AtomicU64::new(0)))
}

fn kernel_nanos() -> &'static [AtomicU64; KERNEL_SLOTS] {
    static SLOTS: OnceLock<[AtomicU64; KERNEL_SLOTS]> = OnceLock::new();
    SLOTS.get_or_init(|| std::array::from_fn(|_| AtomicU64::new(0)))
}

fn kernel_calls() -> &'static [AtomicU64; KERNEL_SLOTS] {
    static SLOTS: OnceLock<[AtomicU64; KERNEL_SLOTS]> = OnceLock::new();
    SLOTS.get_or_init(|| std::array::from_fn(|_| AtomicU64::new(0)))
}

fn record_op(slot: usize, nanos: u64) {
    if slot >= OP_SLOTS {
        return;
    }
    let _ = op_nanos()[slot].fetch_add(nanos, Ordering::Relaxed);
    let _ = op_calls()[slot].fetch_add(1, Ordering::Relaxed);
}

fn record_kernel(slot: usize, nanos: u64) {
    if slot >= KERNEL_SLOTS {
        return;
    }
    let _ = kernel_nanos()[slot].fetch_add(nanos, Ordering::Relaxed);
    let _ = kernel_calls()[slot].fetch_add(1, Ordering::Relaxed);
}

/// Times one operator or kernel call. The scope is a no-op unless profiling is
/// enabled. Nested scopes are allowed; wall times are inclusive.
pub struct Scope {
    operator: Option<usize>,
    kernel: Option<usize>,
    start: Option<Instant>,
}

impl Scope {
    pub fn operator(opcode: u16) -> Self {
        if !enabled() {
            return Self {
                operator: None,
                kernel: None,
                start: None,
            };
        }
        Self {
            operator: Some(opcode as usize),
            kernel: None,
            start: Some(Instant::now()),
        }
    }

    pub fn kernel(kernel: Kernel) -> Self {
        if !enabled() {
            return Self {
                operator: None,
                kernel: None,
                start: None,
            };
        }
        Self {
            operator: None,
            kernel: Some(kernel.index()),
            start: Some(Instant::now()),
        }
    }

    pub fn is_enabled() -> bool {
        enabled()
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(start) = self.start else {
            return;
        };
        let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if let Some(slot) = self.operator {
            record_op(slot, nanos);
        } else if let Some(slot) = self.kernel {
            record_kernel(slot, nanos);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub slot: u16,
    pub name: String,
    pub wall_nanos: u64,
    pub calls: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub operators: Vec<Entry>,
    pub kernels: Vec<Entry>,
}

pub fn snapshot() -> Snapshot {
    let mut operators = Vec::new();
    for slot in 0..OP_SLOTS {
        let calls = op_calls()[slot].load(Ordering::Relaxed);
        if calls == 0 {
            continue;
        }
        operators.push(Entry {
            slot: slot as u16,
            name: String::new(),
            wall_nanos: op_nanos()[slot].load(Ordering::Relaxed),
            calls,
        });
    }
    let mut kernels = Vec::new();
    for slot in 0..=Kernel::Fold.index() {
        let calls = kernel_calls()[slot].load(Ordering::Relaxed);
        let name = match slot {
            0 => Kernel::Sha256.name(),
            1 => Kernel::Transcript.name(),
            2 => Kernel::Merkle.name(),
            3 => Kernel::Matmul.name(),
            5 => Kernel::Lookup.name(),
            6 => Kernel::Rs.name(),
            7 => Kernel::Dft.name(),
            8 => Kernel::Fold.name(),
            _ => "unknown",
        };
        kernels.push(Entry {
            slot: slot as u16,
            name: name.to_owned(),
            wall_nanos: kernel_nanos()[slot].load(Ordering::Relaxed),
            calls,
        });
    }
    Snapshot {
        operators,
        kernels,
    }
}

/// Clear all counters. Intended for tests; stages are separate processes.
pub fn reset() {
    for slot in 0..OP_SLOTS {
        op_nanos()[slot].store(0, Ordering::Relaxed);
        op_calls()[slot].store(0, Ordering::Relaxed);
    }
    for slot in 0..KERNEL_SLOTS {
        kernel_nanos()[slot].store(0, Ordering::Relaxed);
        kernel_calls()[slot].store(0, Ordering::Relaxed);
    }
}

/// Force-enable profiling in tests without depending on environment timing.
pub fn force_enable() {
    let _ = ENABLED.set(true);
}
