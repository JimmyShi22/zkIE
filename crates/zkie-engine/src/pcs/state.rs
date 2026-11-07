//! Private prover-state accounting.
//!
//! Leases are one-shot and non-cloneable. Reservations are checked before they
//! are deducted. Failed, cancelled or crashed leases are not refunded: the
//! exposure may already have happened.

use std::collections::BTreeMap;

use zkie_core::error::{Error, Result};
use zkie_core::session::{
    BudgetItem, BudgetReservation, ExposureRecord, LeaseOutcome, ProverStateStore, StateLease,
};
use zkie_core::types::Digest;

/// Mask domains. Domain separation is part of the hiding state, not a worker
/// index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskDomain {
    RandomTail,
    ExposedColumns,
    OodAnswer,
    Request,
    Batch,
    Leaf,
    Round,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskStreamRequest {
    pub domain: MaskDomain,
    pub request_id: u64,
    pub batch_id: u64,
    pub leaf: u64,
    pub round: u32,
}

pub trait MaskSource {
    fn fill_mask(&mut self, request: &MaskStreamRequest, out: &mut [u8]) -> Result<()>;
}

#[derive(Clone, Debug)]
struct LeaseRecord {
    commitment: Digest,
    reserved: BTreeMap<String, u64>,
    exposed: BTreeMap<String, u64>,
    finished: bool,
}

#[derive(Clone, Debug)]
pub struct BudgetedProverStateStore {
    remaining: BTreeMap<String, u64>,
    leases: BTreeMap<u64, LeaseRecord>,
    last_generation: BTreeMap<u64, u64>,
    next_id: u64,
}

impl BudgetedProverStateStore {
    pub fn new(budget: Vec<BudgetItem>) -> Self {
        let mut remaining = BTreeMap::new();
        for item in budget {
            let entry = remaining.entry(item.kind).or_insert(0u64);
            *entry = entry.saturating_add(item.amount);
        }
        Self {
            remaining,
            leases: BTreeMap::new(),
            last_generation: BTreeMap::new(),
            next_id: 0,
        }
    }

    pub fn remaining(&self, kind: &str) -> u64 {
        self.remaining.get(kind).copied().unwrap_or(0)
    }

    pub fn was_exposed(&self, lease_id: u64) -> bool {
        self.leases
            .get(&lease_id)
            .map(|record| !record.exposed.is_empty())
            .unwrap_or(false)
    }
}

impl ProverStateStore for BudgetedProverStateStore {
    fn reserve(&mut self, request: &BudgetReservation) -> Result<StateLease> {
        let mut requested: BTreeMap<String, u64> = BTreeMap::new();
        for item in &request.items {
            let entry = requested.entry(item.kind.clone()).or_insert(0u64);
            *entry = entry
                .checked_add(item.amount)
                .ok_or(Error::ResourceLimit)?;
        }

        for (kind, amount) in &requested {
            let available = self.remaining.get(kind).copied().unwrap_or(0);
            if *amount > available {
                return Err(Error::LeafBudgetExhausted);
            }
        }

        if let Some(previous) = self.last_generation.get(&request.leaf).copied() {
            if request.generation < previous {
                return Err(Error::StateRollbackSuspected);
            }
        }

        for (kind, amount) in &requested {
            if let Some(entry) = self.remaining.get_mut(kind) {
                *entry = entry
                    .checked_sub(*amount)
                    .ok_or(Error::LeafBudgetExhausted)?;
            }
        }
        self.last_generation
            .insert(request.leaf, request.generation);
        self.next_id = self.next_id.checked_add(1).ok_or(Error::ResourceLimit)?;

        let lease = StateLease::new(
            self.next_id,
            request.commitment,
            request.leaf,
            request.generation,
            request.run_id.clone(),
        );
        self.leases.insert(
            lease.id(),
            LeaseRecord {
                commitment: request.commitment,
                reserved: requested,
                exposed: BTreeMap::new(),
                finished: false,
            },
        );
        Ok(lease)
    }

    fn mark_exposed(&mut self, lease: &StateLease, exposure: &ExposureRecord) -> Result<()> {
        let record = self
            .leases
            .get_mut(&lease.id())
            .ok_or(Error::ProverStateMissing)?;
        if record.finished {
            return Err(Error::InvalidSessionState);
        }
        if record.commitment != lease.commitment() {
            return Err(Error::StateRollbackSuspected);
        }
        for item in &exposure.items {
            let reserved = record.reserved.get(&item.kind).copied().unwrap_or(0);
            let exposed = record.exposed.get(&item.kind).copied().unwrap_or(0);
            let next = exposed
                .checked_add(item.amount)
                .ok_or(Error::LeafBudgetExhausted)?;
            if next > reserved {
                return Err(Error::LeafBudgetExhausted);
            }
            record.exposed.insert(item.kind.clone(), next);
        }
        Ok(())
    }

    fn finish(&mut self, lease: StateLease, outcome: LeaseOutcome) -> Result<()> {
        let record = self
            .leases
            .get_mut(&lease.id())
            .ok_or(Error::ProverStateMissing)?;
        if record.finished {
            return Err(Error::InvalidSessionState);
        }
        if record.commitment != lease.commitment() {
            return Err(Error::StateRollbackSuspected);
        }
        record.finished = true;
        // No refund on any outcome: exposure may already have happened.
        let _ = outcome;
        Ok(())
    }
}

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedLease {
    commitment: Digest,
    leaf: u64,
    generation: u64,
    run_id: String,
    reserved: BTreeMap<String, u64>,
    exposed: BTreeMap<String, u64>,
    finished: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PersistedState {
    ledger_generation: u64,
    remaining: BTreeMap<String, u64>,
    leases: BTreeMap<u64, PersistedLease>,
    last_generation: BTreeMap<u64, u64>,
    next_id: u64,
}

/// File-backed private state with a process lock and a monotonic ledger head.
///
/// A stale lock whose PID is no longer alive is recovered. A ledger whose state
/// generation is below the head generation is treated as a rollback. Leases
/// that never finished remain consumed; there is no refund on recovery.
pub struct FileProverStateStore {
    dir: PathBuf,
    lock_path: PathBuf,
    lock: Option<File>,
    state: PersistedState,
}

impl FileProverStateStore {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|_| Error::Io)?;
        let lock_path = dir.join("state.lock");
        let lock = acquire_process_lock(&lock_path)?;
        let state_path = dir.join("ledger.json");
        let head_path = dir.join("ledger.head");

        let state = if state_path.exists() {
            let bytes = fs::read(&state_path).map_err(|_| Error::StateRollbackSuspected)?;
            serde_json::from_slice(&bytes).map_err(|_| Error::StateRollbackSuspected)?
        } else {
            PersistedState::default()
        };
        let head = if head_path.exists() {
            let text = fs::read_to_string(&head_path).map_err(|_| Error::StateRollbackSuspected)?;
            text.trim()
                .parse::<u64>()
                .map_err(|_| Error::StateRollbackSuspected)?
        } else {
            0
        };
        if head > state.ledger_generation {
            return Err(Error::StateRollbackSuspected);
        }

        let mut store = Self {
            dir,
            lock_path,
            lock: Some(lock),
            state,
        };
        if !state_path.exists() || head < store.state.ledger_generation {
            store.persist()?;
        }
        Ok(store)
    }

    pub fn remaining(&self, kind: &str) -> u64 {
        self.state.remaining.get(kind).copied().unwrap_or(0)
    }

    pub fn was_exposed(&self, lease_id: u64) -> bool {
        self.state
            .leases
            .get(&lease_id)
            .map(|record| !record.exposed.is_empty())
            .unwrap_or(false)
    }

    /// Initialize the durable budget when the ledger has no history yet.
    pub fn seed_budget(&mut self, budget: &[BudgetItem]) -> Result<()> {
        if self.state.ledger_generation != 0
            || !self.state.leases.is_empty()
            || self.state.remaining.values().any(|amount| *amount != 0)
        {
            return Err(Error::InvalidSessionState);
        }
        for item in budget {
            let entry = self.state.remaining.entry(item.kind.clone()).or_insert(0);
            *entry = entry.checked_add(item.amount).ok_or(Error::ResourceLimit)?;
        }
        self.mutate()
    }

    pub fn ledger_generation(&self) -> u64 {
        self.state.ledger_generation
    }

    fn mutate(&mut self) -> Result<()> {
        self.state.ledger_generation = self
            .state
            .ledger_generation
            .checked_add(1)
            .ok_or(Error::StateRollbackSuspected)?;
        self.persist()
    }

    fn persist(&mut self) -> Result<()> {
        let state_bytes =
            serde_json::to_vec(&self.state).map_err(|_| Error::InvalidEncoding)?;
        write_atomic(&self.dir.join("ledger.json"), &state_bytes)?;
        let head = format!("{}\n", self.state.ledger_generation);
        write_atomic(&self.dir.join("ledger.head"), head.as_bytes())?;
        sync_dir(&self.dir)
    }
}

impl ProverStateStore for FileProverStateStore {
    fn reserve(&mut self, request: &BudgetReservation) -> Result<StateLease> {
        let mut requested: BTreeMap<String, u64> = BTreeMap::new();
        for item in &request.items {
            let entry = requested.entry(item.kind.clone()).or_insert(0u64);
            *entry = entry
                .checked_add(item.amount)
                .ok_or(Error::ResourceLimit)?;
        }
        for (kind, amount) in &requested {
            if *amount > self.remaining(kind) {
                return Err(Error::LeafBudgetExhausted);
            }
        }
        if let Some(previous) = self.state.last_generation.get(&request.leaf).copied() {
            if request.generation < previous {
                return Err(Error::StateRollbackSuspected);
            }
        }
        for (kind, amount) in &requested {
            if let Some(entry) = self.state.remaining.get_mut(kind) {
                *entry = entry
                    .checked_sub(*amount)
                    .ok_or(Error::LeafBudgetExhausted)?;
            }
        }
        self.state
            .last_generation
            .insert(request.leaf, request.generation);
        self.state.next_id = self
            .state
            .next_id
            .checked_add(1)
            .ok_or(Error::ResourceLimit)?;
        let lease = StateLease::new(
            self.state.next_id,
            request.commitment,
            request.leaf,
            request.generation,
            request.run_id.clone(),
        );
        self.state.leases.insert(
            lease.id(),
            PersistedLease {
                commitment: request.commitment,
                leaf: request.leaf,
                generation: request.generation,
                run_id: request.run_id.clone(),
                reserved: requested,
                exposed: BTreeMap::new(),
                finished: false,
            },
        );
        self.mutate()?;
        Ok(lease)
    }

    fn mark_exposed(&mut self, lease: &StateLease, exposure: &ExposureRecord) -> Result<()> {
        let record = self
            .state
            .leases
            .get_mut(&lease.id())
            .ok_or(Error::ProverStateMissing)?;
        if record.finished {
            return Err(Error::InvalidSessionState);
        }
        if record.commitment != lease.commitment() {
            return Err(Error::StateRollbackSuspected);
        }
        for item in &exposure.items {
            let reserved = record.reserved.get(&item.kind).copied().unwrap_or(0);
            let exposed = record.exposed.get(&item.kind).copied().unwrap_or(0);
            let next = exposed
                .checked_add(item.amount)
                .ok_or(Error::LeafBudgetExhausted)?;
            if next > reserved {
                return Err(Error::LeafBudgetExhausted);
            }
            record.exposed.insert(item.kind.clone(), next);
        }
        self.mutate()
    }

    fn finish(&mut self, lease: StateLease, _outcome: LeaseOutcome) -> Result<()> {
        let record = self
            .state
            .leases
            .get_mut(&lease.id())
            .ok_or(Error::ProverStateMissing)?;
        if record.finished {
            return Err(Error::InvalidSessionState);
        }
        if record.commitment != lease.commitment() {
            return Err(Error::StateRollbackSuspected);
        }
        record.finished = true;
        self.mutate()
    }
}

impl Drop for FileProverStateStore {
    fn drop(&mut self) {
        self.lock.take();
        let _ = fs::remove_file(&self.lock_path);
    }
}

fn acquire_process_lock(path: &Path) -> Result<File> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            writeln!(file, "{}", std::process::id()).map_err(|_| Error::Io)?;
            file.sync_all().map_err(|_| Error::Io)?;
            Ok(file)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let pid = fs::read_to_string(path)
                .ok()
                .and_then(|text| text.trim().parse::<u32>().ok());
            let alive = pid
                .map(|pid| Path::new(&format!("/proc/{pid}")).exists())
                .unwrap_or(false);
            if alive {
                return Err(Error::StateLocked);
            }
            let _ = fs::remove_file(path);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|_| Error::StateLocked)?;
            writeln!(file, "{}", std::process::id()).map_err(|_| Error::Io)?;
            file.sync_all().map_err(|_| Error::Io)?;
            Ok(file)
        }
        Err(_) => Err(Error::Io),
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp).map_err(|_| Error::Io)?;
    file.write_all(bytes).map_err(|_| Error::Io)?;
    file.sync_all().map_err(|_| Error::Io)?;
    fs::rename(&tmp, path).map_err(|_| Error::Io)
}

fn sync_dir(dir: &Path) -> Result<()> {
    let file = File::open(dir).map_err(|_| Error::Io)?;
    file.sync_all().map_err(|_| Error::Io)
}
