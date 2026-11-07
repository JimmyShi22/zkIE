//! Private-ledger recovery contracts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::error::Error;
use zkie_core::session::{
    BudgetItem, BudgetReservation, ExposureRecord, LeaseOutcome, ProverStateStore, StateLease,
};
use zkie_core::types::Digest;
use zkie_engine::pcs::state::FileProverStateStore;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-state-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn digest(seed: u8) -> Digest {
    Digest([seed; 32])
}

fn item(kind: &str, amount: u64) -> BudgetItem {
    BudgetItem {
        kind: kind.to_owned(),
        amount,
    }
}

fn reservation(
    commitment: u8,
    leaf: u64,
    generation: u64,
    items: Vec<BudgetItem>,
) -> BudgetReservation {
    BudgetReservation {
        commitment: digest(commitment),
        leaf,
        generation,
        run_id: "run-1".into(),
        items,
    }
}

fn open_with_budget(dir: &Path, budget: Vec<BudgetItem>) -> FileProverStateStore {
    let mut store = FileProverStateStore::open(dir).unwrap();
    store.seed_budget(&budget).unwrap();
    store
}

#[test]
fn process_lock_is_exclusive_and_stale_locks_are_recovered() {
    let temp = TempDir::new();
    let first = open_with_budget(temp.path(), vec![item("field", 10)]);
    assert!(matches!(
        FileProverStateStore::open(temp.path()),
        Err(Error::StateLocked)
    ));
    drop(first);
    assert!(FileProverStateStore::open(temp.path()).is_ok());

    fs::write(temp.path().join("state.lock"), "4294967295\n").unwrap();
    assert!(FileProverStateStore::open(temp.path()).is_ok());
}

#[test]
fn reservations_and_exposure_survive_reopen_without_refund() {
    let temp = TempDir::new();
    let mut store = open_with_budget(temp.path(), vec![item("field", 10)]);
    let lease = store
        .reserve(&reservation(1, 0, 0, vec![item("field", 4)]))
        .unwrap();
    store
        .mark_exposed(&lease, &ExposureRecord {
            items: vec![item("field", 3)],
        })
        .unwrap();
    drop(store);

    let mut reopened = FileProverStateStore::open(temp.path()).unwrap();
    assert_eq!(reopened.remaining("field"), 6);
    assert!(reopened.was_exposed(lease.id()));
    let second = reopened
        .reserve(&reservation(2, 1, 0, vec![item("field", 2)]))
        .unwrap();
    drop(reopened);

    let reopened = FileProverStateStore::open(temp.path()).unwrap();
    assert_eq!(reopened.remaining("field"), 4);
    assert!(reopened.was_exposed(lease.id()));
    assert!(!reopened.was_exposed(second.id()));
}

#[test]
fn crashed_lease_keeps_consumed_budget_and_generation_is_monotonic() {
    let temp = TempDir::new();
    let mut store = open_with_budget(temp.path(), vec![item("field", 10)]);
    let _lease = store
        .reserve(&reservation(1, 0, 5, vec![item("field", 6)]))
        .unwrap();
    drop(store);

    let mut reopened = FileProverStateStore::open(temp.path()).unwrap();
    assert_eq!(reopened.remaining("field"), 4);
    assert!(matches!(
        reopened.reserve(&reservation(2, 0, 4, vec![item("field", 1)])),
        Err(Error::StateRollbackSuspected)
    ));
    let lease = reopened
        .reserve(&reservation(3, 0, 5, vec![item("field", 3)]))
        .unwrap();
    assert_eq!(reopened.remaining("field"), 1);
    assert!(matches!(
        reopened.reserve(&reservation(4, 2, 0, vec![item("field", 2)])),
        Err(Error::LeafBudgetExhausted)
    ));
    assert!(matches!(
        reopened.mark_exposed(&lease, &ExposureRecord {
            items: vec![item("field", 4)],
        }),
        Err(Error::LeafBudgetExhausted)
    ));
}

#[test]
fn leases_are_one_shot_and_bound_to_their_commitment() {
    let temp = TempDir::new();
    let mut store = open_with_budget(temp.path(), vec![item("field", 8)]);
    let lease = store
        .reserve(&reservation(1, 0, 0, vec![item("field", 4)]))
        .unwrap();
    let forged = StateLease::new(lease.id(), digest(9), 0, 0, "run-1");
    assert!(matches!(
        store.mark_exposed(&forged, &ExposureRecord {
            items: vec![item("field", 1)],
        }),
        Err(Error::StateRollbackSuspected)
    ));

    store.finish(lease, LeaseOutcome::Completed).unwrap();
    let replay = StateLease::new(1, digest(1), 0, 0, "run-1");
    assert!(matches!(
        store.finish(replay, LeaseOutcome::Failed),
        Err(Error::InvalidSessionState)
    ));
}

#[test]
fn ledger_head_ahead_of_state_is_rejected() {
    let temp = TempDir::new();
    let store = open_with_budget(temp.path(), vec![item("field", 4)]);
    let before = store.ledger_generation();
    drop(store);
    fs::write(
        temp.path().join("ledger.head"),
        format!("{}\n", before + 100),
    )
    .unwrap();
    assert!(matches!(
        FileProverStateStore::open(temp.path()),
        Err(Error::StateRollbackSuspected)
    ));
}
