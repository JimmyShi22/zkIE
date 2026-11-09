//! Isolated candidate for the provider's persistent column count.
//!
//! This is a local, single-machine count-file adapter. It models the required
//! count rules without production wiring: a fixed columns-per-batch plan, an
//! exclusive lock over the live count files, an all-live-table room check
//! before appending, an append-and-recheck of the sampled table, and a durable
//! sync before returning. The lock and directory are trusted same-machine
//! premises. This is not a rollback-proof or cross-machine authority, and a
//! copied directory has an independent count. Those extensions have no
//! counterpart here and are not claimed.
//!
//! A production caller must replace the test-only failure hooks and must decide
//! which live tables are in the set. This module does not select a new leaf,
//! shorten a batch, retry a failed sync, or silently downgrade hiding. The lock
//! is the platform file lock; if that lock cannot be taken the call fails with
//! an I/O error, and there is no thread-mutex fallback. The caller's paths must
//! all be under one trusted directory/lock authority: two root paths that point
//! at one count file under different lock files are not made mutually
//! exclusive here, and arbitrary-directory safety is not claimed. The test-only
//! append/close hooks demonstrate upper-layer error propagation and the absence
//! of a receipt; they are not injected operating-system syscall failures. The
//! removed-count-file test exercises an actual missing-target append open.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::os::unix::io::IntoRawFd;

use zkie_core::error::{Error, Result};

/// The fixed plan for one batch of count files. `columns` is the logical
/// number of columns the plan reserves for every batch; `max_columns` is the
/// table-life limit. The fields are private so a forged plan cannot be built
/// by struct literal; every receiving entry rechecks the values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountPlan {
    columns: u64,
    max_columns: u64,
}

impl CountPlan {
    pub fn new(columns: u64, max_columns: u64) -> Result<Self> {
        let plan = Self {
            columns,
            max_columns,
        };
        plan.check()?;
        Ok(plan)
    }

    pub fn columns(&self) -> u64 {
        self.columns
    }

    pub fn max_columns(&self) -> u64 {
        self.max_columns
    }

    fn check(&self) -> Result<()> {
        if self.columns == 0 || self.max_columns == 0 || self.max_columns < self.columns {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

/// A completed, durable count reservation. There is no cancel/refund method:
/// an abandoned proof keeps its count, as the candidate contract requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountReceipt {
    pub leaf: usize,
    pub added: u64,
    pub after: u64,
}

/// A small set of live count files, one per leaf table. Each file's length is
/// its count in columns. The lock file is under `root`; the count files are
/// supplied by the caller and must already exist.
pub struct CountSet {
    root: PathBuf,
    lock_path: PathBuf,
    counts: Vec<PathBuf>,
    plan: CountPlan,
    #[cfg(test)]
    short_write: bool,
    #[cfg(test)]
    sync_failure: bool,
    #[cfg(test)]
    unlock_failure: bool,
    #[cfg(test)]
    append_failure: bool,
    #[cfg(test)]
    close_failure: bool,
}

impl CountSet {
    pub fn new(root: impl Into<PathBuf>, counts: Vec<PathBuf>, plan: CountPlan) -> Result<Self> {
        plan.check()?;
        let root = root.into();
        if counts.is_empty() || !root.is_dir() {
            return Err(Error::InvalidConfig);
        }
        for path in &counts {
            let metadata = std::fs::symlink_metadata(path).map_err(|_| Error::Io)?;
            if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                return Err(Error::InvalidConfig);
            }
        }
        let lock_path = root.join("count.lock");
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .map_err(|_| Error::Io)?;
        Ok(Self {
            root,
            lock_path,
            counts,
            plan,
            #[cfg(test)]
            short_write: false,
            #[cfg(test)]
            sync_failure: false,
            #[cfg(test)]
            unlock_failure: false,
            #[cfg(test)]
            append_failure: false,
            #[cfg(test)]
            close_failure: false,
        })
    }

    pub fn plan(&self) -> CountPlan {
        self.plan
    }

    pub fn counts(&self) -> Result<Vec<u64>> {
        self.counts.iter().map(|path| read_count(path)).collect()
    }

    /// The startup dry run: take the lock, open every live count file, and
    /// check that every one has room for a complete planned batch. No bytes are
    /// appended. A missing file, lock failure or insufficient room is an error.
    pub fn verify_startup(&self) -> Result<()> {
        let lock = self.open_lock()?;
        lock.lock().map_err(|_| Error::Io)?;
        let result = self.verify_startup_locked();
        let unlock = lock.unlock().map_err(|_| Error::Io);
        match (result, unlock) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn verify_startup_locked(&self) -> Result<()> {
        self.plan.check()?;
        for path in &self.counts {
            check_appendable(path, self.append_failure_for_test())?;
            let count = read_count(path)?;
            let next = count
                .checked_add(self.plan.columns)
                .ok_or(Error::ResourceLimit)?;
            if next > self.plan.max_columns {
                return Err(Error::ResourceLimit);
            }
        }
        Ok(())
    }

    /// Count one batch against leaf `leaf`. The fixed plan columns are checked
    /// against every live count first; the sampled leaf is appended only after
    /// all live tables have room. The append is rechecked, synced once, and
    /// only then is a receipt returned. A rejected or failed call never
    /// shortens, switches leaf, retries the sync, or returns a reservation.
    pub fn begin_batch(&self, leaf: usize) -> Result<CountReceipt> {
        if leaf >= self.counts.len() {
            return Err(Error::InvalidIndex);
        }
        let lock = self.open_lock()?;
        lock.lock().map_err(|_| Error::Io)?;
        let result = self.begin_batch_locked(leaf);
        let unlock_failed = lock.unlock().is_err();
        if result.is_ok() && (unlock_failed || self.unlock_failed_for_test()) {
            return Err(Error::Io);
        }
        result
    }

    fn begin_batch_locked(&self, leaf: usize) -> Result<CountReceipt> {
        self.plan.check()?;
        self.verify_startup_locked()?;
        #[cfg(test)]
        if let Some(milliseconds) = std::env::var("ZKIE_COUNT_CHILD_HOLD_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
        {
            std::thread::sleep(std::time::Duration::from_millis(milliseconds));
        }
        if self.short_write_for_test() {
            let path = &self.counts[leaf];
            let mut file = OpenOptions::new().append(true).open(path).map_err(|_| Error::Io)?;
            let partial = self.plan.columns.saturating_sub(1);
            let _ = write_zeros(&mut file, partial);
            return Err(Error::Io);
        }
        let path = &self.counts[leaf];
        let mut file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|_| Error::Io)?;
        let written = write_zeros(&mut file, self.plan.columns)?;
        if written != self.plan.columns {
            return Err(Error::Io);
        }
        let after = file.metadata().map_err(|_| Error::Io)?.len();
        if after > self.plan.max_columns {
            return Err(Error::ResourceLimit);
        }
        if self.sync_failure_for_test() {
            return Err(Error::Io);
        }
        file.sync_all().map_err(|_| Error::Io)?;
        close_count_file(file)?;
        if self.close_failure_for_test() {
            return Err(Error::Io);
        }
        Ok(CountReceipt {
            leaf,
            added: self.plan.columns,
            after,
        })
    }

    fn open_lock(&self) -> Result<File> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.lock_path)
            .map_err(|_| Error::Io)
    }

    #[cfg(test)]
    pub fn force_short_write(&mut self) {
        self.short_write = true;
    }

    #[cfg(test)]
    pub fn force_sync_failure(&mut self) {
        self.sync_failure = true;
    }

    #[cfg(test)]
    pub fn force_unlock_failure(&mut self) {
        self.unlock_failure = true;
    }

    #[cfg(test)]
    pub fn force_append_failure(&mut self) {
        self.append_failure = true;
    }

    #[cfg(test)]
    pub fn force_close_failure(&mut self) {
        self.close_failure = true;
    }

    #[cfg(test)]
    fn short_write_for_test(&self) -> bool {
        self.short_write
    }

    #[cfg(not(test))]
    fn short_write_for_test(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn sync_failure_for_test(&self) -> bool {
        self.sync_failure
    }

    #[cfg(not(test))]
    fn sync_failure_for_test(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn unlock_failed_for_test(&self) -> bool {
        self.unlock_failure
    }

    #[cfg(not(test))]
    fn unlock_failed_for_test(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn append_failure_for_test(&self) -> bool {
        self.append_failure
    }

    #[cfg(not(test))]
    fn append_failure_for_test(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn close_failure_for_test(&self) -> bool {
        self.close_failure
    }

    #[cfg(not(test))]
    fn close_failure_for_test(&self) -> bool {
        false
    }

    /// The lock file's path is exposed only for tests and diagnostics.
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn read_count(path: &Path) -> Result<u64> {
    let metadata = std::fs::symlink_metadata(path).map_err(|_| Error::Io)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(Error::InvalidConfig);
    }
    Ok(metadata.len())
}

fn check_appendable(path: &Path, injected_failure: bool) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(path)
        .map_err(|_| Error::Io)?;
    if injected_failure {
        return Err(Error::Io);
    }
    let metadata = file.metadata().map_err(|_| Error::Io)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

fn write_zeros(file: &mut File, count: u64) -> Result<u64> {
    let count = usize::try_from(count).map_err(|_| Error::ResourceLimit)?;
    let mut zeros = Vec::new();
    zeros
        .try_reserve_exact(count)
        .map_err(|_| Error::ResourceLimit)?;
    zeros.resize(count, 0);
    match file.write(&zeros) {
        Ok(written) => Ok(written as u64),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Err(Error::Io),
        Err(_) => Err(Error::Io),
    }
}

#[cfg(unix)]
fn close_count_file(file: File) -> Result<()> {
    let descriptor = file.into_raw_fd();
    if unsafe { libc::close(descriptor) } != 0 {
        return Err(Error::Io);
    }
    Ok(())
}

#[cfg(not(unix))]
fn close_count_file(file: File) -> Result<()> {
    drop(file);
    Err(Error::UnsupportedPrimitive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(1);

    fn temp_root(name: &str) -> PathBuf {
        let serial = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zkie-count-{}-{name}-{serial}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn add_leaf(root: &Path, name: &str) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let count = dir.join("count.bin");
        File::create(&count).unwrap();
        count
    }

    fn set_count(path: &Path, count: u64) {
        std::fs::write(path, vec![0u8; count as usize]).unwrap();
    }

    #[test]
    fn all_live_leaves_are_checked_before_append() {
        let root = temp_root("all-live");
        let first = add_leaf(&root, "leaf0");
        let second = add_leaf(&root, "leaf1");
        set_count(&second, 1);
        let set = CountSet::new(
            &root,
            vec![first.clone(), second],
            CountPlan::new(2, 2).unwrap(),
        )
        .unwrap();
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&first).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_full_sampled_leaf_is_not_switched_to_a_roomy_leaf() {
        let root = temp_root("no-switch");
        let first = add_leaf(&root, "leaf0");
        let second = add_leaf(&root, "leaf1");
        set_count(&first, 1);
        let set = CountSet::new(
            &root,
            vec![first.clone(), second.clone()],
            CountPlan::new(2, 2).unwrap(),
        )
        .unwrap();
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&first).unwrap(), 1);
        assert_eq!(read_count(&second).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn append_is_durable_and_restart_preserves_the_count() {
        let root = temp_root("restart");
        let count = add_leaf(&root, "leaf0");
        let plan = CountPlan::new(1, 2).unwrap();
        let set = CountSet::new(&root, vec![count.clone()], plan).unwrap();
        set.verify_startup().unwrap();
        let receipt = set.begin_batch(0).unwrap();
        assert_eq!(receipt.after, 1);
        drop(set);
        let restarted = CountSet::new(&root, vec![count.clone()], plan).unwrap();
        assert_eq!(restarted.counts().unwrap(), vec![1]);
        assert_eq!(restarted.begin_batch(0).unwrap().after, 2);
        assert!(restarted.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn short_write_and_sync_failure_stay_fail_closed() {
        let root = temp_root("short-sync");
        let count = add_leaf(&root, "leaf0");
        let plan = CountPlan::new(3, 6).unwrap();
        let mut set = CountSet::new(&root, vec![count.clone()], plan).unwrap();
        set.force_short_write();
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 2);
        let mut set = CountSet::new(&root, vec![count.clone()], plan).unwrap();
        set.force_sync_failure();
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 5);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unlock_failure_is_not_reported_as_success() {
        let root = temp_root("unlock");
        let count = add_leaf(&root, "leaf0");
        let plan = CountPlan::new(1, 1).unwrap();
        let mut set = CountSet::new(&root, vec![count.clone()], plan).unwrap();
        set.force_unlock_failure();
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn startup_rejects_a_full_live_set() {
        let root = temp_root("startup");
        let count = add_leaf(&root, "leaf0");
        set_count(&count, 1);
        let set = CountSet::new(&root, vec![count], CountPlan::new(1, 1).unwrap()).unwrap();
        assert!(set.verify_startup().is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_count_file_is_not_treated_as_zero() {
        let root = temp_root("missing");
        let missing = root.join("leaf0/count.bin");
        assert!(CountSet::new(&root, vec![missing], CountPlan::new(1, 1).unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn forged_plans_are_rejected_at_every_entry() {
        let root = temp_root("forged-plan");
        let count = add_leaf(&root, "leaf0");
        let zero = CountPlan {
            columns: 0,
            max_columns: 0,
        };
        assert!(zero.check().is_err());
        assert!(CountSet::new(&root, vec![count.clone()], zero).is_err());
        let reversed = CountPlan {
            columns: 2,
            max_columns: 1,
        };
        assert!(reversed.check().is_err());
        assert!(CountSet::new(&root, vec![count.clone()], reversed).is_err());

        let mut set = CountSet::new(
            &root,
            vec![count.clone()],
            CountPlan::new(1, 1).unwrap(),
        )
        .unwrap();
        set.plan.columns = 0;
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 0);

        let mut set = CountSet::new(
            &root,
            vec![count.clone()],
            CountPlan::new(1, 1).unwrap(),
        )
        .unwrap();
        set.plan.max_columns = 0;
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn append_open_call_error_is_rejected_at_startup() {
        let root = temp_root("append-open");
        let count = add_leaf(&root, "leaf0");
        let mut set = CountSet::new(
            &root,
            vec![count.clone()],
            CountPlan::new(1, 1).unwrap(),
        )
        .unwrap();
        // The hook fails inside the append-open helper after the actual open.
        // It proves upper-layer propagation and no append/receipt; it is not a
        // claim that the operating system's open(2) failed.
        set.force_append_failure();
        assert!(set.verify_startup().is_err());
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn close_call_error_propagation_does_not_emit_receipt() {
        let root = temp_root("close-failure");
        let count = add_leaf(&root, "leaf0");
        let mut set = CountSet::new(
            &root,
            vec![count.clone()],
            CountPlan::new(1, 1).unwrap(),
        )
        .unwrap();
        // The hook reports an error after the real close succeeded. This proves
        // the caller does not emit a receipt on the reported close failure; it
        // is not an actual close(2) failure injection.
        set.force_close_failure();
        assert!(set.begin_batch(0).is_err());
        assert_eq!(read_count(&count).unwrap(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn startup_rejects_a_removed_count_file_at_append_open() {
        let root = temp_root("removed-before-open");
        let count = add_leaf(&root, "leaf0");
        let set = CountSet::new(
            &root,
            vec![count.clone()],
            CountPlan::new(1, 1).unwrap(),
        )
        .unwrap();
        // Removing the file after construction makes the real append-open fail
        // with a missing target; no injected flag is used here.
        std::fs::remove_file(&count).unwrap();
        assert!(set.verify_startup().is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn count_files_are_appendable_at_startup() {
        let root = temp_root("appendable");
        let count = add_leaf(&root, "leaf0");
        let set = CountSet::new(
            &root,
            vec![count],
            CountPlan::new(1, 1).unwrap(),
        )
        .unwrap();
        set.verify_startup().unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn child_process_helper() {
        let Some(root) = std::env::var_os("ZKIE_COUNT_CHILD_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let leaves = std::env::var("ZKIE_COUNT_CHILD_LEAVES")
            .unwrap()
            .split(',')
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        let columns = std::env::var("ZKIE_COUNT_CHILD_COLUMNS")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        let max = std::env::var("ZKIE_COUNT_CHILD_MAX")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        let result = CountSet::new(&root, leaves, CountPlan::new(columns, max).unwrap())
            .and_then(|set| set.begin_batch(0));
        match result {
            Ok(_) => println!("CHILD_OK"),
            Err(_) => println!("CHILD_ERR"),
        }
    }

    #[test]
    fn two_processes_serialize_on_one_lock() {
        let root = temp_root("multi-process");
        let count = add_leaf(&root, "leaf0");
        let current = std::env::current_exe().unwrap();
        let mut children = Vec::new();
        for _ in 0..2 {
            let mut command = std::process::Command::new(&current);
            command
                .args([
                    "--exact",
                    "pcs::request::persistent_count::tests::child_process_helper",
                    "--nocapture",
                ])
                .env("ZKIE_COUNT_CHILD_ROOT", &root)
                .env("ZKIE_COUNT_CHILD_LEAVES", &count)
                .env("ZKIE_COUNT_CHILD_COLUMNS", "1")
                .env("ZKIE_COUNT_CHILD_MAX", "1")
                .env("ZKIE_COUNT_CHILD_HOLD_MS", "100")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            children.push(command.spawn().unwrap());
        }
        let outputs = children
            .into_iter()
            .map(|child| child.wait_with_output().unwrap())
            .collect::<Vec<_>>();
        let text = outputs
            .iter()
            .map(|output| {
                String::from_utf8_lossy(&output.stdout).into_owned()
                    + &String::from_utf8_lossy(&output.stderr)
            })
            .collect::<Vec<_>>();
        assert_eq!(text.iter().filter(|text| text.contains("CHILD_OK")).count(), 1);
        assert_eq!(text.iter().filter(|text| text.contains("CHILD_ERR")).count(), 1);
        assert_eq!(read_count(&count).unwrap(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }
}
