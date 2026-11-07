//! Disk-backed scratch tensors.
//!
//! Scratch is created in a caller-supplied directory on a real filesystem; the
//! storage layer never redirects it to tmpfs. A scratch object is writable
//! until sealed, then immutable until released.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use zkie_core::compute::MemoryPermit;
use zkie_core::error::{Error, Result};
use zkie_core::storage::{TensorReader, TensorStorageService};
use zkie_core::types::{
    Digest, ElementRange, ScratchTensorId, TensorId, TensorSpec, TensorStorageRef, TensorView,
};

#[derive(Clone, Debug)]
struct ScratchRecord {
    path: PathBuf,
    spec: TensorSpec,
    sealed: bool,
    released: bool,
}

#[derive(Debug, Default)]
struct ScratchState {
    next_id: u64,
    records: BTreeMap<ScratchTensorId, ScratchRecord>,
}

pub struct ScratchStorage {
    dir: PathBuf,
    state: Mutex<ScratchState>,
}

impl ScratchStorage {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|_| Error::Io)?;
        Ok(Self {
            dir,
            state: Mutex::new(ScratchState::default()),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, ScratchState>> {
        self.state
            .lock()
            .map_err(|_| Error::SynchronizationFailed)
    }
}

impl TensorStorageService for ScratchStorage {
    fn reader(&self, view: &TensorView) -> Result<Arc<dyn TensorReader>> {
        let TensorStorageRef::Scratch(id) = view.storage else {
            return Err(Error::UnsupportedPrimitive);
        };
        let state = self.lock()?;
        let record = state.records.get(&id).ok_or(Error::InvalidIndex)?;
        if !record.sealed || record.released {
            return Err(Error::InvalidSessionState);
        }
        let file = File::open(&record.path).map_err(|_| Error::Io)?;
        let reader: Arc<dyn TensorReader> = Arc::new(ScratchReader {
            file: Mutex::new(file),
            spec: record.spec.clone(),
        });
        Ok(reader)
    }

    fn create_scratch(
        &self,
        spec: &TensorSpec,
        permit: &MemoryPermit,
    ) -> Result<ScratchTensorId> {
        spec.validate()?;
        if permit.request().scratch_disk_bytes < spec.byte_len()? {
            return Err(Error::ResourceLimit);
        }
        let mut state = self.lock()?;
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or(Error::ResourceLimit)?;
        let id = ScratchTensorId(state.next_id);
        let path = self.dir.join(format!("scratch-{}.bin", id.0));
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|_| Error::Io)?;
        state.records.insert(
            id,
            ScratchRecord {
                path,
                spec: spec.clone(),
                sealed: false,
                released: false,
            },
        );
        Ok(id)
    }

    fn write_scratch(
        &self,
        id: ScratchTensorId,
        range: ElementRange,
        canonical_bytes: &[u8],
    ) -> Result<()> {
        let mut state = self.lock()?;
        let record = state.records.get_mut(&id).ok_or(Error::InvalidIndex)?;
        if record.sealed || record.released {
            return Err(Error::InvalidSessionState);
        }
        range.validate(&record.spec)?;
        let expected = range
            .len
            .checked_mul(record.spec.dtype.element_size())
            .ok_or(Error::ShapeMismatch)?;
        if canonical_bytes.len() as u64 != expected {
            return Err(Error::ShapeMismatch);
        }
        let (start_byte, _) = range.byte_range(&record.spec)?;
        let mut file = OpenOptions::new()
            .write(true)
            .open(&record.path)
            .map_err(|_| Error::Io)?;
        file.seek(SeekFrom::Start(start_byte))
            .map_err(|_| Error::Io)?;
        file.write_all(canonical_bytes).map_err(|_| Error::Io)
    }

    fn seal_scratch(&self, id: ScratchTensorId) -> Result<TensorView> {
        let mut state = self.lock()?;
        let record = state.records.get_mut(&id).ok_or(Error::InvalidIndex)?;
        if record.released {
            return Err(Error::InvalidSessionState);
        }
        record.sealed = true;
        let file = OpenOptions::new()
            .write(true)
            .open(&record.path)
            .map_err(|_| Error::Io)?;
        file.sync_all().map_err(|_| Error::Io)?;
        let count = record.spec.element_count()?;
        Ok(TensorView {
            tensor_id: TensorId(id.0),
            spec: record.spec.clone(),
            storage: TensorStorageRef::Scratch(id),
            element_range: ElementRange::new(0, count),
        })
    }

    fn release_scratch(&self, id: ScratchTensorId) -> Result<()> {
        let mut state = self.lock()?;
        let record = state.records.get_mut(&id).ok_or(Error::InvalidIndex)?;
        if record.released {
            return Ok(());
        }
        record.released = true;
        fs::remove_file(&record.path).map_err(|_| Error::Io)
    }
}

struct ScratchReader {
    file: Mutex<File>,
    spec: TensorSpec,
}

impl TensorReader for ScratchReader {
    fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    fn element_count(&self) -> u64 {
        self.spec.element_count().unwrap_or(0)
    }

    fn identity(&self) -> Digest {
        // TODO: bind a caller-supplied scratch identity when assets are wired.
        Digest::ZERO
    }

    fn read_range(
        &self,
        start_element: u64,
        element_count: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let count = u64::try_from(element_count).map_err(|_| Error::InvalidIndex)?;
        let range = ElementRange::new(start_element, count);
        range.validate(&self.spec)?;
        let (start_byte, byte_len) = range.byte_range(&self.spec)?;
        if dst.len() as u64 != byte_len {
            return Err(Error::ShapeMismatch);
        }
        let mut file = self.file.lock().map_err(|_| Error::SynchronizationFailed)?;
        file.seek(SeekFrom::Start(start_byte))
            .map_err(|_| Error::Io)?;
        file.read_exact(dst).map_err(|_| Error::IncompleteArtifact)
    }
}
