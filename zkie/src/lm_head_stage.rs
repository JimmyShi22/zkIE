//! Versioned public and private handoff records for the LM-head stages.
//!
//! The public record carries only plan and commitment data; the private record
//! carries the secret values one prove run needs. Files are created without
//! overwriting, private directories are owner-only and private files are
//! owner-readable only. The budget journal reserves a proof's draw budget once;
//! a second claim fails, and no failure rolls the state back.

use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::runtime_tree::MaskTreePlan;
use zkie_engine::pcs::request::table::IndexReader;
use zkie_ops::proof::base_last_session::LastBaseRandomness;
use zkie_ops::proof::session_baseline::SESSION_PROTOCOL_IDENTITY;

use crate::lm_head_key::{
    BaseStarBinding, LmHeadGuRandomness, LmHeadKeyRandomness, LmHeadMaskRandomness,
    LmHeadOldMaskRandomness, LmHeadOmegaRandomness, LimitedKeyStatement, LimitedSwitchPlan,
};

pub const LM_HEAD_STAGE_VERSION: u64 = 1;
const PUBLIC_FILE: &str = "public.bin";
const PRIVATE_FILE: &str = "prove-state.bin";
const BUDGET_FILE: &str = "budget.state";
const BUDGET_LOCK: &str = "budget.lock";

/// The public old-mask commitment: its tree plan and root.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OldMaskPublic {
    pub plan: MaskTreePlan,
    pub root: Digest,
}

/// One public handoff. It is everything the prove and verify processes may
/// learn without the prover's secret values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmHeadPublicHandoff {
    pub version: u64,
    pub protocol_identity: Digest,
    pub statement: Vec<u8>,
    pub reader: IndexReader,
    pub plan_identifier: Digest,
    pub case_id: Vec<u8>,
    pub random_bytes: Digest,
    pub base: BaseStarBinding,
    pub old_mask: Option<OldMaskPublic>,
    pub fresh_plan: MaskTreePlan,
    pub switches: Vec<LimitedSwitchPlan>,
}

impl LmHeadPublicHandoff {
    /// The statement must decode, validate and agree with the reader, and all
    /// plans must pass their own self-checks.
    pub fn validate(&self) -> Result<LimitedKeyStatement> {
        if self.version != LM_HEAD_STAGE_VERSION
            || self.protocol_identity != SESSION_PROTOCOL_IDENTITY
            || self.plan_identifier == Digest::ZERO
            || self.random_bytes == Digest::ZERO
            || self.case_id.is_empty()
        {
            return Err(Error::InvalidConfig);
        }
        let statement = LimitedKeyStatement::decode(&self.statement)?;
        statement.validate()?;
        if statement.table_root != self.reader.root
            || statement.table_configuration != self.reader.configuration
            || statement.table_shape != self.reader.shape
            || statement.matrix_rows == 0
        {
            return Err(Error::StatementMismatch);
        }
        if self.base.plan.fault().is_some()
            || self.base.positions == 0
            || self.fresh_plan.fault().is_some()
            || self.switches.is_empty()
        {
            return Err(Error::InvalidConfig);
        }
        if let Some(old) = &self.old_mask {
            if old.plan.fault().is_some() || old.root == Digest::ZERO {
                return Err(Error::InvalidConfig);
            }
        }
        for switch in &self.switches {
            if switch.target.main_rows == 0
                || switch.target.mask_rows == 0
                || switch.positions == 0
                || switch.plan.fault().is_some()
            {
                return Err(Error::InvalidConfig);
            }
            if let Some(stage) = &switch.stage_fold {
                if stage.mask_tree.fault().is_some() || stage.layout.fault(&stage.shape).is_some() {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        Ok(statement)
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        encode_bytes(self)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let value: Self = decode_bytes(bytes)?;
        value.validate()?;
        Ok(value)
    }

    pub fn digest(&self) -> Result<Digest> {
        Ok(digest_bytes(&self.encode()?))
    }

    /// Checks the handoff's committed public reader and statement against the
    /// trusted policy table identity. The context's old-side fields were
    /// already checked against the public anchor and asset when it was loaded.
    /// The verifier uses this instead of trusting the handoff's own statement
    /// or roots.
    pub fn verify_trusted_identity(
        &self,
        context: &crate::s7_context::LmHeadContext,
    ) -> Result<LimitedKeyStatement> {
        let statement = self.validate()?;
        context.verify_committed_limited_table_identity(&self.reader, &statement)?;
        Ok(statement)
    }
}

/// The persisted commitment state of the committed table: its public reader
/// and name, the column leaves, the private salts and the commit peak. It is
/// enough to rebuild the table without recomputing any codeword.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmHeadTableState {
    pub reader: IndexReader,
    pub name: Vec<u8>,
    pub leaves: Vec<Digest>,
    pub salts: Vec<Vec<u8>>,
    pub peak_bytes: u64,
}

impl LmHeadTableState {
    pub fn validate(&self) -> Result<()> {
        if self.reader.fault().is_some() || self.name.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let expected = 1usize
            .checked_shl(self.reader.shape.code.log_n)
            .ok_or(Error::ResourceLimit)?;
        if self.leaves.len() != expected || self.salts.len() != expected {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// One private prove state. It is fresh for one session and bound to the
/// public handoff digest; a second prove with the same state is rejected by
/// the budget journal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmHeadPrivateState {
    pub version: u64,
    pub public_digest: Digest,
    pub table: LmHeadTableState,
    pub key: LmHeadKeyRandomness,
    pub gu: LmHeadGuRandomness,
    pub omega: LmHeadOmegaRandomness,
    pub mask: LmHeadMaskRandomness,
    pub old_mask: LmHeadOldMaskRandomness,
    pub master: Digest,
    pub base_salts: Vec<Vec<u8>>,
    pub base_randomness: LastBaseRandomness,
}

impl LmHeadPrivateState {
    pub fn validate(&self, public_digest: Digest) -> Result<()> {
        if self.version != LM_HEAD_STAGE_VERSION
            || self.public_digest != public_digest
            || self.master == Digest::ZERO
            || self.base_salts.is_empty()
            || self.key.salts.is_empty()
        {
            return Err(Error::InvalidConfig);
        }
        self.table.validate()?;
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        encode_bytes(self)
    }

    pub fn decode(bytes: &[u8], public_digest: Digest) -> Result<Self> {
        let value: Self = decode_bytes(bytes)?;
        value.validate(public_digest)?;
        Ok(value)
    }
}

/// The persisted draw budget of one session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LmHeadBudgetState {
    pub version: u64,
    pub session: Digest,
    pub max_columns: u64,
    pub used_columns: u64,
    pub mask_consumed: bool,
}

/// A successful reservation. Dropping it does not release the reservation.
#[derive(Clone, Copy, Debug)]
pub struct LmHeadBudgetClaim {
    pub state: LmHeadBudgetState,
}

pub(crate) fn encode_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(value)
        .map_err(|_| Error::InvalidEncoding)
}

pub(crate) fn decode_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|_| Error::InvalidEncoding)
}

pub(crate) fn private_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

fn public_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o644);
    }
    options
}

pub(crate) fn ensure_private_dir(dir: &Path) -> Result<()> {
    if let Ok(metadata) = std::fs::symlink_metadata(dir) {
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::Io);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(Error::InvalidConfig);
            }
        }
        return Ok(());
    }
    let mut builder = DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir).map_err(|_| Error::Io)
}

pub(crate) fn write_bytes(path: &Path, bytes: &[u8], private: bool) -> Result<()> {
    let mut options = if private {
        private_options()
    } else {
        public_options()
    };
    let mut file = options.open(path).map_err(|_| Error::Io)?;
    file.write_all(bytes).map_err(|_| Error::Io)?;
    file.sync_all().map_err(|_| Error::Io)
}

pub(crate) fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    if let Ok(metadata) = std::fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() {
            return Err(Error::Io);
        }
    }
    let mut file = File::open(path).map_err(|_| Error::Io)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|_| Error::Io)?;
    Ok(bytes)
}

pub fn write_public(dir: &Path, handoff: &LmHeadPublicHandoff) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(|_| Error::Io)?;
    let path = dir.join(PUBLIC_FILE);
    write_bytes(&path, &handoff.encode()?, false)?;
    Ok(path)
}

/// Writes one public stage file. The directory is created if absent; the file
/// is created fresh and never overwrites.
pub fn write_public_file(dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(|_| Error::Io)?;
    let path = dir.join(name);
    write_bytes(&path, bytes, false)?;
    Ok(path)
}

pub fn read_public_file(path: &Path) -> Result<Vec<u8>> {
    read_bytes(path)
}

/// Writes one private stage file under an owner-only directory.
pub fn write_private_file(dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    ensure_private_dir(dir)?;
    let path = dir.join(name);
    write_bytes(&path, bytes, true)?;
    Ok(path)
}

pub fn read_public(path: &Path) -> Result<LmHeadPublicHandoff> {
    LmHeadPublicHandoff::decode(&read_bytes(path)?)
}

pub fn prepare_private_dir(dir: &Path) -> Result<()> {
    ensure_private_dir(dir)
}

pub fn write_private(dir: &Path, state: &LmHeadPrivateState) -> Result<PathBuf> {
    ensure_private_dir(dir)?;
    let path = dir.join(PRIVATE_FILE);
    write_bytes(&path, &state.encode()?, true)?;
    Ok(path)
}

pub fn read_private(path: &Path, public_digest: Digest) -> Result<LmHeadPrivateState> {
    LmHeadPrivateState::decode(&read_bytes(path)?, public_digest)
}

pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    let file = File::open(dir).map_err(|_| Error::Io)?;
    file.sync_all().map_err(|_| Error::Io)
}

fn budget_path(dir: &Path) -> PathBuf {
    dir.join(BUDGET_FILE)
}

fn budget_from(bytes: &[u8]) -> Result<LmHeadBudgetState> {
    let state: LmHeadBudgetState = decode_bytes(bytes)?;
    if state.version != LM_HEAD_STAGE_VERSION || state.session == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    Ok(state)
}

fn write_budget(dir: &Path, state: &LmHeadBudgetState) -> Result<()> {
    let path = budget_path(dir);
    write_bytes(&path, &encode_bytes(state)?, true)
}

/// Writes the session's initial budget. The file is created fresh; a second
/// initialization of the same private directory fails.
pub fn initialize_budget(dir: &Path, session: Digest, max_columns: u64) -> Result<()> {
    if session == Digest::ZERO || max_columns == 0 {
        return Err(Error::InvalidConfig);
    }
    ensure_private_dir(dir)?;
    let state = LmHeadBudgetState {
        version: LM_HEAD_STAGE_VERSION,
        session,
        max_columns,
        used_columns: 0,
        mask_consumed: false,
    };
    write_budget(dir, &state)?;
    sync_dir(dir)
}

pub fn read_budget(dir: &Path) -> Result<LmHeadBudgetState> {
    budget_from(&read_bytes(&budget_path(dir))?)
}

/// Atomically reserves the columns and the mask material for one proof. The
/// journal fences reuse within this directory lifetime; rolling back the whole
/// machine or directory snapshot to an earlier state is an external
/// deployment trust assumption and is not detected here. The
/// lock file is created exclusively; a stale lock or a spent budget makes a
/// later claim fail. A successful claim never rolls back.
pub fn claim_budget(dir: &Path, session: Digest, columns: u64) -> Result<LmHeadBudgetClaim> {
    ensure_private_dir(dir)?;
    if session == Digest::ZERO || columns == 0 {
        return Err(Error::InvalidConfig);
    }
    let path = budget_path(dir);
    let lock = dir.join(BUDGET_LOCK);
    let _lock = private_options().open(&lock).map_err(|_| Error::InvalidSessionState)?;
    let state = budget_from(&read_bytes(&path)?)?;
    if state.session != session
        || state.mask_consumed
        || state.used_columns.checked_add(columns).ok_or(Error::ResourceLimit)? > state.max_columns
    {
        return Err(Error::InvalidSessionState);
    }
    let claimed = LmHeadBudgetState {
        used_columns: state.used_columns + columns,
        mask_consumed: true,
        ..state
    };
    let temp = dir.join("budget.tmp");
    write_bytes(&temp, &encode_bytes(&claimed)?, true)?;
    std::fs::rename(&temp, &path).map_err(|_| Error::Io)?;
    // The replacement is durable before the lock is released: the file is
    // synced, then the parent directory entry, then the lock is removed and
    // the directory synced again. A crash before the first directory sync
    // leaves the old state or the lock in place and every retry fails closed;
    // a crash after it leaves the spent state durable.
    sync_dir(dir)?;
    std::fs::remove_file(&lock).map_err(|_| Error::Io)?;
    sync_dir(dir)?;
    Ok(LmHeadBudgetClaim { state: claimed })
}

/// One file that a verifier needs. The role is a stable name, the path is
/// relative to the manifest directory, and the digest covers the whole file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    pub role: String,
    pub path: String,
    pub bytes: u64,
    pub digest: Digest,
}

/// The versioned list of every byte a verifier needs for this chain. A
/// manifest binds referenced files by whole-file digest and rejects a missing
/// file, a size or digest mismatch, a path outside its directory and a
/// trailing byte in the manifest itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationManifest {
    pub version: u64,
    pub files: Vec<ManifestFile>,
}

pub const VERIFICATION_MANIFEST_VERSION: u64 = 1;
pub const VERIFICATION_MANIFEST_FILE: &str = "verification-manifest.bin";

fn checked_relative(path: &str) -> Result<PathBuf> {
    if path.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let value = Path::new(path);
    if value.is_absolute()
        || value
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(Error::InvalidConfig);
    }
    Ok(value.to_path_buf())
}

impl VerificationManifest {
    pub fn new(mut files: Vec<ManifestFile>) -> Result<Self> {
        if files.is_empty() {
            return Err(Error::IncompleteArtifact);
        }
        files.sort_by(|a, b| a.role.cmp(&b.role));
        let value = Self {
            version: VERIFICATION_MANIFEST_VERSION,
            files,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != VERIFICATION_MANIFEST_VERSION || self.files.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let mut seen = std::collections::BTreeSet::new();
        for file in &self.files {
            if file.role.is_empty() || file.digest == Digest::ZERO {
                return Err(Error::InvalidConfig);
            }
            checked_relative(&file.path)?;
            if !seen.insert(file.role.as_str()) {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        encode_bytes(self)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let value: Self = decode_bytes(bytes)?;
        value.validate()?;
        Ok(value)
    }

    pub fn digest(&self) -> Result<Digest> {
        Ok(zkie_core::artifact::digest_bytes(&self.encode()?))
    }

    /// Checks every referenced file against its declared size and digest.
    pub fn verify(&self, dir: &Path) -> Result<()> {
        self.validate()?;
        for file in &self.files {
            let path = dir.join(checked_relative(&file.path)?);
            let bytes = read_bytes(&path)?;
            if bytes.len() as u64 != file.bytes
                || zkie_core::artifact::digest_bytes(&bytes) != file.digest
            {
                return Err(Error::ArtifactIntegrityMismatch);
            }
        }
        Ok(())
    }
}

/// Writes the manifest and returns its digest. The file is created fresh.
pub fn write_manifest(dir: &Path, manifest: &VerificationManifest) -> Result<Digest> {
    manifest.validate()?;
    write_bytes(
        &dir.join(VERIFICATION_MANIFEST_FILE),
        &manifest.encode()?,
        false,
    )?;
    manifest.digest()
}

pub fn read_manifest(dir: &Path) -> Result<VerificationManifest> {
    VerificationManifest::decode(&read_bytes(&dir.join(VERIFICATION_MANIFEST_FILE))?)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::cs_stage::{Cut, CutPart, SwitchTarget};
    use zkie_engine::pcs::request::runtime_tree::{
        RuntimeTreePart, RuntimeTreePlan,
    };
    use zkie_engine::pcs::request::table::{
        IndexRecord, IndexShape, TableDistances,
    };

    use super::*;
    use zkie_engine::pcs::request::field::{Fp, Fp3};
    use crate::lm_head_key::LIMITED_KEY_VERSION;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn dir() -> PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("zkie-stage-test-{}-{}", std::process::id(), id))
    }

    fn statement() -> Vec<u8> {
        let code = build_code_plan(3, 4, 1).expect("code");
        LimitedKeyStatement {
            version: LIMITED_KEY_VERSION,
            model_commitment: Digest([1; 32]),
            table_root: Digest([2; 32]),
            table_configuration: Digest([3; 32]),
            table_shape: IndexShape {
                code,
                v: 2,
                answers: 1,
                salt_bytes: 40,
                distances: TableDistances {
                    radius: 1,
                    later_radius: 1,
                    m_once: 1,
                    m_later: 1,
                },
            },
            matrix_rows: 4,
            k_lo: 0,
            k_col: 0,
            valid_min: i32::MIN as i64,
            valid_max: i32::MAX as i64,
            no_wf: true,
            trusted_i32_mapping: true,
        }
        .encode()
    }

    fn tree() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 1,
            leaf_class: 0,
            log_n: 3,
            log_n_zk: 3,
            parts: vec![RuntimeTreePart {
                main_symbols: 3,
                mask_symbols: 3,
            }],
            salt_bytes: 40,
            run: Digest([4; 32]),
            configuration: Digest([3; 32]),
        }
    }

    fn sample_public() -> LmHeadPublicHandoff {
        let statement_bytes = statement();
        let statement = LimitedKeyStatement::decode(&statement_bytes).expect("statement");
        let reader = IndexReader {
            shape: statement.table_shape,
            configuration: statement.table_configuration,
            root: statement.table_root,
            record: IndexRecord {
                y: vec![Fp3::ZERO],
            },
        };
        let target = SwitchTarget {
            code: build_code_plan(3, 4, 1).expect("code"),
            mask_code: build_code_plan(3, 4, 1).expect("mask code"),
            main_rows: 1,
            mask_rows: 1,
            answers: 1,
            main: Cut {
                part: CutPart::Main,
                bits: 0,
                first_row: 0,
            },
            rest: Cut {
                part: CutPart::Mask,
                bits: 0,
                first_row: 0,
            },
        };
        LmHeadPublicHandoff {
            version: LM_HEAD_STAGE_VERSION,
            protocol_identity: SESSION_PROTOCOL_IDENTITY,
            statement: statement_bytes,
            reader,
            plan_identifier: Digest([5; 32]),
            case_id: b"case".to_vec(),
            random_bytes: Digest([6; 32]),
            base: BaseStarBinding {
                plan: tree(),
                positions: 2,
            },
            old_mask: Some(OldMaskPublic {
                plan: MaskTreePlan {
                    tree_number: 2,
                    leaf_class: 0,
                    log_n: 3,
                    mask_rows: 1,
                    salt_bytes: 40,
                    run: Digest([4; 32]),
                    configuration: Digest([3; 32]),
                },
                root: Digest([7; 32]),
            }),
            fresh_plan: MaskTreePlan {
                tree_number: 3,
                leaf_class: 0,
                log_n: 3,
                mask_rows: 1,
                salt_bytes: 40,
                run: Digest([4; 32]),
                configuration: Digest([3; 32]),
            },
            switches: vec![LimitedSwitchPlan {
                target,
                positions: 2,
                plan: tree(),
                stage_fold: None,
            }],
        }
    }

    fn sample_private(public_digest: Digest) -> LmHeadPrivateState {
        let statement_bytes = statement();
        let statement = LimitedKeyStatement::decode(&statement_bytes).expect("statement");
        let reader = IndexReader {
            shape: statement.table_shape,
            configuration: statement.table_configuration,
            root: statement.table_root,
            record: IndexRecord {
                y: vec![
                    Fp3::ZERO;
                    (1usize << statement.table_shape.v)
                        * statement.table_shape.answers as usize
                ],
            },
        };
        let n = 1usize << statement.table_shape.code.log_n;
        LmHeadPrivateState {
            version: LM_HEAD_STAGE_VERSION,
            public_digest,
            table: LmHeadTableState {
                reader,
                name: b"name".to_vec(),
                leaves: vec![Digest::ZERO; n],
                salts: vec![vec![0; 40]; n],
                peak_bytes: 0,
            },
            key: LmHeadKeyRandomness {
                tails: vec![vec![Fp::default()]],
                salts: vec![vec![1; 40]],
                pads: Vec::new(),
            },
            gu: LmHeadGuRandomness {
                code_tails: Vec::new(),
                salts: Vec::new(),
                pads: Vec::new(),
            },
            omega: LmHeadOmegaRandomness {
                code_tails: Vec::new(),
                salts: Vec::new(),
                pad: Fp3::ZERO,
            },
            mask: LmHeadMaskRandomness {
                seed: Digest([8; 32]),
                salts: vec![vec![2; 40]],
            },
            old_mask: LmHeadOldMaskRandomness {
                rows: Vec::new(),
                tails: Vec::new(),
                salts: Vec::new(),
            },
            master: Digest([9; 32]),
            base_salts: vec![vec![3; 40]],
            base_randomness: LastBaseRandomness::Fixed(Digest([10; 32])),
        }
    }

    #[test]
    fn public_and_private_records_round_trip_and_reject_damage() {
        let public = sample_public();
        let bytes = public.encode().expect("encode");
        let decoded = LmHeadPublicHandoff::decode(&bytes).expect("decode");
        assert_eq!(decoded.statement, public.statement);
        assert_eq!(decoded.digest().expect("digest"), public.digest().expect("digest"));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(LmHeadPublicHandoff::decode(&trailing).is_err());
        let mut version = public.clone();
        version.version = LM_HEAD_STAGE_VERSION + 1;
        assert!(LmHeadPublicHandoff::decode(&version.encode().expect("encode")).is_err());

        let digest = public.digest().expect("digest");
        let private = sample_private(digest);
        let bytes = private.encode().expect("encode");
        assert!(LmHeadPrivateState::decode(&bytes, digest).is_ok());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(LmHeadPrivateState::decode(&trailing, digest).is_err());
        assert!(LmHeadPrivateState::decode(&bytes, Digest([11; 32])).is_err());
    }

    #[test]
    fn private_files_are_owner_only_and_do_not_overwrite() {
        let dir = dir();
        let public = sample_public();
        let digest = public.digest().expect("digest");
        let private_dir = dir.join("private");
        let state = sample_private(digest);
        let path = write_private(&private_dir, &state).expect("write");
        assert!(write_private(&private_dir, &state).is_err());
        assert!(read_private(&path, digest).is_ok());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&private_dir).expect("dir").permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                std::fs::metadata(&path).expect("file").permissions().mode() & 0o777,
                0o600
            );
        }
        #[cfg(unix)]
        {
            let outside = dir.join("outside");
            std::fs::write(&outside, b"x").expect("outside");
            let _ = std::fs::remove_file(&path);
            std::os::unix::fs::symlink(&outside, &path).expect("symlink");
            assert!(write_private(&private_dir, &state).is_err());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_budget_is_reserved_once_and_never_rolls_back() {
        let dir = dir();
        let session = Digest([12; 32]);
        initialize_budget(&dir, session, 10).expect("initialize");
        assert!(initialize_budget(&dir, session, 10).is_err());
        let claim = claim_budget(&dir, session, 4).expect("claim");
        assert_eq!(claim.state.used_columns, 4);
        assert!(claim.state.mask_consumed);
        let state = read_budget(&dir).expect("state");
        assert_eq!(state.used_columns, 4);
        assert!(claim_budget(&dir, session, 1).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn another_session_cannot_claim_the_budget() {
        let dir = dir();
        let session = Digest([13; 32]);
        initialize_budget(&dir, session, 10).expect("initialize");
        assert!(claim_budget(&dir, Digest([14; 32]), 1).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_successful_claim_is_durable_and_leaves_no_journal_temporary() {
        let dir = dir();
        let session = Digest([15; 32]);
        initialize_budget(&dir, session, 10).expect("initialize");
        let _claim = claim_budget(&dir, session, 4).expect("claim");
        assert!(!dir.join(BUDGET_LOCK).exists());
        assert!(!dir.join("budget.tmp").exists());
        // A fresh reader, as after a restart, sees the spent journal.
        let state = read_budget(&dir).expect("reopen");
        assert_eq!(state.used_columns, 4);
        assert!(state.mask_consumed);
        assert!(claim_budget(&dir, session, 4).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_claims_have_one_winner_and_no_mask_reuse() {
        let dir = dir();
        let session = Digest([16; 32]);
        initialize_budget(&dir, session, 10).expect("initialize");
        let first = dir.clone();
        let second = dir.clone();
        let a = std::thread::spawn(move || claim_budget(&first, session, 4).is_ok());
        let b = std::thread::spawn(move || claim_budget(&second, session, 4).is_ok());
        let wins = [a.join().expect("join"), b.join().expect("join")]
            .into_iter()
            .filter(|ok| *ok)
            .count();
        assert_eq!(wins, 1);
        let state = read_budget(&dir).expect("state");
        assert_eq!(state.used_columns, 4);
        assert!(state.mask_consumed);
        assert!(claim_budget(&dir, session, 4).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_journal_lock_fails_closed_and_keeps_the_old_state() {
        let dir = dir();
        let session = Digest([17; 32]);
        initialize_budget(&dir, session, 10).expect("initialize");
        std::fs::write(dir.join(BUDGET_LOCK), b"stale").expect("lock");
        assert!(claim_budget(&dir, session, 4).is_err());
        let state = read_budget(&dir).expect("state");
        assert_eq!(state.used_columns, 0);
        assert!(!state.mask_consumed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn manifest_file(role: &str, path: &str, bytes: &[u8]) -> ManifestFile {
        ManifestFile {
            role: role.to_string(),
            path: path.to_string(),
            bytes: bytes.len() as u64,
            digest: zkie_core::artifact::digest_bytes(bytes),
        }
    }

    #[test]
    fn the_manifest_binds_whole_files_and_rejects_damage() {
        let dir = dir();
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("artifact.bin"), b"artifact").expect("artifact");
        std::fs::write(dir.join("trace.bin"), b"trace").expect("trace");
        let manifest = VerificationManifest::new(vec![
            manifest_file("artifact", "artifact.bin", b"artifact"),
            manifest_file("trace", "trace.bin", b"trace"),
        ])
        .expect("manifest");
        manifest.verify(&dir).expect("verify");
        let digest = write_manifest(&dir, &manifest).expect("write");
        assert_eq!(read_manifest(&dir).expect("read").digest().expect("digest"), digest);

        std::fs::write(dir.join("artifact.bin"), b"artifacT").expect("tamper");
        assert!(manifest.verify(&dir).is_err());

        std::fs::remove_file(dir.join("trace.bin")).expect("remove");
        assert!(manifest.verify(&dir).is_err());

        let outside = VerificationManifest::new(vec![manifest_file(
            "outside",
            "../outside.bin",
            b"x",
        )]);
        assert!(outside.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_manifest_rejects_trailing_bytes_and_duplicate_roles() {
        let value = VerificationManifest::new(vec![manifest_file("a", "a.bin", b"a")]).expect("m");
        let mut trailing = value.encode().expect("encode");
        trailing.push(0);
        assert!(VerificationManifest::decode(&trailing).is_err());
        let duplicate = VerificationManifest::new(vec![
            manifest_file("a", "a.bin", b"a"),
            manifest_file("a", "b.bin", b"b"),
        ]);
        assert!(duplicate.is_err());
    }
}
