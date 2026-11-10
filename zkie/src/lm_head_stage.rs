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
}

/// One private prove state. It is fresh for one session and bound to the
/// public handoff digest; a second prove with the same state is rejected by
/// the budget journal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmHeadPrivateState {
    pub version: u64,
    pub public_digest: Digest,
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

fn encode_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(value)
        .map_err(|_| Error::InvalidEncoding)
}

fn decode_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|_| Error::InvalidEncoding)
}

fn private_options() -> OpenOptions {
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

fn ensure_private_dir(dir: &Path) -> Result<()> {
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

fn write_bytes(path: &Path, bytes: &[u8], private: bool) -> Result<()> {
    let mut options = if private {
        private_options()
    } else {
        public_options()
    };
    let mut file = options.open(path).map_err(|_| Error::Io)?;
    file.write_all(bytes).map_err(|_| Error::Io)?;
    file.sync_all().map_err(|_| Error::Io)
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
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

pub fn write_private(dir: &Path, state: &LmHeadPrivateState) -> Result<PathBuf> {
    ensure_private_dir(dir)?;
    let path = dir.join(PRIVATE_FILE);
    write_bytes(&path, &state.encode()?, true)?;
    Ok(path)
}

pub fn read_private(path: &Path, public_digest: Digest) -> Result<LmHeadPrivateState> {
    LmHeadPrivateState::decode(&read_bytes(path)?, public_digest)
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
    write_budget(dir, &state)
}

pub fn read_budget(dir: &Path) -> Result<LmHeadBudgetState> {
    budget_from(&read_bytes(&budget_path(dir))?)
}

/// Atomically reserves the columns and the mask material for one proof. The
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
    let _ = std::fs::remove_file(&lock);
    Ok(LmHeadBudgetClaim { state: claimed })
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
                plan: tree(),
                stage_fold: None,
            }],
        }
    }

    fn sample_private(public_digest: Digest) -> LmHeadPrivateState {
        LmHeadPrivateState {
            version: LM_HEAD_STAGE_VERSION,
            public_digest,
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
}
