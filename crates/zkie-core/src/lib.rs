//! zkie-core: shared types, commitments, and execution traits.
pub mod artifact;
pub mod claim;
pub mod common;
pub mod compute;
pub mod error;
pub mod oracle;
pub mod profile;
pub mod progress;
pub mod proof_header;
pub mod storage;
pub mod types;

pub use error::{Error, Result};
