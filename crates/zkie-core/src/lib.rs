//! zkie-core: proving substrate (common + pcs).
pub mod artifact;
pub mod claim;
pub mod common;
pub mod compute;
pub mod error;
pub mod oracle;
pub mod profile;
pub mod progress;
pub mod session;
pub mod storage;
pub mod types;

pub use error::{Error, Result};
