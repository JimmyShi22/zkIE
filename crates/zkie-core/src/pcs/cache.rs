use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use super::whir::Commitment;

pub const BATCH_CACHE_SCHEMA_VERSION: u32 = 2;
const BATCH_CACHE_MAGIC: [u8; 8] = *b"ZKIEBCH2";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchCacheMetadata {
    pub schema_version: u32,
    pub model_id: String,
    pub group_id: String,
    /// Ordered model tensor identifiers. Their position is the committed table index.
    pub table_ids: Vec<String>,
    pub table_arity: u32,
    pub num_tables: u32,
    pub security_level: u32,
    pub pow_bits: u32,
    pub source_digest: [u8; 32],
}

impl BatchCacheMetadata {
    pub fn new(
        model_id: String,
        group_id: String,
        table_ids: Vec<String>,
        table_arity: u32,
        num_tables: u32,
        security_level: u32,
        pow_bits: u32,
        source_digest: [u8; 32],
    ) -> Self {
        assert_eq!(table_ids.len(), num_tables as usize);
        Self {
            schema_version: BATCH_CACHE_SCHEMA_VERSION,
            model_id,
            group_id,
            table_ids,
            table_arity,
            num_tables,
            security_level,
            pow_bits,
            source_digest,
        }
    }
}

pub fn write_header<W: Write>(
    writer: &mut W,
    metadata: &BatchCacheMetadata,
    commitment: &Commitment,
) -> Result<(), Box<bincode::ErrorKind>> {
    assert_eq!(metadata.schema_version, BATCH_CACHE_SCHEMA_VERSION);
    bincode::serialize_into(writer, &(BATCH_CACHE_MAGIC, metadata, commitment))
}

pub fn read_header<R: Read>(
    reader: &mut R,
) -> Result<(BatchCacheMetadata, Commitment), Box<bincode::ErrorKind>> {
    let (magic, metadata, commitment): ([u8; 8], BatchCacheMetadata, Commitment) =
        bincode::deserialize_from(reader)?;
    if magic != BATCH_CACHE_MAGIC || metadata.schema_version != BATCH_CACHE_SCHEMA_VERSION {
        return Err(Box::new(bincode::ErrorKind::Custom(
            "unsupported batch cache format".to_owned(),
        )));
    }
    Ok((metadata, commitment))
}
