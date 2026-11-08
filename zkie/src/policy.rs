//! Trusted verification policy.
//!
//! The policy is fixed by the operator outside the proof bundle. It selects the
//! model version and the expected request shape, and carries the expected
//! anchor and model commitment identities.

use serde::Deserialize;
use zkie_core::artifact::digest_from_hex;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

pub const POLICY_VERSION: u32 = 1;
pub const TOKEN_MAX_SEQUENCE: usize = 512;
pub const TIMESFM_RAW_POINTS: usize = 512;
pub const TIMESFM_PATCH_TOKENS: usize = 16;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyDocument {
    pub policy_version: u32,
    pub model: String,
    pub model_version: u32,
    pub case: Vec<PolicyCase>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyCase {
    pub case_id: String,
    pub sequence_length: Option<usize>,
    pub raw_input_length: Option<usize>,
    pub patch_tokens: Option<usize>,
    pub anchor_id: String,
    pub model_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedRequest {
    pub model: String,
    pub model_version: u32,
    pub case_id: String,
    pub sequence_length: Option<usize>,
    pub raw_input_length: Option<usize>,
    pub patch_tokens: Option<usize>,
    pub anchor_id: Digest,
    pub model_id: Digest,
}

pub fn parse(bytes: &[u8]) -> Result<PolicyDocument> {
    let text = std::str::from_utf8(bytes).map_err(|_| Error::InvalidEncoding)?;
    toml::from_str(text).map_err(|_| Error::InvalidEncoding)
}

pub fn select(
    document: &PolicyDocument,
    case_id: &str,
    model: &str,
    input_len: usize,
) -> Result<TrustedRequest> {
    if document.policy_version != POLICY_VERSION {
        return Err(Error::UnsupportedVersion);
    }
    if document.model != model {
        return Err(Error::StatementMismatch);
    }
    if !matches!(model, "gpt2" | "gemma3" | "timesfm") {
        return Err(Error::UnsupportedPrimitive);
    }
    let mut selected: Option<&PolicyCase> = None;
    let mut seen = std::collections::BTreeSet::new();
    for case in &document.case {
        if case.case_id.is_empty() || !seen.insert(case.case_id.clone()) {
            return Err(Error::StatementMismatch);
        }
        if case.case_id == case_id {
            if selected.is_some() {
                return Err(Error::StatementMismatch);
            }
            selected = Some(case);
        }
    }
    let case = selected.ok_or(Error::StatementMismatch)?;
    match model {
        "gpt2" | "gemma3" => {
            if case.raw_input_length.is_some() || case.patch_tokens.is_some() {
                return Err(Error::InvalidConfig);
            }
            let sequence = case.sequence_length.ok_or(Error::InvalidConfig)?;
            if sequence == 0 || sequence > TOKEN_MAX_SEQUENCE || sequence != input_len {
                return Err(Error::StatementMismatch);
            }
            Ok(TrustedRequest {
                model: document.model.clone(),
                model_version: document.model_version,
                case_id: case.case_id.clone(),
                sequence_length: Some(sequence),
                raw_input_length: None,
                patch_tokens: None,
                anchor_id: digest_from_hex(&case.anchor_id)?,
                model_id: digest_from_hex(&case.model_id)?,
            })
        }
        "timesfm" => {
            if case.sequence_length.is_some() {
                return Err(Error::InvalidConfig);
            }
            let raw = case.raw_input_length.ok_or(Error::InvalidConfig)?;
            let patches = case.patch_tokens.ok_or(Error::InvalidConfig)?;
            if raw != TIMESFM_RAW_POINTS
                || patches != TIMESFM_PATCH_TOKENS
                || raw != input_len
            {
                return Err(Error::StatementMismatch);
            }
            Ok(TrustedRequest {
                model: document.model.clone(),
                model_version: document.model_version,
                case_id: case.case_id.clone(),
                sequence_length: None,
                raw_input_length: Some(raw),
                patch_tokens: Some(patches),
                anchor_id: digest_from_hex(&case.anchor_id)?,
                model_id: digest_from_hex(&case.model_id)?,
            })
        }
        _ => Err(Error::UnsupportedPrimitive),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const B: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn token_policy() -> String {
        format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\n\n[[case]]\ncase_id = \"gpt2-seq512\"\nsequence_length = 512\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\n"
        )
    }

    #[test]
    fn parses_and_selects_token_case() {
        let document = parse(token_policy().as_bytes()).unwrap();
        let selected = select(&document, "gpt2-seq16", "gpt2", 16).unwrap();
        assert_eq!(selected.sequence_length, Some(16));
        assert_eq!(selected.anchor_id.0, [0x11; 32]);
        assert_eq!(selected.model_id.0, [0x22; 32]);
    }

    #[test]
    fn parses_and_selects_timesfm_case() {
        let text = format!(
            "policy_version = 1\nmodel = \"timesfm\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"timesfm-16patch\"\nraw_input_length = 512\npatch_tokens = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\n"
        );
        let document = parse(text.as_bytes()).unwrap();
        let selected = select(&document, "timesfm-16patch", "timesfm", 512).unwrap();
        assert_eq!(selected.raw_input_length, Some(512));
        assert_eq!(selected.patch_tokens, Some(16));
    }

    #[test]
    fn rejects_unknown_fields_and_duplicate_keys() {
        let unknown = token_policy().replace("model_version = 1", "model_version = 1\nextra = 2");
        assert!(parse(unknown.as_bytes()).is_err());
        let duplicate = token_policy().replace("policy_version = 1", "policy_version = 1\npolicy_version = 1");
        assert!(parse(duplicate.as_bytes()).is_err());
    }

    #[test]
    fn rejects_bad_shapes_and_versions() {
        let document = parse(token_policy().as_bytes()).unwrap();
        assert!(select(&document, "gpt2-seq16", "gpt2", 512).is_err());
        assert!(select(&document, "gpt2-seq16", "gemma3", 16).is_err());
        assert!(select(&document, "missing", "gpt2", 16).is_err());
        let mut version = document.clone();
        version.policy_version = 2;
        assert_eq!(
            select(&version, "gpt2-seq16", "gpt2", 16),
            Err(Error::UnsupportedVersion)
        );
        let text = format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"aa\"\nmodel_id = \"{B}\"\n"
        );
        assert!(parse(text.as_bytes()).is_ok());
        let document = parse(text.as_bytes()).unwrap();
        assert!(select(&document, "gpt2-seq16", "gpt2", 16).is_err());
    }

    #[test]
    fn rejects_duplicate_case_ids() {
        let duplicate = token_policy().replace("gpt2-seq512", "gpt2-seq16");
        let document = parse(duplicate.as_bytes()).unwrap();
        assert!(select(&document, "gpt2-seq16", "gpt2", 16).is_err());
    }
}
