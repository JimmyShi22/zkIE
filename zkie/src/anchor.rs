//! `anchor` subcommand: operator-side registration of model assets.
//!
//! The command reads the case input and every registered model asset through
//! the trusted model specifications, rebuilds the complete commitment with the
//! same algorithm used by commit, and prints the anchor identity, input-bound
//! model identity, parameters digest and specification version. It never reads
//! an anchor or identity from a proof bundle.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use zkie_core::artifact::{digest_bytes, digest_from_hex, digest_to_hex};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_ops::proof::model_commitment::{
    commit_assets_with_digests, AssetDigest, ModelCommitment,
};

use crate::report::{parse_flags, path_flag};

pub struct AnchorArgs {
    pub case: PathBuf,
    pub input: PathBuf,
    pub asset_root: PathBuf,
    pub out: Option<PathBuf>,
    pub asset_hashes: Option<PathBuf>,
    pub policy: Option<PathBuf>,
}

impl AnchorArgs {
    pub fn parse(args: &[String]) -> Result<Self> {
        let flags = parse_flags(
            args,
            &["case", "input", "asset-root", "out", "asset-hashes", "policy"],
        )?;
        Ok(Self {
            case: path_flag(&flags, "case")?,
            input: path_flag(&flags, "input")?,
            asset_root: path_flag(&flags, "asset-root")?,
            out: flags.get("out").map(PathBuf::from),
            asset_hashes: flags.get("asset-hashes").map(PathBuf::from),
            policy: flags.get("policy").map(PathBuf::from),
        })
    }
}

fn read_input(input: &Path) -> Result<Vec<i32>> {
    let bytes = std::fs::read(input).map_err(|_| Error::Io)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidEncoding);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
        .collect())
}

struct SelectedCase {
    asset_root: PathBuf,
    case_id: String,
    model: String,
    seq_tokens: usize,
    gt_path: PathBuf,
    gt_bytes: u64,
    gt_sha256: String,
}

fn select_case(case: &Path, input: &Path) -> Result<SelectedCase> {
    let (root, entries) = crate::run::parse_cases(case)?;
    let wanted = std::fs::canonicalize(input).map_err(|_| Error::Io)?;
    let mut selected: Option<(String, usize, Option<String>, Option<u64>, Option<String>)> = None;
    for entry in entries {
        let candidate = root.join(&entry.input);
        let matches = std::fs::canonicalize(&candidate)
            .map(|path| path == wanted)
            .unwrap_or(false);
        if !matches {
            continue;
        }
        if selected.is_some() {
            return Err(Error::StatementMismatch);
        }
        selected = Some((
            entry.id,
            entry.seq_tokens,
            entry.gt_path,
            entry.gt_bytes,
            entry.gt_sha256,
        ));
    }
    let (case_id, seq_tokens, gt_path, gt_bytes, gt_sha256) =
        selected.ok_or(Error::InvalidConfig)?;
    let gt_path = gt_path.ok_or(Error::InvalidConfig)?;
    let gt_bytes = gt_bytes.ok_or(Error::InvalidConfig)?;
    let gt_sha256 = gt_sha256.ok_or(Error::InvalidConfig)?;
    let model = case_id
        .split('-')
        .next()
        .filter(|name| matches!(*name, "gpt2" | "gemma3" | "timesfm"))
        .ok_or(Error::UnsupportedPrimitive)?
        .to_owned();
    let gt_path = root.join(gt_path);
    let expected_root = if model == "gpt2" {
        root.join("models")
    } else {
        root
    };
    let asset_root =
        std::fs::canonicalize(expected_root).map_err(|_| Error::AssetMismatch)?;
    Ok(SelectedCase {
        asset_root,
        case_id,
        model,
        seq_tokens,
        gt_path,
        gt_bytes,
        gt_sha256,
    })
}

fn check_shape(model: &str, values: usize, seq_tokens: usize) -> Result<()> {
    match model {
        "gpt2" | "gemma3" => {
            if seq_tokens == 0
                || seq_tokens > crate::policy::TOKEN_MAX_SEQUENCE
                || values != seq_tokens
            {
                return Err(Error::InvalidConfig);
            }
        }
        "timesfm" => {
            if values != crate::policy::TIMESFM_RAW_POINTS
                || seq_tokens != crate::policy::TIMESFM_PATCH_TOKENS
            {
                return Err(Error::InvalidConfig);
            }
        }
        _ => return Err(Error::UnsupportedPrimitive),
    }
    Ok(())
}

struct Confirmation {
    commitment: ModelCommitment,
    model_id: Digest,
    parameters: Digest,
    spec_version: u32,
    digests: Vec<AssetDigest>,
}

fn confirmation(
    model: &str,
    asset_root: &Path,
    values: &[i32],
) -> Result<Confirmation> {
    let tokens: Vec<u32> = values.iter().map(|value| *value as u32).collect();
    let (model_id, parameters, specs, spec_version) = match model {
        "gpt2" => (
            zkie_models_gpt2::plan::model_commitment(&tokens)?,
            zkie_models_gpt2::plan::parameters_digest()?,
            zkie_models_gpt2::plan::asset_specs(),
            zkie_models_gpt2::plan::MODEL_VERSION,
        ),
        "gemma3" => (
            zkie_models_gemma3::plan::model_commitment(&tokens)?,
            zkie_models_gemma3::plan::parameters_digest()?,
            zkie_models_gemma3::plan::asset_specs(),
            zkie_models_gemma3::plan::MODEL_VERSION,
        ),
        "timesfm" => (
            zkie_models_timesfm::plan::model_commitment(&tokens)?,
            zkie_models_timesfm::plan::parameters_digest()?,
            zkie_models_timesfm::plan::asset_specs(),
            zkie_models_timesfm::plan::MODEL_VERSION,
        ),
        _ => return Err(Error::UnsupportedPrimitive),
    };
    let spec_count = specs.len();
    let (commitment, digests) =
        commit_assets_with_digests(asset_root, model_id, parameters, &specs)?;
    if commitment.anchor.record.model_commitment != model_id
        || commitment.anchor.record.parameters_digest != parameters
        || commitment.assets.len() != spec_count
        || digests.len() != spec_count
    {
        return Err(Error::StatementMismatch);
    }
    Ok(Confirmation {
        commitment,
        model_id,
        parameters,
        spec_version,
        digests,
    })
}

fn compare_asset_hashes(previous: &serde_json::Value, current: &[AssetDigest]) -> Result<()> {
    let list = previous.get("asset_sha256").unwrap_or(previous);
    let list = list.as_array().ok_or(Error::InvalidConfig)?;
    if list.len() != current.len() {
        return Err(Error::StatementMismatch);
    }
    for (entry, digest) in list.iter().zip(current) {
        let asset_id = entry
            .get("asset_id")
            .and_then(|value| value.as_u64())
            .ok_or(Error::InvalidConfig)?;
        let path = entry
            .get("path")
            .and_then(|value| value.as_str())
            .ok_or(Error::InvalidConfig)?;
        let bytes = entry
            .get("bytes")
            .and_then(|value| value.as_u64())
            .ok_or(Error::InvalidConfig)?;
        let sha256 = entry
            .get("sha256")
            .and_then(|value| value.as_str())
            .ok_or(Error::InvalidConfig)?;
        let element_count = entry
            .get("element_count")
            .and_then(|value| value.as_u64())
            .ok_or(Error::InvalidConfig)?;
        let row_width = entry
            .get("row_width")
            .and_then(|value| value.as_u64())
            .ok_or(Error::InvalidConfig)?;
        let encoding = entry
            .get("encoding")
            .and_then(|value| value.as_str())
            .ok_or(Error::InvalidConfig)?;
        if asset_id != digest.asset_id
            || path != digest.path
            || bytes != digest.bytes
            || sha256 != digest_to_hex(&digest.sha256)
            || element_count != digest.element_count
            || row_width != digest.row_width as u64
            || encoding != digest.encoding
        {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_policy(
    path: &Path,
    case: &SelectedCase,
    confirmed: &Confirmation,
    input_len: usize,
    output_type: &str,
    output_elements: usize,
    gt_digest: Digest,
) -> Result<()> {
    let bytes = std::fs::read(path).map_err(|_| Error::Io)?;
    let document = crate::policy::parse(&bytes)?;
    let trusted = crate::policy::select(&document, &case.case_id, &case.model, input_len)?;
    if trusted.model_version != confirmed.spec_version
        || trusted.anchor_id != confirmed.commitment.anchor.record.anchor_id
        || trusted.model_id != confirmed.model_id
    {
        return Err(Error::StatementMismatch);
    }
    if trusted.output.mode == crate::policy::OutputMode::Expected
        && (trusted.output.kind.name() != output_type
            || trusted.output.elements != Some(output_elements)
            || trusted.output.sha256 != Some(gt_digest))
    {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<()> {
    let parsed = AnchorArgs::parse(args)?;
    let case = select_case(&parsed.case, &parsed.input)?;
    let asset_root = std::fs::canonicalize(&parsed.asset_root).map_err(|_| Error::AssetMismatch)?;
    if asset_root != case.asset_root {
        return Err(Error::AssetMismatch);
    }
    let values = read_input(&parsed.input)?;
    check_shape(&case.model, values.len(), case.seq_tokens)?;
    let gt = std::fs::read(&case.gt_path).map_err(|_| Error::Io)?;
    if gt.len() as u64 != case.gt_bytes {
        return Err(Error::ShapeMismatch);
    }
    let gt_digest = digest_from_hex(&case.gt_sha256)?;
    if digest_bytes(&gt) != gt_digest {
        return Err(Error::StatementMismatch);
    }
    if gt.len() % 4 != 0 {
        return Err(Error::InvalidEncoding);
    }
    let gt_values: Vec<i32> = gt
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
        .collect();
    let output_elements = match case.model.as_str() {
        "gpt2" | "gemma3" => {
            if gt_values.len() != case.seq_tokens {
                return Err(Error::ShapeMismatch);
            }
            let vocab = if case.model == "gpt2" {
                zkie_models_gpt2::plan::valid_vocab()
            } else {
                zkie_models_gemma3::plan::valid_vocab()
            };
            for value in &gt_values {
                if *value < 0 || *value as usize >= vocab {
                    return Err(Error::StatementMismatch);
                }
            }
            gt_values.len()
        }
        "timesfm" => {
            let expected = zkie_models_timesfm::plan::output_elements(case.seq_tokens)?;
            if gt_values.len() != expected {
                return Err(Error::ShapeMismatch);
            }
            gt_values.len()
        }
        _ => return Err(Error::UnsupportedPrimitive),
    };
    let output_type = if case.model == "timesfm" {
        "int_output"
    } else {
        "argmax"
    };
    let confirmed = confirmation(&case.model, &asset_root, &values)?;
    if let Some(previous) = &parsed.asset_hashes {
        let bytes = std::fs::read(previous).map_err(|_| Error::Io)?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|_| Error::InvalidEncoding)?;
        compare_asset_hashes(&value, &confirmed.digests)?;
    }
    if let Some(policy) = &parsed.policy {
        check_policy(
            policy,
            &case,
            &confirmed,
            values.len(),
            output_type,
            output_elements,
            gt_digest,
        )?;
    }
    let asset_hashes: Vec<serde_json::Value> = confirmed
        .digests
        .iter()
        .map(|digest| {
            serde_json::json!({
                "asset_id": digest.asset_id,
                "path": digest.path,
                "bytes": digest.bytes,
                "sha256": digest_to_hex(&digest.sha256),
                "element_count": digest.element_count,
                "row_width": digest.row_width,
                "encoding": digest.encoding,
            })
        })
        .collect();
    let registered_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let mut output = serde_json::json!({
        "schema": 1,
        "command": "anchor",
        "case_id": case.case_id,
        "model": case.model,
        "model_version": confirmed.spec_version,
        "spec_version": confirmed.spec_version,
        "model_id": digest_to_hex(&confirmed.model_id),
        "anchor_id": digest_to_hex(&confirmed.commitment.anchor.record.anchor_id),
        "parameters_digest": digest_to_hex(&confirmed.parameters),
        "assets_checked": confirmed.digests.len(),
        "input_elements": values.len(),
        "registered_at_unix": registered_at,
        "output_mode": "expected",
        "expected_output_sha256": case.gt_sha256,
        "output_digest_algorithm": "sha256",
        "output_type": output_type,
        "output_encoding": "i32-le",
        "output_elements": output_elements,
        "gt_bytes": case.gt_bytes,
        "asset_sha256": asset_hashes,
        "asset_baseline": "self-built change-detection baseline; not verified against externally published model material; model authorization policy pending",
    });
    if case.model == "timesfm" {
        output["raw_input_length"] = serde_json::json!(values.len());
        output["patch_tokens"] = serde_json::json!(case.seq_tokens);
    } else {
        output["sequence_length"] = serde_json::json!(values.len());
    }
    let body = serde_json::to_vec_pretty(&output).map_err(|_| Error::InvalidEncoding)?;
    if let Some(path) = &parsed.out {
        std::fs::write(path, &body).map_err(|_| Error::Io)?;
    }
    println!("{}", String::from_utf8_lossy(&body));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-anchor-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_case(root: &Path, inputs: &[(&str, &str)]) -> PathBuf {
        let assets = root.join("models");
        std::fs::create_dir_all(&assets).unwrap();
        let mut text = format!("root = \"{}\"\n", root.to_string_lossy());
        for (id, input) in inputs {
            text.push_str(&format!(
                "\n[[case]]\nid = \"{id}\"\ninput = {{ path = \"{input}\" }}\nseq_tokens = 16\ninteger_gt = {{ path = \"gt_i32.bin\", bytes = 4, sha256 = \"{}\" }}\n",
                "0".repeat(64)
            ));
        }
        let case = root.join("case.toml");
        std::fs::write(&case, text).unwrap();
        case
    }

    #[test]
    fn selects_the_entry_matching_the_input() {
        let root = temp_dir();
        std::fs::write(root.join("a_i32.bin"), vec![0u8; 64]).unwrap();
        std::fs::write(root.join("b_i32.bin"), vec![0u8; 2048]).unwrap();
        let case = write_case(&root, &[("gpt2-seq16", "a_i32.bin"), ("gpt2-seq512", "b_i32.bin")]);
        let selected = select_case(&case, &root.join("b_i32.bin")).unwrap();
        assert_eq!(selected.case_id, "gpt2-seq512");
        assert_eq!(selected.model, "gpt2");
        assert_eq!(selected.asset_root, std::fs::canonicalize(root.join("models")).unwrap());
        assert!(select_case(&case, &root.join("missing.bin")).is_err());
    }

    #[test]
    fn rejects_ambiguous_inputs_and_bad_shapes() {
        let root = temp_dir();
        std::fs::write(root.join("a_i32.bin"), vec![0u8; 64]).unwrap();
        let case = write_case(
            &root,
            &[("gpt2-seq16", "a_i32.bin"), ("gpt2-seq16", "a_i32.bin")],
        );
        assert!(select_case(&case, &root.join("a_i32.bin")).is_err());
        assert!(check_shape("gpt2", 16, 16).is_ok());
        assert!(check_shape("gpt2", 16, 512).is_err());
        assert!(check_shape("gemma3", 0, 0).is_err());
        assert!(check_shape("timesfm", 512, 16).is_ok());
        assert!(check_shape("timesfm", 16, 16).is_err());
        assert!(check_shape("unknown", 1, 1).is_err());
        std::fs::write(root.join("bad_i32.bin"), vec![0u8; 5]).unwrap();
        assert!(read_input(&root.join("bad_i32.bin")).is_err());
    }
}
