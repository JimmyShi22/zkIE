//! Asset check: every referenced path must exist, match its declared byte
//! length and SHA-256. Blocked GT entries carry no path and are not silently
//! treated as PASS.

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[derive(Default, Debug)]
struct Record {
    path: Option<String>,
    bytes: Option<u64>,
    sha256: Option<String>,
}

#[derive(Default, Debug)]
pub struct CheckReport {
    pub checked: usize,
    pub missing: usize,
    pub mismatched: usize,
    pub bad_patch_tokens: usize,
}

pub fn main(args: &[String]) -> i32 {
    match check(args) {
        Ok(report) => {
            println!(
                "{{\"ok\":{},\"checked\":{},\"missing\":{},\"mismatched\":{},\"bad_patch_tokens\":{}}}",
                report.missing == 0 && report.mismatched == 0 && report.bad_patch_tokens == 0,
                report.checked,
                report.missing,
                report.mismatched,
                report.bad_patch_tokens
            );
            if report.missing == 0 && report.mismatched == 0 && report.bad_patch_tokens == 0 {
                0
            } else {
                1
            }
        }
        Err(error) => {
            eprintln!("asset check failed: {error}");
            1
        }
    }
}

pub fn check(args: &[String]) -> Result<CheckReport, String> {
    let mut root: Option<PathBuf> = None;
    let mut cases: Vec<String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--root" => {
                index += 1;
                root = Some(PathBuf::from(args.get(index).ok_or("missing value for --root")?));
            }
            "--cases" => {
                index += 1;
                while index < args.len() && !args[index].starts_with("--") {
                    cases.push(args[index].clone());
                    index += 1;
                }
                if cases.is_empty() {
                    return Err("missing value for --cases".into());
                }
                continue;
            }
            "--out" => {
                index += 1;
                out = Some(PathBuf::from(args.get(index).ok_or("missing value for --out")?));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }
    let root = root.ok_or("missing --root")?;
    if !root.is_dir() {
        return Err(format!("asset root is not a directory: {}", root.display()));
    }
    let mut report = CheckReport::default();
    let mut unique = std::collections::BTreeSet::new();
    for case in cases {
        let text = fs::read_to_string(&case).map_err(|error| format!("{case}: {error}"))?;
        let case_root = parse_root(&text).unwrap_or_else(|| root.clone());
        check_patch_tokens(&text, &mut report)?;
        for record in records(&text) {
            let path = match record.path {
                Some(path) => path,
                None => continue,
            };
            report.checked += 1;
            let resolved = if Path::new(&path).is_absolute() {
                PathBuf::from(&path)
            } else {
                case_root.join(&path)
            };
            if !resolved.exists() {
                eprintln!("missing asset: {}", resolved.display());
                report.missing += 1;
                continue;
            }
            let metadata = fs::metadata(&resolved).map_err(|error| format!("{}: {error}", resolved.display()))?;
            if let Some(expected) = record.bytes {
                if metadata.len() != expected {
                    eprintln!("length mismatch: {} expected {} got {}", resolved.display(), expected, metadata.len());
                    report.mismatched += 1;
                    continue;
                }
            }
            if let Some(expected) = record.sha256 {
                let actual = sha256_file(&resolved)?;
                if actual != expected.to_ascii_lowercase() {
                    eprintln!("digest mismatch: {}", resolved.display());
                    report.mismatched += 1;
                    continue;
                }
            }
            unique.insert(resolved);
        }
    }
    if let Some(out) = out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
        }
        let body = format!(
            "{{\n  \"checked\": {},\n  \"unique_files\": {},\n  \"missing\": {},\n  \"mismatched\": {},\n  \"bad_patch_tokens\": {}\n}}\n",
            report.checked,
            unique.len(),
            report.missing,
            report.mismatched,
            report.bad_patch_tokens
        );
        fs::write(&out, body).map_err(|error| format!("{}: {error}", out.display()))?;
    }
    Ok(report)
}

fn check_patch_tokens(text: &str, report: &mut CheckReport) -> Result<(), String> {
    let mut in_case = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[[") {
            in_case = trimmed.starts_with("[[case]]");
            continue;
        }
        if !in_case {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("patch_tokens = ") {
            let value: usize = rest.trim().parse().map_err(|_| "invalid patch_tokens".to_string())?;
            if value != 16 {
                report.bad_patch_tokens += 1;
            }
        }
    }
    Ok(())
}

fn records(text: &str) -> Vec<Record> {
    let mut records = Vec::new();
    let mut current: Option<Record> = None;
    let mut flush = |current: &mut Option<Record>, records: &mut Vec<Record>| {
        if let Some(record) = current.take() {
            if record.path.is_some() {
                records.push(record);
            }
        }
    };
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[[") {
            flush(&mut current, &mut records);
            continue;
        }
        if trimmed.starts_with("path = \"") {
            flush(&mut current, &mut records);
            current = Some(Record::default());
        }
        if let Some(record) = current.as_mut() {
            if let Some(rest) = trimmed.strip_prefix("path = \"") {
                if let Some(end) = rest.find(char::from(34)) {
                    record.path = Some(rest[..end].to_owned());
                }
            }
            if let Some(rest) = trimmed.strip_prefix("bytes = ") {
                record.bytes = rest.trim().parse().ok();
            }
            if let Some(rest) = trimmed.strip_prefix("sha256 = \"") {
                if let Some(end) = rest.find(char::from(34)) {
                    record.sha256 = Some(rest[..end].to_owned());
                }
            }
        }
    }
    flush(&mut current, &mut records);
    records
}

fn parse_root(text: &str) -> Option<PathBuf> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("root = ") {
            return Some(PathBuf::from(unquote(rest)));
        }
    }
    None
}

fn unquote(value: &str) -> String {
    value.trim().trim_matches(char::from(34)).to_owned()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    use std::io::Read;
    loop {
        let read = file.read(&mut buffer).map_err(|error| format!("{}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    Ok(text)
}
