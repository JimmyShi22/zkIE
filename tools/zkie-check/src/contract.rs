//! Contract check for the scope document and acceptance matrix.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use crate::parse_flags;

pub fn main(args: &[String]) -> i32 {
    match check(args) {
        Ok(summary) => {
            println!("{}", summary.to_json());
            0
        }
        Err(error) => {
            eprintln!("contract check failed: {error}");
            1
        }
    }
}

#[derive(Default)]
pub struct Summary {
    pub requirements: usize,
    pub planned: usize,
    pub blocked: usize,
    pub deferred: usize,
}

impl Summary {
    fn to_json(&self) -> String {
        format!(
            "{{\"ok\":true,\"requirements\":{},\"planned\":{},\"blocked\":{},\"deferred\":{}}}",
            self.requirements, self.planned, self.blocked, self.deferred
        )
    }
}

pub fn check(args: &[String]) -> Result<Summary, String> {
    let flags = parse_flags(args, &["scope", "matrix"])?;
    let scope_path = flags.get("scope").ok_or("missing --scope")?;
    let matrix_path = flags.get("matrix").ok_or("missing --matrix")?;

    let scope = fs::read_to_string(scope_path)
        .map_err(|error| format!("cannot read scope: {error}"))?;
    if scope.trim().is_empty() || !scope.contains("must-do matrix") {
        return Err("scope document is missing the case matrix section".into());
    }

    let matrix = fs::read_to_string(matrix_path)
        .map_err(|error| format!("cannot read matrix: {error}"))?;
    let requirements = parse_requirements(&matrix)?;
    if requirements.is_empty() {
        return Err("matrix contains no requirements".into());
    }

    let mut summary = Summary {
        requirements: requirements.len(),
        ..Summary::default()
    };
    let mut ids = BTreeSet::new();
    for requirement in &requirements {
        for key in [
            "id",
            "category",
            "statement",
            "doc",
            "files",
            "task",
            "acceptance",
            "evidence",
            "status",
        ] {
            if !requirement.contains_key(key) {
                return Err(format!("requirement is missing key `{key}`"));
            }
        }
        let id = requirement.get("id").unwrap();
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate requirement id `{id}`"));
        }
        match requirement.get("category").map(String::as_str) {
            Some("correctness" | "resource" | "performance" | "deferred") => {}
            other => return Err(format!("invalid category `{other:?}` for `{id}`")),
        }
        match requirement.get("status").map(String::as_str) {
            Some("planned") => summary.planned += 1,
            Some("blocked") => summary.blocked += 1,
            Some("deferred") => summary.deferred += 1,
            other => return Err(format!("invalid status `{other:?}` for `{id}`")),
        }
    }
    Ok(summary)
}

fn parse_requirements(text: &str) -> Result<Vec<BTreeMap<String, String>>, String> {
    let mut requirements = Vec::new();
    let mut current: Option<BTreeMap<String, String>> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "[[requirement]]" {
            if let Some(previous) = current.take() {
                requirements.push(previous);
            }
            current = Some(BTreeMap::new());
            continue;
        }
        let Some(requirement) = current.as_mut() else {
            continue;
        };
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || key.starts_with('#') {
            continue;
        }
        requirement.insert(key.to_owned(), unquote(value.trim()));
    }
    if let Some(previous) = current {
        requirements.push(previous);
    }
    Ok(requirements)
}

fn unquote(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('"')
        .trim_end_matches('"')
        .to_owned()
}
