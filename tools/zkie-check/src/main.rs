//! zkie-check: contract, asset and run-report validation.

mod assets;
mod contract;
mod report;

use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("contract") => contract::main(&args[1..]),
        Some("assets") => assets::main(&args[1..]),
        Some("report") => report::main(&args[1..]),
        _ => {
            print_usage();
            2
        }
    };
    std::process::exit(code);
}

pub(crate) fn parse_flags(
    args: &[String],
    allowed: &[&str],
) -> Result<BTreeMap<String, String>, String> {
    let mut flags = BTreeMap::new();
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let name = flag
            .strip_prefix("--")
            .filter(|name| !name.is_empty())
            .ok_or_else(|| format!("unexpected argument: {flag}"))?;
        if !allowed.contains(&name) {
            return Err(format!("unknown flag: --{name}"));
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value for --{name}"))?;
        if flags.insert(name.to_owned(), value.clone()).is_some() {
            return Err(format!("duplicate flag: --{name}"));
        }
        index += 2;
    }
    Ok(flags)
}

fn print_usage() {
    println!(
        "usage:\n  zkie-check contract --scope <scope.md> --matrix <acceptance-matrix.toml>\n  zkie-check assets --root <asset-root> --cases <case.toml>... --out <path>\n  zkie-check report --input <run-dir> --require-profile <name>"
    );
}
