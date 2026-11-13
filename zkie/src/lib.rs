//! zkie library: three-stage command implementations and measurement.

pub mod anchor;
mod bounded_trace;
pub mod commitment_ledger;
pub mod commit;
pub mod output;
pub mod provider;
pub mod progress;
pub mod policy;
pub mod profile;
pub mod prove;
pub mod report;
pub mod run;
pub mod s7_context;
pub mod lm_head_key;
pub mod lm_head_stage;
pub mod verify;

use provider::production_provider;

pub fn run_cli(args: Vec<String>) -> i32 {
    let Some(command) = args.first().map(String::as_str) else {
        print_usage();
        return 0;
    };
    let rest = &args[1..];
    match command {
        "anchor" => exit_code(anchor::run(rest)),
        "commit" => exit_code(commit::run(rest, &production_provider())),
        "prove" => exit_code(prove::run(rest, &production_provider())),
        "verify" => match verify::run(rest, &production_provider()) {
            Ok(receipt) if receipt.accepted => 0,
            Ok(_) => 1,
            Err(error) => {
                eprintln!("{error}");
                2
            }
        },
        "matrix" => match run::matrix(rest) {
            Ok(_) => 0,
            Err(error) => { eprintln!("{error}"); 2 }
        },
        "decode-matrix" => match run::decode_matrix(rest) {
            Ok(_) => 0,
            Err(error) => { eprintln!("{error}"); 2 }
        },
        "profile-matrix" => match profile::profile_matrix(rest) {
            Ok(_) => 0,
            Err(error) => { eprintln!("{error}"); 2 }
        },
        "run" => match run::run(rest) {
            Ok(codes) if codes.iter().all(|code| *code == 0) => 0,
            Ok(_) => 1,
            Err(error) => {
                eprintln!("{error}");
                2
            }
        },
        "help" | "--help" | "-h" => {
            print_usage();
            0
        }
        _ => {
            eprintln!("unknown command: {command}");
            print_usage();
            2
        }
    }
}

fn exit_code<T>(result: zkie_core::error::Result<T>) -> i32 {
    match result {
        Ok(_) => 0,
        Err(error) => {
            eprintln!("{error}");
            2
        }
    }
}

pub fn print_usage() {
    println!(
        "usage:\n  zkie anchor --case <path> --input <path> --asset-root <path> [--out <path>]\n  zkie matrix --cases <case.toml>... --cold-runs 1 --warm-runs 3 --out <dir>\n  zkie decode-matrix --cases <case.toml>... --prompt-len 16 --decode-lengths 16 512 --out <dir>\n  zkie profile-matrix --cases <case.toml>... --threads 1 8 64 --runs 3 --out <dir>\n  zkie commit --case <path> --assets-manifest <path> --parameters <path> --out <dir> --run-dir <dir>\n  zkie prove --commit <dir> --input <path> --model-id <hex> --out <dir> --run-dir <dir>\n  zkie verify --public <dir> --statement <path> --proof <path> --public-inputs <path> --model-id <hex> --policy <path> --run-dir <dir>\n  zkie run --commit-policy create|reuse [--commit-dir <dir>] --case <path> --assets-manifest <path> --parameters <path> --input <path> --model-id <hex> --policy <path> --run-dir <dir>"
    );
}
