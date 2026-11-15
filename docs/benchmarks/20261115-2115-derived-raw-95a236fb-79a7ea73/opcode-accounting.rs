use std::fs::File;
use std::io::BufReader;

use zkie_ops::op::OpCode;
use zkie_ops::proof::model_commitment::SharedTrace;

fn derived(opcode: u16) -> Option<&'static str> {
    const SET: [(OpCode, &str); 9] = [
        (OpCode::AffineScale, "AffineScale"),
        (OpCode::Softmax, "Softmax"),
        (OpCode::SoftmaxRows, "SoftmaxRows"),
        (OpCode::Hadamard, "Hadamard"),
        (OpCode::GatedFfn, "GatedFfn"),
        (OpCode::ProfileLayerNorm, "ProfileLayerNorm"),
        (OpCode::ProfileRmsNorm, "ProfileRmsNorm"),
        (OpCode::ProfileRope, "ProfileRope"),
        (OpCode::ProfileStandardize, "ProfileStandardize"),
    ];
    SET.iter()
        .find(|(code, _)| *code as u16 == opcode)
        .map(|(_, name)| *name)
}

fn account(name: &str, path: &str) {
    let file = File::open(path).unwrap();
    let trace: SharedTrace = bincode::deserialize_from(BufReader::new(file)).unwrap();
    let mut rows: Vec<(&'static str, usize, u64)> = Vec::new();
    for step in &trace.steps {
        let Some(opcode) = derived(step.message.opcode) else {
            continue;
        };
        let values: u64 = step
            .message
            .proof
            .lengths
            .raw
            .iter()
            .map(|value| *value as u64)
            .sum();
        match rows.iter_mut().find(|(name, _, _)| *name == opcode) {
            Some((_, count, total)) => {
                *count += 1;
                *total += values;
            }
            None => rows.push((opcode, 1, values)),
        }
    }
    let mut total_steps = 0usize;
    let mut total_values = 0u64;
    println!("{name}:");
    for (opcode, count, values) in &rows {
        total_steps += count;
        total_values += values;
        println!(
            "  {opcode}: steps={count} raw_values={values} value_bytes={}",
            values * 16
        );
    }
    println!(
        "  total: steps={total_steps} raw_values={total_values} value_bytes={}",
        total_values * 16
    );
}

#[test]
fn derived_raw_accounting() {
    let base = std::env::var("DERIVED_TRACE_DIR").expect("the trace directory");
    for case in [
        "gpt2-seq16",
        "gpt2-seq512",
        "gemma3-seq16",
        "gemma3-seq512",
        "timesfm-16patch",
    ] {
        let path = format!("{base}/before/{case}/proof/proof.bin");
        account(case, &path);
    }
}
