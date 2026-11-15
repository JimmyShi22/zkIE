use zkie_ops::op::OpCode;
use zkie_ops::proof::plan::{PlanConfigValue, TracePlan};

fn literal(config: &[PlanConfigValue], index: usize) -> i64 {
    match config.get(index) {
        Some(PlanConfigValue::Literal(value)) => *value,
        _ => panic!("the matrix geometry is literal"),
    }
}

fn account(name: &str, plan: &TracePlan) {
    let mut rows: Vec<(OpCode, usize, u64)> = Vec::new();
    for step in &plan.steps {
        if !matches!(
            step.opcode,
            OpCode::MatMul | OpCode::Projection | OpCode::RegressionHead
        ) {
            continue;
        }
        let values = (literal(&step.config, 0) as u64) * (literal(&step.config, 1) as u64);
        match rows.iter_mut().find(|(opcode, _, _)| *opcode == step.opcode) {
            Some((_, count, total)) => {
                *count += 1;
                *total += values;
            }
            None => rows.push((step.opcode, 1, values)),
        }
    }
    let mut total_values = 0u64;
    let mut total_steps = 0usize;
    println!("{name}:");
    for (opcode, count, values) in &rows {
        total_steps += count;
        total_values += values;
        println!(
            "  {opcode:?}: steps={count} raw_values={values} value_bytes={}",
            values * 16
        );
    }
    println!(
        "  total: steps={total_steps} raw_values={total_values} value_bytes={}",
        total_values * 16
    );
}

#[test]
fn raw_dedup_accounting() {
    account(
        "gpt2 seq16",
        &zkie_models_gpt2::plan::trace_plan(16).unwrap(),
    );
    account(
        "gpt2 seq512",
        &zkie_models_gpt2::plan::trace_plan(512).unwrap(),
    );
    account(
        "gemma3 seq16",
        &zkie_models_gemma3::plan::trace_plan(16).unwrap(),
    );
    account(
        "gemma3 seq512",
        &zkie_models_gemma3::plan::trace_plan(512).unwrap(),
    );
    account(
        "timesfm 16-patch",
        &zkie_models_timesfm::plan::trace_plan(16).unwrap(),
    );
}
