//! Read-only consumer inventory for the proof-size consumer closure step.
//!
//! Enumerates every plan-level asset consumer for the production plans of the
//! three models and prints a per-asset consumer list. It does not run any
//! prover, verifier or key material.

use std::collections::{BTreeMap, BTreeSet};

use zkie_ops::op::OpCode;
use zkie_ops::proof::model_commitment::ModelAssetSpec;
use zkie_ops::proof::plan::{PlanSource, TracePlan};

fn collect_assets(source: &PlanSource, out: &mut Vec<String>) {
    match source {
        PlanSource::Asset { path, .. } => out.push(path.clone()),
        PlanSource::Layout(inner, _) => collect_assets(inner, out),
        PlanSource::Concat(parts) | PlanSource::HeadConcat { parts, .. } => {
            for part in parts {
                collect_assets(part, out);
            }
        }
        PlanSource::Prev { .. } | PlanSource::Constant(_) => {}
    }
}

fn op_name(code: OpCode) -> String {
    format!("{code:?}")
}

fn main() {
    let plans: Vec<(&str, &str, usize, TracePlan)> = vec![
        ("gpt2", "gpt2-seq16", 16, zkie_models_gpt2::plan::trace_plan(16).unwrap()),
        ("gpt2", "gpt2-seq512", 512, zkie_models_gpt2::plan::trace_plan(512).unwrap()),
        ("gemma3", "gemma3-seq16", 16, zkie_models_gemma3::plan::trace_plan(16).unwrap()),
        ("gemma3", "gemma3-seq512", 512, zkie_models_gemma3::plan::trace_plan(512).unwrap()),
        ("timesfm", "timesfm-16patch", 16, zkie_models_timesfm::plan::trace_plan(16).unwrap()),
    ];

    let mut consumers: BTreeMap<String, Vec<(String, String, usize, String, usize)>> =
        BTreeMap::new();
    for (model, case, _n, plan) in &plans {
        for (step_index, step) in plan.steps.iter().enumerate() {
            for (port, input) in step.inputs.iter().enumerate() {
                let mut assets = Vec::new();
                collect_assets(input, &mut assets);
                for path in assets {
                    consumers.entry(path).or_default().push((
                        (*model).to_string(),
                        (*case).to_string(),
                        step_index,
                        op_name(step.opcode),
                        port,
                    ));
                }
            }
        }
    }

    println!("asset\tconsumers\tby_op\tcases");
    for (path, uses) in &consumers {
        let mut by_op: BTreeMap<String, usize> = BTreeMap::new();
        let mut cases: BTreeSet<String> = BTreeSet::new();
        for (_, case, _, op, _) in uses {
            *by_op.entry(op.clone()).or_insert(0) += 1;
            cases.insert(case.clone());
        }
        let ops = by_op
            .iter()
            .map(|(op, count)| format!("{op}:{count}"))
            .collect::<Vec<_>>()
            .join(",");
        let cases = cases.into_iter().collect::<Vec<_>>().join(",");
        println!("{path}\t{}\t{ops}\t{cases}", uses.len());
    }

    println!();
    println!("detail ass\tmodel\tcase\tstep\top\tport");
    let mut detail: Vec<(&String, &(String, String, usize, String, usize))> = Vec::new();
    for (path, uses) in &consumers {
        for use_ in uses {
            detail.push((path, use_));
        }
    }
    detail.sort_by(|a, b| {
        (a.0, &a.1 .1, a.1 .2, &a.1 .3, a.1 .4)
            .cmp(&(b.0, &b.1 .1, b.1 .2, &b.1 .3, b.1 .4))
    });
    for (path, (model, case, step, op, port)) in detail {
        println!("{path}\t{model}\t{case}\t{step}\t{op}\t{port}");
    }

    println!();
    println!("assets without a plan consumer");
    let specs: Vec<(&str, Vec<ModelAssetSpec>)> = vec![
        ("gpt2", zkie_models_gpt2::plan::asset_specs()),
        ("gemma3", zkie_models_gemma3::plan::asset_specs()),
        ("timesfm", zkie_models_timesfm::plan::asset_specs()),
    ];
    for (model, specs) in specs {
        for spec in specs {
            if !consumers.contains_key(&spec.path) {
                println!("{model}\t{}\tunused", spec.path);
            }
        }
    }
}
