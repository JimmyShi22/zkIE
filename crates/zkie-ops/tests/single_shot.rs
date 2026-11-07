//! Single-shot: D next-token steps in one proof, compared with a step-wise
//! reference.

#[path = "support/single_shot_fixture.rs"]
mod fixture;

use zkie_ops::single_shot::{prove_single_shot, verify_single_shot};

#[test]
fn decode_lengths_one_two_four_eight_match_reference() {
    for decode_len in [1usize, 2, 4, 8] {
        let (public, openings, witnesses) = fixture::fixture(decode_len);
        let expected_tokens = public.statement.output_tokens.clone();
        let gaps = vec![0i64; decode_len];
        let proof = prove_single_shot(&public, &witnesses, &openings, gaps).unwrap();
        assert_eq!(proof.steps.len(), decode_len);
        for (index, step) in proof.steps.iter().enumerate() {
            // Independent step reference: position P-1+i and token i%V.
            assert_eq!(step.position as usize, fixture::PROMPT - 1 + index);
            assert_eq!(step.token, expected_tokens[index]);
            assert_eq!(step.token as usize, index % fixture::VOCAB);
        }
        let result = verify_single_shot(&public, &proof).unwrap();
        assert!(result.accepted);
    }
}

#[test]
fn removing_or_editing_a_step_is_rejected() {
    let decode_len = 4;
    let (public, openings, witnesses) = fixture::fixture(decode_len);
    let proof = prove_single_shot(&public, &witnesses, &openings, vec![0; decode_len]).unwrap();

    let mut missing = proof.clone();
    missing.steps.pop();
    assert!(verify_single_shot(&public, &missing).is_err());

    let mut changed_token = proof.clone();
    changed_token.steps[0].token = (changed_token.steps[0].token + 1) % 4;
    assert!(verify_single_shot(&public, &changed_token).is_err());

    let mut changed_position = proof.clone();
    changed_position.steps[1].position += 1;
    assert!(verify_single_shot(&public, &changed_position).is_err());

    let mut changed_statement = proof.clone();
    changed_statement.statement_digest = zkie_core::types::Digest([9u8; 32]);
    assert!(verify_single_shot(&public, &changed_statement).is_err());
}

#[test]
fn a_single_proof_covers_the_whole_decode() {
    let decode_len = 8;
    let (public, openings, witnesses) = fixture::fixture(decode_len);
    let proof = prove_single_shot(&public, &witnesses, &openings, vec![0; decode_len]).unwrap();
    assert_eq!(proof.steps.len(), public.statement.coverage.decode_len as usize);
    assert_eq!(
        proof.statement_digest,
        zkie_ops::proof::request::output_policy::statement_digest(&public.statement).unwrap()
    );
}

fn model_layers() -> Vec<zkie_ops::proof::model::ModelLayer> {
    use zkie_ops::proof::model::ModelLayer;
    let mut first_weights = vec![0i64; 9];
    for index in 0..3 {
        first_weights[index * 3 + index] = 1;
    }
    let mut second_weights = vec![0i64; 12];
    second_weights[0] = 1;
    vec![
        ModelLayer {
            rows: 3,
            cols: 3,
            shift: 0,
            relu: true,
            weights: first_weights,
            bias: vec![0, 0, 0],
        },
        ModelLayer {
            rows: 4,
            cols: 3,
            shift: 0,
            relu: false,
            weights: second_weights,
            bias: vec![1, 0, 0, 0],
        },
    ]
}

fn model_witness(input: Vec<i64>) -> zkie_ops::proof::model::ModelWitness {
    use zkie_ops::proof::model::{LayerWitness, ModelWitness};
    let model = model_layers();
    let mut values = input.clone();
    let mut layers = Vec::new();
    for layer in &model {
        let mut raw = Vec::new();
        let mut remainders = Vec::new();
        let mut pre = Vec::new();
        let mut output = Vec::new();
        let mut bits = Vec::new();
        for row in 0..layer.rows {
            let mut sum = layer.bias[row] as i128;
            for column in 0..layer.cols {
                sum += layer.weights[row * layer.cols + column] as i128 * values[column] as i128;
            }
            let scale = 1i128 << layer.shift;
            let quotient = sum / scale;
            let remainder = sum - quotient * scale;
            raw.push(sum);
            remainders.push(remainder as i64);
            pre.push(quotient as i64);
            if layer.relu {
                bits.push(if quotient > 0 { 1u8 } else { 0u8 });
                output.push((quotient as i64).max(0));
            } else {
                bits.push(0);
                output.push(quotient as i64);
            }
        }
        values = output.clone();
        layers.push(LayerWitness {
            input: if layers.is_empty() {
                input.clone()
            } else {
                Vec::new()
            },
            raw,
            remainders,
            pre_activation: pre,
            output,
            bits,
        });
    }
    // Fill each layer's input from the previous layer output / model input.
    let mut current = input.clone();
    for layer in &mut layers {
        layer.input = current.clone();
        current = layer.output.clone();
    }
    ModelWitness {
        input,
        layers,
        logits: current,
    }
}

#[test]
fn model_logits_are_bound_layer_by_layer() {
    use zkie_ops::proof::model::{prove_model, verify_model};
    let decode_len = 4;
    for step in 0..decode_len {
        let mut statement = fixture::statement(decode_len);
        statement.output_tokens = vec![0; decode_len];
        let input = vec![1 + step as i64, 2, 3];
        let witness = model_witness(input.clone());
        let (public, proof) =
            prove_model(&statement, &model_layers(), &witness, 0).unwrap();
        verify_model(&public, &proof).unwrap();

        // Tampering a weight row: authenticated opening no longer matches.
        let mut tampered_weights = proof.clone();
        tampered_weights.layers[1].weights[0].values[0] += 1;
        assert!(verify_model(&public, &tampered_weights).is_err());

        // Tampering an intermediate row.
        let mut tampered_row = proof.clone();
        tampered_row.layers[0].output[0].values[0] += 1;
        assert!(verify_model(&public, &tampered_row).is_err());

        // Tampering the logits row.
        let mut tampered_logits = proof.clone();
        let last = tampered_logits.layers.len() - 1;
        tampered_logits.layers[last].output[0].values[0] += 1;
        assert!(verify_model(&public, &tampered_logits).is_err());

        // Tampering relation parameters: shift and weight bounds.
        let mut changed_shift = public.clone();
        changed_shift.layers[0].2 = 1;
        assert!(verify_model(&changed_shift, &proof).is_err());
        let mut changed_bound = public.clone();
        changed_bound.weight_bounds = (0, 0);
        assert!(verify_model(&changed_bound, &proof).is_err());

        // Breaking the layer edge: layer 1 must consume layer 0's output.
        let mut broken_edge = proof.clone();
        broken_edge.layers[1].input = broken_edge.layers[0].input.clone();
        assert!(verify_model(&public, &broken_edge).is_err());

        // Changing the output token expectation.
        let mut changed_token = public.clone();
        changed_token.expected_token = 1;
        assert!(verify_model(&changed_token, &proof).is_err());
        assert_eq!(input.len(), 3);
    }
}

#[test]
fn nonlinear_stages_are_bound_to_model_logits_and_tamper_rejects() {
    use zkie_ops::argmax::ArgmaxOp;
    use zkie_ops::causal_attention::CausalAttentionOp;
    use zkie_ops::kv_cache::KvCacheOp;
    use zkie_ops::layer_norm::LayerNormOp;
    use zkie_ops::patchify::PatchifyOp;
    use zkie_ops::proof::model::prove_model;
    use zkie_ops::proof::model_chain::{
        prove_op_chain, value_binding, verify_op_chain, OpChainStage,
    };
    use zkie_ops::proof::relations::RelationConfig;
    use zkie_ops::registry::PureForward;
    use zkie_ops::rms_norm::RmsNormOp;
    use zkie_ops::softmax::SoftmaxOp;
    use zkie_ops::topk::TopKOp;

    fn stage<T: PureForward + RelationConfig + 'static>(
        op: T,
        inputs: Vec<Vec<i64>>,
    ) -> (OpChainStage, Vec<Vec<i64>>) {
        let config = op.relation_config(&inputs).unwrap();
        let outputs = op.forward_values(&inputs).unwrap().outputs;
        (
            OpChainStage {
                op: Box::new(op),
                config,
                inputs,
            },
            outputs,
        )
    }

    let decode_len = 1;
    let mut statement = fixture::statement(decode_len);
    statement.output_tokens = vec![0; decode_len];
    let witness = model_witness(vec![1, 2, 3]);
    let (_model_public, model_proof) =
        prove_model(&statement, &model_layers(), &witness, 0).unwrap();
    let logits = witness.logits.clone();
    let logits_from_proof: Vec<i64> = model_proof
        .layers
        .last()
        .unwrap()
        .output
        .iter()
        .map(|opening| opening.values[0] as i64)
        .collect();
    assert_eq!(logits_from_proof, logits);
    let logits_binding = value_binding(&[logits_from_proof]);

    let (softmax, probs) = stage(
        SoftmaxOp { shift: 4 },
        vec![logits.clone(), vec![1, 1, 1, 1]],
    );
    let (norm, normalized) = stage(
        LayerNormOp {
            rows: 1,
            cols: 4,
            valid_cols: 4,
            shift: 2,
            clamp: None,
        },
        vec![probs[0].clone(), vec![1, 1, 1, 1]],
    );
    let (rms, rms_out) = stage(
        RmsNormOp {
            rows: 1,
            cols: 4,
            valid_cols: 4,
            shift: 2,
            clamp: None,
        },
        vec![normalized[0].clone(), vec![1, 1, 1, 1]],
    );
    let (topk, topk_out) = stage(TopKOp { k: 2 }, vec![rms_out[0].clone()]);
    let (argmax, _) = stage(ArgmaxOp, vec![topk_out[0].clone()]);
    let stages = vec![softmax, norm, rms, topk, argmax];
    let (chain_public, chain_proof) = prove_op_chain(logits_binding, &stages).unwrap();
    let final_values = verify_op_chain(&chain_public, &chain_proof, logits_binding).unwrap();
    assert_eq!(final_values.len(), 1);

    let mut tampered_output = chain_proof.clone();
    tampered_output.stages[0].outputs[0][0].values[0] =
        tampered_output.stages[0].outputs[0][0].values[0].wrapping_add(1);
    assert!(verify_op_chain(&chain_public, &tampered_output, logits_binding).is_err());

    let mut tampered_aux = chain_proof.clone();
    tampered_aux.stages[0].aux[1][0].values[0] =
        tampered_aux.stages[0].aux[1][0].values[0].wrapping_add(1);
    assert!(verify_op_chain(&chain_public, &tampered_aux, logits_binding).is_err());

    let mut tampered_config = chain_public.clone();
    tampered_config.stages[1].config[0] = 2;
    assert!(verify_op_chain(&tampered_config, &chain_proof, logits_binding).is_err());

    let mut broken_edge = chain_proof.clone();
    broken_edge.stages[1].inputs[0][0].values[0] =
        broken_edge.stages[1].inputs[0][0].values[0].wrapping_add(1);
    assert!(verify_op_chain(&chain_public, &broken_edge, logits_binding).is_err());

    assert!(verify_op_chain(
        &chain_public,
        &chain_proof,
        zkie_core::types::Digest([0xEE; 32])
    )
    .is_err());

    // Attention and KV cache stages are also checked as relations and bound to
    // the same model logits root through their first input oracle.
    let (attention, attention_out) = stage(
        CausalAttentionOp {
            seq: 2,
            head_dim: 2,
            shift: 0,
        },
        vec![
            logits.clone(),
            vec![1, 0, 0, 1],
            vec![10, 20, 30, 40],
            vec![1, 1, 1],
        ],
    );
    let (patch, _) = stage(PatchifyOp { patch: 2 }, vec![attention_out[0].clone()]);
    let (attention_public, attention_proof) =
        prove_op_chain(logits_binding, &[attention, patch]).unwrap();
    verify_op_chain(&attention_public, &attention_proof, logits_binding).unwrap();
    let mut bad_attention = attention_proof.clone();
    bad_attention.stages[0].aux[2][0].values[0] =
        bad_attention.stages[0].aux[2][0].values[0].wrapping_add(1);
    assert!(verify_op_chain(&attention_public, &bad_attention, logits_binding).is_err());

    let (kv, _) = stage(
        KvCacheOp {
            capacity: 2,
            width: 2,
            position: 1,
        },
        vec![logits.clone(), logits.clone(), vec![1, 2], vec![3, 4]],
    );
    let (kv_public, kv_proof) = prove_op_chain(logits_binding, &[kv]).unwrap();
    verify_op_chain(&kv_public, &kv_proof, logits_binding).unwrap();
    let mut bad_kv = kv_proof.clone();
    bad_kv.stages[0].outputs[0][0].values[0] =
        bad_kv.stages[0].outputs[0][0].values[0].wrapping_add(1);
    assert!(verify_op_chain(&kv_public, &bad_kv, logits_binding).is_err());
}
