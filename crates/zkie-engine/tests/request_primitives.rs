//! Golden-vector and rejection checks for chain, PoW and code primitives.

use serde_json::Value;
use zkie_core::artifact::digest_from_hex;
use zkie_core::error::Error;
use zkie_core::types::Digest;

use zkie_engine::pcs::request::chain::{
    Chain, ChainPrefix, KIND_ALGEBRAIC, KIND_SEAL,
};
use zkie_engine::pcs::request::code::{
    build_code_plan, encode_row_fp, encode_row_fp3, evaluate_ood_fp, evaluate_ood_fp3, point,
    CodeParams,
};
use zkie_engine::pcs::request::field::{fp_from_u64, fp_to_u64, Fp3};
use zkie_engine::pcs::request::params::{PowPolicy, RequestParams, REQUEST_CHAIN_LABEL};
use zkie_engine::pcs::request::pow::{search_nonce, verify_nonce, work_test};

fn vectors() -> Value {
    serde_json::from_str(include_str!("vectors/request/primitives.json")).unwrap()
}

fn hex_bytes(value: &str) -> Vec<u8> {
    (0..value.len() / 2)
        .map(|index| u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap())
        .collect()
}

fn decimal(value: &Value) -> u64 {
    if let Some(text) = value.as_str() {
        text.parse().unwrap()
    } else {
        value.as_u64().unwrap()
    }
}

fn digest32(value: &Value) -> Digest {
    digest_from_hex(value.as_str().unwrap()).unwrap()
}

fn words(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(decimal)
        .collect()
}

#[test]
fn chain_start_matches_vectors() {
    let vectors = vectors();
    for case in vectors["chain_start"].as_array().unwrap() {
        let parameters = digest32(&case["parameters_hex"]).0;
        let parameters = Digest(parameters);
        let statement = hex_bytes(case["statement_hex"].as_str().unwrap());
        let record = hex_bytes(case["record_hex"].as_str().unwrap());
        let random = case["random_hex"].as_str().map(|text| {
            let bytes = hex_bytes(text);
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bytes);
            Digest(digest)
        });
        let chain = Chain::begin(
            case["label"].as_u64().unwrap(),
            &parameters,
            &statement,
            &record,
            random.as_ref(),
        )
        .unwrap();
        assert_eq!(
            chain.value(),
            &digest_from_hex(case["rho_hex"].as_str().unwrap()).unwrap()
        );
    }
}

#[test]
fn round_calls_match_vectors_and_bind_position() {
    let vectors = vectors();
    for case in vectors["round_call"].as_array().unwrap() {
        let prefix = ChainPrefix {
            rho_prev: digest_from_hex(case["rho_hex"].as_str().unwrap()).unwrap(),
            label: case["label"].as_u64().unwrap(),
            round: case["round"].as_u64().unwrap(),
            message: words(&case["message"]),
            kind: case["kind"].as_u64().unwrap() as u32,
            work_bits: case["work_bits"].as_u64().unwrap() as u8,
        };
        let nonce = decimal(&case["nonce"]);
        let call = prefix.call(nonce, 0).unwrap();
        assert_eq!(call.to_vec(), hex_bytes(case["call0_hex"].as_str().unwrap()));

        let mut moved = prefix.clone();
        moved.round += 1;
        assert_ne!(moved.call(nonce, 0).unwrap(), call);
        let mut other_message = prefix.clone();
        other_message.message.push(99);
        assert_ne!(other_message.call(nonce, 0).unwrap(), call);
    }
}

#[test]
fn verifier_draws_match_vectors() {
    let vectors = vectors();
    let case = &vectors["draws"];
    let label = case["label"].as_u64().unwrap();
    let rho = digest_from_hex(case["rho_hex"].as_str().unwrap()).unwrap();
    let message = words(&case["message"]);
    let kind = case["kind"].as_u64().unwrap() as u32;

    let mut chain = Chain::resume(label, rho);
    let mut open = chain.message();
    for word in &message {
        open.push_u64(*word);
    }
    let prefix = open.prefix(kind, 0).unwrap();
    assert!(chain.step(&prefix, 0).unwrap());

    let calls = case["calls_hex"].as_array().unwrap();
    let first = hex_bytes(calls[0].as_str().unwrap());
    let mut expected_value = [0u8; 32];
    expected_value.copy_from_slice(&first[..32]);
    assert_eq!(chain.value(), &Digest(expected_value));
    for (index, call) in calls.iter().enumerate() {
        assert_eq!(
            prefix.call(0, index as u64).unwrap().to_vec(),
            hex_bytes(call.as_str().unwrap())
        );
    }

    assert_eq!(fp_to_u64(chain.draw_f().unwrap()), decimal(&case["f"]));
    let extension = chain.draw_e().unwrap();
    let expected = case["e"].as_array().unwrap();
    assert_eq!(fp_to_u64(extension.c[0]), decimal(&expected[0]));
    assert_eq!(fp_to_u64(extension.c[1]), decimal(&expected[1]));
    assert_eq!(fp_to_u64(extension.c[2]), decimal(&expected[2]));
    let nonzero = chain.draw_e_nonzero().unwrap();
    let expected = case["e_nonzero"].as_array().unwrap();
    assert_eq!(fp_to_u64(nonzero.c[0]), decimal(&expected[0]));
    assert_eq!(fp_to_u64(nonzero.c[1]), decimal(&expected[1]));
    assert_eq!(fp_to_u64(nonzero.c[2]), decimal(&expected[2]));
    assert!(!nonzero.is_zero());
    let log_n = case["position_log_n"].as_u64().unwrap() as u32;
    assert_eq!(chain.draw_position(log_n).unwrap(), decimal(&case["position"]));
}

#[test]
fn proof_of_work_search_and_verification() {
    let vectors = vectors();
    let case = &vectors["pow"];
    let prefix = ChainPrefix {
        rho_prev: digest_from_hex(case["rho_hex"].as_str().unwrap()).unwrap(),
        label: case["label"].as_u64().unwrap(),
        round: case["round"].as_u64().unwrap(),
        message: words(&case["message"]),
        kind: case["kind"].as_u64().unwrap() as u32,
        work_bits: case["work_bits"].as_u64().unwrap() as u8,
    };
    let expected_nonce = decimal(&case["nonce"]);
    assert_eq!(search_nonce(&prefix, 1_000_000).unwrap(), expected_nonce);
    assert!(verify_nonce(&prefix, expected_nonce, prefix.work_bits).unwrap());
    assert!(!verify_nonce(&prefix, expected_nonce, prefix.work_bits - 1).unwrap());

    let mut wrong_position = prefix.clone();
    wrong_position.round += 1;
    assert_ne!(
        prefix.first_word(expected_nonce).unwrap(),
        wrong_position.first_word(expected_nonce).unwrap()
    );

    assert!(work_test(0, 0).unwrap());
    assert!(work_test(0, 64).unwrap());
    assert!(!work_test(1, 1).unwrap());
    assert!(matches!(work_test(0, 65), Err(Error::InvalidConfig)));

    let mut empty = prefix.clone();
    empty.work_bits = 0;
    assert_eq!(search_nonce(&empty, 4).unwrap(), 0);
    assert!(!verify_nonce(&empty, 1, 0).unwrap());
    assert!(matches!(search_nonce(&prefix, 0), Err(Error::ResourceLimit)));
}

#[test]
fn code_plan_and_encoding_match_vectors() {
    let vectors = vectors();
    for case in vectors["code"].as_array().unwrap() {
        let plan = build_code_plan(
            case["log_n"].as_u64().unwrap() as u32,
            case["ell"].as_u64().unwrap(),
            case["r"].as_u64().unwrap(),
        )
        .unwrap();
        let message: Vec<_> = case["message"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| fp_from_u64(decimal(value)))
            .collect();
        let tail: Vec<_> = case["tail"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| fp_from_u64(decimal(value)))
            .collect();
        let word = encode_row_fp(&plan, &message, &tail).unwrap();
        let expected: Vec<_> = case["word"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| decimal(value))
            .collect();
        assert_eq!(
            word.iter().map(|value| fp_to_u64(*value)).collect::<Vec<_>>(),
            expected
        );
        let z = fp_from_u64(decimal(&case["ood_z"]));
        assert_eq!(
            fp_to_u64(evaluate_ood_fp(&plan, &message, &tail, z).unwrap()),
            decimal(&case["ood"])
        );
        assert_eq!(
            fp_to_u64(point(plan.log_n, 0).unwrap()),
            7
        );
    }

    assert!(matches!(build_code_plan(0, 1, 0), Err(Error::InvalidConfig)));
    assert!(matches!(build_code_plan(2, 0, 0), Err(Error::InvalidConfig)));
    assert!(matches!(build_code_plan(2, 5, 0), Err(Error::InvalidConfig)));
    assert!(matches!(build_code_plan(2, 2, 3), Err(Error::InvalidConfig)));
}

#[test]
fn extension_code_is_coordinate_wise() {
    let plan = CodeParams {
        log_n: 2,
        ell: 1,
        r: 1,
    };
    let message = [Fp3::new(
        fp_from_u64(3),
        fp_from_u64(5),
        fp_from_u64(7),
    )];
    let tail = [Fp3::new(
        fp_from_u64(11),
        fp_from_u64(13),
        fp_from_u64(17),
    )];
    let word = encode_row_fp3(&plan, &message, &tail).unwrap();
    assert_eq!(word.len(), 4);
    for position in 0..4 {
        let z = Fp3::from_base(point(plan.log_n, position).unwrap());
        assert_eq!(
            word[position as usize],
            evaluate_ood_fp3(&plan, &message, &tail, z).unwrap()
        );
    }
}

#[test]
fn parameter_digest_matches_vector_and_binds_shape() {
    let vectors = vectors();
    let case = &vectors["params_digest"];
    let code = CodeParams {
        log_n: case["log_n"].as_u64().unwrap() as u32,
        ell: case["ell"].as_u64().unwrap(),
        r: case["r"].as_u64().unwrap(),
    };
    let pow = PowPolicy::new(
        case["request_work_bits"].as_u64().unwrap() as u8,
        case["shared_work_bits"].as_u64().unwrap() as u8,
        case["final_work_bits"].as_u64().unwrap() as u8,
    )
    .unwrap();
    let params = RequestParams::new(code, pow).unwrap();
    assert_eq!(
        params.digest().unwrap(),
        digest_from_hex(case["digest_hex"].as_str().unwrap()).unwrap()
    );
    assert_eq!(REQUEST_CHAIN_LABEL, 0x4652_4d01);
    assert_eq!(KIND_ALGEBRAIC, 1);
    assert_eq!(KIND_SEAL, 6);

    let changed = RequestParams::new(
        CodeParams {
            log_n: code.log_n,
            ell: code.ell + 1,
            r: code.r,
        },
        pow,
    )
    .unwrap();
    assert_ne!(params.digest().unwrap(), changed.digest().unwrap());
}
