//! Golden-vector checks for exact integers, the fields and the hash.

#[path = "support/integer_oracle.rs"]
mod integer_oracle;

use serde_json::Value;
use zkie_core::artifact::digest_from_hex;
use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::Error;

use zkie_engine::pcs::request::field::{
    fp_from_u64, fp_inverse, fp_to_u64, Fp, Fp3, MODULUS,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_engine::primitives::{add_with_carry, div_rem_checked, mul_add_carry, rq, sub_with_borrow, WideU256};
use zkie_engine::primitives::transcript::{hash_labeled, hash_labeled_64, Blake2b};

fn vectors() -> Value {
    serde_json::from_str(include_str!("vectors/request/primitives.json")).unwrap()
}

fn hex_bytes(value: &str) -> Vec<u8> {
    assert!(value.len() % 2 == 0);
    (0..value.len() / 2)
        .map(|index| u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap())
        .collect()
}

fn hex16(value: &str) -> [u8; 16] {
    let bytes = hex_bytes(value);
    let mut out = [0u8; 16];
    out.copy_from_slice(&bytes);
    out
}

fn decimal(value: &Value) -> u64 {
    if let Some(text) = value.as_str() {
        text.parse().unwrap()
    } else {
        value.as_u64().unwrap()
    }
}

fn signed(value: &Value) -> i128 {
    value.as_str().unwrap().parse().unwrap()
}

fn wide_from_decimal(text: &str) -> WideU256 {
    let mut value = WideU256::ZERO;
    for byte in text.bytes() {
        value = value.mul_u64(10).unwrap();
        value = value
            .checked_add(WideU256::from_u64((byte - b'0') as u64))
            .unwrap();
    }
    value
}

fn wide_to_decimal(mut value: WideU256) -> String {
    if value.is_zero() {
        return "0".into();
    }
    let ten = WideU256::from_u64(10);
    let mut digits = Vec::new();
    while !value.is_zero() {
        let (quotient, remainder) = value.div_rem(ten).unwrap();
        digits.push((remainder.limbs[0] as u8 + b'0') as char);
        value = quotient;
    }
    digits.iter().rev().collect()
}

#[test]
fn blake2b_matches_independent_vectors() {
    let vectors = vectors();
    for case in vectors["blake2b"].as_array().unwrap() {
        let out_len = case["out_len"].as_u64().unwrap() as usize;
        let data = hex_bytes(case["data_hex"].as_str().unwrap());
        let salt = hex16(case["salt_hex"].as_str().unwrap());
        let personal = hex16(case["personal_hex"].as_str().unwrap());
        let mut hasher = Blake2b::with_salt_personal(out_len, salt, personal);
        hasher.update(&data);
        let digest = hasher.finalize();
        let expected = hex_bytes(case["digest_hex"].as_str().unwrap());
        assert_eq!(digest, expected);
    }
}

#[test]
fn labelled_hashes_match_vectors() {
    let vectors = vectors();
    for case in vectors["hash_labeled"].as_array().unwrap() {
        let tag = case["tag"].as_u64().unwrap();
        let index = case["index"].as_u64().unwrap();
        let data = hex_bytes(case["data_hex"].as_str().unwrap());
        let start = case["start_hex"]
            .as_str()
            .map(|text| digest_from_hex(text).unwrap());
        let digest = hash_labeled(tag, index, start.as_ref(), &data).unwrap();
        assert_eq!(digest, digest_from_hex(case["digest_hex"].as_str().unwrap()).unwrap());
    }
    for case in vectors["hash_labeled_64"].as_array().unwrap() {
        let tag = case["tag"].as_u64().unwrap();
        let index = case["index"].as_u64().unwrap();
        let data = hex_bytes(case["data_hex"].as_str().unwrap());
        let start = digest_from_hex(case["start_hex"].as_str().unwrap()).unwrap();
        let digest = hash_labeled_64(tag, index, &start, &data).unwrap();
        assert_eq!(digest.to_vec(), hex_bytes(case["digest_hex"].as_str().unwrap()));
    }
    assert!(hash_labeled(160, 1, None, b"x").is_ok());
    assert!(matches!(
        hash_labeled(162, 1, None, b"x"),
        Err(Error::InvalidEncoding)
    ));
}

#[test]
fn base_field_vectors() {
    let vectors = vectors();
    for case in vectors["field_add"].as_array().unwrap() {
        let a = fp_from_u64(decimal(&case["a"]));
        let b = fp_from_u64(decimal(&case["b"]));
        assert_eq!(fp_to_u64(a + b), decimal(&case["sum"]));
    }
    for case in vectors["field_mul"].as_array().unwrap() {
        let a = fp_from_u64(decimal(&case["a"]));
        let b = fp_from_u64(decimal(&case["b"]));
        assert_eq!(fp_to_u64(a * b), decimal(&case["product"]));
    }
    for case in vectors["field_inv"].as_array().unwrap() {
        let a = fp_from_u64(decimal(&case["a"]));
        assert_eq!(fp_to_u64(fp_inverse(a).unwrap()), decimal(&case["inverse"]));
        assert_eq!(
            fp_to_u64(a * fp_inverse(a).unwrap()),
            1
        );
    }
    assert!(matches!(fp_inverse(fp_from_u64(0)), Err(Error::DivisionByZero)));
}

fn fp3_from_case(values: &Value) -> Fp3 {
    let entries = values.as_array().unwrap();
    Fp3::new(
        fp_from_u64(decimal(&entries[0])),
        fp_from_u64(decimal(&entries[1])),
        fp_from_u64(decimal(&entries[2])),
    )
}

#[test]
fn extension_field_vectors_and_canonical_encoding() {
    let vectors = vectors();
    for case in vectors["ext_mul"].as_array().unwrap() {
        let a = fp3_from_case(&case["a"]);
        let b = fp3_from_case(&case["b"]);
        let product = a.mul(&b);
        let expected = fp3_from_case(&case["product"]);
        assert_eq!(product, expected);
        assert_eq!(product.encode(), expected.encode());
    }
    for case in vectors["ext_inv"].as_array().unwrap() {
        let a = fp3_from_case(&case["a"]);
        let inverse = a.inverse().unwrap();
        assert_eq!(inverse, fp3_from_case(&case["inverse"]));
        assert_eq!(a.mul(&inverse), Fp3::ONE);
        assert_eq!(a.encode().to_vec(), hex_bytes(case["bytes_hex"].as_str().unwrap()));
        assert_eq!(Fp3::decode(&a.encode()).unwrap(), a);
    }

    let mut non_canonical = [0u8; 24];
    non_canonical[..8].copy_from_slice(&MODULUS.to_le_bytes());
    assert!(matches!(Fp3::decode(&non_canonical), Err(Error::InvalidEncoding)));
    assert!(matches!(Fp3::decode(&[0u8; 23]), Err(Error::InvalidEncoding)));
    assert!(matches!(
        fp_inverse(fp_from_u64(0)).map(|_| ()),
        Err(Error::DivisionByZero)
    ));
}

#[test]
fn rounding_vectors_use_mathematical_floor() {
    let vectors = vectors();
    for case in vectors["rq"].as_array().unwrap() {
        let a = signed(&case["a"]);
        let k = case["k"].as_u64().unwrap() as u32;
        assert_eq!(rq(a, k).unwrap(), signed(&case["out"]));
        assert_eq!(rq(a, k).unwrap(), integer_oracle::rq_oracle(a, k));
    }
    for (a, k) in [(-7i128, 2u32), (-8, 2), (7, 2), (-1, 3), (0, 5), (9, 0)] {
        assert_eq!(rq(a, k).unwrap(), integer_oracle::rq_oracle(a, k));
    }
    assert!(matches!(rq(0, 128), Err(Error::BoundViolation)));
}

#[test]
fn wide_integer_vectors_and_carry_helpers() {
    let vectors = vectors();
    for case in vectors["wide_add"].as_array().unwrap() {
        let a = wide_from_decimal(case["a"].as_str().unwrap());
        let b = wide_from_decimal(case["b"].as_str().unwrap());
        let sum = a.checked_add(b).unwrap();
        assert_eq!(wide_to_decimal(sum), case["sum"].as_str().unwrap());
    }
    for case in vectors["wide_sub"].as_array().unwrap() {
        let a = wide_from_decimal(case["a"].as_str().unwrap());
        let b = wide_from_decimal(case["b"].as_str().unwrap());
        let difference = a.checked_sub(b).unwrap();
        assert_eq!(
            wide_to_decimal(difference),
            case["difference"].as_str().unwrap()
        );
    }
    for case in vectors["wide_mul_u64"].as_array().unwrap() {
        let a = wide_from_decimal(case["a"].as_str().unwrap());
        let factor = case["factor"].as_str().unwrap().parse().unwrap();
        let product = a.mul_u64(factor).unwrap();
        assert_eq!(
            wide_to_decimal(product),
            case["product"].as_str().unwrap()
        );
    }
    for case in vectors["wide_div"].as_array().unwrap() {
        let a = wide_from_decimal(case["a"].as_str().unwrap());
        let divisor = wide_from_decimal(case["divisor"].as_str().unwrap());
        let (quotient, remainder) = a.div_rem(divisor).unwrap();
        assert_eq!(
            wide_to_decimal(quotient),
            case["quotient"].as_str().unwrap()
        );
        assert_eq!(
            wide_to_decimal(remainder),
            case["remainder"].as_str().unwrap()
        );
    }
    assert!(matches!(
        WideU256::from_u64(3).div_rem(WideU256::ZERO),
        Err(Error::DivisionByZero)
    ));

    assert_eq!(add_with_carry(u64::MAX, 1, 0), (0, 1));
    assert_eq!(sub_with_borrow(0, 1, 0), (u64::MAX, 1));
    assert_eq!(mul_add_carry(u64::MAX, 2, 1, 0), (u64::MAX, 1));
    assert_eq!(add_with_carry(u64::MAX, 0, 1), (0, 1));
    assert_eq!(mul_add_carry(1 << 63, 2, 0, 0), (0, 1));

    assert_eq!(integer_oracle::add_oracle(5, 7), Some(12));
    assert_eq!(integer_oracle::mul_oracle(9, 9), Some(81));
    assert_eq!(integer_oracle::div_oracle(100, 7), Some((14, 2)));
    assert_eq!(integer_oracle::div_oracle(1, 0), None);
}

#[test]
fn exact_division_rejects_zero_and_matches_euclid() {
    assert!(matches!(div_rem_checked(5, 0), Err(Error::DivisionByZero)));
    for (a, b) in [(7i128, 3i128), (-7, 3), (7, -3), (-7, -3), (0, 5)] {
        let (quotient, remainder) = div_rem_checked(a, b).unwrap();
        assert_eq!((quotient, remainder), integer_oracle::euclid_oracle(a, b));
    }
}

#[test]
fn canonical_codec_rejects_non_canonical_and_trailing_bytes() {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(7);
    writer.put_bytes(b"abc");
    let bytes = writer.into_bytes();
    let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
    assert_eq!(reader.get_u64().unwrap(), 7);
    assert_eq!(reader.get_bytes().unwrap(), b"abc");
    reader.finish().unwrap();

    let mut trailing = bytes.clone();
    trailing.push(0);
    let mut reader = CanonicalReader::new(&trailing, DecodeLimits::unlimited());
    let _ = reader.get_u64().unwrap();
    let _ = reader.get_bytes().unwrap();
    assert!(matches!(reader.finish(), Err(Error::InvalidEncoding)));

    let mut non_canonical = [0u8; 8];
    non_canonical.copy_from_slice(&MODULUS.to_le_bytes());
    let mut reader = CanonicalReader::new(&non_canonical, DecodeLimits::unlimited());
    assert!(matches!(reader.get_fp(), Err(Error::InvalidEncoding)));

    let mut limit = CanonicalWriter::new();
    limit.put_bytes(&[0u8; 8]);
    let bytes = limit.into_bytes();
    let limits = DecodeLimits {
        max_bytes: 4,
        max_elements: 4,
    };
    let mut reader = CanonicalReader::new(&bytes, limits);
    assert!(matches!(reader.get_bytes(), Err(Error::InvalidEncoding)));
    let _: Fp = fp_from_u64(3);
}

