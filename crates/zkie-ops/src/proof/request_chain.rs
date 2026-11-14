//! The request-level chain's transcript section.
//!
//! A request's sub-proof leaves three masked sumchecks: the kept token
//! sumcheck, the batched stack and the weight merge. Their proofs travel as
//! one section, after the sub-proof's other messages and before the chain
//! value. The section carries its own version; the round and end-value counts
//! come from the plan's claim shapes, and a section that does not match them
//! is refused rather than reinterpreted.

use zkie_core::error::{Error, Result};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::psc_general::GeneralProof;

/// The section's version.
pub const REQUEST_CHAIN_VERSION: u64 = 1;

/// One request chain's proofs, in the order the chain runs them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestChainMessages {
    pub token: GeneralProof,
    pub stack: GeneralProof,
    pub merge: GeneralProof,
}

/// The round counts and shown end-value counts the plan expects of each proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestChainShape {
    pub token_rounds: usize,
    pub token_ends: usize,
    pub stack_rounds: usize,
    pub stack_ends: usize,
    pub merge_rounds: usize,
    pub merge_ends: usize,
}

impl RequestChainMessages {
    pub fn encode(&self, shape: RequestChainShape) -> Result<Vec<u8>> {
        if self.token.rounds.len() != shape.token_rounds
            || self.token.end_values.len() != shape.token_ends
            || self.stack.rounds.len() != shape.stack_rounds
            || self.stack.end_values.len() != shape.stack_ends
            || self.merge.rounds.len() != shape.merge_rounds
            || self.merge.end_values.len() != shape.merge_ends
        {
            return Err(Error::InvalidConfig);
        }
        let mut writer = CanonicalWriter::new();
        self.encode_into(&mut writer);
        Ok(writer.into_bytes())
    }

    /// Writes the section without the shape check; the decoder still checks
    /// every stored count against the plan.
    pub fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(REQUEST_CHAIN_VERSION);
        self.token.encode_into(writer);
        self.stack.encode_into(writer);
        self.merge.encode_into(writer);
    }

    pub fn decode(reader: &mut CanonicalReader, shape: RequestChainShape) -> Result<Self> {
        if reader.get_u64()? != REQUEST_CHAIN_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let token = GeneralProof::decode(reader, shape.token_rounds, shape.token_ends)?;
        let stack = GeneralProof::decode(reader, shape.stack_rounds, shape.stack_ends)?;
        let merge = GeneralProof::decode(reader, shape.merge_rounds, shape.merge_ends)?;
        Ok(Self {
            token,
            stack,
            merge,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use zkie_core::common::codec::DecodeLimits;
    use zkie_engine::pcs::request::field::Fp3;
    use zkie_engine::pcs::request::field::fp_from_u64;

    use crate::proof::psc::PscRound;

    fn f(value: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(value))
    }

    fn proof(rounds: usize, ends: usize, base: u64) -> GeneralProof {
        GeneralProof {
            hiding_sum: f(base),
            rounds: (0..rounds)
                .map(|index| PscRound {
                    coefficients: vec![f(base + 1 + index as u64)],
                })
                .collect(),
            y_round: PscRound {
                coefficients: vec![f(base + 100)],
            },
            end_values: (0..ends).map(|index| f(base + 200 + index as u64)).collect(),
        }
    }

    fn shape() -> RequestChainShape {
        RequestChainShape {
            token_rounds: 2,
            token_ends: 1,
            stack_rounds: 3,
            stack_ends: 2,
            merge_rounds: 4,
            merge_ends: 1,
        }
    }

    fn messages() -> RequestChainMessages {
        RequestChainMessages {
            token: proof(2, 1, 10),
            stack: proof(3, 2, 20),
            merge: proof(4, 1, 30),
        }
    }

    #[test]
    fn the_section_round_trips_with_its_shape() {
        let messages = messages();
        let bytes = messages.encode(shape()).unwrap();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert_eq!(
            RequestChainMessages::decode(&mut reader, shape()).unwrap(),
            messages
        );
    }

    #[test]
    fn another_version_or_shape_is_refused() {
        let messages = messages();
        let mut bytes = messages.encode(shape()).unwrap();
        // The first eight bytes are the version.
        bytes[7] = 9;
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(RequestChainMessages::decode(&mut reader, shape()).is_err());
        let bytes = messages.encode(shape()).unwrap();
        // The encoder refuses proofs of another shape.
        let mut wrong = shape();
        wrong.stack_rounds = 2;
        assert!(messages.encode(wrong).is_err());
        // The decoder refuses stored counts that differ from the plan's.
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        let mut wrong = shape();
        wrong.merge_ends = 2;
        assert!(RequestChainMessages::decode(&mut reader, wrong).is_err());
        // Truncated bytes are refused.
        let mut reader = CanonicalReader::new(&bytes[..bytes.len() - 4], DecodeLimits::unlimited());
        assert!(RequestChainMessages::decode(&mut reader, shape()).is_err());
    }
}
