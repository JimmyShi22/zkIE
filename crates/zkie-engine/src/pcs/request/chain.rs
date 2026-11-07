//! Hash-chain primitives shared by the three protocol chains.
//!
//! A chain is a Fiat-Shamir transcript: each round absorbs a prover message
//! and derives the verifier's challenge words from the round's answer blocks.

use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::field::{Fp, Fp3, MODULUS};
use crate::pcs::request::params::{DRAW_TRIES, NONZERO_TRIES};
use crate::pcs::request::pow::work_test;
use crate::primitives::transcript::{chain_start, hash_labeled_64, tag_word, CHAIN_QUERY};

pub const KIND_ALGEBRAIC: u32 = 1;
pub const KIND_EPSILON: u32 = 2;
pub const KIND_SEED: u32 = 3;
pub const KIND_POSITIONS: u32 = 4;
pub const KIND_BITS: u32 = 5;
pub const KIND_SEAL: u32 = 6;
pub const KIND_EMPTY: u32 = 7;

/// One round's query before the nonce: everything a nonce search varies around.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainPrefix {
    pub rho_prev: Digest,
    pub label: u64,
    pub round: u64,
    pub message: Vec<u64>,
    pub kind: u32,
    pub work_bits: u8,
}

impl ChainPrefix {
    /// Call `j` of this round: 64 answer bytes.
    pub fn call(&self, nonce: u64, j: u64) -> Result<[u8; 64]> {
        if self.label >> 48 != 0 || self.round >> 63 != 0 || j >> 48 != 0 {
            return Err(Error::InvalidEncoding);
        }
        if self.work_bits > 64 {
            return Err(Error::InvalidConfig);
        }
        if self.kind > u8::MAX as u32 {
            return Err(Error::InvalidEncoding);
        }
        let mut data = Vec::with_capacity(self.message.len() * 8 + 16);
        for word in &self.message {
            data.extend_from_slice(&word.to_le_bytes());
        }
        let closing = (self.kind as u64) | ((self.work_bits as u64) << 8) | (j << 16);
        data.extend_from_slice(&closing.to_le_bytes());
        data.extend_from_slice(&nonce.to_le_bytes());
        hash_labeled_64(
            tag_word(CHAIN_QUERY, self.label),
            self.round,
            &self.rho_prev,
            &data,
        )
    }

    /// The first 64-bit word of call 0, which the work test reads.
    pub fn first_word(&self, nonce: u64) -> Result<u64> {
        let call = self.call(nonce, 0)?;
        Ok(u64::from_le_bytes(
            call[..8].try_into().map_err(|_| Error::InvalidEncoding)?,
        ))
    }
}

/// An open prover message for the next round.
pub struct ChainMessage {
    rho_prev: Digest,
    label: u64,
    round: u64,
    message: Vec<u64>,
}

impl ChainMessage {
    pub fn push_u64(&mut self, word: u64) {
        self.message.push(word);
    }

    pub fn push_fp(&mut self, value: Fp) {
        self.message.push(value.as_canonical_u64());
    }

    pub fn push_fp3(&mut self, value: &Fp3) {
        for coefficient in value.c {
            self.message.push(coefficient.as_canonical_u64());
        }
    }

    pub fn prefix(self, kind: u32, work_bits: u8) -> Result<ChainPrefix> {
        if work_bits > 64 {
            return Err(Error::InvalidConfig);
        }
        if kind > u8::MAX as u32 {
            return Err(Error::InvalidEncoding);
        }
        Ok(ChainPrefix {
            rho_prev: self.rho_prev,
            label: self.label,
            round: self.round,
            message: self.message,
            kind,
            work_bits,
        })
    }
}

/// One chain. It holds the current value, the round count and, after a round,
/// the verifier's answer words that the draws consume.
#[derive(Clone, Debug)]
pub struct Chain {
    label: u64,
    rho: Digest,
    round: u64,
    query: Option<ChainPrefix>,
    nonce: u64,
    call_index: u64,
    words: [u64; 8],
    next: usize,
    have: usize,
}

impl Chain {
    pub fn begin(
        label: u64,
        parameters: &Digest,
        statement: &[u8],
        record: &[u8],
        random_bytes: Option<&Digest>,
    ) -> Result<Self> {
        let rho = chain_start(label, parameters, statement, record, random_bytes)?;
        Ok(Self {
            label,
            rho,
            round: 0,
            query: None,
            nonce: 0,
            call_index: 0,
            words: [0u64; 8],
            next: 0,
            have: 0,
        })
    }

    /// Resume a chain from a transcript value. Verifiers use the value they
    /// recomputed, never a value read from a proof.
    pub fn resume(label: u64, rho: Digest) -> Self {
        Self {
            label,
            rho,
            round: 0,
            query: None,
            nonce: 0,
            call_index: 0,
            words: [0u64; 8],
            next: 0,
            have: 0,
        }
    }

    pub fn label(&self) -> u64 {
        self.label
    }

    pub fn value(&self) -> &Digest {
        &self.rho
    }

    pub fn rounds(&self) -> u64 {
        self.round
    }

    pub fn message(&self) -> ChainMessage {
        ChainMessage {
            rho_prev: self.rho,
            label: self.label,
            round: self.round,
            message: Vec::new(),
        }
    }

    /// Advance one round. The prefix must belong to this chain position; a
    /// failed work test leaves the chain unchanged and returns `false`.
    pub fn step(&mut self, prefix: &ChainPrefix, nonce: u64) -> Result<bool> {
        if prefix.rho_prev != self.rho || prefix.label != self.label || prefix.round != self.round {
            return Err(Error::StatementMismatch);
        }
        if prefix.work_bits == 0 && nonce != 0 {
            return Err(Error::InvalidEncoding);
        }
        let first = prefix.call(nonce, 0)?;
        let first_word = u64::from_le_bytes(
            first[..8]
                .try_into()
                .map_err(|_| Error::InvalidEncoding)?,
        );
        if !work_test(first_word, prefix.work_bits)? {
            return Ok(false);
        }
        let half = if prefix.work_bits > 0 { 1 } else { 0 };
        let mut next_value = [0u8; 32];
        next_value.copy_from_slice(&first[half * 32..(half + 1) * 32]);
        self.rho = Digest(next_value);
        self.round = self.round.checked_add(1).ok_or(Error::ResourceLimit)?;
        self.query = Some(prefix.clone());
        self.nonce = nonce;
        self.call_index = 1;
        self.next = 0;
        self.have = 0;
        if prefix.work_bits == 0 {
            for index in 0..4 {
                self.words[index] = u64::from_le_bytes(
                    first[32 + index * 8..40 + index * 8]
                        .try_into()
                        .map_err(|_| Error::InvalidEncoding)?,
                );
            }
            self.have = 4;
        }
        Ok(true)
    }

    fn next_word(&mut self) -> Result<u64> {
        if self.next == self.have {
            let prefix = self
                .query
                .as_ref()
                .ok_or(Error::InvalidSessionState)?
                .clone();
            let call = prefix.call(self.nonce, self.call_index)?;
            self.call_index = self.call_index.checked_add(1).ok_or(Error::ResourceLimit)?;
            for index in 0..8 {
                self.words[index] = u64::from_le_bytes(
                    call[index * 8..(index + 1) * 8]
                        .try_into()
                        .map_err(|_| Error::InvalidEncoding)?,
                );
            }
            self.next = 0;
            self.have = 8;
        }
        let word = self.words[self.next];
        self.next += 1;
        Ok(word)
    }

    /// One base-field challenge: the next word below the modulus, or zero
    /// after the fixed number of rejected words.
    pub fn draw_f(&mut self) -> Result<Fp> {
        for _ in 0..DRAW_TRIES {
            let word = self.next_word()?;
            if word < MODULUS {
                return Ok(Fp::from_u64(word));
            }
        }
        Ok(Fp::ZERO)
    }

    /// One extension-field challenge: three base-field draws.
    pub fn draw_e(&mut self) -> Result<Fp3> {
        Ok(Fp3::new(self.draw_f()?, self.draw_f()?, self.draw_f()?))
    }

    /// A non-zero extension challenge: the first non-zero of the fixed number
    /// of extension draws, else one.
    pub fn draw_e_nonzero(&mut self) -> Result<Fp3> {
        for _ in 0..NONZERO_TRIES {
            let value = self.draw_e()?;
            if !value.is_zero() {
                return Ok(value);
            }
        }
        Ok(Fp3::ONE)
    }

    /// A position below `2^log_n`: the low bits of the next word.
    pub fn draw_position(&mut self, log_n: u32) -> Result<u64> {
        if log_n >= 64 {
            return Err(Error::InvalidConfig);
        }
        let word = self.next_word()?;
        Ok(word & ((1u64 << log_n) - 1))
    }
}
