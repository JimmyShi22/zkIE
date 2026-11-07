//! Protocol hash: BLAKE2b on labelled strings.
//!
//! A hashed string begins with a sixteen-byte label, the tag's first word and
//! then the index, both little-endian. A hash with a start uses the start's
//! first sixteen bytes as BLAKE2b's salt and the next sixteen as its
//! personalization. Digests are 32 bytes; a call of a chain round is 64 bytes.
//!
//! The BLAKE2b compression function is implemented here so that the protocol's
//! digest lengths and salt/personalization are not replaced by a default
//! instance.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

pub const DIGEST_BYTES: usize = 32;
pub const CHAIN_DIGEST_BYTES: usize = 64;
pub const LABEL_BYTES: usize = 16;

pub const CHAIN_START: u16 = 160;
pub const CHAIN_QUERY: u16 = 162;
pub const LEAF_MAIN: u16 = 164;
pub const TREE_NODE: u16 = 166;
pub const STATEMENT: u16 = 168;
pub const PARAMETERS: u16 = 169;
pub const TEST_COINS: u16 = 171;
pub const RUN: u16 = 172;
pub const LEAF_BINDING: u16 = 173;
pub const LEAF_STORED_MAIN: u16 = 174;
pub const COIN_INPUTS: u16 = 176;
pub const BINDING_NODE: u16 = 178;
pub const PUBLISHED_ID: u16 = 179;

const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[inline]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], counter: u128, last: bool) {
    let mut message = [0u64; 16];
    for index in 0..16 {
        message[index] = u64::from_le_bytes(
            block[index * 8..(index + 1) * 8]
                .try_into()
                .expect("eight bytes"),
        );
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= counter as u64;
    v[13] ^= (counter >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for round in 0..12 {
        let sigma = &SIGMA[round % 10];
        g(&mut v, 0, 4, 8, 12, message[sigma[0]], message[sigma[1]]);
        g(&mut v, 1, 5, 9, 13, message[sigma[2]], message[sigma[3]]);
        g(&mut v, 2, 6, 10, 14, message[sigma[4]], message[sigma[5]]);
        g(&mut v, 3, 7, 11, 15, message[sigma[6]], message[sigma[7]]);
        g(&mut v, 0, 5, 10, 15, message[sigma[8]], message[sigma[9]]);
        g(&mut v, 1, 6, 11, 12, message[sigma[10]], message[sigma[11]]);
        g(&mut v, 2, 7, 8, 13, message[sigma[12]], message[sigma[13]]);
        g(&mut v, 3, 4, 9, 14, message[sigma[14]], message[sigma[15]]);
    }
    for index in 0..8 {
        h[index] ^= v[index] ^ v[index + 8];
    }
}

/// Sequential, unkeyed-by-default BLAKE2b with optional key, salt and
/// personalization, and a digest length of 1 to 64 bytes.
pub struct Blake2b {
    h: [u64; 8],
    counter: u128,
    buffer: [u8; 128],
    buffered: usize,
    out_len: usize,
}

impl Blake2b {
    pub fn new(out_len: usize) -> Self {
        Self::keyed(out_len, &[], [0u8; 16], [0u8; 16]).expect("valid BLAKE2b parameters")
    }

    pub fn with_salt_personal(out_len: usize, salt: [u8; 16], personal: [u8; 16]) -> Self {
        Self::keyed(out_len, &[], salt, personal).expect("valid BLAKE2b parameters")
    }

    pub fn keyed(
        out_len: usize,
        key: &[u8],
        salt: [u8; 16],
        personal: [u8; 16],
    ) -> Result<Self> {
        if out_len == 0 || out_len > 64 || key.len() > 64 {
            return Err(Error::InvalidEncoding);
        }
        let mut h = IV;
        h[0] ^= 0x0101_0000u64 ^ ((key.len() as u64) << 8) ^ out_len as u64;
        h[4] ^= u64::from_le_bytes(salt[0..8].try_into().map_err(|_| Error::InvalidEncoding)?);
        h[5] ^= u64::from_le_bytes(salt[8..16].try_into().map_err(|_| Error::InvalidEncoding)?);
        h[6] ^= u64::from_le_bytes(
            personal[0..8]
                .try_into()
                .map_err(|_| Error::InvalidEncoding)?,
        );
        h[7] ^= u64::from_le_bytes(
            personal[8..16]
                .try_into()
                .map_err(|_| Error::InvalidEncoding)?,
        );
        let mut hasher = Self {
            h,
            counter: 0,
            buffer: [0u8; 128],
            buffered: 0,
            out_len,
        };
        if !key.is_empty() {
            hasher.buffer[..key.len()].copy_from_slice(key);
            hasher.buffered = 128;
        }
        Ok(hasher)
    }

    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            if self.buffered == 128 {
                self.counter = self.counter.wrapping_add(128);
                let block = self.buffer;
                compress(&mut self.h, &block, self.counter, false);
                self.buffered = 0;
            }
            let take = core::cmp::min(128 - self.buffered, data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
        }
    }

    pub fn finalize(mut self) -> Vec<u8> {
        self.counter = self.counter.wrapping_add(self.buffered as u128);
        for index in self.buffered..128 {
            self.buffer[index] = 0;
        }
        let block = self.buffer;
        compress(&mut self.h, &block, self.counter, true);
        let mut out = Vec::with_capacity(self.out_len);
        for word in self.h {
            out.extend_from_slice(&word.to_le_bytes());
            if out.len() >= self.out_len {
                break;
            }
        }
        out.truncate(self.out_len);
        out
    }
}

/// The first word of a tag: the domain number in the low sixteen bits and the
/// tree number (or a chain's label) above it.
pub fn tag_word(domain: u16, tree: u64) -> u64 {
    domain as u64 | (tree << 16)
}

fn label_bytes(tag: u64, index: u64) -> Result<[u8; LABEL_BYTES]> {
    if index >> 63 != 0 {
        return Err(Error::InvalidEncoding);
    }
    let mut label = [0u8; LABEL_BYTES];
    label[..8].copy_from_slice(&tag.to_le_bytes());
    label[8..].copy_from_slice(&index.to_le_bytes());
    Ok(label)
}

fn has_start(domain: u16) -> bool {
    matches!(
        domain,
        CHAIN_QUERY | TEST_COINS | LEAF_MAIN | LEAF_BINDING | LEAF_STORED_MAIN | TREE_NODE | BINDING_NODE
    )
}

fn digest_from_bytes(bytes: &[u8]) -> Result<Digest> {
    if bytes.len() != DIGEST_BYTES {
        return Err(Error::InvalidEncoding);
    }
    let mut digest = [0u8; DIGEST_BYTES];
    digest.copy_from_slice(bytes);
    Ok(Digest(digest))
}

/// Hash a labelled string of arbitrary length to 32 bytes.
pub fn hash_labeled(
    tag: u64,
    index: u64,
    start: Option<&Digest>,
    data: &[u8],
) -> Result<Digest> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Transcript);
    let label = label_bytes(tag, index)?;
    let mut hasher = match start {
        Some(start) if has_start((tag & 0xffff) as u16) => {
            let salt: [u8; 16] = start.0[0..16].try_into().map_err(|_| Error::InvalidEncoding)?;
            let personal: [u8; 16] =
                start.0[16..32].try_into().map_err(|_| Error::InvalidEncoding)?;
            Blake2b::with_salt_personal(DIGEST_BYTES, salt, personal)
        }
        Some(_) => return Err(Error::InvalidEncoding),
        None => {
            if has_start((tag & 0xffff) as u16) {
                return Err(Error::InvalidEncoding);
            }
            Blake2b::new(DIGEST_BYTES)
        }
    };
    hasher.update(&label);
    hasher.update(data);
    digest_from_bytes(&hasher.finalize())
}

/// Hash a labelled string of arbitrary length to 64 bytes.
pub fn hash_labeled_64(
    tag: u64,
    index: u64,
    start: &Digest,
    data: &[u8],
) -> Result<[u8; CHAIN_DIGEST_BYTES]> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Transcript);
    let label = label_bytes(tag, index)?;
    let salt: [u8; 16] = start.0[0..16].try_into().map_err(|_| Error::InvalidEncoding)?;
    let personal: [u8; 16] = start.0[16..32].try_into().map_err(|_| Error::InvalidEncoding)?;
    let mut hasher = Blake2b::with_salt_personal(CHAIN_DIGEST_BYTES, salt, personal);
    hasher.update(&label);
    hasher.update(data);
    let out = hasher.finalize();
    let mut digest = [0u8; CHAIN_DIGEST_BYTES];
    digest.copy_from_slice(&out);
    Ok(digest)
}

/// The run's value: `H(RUN, index = label; parameters digest, random bytes)`.
pub fn run_value(label: u64, parameters: &Digest, random_bytes: &Digest) -> Result<Digest> {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&parameters.0);
    data.extend_from_slice(&random_bytes.0);
    hash_labeled(tag_word(RUN, 0), label, None, &data)
}

fn push_length_prefixed(data: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    data.extend_from_slice(&(value.len() as u64).to_le_bytes());
    data.extend_from_slice(value);
    Ok(())
}

/// The first value of a chain.
///
/// The string binds the parameters' digest, the statement, an optional record
/// and the (optional) run value derived from the chain's random bytes.
pub fn chain_start(
    label: u64,
    parameters: &Digest,
    statement: &[u8],
    record: &[u8],
    random_bytes: Option<&Digest>,
) -> Result<Digest> {
    if label >> 48 != 0 {
        return Err(Error::InvalidEncoding);
    }
    let mut data = Vec::new();
    data.extend_from_slice(&parameters.0);
    push_length_prefixed(&mut data, statement)?;
    push_length_prefixed(&mut data, record)?;
    data.extend_from_slice(&(if random_bytes.is_some() { 1u64 } else { 0 }).to_le_bytes());
    if let Some(random_bytes) = random_bytes {
        data.extend_from_slice(&run_value(label, parameters, random_bytes)?.0);
    }
    hash_labeled(tag_word(CHAIN_START, label), label, None, &data)
}
