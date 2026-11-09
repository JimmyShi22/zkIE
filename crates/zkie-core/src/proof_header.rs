//! Isolated proof-header parsing for a tentative test encoding.
//!
//! The encoding is not a final external format. A header is exactly eight
//! bytes: four tag bytes followed by one little-endian version word. The first
//! tag byte is the low byte when the tag is viewed as a little-endian word.
//!
//! [`read_header`] consumes exactly the eight header bytes and performs no
//! trailing-byte check. Trailing bytes are a property of the complete proof
//! parser, not of the header parser.
//!
//! A successful header read proves only that the tag matches and that the
//! version word equals this build's single compile-time version. It does not
//! establish soundness, hiding or single-shot semantics. Trusted transcript
//! binding (protocol identity, configuration, plan, statement, output and
//! table identity) is bound before the first challenge when this encoding is
//! integrated into production; this module makes no claim that any proof
//! passes.

use std::io::{ErrorKind, Read};

use crate::error::{Error, Result};

/// Four tag bytes. The first byte is the low byte when the tag is read as a
/// little-endian word.
pub const TAG_PROOF: [u8; 4] = [b'Z', b'K', b'P', b'R'];

/// The single version word this build accepts. The bytes are `Z`, `K`, `0`,
/// `1`, with the first byte in the low byte. The value is compile-time
/// authority: a header word is compared with it and can never extend the set
/// of accepted words.
pub const PROOF_VERSION_WORD: u32 = 0x3130_4B5A;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofHeader {
    pub tag: [u8; 4],
    pub version_word: u32,
}

/// Reads and validates exactly one eight-byte header.
///
/// A short header is an encoding error. A wrong tag is a proof error and an
/// unknown version is an unsupported version. Any other read failure is an I/O
/// error. The function returns on the first failure; the caller must not
/// continue parsing the same input.
pub fn read_header<R: Read>(reader: &mut R) -> Result<ProofHeader> {
    let tag = read_word(reader)?;
    let version_word = u32::from_le_bytes(read_word(reader)?);
    if tag != TAG_PROOF {
        return Err(Error::InvalidProof);
    }
    if version_word != PROOF_VERSION_WORD {
        return Err(Error::UnsupportedVersion);
    }
    Ok(ProofHeader { tag, version_word })
}

fn read_word<R: Read>(reader: &mut R) -> Result<[u8; 4]> {
    let mut word = [0u8; 4];
    read_exact_bytes(reader, &mut word)?;
    Ok(word)
}

fn read_exact_bytes<R: Read>(reader: &mut R, out: &mut [u8]) -> Result<()> {
    let mut filled = 0;
    while filled < out.len() {
        match reader.read(&mut out[filled..]) {
            Ok(0) => return Err(Error::InvalidEncoding),
            Ok(read) => filled += read,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(_) => return Err(Error::Io),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};

    fn legal_header() -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&TAG_PROOF);
        bytes[4..].copy_from_slice(&PROOF_VERSION_WORD.to_le_bytes());
        bytes
    }

    fn header_error(bytes: &[u8]) -> &'static str {
        read_header(&mut Cursor::new(bytes)).unwrap_err().code()
    }

    #[test]
    fn legal_header_is_accepted_and_consumes_eight_bytes() {
        let mut cursor = Cursor::new(legal_header());
        let header = read_header(&mut cursor).unwrap();
        assert_eq!(header.tag, TAG_PROOF);
        assert_eq!(header.version_word, PROOF_VERSION_WORD);
        assert_eq!(cursor.position(), 8);
    }

    #[test]
    fn header_does_not_check_trailing_bytes() {
        let mut bytes = legal_header().to_vec();
        bytes.push(0);
        let mut cursor = Cursor::new(bytes);
        assert!(read_header(&mut cursor).is_ok());
        assert_eq!(cursor.position(), 8);
    }

    #[test]
    fn every_short_header_is_invalid_encoding() {
        let bytes = legal_header();
        for len in 0..8 {
            assert_eq!(header_error(&bytes[..len]), "invalid_encoding", "len {len}");
        }
    }

    #[test]
    fn wrong_tag_is_rejected_before_version() {
        let mut bytes = legal_header();
        bytes[0] ^= 1;
        assert_eq!(header_error(&bytes), "invalid_proof");
    }

    #[test]
    fn unknown_version_is_unsupported() {
        let mut bytes = legal_header();
        bytes[4..].copy_from_slice(&PROOF_VERSION_WORD.wrapping_add(1).to_le_bytes());
        assert_eq!(header_error(&bytes), "unsupported_version");
    }

    struct ErrorAfter {
        data: Vec<u8>,
        position: usize,
        fail_at: usize,
    }

    impl Read for ErrorAfter {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.position >= self.fail_at {
                return Err(io::Error::other("injected read error"));
            }
            let available = self.fail_at.min(self.data.len()) - self.position;
            let count = available.min(out.len());
            out[..count].copy_from_slice(&self.data[self.position..self.position + count]);
            self.position += count;
            Ok(count)
        }
    }

    #[test]
    fn io_error_before_the_header_is_io() {
        let mut reader = ErrorAfter {
            data: legal_header().to_vec(),
            position: 0,
            fail_at: 0,
        };
        assert_eq!(read_header(&mut reader).unwrap_err().code(), "io");
    }

    #[test]
    fn io_error_after_the_tag_is_io() {
        let mut reader = ErrorAfter {
            data: legal_header().to_vec(),
            position: 0,
            fail_at: 4,
        };
        assert_eq!(read_header(&mut reader).unwrap_err().code(), "io");
    }

    #[test]
    fn io_error_in_the_middle_of_a_word_is_io() {
        let mut reader = ErrorAfter {
            data: legal_header().to_vec(),
            position: 0,
            fail_at: 2,
        };
        assert_eq!(read_header(&mut reader).unwrap_err().code(), "io");
    }

    // Fixture-only demonstration of a complete parse: header, then a synthetic
    // body, then the end-of-input gate. This is not production code and is not
    // a claim that `read_header` alone checks trailing bytes.
    fn read_fixture_body<R: Read>(reader: &mut R) -> Result<()> {
        let mut length = [0u8; 4];
        read_exact_bytes(reader, &mut length)?;
        let mut remaining = u32::from_le_bytes(length) as usize;
        let mut buffer = [0u8; 64];
        while remaining > 0 {
            let take = remaining.min(buffer.len());
            read_exact_bytes(reader, &mut buffer[..take])?;
            remaining -= take;
        }
        Ok(())
    }

    fn read_fixture_proof<R: Read>(reader: &mut R) -> Result<ProofHeader> {
        let header = read_header(reader)?;
        read_fixture_body(reader)?;
        let mut trailing = [0u8; 1];
        match reader.read(&mut trailing) {
            Ok(0) => Ok(header),
            Ok(_) => Err(Error::InvalidProof),
            Err(_) => Err(Error::Io),
        }
    }

    fn fixture_bytes(body: &[u8], trailing: bool) -> Vec<u8> {
        let mut bytes = legal_header().to_vec();
        bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
        bytes.extend_from_slice(body);
        if trailing {
            bytes.push(0);
        }
        bytes
    }

    #[test]
    fn fixture_parser_consumes_header_and_body() {
        let bytes = fixture_bytes(&[0x11, 0x22, 0x33], false);
        let mut cursor = Cursor::new(&bytes);
        let header = read_fixture_proof(&mut cursor).unwrap();
        assert_eq!(header.tag, TAG_PROOF);
        assert_eq!(cursor.position(), bytes.len() as u64);
    }

    #[test]
    fn fixture_parser_rejects_a_trailing_byte() {
        let bytes = fixture_bytes(&[0x11, 0x22, 0x33], true);
        let error = read_fixture_proof(&mut Cursor::new(&bytes)).unwrap_err();
        assert_eq!(error.code(), "invalid_proof");
        let mut cursor = Cursor::new(&bytes);
        assert!(read_header(&mut cursor).is_ok());
        assert_eq!(cursor.position(), 8);
    }

    #[test]
    fn fixture_parser_rejects_a_truncated_body() {
        let mut bytes = fixture_bytes(&[0x11, 0x22, 0x33], false);
        bytes.pop();
        assert_eq!(
            read_fixture_proof(&mut Cursor::new(&bytes))
                .unwrap_err()
                .code(),
            "invalid_encoding"
        );
    }
}
