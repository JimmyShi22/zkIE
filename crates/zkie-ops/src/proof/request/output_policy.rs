//! Output policy: the fixed seven-word acceptance policy, exact match and
//! clipped gap. The policy is part of the statement and of the statement
//! digest; the proof carries only tokens and per-position gaps.

use zkie_core::error::{Error, Result};
use zkie_core::session::{
    ClippedGapPolicy, DecodeCoverage, EosPolicy, OutputPolicy, Statement, VerificationResult,
};
use zkie_core::types::Digest;
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

pub const POLICY_WORDS: usize = 7;
pub const POLICY_BYTES: usize = POLICY_WORDS * 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyKind {
    ExactMatch,
    ClippedGap,
}

/// The seven words `C, theta, cap, s, sh, sT, F`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyParams {
    pub c: u64,
    pub theta: u64,
    pub cap: u64,
    pub s: u64,
    pub sh: u64,
    pub st: u64,
    pub f: u64,
}

impl PolicyParams {
    pub fn exact_match(sh: u64) -> Result<Self> {
        let params = Self {
            c: 1u64 << sh,
            theta: 1u64 << sh,
            cap: 0,
            s: 0,
            sh,
            st: sh.checked_sub(4).ok_or(Error::InvalidConfig)?,
            f: 24,
        };
        params.validate()?;
        Ok(params)
    }

    pub fn clipped(
        c: u64,
        theta: u64,
        cap: u64,
        s: u64,
        sh: u64,
        st: u64,
    ) -> Result<Self> {
        let params = Self {
            c,
            theta,
            cap,
            s,
            sh,
            st,
            f: 24,
        };
        params.validate()?;
        Ok(params)
    }

    pub fn encode(&self) -> [u8; POLICY_BYTES] {
        let mut words = [0u8; POLICY_BYTES];
        for (index, value) in [self.c, self.theta, self.cap, self.s, self.sh, self.st, self.f]
            .iter()
            .enumerate()
        {
            words[index * 8..(index + 1) * 8].copy_from_slice(&value.to_le_bytes());
        }
        words
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != POLICY_BYTES {
            return Err(Error::InvalidEncoding);
        }
        let mut words = [0u64; POLICY_WORDS];
        for (index, word) in words.iter_mut().enumerate() {
            *word = u64::from_le_bytes(
                bytes[index * 8..(index + 1) * 8]
                    .try_into()
                    .map_err(|_| Error::InvalidEncoding)?,
            );
        }
        Ok(Self {
            c: words[0],
            theta: words[1],
            cap: words[2],
            s: words[3],
            sh: words[4],
            st: words[5],
            f: words[6],
        })
    }

    /// The protocol's policy-word tests. Kinds other than exact/clipped are
    /// rejected for a whole call.
    pub fn validate(&self) -> Result<PolicyKind> {
        if self.theta > self.c
            || self.cap >= (1u64 << 16)
            || self.f != 24
            || self.s > 63
            || (self.sh != 7 && self.sh != 8)
        {
            return Err(Error::InvalidConfig);
        }
        if self.st.checked_add(4) == Some(self.sh) {
            let exact = self.c == (1u64 << self.sh)
                && self.theta == (1u64 << self.sh)
                && self.cap == 0;
            return Ok(if exact {
                PolicyKind::ExactMatch
            } else {
                PolicyKind::ClippedGap
            });
        }
        if self.st >= self.sh.saturating_add(4) && self.st <= self.sh.saturating_add(10) {
            return Err(Error::UnsupportedPrimitive);
        }
        Err(Error::InvalidConfig)
    }

    pub fn base_cost(&self) -> u64 {
        1u64 << self.sh
    }
}

fn clipped_params(clipped: &ClippedGapPolicy) -> Result<PolicyParams> {
    PolicyParams::decode(&clipped.parameters)
}

/// Digest of the encoded policy words.
pub fn policy_digest(policy: &OutputPolicy) -> Result<Digest> {
    let mut writer = CanonicalWriter::new();
    match policy {
        OutputPolicy::ExactMatch => writer.put_u64(0),
        OutputPolicy::ClippedGap(clipped) => {
            writer.put_u64(1);
            let params = clipped_params(clipped)?;
            writer.put_bytes(&params.encode());
        }
    }
    hash_labeled(tag_word(STATEMENT, 11), 0, None, &writer.into_bytes())
}

pub fn validate_statement_policy(statement: &Statement) -> Result<()> {
    statement.validate()?;
    if let OutputPolicy::ClippedGap(clipped) = &statement.output_policy {
        if policy_digest(&statement.output_policy)? != clipped.parameters_digest {
            return Err(Error::StatementMismatch);
        }
        let params = clipped_params(clipped)?;
        let kind = params.validate()?;
        if kind == PolicyKind::ExactMatch {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

fn policy_params(statement: &Statement) -> Result<(PolicyKind, PolicyParams)> {
    match &statement.output_policy {
        OutputPolicy::ExactMatch => {
            let params = PolicyParams {
                c: 1u64 << 7,
                theta: 1u64 << 7,
                cap: 0,
                s: 0,
                sh: 7,
                st: 3,
                f: 24,
            };
            let kind = params.validate()?;
            Ok((kind, params))
        }
        OutputPolicy::ClippedGap(clipped) => {
            let params = clipped_params(clipped)?;
            let kind = params.validate()?;
            Ok((kind, params))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputProof {
    pub tokens: Vec<u32>,
    pub gaps: Vec<i64>,
}

fn check_accounting(
    params: PolicyParams,
    kind: PolicyKind,
    token_count: usize,
    gaps: &[i64],
) -> Result<()> {
    if gaps.len() != token_count {
        return Err(Error::ShapeMismatch);
    }
    if params.cap > token_count as u64 {
        return Err(Error::InvalidConfig);
    }
    let base = params.base_cost() as i128;
    let mut sum = 0i128;
    let mut above = 0u64;
    for gap in gaps {
        if *gap < 0 {
            return Err(Error::BoundViolation);
        }
        let cost = base
            .checked_add(*gap as i128)
            .ok_or(Error::BoundViolation)?;
        sum = sum
            .checked_add(cost.min(params.c as i128))
            .ok_or(Error::BoundViolation)?;
        if cost > params.c as i128 {
            above = above.checked_add(1).ok_or(Error::ResourceLimit)?;
        }
    }
    let limit = (params.theta as i128)
        .checked_mul(token_count as i128)
        .ok_or(Error::BoundViolation)?;
    if sum > limit || above > params.cap {
        return Err(Error::InvalidProof);
    }
    if kind == PolicyKind::ExactMatch && gaps.iter().any(|gap| *gap != 0) {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn check_output(statement: &Statement, proof: &OutputProof) -> Result<()> {
    validate_statement_policy(statement)?;
    if proof.tokens != statement.output_tokens {
        return Err(Error::StatementMismatch);
    }
    let (kind, params) = policy_params(statement)?;
    check_accounting(params, kind, proof.tokens.len(), &proof.gaps)
}

pub fn prove_output(statement: &Statement, proof: OutputProof) -> Result<OutputProof> {
    check_output(statement, &proof)?;
    Ok(proof)
}

pub fn verify_output(statement: &Statement, proof: &OutputProof) -> Result<VerificationResult> {
    check_output(statement, proof)?;
    Ok(VerificationResult {
        accepted: true,
        statement_digest: statement_digest(statement)?,
        output_policy: statement.output_policy.clone(),
    })
}

pub fn coverage_digest(coverage: &DecodeCoverage) -> Result<Digest> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(coverage.decode_len as u64);
    writer.put_u64(coverage.covered_positions.len() as u64);
    for position in &coverage.covered_positions {
        writer.put_u64(*position as u64);
    }
    hash_labeled(tag_word(STATEMENT, 12), 0, None, &writer.into_bytes())
}

fn eos_code(policy: EosPolicy) -> u64 {
    match policy {
        EosPolicy::FixedLength => 0,
        EosPolicy::StopAtFirstEos => 1,
    }
}

pub fn statement_digest(statement: &Statement) -> Result<Digest> {
    statement.validate()?;
    let mut writer = CanonicalWriter::new();
    writer.put_digest(&statement.protocol_id);
    writer.put_u64(statement.protocol_version as u64);
    writer.put_digest(&statement.parameter_digest);
    writer.put_digest(&statement.model_id);
    writer.put_digest(&statement.graph_digest);
    writer.put_digest(&statement.quantization_digest);
    writer.put_digest(&statement.tables_digest);
    writer.put_u64(statement.prompt_tokens.len() as u64);
    for token in &statement.prompt_tokens {
        writer.put_u64(*token as u64);
    }
    writer.put_u64(statement.output_tokens.len() as u64);
    for token in &statement.output_tokens {
        writer.put_u64(*token as u64);
    }
    writer.put_digest(&coverage_digest(&statement.coverage)?);
    writer.put_u64(eos_code(statement.eos_policy));
    writer.put_digest(&policy_digest(&statement.output_policy)?);
    hash_labeled(tag_word(STATEMENT, 13), 0, None, &writer.into_bytes())
}
