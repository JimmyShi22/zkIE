//! The consumer auxiliary proof file and its totals.
//!
//! The file carries the three public stages' messages — the first stage, the
//! shared stage and the second own group — under one versioned, canonical
//! container. A digest over the whole body refuses any changed byte before a
//! single message is decoded against its plan, and the totals repeat the three
//! seals and the two bound roots the acceptance checked.

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::table::SampleSeed;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::consumer_first::{
    ConsumerFirstMessages, ConsumerFirstObligations, ConsumerFirstPlan,
};
use crate::proof::consumer_plan::ConsumerPlanAssembly;
use crate::proof::consumer_second::{
    consumer_total_acceptance, ConsumerOwnPlan, ConsumerOwnProof, ConsumerTotalAccepted,
};
use crate::proof::consumer_shared::{
    consumer_binding_root, ConsumerSharedGeometry, ConsumerSharedStage,
};
use crate::proof::hidden_send_session::{
    decode_shared_messages, encode_shared_messages, HiddenSharedMessages,
};
use crate::proof::lin_binding::{LinBindingPath, LinBindingRoot};
use zkie_engine::pcs::request::table::IndexReader;

/// Version of the consumer proof file.
pub const CONSUMER_PROOF_VERSION: u64 = 2;

/// The accepted totals repeated in the file: the three seals, the Eval target
/// and the two committed roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerProofTotals {
    pub first_seal: Digest,
    pub binding_root: LinBindingRoot,
    pub shared_seal: Digest,
    pub second_seal: Digest,
    pub eval_target: Fp3,
    pub w_root: Digest,
    pub omega_root: Digest,
}

impl ConsumerProofTotals {
    pub fn fault(&self) -> Option<&'static str> {
        if self.first_seal == Digest::ZERO
            || self.binding_root.fault().is_some()
            || self.shared_seal == Digest::ZERO
            || self.second_seal == Digest::ZERO
            || self.w_root == Digest::ZERO
            || self.omega_root == Digest::ZERO
        {
            return Some("a consumer proof total is empty");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidProof))
    }

    fn put(&self, writer: &mut CanonicalWriter) {
        writer.put_digest(&self.first_seal);
        writer.put_digest(&self.binding_root.digest);
        writer.put_u64(self.binding_root.places);
        writer.put_digest(&self.shared_seal);
        writer.put_digest(&self.second_seal);
        writer.put_fp3(&self.eval_target);
        writer.put_digest(&self.w_root);
        writer.put_digest(&self.omega_root);
    }

    fn get(reader: &mut CanonicalReader<'_>) -> Result<Self> {
        Ok(Self {
            first_seal: reader.get_digest()?,
            binding_root: LinBindingRoot {
                digest: reader.get_digest()?,
                places: reader.get_u64()?,
            },
            shared_seal: reader.get_digest()?,
            second_seal: reader.get_digest()?,
            eval_target: reader.get_fp3()?,
            w_root: reader.get_digest()?,
            omega_root: reader.get_digest()?,
        })
    }
}

/// One consumer auxiliary proof file: the three stage messages and the
/// accepted totals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerProofFile {
    pub first: Vec<u8>,
    pub shared: Vec<u8>,
    pub own: Vec<u8>,
    /// The per-request binding path from the verified first seal to the
    /// shared chain's root. The verifier rebuilds the root from its own seal;
    /// the path is checked, not trusted.
    pub binding_path: LinBindingPath,
    /// The prover-dependent plan values of the first stage. The verifier
    /// rebuilds the plan structure from the trusted facts and reads only these
    /// committed values; the first chain and the group bind every one of them.
    pub assembly: Vec<u8>,
    pub totals: ConsumerProofTotals,
}

impl ConsumerProofFile {
    /// Assembles the file from the verified stage results.
    pub fn assemble(
        first_plan: &ConsumerFirstPlan,
        first: &ConsumerFirstMessages,
        shared: &HiddenSharedMessages,
        own: &ConsumerOwnProof,
        accepted: &ConsumerTotalAccepted,
        binding_path: LinBindingPath,
    ) -> Result<Self> {
        let totals = ConsumerProofTotals {
            first_seal: accepted.first_seal,
            binding_root: accepted.binding_root,
            shared_seal: accepted.shared_seal,
            second_seal: accepted.second_seal,
            eval_target: accepted.eval_target,
            w_root: accepted.w_root,
            omega_root: accepted.omega_root,
        };
        Self::assemble_with_totals(first_plan, first, shared, own, totals, binding_path)
    }

    /// The same assembly from the stage results the prover already holds. The
    /// totals carry exactly the seals and roots the stages produced; the
    /// verifier never reads them as evidence, it only compares them with its
    /// own decoded stages before running the chain.
    pub fn assemble_with_totals(
        first_plan: &ConsumerFirstPlan,
        first: &ConsumerFirstMessages,
        shared: &HiddenSharedMessages,
        own: &ConsumerOwnProof,
        totals: ConsumerProofTotals,
        binding_path: LinBindingPath,
    ) -> Result<Self> {
        let group = first_plan.group.as_ref().ok_or(Error::InvalidConfig)?;
        let assembly = ConsumerPlanAssembly {
            member: first_plan.member.clone(),
            target: first_plan.target,
            mask_root: group.mask_root,
            raw_roots: first_plan.draft.raw.raw.roots.clone(),
            bits_roots: first_plan.draft.raw.bits.roots.clone(),
        };
        let file = Self {
            first: first.encode(first_plan)?,
            shared: encode_shared_messages(shared)?,
            own: encode_own_proof(own)?,
            binding_path,
            assembly: assembly.encode()?,
            totals,
        };
        file.check()?;
        Ok(file)
    }

    /// Decodes the prover-dependent plan values carried by this file.
    pub fn plan_assembly(&self) -> Result<ConsumerPlanAssembly> {
        ConsumerPlanAssembly::decode(&self.assembly)
    }

    pub fn fault(&self) -> Option<&'static str> {
        self.totals.fault()
    }

    pub fn check(&self) -> Result<()> {
        self.totals.check()?;
        if self.first.is_empty()
            || self.shared.is_empty()
            || self.own.is_empty()
            || self.assembly.is_empty()
        {
            return Err(Error::InvalidProof);
        }
        self.plan_assembly()?;
        Ok(())
    }

    fn body(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_PROOF_VERSION);
        writer.put_bytes(&self.first);
        writer.put_bytes(&self.shared);
        writer.put_bytes(&self.own);
        writer.put_bytes(&self.binding_path.encode());
        writer.put_bytes(&self.assembly);
        self.totals.put(&mut writer);
        writer.into_bytes()
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.check()?;
        let body = self.body();
        let digest = digest_bytes(&body);
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(&body);
        writer.put_digest(&digest);
        Ok(writer.into_bytes())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader =
            CanonicalReader::new(bytes, zkie_core::common::codec::DecodeLimits::unlimited());
        let body = reader.get_bytes()?;
        let digest = reader.get_digest()?;
        reader.finish()?;
        if digest_bytes(body) != digest {
            return Err(Error::InvalidProof);
        }
        let mut body_reader =
            CanonicalReader::new(body, zkie_core::common::codec::DecodeLimits::unlimited());
        if body_reader.get_u64()? != CONSUMER_PROOF_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let first = body_reader.get_bytes()?.to_vec();
        let shared = body_reader.get_bytes()?.to_vec();
        let own = body_reader.get_bytes()?.to_vec();
        let binding_path = LinBindingPath::decode(body_reader.get_bytes()?)?;
        let assembly = body_reader.get_bytes()?.to_vec();
        let totals = ConsumerProofTotals::get(&mut body_reader)?;
        body_reader.finish()?;
        let file = Self {
            first,
            shared,
            own,
            binding_path,
            assembly,
            totals,
        };
        file.check()?;
        Ok(file)
    }

    /// Decodes the own group proof that this file carries.
    pub fn own_proof(&self) -> Result<ConsumerOwnProof> {
        decode_own_proof(&self.own)
    }
}

/// The sidecar file name in the proof output directory.
pub const CONSUMER_PROOF_FILE: &str = "consumer_proof.bin";

/// The largest admitted sidecar file. The three stages' messages are far
/// below it; a larger file is refused before it is read.
pub const CONSUMER_PROOF_MAX_BYTES: u64 = 64 * 1024 * 1024;

impl ConsumerProofFile {
    /// Writes the sidecar file beside the proof artifact. The bytes are
    /// written to a temporary name in the same directory and renamed, so a
    /// reader never observes a partial file.
    pub fn write_to(&self, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let bytes = self.encode()?;
        let path = dir.join(CONSUMER_PROOF_FILE);
        let temporary = dir.join(format!(".{CONSUMER_PROOF_FILE}.tmp"));
        std::fs::write(&temporary, &bytes).map_err(|_| Error::Io)?;
        std::fs::rename(&temporary, &path).map_err(|_| Error::Io)?;
        Ok(path)
    }

    /// Reads and decodes the sidecar file. A missing, oversized or malformed
    /// file is refused; the container digest is checked before any section is
    /// decoded against its plan.
    pub fn read_from(dir: &std::path::Path) -> Result<Self> {
        let path = dir.join(CONSUMER_PROOF_FILE);
        let metadata = std::fs::metadata(&path).map_err(|_| Error::InvalidProof)?;
        if metadata.len() > CONSUMER_PROOF_MAX_BYTES {
            return Err(Error::ResourceLimit);
        }
        let bytes = std::fs::read(&path).map_err(|_| Error::InvalidProof)?;
        Self::decode(&bytes)
    }
}

/// Verifies one consumer proof file in the global order: first stage, shared
/// stage, own group and total acceptance. The file's totals must equal the
/// decoded stage seals and bound roots before any stage runs; every stage
/// then verifies against its own plan.
#[allow(clippy::too_many_arguments)]
pub fn verify_consumer_chain(
    first_plan: &ConsumerFirstPlan,
    activation: &[Fp3],
    outputs: &[Vec<i64>],
    geometry: &ConsumerSharedGeometry,
    shared_stage: &ConsumerSharedStage,
    reader: &IndexReader,
    own_plan: &ConsumerOwnPlan,
    file: &ConsumerProofFile,
) -> Result<ConsumerTotalAccepted> {
    file.check()?;
    let first = ConsumerFirstMessages::decode(first_plan, &file.first)?;
    let shared = decode_shared_messages(&file.shared)?;
    let own = file.own_proof()?;
    let binding_root = consumer_binding_root(
        &first.first_seal,
        &geometry.binding_shape,
        &geometry.binding_configuration,
        &file.binding_path,
    )?;
    if file.totals.first_seal != first.first_seal
        || file.totals.binding_root != binding_root
        || file.totals.shared_seal != shared.seal
        || file.totals.second_seal != own.seal
        || file.totals.eval_target != first.eval_target
        || file.totals.w_root != first.w_root
        || file.totals.omega_root != own_plan.omega_root
    {
        return Err(Error::InvalidProof);
    }
    consumer_total_acceptance(
        first_plan,
        activation,
        outputs,
        &first,
        geometry,
        shared_stage,
        reader,
        &shared,
        &file.binding_path,
        own_plan,
        &own,
    )
}

fn put_seed(writer: &mut CanonicalWriter, seed: &SampleSeed) {
    writer.put_u64(seed.z.len() as u64);
    for value in &seed.z {
        writer.put_fp3(value);
    }
    writer.put_fp3(&seed.beta);
}

fn get_seed(reader: &mut CanonicalReader<'_>) -> Result<SampleSeed> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count == 0 || count > 64 {
        return Err(Error::ResourceLimit);
    }
    let mut z = Vec::with_capacity(count);
    for _ in 0..count {
        z.push(reader.get_fp3()?);
    }
    let beta = reader.get_fp3()?;
    if beta == Fp3::ZERO {
        return Err(Error::InvalidProof);
    }
    Ok(SampleSeed { z, beta })
}

/// The own group proof bytes: the drawn W seeds, their answers, the proximity
/// messages and the seal.
pub fn encode_own_proof(own: &ConsumerOwnProof) -> Result<Vec<u8>> {
    if own.w_seeds.is_empty() || own.w_seeds.len() != own.w_answers.len() {
        return Err(Error::InvalidProof);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_u64(CONSUMER_PROOF_VERSION);
    writer.put_u64(own.w_seeds.len() as u64);
    for seed in &own.w_seeds {
        put_seed(&mut writer, seed);
    }
    for answer in &own.w_answers {
        writer.put_fp3(answer);
    }
    writer.put_bytes(&own.messages);
    writer.put_digest(&own.seal);
    Ok(writer.into_bytes())
}

pub fn decode_own_proof(bytes: &[u8]) -> Result<ConsumerOwnProof> {
    let mut reader =
        CanonicalReader::new(bytes, zkie_core::common::codec::DecodeLimits::unlimited());
    if reader.get_u64()? != CONSUMER_PROOF_VERSION {
        return Err(Error::InvalidEncoding);
    }
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count == 0 || count > 64 {
        return Err(Error::ResourceLimit);
    }
    let mut w_seeds = Vec::with_capacity(count);
    for _ in 0..count {
        w_seeds.push(get_seed(&mut reader)?);
    }
    let mut w_answers = Vec::with_capacity(count);
    for _ in 0..count {
        w_answers.push(reader.get_fp3()?);
    }
    let messages = reader.get_bytes()?.to_vec();
    let seal = reader.get_digest()?;
    reader.finish()?;
    if messages.is_empty() || seal == Digest::ZERO {
        return Err(Error::InvalidProof);
    }
    Ok(ConsumerOwnProof {
        w_seeds,
        w_answers,
        messages,
        seal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use zkie_engine::pcs::request::field::fp_from_u64;

    fn fp(value: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(value))
    }

    fn own() -> ConsumerOwnProof {
        ConsumerOwnProof {
            w_seeds: vec![SampleSeed {
                z: vec![fp(1), fp(2), fp(3)],
                beta: fp(5),
            }],
            w_answers: vec![fp(9)],
            messages: vec![1, 2, 3, 4],
            seal: Digest([0x11; 32]),
        }
    }

    fn assembly() -> Vec<u8> {
        ConsumerPlanAssembly {
            member: crate::proof::hidden_member::PendingRow {
                weights: vec![crate::proof::hidden_member::WeightBlock::Scalar {
                    row: 0,
                    value: fp(1),
                }],
                psi: vec![fp(2)],
                target: fp(3),
                masks: Vec::new(),
            },
            target: fp(4),
            mask_root: Digest([0x31; 32]),
            raw_roots: vec![Digest([0x32; 32])],
            bits_roots: vec![Digest([0x33; 32])],
        }
        .encode()
        .unwrap()
    }

    fn totals() -> ConsumerProofTotals {
        ConsumerProofTotals {
            first_seal: Digest([0x21; 32]),
            binding_root: LinBindingRoot {
                digest: Digest([0x24; 32]),
                places: 1,
            },
            shared_seal: Digest([0x22; 32]),
            second_seal: Digest([0x23; 32]),
            eval_target: fp(7),
            w_root: Digest([0x24; 32]),
            omega_root: Digest([0x25; 32]),
        }
    }

    #[test]
    fn the_own_proof_round_trips_and_changed_bytes_are_refused() {
        let proof = own();
        let bytes = encode_own_proof(&proof).unwrap();
        assert_eq!(decode_own_proof(&bytes).unwrap(), proof);
        // The container checks the structure; a changed seal still decodes
        // and is caught by the file digest and by the stage verification. A
        // zero seed coordinate count is a structural error and is refused.
        let mut changed = bytes.clone();
        let last = changed.len() - 1;
        changed[last] ^= 1;
        assert_ne!(decode_own_proof(&changed).unwrap().seal, proof.seal);
        let mut broken = bytes.clone();
        broken[16..24].copy_from_slice(&0u64.to_le_bytes());
        assert!(decode_own_proof(&broken).is_err());
        let empty = ConsumerOwnProof {
            w_seeds: Vec::new(),
            w_answers: Vec::new(),
            messages: vec![1],
            seal: Digest([0x11; 32]),
        };
        assert!(encode_own_proof(&empty).is_err());
    }

    #[test]
    fn the_sidecar_file_round_trips_and_a_missing_or_oversized_file_is_refused() {
        let root = std::env::temp_dir().join(format!(
            "zkie-consumer-proof-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let file = ConsumerProofFile {
            first: vec![9, 9, 9],
            shared: vec![8, 8],
            own: encode_own_proof(&own()).unwrap(),
            binding_path: LinBindingPath {
                place: 0,
                salt: vec![7u8; 40],
                siblings: Vec::new(),
            },
            assembly: assembly(),
            totals: totals(),
        };
        let path = file.write_to(&root).unwrap();
        assert!(path.ends_with(CONSUMER_PROOF_FILE));
        assert_eq!(ConsumerProofFile::read_from(&root).unwrap(), file);
        // A missing file is refused.
        let empty = root.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(ConsumerProofFile::read_from(&empty).is_err());
        // A file above the size bound is refused before it is read.
        let big = root.join("big");
        std::fs::create_dir_all(&big).unwrap();
        std::fs::write(
            big.join(CONSUMER_PROOF_FILE),
            vec![0u8; (CONSUMER_PROOF_MAX_BYTES + 1) as usize],
        )
        .unwrap();
        assert!(ConsumerProofFile::read_from(&big).is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_proof_file_round_trips_and_a_changed_body_is_refused() {
        let file = ConsumerProofFile {
            first: vec![9, 9, 9],
            shared: vec![8, 8],
            own: encode_own_proof(&own()).unwrap(),
            binding_path: LinBindingPath {
                place: 0,
                salt: vec![7u8; 40],
                siblings: Vec::new(),
            },
            assembly: assembly(),
            totals: totals(),
        };
        let bytes = file.encode().unwrap();
        assert_eq!(ConsumerProofFile::decode(&bytes).unwrap(), file);
        assert_eq!(file.own_proof().unwrap(), own());
        let mut changed = bytes.clone();
        let mid = changed.len() / 2;
        changed[mid] ^= 0x40;
        assert!(ConsumerProofFile::decode(&changed).is_err());
        // A missing section or an empty total is refused.
        let mut bad = file.clone();
        bad.first.clear();
        assert!(bad.encode().is_err());
        let mut bad = file;
        bad.totals.second_seal = Digest::ZERO;
        assert!(bad.encode().is_err());
    }
}
