//! Shared committed-message and product-sumcheck row-binding primitives.
//!
//! A committed product-sumcheck relation has the same mechanical shape in
//! every relation that uses it:
//!
//! * it declares one or more SEND groups whose value rows stay with the
//!   prover and whose roots enter the proof message;
//! * it draws sample answers and opening positions, and sends one runtime
//!   opening per position and group;
//! * it runs one masked product-sumcheck transcript per position, sends the
//!   masked point values and settles each transcript into point claims on the
//!   committed rows.
//!
//! The relation-specific polynomials, masks and row equations stay with the
//! relation. This module owns the message codecs, the position-domain
//! challenge source and the checked binding shell, so all relations use one
//! implementation of those mechanics. The codecs are byte-level: field order
//! and length prefixes are part of the wire format.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, KIND_EMPTY};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::runtime_tree::{RuntimeTreeOpening, RuntimeTreePlan};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::hidden_send_session::{SendGroup, SendGroupEntry, SendSamples, WShape};
use crate::proof::psc::{DotOutput, DotProof, PscChallenges, PscRound, SHOWN_MAPS};

/// The committed group declaration carried by a proof message. The value rows
/// and the retained commit trees are witness state: they stay with the prover
/// and are never part of the verifier-held message or of its encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedGroupMessage {
    pub group: SendGroup,
    /// The committed roots in position order, derived from the group entries.
    pub roots: Vec<Digest>,
}

impl CommittedGroupMessage {
    pub fn new(group: SendGroup) -> Self {
        let roots = group.entries.iter().map(|entry| entry.root).collect();
        Self { group, roots }
    }
}

/// One declared and opened committed group: the group message, its SEND
/// sample, the drawn opening positions and the runtime openings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedPscGroup {
    pub message: CommittedGroupMessage,
    pub sample: SendSamples,
    pub positions: Vec<Vec<u64>>,
    pub openings: Vec<RuntimeTreeOpening>,
}

/// A borrowed view of [`CommittedPscGroup`], for callers that keep the four
/// parts in separate fields.
#[derive(Clone, Copy, Debug)]
pub struct CommittedPscGroupRef<'a> {
    pub message: &'a CommittedGroupMessage,
    pub sample: &'a SendSamples,
    pub positions: &'a [Vec<u64>],
    pub openings: &'a [RuntimeTreeOpening],
}

impl<'a> CommittedPscGroupRef<'a> {
    pub fn from_group(group: &'a CommittedPscGroup) -> Self {
        Self {
            message: &group.message,
            sample: &group.sample,
            positions: &group.positions,
            openings: &group.openings,
        }
    }

    /// Writes the group message, the SEND sample, the opening positions and
    /// the runtime openings, in that order.
    pub fn encode_into(&self, writer: &mut CanonicalWriter) -> Result<()> {
        writer.put_bytes(&self.message.group.encode());
        writer.put_bytes(&self.sample.encode());
        put_positions(writer, self.positions)?;
        put_openings(writer, &self.message.group.entries, self.openings)
    }
}

impl CommittedPscGroup {
    pub fn encode_into(&self, writer: &mut CanonicalWriter) -> Result<()> {
        CommittedPscGroupRef::from_group(self).encode_into(writer)
    }

    /// Reads one declared and opened group. The runtime tree plans come from
    /// the trusted SEND shape, never from the message.
    pub fn decode(
        reader: &mut CanonicalReader,
        shape: &WShape,
        position_count: u32,
    ) -> Result<Self> {
        let message = CommittedGroupMessage::new(SendGroup::decode(reader.get_bytes()?)?);
        let sample = SendSamples::decode(reader.get_bytes()?)?;
        let positions = get_positions(reader)?;
        let openings = get_openings(reader, shape, position_count, &positions)?;
        Ok(Self {
            message,
            sample,
            positions,
            openings,
        })
    }

    /// Shape and sampling checks that a caller performs before sending or
    /// accepting the group. The message itself was decoded, so the roots and
    /// tree plans are the trusted ones.
    pub fn fault(&self, shape: &WShape, position_count: u32) -> Option<&'static str> {
        let count = position_count as usize;
        if self.message.group.entries.len() != count
            || self.message.roots.len() != count
            || self.positions.len() != count
            || self.openings.len() != count
            || self.sample.seed.len() != shape.answers as usize
            || self.sample.answers.len() != count
        {
            return Some("a committed PSC group does not have one sampled member per position");
        }
        for (index, (list, opening)) in self.positions.iter().zip(&self.openings).enumerate() {
            if self.message.group.entries[index].position != index as u32 {
                return Some("a committed PSC group is not in position order");
            }
            if list.is_empty() || opening.positions != *list {
                return Some("a committed PSC opening is not the sampled position list");
            }
            if self.message.roots[index] != self.message.group.entries[index].root {
                return Some("a committed PSC group root is not its member root");
            }
        }
        None
    }
}

/// One position's settled product-sumcheck binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PscPointBinding<const POINTS: usize> {
    /// The fresh point of the committed-product relation.
    pub q: Vec<Fp3>,
    /// The masked point values sent by the prover.
    pub point_values: [Fp3; POINTS],
    /// The linear claims that the transcript leaves for the relation rows.
    pub output: DotOutput,
}

/// The settled bindings of every position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PscRowBinding<const POINTS: usize> {
    pub positions: Vec<PscPointBinding<POINTS>>,
}

impl<const POINTS: usize> PscRowBinding<POINTS> {
    pub fn fault(&self, position_count: usize, variables: usize) -> Option<&'static str> {
        if self.positions.len() != position_count {
            return Some("a PSC row binding does not have one position per instance");
        }
        if self
            .positions
            .iter()
            .any(|binding| binding.q.len() != variables || binding.output.point.len() != variables)
        {
            return Some("a PSC row binding has another variable count");
        }
        None
    }
}

fn put_positions(writer: &mut CanonicalWriter, positions: &[Vec<u64>]) -> Result<()> {
    writer.put_u64(positions.len() as u64);
    for list in positions {
        writer.put_u64(list.len() as u64);
        for position in list {
            writer.put_u64(*position);
        }
    }
    Ok(())
}

fn get_positions(reader: &mut CanonicalReader) -> Result<Vec<Vec<u64>>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        let length = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if length == 0 || length > (1 << 20) {
            return Err(Error::InvalidEncoding);
        }
        let mut list = Vec::with_capacity(length);
        for _ in 0..length {
            list.push(reader.get_u64()?);
        }
        positions.push(list);
    }
    Ok(positions)
}

fn put_openings(
    writer: &mut CanonicalWriter,
    entries: &[SendGroupEntry],
    openings: &[RuntimeTreeOpening],
) -> Result<()> {
    if entries.len() != openings.len() {
        return Err(Error::ShapeMismatch);
    }
    writer.put_u64(openings.len() as u64);
    for (entry, opening) in entries.iter().zip(openings) {
        writer.put_bytes(
            &zkie_engine::pcs::request::runtime_tree::encode_runtime_opening(
                &entry.tree,
                opening,
            )?,
        );
    }
    Ok(())
}

fn get_openings(
    reader: &mut CanonicalReader,
    shape: &WShape,
    position_count: u32,
    opening_positions: &[Vec<u64>],
) -> Result<Vec<RuntimeTreeOpening>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count != position_count as usize || opening_positions.len() != position_count as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut openings = Vec::with_capacity(count);
    for position in 0..position_count {
        let tree = shape.tree_plan(position)?;
        let opening = zkie_engine::pcs::request::runtime_tree::decode_runtime_opening(
            &tree,
            &opening_positions[position as usize],
            reader.get_bytes()?,
        )?;
        if opening.positions != opening_positions[position as usize] {
            return Err(Error::InvalidEncoding);
        }
        openings.push(opening);
    }
    Ok(openings)
}

pub(crate) fn put_tree_plan(writer: &mut CanonicalWriter, plan: &RuntimeTreePlan) {
    writer.put_u64(plan.tree_number);
    writer.put_u64(plan.leaf_class as u64);
    writer.put_u64(plan.log_n as u64);
    writer.put_u64(plan.log_n_zk as u64);
    writer.put_u64(plan.parts.len() as u64);
    for part in &plan.parts {
        writer.put_u64(part.main_symbols);
        writer.put_u64(part.mask_symbols);
    }
    writer.put_u64(plan.salt_bytes as u64);
    writer.put_digest(&plan.run);
    writer.put_digest(&plan.configuration);
}

pub(crate) fn get_tree_plan(reader: &mut CanonicalReader) -> Result<RuntimeTreePlan> {
    let tree_number = reader.get_u64()?;
    let leaf_class = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let log_n = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let log_n_zk = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count == 0 || count > 4 {
        return Err(Error::InvalidEncoding);
    }
    let mut parts = Vec::with_capacity(count);
    for _ in 0..count {
        parts.push(zkie_engine::pcs::request::runtime_tree::RuntimeTreePart {
            main_symbols: reader.get_u64()?,
            mask_symbols: reader.get_u64()?,
        });
    }
    let salt_bytes = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let run = reader.get_digest()?;
    let configuration = reader.get_digest()?;
    Ok(RuntimeTreePlan {
        tree_number,
        leaf_class,
        log_n,
        log_n_zk,
        parts,
        salt_bytes,
        run,
        configuration,
    })
}

/// Writes the version, the number of transcripts, the masked point values and
/// the product-sumcheck transcripts, in that order.
pub fn put_psc_transcript<const POINTS: usize>(
    writer: &mut CanonicalWriter,
    version: u64,
    psc: &[DotProof],
    point_values: &[[Fp3; POINTS]],
) -> Result<()> {
    writer.put_u64(version);
    writer.put_u64(psc.len() as u64);
    writer.put_u64(point_values.len() as u64);
    for values in point_values {
        for value in values {
            writer.put_fp3(value);
        }
    }
    for proof in psc {
        writer.put_fp3(&proof.hiding_sum);
        writer.put_u64(proof.rounds.len() as u64);
        for round in &proof.rounds {
            writer.put_u64(round.coefficients.len() as u64);
            for coefficient in &round.coefficients {
                writer.put_fp3(coefficient);
            }
        }
        writer.put_u64(proof.y_round.coefficients.len() as u64);
        for coefficient in &proof.y_round.coefficients {
            writer.put_fp3(coefficient);
        }
        writer.put_u64(proof.end_values.len() as u64);
        for value in &proof.end_values {
            writer.put_fp3(value);
        }
    }
    Ok(())
}

/// Reads a product-sumcheck transcript batch with `position_count` instances
/// of `variables` Boolean variables and the fixed number of shown maps.
pub fn get_psc_transcript<const POINTS: usize>(
    reader: &mut CanonicalReader,
    version: u64,
    position_count: u32,
    variables: usize,
) -> Result<(Vec<DotProof>, Vec<[Fp3; POINTS]>)> {
    if reader.get_u64()? != version {
        return Err(Error::InvalidEncoding);
    }
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count != position_count as usize {
        return Err(Error::ShapeMismatch);
    }
    let value_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if value_count != position_count as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut point_values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        let mut values = [Fp3::ZERO; POINTS];
        for value in &mut values {
            *value = reader.get_fp3()?;
        }
        point_values.push(values);
    }
    let mut psc = Vec::with_capacity(count);
    for _ in 0..count {
        let hiding_sum = reader.get_fp3()?;
        let round_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if round_count != variables {
            return Err(Error::ShapeMismatch);
        }
        let mut rounds = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            let length = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            if length != 2 {
                return Err(Error::ShapeMismatch);
            }
            let mut coefficients = Vec::with_capacity(length);
            for _ in 0..length {
                coefficients.push(reader.get_fp3()?);
            }
            rounds.push(PscRound { coefficients });
        }
        let y_len = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if y_len != 2 {
            return Err(Error::ShapeMismatch);
        }
        let mut y_coefficients = Vec::with_capacity(y_len);
        for _ in 0..y_len {
            y_coefficients.push(reader.get_fp3()?);
        }
        let end_len = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if end_len != SHOWN_MAPS {
            return Err(Error::ShapeMismatch);
        }
        let mut end_values = [Fp3::ZERO; SHOWN_MAPS];
        for value in &mut end_values {
            *value = reader.get_fp3()?;
        }
        psc.push(DotProof {
            hiding_sum,
            rounds,
            y_round: PscRound {
                coefficients: y_coefficients,
            },
            end_values,
        });
    }
    Ok((psc, point_values))
}

/// The transcript tags of one committed product-sumcheck relation. The
/// version and the tag words separate the instances of different relations
/// and the position word separates the instances of different positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PscTranscriptTags {
    pub version: u64,
    pub hiding_sum: u64,
    pub round: u64,
    pub end: u64,
    /// The masked claim of a general sumcheck, absorbed before the first
    /// challenge that multiplies the target.
    pub claim: u64,
}

/// One position's chain-backed challenge source. The position is part of
/// every absorbed message so the instances cannot be reordered or replayed.
pub struct PositionPscChallenges<'a> {
    chain: &'a mut Chain,
    tags: PscTranscriptTags,
    position: u64,
}

impl<'a> PositionPscChallenges<'a> {
    pub fn new(chain: &'a mut Chain, tags: PscTranscriptTags, position: u64) -> Self {
        Self {
            chain,
            tags,
            position,
        }
    }
}

impl PositionPscChallenges<'_> {
    /// Absorbs the masked claim before the first sumcheck challenge. The
    /// general sumcheck's first challenge multiplies the target, so every
    /// value that enters the target must be fixed in the transcript first.
    pub fn send_claim(&mut self, value: Fp3) -> Result<()> {
        let mut message = self.chain.message();
        message.push_u64(self.tags.version);
        message.push_u64(self.tags.claim);
        message.push_u64(self.position);
        message.push_fp3(&value);
        let prefix =
            message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

impl PscChallenges for PositionPscChallenges<'_> {
    fn draw(&mut self) -> Result<Fp3> {
        self.chain.draw_e()
    }

    fn draw_nonzero(&mut self) -> Result<Fp3> {
        self.chain.draw_e_nonzero()
    }

    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()> {
        let mut message = self.chain.message();
        message.push_u64(self.tags.version);
        message.push_u64(self.tags.hiding_sum);
        message.push_u64(self.position);
        message.push_fp3(&value);
        let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }

    fn send_round(&mut self, index: usize, round: &PscRound) -> Result<()> {
        let mut message = self.chain.message();
        message.push_u64(self.tags.version);
        message.push_u64(self.tags.round);
        message.push_u64(self.position);
        message.push_u64(index as u64);
        message.push_u64(round.coefficients.len() as u64);
        for coefficient in &round.coefficients {
            message.push_fp3(coefficient);
        }
        let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }

    fn send_end_values(&mut self, values: &[Fp3; SHOWN_MAPS]) -> Result<()> {
        let mut message = self.chain.message();
        message.push_u64(self.tags.version);
        message.push_u64(self.tags.end);
        message.push_u64(self.position);
        message.push_u64(values.len() as u64);
        for value in values {
            message.push_fp3(value);
        }
        let prefix = message.prefix(KIND_EMPTY, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

impl crate::proof::psc_general::GeneralChallenges for PositionPscChallenges<'_> {
    fn draw(&mut self) -> Result<Fp3> {
        PscChallenges::draw(self)
    }

    fn draw_nonzero(&mut self) -> Result<Fp3> {
        PscChallenges::draw_nonzero(self)
    }

    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()> {
        PscChallenges::send_hiding_sum(self, value)
    }

    fn send_round(&mut self, index: usize, round: &PscRound) -> Result<()> {
        PscChallenges::send_round(self, index, round)
    }

    fn send_end_values(&mut self, values: &[Fp3]) -> Result<()> {
        let mut message = self.chain.message();
        message.push_u64(self.tags.version);
        message.push_u64(self.tags.end);
        message.push_u64(self.position);
        message.push_u64(values.len() as u64);
        for value in values {
            message.push_fp3(value);
        }
        let prefix = message.prefix(KIND_EMPTY, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

/// Absorbs one position's masked point values before the relation draws the
/// next point.
pub fn absorb_point(
    chain: &mut Chain,
    version: u64,
    tag: u64,
    position: u64,
    values: &[Fp3],
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(version);
    message.push_u64(tag);
    message.push_u64(position);
    message.push_u64(values.len() as u64);
    for value in values {
        message.push_fp3(value);
    }
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::codec::DecodeLimits;
    use zkie_core::common::field::PrimeField64;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::gu::GuOracleShape;
    use zkie_engine::pcs::request::runtime_tree::RuntimeTreePart;

    use zkie_engine::pcs::request::table::SampleSeed;

    use crate::proof::hidden_send_session::SendGroupEntry;

    fn tree_plan() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 7,
            leaf_class: 1,
            log_n: 4,
            log_n_zk: 4,
            parts: vec![RuntimeTreePart {
                main_symbols: 1,
                mask_symbols: 1,
            }],
            salt_bytes: 40,
            run: Digest([0x11; 32]),
            configuration: Digest([0x22; 32]),
        }
    }

    fn shape() -> WShape {
        WShape::new(
            4,
            build_code_plan(4, 8, 1).unwrap(),
            1,
            1,
            40,
            7,
            1,
            Digest([0x11; 32]),
            Digest([0x22; 32]),
        )
        .unwrap()
    }

    fn group() -> SendGroup {
        SendGroup {
            group: 3,
            entries: vec![SendGroupEntry {
                position: 0,
                oracle: 9,
                shape: GuOracleShape {
                    code: build_code_plan(4, 8, 1).unwrap(),
                    main_rows: 1,
                    main_deg: 3,
                    mask_rows: 0,
                    mask_code: None,
                    answers: 1,
                },
                tree: tree_plan(),
                root: Digest([0x33; 32]),
            }],
        }
    }

    fn proof() -> DotProof {
        DotProof {
            hiding_sum: Fp3::ONE,
            rounds: vec![
                PscRound {
                    coefficients: vec![Fp3::ONE, Fp3::ZERO],
                },
                PscRound {
                    coefficients: vec![Fp3::ZERO, Fp3::ONE],
                },
            ],
            y_round: PscRound {
                coefficients: vec![Fp3::ONE, Fp3::ONE],
            },
            end_values: [Fp3::ONE, Fp3::ZERO],
        }
    }

    fn run_challenges(
        chain: &mut Chain,
        tags: PscTranscriptTags,
        position: u64,
    ) -> (Fp3, Fp3, Fp3) {
        let mut source = PositionPscChallenges::new(chain, tags, position);
        source.send_hiding_sum(Fp3::ONE).unwrap();
        let first = source.draw().unwrap();
        source
            .send_round(
                0,
                &PscRound {
                    coefficients: vec![Fp3::ONE, Fp3::ZERO],
                },
            )
            .unwrap();
        let second = source.draw_nonzero().unwrap();
        source.send_end_values(&[Fp3::ONE, Fp3::ONE]).unwrap();
        let third = source.draw().unwrap();
        (first, second, third)
    }

    #[test]
    fn positions_and_tree_plans_round_trip() {
        let positions = vec![vec![0u64, 3], vec![5]];
        let mut writer = CanonicalWriter::new();
        put_positions(&mut writer, &positions).unwrap();
        let bytes = writer.into_bytes();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert_eq!(get_positions(&mut reader).unwrap(), positions);
        reader.finish().unwrap();

        let plan = tree_plan();
        let mut writer = CanonicalWriter::new();
        put_tree_plan(&mut writer, &plan);
        let bytes = writer.into_bytes();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert_eq!(get_tree_plan(&mut reader).unwrap(), plan);
        reader.finish().unwrap();
    }

    #[test]
    fn malformed_position_and_tree_messages_are_rejected() {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(1);
        writer.put_u64(0);
        let bytes = writer.into_bytes();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(get_positions(&mut reader).is_err());

        let mut writer = CanonicalWriter::new();
        for word in [9u64, 1, 4, 4, 0] {
            writer.put_u64(word);
        }
        writer.put_u64(40);
        writer.put_digest(&Digest([1; 32]));
        writer.put_digest(&Digest([2; 32]));
        let bytes = writer.into_bytes();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(get_tree_plan(&mut reader).is_err());
    }

    #[test]
    fn product_sumcheck_transcript_round_trips() {
        let psc = vec![proof(), proof()];
        let points = vec![
            [Fp3::ONE, Fp3::ZERO, Fp3::ONE],
            [Fp3::ZERO, Fp3::ONE, Fp3::ONE],
        ];
        let mut writer = CanonicalWriter::new();
        put_psc_transcript(&mut writer, 4, &psc, &points).unwrap();
        let bytes = writer.into_bytes();
        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        let (decoded_psc, decoded_points) =
            get_psc_transcript::<3>(&mut reader, 4, 2, 2).unwrap();
        assert_eq!(decoded_psc, psc);
        assert_eq!(decoded_points, points);
        reader.finish().unwrap();
    }

    #[test]
    fn malformed_product_sumcheck_transcripts_are_rejected() {
        let psc = vec![proof()];
        let points = vec![[Fp3::ONE, Fp3::ZERO, Fp3::ONE]];
        let mut writer = CanonicalWriter::new();
        put_psc_transcript(&mut writer, 4, &psc, &points).unwrap();
        let bytes = writer.into_bytes();

        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(get_psc_transcript::<3>(&mut reader, 5, 1, 2).is_err());

        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(get_psc_transcript::<3>(&mut reader, 4, 2, 2).is_err());

        let mut reader = CanonicalReader::new(&bytes, DecodeLimits::unlimited());
        assert!(get_psc_transcript::<3>(&mut reader, 4, 1, 1).is_err());

        let mut trailing = bytes.clone();
        trailing.push(0);
        let mut reader = CanonicalReader::new(&trailing, DecodeLimits::unlimited());
        get_psc_transcript::<3>(&mut reader, 4, 1, 2).unwrap();
        assert!(reader.finish().is_err());
    }

    #[test]
    fn committed_groups_derive_roots_and_reject_broken_shapes() {
        let message = CommittedGroupMessage::new(group());
        assert_eq!(message.roots, vec![Digest([0x33; 32])]);

        let mut committed = CommittedPscGroup {
            message,
            sample: SendSamples {
                seed: vec![SampleSeed {
                    z: Vec::new(),
                    beta: Fp3::ONE,
                }],
                answers: vec![Fp3::ONE],
            },
            positions: vec![vec![0]],
            openings: vec![RuntimeTreeOpening {
                positions: vec![0],
                columns: Vec::new(),
                given: Vec::new(),
            }],
        };
        assert!(committed.fault(&shape(), 1).is_none());

        let mut changed = committed.clone();
        changed.positions = vec![Vec::new()];
        assert!(changed.fault(&shape(), 1).is_some());

        let mut changed = committed.clone();
        changed.openings[0].positions = vec![1];
        assert!(changed.fault(&shape(), 1).is_some());

        let mut changed = committed.clone();
        changed.sample.answers.clear();
        assert!(changed.fault(&shape(), 1).is_some());

        committed.sample.seed.clear();
        assert!(committed.fault(&shape(), 1).is_some());
    }

    #[test]
    fn challenge_sources_are_deterministic_and_position_domain_separated() {
        let tags = PscTranscriptTags {
            version: 1,
            hiding_sum: 20,
            round: 21,
            end: 22,
            claim: 23,
        };
        let mut first = Chain::begin(41, &Digest::ZERO, b"committed psc", &[], None).unwrap();
        let first = run_challenges(&mut first, tags, 0);
        let mut second = Chain::begin(41, &Digest::ZERO, b"committed psc", &[], None).unwrap();
        assert_eq!(first, run_challenges(&mut second, tags, 0));

        let mut other_position =
            Chain::begin(41, &Digest::ZERO, b"committed psc", &[], None).unwrap();
        assert_ne!(first, run_challenges(&mut other_position, tags, 1));

        let mut other_tags = Chain::begin(41, &Digest::ZERO, b"committed psc", &[], None).unwrap();
        let changed = PscTranscriptTags {
            version: 2,
            ..tags
        };
        assert_ne!(first, run_challenges(&mut other_tags, changed, 0));
    }

    #[test]
    fn claim_absorption_precedes_the_first_challenge() {
        use crate::proof::psc::Weight;
        use crate::proof::psc_general::{
            general_masks, general_variables, prove_general, verify_general,
            GeneralMaskEntries, Summand, WeightedProductInner,
        };

        let z = [Fp3::from_u64(2), Fp3::from_u64(3)];
        let x = [1u64, 2, 3, 4].map(Fp3::from_u64);
        let omega = [5u64, 6, 7, 8].map(Fp3::from_u64);
        let eq = |value: Fp3, bit: bool| {
            if bit {
                value
            } else {
                Fp3::ONE.sub(&value)
            }
        };
        let mut target = Fp3::ZERO;
        for index in 0..4 {
            let factor = eq(z[0], index & 1 == 1).mul(&eq(z[1], index & 2 == 2));
            target = target.add(&factor.mul(&x[index]).mul(&omega[index]));
        }
        let variables = [
            Weight::equality(z[0], 3).unwrap(),
            Weight::equality(z[1], 3).unwrap(),
        ];
        let entries = crate::proof::psc::hp_entries(
            &general_variables(&variables, &Summand::Product).unwrap(),
        )
        .unwrap();
        assert_eq!(
            entries + 2,
            general_masks(&Summand::Product, &variables).unwrap()
        );
        let hiding = (0..entries)
            .map(|index| Fp3::from_u64(100 + index as u64))
            .collect::<Vec<_>>();
        let masks = GeneralMaskEntries {
            hiding,
            shown: vec![Fp3::from_u64(200), Fp3::from_u64(201)],
        };
        let tags = PscTranscriptTags {
            version: 1,
            hiding_sum: 60,
            round: 61,
            end: 62,
            claim: 63,
        };
        let mut inner = WeightedProductInner::new(&x, &omega, &z).unwrap();
        let mut prover_chain =
            Chain::begin(88, &Digest::ZERO, b"claim order", &[], None).unwrap();
        let mut prover = PositionPscChallenges::new(&mut prover_chain, tags, 0);
        prover.send_claim(target).unwrap();
        let (proof, proven) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &Summand::Product,
            None,
            &mut inner,
            &masks,
            &mut prover,
        )
        .unwrap();

        let mut verifier_chain =
            Chain::begin(88, &Digest::ZERO, b"claim order", &[], None).unwrap();
        let mut verifier = PositionPscChallenges::new(&mut verifier_chain, tags, 0);
        verifier.send_claim(target).unwrap();
        let verified = verify_general(
            &variables,
            &[],
            target,
            &Summand::Product,
            None,
            &proof,
            masks.hiding.len(),
            &mut verifier,
        )
        .unwrap();
        assert_eq!(proven, verified);

        // The old form leaves the claim out of the pre-challenge transcript:
        // the challenges differ, so the verifier derives a different output
        // claim set. In the chain the committed claim binding then rejects it.
        let mut old_chain =
            Chain::begin(88, &Digest::ZERO, b"claim order", &[], None).unwrap();
        let mut old = PositionPscChallenges::new(&mut old_chain, tags, 0);
        let replayed = verify_general(
            &variables,
            &[],
            target,
            &Summand::Product,
            None,
            &proof,
            masks.hiding.len(),
            &mut old,
        )
        .unwrap();
        assert_ne!(proven, replayed);
    }

    #[test]
    fn absorbed_points_are_deterministic_and_tagged() {
        let mut first = Chain::begin(41, &Digest::ZERO, b"committed point", &[], None).unwrap();
        absorb_point(&mut first, 1, 38, 0, &[Fp3::ONE, Fp3::ZERO]).unwrap();
        let first_value = first.draw_e().unwrap();
        let mut second = Chain::begin(41, &Digest::ZERO, b"committed point", &[], None).unwrap();
        absorb_point(&mut second, 1, 38, 0, &[Fp3::ONE, Fp3::ZERO]).unwrap();
        assert_eq!(first_value, second.draw_e().unwrap());
        let mut other_tag = Chain::begin(41, &Digest::ZERO, b"committed point", &[], None).unwrap();
        absorb_point(&mut other_tag, 1, 39, 0, &[Fp3::ONE, Fp3::ZERO]).unwrap();
        assert_ne!(first_value, other_tag.draw_e().unwrap());
    }
}
