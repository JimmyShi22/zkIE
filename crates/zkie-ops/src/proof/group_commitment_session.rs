//! Isolated fixture group commitment, SEND sample registration and opening.
//!
//! Each member oracle of the fixture group has its own runtime tree, root,
//! code, message rows, encoding tail and reserved sample pad. The ordered root
//! message is absorbed into the session chain; the shared seed is drawn from
//! the same root round; the complete ordered sample answers are received and
//! absorbed; the sampled registry and sample-form claims are registered; only
//! then are the member opening positions drawn and checked and the chain handed
//! to BAT. Every member keeps its own oracle identity and opening positions.
//! The ordered list of roots is not called one group root, and the chain
//! transcript value is a transcript value, not a group commitment root.
//!
//! The roots and plans are fixture values checked against the fixture's own
//! independently re-encoded expectation. Building the expected roots and the
//! prover witness from the same deterministic fixture values is construction
//! sharing, not a verifier commitment proof; the verifier's checks are the
//! opening against the received root and the sample-form claim over the received
//! answer. The roots are not SEND receipts, not G_u/asset roots, and not a
//! production trust anchor. The sample pads are public deterministic fixture
//! values encoded into the member commitment; they are not hiding.
//!
//! The shared seed uses one seed per answer number and the largest sample cube
//! over the members; every member reads the prefix its own shape needs. The
//! fixture supports this through the existing shared-seed validation and sample
//! row builder. Unsupported plans, shapes, answer counts or seed lengths fail
//! closed. A caller that replaces both the witness and the fixture expectation
//! has no independent trusted binding here; that remains unclosed.
//!
//! The declaration round uses the existing seed kind for the ordered root
//! message (the seed is drawn from that round), the empty kind to close the
//! answer round, and the positions kind for the opening positions. This is not
//! a production SEND message, a hiding guarantee or a complete group
//! authentication.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{
    Chain, ChainMessage, KIND_EMPTY, KIND_POSITIONS, KIND_SEED,
};
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::gu::{GuNotice, GuOracleShape, GuRegistry, GuShape};
use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
use zkie_engine::pcs::request::read::unique_sorted;
use zkie_engine::pcs::request::relation::{
    gu_layout_fault, sample_cube, sample_row, validate_shared_sample_seeds, GuRelationRow,
};
use zkie_engine::pcs::request::runtime_tree::{
    short_position, verify_runtime_opening, RuntimePartRows, RuntimeTree, RuntimeTreeOpening,
    RuntimeTreePart, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::{
    IndexShape, SampleSeed, TableDistances,
};

/// The fixture's expected declaration of one group member: its oracle identity,
/// codes, row and answer counts, runtime tree and independently re-encoded root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupMemberPlan {
    pub oracle: u64,
    pub group: u64,
    pub main_code: CodeParams,
    pub mask_code: CodeParams,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub answers: u32,
    pub tree: RuntimeTreePlan,
    pub root: Digest,
}

impl GroupMemberPlan {
    fn validate(&self) -> Result<()> {
        build_code_plan(self.main_code.log_n, self.main_code.ell, self.main_code.r)?;
        build_code_plan(self.mask_code.log_n, self.mask_code.ell, self.mask_code.r)?;
        if self.main_rows == 0
            || self.mask_rows == 0
            || self.answers == 0
            || self.root == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        if self.tree.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let main_symbols = self
            .main_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        let mask_symbols = self
            .mask_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        if self.tree.parts.len() != 1
            || self.tree.parts[0]
                != (RuntimeTreePart {
                    main_symbols,
                    mask_symbols,
                })
            || self.tree.log_n != self.main_code.log_n
            || self.tree.log_n_zk != self.mask_code.log_n
        {
            return Err(Error::InvalidConfig);
        }
        if (self.mask_rows > 0 && self.mask_code.ell < self.answers as u64)
            || (self.mask_rows == 0 && self.main_code.ell < self.answers as u64)
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

/// The public root declaration of one member. It carries the member identity,
/// codes, rows, answers, runtime tree and root; the session compares the
/// complete entry with its expected plan before absorbing the ordered message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupRootEntry {
    pub oracle: u64,
    pub group: u64,
    pub main_code: CodeParams,
    pub mask_code: CodeParams,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub answers: u32,
    pub tree: RuntimeTreePlan,
    pub root: Digest,
}

impl GroupRootEntry {
    pub fn from_plan(plan: &GroupMemberPlan) -> Self {
        Self {
            oracle: plan.oracle,
            group: plan.group,
            main_code: plan.main_code,
            mask_code: plan.mask_code,
            main_rows: plan.main_rows,
            mask_rows: plan.mask_rows,
            answers: plan.answers,
            tree: plan.tree.clone(),
            root: plan.root,
        }
    }
}

/// The ordered multi-oracle root message. There is one entry per member and no
/// combined root field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupRootMessage {
    pub entries: Vec<GroupRootEntry>,
}

impl GroupRootMessage {
    pub fn from_plans(plans: &[GroupMemberPlan]) -> Self {
        Self {
            entries: plans.iter().map(GroupRootEntry::from_plan).collect(),
        }
    }
}

fn tree_plan(
    oracle: u64,
    tree_number: u64,
    leaf_class: u32,
    main_code: CodeParams,
    mask_code: CodeParams,
    main_rows: u64,
    mask_rows: u64,
) -> Result<RuntimeTreePlan> {
    let main_symbols = main_rows
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    let mask_symbols = mask_rows
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    let run_byte = (oracle as u8).wrapping_mul(17) ^ 0x61;
    let plan = RuntimeTreePlan {
        tree_number,
        leaf_class,
        log_n: main_code.log_n,
        log_n_zk: mask_code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols,
            mask_symbols,
        }],
        salt_bytes: LEAF_SALT_MIN_BYTES,
        run: Digest([run_byte; 32]),
        configuration: Digest([0x62; 32]),
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(plan)
}

fn fixture_index_shape() -> Result<IndexShape> {
    Ok(IndexShape {
        code: build_code_plan(3, 4, 1)?,
        v: 2,
        answers: 1,
        salt_bytes: LEAF_SALT_MIN_BYTES,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: 0,
            m_later: 0,
        },
    })
}

/// Builds the trusted G_u-shaped view of every member plan. The view supplies
/// the sample cube, pads and shared-seed semantics; it is fixture data, not a
/// production statement.
pub fn member_gu_shapes(plans: &[GroupMemberPlan]) -> Result<Vec<GuShape>> {
    if plans.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let index = fixture_index_shape()?;
    let group = plans[0].group;
    let mut shapes = Vec::with_capacity(plans.len());
    for plan in plans {
        plan.validate()?;
        if plan.group != group {
            return Err(Error::InvalidConfig);
        }
        let shape = GuShape {
            oracle: plan.oracle,
            group: plan.group,
            tree: plan.tree.clone(),
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: plan.main_code,
                main_rows: plan.main_rows,
                main_deg: E_DEGREE as u32,
                mask_rows: plan.mask_rows,
                mask_code: Some(plan.mask_code),
                answers: plan.answers,
            },
            index,
        };
        if shape.fault().is_some() || gu_layout_fault(&shape).is_some() {
            return Err(Error::InvalidConfig);
        }
        shapes.push(shape);
    }
    Ok(shapes)
}

/// The shared answer order: by answer number, then member order. Each member
/// contributes only its own answers; the shared seed set has one seed per
/// answer number up to the largest answer count.
pub fn shared_answer_order(shapes: &[GuShape]) -> Result<Vec<(u32, u64)>> {
    if shapes.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut max_answers = 0u32;
    for shape in shapes {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        max_answers = max_answers.max(shape.u.answers);
    }
    let mut order = Vec::new();
    for answer in 0..max_answers {
        for shape in shapes {
            if answer < shape.u.answers {
                order.push((answer, shape.oracle));
            }
        }
    }
    if order.is_empty() {
        return Err(Error::InvalidConfig);
    }
    Ok(order)
}

fn fixture_value(tag: u64, oracle: u64, base: u64, index: usize) -> Fp3 {
    Fp3::from_u64(
        base.wrapping_add(tag.wrapping_mul(0x1000))
            .wrapping_add(oracle.wrapping_mul(0x200))
            .wrapping_add(index as u64),
    )
}

/// One committed fixture member oracle: its messages, encoding tails, salts,
/// codewords, reserved sample pads and runtime tree. The witness is public,
/// deterministic and provides no hiding; it is a test fixture, not a SEND
/// receipt.
pub struct CommittedGroupMember {
    plan: GroupMemberPlan,
    main_messages: Vec<Vec<Fp3>>,
    main_tails: Vec<Vec<Fp3>>,
    mask_messages: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
    salts: Vec<Vec<u8>>,
    main_codewords: Vec<Vec<Fp3>>,
    mask_codewords: Vec<Vec<Fp3>>,
    /// Pad slots: (is_mask, row, place, value).
    pads: Vec<(bool, usize, usize, Fp3)>,
    tree: RuntimeTree,
}

/// The shape of one fixture member before its messages are built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemberFixture {
    pub tag: u64,
    pub oracle: u64,
    pub group: u64,
    pub tree_number: u64,
    pub leaf_class: u32,
    pub main_code: CodeParams,
    pub mask_code: CodeParams,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub answers: u32,
    pub pad_shift: Fp3,
}

impl CommittedGroupMember {
    pub fn build(config: MemberFixture) -> Result<Self> {
        build_code_plan(
            config.main_code.log_n,
            config.main_code.ell,
            config.main_code.r,
        )?;
        build_code_plan(
            config.mask_code.log_n,
            config.mask_code.ell,
            config.mask_code.r,
        )?;
        if config.main_rows == 0 || config.mask_rows == 0 || config.answers == 0 {
            return Err(Error::InvalidConfig);
        }
        let tree = tree_plan(
            config.oracle,
            config.tree_number,
            config.leaf_class,
            config.main_code,
            config.mask_code,
            config.main_rows,
            config.mask_rows,
        )?;
        let main_rows = usize::try_from(config.main_rows).map_err(|_| Error::ResourceLimit)?;
        let mask_rows = usize::try_from(config.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let main_columns =
            usize::try_from(config.main_code.ell).map_err(|_| Error::ResourceLimit)?;
        let mask_columns =
            usize::try_from(config.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let mut main_messages = (0..main_rows)
            .map(|row| {
                (0..main_columns)
                    .map(|column| {
                        fixture_value(
                            config.tag,
                            config.oracle,
                            0x1100,
                            row * main_columns + column,
                        )
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let main_tail_columns =
            usize::try_from(config.main_code.r).map_err(|_| Error::ResourceLimit)?;
        let main_tails = (0..main_rows)
            .map(|row| {
                (0..main_tail_columns)
                    .map(|column| {
                        fixture_value(config.tag, config.oracle, 0x2200, row * 0x10 + column)
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let mut mask_messages = (0..mask_rows)
            .map(|row| {
                (0..mask_columns)
                    .map(|column| {
                        fixture_value(
                            config.tag,
                            config.oracle,
                            0x3300,
                            row * mask_columns + column,
                        )
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let mask_tail_columns =
            usize::try_from(config.mask_code.r).map_err(|_| Error::ResourceLimit)?;
        let mask_tails = (0..mask_rows)
            .map(|row| {
                (0..mask_tail_columns)
                    .map(|column| {
                        fixture_value(config.tag, config.oracle, 0x4400, row * 0x10 + column)
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        if config.mask_rows > 0 && config.mask_code.ell < config.answers as u64 {
            return Err(Error::InvalidConfig);
        }
        // Reserve one pad per answer before the commitment. The pad value is
        // part of the encoded message; the seed only supplies its coefficient.
        let mut pads = Vec::with_capacity(config.answers as usize);
        for answer in 0..config.answers {
            let value = Fp3::from_u64(
                0x9000u64
                    .wrapping_add(config.tag.wrapping_mul(0x100))
                    .wrapping_add(config.oracle.wrapping_mul(0x10))
                    .wrapping_add(answer as u64),
            )
            .add(&config.pad_shift);
            if config.mask_rows > 0 {
                let place = usize::try_from(
                    config.mask_code.ell - config.answers as u64 + answer as u64,
                )
                .map_err(|_| Error::ResourceLimit)?;
                let row = mask_rows - 1;
                let slot = mask_messages
                    .get_mut(row)
                    .and_then(|entries| entries.get_mut(place))
                    .ok_or(Error::ShapeMismatch)?;
                *slot = value;
                pads.push((true, row, place, value));
            } else {
                let place = usize::try_from(
                    config.main_code.ell - config.answers as u64 + answer as u64,
                )
                .map_err(|_| Error::ResourceLimit)?;
                let row = main_rows - 1;
                let slot = main_messages
                    .get_mut(row)
                    .and_then(|entries| entries.get_mut(place))
                    .ok_or(Error::ShapeMismatch)?;
                *slot = value;
                pads.push((false, row, place, value));
            }
        }
        let salts: Vec<Vec<u8>> = (0..tree.columns())
            .map(|position| {
                let byte = ((position as u8).wrapping_mul(0x1d))
                    ^ (config.tag as u8).wrapping_mul(3)
                    ^ (config.oracle as u8)
                    ^ 0x5a;
                vec![byte; tree.salt_bytes]
            })
            .collect();
        let main_codewords = main_messages
            .iter()
            .zip(&main_tails)
            .map(|(message, tail)| encode_row_fp3(&config.main_code, message, tail))
            .collect::<Result<Vec<_>>>()?;
        let mask_codewords = mask_messages
            .iter()
            .zip(&mask_tails)
            .map(|(message, tail)| encode_row_fp3(&config.mask_code, message, tail))
            .collect::<Result<Vec<_>>>()?;
        let committed = RuntimeTree::commit(
            tree,
            &[RuntimePartRows {
                main: main_codewords.clone(),
                mask: mask_codewords.clone(),
                main_deg: E_DEGREE as u32,
            }],
            &salts,
        )?;
        let plan = GroupMemberPlan {
            oracle: config.oracle,
            group: config.group,
            main_code: config.main_code,
            mask_code: config.mask_code,
            main_rows: config.main_rows,
            mask_rows: config.mask_rows,
            answers: config.answers,
            tree: committed.plan().clone(),
            root: *committed.root(),
        };
        plan.validate()?;
        Ok(Self {
            plan,
            main_messages,
            main_tails,
            mask_messages,
            mask_tails,
            salts,
            main_codewords,
            mask_codewords,
            pads,
            tree: committed,
        })
    }

    pub fn plan(&self) -> &GroupMemberPlan {
        &self.plan
    }

    pub fn main_messages(&self) -> &[Vec<Fp3>] {
        &self.main_messages
    }

    pub fn mask_messages(&self) -> &[Vec<Fp3>] {
        &self.mask_messages
    }

    /// The committed main encoding tails, one row per main message row.
    pub fn main_tails(&self) -> &[Vec<Fp3>] {
        &self.main_tails
    }

    /// The committed mask encoding tails, one row per mask message row.
    pub fn mask_tails(&self) -> &[Vec<Fp3>] {
        &self.mask_tails
    }

    pub fn pad_slots(&self) -> &[(bool, usize, usize, Fp3)] {
        &self.pads
    }

    pub fn gu_witness(&self) -> zkie_engine::pcs::request::relation::GuWitness {
        let mut main = Vec::new();
        for row in &self.main_messages {
            main.extend_from_slice(row);
        }
        let mut mask = Vec::new();
        for row in &self.mask_messages {
            mask.extend_from_slice(row);
        }
        zkie_engine::pcs::request::relation::GuWitness { main, mask }
    }

    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.tree.open(positions)
    }

    /// Re-encodes every stored message and tail and re-commits the member tree.
    /// This is the fixture's independent encoding-to-root check.
    pub fn recompute_root(&self) -> Result<Digest> {
        let main_codewords = self
            .main_messages
            .iter()
            .zip(&self.main_tails)
            .map(|(message, tail)| encode_row_fp3(&self.plan.main_code, message, tail))
            .collect::<Result<Vec<_>>>()?;
        let mask_codewords = self
            .mask_messages
            .iter()
            .zip(&self.mask_tails)
            .map(|(message, tail)| encode_row_fp3(&self.plan.mask_code, message, tail))
            .collect::<Result<Vec<_>>>()?;
        let tree = RuntimeTree::commit(
            self.plan.tree.clone(),
            &[RuntimePartRows {
                main: main_codewords,
                mask: mask_codewords,
                main_deg: E_DEGREE as u32,
            }],
            &self.salts,
        )?;
        Ok(*tree.root())
    }

    /// The leaf symbols at one long position, in the runtime tree's order:
    /// every main row's codeword, then every mask row's codeword at the short
    /// position. The salt is handled by the opening.
    pub fn column_symbols(&self, position: u64) -> Result<Vec<Fp>> {
        let columns = self.plan.tree.columns();
        if position >= columns {
            return Err(Error::InvalidIndex);
        }
        let short = short_position(
            self.plan.tree.log_n,
            self.plan.tree.log_n_zk,
            position,
        )?;
        let mut symbols = Vec::new();
        for codeword in &self.main_codewords {
            let value = codeword
                .get(position as usize)
                .ok_or(Error::ShapeMismatch)?;
            symbols.extend(value.c.iter().copied());
        }
        for codeword in &self.mask_codewords {
            let value = codeword
                .get(short as usize)
                .ok_or(Error::ShapeMismatch)?;
            symbols.extend(value.c.iter().copied());
        }
        if symbols.len() as u64 != self.plan.tree.width()? {
            return Err(Error::ShapeMismatch);
        }
        Ok(symbols)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    RootsBound,
    SeedsDrawn,
    AnswersCollecting,
    AnswersAbsorbed,
    PositionsDrawn,
    Opened,
}

fn push_bytes(message: &mut ChainMessage, bytes: &[u8]) {
    message.push_u64(bytes.len() as u64);
    for chunk in bytes.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        message.push_u64(u64::from_le_bytes(word));
    }
}

fn push_code(message: &mut ChainMessage, code: CodeParams) {
    message.push_u64(code.log_n as u64);
    message.push_u64(code.ell);
    message.push_u64(code.r);
}

/// The verifier-side group declaration and SEND session. It owns the chain
/// through the roots, shared seed, complete ordered answers, sampled
/// registration, opening positions and member openings, and then transfers the
/// same chain to the BAT stage. It is not a production SEND session.
pub struct GroupCommitmentSession {
    chain: Chain,
    expected: Vec<GroupMemberPlan>,
    shapes: Vec<GuShape>,
    stage: Stage,
    roots: Vec<Digest>,
    seeds: Vec<SampleSeed>,
    answer_order: Vec<(u32, u64)>,
    received: Vec<Fp3>,
    sample_rows: Vec<GuRelationRow>,
    positions: Vec<Vec<u64>>,
}

impl GroupCommitmentSession {
    pub fn new(chain: Chain, expected: Vec<GroupMemberPlan>) -> Result<Self> {
        if expected.is_empty() || chain.rounds() != 0 {
            return Err(Error::InvalidSessionState);
        }
        for (index, plan) in expected.iter().enumerate() {
            plan.validate()?;
            if expected[..index]
                .iter()
                .any(|other| other.oracle == plan.oracle)
            {
                return Err(Error::InvalidConfig);
            }
            if index > 0
                && (expected[index - 1].oracle >= plan.oracle
                    || expected[index - 1].group != plan.group)
            {
                return Err(Error::InvalidConfig);
            }
        }
        let shapes = member_gu_shapes(&expected)?;
        let answer_order = shared_answer_order(&shapes)?;
        Ok(Self {
            chain,
            expected,
            shapes,
            stage: Stage::Fresh,
            roots: Vec::new(),
            seeds: Vec::new(),
            answer_order,
            received: Vec::new(),
            sample_rows: Vec::new(),
            positions: Vec::new(),
        })
    }

    /// Receives the ordered multi-oracle root message. Every entry must equal
    /// the fixture's expected plan and root; the complete message is then
    /// absorbed into the session chain as the SEND root/seed round.
    pub fn bind_roots(&mut self, message: &GroupRootMessage) -> Result<()> {
        if self.stage != Stage::Fresh {
            return Err(Error::InvalidSessionState);
        }
        if message.entries.len() != self.expected.len() {
            return Err(Error::ShapeMismatch);
        }
        for (entry, plan) in message.entries.iter().zip(&self.expected) {
            if entry.oracle != plan.oracle
                || entry.group != plan.group
                || entry.main_code != plan.main_code
                || entry.mask_code != plan.mask_code
                || entry.main_rows != plan.main_rows
                || entry.mask_rows != plan.mask_rows
                || entry.answers != plan.answers
                || entry.tree != plan.tree
                || entry.root != plan.root
            {
                return Err(Error::StatementMismatch);
            }
        }
        let mut out = self.chain.message();
        out.push_u64(message.entries.len() as u64);
        for entry in &message.entries {
            out.push_u64(entry.oracle);
            out.push_u64(entry.group);
            push_code(&mut out, entry.main_code);
            push_code(&mut out, entry.mask_code);
            out.push_u64(entry.main_rows);
            out.push_u64(entry.mask_rows);
            out.push_u64(entry.answers as u64);
            push_bytes(&mut out, &entry.tree.encode());
            push_bytes(&mut out, &entry.root.0);
        }
        let prefix = out.prefix(KIND_SEED, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.roots = message.entries.iter().map(|entry| entry.root).collect();
        self.stage = Stage::RootsBound;
        Ok(())
    }

    /// Draws the shared seed after the fixed roots. One seed per answer number
    /// up to the largest answer count; each seed carries the largest sample
    /// cube's coordinates. Every member reads its own prefix.
    pub fn draw_shared_seeds(&mut self) -> Result<()> {
        if self.stage != Stage::RootsBound {
            return Err(Error::InvalidSessionState);
        }
        let refs: Vec<&GuShape> = self.shapes.iter().collect();
        let mut bits = 0u32;
        let mut answers = 0u32;
        for shape in &refs {
            let cube = sample_cube(shape)?;
            bits = bits.max(cube.bits);
            answers = answers.max(shape.u.answers);
        }
        if bits == 0 {
            return Err(Error::InvalidConfig);
        }
        let mut seeds = Vec::with_capacity(answers as usize);
        for _ in 0..answers {
            let mut z = Vec::with_capacity(bits as usize);
            for _ in 0..bits {
                z.push(self.chain.draw_e()?);
            }
            let beta = self.chain.draw_e_nonzero()?;
            seeds.push(SampleSeed { z, beta });
        }
        validate_shared_sample_seeds(&refs, &seeds)?;
        self.seeds = seeds;
        self.stage = Stage::SeedsDrawn;
        Ok(())
    }

    /// Receives one answer in the fixed shared order: by answer number, then
    /// member order. The last answer closes the answer round. The registry and
    /// sample-form claims are then built from the bound roots, the drawn seeds
    /// and the received answers; the prover's own evaluation is not compared.
    pub fn receive_answer(&mut self, oracle: u64, target: Fp3) -> Result<()> {
        if self.stage != Stage::SeedsDrawn && self.stage != Stage::AnswersCollecting {
            return Err(Error::InvalidSessionState);
        }
        let next = self
            .answer_order
            .get(self.received.len())
            .copied()
            .ok_or(Error::InvalidSessionState)?;
        if next.1 != oracle {
            return Err(Error::StatementMismatch);
        }
        self.received.push(target);
        if self.received.len() == self.answer_order.len() {
            let mut out = self.chain.message();
            for answer in &self.received {
                out.push_fp3(answer);
            }
            let prefix = out.prefix(KIND_EMPTY, 0)?;
            if !self.chain.step(&prefix, 0)? {
                return Err(Error::InvalidProof);
            }
            let mut registry = GuRegistry::new();
            for (shape, root) in self.shapes.iter().zip(&self.roots) {
                registry.receive(
                    shape,
                    &GuNotice {
                        root: *root,
                        oracle: shape.oracle,
                        tree_number: shape.tree.tree_number,
                        part: shape.part,
                    },
                )?;
            }
            let mut rows = Vec::with_capacity(self.answer_order.len());
            for (index, (answer, oracle)) in self.answer_order.iter().copied().enumerate() {
                let shape = self
                    .shapes
                    .iter()
                    .find(|shape| shape.oracle == oracle)
                    .ok_or(Error::StatementMismatch)?;
                rows.push(sample_row(
                    shape,
                    &registry,
                    &self.seeds,
                    answer,
                    self.received[index],
                )?);
            }
            self.sample_rows = rows;
            self.stage = Stage::AnswersAbsorbed;
        } else {
            self.stage = Stage::AnswersCollecting;
        }
        Ok(())
    }

    /// Draws the member opening positions after the complete answers. This is
    /// the "later positions" phase; it never runs between the seed and the
    /// answers.
    pub fn draw_positions(&mut self, per_oracle: usize) -> Result<Vec<Vec<u64>>> {
        if self.stage != Stage::AnswersAbsorbed {
            return Err(Error::InvalidSessionState);
        }
        if per_oracle == 0 {
            return Err(Error::InvalidConfig);
        }
        let mut out = self.chain.message();
        out.push_u64(self.expected.len() as u64);
        out.push_u64(per_oracle as u64);
        let prefix = out.prefix(KIND_POSITIONS, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let mut positions = Vec::with_capacity(self.expected.len());
        for plan in &self.expected {
            let mut drawn = Vec::with_capacity(per_oracle);
            for _ in 0..per_oracle {
                drawn.push(self.chain.draw_position(plan.tree.log_n)?);
            }
            positions.push(unique_sorted(&drawn));
        }
        self.positions = positions.clone();
        self.stage = Stage::PositionsDrawn;
        Ok(positions)
    }

    /// Verifies every member opening against its received root at the drawn
    /// positions. A changed row, tail, salt, path or a witness from another
    /// commitment fails here.
    pub fn verify_openings(&mut self, openings: &[RuntimeTreeOpening]) -> Result<()> {
        if self.stage != Stage::PositionsDrawn {
            return Err(Error::InvalidSessionState);
        }
        if openings.len() != self.expected.len() {
            return Err(Error::ShapeMismatch);
        }
        for ((plan, positions), opening) in
            self.expected.iter().zip(&self.positions).zip(openings)
        {
            if opening.positions != *positions {
                return Err(Error::InvalidIndex);
            }
            verify_runtime_opening(&plan.tree, &plan.root, positions, opening)?;
        }
        self.stage = Stage::Opened;
        Ok(())
    }

    /// Transfers the advanced chain to the BAT stage. The session is consumed;
    /// a missing root/seed/answer/opening stage is rejected.
    pub fn into_chain(self) -> Result<Chain> {
        if self.stage != Stage::Opened {
            return Err(Error::InvalidSessionState);
        }
        Ok(self.chain)
    }

    pub fn roots(&self) -> &[Digest] {
        &self.roots
    }

    pub fn seeds(&self) -> &[SampleSeed] {
        &self.seeds
    }

    pub fn answer_order(&self) -> &[(u32, u64)] {
        &self.answer_order
    }

    pub fn sample_rows(&self) -> &[GuRelationRow] {
        &self.sample_rows
    }

    pub fn positions(&self) -> &[Vec<u64>] {
        &self.positions
    }

    pub fn chain_value(&self) -> &Digest {
        self.chain.value()
    }

    pub fn chain_rounds(&self) -> u64 {
        self.chain.rounds()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::PrimeCharacteristicRing;
    use zkie_engine::pcs::request::chain::Chain;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

    fn chain(seed: &[u8]) -> Chain {
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest::ZERO, seed, &[], None).unwrap()
    }

    fn code() -> CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn config(tag: u64, oracle: u64, pad_shift: Fp3) -> MemberFixture {
        MemberFixture {
            tag,
            oracle,
            group: 0,
            tree_number: 10 + oracle,
            leaf_class: 4,
            main_code: code(),
            mask_code: code(),
            main_rows: 2,
            mask_rows: 2,
            answers: 1,
            pad_shift,
        }
    }

    fn group_members_with_pad(tag: u64, pad_shift: Fp3) -> Vec<CommittedGroupMember> {
        (0..3)
            .map(|oracle| CommittedGroupMember::build(config(tag, oracle, pad_shift)).unwrap())
            .collect()
    }

    fn group_members(tag: u64) -> Vec<CommittedGroupMember> {
        group_members_with_pad(tag, Fp3::ZERO)
    }

    fn plans(members: &[CommittedGroupMember]) -> Vec<GroupMemberPlan> {
        members.iter().map(|member| member.plan().clone()).collect()
    }

    fn send_answers(session: &mut GroupCommitmentSession) -> Result<()> {
        session.draw_shared_seeds()?;
        let order = session.answer_order().to_vec();
        for (_, oracle) in order {
            session.receive_answer(oracle, Fp3::from_u64(0x700 + oracle))?;
        }
        Ok(())
    }

    fn bound_session(
        tag: u64,
        seed: &[u8],
    ) -> (GroupCommitmentSession, Vec<CommittedGroupMember>, Vec<Vec<u64>>) {
        let members = group_members(tag);
        let expected = plans(&members);
        let mut session = GroupCommitmentSession::new(chain(seed), expected.clone()).unwrap();
        session
            .bind_roots(&GroupRootMessage::from_plans(&expected))
            .unwrap();
        send_answers(&mut session).unwrap();
        let positions = session.draw_positions(2).unwrap();
        (session, members, positions)
    }

    #[test]
    fn roots_seeds_answers_and_openings_follow_the_chain_order() {
        let members = group_members(41);
        let expected = plans(&members);
        let mut session =
            GroupCommitmentSession::new(chain(b"group binding"), expected.clone()).unwrap();
        assert_eq!(session.chain_rounds(), 0);
        session
            .bind_roots(&GroupRootMessage::from_plans(&expected))
            .unwrap();
        assert_eq!(session.chain_rounds(), 1);
        assert_eq!(session.roots().len(), 3);
        assert_eq!(
            session.roots(),
            &expected.iter().map(|plan| plan.root).collect::<Vec<_>>()
        );
        session.draw_shared_seeds().unwrap();
        assert_eq!(session.chain_rounds(), 1);
        assert_eq!(session.answer_order().len(), 3);
        assert_eq!(session.seeds().len(), 1);
        let order = session.answer_order().to_vec();
        for (_, oracle) in order {
            session
                .receive_answer(oracle, Fp3::from_u64(0x700 + oracle))
                .unwrap();
        }
        assert_eq!(session.chain_rounds(), 2);
        assert_eq!(session.sample_rows().len(), 3);
        let positions = session.draw_positions(2).unwrap();
        assert_eq!(session.chain_rounds(), 3);
        assert_eq!(positions.len(), 3);
        let openings = members
            .iter()
            .zip(&positions)
            .map(|(member, list)| member.open(list))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        session.verify_openings(&openings).unwrap();
        let rounds = session.chain_rounds();
        let value = *session.chain_value();
        let chain = session.into_chain().unwrap();
        assert_eq!(chain.rounds(), rounds);
        assert_eq!(*chain.value(), value);
        assert_eq!(rounds, 3);
    }

    #[test]
    fn independent_reencoding_matches_every_member_root_and_layout() {
        let members = group_members(42);
        for member in &members {
            assert_eq!(member.recompute_root().unwrap(), member.plan().root);
        }
        let (mut session, members, positions) = bound_session(42, b"reencoding");
        assert_eq!(session.positions(), positions.as_slice());
        let openings = members
            .iter()
            .zip(&positions)
            .map(|(member, list)| member.open(list))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        for ((member, opening), list) in members.iter().zip(&openings).zip(&positions) {
            for (at, position) in list.iter().enumerate() {
                assert_eq!(
                    opening.columns[at].symbols,
                    member.column_symbols(*position).unwrap()
                );
            }
        }
        session.verify_openings(&openings).unwrap();
    }

    #[test]
    fn send_order_rejects_early_positions_wrong_oracle_and_extra_answers() {
        let members = group_members(43);
        let expected = plans(&members);
        let mut session =
            GroupCommitmentSession::new(chain(b"send order"), expected.clone()).unwrap();
        assert!(session.draw_shared_seeds().is_err());
        assert!(session
            .receive_answer(0, Fp3::ONE)
            .is_err());
        session
            .bind_roots(&GroupRootMessage::from_plans(&expected))
            .unwrap();
        session.draw_shared_seeds().unwrap();
        assert!(session.draw_positions(1).is_err());
        assert!(session
            .receive_answer(1, Fp3::ONE)
            .is_err());
        session
            .receive_answer(0, Fp3::from_u64(1))
            .unwrap();
        assert!(session.receive_answer(0, Fp3::from_u64(2)).is_err());
        session
            .receive_answer(1, Fp3::from_u64(3))
            .unwrap();
        assert!(session.draw_positions(1).is_err());
        session
            .receive_answer(2, Fp3::from_u64(4))
            .unwrap();
        assert!(session
            .receive_answer(2, Fp3::from_u64(5))
            .is_err());
        assert!(session.draw_positions(1).is_ok());
    }

    #[test]
    fn wrong_order_metadata_and_replaced_roots_are_rejected() {
        let members = group_members(44);
        let expected = plans(&members);
        let message = GroupRootMessage::from_plans(&expected);
        let mut session =
            GroupCommitmentSession::new(chain(b"root checks"), expected.clone()).unwrap();

        let mut swapped = message.clone();
        swapped.entries.swap(0, 1);
        assert!(matches!(
            session.bind_roots(&swapped),
            Err(Error::StatementMismatch)
        ));

        let mut changed_tree = message.clone();
        changed_tree.entries[0].tree.tree_number += 1;
        assert!(matches!(
            session.bind_roots(&changed_tree),
            Err(Error::StatementMismatch)
        ));

        let mut changed_rows = message.clone();
        changed_rows.entries[1].main_rows += 1;
        assert!(matches!(
            session.bind_roots(&changed_rows),
            Err(Error::StatementMismatch)
        ));

        let mut changed_oracle = message.clone();
        changed_oracle.entries[1].oracle += 7;
        assert!(matches!(
            session.bind_roots(&changed_oracle),
            Err(Error::StatementMismatch)
        ));

        let mut changed_code = message.clone();
        changed_code.entries[0].main_code = build_code_plan(4, 4, 1).unwrap();
        assert!(matches!(
            session.bind_roots(&changed_code),
            Err(Error::StatementMismatch)
        ));

        let mut changed_root = message.clone();
        changed_root.entries[2].root.0[0] ^= 1;
        assert!(matches!(
            session.bind_roots(&changed_root),
            Err(Error::StatementMismatch)
        ));

        let other = group_members(45);
        let mut replaced_witness = message.clone();
        replaced_witness.entries[0].root = other[0].plan().root;
        assert!(matches!(
            session.bind_roots(&replaced_witness),
            Err(Error::StatementMismatch)
        ));

        assert_eq!(session.chain_rounds(), 0);
        session.bind_roots(&message).unwrap();
        session.draw_shared_seeds().unwrap();
        assert!(matches!(
            session.bind_roots(&message),
            Err(Error::InvalidSessionState)
        ));
    }

    #[test]
    fn changed_rows_tails_and_openings_fail_the_opening_check() {
        for tamper in 0..4 {
            let (mut session, members, positions) = bound_session(46, b"opening checks");
            let openings = members
                .iter()
                .zip(&positions)
                .map(|(member, list)| member.open(list))
                .collect::<Result<Vec<_>>>()
                .unwrap();
            let mut bad = openings.clone();
            match tamper {
                0 => {
                    bad[0].columns[0].symbols[0] =
                        bad[0].columns[0].symbols[0] + Fp::ONE
                }
                1 => bad[1].columns[0].salt[0] ^= 1,
                2 => bad[2].given[0].0[0] ^= 1,
                _ => bad[0].positions = Vec::new(),
            }
            assert!(session.verify_openings(&bad).is_err());
            session.verify_openings(&openings).unwrap();
        }

        let (mut session, _members, positions) = bound_session(47, b"other witness");
        let other = group_members(48);
        let bad = other
            .iter()
            .zip(&positions)
            .map(|(member, list)| member.open(list))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(session.verify_openings(&bad).is_err());
    }

    #[test]
    fn pad_changes_root_and_answer_under_a_fixed_seed() {
        let first = group_members_with_pad(49, Fp3::ZERO);
        let second = group_members_with_pad(49, Fp3::ONE);
        let first_plans = plans(&first);
        let second_plans = plans(&second);
        assert_ne!(first_plans[0].root, second_plans[0].root);
        let shapes = member_gu_shapes(&first_plans).unwrap();
        let bits = shapes[0].sample_bits().unwrap() as usize;
        let seeds = vec![SampleSeed {
            z: (0..bits).map(|index| Fp3::from_u64(3 + index as u64)).collect(),
            beta: Fp3::ONE,
        }];
        let mut first_registry = GuRegistry::new();
        let mut second_registry = GuRegistry::new();
        for (shape, plan) in shapes.iter().zip(&first_plans) {
            first_registry
                .receive(
                    shape,
                    &GuNotice {
                        root: plan.root,
                        oracle: shape.oracle,
                        tree_number: shape.tree.tree_number,
                        part: shape.part,
                    },
                )
                .unwrap();
        }
        for (shape, plan) in shapes.iter().zip(&second_plans) {
            second_registry
                .receive(
                    shape,
                    &GuNotice {
                        root: plan.root,
                        oracle: shape.oracle,
                        tree_number: shape.tree.tree_number,
                        part: shape.part,
                    },
                )
                .unwrap();
        }
        let order = shared_answer_order(&shapes).unwrap();
        let mut first_answers = Vec::new();
        let mut second_answers = Vec::new();
        for (answer, oracle) in order {
            let shape = shapes
                .iter()
                .find(|shape| shape.oracle == oracle)
                .unwrap();
            let first_member = first
                .iter()
                .find(|member| member.plan().oracle == oracle)
                .unwrap();
            let second_member = second
                .iter()
                .find(|member| member.plan().oracle == oracle)
                .unwrap();
            first_answers.push(
                sample_row(shape, &first_registry, &seeds, answer, Fp3::ZERO)
                    .unwrap()
                    .evaluate(&first_member.gu_witness())
                    .unwrap(),
            );
            second_answers.push(
                sample_row(shape, &second_registry, &seeds, answer, Fp3::ZERO)
                    .unwrap()
                    .evaluate(&second_member.gu_witness())
                    .unwrap(),
            );
        }
        assert_ne!(first_answers, second_answers);
    }

    #[test]
    fn order_and_plan_faults_fail_closed() {
        let members = group_members(50);
        let expected = plans(&members);
        let mut rounds_session =
            GroupCommitmentSession::new(chain(b"rounds"), expected.clone()).unwrap();
        rounds_session
            .bind_roots(&GroupRootMessage::from_plans(&expected))
            .unwrap();

        let mut duplicate = expected.clone();
        duplicate[1].oracle = duplicate[0].oracle;
        assert!(GroupCommitmentSession::new(chain(b"duplicate"), duplicate).is_err());
        let mut unsorted = expected.clone();
        unsorted.swap(0, 1);
        assert!(GroupCommitmentSession::new(chain(b"unsorted"), unsorted).is_err());
        let mut zero_root = expected.clone();
        zero_root[0].root = Digest::ZERO;
        assert!(GroupCommitmentSession::new(chain(b"zero"), zero_root).is_err());
        let mut wrong_widths = expected.clone();
        wrong_widths[0].tree.parts[0].main_symbols += 1;
        assert!(GroupCommitmentSession::new(chain(b"widths"), wrong_widths).is_err());
        let mut wrong_answers = expected.clone();
        wrong_answers[0].answers = 0;
        assert!(GroupCommitmentSession::new(chain(b"answers"), wrong_answers).is_err());
        let mut wrong_group = expected.clone();
        wrong_group[2].group += 1;
        assert!(GroupCommitmentSession::new(chain(b"group"), wrong_group).is_err());

        let early = GroupCommitmentSession::new(chain(b"early"), expected.clone()).unwrap();
        assert!(early.into_chain().is_err());

        let mut session =
            GroupCommitmentSession::new(chain(b"stages"), expected.clone()).unwrap();
        assert!(session.draw_shared_seeds().is_err());
        assert!(session.draw_positions(1).is_err());
        assert!(session.verify_openings(&[]).is_err());
        session
            .bind_roots(&GroupRootMessage::from_plans(&expected))
            .unwrap();
        session.draw_shared_seeds().unwrap();
        let order = session.answer_order().to_vec();
        for (_, oracle) in order {
            session
                .receive_answer(oracle, Fp3::from_u64(1))
                .unwrap();
        }
        let positions = session.draw_positions(1).unwrap();
        assert!(session.draw_positions(1).is_err());
        let openings = members
            .iter()
            .zip(&positions)
            .map(|(member, list)| member.open(list))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        session.verify_openings(&openings).unwrap();
        assert!(session.verify_openings(&openings).is_err());
        assert!(session.into_chain().is_ok());
    }
}
