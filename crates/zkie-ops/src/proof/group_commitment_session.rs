//! Isolated fixture group commitment and opening demonstration.
//!
//! Each member oracle of the fixture group has its own runtime tree, root,
//! code and encoding tail. The ordered root message is absorbed into the session
//! chain before the BAT challenge, and the member openings are verified against
//! the received roots before the chain is handed to BAT. Every member keeps its
//! own oracle identity and opening positions; the ordered list of roots is not
//! called one group root, and the chain's transcript value after the root
//! message is a transcript value, not a group commitment root.
//!
//! The roots and plans are fixture values checked against the fixture's own
//! independently re-encoded expectation. Building the expected roots and the
//! prover witness from the same deterministic fixture values is construction
//! sharing, not a verifier commitment proof; the verifier's check is the
//! opening against the received root, plus the root/plan equality against the
//! fixture expectation. The roots are not SEND receipts, not G_u/asset roots,
//! and not a production trust anchor. The opening check verifies that the
//! supplied member opening leads to the received member root for the drawn
//! positions, not every row. A caller that replaces both the witness and the
//! fixture expectation has no independent trusted binding in this fixture; that
//! remains unclosed.
//!
//! The declaration round uses the existing seal kind as the fixture's root
//! message kind and the existing positions kind for the opening positions. It
//! does not claim the production seal semantics, a real SEND round, a hiding
//! guarantee or a complete group authentication. Unsupported or malformed
//! plans, duplicate oracle ids, mismatched roots and order violations fail
//! closed.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, ChainMessage, KIND_POSITIONS, KIND_SEAL};
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
use zkie_engine::pcs::request::read::unique_sorted;
use zkie_engine::pcs::request::runtime_tree::{
    short_position, verify_runtime_opening, RuntimePartRows, RuntimeTree, RuntimeTreeOpening,
    RuntimeTreePart, RuntimeTreePlan,
};

/// The fixture's expected declaration of one group member: its oracle identity,
/// codes, row counts, runtime tree and independently re-encoded root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupMemberPlan {
    pub oracle: u64,
    pub main_code: CodeParams,
    pub mask_code: CodeParams,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub tree: RuntimeTreePlan,
    pub root: Digest,
}

impl GroupMemberPlan {
    fn validate(&self) -> Result<()> {
        build_code_plan(self.main_code.log_n, self.main_code.ell, self.main_code.r)?;
        build_code_plan(self.mask_code.log_n, self.mask_code.ell, self.mask_code.r)?;
        if self.main_rows == 0 || self.mask_rows == 0 || self.root == Digest::ZERO {
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
        Ok(())
    }
}

/// The public root declaration of one member. It carries the member identity,
/// codes, rows, runtime tree and root; the session compares the complete entry
/// with its expected plan before absorbing the ordered message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupRootEntry {
    pub oracle: u64,
    pub main_code: CodeParams,
    pub mask_code: CodeParams,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub tree: RuntimeTreePlan,
    pub root: Digest,
}

impl GroupRootEntry {
    pub fn from_plan(plan: &GroupMemberPlan) -> Self {
        Self {
            oracle: plan.oracle,
            main_code: plan.main_code,
            mask_code: plan.mask_code,
            main_rows: plan.main_rows,
            mask_rows: plan.mask_rows,
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

fn fixture_value(tag: u64, oracle: u64, base: u64, index: usize) -> Fp3 {
    Fp3::from_u64(
        base.wrapping_add(tag.wrapping_mul(0x1000))
            .wrapping_add(oracle.wrapping_mul(0x200))
            .wrapping_add(index as u64),
    )
}

/// One committed fixture member oracle: its messages, encoding tails, salts,
/// codewords and runtime tree. The witness is public, deterministic and
/// provides no hiding; it is a test fixture, not a SEND receipt.
pub struct CommittedGroupMember {
    plan: GroupMemberPlan,
    main_messages: Vec<Vec<Fp3>>,
    main_tails: Vec<Vec<Fp3>>,
    mask_messages: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
    salts: Vec<Vec<u8>>,
    main_codewords: Vec<Vec<Fp3>>,
    mask_codewords: Vec<Vec<Fp3>>,
    tree: RuntimeTree,
}

impl CommittedGroupMember {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tag: u64,
        oracle: u64,
        tree_number: u64,
        leaf_class: u32,
        main_code: CodeParams,
        mask_code: CodeParams,
        main_rows: u64,
        mask_rows: u64,
    ) -> Result<Self> {
        build_code_plan(main_code.log_n, main_code.ell, main_code.r)?;
        build_code_plan(mask_code.log_n, mask_code.ell, mask_code.r)?;
        let tree = tree_plan(
            oracle, tree_number, leaf_class, main_code, mask_code, main_rows, mask_rows,
        )?;
        if main_rows == 0 || mask_rows == 0 {
            return Err(Error::InvalidConfig);
        }
        let main_rows = usize::try_from(main_rows).map_err(|_| Error::ResourceLimit)?;
        let mask_rows = usize::try_from(mask_rows).map_err(|_| Error::ResourceLimit)?;
        let main_columns =
            usize::try_from(main_code.ell).map_err(|_| Error::ResourceLimit)?;
        let mask_columns =
            usize::try_from(mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let main_messages = (0..main_rows)
            .map(|row| {
                (0..main_columns)
                    .map(|column| {
                        fixture_value(tag, oracle, 0x1100, row * main_columns + column)
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let main_tail_columns =
            usize::try_from(main_code.r).map_err(|_| Error::ResourceLimit)?;
        let main_tails = (0..main_rows)
            .map(|row| {
                (0..main_tail_columns)
                    .map(|column| fixture_value(tag, oracle, 0x2200, row * 0x10 + column))
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let mask_messages = (0..mask_rows)
            .map(|row| {
                (0..mask_columns)
                    .map(|column| {
                        fixture_value(tag, oracle, 0x3300, row * mask_columns + column)
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let mask_tail_columns =
            usize::try_from(mask_code.r).map_err(|_| Error::ResourceLimit)?;
        let mask_tails = (0..mask_rows)
            .map(|row| {
                (0..mask_tail_columns)
                    .map(|column| fixture_value(tag, oracle, 0x4400, row * 0x10 + column))
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let salts: Vec<Vec<u8>> = (0..tree.columns())
            .map(|position| {
                let byte = ((position as u8).wrapping_mul(0x1d))
                    ^ (tag as u8).wrapping_mul(3)
                    ^ (oracle as u8)
                    ^ 0x5a;
                vec![byte; tree.salt_bytes]
            })
            .collect();
        let main_codewords = main_messages
            .iter()
            .zip(&main_tails)
            .map(|(message, tail)| encode_row_fp3(&main_code, message, tail))
            .collect::<Result<Vec<_>>>()?;
        let mask_codewords = mask_messages
            .iter()
            .zip(&mask_tails)
            .map(|(message, tail)| encode_row_fp3(&mask_code, message, tail))
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
            oracle,
            main_code,
            mask_code,
            main_rows: main_rows as u64,
            mask_rows: mask_rows as u64,
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
    /// every main row's coefficients, then every mask row's coefficients at the
    /// short position, then the salt is handled by the opening. This is used by
    /// tests to check the opened member layout.
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
        // The widths are the plan's; the check is explicit so a malformed
        // fixture cannot silently open a short column.
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

/// The verifier-side group declaration session. It owns the chain until the
/// roots and openings are checked and then transfers the same chain to the BAT
/// stage. It is not a production SEND session and does not authenticate the
/// roots beyond the fixture's expected plans.
pub struct GroupCommitmentSession {
    chain: Chain,
    expected: Vec<GroupMemberPlan>,
    stage: Stage,
    roots: Vec<Digest>,
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
            if index > 0 && expected[index - 1].oracle >= plan.oracle {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(Self {
            chain,
            expected,
            stage: Stage::Fresh,
            roots: Vec::new(),
            positions: Vec::new(),
        })
    }

    /// Receives the ordered multi-oracle root message. Every entry must equal
    /// the fixture's expected plan and root; the complete message is then
    /// absorbed into the session chain before any BAT challenge.
    pub fn bind_roots(&mut self, message: &GroupRootMessage) -> Result<()> {
        if self.stage != Stage::Fresh {
            return Err(Error::InvalidSessionState);
        }
        if message.entries.len() != self.expected.len() {
            return Err(Error::ShapeMismatch);
        }
        for (entry, plan) in message.entries.iter().zip(&self.expected) {
            if entry.oracle != plan.oracle
                || entry.main_code != plan.main_code
                || entry.mask_code != plan.mask_code
                || entry.main_rows != plan.main_rows
                || entry.mask_rows != plan.mask_rows
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
            push_code(&mut out, entry.main_code);
            push_code(&mut out, entry.mask_code);
            out.push_u64(entry.main_rows);
            out.push_u64(entry.mask_rows);
            push_bytes(&mut out, &entry.tree.encode());
            push_bytes(&mut out, &entry.root.0);
        }
        let prefix = out.prefix(KIND_SEAL, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.roots = message.entries.iter().map(|entry| entry.root).collect();
        self.stage = Stage::RootsBound;
        Ok(())
    }

    /// Draws the verifier's opening positions for every member from the
    /// session chain, after the roots and before the BAT challenge. The member
    /// order and each member's own depth are preserved; repeated draws are
    /// deduplicated.
    pub fn draw_positions(&mut self, per_oracle: usize) -> Result<Vec<Vec<u64>>> {
        if self.stage != Stage::RootsBound {
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
    /// positions. This is the opening-layer rejection source: a changed row,
    /// tail, salt, path or a witness from another commitment fails here.
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
    /// a missing root/opening stage is rejected.
    pub fn into_chain(self) -> Result<Chain> {
        if self.stage != Stage::Opened {
            return Err(Error::InvalidSessionState);
        }
        Ok(self.chain)
    }

    pub fn roots(&self) -> &[Digest] {
        &self.roots
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

    fn group_members(tag: u64) -> Vec<CommittedGroupMember> {
        (0..3)
            .map(|oracle| {
                CommittedGroupMember::new(
                    tag,
                    oracle,
                    10 + oracle,
                    4,
                    code(),
                    code(),
                    2,
                    2,
                )
                .unwrap()
            })
            .collect()
    }

    fn plans(members: &[CommittedGroupMember]) -> Vec<GroupMemberPlan> {
        members.iter().map(|member| member.plan().clone()).collect()
    }

    fn bound_session(
        tag: u64,
        seed: &[u8],
    ) -> (GroupCommitmentSession, Vec<CommittedGroupMember>, Vec<Vec<u64>>) {
        let members = group_members(tag);
        let expected = plans(&members);
        let mut session =
            GroupCommitmentSession::new(chain(seed), expected).unwrap();
        session
            .bind_roots(&GroupRootMessage::from_plans(&plans(&members)))
            .unwrap();
        let positions = session.draw_positions(2).unwrap();
        (session, members, positions)
    }

    #[test]
    fn roots_are_bound_in_order_before_the_openings_and_the_chain_moves() {
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
        assert_eq!(session.roots(), &expected.iter().map(|p| p.root).collect::<Vec<_>>());
        let positions = session.draw_positions(2).unwrap();
        assert_eq!(session.chain_rounds(), 2);
        assert_eq!(positions.len(), 3);
        assert!(positions.iter().all(|list| !list.is_empty()));
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
        assert_eq!(rounds, 2);
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
    fn wrong_order_metadata_and_replaced_roots_are_rejected() {
        let members = group_members(43);
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

        let other = group_members(48);
        let mut replaced_witness = message.clone();
        replaced_witness.entries[0].root = other[0].plan().root;
        assert!(matches!(
            session.bind_roots(&replaced_witness),
            Err(Error::StatementMismatch)
        ));

        assert_eq!(session.chain_rounds(), 0);
        session.bind_roots(&message).unwrap();
        assert!(matches!(
            session.bind_roots(&message),
            Err(Error::InvalidSessionState)
        ));
    }

    #[test]
    fn changed_rows_tails_and_openings_fail_the_opening_check() {
        let (mut session, members, positions) = bound_session(44, b"opening checks");
        let openings = members
            .iter()
            .zip(&positions)
            .map(|(member, list)| member.open(list))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(session.verify_openings(&openings).is_ok());
        // A repeated verification after success is an order violation, so the
        // tamper checks below use fresh sessions.
        for tamper in 0..4 {
            let (mut session, members, positions) = bound_session(44, b"opening checks");
            let openings = members
                .iter()
                .zip(&positions)
                .map(|(member, list)| member.open(list))
                .collect::<Result<Vec<_>>>()
                .unwrap();
            let mut bad = openings.clone();
            match tamper {
                0 => bad[0].columns[0].symbols[0] = bad[0].columns[0].symbols[0] + Fp::ONE,
                1 => bad[1].columns[0].salt[0] ^= 1,
                2 => bad[2].given[0].0[0] ^= 1,
                _ => bad[0].positions = Vec::new(),
            }
            assert!(session.verify_openings(&bad).is_err());
            session.verify_openings(&openings).unwrap();
        }

        // A witness from another commitment at the same positions is rejected
        // by the opening check against the fixed roots.
        let (mut session, _members, positions) = bound_session(45, b"other witness");
        let other = group_members(46);
        let bad = other
            .iter()
            .zip(&positions)
            .map(|(member, list)| member.open(list))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(session.verify_openings(&bad).is_err());
    }

    #[test]
    fn order_and_plan_faults_fail_closed() {
        let members = group_members(47);
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

        let early = GroupCommitmentSession::new(chain(b"early"), expected.clone()).unwrap();
        assert!(early.into_chain().is_err());

        let mut session =
            GroupCommitmentSession::new(chain(b"stages"), expected.clone()).unwrap();
        assert!(session.draw_positions(1).is_err());
        assert!(session.verify_openings(&[]).is_err());
        session
            .bind_roots(&GroupRootMessage::from_plans(&expected))
            .unwrap();
        assert!(session.draw_positions(0).is_err());
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
