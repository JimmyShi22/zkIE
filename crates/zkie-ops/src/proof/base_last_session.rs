//! Last-stage BASE of a proximity test.
//!
//! The input is the final switched oracle's GroupRow: one main row and one mask
//! row of one oracle message, with the linear claim left pending by the switch.
//! The verifier authenticates the opened symbols against the switched oracle's
//! declared root and against an independent masking oracle that was committed
//! before lambda; it then checks the linear relation on the masked message and
//! the encoding symbols at the drawn positions. No FOLD handoff, no extra fresh
//! row and no second commitment of the same rows are introduced: the masking
//! oracle is the only new commitment, and the final oracle message itself is the
//! witness.
//!
//! The masking oracle is drawn from caller-provided private randomness and is
//! single-use in this session. target* and the masking root precede lambda; the
//! masked rows M' = M* + lambda M and R' = R* + lambda R follow it; the positions
//! precede the openings. The final claim's target is used as received from the
//! switch (the production target shift is zero); a wrong target is rejected by
//! the linear check, not corrected.

use zkie_core::common::codec::DecodeLimits;
use serde::{Deserialize, Serialize};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::base_stage::{self, BaseRows};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC, KIND_SEAL};
use zkie_engine::pcs::request::code::encode_row_fp3;
use zkie_engine::pcs::request::field::{Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_engine::pcs::request::read::{draw_positions, unique_sorted};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_engine::pcs::request::runtime_tree::{
    decode_runtime_opening, encode_runtime_opening, verify_runtime_opening, RuntimePartRows,
    RuntimeTree, RuntimeTreeOpening, RuntimeTreePlan,
};

const STAGE_STAR: u64 = 24;
const STAGE_LAMBDA: u64 = 25;
const STAGE_ROWS: u64 = 26;
const STAGE_POSITIONS: u64 = 27;
const STAGE_SEAL: u64 = 28;

/// The private randomness of the BASE masking oracle. Tests pass an explicit
/// fixed seed; production draws the seed from the operating system generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LastBaseRandomness {
    Fixed(Digest),
    Private,
}

impl LastBaseRandomness {
    fn seed(self) -> Result<Digest> {
        match self {
            LastBaseRandomness::Fixed(seed) => Ok(seed),
            LastBaseRandomness::Private => {
                let mut bytes = [0u8; 32];
                getrandom::fill(&mut bytes).map_err(|_| Error::EntropyUnavailable)?;
                Ok(Digest(bytes))
            }
        }
    }
}

/// The masking oracle's message and random tail: one main row and one mask row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LastBaseStar {
    pub main: Vec<Fp3>,
    pub main_tail: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub mask_tail: Vec<Fp3>,
}

/// What the prover sends before lambda.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LastBaseMessages {
    pub star_root: Digest,
    pub target_star: Fp3,
}

fn dot(left: &[Fp3], right: &[Fp3]) -> Result<Fp3> {
    if left.len() != right.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (a, b) in left.iter().zip(right) {
        sum = sum.add(&a.mul(b));
    }
    Ok(sum)
}

fn draw_star(
    rows: &SingleRows,
    randomness: &LastBaseRandomness,
) -> Result<LastBaseStar> {
    let mut coins = ByteMaskCoins::new(randomness.seed()?)?;
    let mut draw = |count: u64| -> Result<Vec<Fp3>> {
        let count = usize::try_from(count).map_err(|_| Error::ResourceLimit)?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(coins.draw()?.0);
        }
        Ok(out)
    };
    Ok(LastBaseStar {
        main: draw(rows.code.ell)?,
        main_tail: draw(rows.code.r)?,
        mask: draw(rows.mask_code.ell)?,
        mask_tail: draw(rows.mask_code.r)?,
    })
}

fn target_star(rows: &SingleRows, star: &LastBaseStar) -> Result<Fp3> {
    if rows.masks.len() != 1 || rows.mask_claims.len() != 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    Ok(dot(&rows.claim, &star.main)?.add(&dot(&rows.mask_claims[0], &star.mask)?))
}

fn star_encoded_rows(
    rows: &SingleRows,
    star: &LastBaseStar,
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    let main = encode_row_fp3(&rows.code, &star.main, &star.main_tail)?;
    let mask = encode_row_fp3(&rows.mask_code, &star.mask, &star.mask_tail)?;
    Ok((main, mask))
}

fn check_plan(
    plan: &RuntimeTreePlan,
    rows: &SingleRows,
) -> Result<()> {
    if plan.fault().is_some()
        || plan.parts.len() != 1
        || plan.parts[0].main_symbols != E_DEGREE as u64
        || plan.parts[0].mask_symbols != E_DEGREE as u64
        || plan.log_n != rows.code.log_n
        || plan.log_n_zk != rows.mask_code.log_n
    {
        return Err(Error::UnsupportedPrimitive);
    }
    Ok(())
}

fn absorb(chain: &mut Chain, stage: u64, words: &[u64]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(stage);
    for word in words {
        message.push_u64(*word);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn push_digest(words: &mut Vec<u64>, digest: &Digest) {
    for chunk in digest.0.chunks_exact(8) {
        words.push(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
}

fn symbol_from(symbols: &[zkie_engine::pcs::request::field::Fp]) -> Result<Fp3> {
    if symbols.len() != E_DEGREE {
        return Err(Error::ShapeMismatch);
    }
    Ok(Fp3::new(symbols[0], symbols[1], symbols[2]))
}

/// The prover side of the last-stage BASE: the final SingleRows witness, the
/// independent masking rows and their committed oracle tree.
pub struct LastBaseProver {
    rows: SingleRows,
    star: LastBaseStar,
    plan: RuntimeTreePlan,
    tree: RuntimeTree,
}

impl LastBaseProver {
    pub fn new(
        rows: SingleRows,
        plan: RuntimeTreePlan,
        salts: &[Vec<u8>],
        randomness: &LastBaseRandomness,
    ) -> Result<Self> {
        rows.base_ready()?;
        if rows.masks.len() != 1 || rows.mask_claims.len() != 1 {
            return Err(Error::UnsupportedPrimitive);
        }
        check_plan(&plan, &rows)?;
        if salts.len() as u64 != plan.columns() {
            return Err(Error::ShapeMismatch);
        }
        let star = draw_star(&rows, randomness)?;
        let (main, mask) = star_encoded_rows(&rows, &star)?;
        let tree = RuntimeTree::commit(
            plan.clone(),
            &[RuntimePartRows {
                main: vec![main],
                mask: vec![mask],
                main_deg: E_DEGREE as u32,
            }],
            salts,
        )?;
        Ok(Self {
            rows,
            star,
            plan,
            tree,
        })
    }

    pub fn star_root(&self) -> Digest {
        *self.tree.root()
    }

    pub fn star(&self) -> &LastBaseStar {
        &self.star
    }

    pub fn messages(&self) -> Result<LastBaseMessages> {
        Ok(LastBaseMessages {
            star_root: *self.tree.root(),
            target_star: target_star(&self.rows, &self.star)?,
        })
    }

    pub fn base_rows(&self, lambda: Fp3) -> Result<BaseRows> {
        base_stage::prepare(
            &self.rows,
            &self.star.main,
            &self.star.main_tail,
            &self.star.mask,
            &self.star.mask_tail,
            lambda,
        )
    }

    pub fn open_star(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.tree.open(positions)
    }

    pub fn plan(&self) -> &RuntimeTreePlan {
        &self.plan
    }
}

fn put_vec(writer: &mut CanonicalWriter, values: &[Fp3]) {
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_fp3(value);
    }
}

fn get_vec(reader: &mut CanonicalReader<'_>, expected: u64) -> Result<Vec<Fp3>> {
    if reader.get_u64()? != expected {
        return Err(Error::ShapeMismatch);
    }
    let expected = usize::try_from(expected).map_err(|_| Error::ResourceLimit)?;
    let mut values = Vec::with_capacity(expected);
    for _ in 0..expected {
        values.push(reader.get_fp3()?);
    }
    Ok(values)
}

/// The prover's byte artifact for one last-stage BASE: target*, the masked
/// rows and tails, and the two authenticated openings in message order.
pub fn encode_last_base_proof(
    messages: &LastBaseMessages,
    prepared: &BaseRows,
    star_opening: &RuntimeTreeOpening,
    final_opening: &RuntimeTreeOpening,
    star_plan: &RuntimeTreePlan,
    final_plan: &RuntimeTreePlan,
) -> Result<Vec<u8>> {
    if star_opening.positions != final_opening.positions {
        return Err(Error::InvalidIndex);
    }
    let star_bytes = encode_runtime_opening(star_plan, star_opening)?;
    let final_bytes = encode_runtime_opening(final_plan, final_opening)?;
    let mut writer = CanonicalWriter::new();
    writer.put_digest(&messages.star_root);
    writer.put_fp3(&messages.target_star);
    put_vec(&mut writer, &prepared.main_prime);
    put_vec(&mut writer, &prepared.main_tail_prime);
    put_vec(&mut writer, &prepared.mask_prime);
    put_vec(&mut writer, &prepared.mask_tail_prime);
    writer.put_bytes(&star_bytes);
    writer.put_bytes(&final_bytes);
    Ok(writer.into_bytes())
}

/// Verifies one last-stage BASE from its byte artifact: the verifier draws
/// lambda and the positions from the transcript, decodes exactly the prover's
/// messages, and requires the artifact to end after the openings.
pub fn verify_last_base_proof(
    session: LastBaseSession,
    positions_count: usize,
    bytes: &[u8],
) -> Result<Digest> {
    verify_last_base_proof_with_positions(session, positions_count, bytes).map(|(seal, _)| seal)
}

/// The same verification, also returning the BASE positions the verifier
/// drew. The ledger harness uses them to record the LastBase openings under
/// the same external keys as the rest of the run.
pub fn verify_last_base_proof_with_positions(
    mut session: LastBaseSession,
    positions_count: usize,
    bytes: &[u8],
) -> Result<(Digest, Vec<u64>)> {
    let limits = DecodeLimits {
        max_bytes: bytes.len(),
        max_elements: bytes.len(),
    };
    let mut reader = CanonicalReader::new(bytes, limits);
    let star_root = reader.get_digest()?;
    let target_star = reader.get_fp3()?;
    session.receive_star(star_root, target_star)?;
    let lambda = session.draw_lambda()?;
    let main_prime = get_vec(&mut reader, session.rows.code.ell)?;
    let main_tail_prime = get_vec(&mut reader, session.rows.code.r)?;
    let mask_prime = get_vec(&mut reader, session.rows.mask_code.ell)?;
    let mask_tail_prime = get_vec(&mut reader, session.rows.mask_code.r)?;
    session.receive_rows(BaseRows {
        target_star,
        lambda,
        main_prime,
        main_tail_prime,
        mask_prime,
        mask_tail_prime,
    })?;
    let positions = session.draw_base_positions(positions_count)?;
    let star_bytes = reader.get_bytes()?;
    let star_opening = decode_runtime_opening(&session.star_plan, &positions, star_bytes)?;
    let final_bytes = reader.get_bytes()?;
    let final_opening = decode_runtime_opening(&session.final_plan, &positions, final_bytes)?;
    session.verify_openings(&star_opening, &final_opening)?;
    let sealed = session.finish()?;
    reader.finish()?;
    Ok((sealed, positions))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Star,
    Lambda,
    Rows,
    Positions,
    Openings,
    Sealed,
}

/// The verifier side of the last-stage BASE. It consumes the final switched
/// oracle's fixed plan and declared root and the independent masking oracle's
/// plan; it never rebuilds either root from prover data.
pub struct LastBaseSession {
    chain: Chain,
    rows: SingleRows,
    final_plan: RuntimeTreePlan,
    final_root: Digest,
    star_plan: RuntimeTreePlan,
    stage: Stage,
    star_root: Option<Digest>,
    target_star: Option<Fp3>,
    lambda: Option<Fp3>,
    prepared: Option<BaseRows>,
    positions: Vec<u64>,
}

impl LastBaseSession {
    pub fn new(
        chain: Chain,
        rows: SingleRows,
        final_plan: RuntimeTreePlan,
        final_root: Digest,
        star_plan: RuntimeTreePlan,
    ) -> Result<Self> {
        rows.base_ready()?;
        if rows.masks.len() != 1 || rows.mask_claims.len() != 1 {
            return Err(Error::UnsupportedPrimitive);
        }
        check_plan(&final_plan, &rows)?;
        check_plan(&star_plan, &rows)?;
        if final_root == Digest::ZERO || star_plan == final_plan {
            return Err(Error::StatementMismatch);
        }
        Ok(Self {
            chain,
            rows,
            final_plan,
            final_root,
            star_plan,
            stage: Stage::Fresh,
            star_root: None,
            target_star: None,
            lambda: None,
            prepared: None,
            positions: Vec::new(),
        })
    }

    pub fn chain_value(&self) -> &Digest {
        self.chain.value()
    }

    pub fn positions(&self) -> &[u64] {
        &self.positions
    }

    /// Receives the masking oracle's commitment and target* before lambda.
    pub fn receive_star(&mut self, star_root: Digest, target_star: Fp3) -> Result<()> {
        if self.stage != Stage::Fresh || star_root == Digest::ZERO || star_root == self.final_root {
            return Err(Error::StatementMismatch);
        }
        let mut words = Vec::new();
        words.push(self.star_plan.tree_number);
        words.push(self.star_plan.leaf_class as u64);
        words.push(1);
        words.push(1);
        words.push(self.rows.code.log_n as u64);
        words.push(self.rows.code.ell);
        words.push(self.rows.code.r);
        words.push(self.rows.mask_code.log_n as u64);
        words.push(self.rows.mask_code.ell);
        words.push(self.rows.mask_code.r);
        push_digest(&mut words, &star_root);
        absorb(&mut self.chain, STAGE_STAR, &words)?;
        let mut message = self.chain.message();
        message.push_fp3(&target_star);
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.star_root = Some(star_root);
        self.target_star = Some(target_star);
        self.stage = Stage::Star;
        Ok(())
    }

    pub fn draw_lambda(&mut self) -> Result<Fp3> {
        if self.stage != Stage::Star {
            return Err(Error::InvalidSessionState);
        }
        absorb(&mut self.chain, STAGE_LAMBDA, &[])?;
        let lambda = self.chain.draw_e()?;
        self.lambda = Some(lambda);
        self.stage = Stage::Lambda;
        Ok(lambda)
    }

    /// Receives the masked rows and tails after lambda, checks the linear
    /// relation with the verifier's own claim and the received target*, and
    /// advances to the position draw.
    pub fn receive_rows(&mut self, prepared: BaseRows) -> Result<()> {
        if self.stage != Stage::Lambda {
            return Err(Error::InvalidSessionState);
        }
        let lambda = self.lambda.ok_or(Error::InvalidSessionState)?;
        let target_star = self.target_star.ok_or(Error::InvalidSessionState)?;
        if prepared.lambda != lambda || prepared.target_star != target_star {
            return Err(Error::StatementMismatch);
        }
        let mut words = Vec::new();
        for value in &prepared.main_prime {
            words.push(value);
        }
        for value in &prepared.mask_prime {
            words.push(value);
        }
        for value in &prepared.main_tail_prime {
            words.push(value);
        }
        for value in &prepared.mask_tail_prime {
            words.push(value);
        }
        let mut message = self.chain.message();
        message.push_u64(STAGE_ROWS);
        message.push_u64(words.len() as u64);
        for value in &words {
            message.push_fp3(value);
        }
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        if !base_stage::linear_holds(
            &self.rows.claim,
            &self.rows.mask_claims[0],
            self.rows.target,
            &prepared,
        )? {
            return Err(Error::InvalidProof);
        }
        self.prepared = Some(prepared);
        self.stage = Stage::Rows;
        Ok(())
    }

    /// Draws the BASE positions and returns their sorted unique list.
    pub fn draw_base_positions(&mut self, count: usize) -> Result<Vec<u64>> {
        if self.stage != Stage::Rows || count == 0 {
            return Err(Error::InvalidSessionState);
        }
        absorb(&mut self.chain, STAGE_POSITIONS, &[count as u64])?;
        let drawn = draw_positions(&mut self.chain, count, self.rows.code.log_n)?;
        let unique = unique_sorted(&drawn);
        if unique.is_empty() {
            return Err(Error::InvalidIndex);
        }
        self.positions = unique.clone();
        self.stage = Stage::Positions;
        Ok(unique)
    }

    /// Authenticates the masking oracle and the final switched oracle at the
    /// drawn positions and checks the encoding symbol relations.
    pub fn verify_openings(
        &mut self,
        star_opening: &RuntimeTreeOpening,
        final_opening: &RuntimeTreeOpening,
    ) -> Result<()> {
        if self.stage != Stage::Positions {
            return Err(Error::InvalidSessionState);
        }
        let star_root = self.star_root.ok_or(Error::InvalidSessionState)?;
        let lambda = self.lambda.ok_or(Error::InvalidSessionState)?;
        let prepared = self.prepared.as_ref().ok_or(Error::InvalidSessionState)?;
        if star_opening.positions != self.positions || final_opening.positions != self.positions {
            return Err(Error::InvalidIndex);
        }
        verify_runtime_opening(&self.star_plan, &star_root, &self.positions, star_opening)?;
        verify_runtime_opening(
            &self.final_plan,
            &self.final_root,
            &self.positions,
            final_opening,
        )?;
        let encoded_main = encode_row_fp3(
            &self.rows.code,
            &prepared.main_prime,
            &prepared.main_tail_prime,
        )?;
        let encoded_mask = encode_row_fp3(
            &self.rows.mask_code,
            &prepared.mask_prime,
            &prepared.mask_tail_prime,
        )?;
        let mask_n = self.rows.mask_code.n();
        for (index, position) in self.positions.iter().enumerate() {
            let star_column = &star_opening.columns[index];
            let final_column = &final_opening.columns[index];
            if star_column.symbols.len() != 2 * E_DEGREE
                || final_column.symbols.len() != 2 * E_DEGREE
            {
                return Err(Error::ShapeMismatch);
            }
            let star_main = symbol_from(&star_column.symbols[..E_DEGREE])?;
            let star_mask = symbol_from(&star_column.symbols[E_DEGREE..])?;
            let old_main = symbol_from(&final_column.symbols[..E_DEGREE])?;
            let old_mask = symbol_from(&final_column.symbols[E_DEGREE..])?;
            let x = usize::try_from(*position).map_err(|_| Error::ResourceLimit)?;
            let mask_x = usize::try_from(*position % mask_n)
                .map_err(|_| Error::ResourceLimit)?;
            let encoded_main_at = *encoded_main.get(x).ok_or(Error::InvalidIndex)?;
            let encoded_mask_at = *encoded_mask.get(mask_x).ok_or(Error::InvalidIndex)?;
            if !base_stage::symbol_holds(encoded_main_at, star_main, lambda, old_main)
                || !base_stage::symbol_holds(encoded_mask_at, star_mask, lambda, old_mask)
            {
                return Err(Error::InvalidProof);
            }
        }
        self.stage = Stage::Openings;
        Ok(())
    }

    /// Seals the transcript after the last opening check. No message or
    /// challenge follows; the returned digest is the session's final value.
    pub fn finish(&mut self) -> Result<Digest> {
        if self.stage != Stage::Openings {
            return Err(Error::InvalidSessionState);
        }
        absorb(&mut self.chain, STAGE_SEAL, &[])?;
        let mut message = self.chain.message();
        let prefix = message.prefix(KIND_SEAL, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.stage = Stage::Sealed;
        Ok(*self.chain.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan;

    const CONFIG: Digest = Digest([0x55; 32]);

    fn salts(columns: usize, seed: u8) -> Vec<Vec<u8>> {
        (0..columns)
            .map(|position| vec![(position as u8) ^ seed; 40])
            .collect()
    }

    fn rows() -> SingleRows {
        let code = build_code_plan(3, 4, 1).expect("code");
        let mask_code = build_code_plan(3, 4, 1).expect("mask code");
        SingleRows {
            code,
            mask_code,
            main: (0..4).map(|i| Fp3::from_u64(0x10 + i)).collect(),
            main_tail: vec![Fp3::from_u64(0x20)],
            masks: vec![(0..4).map(|i| Fp3::from_u64(0x30 + i)).collect()],
            mask_tails: vec![vec![Fp3::from_u64(0x40)]],
            claim: vec![Fp3::ONE, Fp3::ZERO, Fp3::ZERO, Fp3::ZERO],
            mask_claims: vec![vec![Fp3::ZERO, Fp3::ONE, Fp3::ZERO, Fp3::ZERO]],
            target: Fp3::from_u64(0x10).add(&Fp3::from_u64(0x31)),
        }
    }

    fn star_plan() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 30,
            leaf_class: 0,
            log_n: 3,
            log_n_zk: 3,
            parts: vec![zkie_engine::pcs::request::runtime_tree::RuntimeTreePart {
                main_symbols: 3,
                mask_symbols: 3,
            }],
            salt_bytes: 40,
            run: Digest([0x91; 32]),
            configuration: CONFIG,
        }
    }

    fn final_plan() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 31,
            leaf_class: 0,
            log_n: 3,
            log_n_zk: 3,
            parts: vec![zkie_engine::pcs::request::runtime_tree::RuntimeTreePart {
                main_symbols: 3,
                mask_symbols: 3,
            }],
            salt_bytes: 40,
            run: Digest([0x92; 32]),
            configuration: CONFIG,
        }
    }

    struct FinalOracle {
        tree: RuntimeTree,
        plan: RuntimeTreePlan,
    }

    fn final_oracle(rows: &SingleRows) -> FinalOracle {
        let plan = final_plan();
        let main = encode_row_fp3(&rows.code, &rows.main, &rows.main_tail).expect("main");
        let mask = encode_row_fp3(&rows.mask_code, &rows.masks[0], &rows.mask_tails[0])
            .expect("mask");
        let tree = RuntimeTree::commit(
            plan.clone(),
            &[RuntimePartRows {
                main: vec![main],
                mask: vec![mask],
                main_deg: E_DEGREE as u32,
            }],
            &salts(plan.columns() as usize, 0x71),
        )
        .expect("final tree");
        FinalOracle { tree, plan }
    }

    fn prover_session(
        rows: &SingleRows,
        final_oracle: &FinalOracle,
        randomness: &LastBaseRandomness,
    ) -> (LastBaseProver, Digest) {
        let plan = star_plan();
        let prover = LastBaseProver::new(
            rows.clone(),
            plan.clone(),
            &salts(plan.columns() as usize, 0x72),
            randomness,
        )
        .expect("prover");
        (prover, *final_oracle.tree.root())
    }

    fn run(
        rows: &SingleRows,
        final_oracle: &FinalOracle,
        randomness: &LastBaseRandomness,
        tamper_final_root: bool,
        tamper_star_root: bool,
        tamper_star_opening: bool,
        tamper_final_opening: bool,
        tamper_prime: bool,
        tamper_target_star: bool,
        tamper_claim_target: bool,
    ) -> Result<Digest> {
        let (prover, final_root) = prover_session(rows, final_oracle, randomness);
        let messages = prover.messages()?;
        let root = if tamper_final_root {
            Digest([0xee; 32])
        } else {
            final_root
        };
        let star_root = if tamper_star_root {
            Digest([0xef; 32])
        } else {
            messages.star_root
        };
        let target_star = if tamper_target_star {
            messages.target_star.add(&Fp3::ONE)
        } else {
            messages.target_star
        };
        let mut verifier_rows = rows.clone();
        if tamper_claim_target {
            verifier_rows.target = verifier_rows.target.add(&Fp3::ONE);
        }
        let chain = Chain::begin(99, &Digest::ZERO, b"last base test", &[], None)?;
        let mut session = LastBaseSession::new(
            chain,
            verifier_rows,
            final_oracle.plan.clone(),
            root,
            prover.plan().clone(),
        )?;
        session.receive_star(star_root, target_star)?;
        let lambda = session.draw_lambda()?;
        let mut prepared = prover.base_rows(lambda)?;
        if tamper_prime {
            prepared.main_prime[0] = prepared.main_prime[0].add(&Fp3::ONE);
        }
        session.receive_rows(prepared)?;
        let positions = session.draw_base_positions(2)?;
        let mut star_opening = prover.open_star(&positions)?;
        let mut final_opening = final_oracle.tree.open(&positions)?;
        if tamper_star_opening {
            star_opening.columns[0].symbols[0] =
                star_opening.columns[0].symbols[0] + zkie_engine::pcs::request::field::Fp::new(1);
        }
        if tamper_final_opening {
            final_opening.columns[0].symbols[0] =
                final_opening.columns[0].symbols[0] + zkie_engine::pcs::request::field::Fp::new(1);
        }
        session.verify_openings(&star_opening, &final_opening)?;
        session.finish()
    }

    fn randomness() -> LastBaseRandomness {
        LastBaseRandomness::Fixed(Digest([0x33; 32]))
    }

    #[test]
    fn the_fixed_seed_is_reproducible_and_the_private_source_is_available() {
        let fixed = LastBaseRandomness::Fixed(Digest([0x44; 32]));
        assert_eq!(fixed.seed().unwrap(), Digest([0x44; 32]));
        assert!(LastBaseRandomness::Private.seed().is_ok());
    }

    #[test]
    fn the_last_base_accepts_an_honest_run_and_seals() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        let sealed = run(&rows, &oracle, &randomness(), false, false, false, false, false, false, false)
            .expect("honest base");
        assert_ne!(sealed, Digest::ZERO);
    }

    fn typed_and_encoded(
        rows: &SingleRows,
        oracle: &FinalOracle,
    ) -> (Digest, Vec<u8>, LastBaseProver) {
        let (prover, root) = prover_session(rows, oracle, &randomness());
        let messages = prover.messages().expect("messages");
        let chain = Chain::begin(99, &Digest::ZERO, b"last base test", &[], None).unwrap();
        let mut session = LastBaseSession::new(
            chain,
            rows.clone(),
            oracle.plan.clone(),
            root,
            prover.plan().clone(),
        )
        .unwrap();
        session
            .receive_star(messages.star_root, messages.target_star)
            .unwrap();
        let lambda = session.draw_lambda().unwrap();
        let prepared = prover.base_rows(lambda).unwrap();
        session.receive_rows(prepared.clone()).unwrap();
        let positions = session.draw_base_positions(2).unwrap();
        let star_opening = prover.open_star(&positions).unwrap();
        let final_opening = oracle.tree.open(&positions).unwrap();
        session
            .verify_openings(&star_opening, &final_opening)
            .unwrap();
        let sealed = session.finish().unwrap();
        let bytes = encode_last_base_proof(
            &messages,
            &prepared,
            &star_opening,
            &final_opening,
            prover.plan(),
            &oracle.plan,
        )
        .unwrap();
        (sealed, bytes, prover)
    }

    fn fresh_session(
        rows: &SingleRows,
        oracle: &FinalOracle,
        prover: &LastBaseProver,
    ) -> LastBaseSession {
        let chain = Chain::begin(99, &Digest::ZERO, b"last base test", &[], None).unwrap();
        LastBaseSession::new(
            chain,
            rows.clone(),
            oracle.plan.clone(),
            *oracle.tree.root(),
            prover.plan().clone(),
        )
        .unwrap()
    }

    #[test]
    fn the_last_base_proof_decodes_to_the_same_seal_and_rejects_leftovers() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        let (typed, bytes, prover) = typed_and_encoded(&rows, &oracle);
        let session = fresh_session(&rows, &oracle, &prover);
        let decoded = verify_last_base_proof(session, 2, &bytes).expect("decoded proof");
        assert_eq!(decoded, typed);

        let mut extra = bytes.clone();
        extra.push(0);
        let session = fresh_session(&rows, &oracle, &prover);
        assert!(matches!(
            verify_last_base_proof(session, 2, &extra),
            Err(Error::InvalidEncoding)
        ));

        let short = &bytes[..bytes.len() - 1];
        let session = fresh_session(&rows, &oracle, &prover);
        assert!(matches!(
            verify_last_base_proof(session, 2, short),
            Err(Error::InvalidEncoding)
        ));
    }

    #[test]
    fn the_last_base_proof_rejects_a_tampered_masked_row() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        let (_typed, mut bytes, prover) = typed_and_encoded(&rows, &oracle);
        // The masked main row starts after the digest and target* and its
        // length word: 32 + 24 + 8 bytes.
        let at = 32 + 24 + 8;
        bytes[at] ^= 1;
        let session = fresh_session(&rows, &oracle, &prover);
        assert!(matches!(
            verify_last_base_proof(session, 2, &bytes),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_last_base_rejects_a_wrong_final_root() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        assert!(matches!(
            run(&rows, &oracle, &randomness(), true, false, false, false, false, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_last_base_rejects_a_wrong_star_root() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        assert!(matches!(
            run(&rows, &oracle, &randomness(), false, true, false, false, false, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_last_base_rejects_tampered_star_symbols() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        assert!(matches!(
            run(&rows, &oracle, &randomness(), false, false, true, false, false, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_last_base_rejects_tampered_final_symbols() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        assert!(matches!(
            run(&rows, &oracle, &randomness(), false, false, false, true, false, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_last_base_rejects_tampered_masked_rows() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        assert!(matches!(
            run(&rows, &oracle, &randomness(), false, false, false, false, true, false, false),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_last_base_rejects_a_shifted_target_star() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        assert!(matches!(
            run(
                &rows,
                &oracle,
                &randomness(),
                false,
                false,
                false,
                false,
                false,
                true,
                false
            ),
            Err(Error::StatementMismatch)
        ));
    }

    #[test]
    fn the_last_base_rejects_a_wrong_claim_target() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        assert!(matches!(
            run(
                &rows,
                &oracle,
                &randomness(),
                false,
                false,
                false,
                false,
                false,
                false,
                true
            ),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn the_last_base_rejects_messages_out_of_order() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        let (prover, root) = prover_session(&rows, &oracle, &randomness());
        let chain = Chain::begin(99, &Digest::ZERO, b"last base order", &[], None).unwrap();
        let mut session = LastBaseSession::new(
            chain,
            rows.clone(),
            oracle.plan.clone(),
            root,
            prover.plan().clone(),
        )
        .unwrap();
        assert!(matches!(
            session.draw_lambda(),
            Err(Error::InvalidSessionState)
        ));
        let messages = prover.messages().unwrap();
        session.receive_star(messages.star_root, messages.target_star).unwrap();
        let lambda = session.draw_lambda().unwrap();
        let _prepared = prover.base_rows(lambda).unwrap();
        assert!(matches!(
            session.verify_openings(&prover.open_star(&[1]).unwrap(), &oracle.tree.open(&[1]).unwrap()),
            Err(Error::InvalidSessionState)
        ));
        assert!(session.finish().is_err());
    }

    #[test]
    fn the_last_base_rejects_an_unsupported_layout() {
        let mut many = rows();
        many.masks.push(many.masks[0].clone());
        many.mask_tails.push(many.mask_tails[0].clone());
        many.mask_claims.push(many.mask_claims[0].clone());
        let oracle = final_oracle(&rows());
        let chain = Chain::begin(99, &Digest::ZERO, b"last base layout", &[], None).unwrap();
        assert!(LastBaseSession::new(
            chain,
            many,
            oracle.plan.clone(),
            *oracle.tree.root(),
            star_plan(),
        )
        .is_err());
    }

    #[test]
    fn star_values_are_fresh_one_time_pads_for_the_target_and_rows() {
        let rows = rows();
        let plan = star_plan();
        let salt_values = salts(plan.columns() as usize, 0x72);
        let first = LastBaseProver::new(
            rows.clone(),
            plan.clone(),
            &salt_values,
            &LastBaseRandomness::Fixed(Digest([0x33; 32])),
        )
        .unwrap();
        let second = LastBaseProver::new(
            rows.clone(),
            plan.clone(),
            &salt_values,
            &LastBaseRandomness::Fixed(Digest([0x44; 32])),
        )
        .unwrap();
        assert_ne!(first.star().main, second.star().main);
        assert_ne!(first.star().mask, second.star().mask);
        assert_ne!(first.star().main_tail, second.star().main_tail);
        assert_ne!(first.star().mask_tail, second.star().mask_tail);
        let lambda = Fp3::from_u64(7);
        let prepared = first.base_rows(lambda).unwrap();
        let target = dot(&rows.claim, &first.star().main)
            .unwrap()
            .add(&dot(&rows.mask_claims[0], &first.star().mask).unwrap());
        assert_eq!(prepared.target_star, target);
        assert_eq!(prepared.target_star, first.messages().unwrap().target_star);
        for (index, value) in prepared.main_prime.iter().enumerate() {
            assert_eq!(
                *value,
                first.star().main[index].add(&lambda.mul(&rows.main[index]))
            );
        }
        for (index, value) in prepared.mask_prime.iter().enumerate() {
            assert_eq!(
                *value,
                first.star().mask[index].add(&lambda.mul(&rows.masks[0][index]))
            );
        }
        assert_eq!(
            prepared.main_tail_prime[0],
            first.star().main_tail[0].add(&lambda.mul(&rows.main_tail[0]))
        );
        assert_eq!(
            prepared.mask_tail_prime[0],
            first.star().mask_tail[0].add(&lambda.mul(&rows.mask_tails[0][0]))
        );
    }

    #[test]
    fn the_last_base_rejects_a_second_masking_oracle() {
        let rows = rows();
        let oracle = final_oracle(&rows);
        let (prover, root) = prover_session(&rows, &oracle, &randomness());
        let chain = Chain::begin(99, &Digest::ZERO, b"last base reuse", &[], None).unwrap();
        let mut session = LastBaseSession::new(
            chain,
            rows.clone(),
            oracle.plan.clone(),
            root,
            prover.plan().clone(),
        )
        .unwrap();
        let messages = prover.messages().unwrap();
        session.receive_star(messages.star_root, messages.target_star).unwrap();
        assert!(session
            .receive_star(messages.star_root, messages.target_star)
            .is_err());
    }
}
