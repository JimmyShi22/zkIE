//! FOLDEQ session skeleton (foundation only).
//!
//! The verifier's plan holds the committed table, the fold shape, the group
//! registry and the budgets/counts; every number comes from the verified
//! statement/plan, not from a proof. The session fixes the order: the complete
//! first query, the record seeds, one FOLD start (fresh mask plus hiding sum),
//! the column query round and the column reads. The mask's oracle root, the
//! FOLD round polynomials, the record fold/table test and the linkage tests
//! are not in this skeleton and no closure is claimed.

use zkie_core::error::{Error, Result};

use crate::pcs::request::chain::{Chain, KIND_EPSILON, KIND_POSITIONS};
use crate::pcs::request::field::{Fp, Fp3};
use crate::pcs::request::fold::{
    fold_mask_notice, fold_mask_shape_fault, fold_start, FoldMask, FoldShape, FoldStart,
    MaskCoins, MaskOracle, VerifiedGroups,
};
use crate::pcs::request::read::{draw_positions, VerifierReads};
use crate::pcs::request::table::{
    record_seeds, ColumnOpening, IndexName, IndexReader, SampleSeed,
};

/// The verifier's FOLDEQ plan. The reader is the committed table, the fold
/// shape and group registry come from the statement, and the seed/read budgets
/// and column draw count come from the plan. None of these is taken from a
/// proof.
#[derive(Clone, Debug)]
pub struct FoldeqPlan {
    pub reader: IndexReader,
    pub fold: FoldShape,
    pub groups: VerifiedGroups,
    pub group_id: u64,
    pub leaf_class: u32,
    pub seed_budget: usize,
    pub read_budget: u64,
    pub column_draws: usize,
}

impl FoldeqPlan {
    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.reader.fault() {
            return Some(fault);
        }
        if let Some(fault) = fold_mask_shape_fault(&self.fold) {
            return Some(fault);
        }
        let group = match self.groups.get(self.group_id) {
            Some(group) => group,
            None => return Some("the plan names a group the verifier does not hold"),
        };
        if group.members.is_empty() {
            return Some("the plan's group has no members");
        }
        let draws = match u64::try_from(self.column_draws) {
            Ok(draws) => draws,
            Err(_) => return Some("the column draw count does not fit"),
        };
        if draws > self.read_budget {
            return Some("the column draws exceed the read budget");
        }
        let answers = self.reader.shape.answers as usize;
        if self.seed_budget < answers {
            return Some("the seed budget is below the answer count");
        }
        None
    }

    /// The bytes the first query binds: the complete table name (commitment and
    /// record), before any challenge that reads it.
    pub fn first_query(&self) -> Vec<u8> {
        IndexName {
            commitment: self.reader.commitment(),
            record: self.reader.record.clone(),
        }
        .encode()
    }

    /// The verifier checks the prover's first-query bytes against its plan.
    pub fn check_first_query(&self, sent: &[u8]) -> Result<()> {
        if sent != self.first_query() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    /// The record seeds, with the plan's seed budget.
    pub fn seeds(&self) -> Result<Vec<SampleSeed>> {
        record_seeds(&self.reader.shape, &self.reader.name(), self.seed_budget)
    }

    /// The verifier's fresh-mask view for the plan's group and shape.
    pub fn mask_oracle(&self, id: u64) -> Result<MaskOracle> {
        let notice = fold_mask_notice(&self.fold)?;
        Ok(MaskOracle::from_notice(id, self.group_id, &notice))
    }

    /// Check and open one FOLD start. The mask must have been sent under the
    /// id the verifier checks, and its coefficients and tail must still be the
    /// drawn values.
    pub fn start(&self, mask: &FoldMask, coins: &impl MaskCoins) -> Result<FoldStart> {
        let id = mask.id().ok_or(Error::InvalidProof)?;
        let oracle = self.mask_oracle(id)?;
        fold_start(&self.fold, mask, coins, &oracle, &self.groups)
    }

    /// Draw the plan's column query positions from the caller's chain. The
    /// draws are not deduplicated; a repeated position is a repeated read.
    pub fn draw_queries(&self, chain: &mut Chain) -> Result<Vec<u64>> {
        draw_positions(chain, self.column_draws, self.reader.shape.code.log_n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    FirstQuery,
    Seeds,
    Started,
    Queried,
}

/// The verifier's FOLDEQ session. It owns the chain, the plan and the read
/// store; the prover's openings are passed in. This is a joint-development
/// harness, not a production verifier.
pub struct FoldeqSession {
    plan: FoldeqPlan,
    chain: Chain,
    reads: VerifierReads,
    stage: Stage,
    start: Option<FoldStart>,
    epsilon: Option<Fp3>,
    queries: Vec<u64>,
}

impl FoldeqSession {
    pub fn new(plan: FoldeqPlan, chain: Chain) -> Result<Self> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let reads = VerifierReads::new(plan.read_budget);
        Ok(Self {
            plan,
            chain,
            reads,
            stage: Stage::FirstQuery,
            start: None,
            epsilon: None,
            queries: Vec::new(),
        })
    }

    pub fn plan(&self) -> &FoldeqPlan {
        &self.plan
    }

    pub fn queries(&self) -> &[u64] {
        &self.queries
    }

    pub fn epsilon(&self) -> Option<Fp3> {
        self.epsilon
    }

    pub fn start(&self) -> Option<&FoldStart> {
        self.start.as_ref()
    }

    /// The prover's first-query bytes must equal the plan's complete table
    /// name. The caller must also have started the chain on the same bytes.
    pub fn bind_first_query(&mut self, sent: &[u8]) -> Result<()> {
        if self.stage != Stage::FirstQuery {
            return Err(Error::InvalidSessionState);
        }
        self.plan.check_first_query(sent)?;
        self.stage = Stage::Seeds;
        Ok(())
    }

    /// The verifier's record seeds, from the plan and the complete first query.
    pub fn seeds(&self) -> Result<Vec<SampleSeed>> {
        if self.stage != Stage::Seeds {
            return Err(Error::InvalidSessionState);
        }
        self.plan.seeds()
    }

    /// Check the fresh mask and send the mask notice plus the hiding sum. The
    /// start round draws epsilon. The mask's oracle root is not in this
    /// skeleton; the notice fields and the sum are bound here and the root
    /// binding stays open.
    pub fn start_fold(&mut self, mask: &FoldMask, coins: &impl MaskCoins) -> Result<()> {
        if self.stage != Stage::Seeds {
            return Err(Error::InvalidSessionState);
        }
        let start = self.plan.start(mask, coins)?;
        let notice = start.draft.notice;
        let mut message = self.chain.message();
        message.push_u64(notice.mask_rows);
        message.push_u64(notice.mask_code.log_n as u64);
        message.push_u64(notice.mask_code.ell);
        message.push_u64(notice.mask_code.r);
        message.push_u64(notice.answers as u64);
        message.push_u64(notice.leaf_class as u64);
        message.push_fp3(&start.sum);
        let prefix = message.prefix(KIND_EPSILON, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.epsilon = Some(self.chain.draw_e()?);
        self.start = Some(start);
        self.stage = Stage::Started;
        Ok(())
    }

    /// Step the column query round and draw the plan's positions.
    pub fn draw_queries(&mut self) -> Result<()> {
        if self.stage != Stage::Started {
            return Err(Error::InvalidSessionState);
        }
        let prefix = self.chain.message().prefix(KIND_POSITIONS, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.queries = self.plan.draw_queries(&mut self.chain)?;
        self.stage = Stage::Queried;
        Ok(())
    }

    /// Verify the prover's openings for the drawn positions, charge the reads
    /// and remember the values. A repeated position must return the same
    /// values; an exhausted budget fails closed.
    pub fn read_queries(&mut self, openings: &[ColumnOpening]) -> Result<Vec<Vec<Fp>>> {
        if self.stage != Stage::Queried {
            return Err(Error::InvalidSessionState);
        }
        if openings.len() != self.queries.len() {
            return Err(Error::ShapeMismatch);
        }
        let Self {
            plan,
            reads,
            queries,
            ..
        } = self;
        reads.read(&plan.reader, queries, openings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::field::MODULUS;
    use crate::pcs::request::fold::{fold_mask_make, fold_mask_make_tail, VerifiedGroup};
    use crate::pcs::request::params::{LEAF_SALT_MIN_BYTES, REQUEST_CHAIN_LABEL};
    use crate::pcs::request::read::ProverReads;
    use crate::pcs::request::table::{
        commit_table, IndexShape, TableDistances,
    };
    use zkie_core::common::field::PrimeCharacteristicRing;
    use zkie_core::types::Digest;

    fn shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn committed() -> crate::pcs::request::table::CommittedTable {
        let shape = shape();
        let rows = shape.rows() as usize;
        let messages: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..shape.code.ell)
                    .map(|index| Fp::from_u64((row * 10 + index as usize + 1) as u64))
                    .collect()
            })
            .collect();
        let tails: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..shape.code.r)
                    .map(|index| Fp::from_u64((row + index as usize + 1) as u64))
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<u8>> = (0..shape.code.n() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); shape.salt_bytes])
            .collect();
        commit_table(shape, Digest([7; 32]), &messages, &tails, &salts).unwrap()
    }

    fn fold_shape() -> FoldShape {
        FoldShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k: 2,
            h: 1,
            mask_leaf_class: 3,
        }
    }

    fn plan_fixture() -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        let table = committed();
        let plan = FoldeqPlan {
            reader: table.reader.clone(),
            fold: fold_shape(),
            groups: VerifiedGroups {
                groups: vec![VerifiedGroup {
                    group: 1,
                    members: vec![1, 2],
                    fresh: Some(9),
                }],
            },
            group_id: 1,
            leaf_class: 3,
            seed_budget: table.reader.shape.answers as usize,
            read_budget: 2,
            column_draws: 2,
        };
        (plan, table)
    }

    fn session() -> (FoldeqSession, crate::pcs::request::table::CommittedTable) {
        let (plan, table) = plan_fixture();
        let chain = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            &plan.first_query(),
            &[],
            None,
        )
        .unwrap();
        (FoldeqSession::new(plan, chain).unwrap(), table)
    }

    #[test]
    fn first_query_and_stage_order() {
        let (mut session, table) = session();
        assert!(session.seeds().is_err());
        let mut sent = session.plan().first_query();
        sent[0] ^= 1;
        assert!(session.bind_first_query(&sent).is_err());
        let good = session.plan().first_query();
        session.bind_first_query(&good).unwrap();
        assert_eq!(session.seeds().unwrap().len(), 1);
        assert!(session.draw_queries().is_err());
        assert!(session.read_queries(&[]).is_err());
        let _ = table;
    }

    #[test]
    fn start_queries_and_reads() {
        let (mut session, table) = session();
        let good = session.plan().first_query();
        session.bind_first_query(&good).unwrap();
        let _seeds = session.seeds().unwrap();
        let shape = session.plan().fold;
        let mut source = TestCoins::new(Digest([9; 32]));
        let mut mask = fold_mask_make(&shape, &mut source).unwrap();
        fold_mask_make_tail(&shape, &mut mask, &mut source).unwrap();
        mask.sent_as(9).unwrap();
        session.start_fold(&mask, &source).unwrap();
        assert!(session.epsilon().is_some());
        session.draw_queries().unwrap();
        assert_eq!(session.queries().len(), 2);
        let mut prover = ProverReads::new(2);
        let openings = prover.open(&table, session.queries()).unwrap();
        let columns = session.read_queries(&openings).unwrap();
        assert_eq!(columns.len(), 2);
        // A second read charges past the plan's budget.
        assert!(session.read_queries(&openings).is_err());
    }

    #[test]
    fn plan_faults_are_rejected() {
        let (mut plan, _table) = plan_fixture();
        plan.seed_budget = 0;
        let chain = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            &plan.first_query(),
            &[],
            None,
        )
        .unwrap();
        assert!(FoldeqSession::new(plan, chain).is_err());
        let (mut plan, _table) = plan_fixture();
        plan.read_budget = 1;
        plan.column_draws = 2;
        let chain = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            &plan.first_query(),
            &[],
            None,
        )
        .unwrap();
        assert!(FoldeqSession::new(plan, chain).is_err());
        let (mut plan, _table) = plan_fixture();
        plan.group_id = 7;
        let chain = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            &plan.first_query(),
            &[],
            None,
        )
        .unwrap();
        assert!(FoldeqSession::new(plan, chain).is_err());
    }

    // A local numbered source for the joint harness tests; it is not a
    // production randomness source and provides no hiding.
    struct TestCoins {
        seed: Digest,
        draw: u32,
    }

    impl TestCoins {
        fn new(seed: Digest) -> Self {
            Self { seed, draw: 0 }
        }
    }

    impl MaskCoins for TestCoins {
        fn draw(&mut self) -> Result<(Fp3, u32)> {
            let number = self.draw;
            self.draw = self.draw.checked_add(1).ok_or(Error::ResourceLimit)?;
            let digest = test_digest(self.seed, number)?;
            let mut coefficients = [Fp::ZERO; 3];
            for (index, coefficient) in coefficients.iter_mut().enumerate() {
                let word = u64::from_le_bytes(
                    digest.0[index * 8..(index + 1) * 8].try_into().unwrap(),
                );
                *coefficient = Fp::from_u64(word % MODULUS);
            }
            Ok((Fp3::new(coefficients[0], coefficients[1], coefficients[2]), number))
        }

        fn redraw(&self, number: u32) -> Result<Fp3> {
            let digest = test_digest(self.seed, number)?;
            let mut coefficients = [Fp::ZERO; 3];
            for (index, coefficient) in coefficients.iter_mut().enumerate() {
                let word = u64::from_le_bytes(
                    digest.0[index * 8..(index + 1) * 8].try_into().unwrap(),
                );
                *coefficient = Fp::from_u64(word % MODULUS);
            }
            Ok(Fp3::new(coefficients[0], coefficients[1], coefficients[2]))
        }
    }

    fn test_digest(seed: Digest, number: u32) -> Result<Digest> {
        use crate::primitives::transcript::{hash_labeled, tag_word, TEST_COINS};
        hash_labeled(tag_word(TEST_COINS, 0), number as u64, Some(&seed), &[])
    }
}
