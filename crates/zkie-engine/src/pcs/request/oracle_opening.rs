//! Isolated candidate for non-stored oracle openings.
//!
//! The stored-table path stays in `read`/`table`. This module models only the
//! chain-source and plan checks of a non-stored oracle opening: the list and
//! root must have been recorded by a current session or a sealed predecessor,
//! the plan's salt/run/configuration must match, and the actual runtime-tree
//! opening must lead to that root. It is a fixture boundary: chain values and
//! roots are supplied by the isolated constructor, not decoded from real proof
//! bytes, and only a shared node/leaf store is modeled. It is not the general
//! reference `receive_opening`, does not close a production session, and does
//! not claim hiding, table-test or later-proof closure.
//!
//! Production blockers left explicit: a TreeBook-style opening budget and
//! cross-chain adoption (`count_opening`/`adopt`/`list_for`), a trusted
//! `open_oracles` source, `fold_mask_as_drawn` accounting, the provider hiding
//! ledger, canonical proof-byte decoding, and the IOPP/CS/BASE continuation are
//! not connected. A successful opening here is only an isolated Merkle/plan
//! subset, not a hiding or proof-acceptance result.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::Chain;

use crate::pcs::request::read::{draw_positions, unique_sorted};
use crate::pcs::request::runtime_tree::{
    verify_runtime_opening_shared, OpeningNodeStore, RuntimeTreeOpening, RuntimeTreePlan,
};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReceivedRoot {
    pub root: Digest,
    pub run: Digest,
    pub configuration: Digest,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct KnownChain {
    lists: BTreeSet<Vec<u64>>,
    roots: BTreeMap<Digest, ReceivedRoot>,
}

/// A sealed predecessor. Its fields are private; only `OpeningSession::seal`
/// creates one. A follow session can use the predecessor's list/root provenance.
#[derive(Clone, Debug)]
pub struct SealedChain {
    label: u64,
    value: Digest,
    known: KnownChain,
}

impl SealedChain {
    pub fn label(&self) -> u64 {
        self.label
    }

    pub fn value(&self) -> Digest {
        self.value
    }
}

/// A source issued for a recorded list/root. Fields are private so ordinary
/// callers cannot label an arbitrary list/root as coming from a chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningSource {
    session: u64,
    origin: Digest,
    list: Vec<u64>,
    root: Digest,
}

/// The isolated verifier-side chain state. It records what `draw_positions`
/// drew and what `receive_root` received; these are fixture records, not a
/// decoded production proof.
#[derive(Clone, Debug)]
pub struct OpeningSession {
    id: u64,
    chain: Chain,
    configuration: Digest,
    current_value: Digest,
    current: KnownChain,
    prior: Vec<SealedChain>,
    nodes: BTreeMap<Digest, OpeningNodeStore>,
}

impl OpeningSession {
    pub fn new(chain: Chain, configuration: Digest) -> Self {
        let current_value = *chain.value();
        Self {
            id: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            chain,
            configuration,
            current_value,
            current: KnownChain::default(),
            prior: Vec::new(),
            nodes: BTreeMap::new(),
        }
    }

    pub fn draw_positions(&mut self, count: usize, log_n: u32) -> Result<Vec<u64>> {
        let draws = draw_positions(&mut self.chain, count, log_n)?;
        let unique = unique_sorted(&draws);
        self.current.lists.insert(unique.clone());
        Ok(unique)
    }

    /// Close one empty algebraic round so this isolated chain has challenge
    /// words. The round position is fixture state, not a decoded proof.
    pub fn advance_empty_round(&mut self) -> Result<()> {
        let prefix = self
            .chain
            .message()
            .prefix(crate::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }

    pub fn receive_root(
        &mut self,
        root: Digest,
        run: Digest,
        configuration: Digest,
    ) -> Result<()> {
        if self.current.roots.contains_key(&root) {
            return Err(Error::InvalidSessionState);
        }
        self.current.roots.insert(
            root,
            ReceivedRoot {
                root,
                run,
                configuration,
            },
        );
        Ok(())
    }

    pub fn seal(self) -> SealedChain {
        SealedChain {
            label: self.chain.label(),
            value: self.current_value,
            known: self.current,
        }
    }

    /// Start a session that may use the predecessor's recorded lists and roots.
    /// The new chain must be resumed at the sealed value; this is the isolated
    /// form of a followed chain, not a proof-byte decoder.
    pub fn follow(
        prior: &SealedChain,
        chain: Chain,
        configuration: Digest,
    ) -> Result<Self> {
        if chain.label() != prior.label || *chain.value() != prior.value {
            return Err(Error::StatementMismatch);
        }
        let mut session = Self::new(chain, configuration);
        session.prior.push(prior.clone());
        Ok(session)
    }

    /// Issue a source only for a list/root recorded by this session or one of
    /// its sealed predecessors. An unrelated chain cannot pass this boundary.
    pub fn authorize(&self, list: &[u64], root: Digest) -> Result<OpeningSource> {
        let mut unique = list.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique != list || list.is_empty() {
            return Err(Error::InvalidIndex);
        }
        if self.current.lists.contains(list) && self.current.roots.contains_key(&root) {
            return Ok(OpeningSource {
                session: self.id,
                origin: self.current_value,
                list: list.to_vec(),
                root,
            });
        }
        for prior in &self.prior {
            if prior.known.lists.contains(list) && prior.known.roots.contains_key(&root) {
                return Ok(OpeningSource {
                    session: self.id,
                    origin: prior.value,
                    list: list.to_vec(),
                    root,
                });
            }
        }
        Err(Error::StatementMismatch)
    }

    fn known_source(&self, source: &OpeningSource) -> Result<&ReceivedRoot> {
        if source.session != self.id {
            return Err(Error::StatementMismatch);
        }
        if source.origin == self.current_value {
            if self.current.lists.contains(&source.list) {
                return self
                    .current
                    .roots
                    .get(&source.root)
                    .ok_or(Error::StatementMismatch);
            }
        }
        for prior in &self.prior {
            if prior.value == source.origin
                && prior.known.lists.contains(&source.list)
            {
                return prior
                    .known
                    .roots
                    .get(&source.root)
                    .ok_or(Error::StatementMismatch);
            }
        }
        Err(Error::StatementMismatch)
    }

    #[cfg(test)]
    pub fn forge_source_for_test(
        &self,
        list: Vec<u64>,
        root: Digest,
        origin: Digest,
    ) -> OpeningSource {
        OpeningSource {
            session: self.id,
            origin,
            list,
            root,
        }
    }

    /// Verify the isolated non-stored opening subset. The plan salt, run and
    /// configuration and the recorded source are checked before the actual
    /// runtime opening. The node store is committed only after success.
    pub fn verify_non_stored(
        &mut self,
        plan: &NonStoredPlan,
        source: &OpeningSource,
        opening: &RuntimeTreeOpening,
    ) -> Result<()> {
        if plan.stored || plan.tree.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if plan.plan_salt_bytes != plan.tree.salt_bytes {
            return Err(Error::ShapeMismatch);
        }
        if plan.tree.configuration != self.configuration {
            return Err(Error::StatementMismatch);
        }
        let binding = self.known_source(source)?;
        if binding.run != plan.tree.run || binding.configuration != plan.tree.configuration {
            return Err(Error::StatementMismatch);
        }
        if opening.positions != source.list {
            return Err(Error::InvalidIndex);
        }
        let prior = self
            .nodes
            .get(&source.root)
            .cloned()
            .unwrap_or_default();
        let staged = verify_runtime_opening_shared(
            &plan.tree,
            &source.root,
            &source.list,
            opening,
            &prior,
        )?;
        self.nodes.insert(source.root, staged);
        Ok(())
    }
}

/// The non-stored plan subset: a runtime tree plus the plan's salt length and
/// the stored-table marker. The marker must be false on this path.
#[derive(Clone, Debug)]
pub struct NonStoredPlan {
    pub tree: RuntimeTreePlan,
    pub plan_salt_bytes: usize,
    pub stored: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::runtime_tree::{
        verify_runtime_opening, RuntimePartRows, RuntimeTree, RuntimeTreePart,
    };
    use crate::pcs::request::field::Fp3;

    fn plan() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 3,
            leaf_class: 1,
            log_n: 2,
            log_n_zk: 0,
            parts: vec![RuntimeTreePart {
                main_symbols: 2,
                mask_symbols: 0,
            }],
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        }
    }

    fn tree() -> (RuntimeTree, NonStoredPlan) {
        let plan = plan();
        let rows = vec![
            vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3), Fp3::from_u64(4)],
            vec![Fp3::from_u64(5), Fp3::from_u64(6), Fp3::from_u64(7), Fp3::from_u64(8)],
        ];
        let salts: Vec<Vec<u8>> = (0..4)
            .map(|position| vec![(position as u8) + 1; LEAF_SALT_MIN_BYTES])
            .collect();
        let tree = RuntimeTree::commit(
            plan.clone(),
            &[RuntimePartRows {
                main: rows,
                mask: Vec::new(),
                main_deg: 1,
            }],
            &salts,
        )
        .unwrap();
        (
            tree,
            NonStoredPlan {
                tree: plan,
                plan_salt_bytes: LEAF_SALT_MIN_BYTES,
                stored: false,
            },
        )
    }

    fn chain() -> Chain {
        Chain::begin(41, &Digest::ZERO, b"non-stored opening", &[], None).unwrap()
    }

    fn make_session() -> OpeningSession {
        OpeningSession::new(chain(), Digest([7; 32]))
    }

    fn valid(session: &mut OpeningSession, plan: &NonStoredPlan, tree: &RuntimeTree) -> (OpeningSource, RuntimeTreeOpening, Vec<u64>) {
        session.advance_empty_round().unwrap();
        let list = session.draw_positions(2, 2).unwrap();
        session
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let source = session.authorize(&list, *tree.root()).unwrap();
        let opening = tree.open(&list).unwrap();
        (source, opening, list)
    }

    #[test]
    fn legal_current_opening_is_accepted_and_records_nodes() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (source, opening, _list) = valid(&mut session, &plan, &tree);
        session.verify_non_stored(&plan, &source, &opening).unwrap();
        assert!(!session.nodes.is_empty());
    }

    #[test]
    fn legal_sealed_predecessor_is_accepted() {
        let (tree, plan) = tree();
        let mut first = make_session();
        first.advance_empty_round().unwrap();
        let list = first.draw_positions(2, 2).unwrap();
        first
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let sealed = first.seal();
        let resumed = Chain::resume(sealed.label(), sealed.value());
        let mut second = OpeningSession::follow(&sealed, resumed, Digest([7; 32])).unwrap();
        let source = second.authorize(&list, *tree.root()).unwrap();
        let opening = tree.open(&list).unwrap();
        second.verify_non_stored(&plan, &source, &opening).unwrap();
    }

    #[test]
    fn unrelated_or_forged_source_is_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (source, opening, _list) = valid(&mut session, &plan, &tree);
        let mut other = make_session();
        assert!(other.verify_non_stored(&plan, &source, &opening).is_err());
        let forged = other.forge_source_for_test(vec![0], *tree.root(), *other.chain.value());
        assert!(other.verify_non_stored(&plan, &forged, &opening).is_err());
    }

    #[test]
    fn wrong_stored_run_config_salt_and_root_are_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (source, opening, _list) = valid(&mut session, &plan, &tree);

        let mut bad = plan.clone();
        bad.stored = true;
        assert!(session.verify_non_stored(&bad, &source, &opening).is_err());
        let mut bad = plan.clone();
        bad.tree.run = Digest([8; 32]);
        assert!(session.verify_non_stored(&bad, &source, &opening).is_err());
        let mut bad = plan.clone();
        bad.tree.configuration = Digest([6; 32]);
        assert!(session.verify_non_stored(&bad, &source, &opening).is_err());
        let mut bad = plan.clone();
        bad.plan_salt_bytes += 1;
        assert!(session.verify_non_stored(&bad, &source, &opening).is_err());

        let mut bad = opening.clone();
        bad.positions = vec![0];
        assert!(session.verify_non_stored(&plan, &source, &bad).is_err());
        let nodes_before = session.nodes.clone();
        let mut bad = opening.clone();
        bad.columns[0].symbols[0] = crate::pcs::request::field::fp_from_u64(999);
        assert!(session.verify_non_stored(&plan, &source, &bad).is_err());
        assert_eq!(session.nodes, nodes_before);
        session.verify_non_stored(&plan, &source, &opening).unwrap();
    }

    #[test]
    fn wrong_list_and_root_sources_are_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (_source, _opening, list) = valid(&mut session, &plan, &tree);
        assert!(session.authorize(&[0], *tree.root()).is_err());
        assert!(session.authorize(&list, Digest([5; 32])).is_err());
    }

    #[test]
    fn shared_node_conflict_and_failure_do_not_pollute_store() {
        let (tree, plan) = tree();
        let list = vec![0u64, 1];
        let opening = tree.open(&list).unwrap();
        let base = verify_runtime_opening_shared(
            &plan.tree,
            tree.root(),
            &list,
            &opening,
            &OpeningNodeStore::default(),
        )
        .unwrap();
        let before = base.clone();
        let mut conflict = base.clone();
        conflict.insert_for_test(*tree.root(), 0, list[0], Digest([111; 32]));
        assert!(verify_runtime_opening_shared(&plan.tree, tree.root(), &list, &opening, &conflict).is_err());
        assert_eq!(base, before);
        assert!(verify_runtime_opening(&plan.tree, tree.root(), &list, &opening).is_ok());
    }
    #[test]
    fn recorded_root_from_another_tree_is_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        session.advance_empty_round().unwrap();
        let list = session.draw_positions(2, 2).unwrap();
        let mut other_plan = plan.tree.clone();
        other_plan.run = Digest([10; 32]);
        let rows = vec![
            vec![Fp3::from_u64(11), Fp3::from_u64(12), Fp3::from_u64(13), Fp3::from_u64(14)],
            vec![Fp3::from_u64(15), Fp3::from_u64(16), Fp3::from_u64(17), Fp3::from_u64(18)],
        ];
        let salts: Vec<Vec<u8>> = (0..4)
            .map(|position| vec![(position as u8) + 21; LEAF_SALT_MIN_BYTES])
            .collect();
        let other_tree = RuntimeTree::commit(
            other_plan,
            &[RuntimePartRows {
                main: rows,
                mask: Vec::new(),
                main_deg: 1,
            }],
            &salts,
        )
        .unwrap();
        assert_ne!(*tree.root(), *other_tree.root());
        let mut other = make_session();
        other.advance_empty_round().unwrap();
        let other_list = other.draw_positions(2, 2).unwrap();
        other
            .receive_root(*other_tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let source = other.authorize(&other_list, *other_tree.root()).unwrap();
        let opening = other_tree.open(&other_list).unwrap();
        assert!(other.verify_non_stored(&plan, &source, &opening).is_err());
        // The original tree still verifies independently.
        session
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let source = session.authorize(&list, *tree.root()).unwrap();
        let opening = tree.open(&list).unwrap();
        session.verify_non_stored(&plan, &source, &opening).unwrap();
    }

}
