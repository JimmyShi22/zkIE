//! Isolated candidate for non-stored oracle openings and chain provenance.
//!
//! The stored-table path stays in `read`/`table`. This module models only the
//! chain-source, position-domain and plan checks of a non-stored opening: the
//! list and root must have been recorded by the current chain or a sealed
//! ancestor, their source identities and domains are checked separately, and
//! the runtime-tree opening must lead to that root. It is a fixture boundary:
//! chain values and roots are supplied by the isolated constructor, not
//! decoded or absorbed from real proof bytes, and the order of `draw` and
//! `receive_root` is not a protocol-order claim. It is not the general
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
    pub sender: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct KnownChain {
    marker: u64,
    lists: BTreeSet<(u32, Vec<u64>)>,
    roots: BTreeMap<Digest, ReceivedRoot>,
}

/// A drawn position list with the chain marker and domain it was drawn under.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningList {
    pub domain: u32,
    pub positions: Vec<u64>,
    marker: u64,
}

/// A sealed predecessor. Its fields are private; only `OpeningSession::seal`
/// creates one. Ancestors and node records are carried transitively.
#[derive(Clone, Debug)]
pub struct SealedChain {
    label: u64,
    value: Digest,
    known: KnownChain,
    prior: Vec<SealedChain>,
    nodes: BTreeMap<Digest, OpeningNodeStore>,
}

impl SealedChain {
    pub fn label(&self) -> u64 {
        self.label
    }

    pub fn value(&self) -> Digest {
        self.value
    }
}

/// A source issued for a recorded list/root. The list and root origins are
/// separate, so a list drawn by one chain can pair with a root received by an
/// ancestor. Fields are private so ordinary callers cannot label an arbitrary
/// list/root as coming from a chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningSource {
    session: u64,
    list_marker: u64,
    root_marker: u64,
    domain: u32,
    list: Vec<u64>,
    root: Digest,
    root_sender: u64,
}

/// The isolated verifier-side chain state. It records what `draw_positions`
/// drew and what `receive_root` received; these are fixture records, not a
/// decoded production proof or a protocol-order claim.
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
        let marker = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        Self {
            id: marker,
            chain,
            configuration,
            current_value,
            current: KnownChain {
                marker,
                ..KnownChain::default()
            },
            prior: Vec::new(),
            nodes: BTreeMap::new(),
        }
    }

    /// Close one empty algebraic round so this isolated chain has challenge
    /// words. The current chain value and marker are refreshed; records made
    /// before a round are refused rather than kept under the new value.
    pub fn advance_empty_round(&mut self) -> Result<()> {
        if !self.current.lists.is_empty() || !self.current.roots.is_empty() {
            return Err(Error::InvalidSessionState);
        }
        let prefix = self
            .chain
            .message()
            .prefix(crate::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.current_value = *self.chain.value();
        Ok(())
    }

    pub fn draw_positions(&mut self, count: usize, log_n: u32) -> Result<OpeningList> {
        let draws = draw_positions(&mut self.chain, count, log_n)?;
        let unique = unique_sorted(&draws);
        self.current
            .lists
            .insert((log_n, unique.clone()));
        Ok(OpeningList {
            domain: log_n,
            positions: unique,
            marker: self.current.marker,
        })
    }

    /// Record a root with the identity of the chain that received it. This is
    /// a fixture record; it does not absorb or decode a real chain message.
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
                sender: self.current.marker,
            },
        );
        Ok(())
    }

    pub fn seal(self) -> SealedChain {
        SealedChain {
            label: self.chain.label(),
            value: self.current_value,
            known: self.current,
            prior: self.prior,
            nodes: self.nodes,
        }
    }

    /// Start a session that may use the predecessor's recorded lists, roots
    /// and node records. The new chain must be resumed at the sealed value; the
    /// predecessor's own ancestors remain available.
    pub fn follow(
        prior: &SealedChain,
        chain: Chain,
        configuration: Digest,
    ) -> Result<Self> {
        if chain.label() != prior.label || *chain.value() != prior.value {
            return Err(Error::StatementMismatch);
        }
        let mut session = Self::new(chain, configuration);
        let mut ancestors = prior.prior.clone();
        ancestors.push(prior.clone());
        session.prior = ancestors;
        session.nodes = prior.nodes.clone();
        Ok(session)
    }

    fn all_chains(&self) -> impl Iterator<Item = &KnownChain> {
        std::iter::once(&self.current).chain(self.prior.iter().map(|prior| &prior.known))
    }

    /// Issue a source only when the list and the root are each recorded, not
    /// necessarily by the same chain. A list drawn by the current chain can
    /// pair with a root received by an ancestor or the other way round.
    pub fn authorize(&self, list: &OpeningList, root: Digest) -> Result<OpeningSource> {
        let mut unique = list.positions.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique != list.positions || list.positions.is_empty() {
            return Err(Error::InvalidIndex);
        }
        let list_chain = self
            .all_chains()
            .find(|chain| {
                chain.marker == list.marker
                    && chain.lists.contains(&(list.domain, list.positions.clone()))
            })
            .ok_or(Error::StatementMismatch)?;
        let root_chain = self
            .all_chains()
            .find(|chain| chain.roots.contains_key(&root))
            .ok_or(Error::StatementMismatch)?;
        let binding = root_chain.roots.get(&root).ok_or(Error::StatementMismatch)?;
        Ok(OpeningSource {
            session: self.id,
            list_marker: list_chain.marker,
            root_marker: root_chain.marker,
            domain: list.domain,
            list: list.positions.clone(),
            root,
            root_sender: binding.sender,
        })
    }

    fn known_source(&self, source: &OpeningSource) -> Result<&ReceivedRoot> {
        if source.session != self.id {
            return Err(Error::StatementMismatch);
        }
        let list_chain = self
            .all_chains()
            .find(|chain| {
                chain.marker == source.list_marker
                    && chain.lists.contains(&(source.domain, source.list.clone()))
            })
            .ok_or(Error::StatementMismatch)?;
        let _ = list_chain;
        let root_chain = self
            .all_chains()
            .find(|chain| chain.marker == source.root_marker)
            .ok_or(Error::StatementMismatch)?;
        let binding = root_chain
            .roots
            .get(&source.root)
            .ok_or(Error::StatementMismatch)?;
        if binding.sender != source.root_sender {
            return Err(Error::StatementMismatch);
        }
        Ok(binding)
    }

    #[cfg(test)]
    pub fn forge_source_for_test(
        &self,
        list: Vec<u64>,
        root: Digest,
        list_marker: u64,
        root_marker: u64,
        domain: u32,
    ) -> OpeningSource {
        OpeningSource {
            session: self.id,
            list_marker,
            root_marker,
            domain,
            list,
            root,
            root_sender: root_marker,
        }
    }

    /// Verify the isolated non-stored opening subset. The list domain, root
    /// source, plan salt/run/configuration and actual runtime opening are
    /// checked before the shared node store is committed.
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
        if source.domain != plan.tree.log_n {
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
    use crate::pcs::request::field::Fp3;
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::runtime_tree::{
        verify_runtime_opening, RuntimePartRows, RuntimeTree, RuntimeTreePart,
    };

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

    fn valid(
        session: &mut OpeningSession,
        plan: &NonStoredPlan,
        tree: &RuntimeTree,
    ) -> (OpeningSource, RuntimeTreeOpening, OpeningList) {
        session.advance_empty_round().unwrap();
        let list = session.draw_positions(2, 2).unwrap();
        session
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let source = session.authorize(&list, *tree.root()).unwrap();
        let opening = tree.open(&list.positions).unwrap();
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
    fn legal_sealed_predecessor_and_mixed_sources_are_accepted() {
        let (tree, plan) = tree();
        let mut first = make_session();
        first.advance_empty_round().unwrap();
        let first_list = first.draw_positions(2, 2).unwrap();
        first
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let sealed_first = first.seal();
        let resumed = Chain::resume(sealed_first.label(), sealed_first.value());
        let mut second = OpeningSession::follow(&sealed_first, resumed, Digest([7; 32])).unwrap();
        second.advance_empty_round().unwrap();
        let second_list = second.draw_positions(1, 2).unwrap();
        let source_from_prior_list = second
            .authorize(&first_list, *tree.root())
            .unwrap();
        let opening_prior = tree.open(&first_list.positions).unwrap();
        second
            .verify_non_stored(&plan, &source_from_prior_list, &opening_prior)
            .unwrap();
        let source_mixed = second.authorize(&second_list, *tree.root()).unwrap();
        let opening_second = tree.open(&second_list.positions).unwrap();
        second
            .verify_non_stored(&plan, &source_mixed, &opening_second)
            .unwrap();
    }

    #[test]
    fn sealed_value_must_be_the_advanced_chain_value() {
        let mut first = make_session();
        let old = first.current_value;
        first.advance_empty_round().unwrap();
        let new = first.current_value;
        assert_ne!(old, new);
        let sealed = first.seal();
        assert_eq!(sealed.value(), new);
        assert!(OpeningSession::follow(&sealed, Chain::resume(sealed.label(), old), Digest([7; 32])).is_err());
        assert!(OpeningSession::follow(&sealed, Chain::resume(sealed.label(), new), Digest([7; 32])).is_ok());
    }

    #[test]
    fn unrelated_or_forged_source_is_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (source, opening, _list) = valid(&mut session, &plan, &tree);
        let mut other = make_session();
        assert!(other.verify_non_stored(&plan, &source, &opening).is_err());
        let forged = other.forge_source_for_test(
            vec![0],
            *tree.root(),
            other.current.marker,
            other.current.marker,
            2,
        );
        assert!(other.verify_non_stored(&plan, &forged, &opening).is_err());
    }

    #[test]
    fn identity_and_domain_mismatches_are_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (source, opening, list) = valid(&mut session, &plan, &tree);
        let mut same_value = OpeningSession::new(
            Chain::resume(41, *session.chain.value()),
            Digest([7; 32]),
        );
        assert!(same_value.verify_non_stored(&plan, &source, &opening).is_err());
        let wrong_domain = OpeningList {
            domain: 1,
            positions: list.positions.clone(),
            marker: list.marker,
        };
        assert!(session.authorize(&wrong_domain, *tree.root()).is_err());
        let wrong_list_marker = session.forge_source_for_test(
            source.list.clone(),
            source.root,
            source.list_marker + 1,
            source.root_marker,
            source.domain,
        );
        assert!(session.verify_non_stored(&plan, &wrong_list_marker, &opening).is_err());
        assert!(session.verify_non_stored(&plan, &source, &opening).is_ok());
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
        let short = OpeningList {
            domain: 2,
            positions: vec![0],
            marker: list.marker,
        };
        assert!(session.authorize(&short, *tree.root()).is_err());
        assert!(session.authorize(&list, Digest([5; 32])).is_err());
    }

    #[test]
    fn shared_node_conflict_survives_follow_and_failure_does_not_pollute() {
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

        let mut first = make_session();
        first.advance_empty_round().unwrap();
        let list = first.draw_positions(2, 2).unwrap();
        first
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let source = first.authorize(&list, *tree.root()).unwrap();
        let opening = tree.open(&list.positions).unwrap();
        first.verify_non_stored(&plan, &source, &opening).unwrap();
        let sealed = first.seal();
        let resumed = Chain::resume(sealed.label(), sealed.value());
        let mut second = OpeningSession::follow(&sealed, resumed, Digest([7; 32])).unwrap();
        let mut conflict = OpeningNodeStore::default();
        conflict.insert_for_test(*tree.root(), 0, list.positions[0], Digest([111; 32]));
        second.nodes.insert(*tree.root(), conflict);
        assert!(second.verify_non_stored(&plan, &source, &opening).is_err());
        assert!(verify_runtime_opening(&plan.tree, tree.root(), &list.positions, &opening).is_ok());
    }
    #[test]
    fn multi_level_ancestors_and_mixed_roots_are_accepted() {
        let (tree, plan) = tree();
        let mut a = make_session();
        a.advance_empty_round().unwrap();
        let list_a = a.draw_positions(2, 2).unwrap();
        a.receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let sealed_a = a.seal();

        let mut b = OpeningSession::follow(
            &sealed_a,
            Chain::resume(sealed_a.label(), sealed_a.value()),
            Digest([7; 32]),
        )
        .unwrap();
        b.advance_empty_round().unwrap();
        let list_b = b.draw_positions(1, 2).unwrap();
        let sealed_b = b.seal();

        let mut c = OpeningSession::follow(
            &sealed_b,
            Chain::resume(sealed_b.label(), sealed_b.value()),
            Digest([7; 32]),
        )
        .unwrap();
        c.advance_empty_round().unwrap();
        let list_c = c.draw_positions(1, 2).unwrap();

        let from_a = c.authorize(&list_a, *tree.root()).unwrap();
        let opening_a = tree.open(&list_a.positions).unwrap();
        c.verify_non_stored(&plan, &from_a, &opening_a).unwrap();

        let mixed = c.authorize(&list_c, *tree.root()).unwrap();
        let opening_c = tree.open(&list_c.positions).unwrap();
        c.verify_non_stored(&plan, &mixed, &opening_c).unwrap();

        let from_b = c.authorize(&list_b, *tree.root()).unwrap();
        let opening_b = tree.open(&list_b.positions).unwrap();
        c.verify_non_stored(&plan, &from_b, &opening_b).unwrap();
    }

    #[test]
    fn root_sender_identity_and_fixture_order_are_explicit() {
        let (tree, plan) = tree();
        let mut session = make_session();
        session.advance_empty_round().unwrap();
        // The fixture records the root before drawing; this is not a claim
        // about the protocol order, which differs between foldeq and CS.
        session
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let list = session.draw_positions(2, 2).unwrap();
        let source = session.authorize(&list, *tree.root()).unwrap();
        let opening = tree.open(&list.positions).unwrap();
        session.verify_non_stored(&plan, &source, &opening).unwrap();
        let mut bad = source.clone();
        bad.root_sender += 1;
        assert!(session.verify_non_stored(&plan, &bad, &opening).is_err());
    }

}
