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
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::sync::atomic::AtomicBool;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::Chain;
use crate::pcs::request::read::{draw_positions, unique_sorted};
use crate::pcs::request::runtime_tree::{
    verify_runtime_opening_shared, OpeningNodeStore, RuntimeTreeOpening, RuntimeTreePlan,
};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
#[cfg(test)]
static COMMIT_DELAY: AtomicBool = AtomicBool::new(false);

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

/// A non-forgeable handle for one received-root event. Only
/// `OpeningSession::receive_root` creates it; it carries the receiving chain
/// identity so the same root value from another sender cannot be substituted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootHandle {
    root: Digest,
    sender: u64,
    marker: u64,
}

/// A sealed predecessor snapshot. Its fields are private; only
/// `OpeningSession::seal` creates one. This is an isolated provenance snapshot,
/// not the reference seal-close round. Ancestors and node records are carried
/// transitively.
#[derive(Clone, Debug)]
pub struct SealedChain {
    label: u64,
    value: Digest,
    known: KnownChain,
    ancestors: Vec<KnownChain>,
    nodes: Arc<Mutex<BTreeMap<Digest, OpeningNodeStore>>>,
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
///
/// This type is deliberately not `Clone`: one marker/session identity cannot be
/// copied into two private record sets. A `SealedChain` may be followed by
/// siblings; their node records use one shared lock so updates are visible
/// across the siblings, but this remains one isolated verdict model.
pub struct OpeningSession {
    id: u64,
    chain: Chain,
    configuration: Digest,
    current_value: Digest,
    current: KnownChain,
    ancestors: Vec<KnownChain>,
    nodes: Arc<Mutex<BTreeMap<Digest, OpeningNodeStore>>>,
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
            ancestors: Vec::new(),
            nodes: Arc::new(Mutex::new(BTreeMap::new())),
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

    #[cfg(test)]
    pub fn record_list_for_test(&mut self, domain: u32, positions: Vec<u64>) -> OpeningList {
        let mut unique = positions.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique, positions);
        self.current.lists.insert((domain, positions.clone()));
        OpeningList {
            domain,
            positions,
            marker: self.current.marker,
        }
    }

    /// Record a root with the identity of the chain that received it. This is
    /// a fixture record; it does not absorb or decode a real chain message.
    pub fn receive_root(
        &mut self,
        root: Digest,
        run: Digest,
        configuration: Digest,
    ) -> Result<RootHandle> {
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
        Ok(RootHandle {
            root,
            sender: self.current.marker,
            marker: self.current.marker,
        })
    }

    /// Return the isolated provenance snapshot. This does not perform the
    /// reference seal-close round or emit a seal word: it records the current
    /// fixture value, ancestors and shared node store for continuation. A
    /// same-label/resumed chain is an isolated modeling device, not a complete
    /// sealed-chain proof.
    pub fn seal(self) -> SealedChain {
        SealedChain {
            label: self.chain.label(),
            value: self.current_value,
            known: self.current,
            ancestors: self.ancestors,
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
        let mut ancestors = prior.ancestors.clone();
        ancestors.push(prior.known.clone());
        session.ancestors = ancestors;
        session.nodes = Arc::clone(&prior.nodes);
        Ok(session)
    }

    fn all_chains(&self) -> impl Iterator<Item = &KnownChain> {
        std::iter::once(&self.current).chain(self.ancestors.iter())
    }

    /// Issue a source only when the list and the root are each recorded, not
    /// necessarily by the same chain. A list drawn by the current chain can
    /// pair with a root received by an ancestor or the other way round.
    pub fn authorize(&self, list: &OpeningList, root: &RootHandle) -> Result<OpeningSource> {
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
            .find(|chain| chain.marker == root.marker && chain.roots.contains_key(&root.root))
            .ok_or(Error::StatementMismatch)?;
        let binding = root_chain
            .roots
            .get(&root.root)
            .ok_or(Error::StatementMismatch)?;
        if binding.sender != root.sender {
            return Err(Error::StatementMismatch);
        }
        Ok(OpeningSource {
            session: self.id,
            list_marker: list_chain.marker,
            root_marker: root_chain.marker,
            domain: list.domain,
            list: list.positions.clone(),
            root: root.root,
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
        // The read, conflict check, merge and commit of one root's node
        // record are one critical section. Keeping the lock across the walk
        // prevents two sibling chains from overwriting each other's accepted
        // vertices; a failed walk leaves the committed map unchanged.
        let mut nodes = self
            .nodes
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        let prior = nodes
            .get(&source.root)
            .cloned()
            .unwrap_or_default();
        #[cfg(test)]
        if COMMIT_DELAY.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let staged = verify_runtime_opening_shared(
            &plan.tree,
            &source.root,
            &source.list,
            opening,
            &prior,
        )?;
        nodes.insert(source.root, staged);
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

    fn nodes_snapshot(session: &OpeningSession) -> BTreeMap<Digest, OpeningNodeStore> {
        session.nodes.lock().unwrap().clone()
    }

    fn valid(
        session: &mut OpeningSession,
        plan: &NonStoredPlan,
        tree: &RuntimeTree,
    ) -> (RootHandle, OpeningSource, RuntimeTreeOpening, OpeningList) {
        session.advance_empty_round().unwrap();
        let list = session.draw_positions(2, 2).unwrap();
        let root = session
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let source = session.authorize(&list, &root).unwrap();
        let opening = tree.open(&list.positions).unwrap();
        (root, source, opening, list)
    }

    #[test]
    fn legal_current_opening_is_accepted_and_records_nodes() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (_root, source, opening, _list) = valid(&mut session, &plan, &tree);
        session.verify_non_stored(&plan, &source, &opening).unwrap();
        assert!(!nodes_snapshot(&session).is_empty());
    }

    #[test]
    fn legal_sealed_predecessor_and_mixed_sources_are_accepted() {
        let (tree, plan) = tree();
        let mut first = make_session();
        first.advance_empty_round().unwrap();
        let first_list = first.draw_positions(2, 2).unwrap();
        let first_root = first
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let sealed_first = first.seal();
        let resumed = Chain::resume(sealed_first.label(), sealed_first.value());
        let mut second = OpeningSession::follow(&sealed_first, resumed, Digest([7; 32])).unwrap();
        second.advance_empty_round().unwrap();
        let second_list = second.draw_positions(1, 2).unwrap();
        let source_from_prior_list = second.authorize(&first_list, &first_root).unwrap();
        let opening_prior = tree.open(&first_list.positions).unwrap();
        second
            .verify_non_stored(&plan, &source_from_prior_list, &opening_prior)
            .unwrap();
        let source_mixed = second.authorize(&second_list, &first_root).unwrap();
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
        let (_root, source, opening, _list) = valid(&mut session, &plan, &tree);
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
        let (root, source, opening, list) = valid(&mut session, &plan, &tree);
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
        assert!(session.authorize(&wrong_domain, &root).is_err());
        let wrong_list_marker = session.forge_source_for_test(
            source.list.clone(),
            source.root,
            source.list_marker + 1,
            source.root_marker,
            source.domain,
        );
        assert!(session.verify_non_stored(&plan, &wrong_list_marker, &opening).is_err());
        let mut wrong_sender = source.clone();
        wrong_sender.root_sender += 1;
        assert!(session.verify_non_stored(&plan, &wrong_sender, &opening).is_err());
        assert!(session.verify_non_stored(&plan, &source, &opening).is_ok());
    }

    #[test]
    fn wrong_stored_run_config_salt_and_root_are_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (_root, source, opening, _list) = valid(&mut session, &plan, &tree);

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
        let nodes_before = nodes_snapshot(&session);
        let mut bad = opening.clone();
        bad.columns[0].symbols[0] = crate::pcs::request::field::fp_from_u64(999);
        assert!(session.verify_non_stored(&plan, &source, &bad).is_err());
        assert_eq!(nodes_snapshot(&session), nodes_before);
        session.verify_non_stored(&plan, &source, &opening).unwrap();
    }

    #[test]
    fn wrong_list_and_root_handles_are_rejected() {
        let (tree, plan) = tree();
        let mut session = make_session();
        let (_root, _source, _opening, list) = valid(&mut session, &plan, &tree);
        let short = OpeningList {
            domain: 2,
            positions: vec![0],
            marker: list.marker,
        };
        let root = session
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap_err();
        assert!(root.to_string().is_empty() || true); // no-op; keep the duplicate-root rejection explicit below
        let mut other = make_session();
        other.advance_empty_round().unwrap();
        let other_handle = other
            .receive_root(Digest([5; 32]), plan.tree.run, plan.tree.configuration)
            .unwrap();
        assert!(session.authorize(&short, &other_handle).is_err());
    }

    #[test]
    fn shared_node_conflict_survives_follow_and_failure_does_not_pollute() {
        let (tree, plan) = tree();
        let mut first = make_session();
        let (root, source, opening, list) = valid(&mut first, &plan, &tree);
        first.verify_non_stored(&plan, &source, &opening).unwrap();
        let sealed = first.seal();
        let resumed = Chain::resume(sealed.label(), sealed.value());
        let mut second = OpeningSession::follow(&sealed, resumed, Digest([7; 32])).unwrap();
        let inherited_source = second.authorize(&list, &root).unwrap();
        second.verify_non_stored(&plan, &inherited_source, &opening).unwrap();
        let before = nodes_snapshot(&second);
        let mut conflict = before.get(tree.root()).cloned().unwrap_or_default();
        conflict.insert_for_test(*tree.root(), 0, list.positions[0], Digest([111; 32]));
        second.nodes.lock().unwrap().insert(*tree.root(), conflict);
        let before_failure = nodes_snapshot(&second);
        assert!(second.verify_non_stored(&plan, &inherited_source, &opening).is_err());
        assert_eq!(nodes_snapshot(&second), before_failure);
        assert!(!before.is_empty());
        assert!(verify_runtime_opening(&plan.tree, tree.root(), &list.positions, &opening).is_ok());
    }

    #[test]
    fn sibling_chains_share_nodes_and_clone_is_not_available() {
        let (tree, plan) = tree();
        let mut first = make_session();
        let (root, _source, _opening_first, _list_first) = valid(&mut first, &plan, &tree);
        let sealed = first.seal();
        let mut left = OpeningSession::follow(
            &sealed,
            Chain::resume(sealed.label(), sealed.value()),
            Digest([7; 32]),
        )
        .unwrap();
        let mut right = OpeningSession::follow(
            &sealed,
            Chain::resume(sealed.label(), sealed.value()),
            Digest([7; 32]),
        )
        .unwrap();
        let list_left = left.record_list_for_test(2, vec![0]);
        let list_right = right.record_list_for_test(2, vec![2]);
        let source_left = left.authorize(&list_left, &root).unwrap();
        let source_right = right.authorize(&list_right, &root).unwrap();
        let opening_left = tree.open(&list_left.positions).unwrap();
        let opening_right = tree.open(&list_right.positions).unwrap();

        let after_first = OpeningNodeStore::default();
        let after_left = verify_runtime_opening_shared(
            &plan.tree,
            tree.root(),
            &list_left.positions,
            &opening_left,
            &after_first,
        )
        .unwrap();

        left.verify_non_stored(&plan, &source_left, &opening_left).unwrap();
        assert_eq!(nodes_snapshot(&right).get(tree.root()), Some(&after_left));
        right.verify_non_stored(&plan, &source_right, &opening_right).unwrap();
        let union = verify_runtime_opening_shared(
            &plan.tree,
            tree.root(),
            &list_right.positions,
            &opening_right,
            &after_left,
        )
        .unwrap();
        assert!(after_left.len() < union.len());
        assert_eq!(nodes_snapshot(&left).get(tree.root()), Some(&union));
        assert_eq!(nodes_snapshot(&right).get(tree.root()), Some(&union));
    }
    #[test]
    fn multi_level_ancestors_and_mixed_roots_are_accepted() {
        let (tree, plan) = tree();
        let mut a = make_session();
        a.advance_empty_round().unwrap();
        let list_a = a.draw_positions(2, 2).unwrap();
        let root_a = a
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let sealed_a = a.seal();
        assert_eq!(sealed_a.ancestors.len(), 0);

        let mut b = OpeningSession::follow(
            &sealed_a,
            Chain::resume(sealed_a.label(), sealed_a.value()),
            Digest([7; 32]),
        )
        .unwrap();
        b.advance_empty_round().unwrap();
        let list_b = b.draw_positions(1, 2).unwrap();
        let sealed_b = b.seal();
        assert_eq!(sealed_b.ancestors.len(), 1);

        let mut c = OpeningSession::follow(
            &sealed_b,
            Chain::resume(sealed_b.label(), sealed_b.value()),
            Digest([7; 32]),
        )
        .unwrap();
        c.advance_empty_round().unwrap();
        let list_c = c.draw_positions(1, 2).unwrap();
        assert_eq!(c.ancestors.len(), 2);

        let from_a = c.authorize(&list_a, &root_a).unwrap();
        let opening_a = tree.open(&list_a.positions).unwrap();
        c.verify_non_stored(&plan, &from_a, &opening_a).unwrap();

        let mixed = c.authorize(&list_c, &root_a).unwrap();
        let opening_c = tree.open(&list_c.positions).unwrap();
        c.verify_non_stored(&plan, &mixed, &opening_c).unwrap();

        let from_b = c.authorize(&list_b, &root_a).unwrap();
        let opening_b = tree.open(&list_b.positions).unwrap();
        c.verify_non_stored(&plan, &from_b, &opening_b).unwrap();
    }

    #[test]
    fn root_sender_identity_and_fixture_order_are_explicit() {
        let (tree, plan) = tree();
        let mut session = make_session();
        session.advance_empty_round().unwrap();
        // Fixture order: root before draw. This is not a protocol-order claim.
        let root = session
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let list = session.draw_positions(2, 2).unwrap();
        let source = session.authorize(&list, &root).unwrap();
        let opening = tree.open(&list.positions).unwrap();
        session.verify_non_stored(&plan, &source, &opening).unwrap();
        let mut bad = source.clone();
        bad.root_sender += 1;
        assert!(session.verify_non_stored(&plan, &bad, &opening).is_err());

        let mut other = make_session();
        other.advance_empty_round().unwrap();
        let other_root = other
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        assert!(session.authorize(&list, &other_root).is_err());
    }
    #[test]
    fn concurrent_siblings_merge_valid_openings_atomically() {
        use std::sync::Barrier;
        let (tree, plan) = tree();
        let mut first = make_session();
        first.advance_empty_round().unwrap();
        let _first_list = first.draw_positions(1, 2).unwrap();
        let root = first
            .receive_root(*tree.root(), plan.tree.run, plan.tree.configuration)
            .unwrap();
        let sealed = first.seal();
        let mut left = OpeningSession::follow(
            &sealed,
            Chain::resume(sealed.label(), sealed.value()),
            Digest([7; 32]),
        )
        .unwrap();
        let mut right = OpeningSession::follow(
            &sealed,
            Chain::resume(sealed.label(), sealed.value()),
            Digest([7; 32]),
        )
        .unwrap();
        let left_list = left.record_list_for_test(2, vec![0]);
        let right_list = right.record_list_for_test(2, vec![2]);
        let left_source = left.authorize(&left_list, &root).unwrap();
        let right_source = right.authorize(&right_list, &root).unwrap();
        let left_opening = tree.open(&left_list.positions).unwrap();
        let right_opening = tree.open(&right_list.positions).unwrap();

        let barrier = Barrier::new(2);
        let plan_ref = &plan;
        let left_check = left_opening.clone();
        let right_check = right_opening.clone();
        let left_ref = &mut left;
        let right_ref = &mut right;
        COMMIT_DELAY.store(true, Ordering::Relaxed);
        let (left_ok, right_ok) = std::thread::scope(|scope| {
            let left_barrier = &barrier;
            let left_handle = scope.spawn(move || {
                left_barrier.wait();
                left_ref
                    .verify_non_stored(plan_ref, &left_source, &left_check)
                    .is_ok()
            });
            let right_barrier = &barrier;
            let right_handle = scope.spawn(move || {
                right_barrier.wait();
                right_ref
                    .verify_non_stored(plan_ref, &right_source, &right_check)
                    .is_ok()
            });
            (
                left_handle.join().unwrap(),
                right_handle.join().unwrap(),
            )
        });
        COMMIT_DELAY.store(false, Ordering::Relaxed);
        assert!(left_ok && right_ok);
        let left_store = verify_runtime_opening_shared(
            &plan.tree,
            tree.root(),
            &left_list.positions,
            &left_opening,
            &OpeningNodeStore::default(),
        )
        .unwrap();
        let union = verify_runtime_opening_shared(
            &plan.tree,
            tree.root(),
            &right_list.positions,
            &right_opening,
            &left_store,
        )
        .unwrap();
        let snapshot = nodes_snapshot(&left);
        assert!(left_store.len() < union.len());
        assert_eq!(snapshot.get(tree.root()), Some(&union));
    }

}
