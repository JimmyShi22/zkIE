//! Isolated candidate for per-tree opening counts and position lists.
//!
//! This models the tree bookkeeping at the candidate level: a
//! non-empty draw list with a sorted set, a short list for a shorter tree, one
//! list count per opening, positions accumulated over all lists (repeats
//! across lists count), whole-tree opens, one-list restrictions, per-part
//! main/mask hiding limits checked before output, and release only after every
//! part has been read and is not held. It is not production wiring: the tree
//! roots, groups and opening bytes are caller fixtures, and no real proof byte
//! decoder, table test, IOPP/CS/BASE stage or persistent tree book is
//! connected. Empty ids are an explicit no-op; an empty draw list is invalid
//! and never becomes a successful opening.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use zkie_core::error::{Error, Result};

/// A non-empty position list. The sorted set is the opening list; the draws
/// preserve the original order and repeats.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionList {
    log_n: u32,
    draws: Vec<u64>,
    set: Vec<u64>,
}

impl PositionList {
    /// A draw list must be non-empty; each draw must fit `2^log_n`. Repeats
    /// are removed from the set but retained in the draw order.
    pub fn from_draws(draws: Vec<u64>, log_n: u32) -> Result<Self> {
        if log_n >= 64 || draws.is_empty() {
            return Err(Error::InvalidIndex);
        }
        let mut set = draws.clone();
        set.sort_unstable();
        set.dedup();
        if set.last().map_or(true, |last| *last >> log_n != 0) {
            return Err(Error::InvalidIndex);
        }
        Ok(Self { log_n, draws, set })
    }

    pub fn log_n(&self) -> u32 {
        self.log_n
    }

    pub fn size(&self) -> usize {
        self.set.len()
    }

    pub fn draws(&self) -> &[u64] {
        &self.draws
    }

    pub fn all(&self) -> &[u64] {
        &self.set
    }

    pub fn index_of(&self, position: u64) -> Result<usize> {
        self.set
            .binary_search(&position)
            .map_err(|_| Error::InvalidIndex)
    }

    /// The list read for a shorter tree: each draw is reduced modulo
    /// `2^log_short`, then sorted and deduplicated. It is still a non-empty
    /// list because the source draws are non-empty.
    pub fn short_list(&self, log_short: u32) -> Result<Self> {
        if log_short > self.log_n {
            return Err(Error::InvalidIndex);
        }
        let mask = if log_short == 0 {
            0
        } else {
            (1u64 << log_short) - 1
        };
        let draws = self.draws.iter().map(|position| position & mask).collect();
        Self::from_draws(draws, log_short)
    }
}

#[derive(Clone, Debug)]
pub struct TreeSpec {
    pub log_n: u32,
    pub parts: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct OracleSpec {
    pub id: u64,
    pub groups: Vec<u32>,
    pub answers: u32,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub main_hiding: u64,
    pub mask_hiding: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeCounts {
    pub lists: u64,
    pub positions: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenedTree {
    pub tree: usize,
    pub positions: u64,
    pub list: PositionList,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Opened {
    pub trees: Vec<OpenedTree>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOpenedTree {
    pub tree: usize,
    pub list: PositionList,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadOpened {
    pub trees: Vec<ReadOpenedTree>,
}

struct TreeState {
    log_n: u32,
    parts: Vec<u64>,
    one_list: bool,
    counts: Arc<Mutex<TreeCounts>>,
}

/// A candidate tree book. It is move-only conceptually: the counts are shared
/// only through `adopt`, which is the cross-chain adoption path, not through a
/// general clone.
pub struct TreeBook {
    trees: Vec<TreeState>,
    oracles: BTreeMap<u64, OracleSpec>,
    place: BTreeMap<u64, (usize, usize)>,
    held: BTreeSet<u64>,
}

impl TreeBook {
    pub fn new(trees: Vec<TreeSpec>, oracles: Vec<OracleSpec>) -> Result<Self> {
        if trees.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let mut place = BTreeMap::new();
        let mut specs = BTreeMap::new();
        for (tree_index, tree) in trees.iter().enumerate() {
            if tree.log_n >= 64 || tree.parts.is_empty() {
                return Err(Error::InvalidConfig);
            }
            for (part_index, id) in tree.parts.iter().enumerate() {
                if place.insert(*id, (tree_index, part_index)).is_some() {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        for oracle in oracles {
            let id = oracle.id;
            if specs.insert(id, oracle).is_some() || !place.contains_key(&id) {
                return Err(Error::InvalidConfig);
            }
        }
        if specs.len() != place.len() || specs.keys().any(|id| !place.contains_key(id)) {
            return Err(Error::InvalidConfig);
        }
        let states = trees
            .into_iter()
            .map(|tree| {
                let one_list = tree
                    .parts
                    .iter()
                    .any(|id| specs.get(id).map_or(true, |spec| spec.answers == 0));
                TreeState {
                    log_n: tree.log_n,
                    parts: tree.parts,
                    one_list,
                    counts: Arc::new(Mutex::new(TreeCounts::default())),
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            trees: states,
            oracles: specs,
            place,
            held: BTreeSet::new(),
        })
    }

    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    pub fn counts(&self, tree: usize) -> Result<TreeCounts> {
        let state = self.trees.get(tree).ok_or(Error::InvalidIndex)?;
        let counts = state.counts.lock().map_err(|_| Error::InvalidSessionState)?;
        Ok(*counts)
    }

    pub fn set_held(&mut self, id: u64, held: bool) -> Result<()> {
        if !self.place.contains_key(&id) {
            return Err(Error::InvalidIndex);
        }
        if held {
            self.held.insert(id);
        } else {
            self.held.remove(&id);
        }
        Ok(())
    }

    pub fn is_held(&self, id: u64) -> bool {
        self.held.contains(&id)
    }

    fn tree_of(&self, id: u64) -> Result<usize> {
        self.place
            .get(&id)
            .map(|(tree, _)| *tree)
            .ok_or(Error::InvalidIndex)
    }

    fn trees_of(&self, ids: &[u64]) -> Result<Vec<usize>> {
        let mut trees = BTreeSet::new();
        for id in ids {
            trees.insert(self.tree_of(*id)?);
        }
        Ok(trees.into_iter().collect())
    }

    /// Every part of every touched tree must be named. Empty ids are an
    /// explicit no-op handled by the callers.
    fn whole_trees(&self, ids: &[u64]) -> Result<bool> {
        for tree in self.trees_of(ids)? {
            for part in &self.trees[tree].parts {
                if !ids.contains(part) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// One opening of the tree: count the list, enforce the group agreement and
    /// the one-list restriction. No positions are counted here; that happens
    /// only when an actual opening is produced.
    pub fn count_opening(&self, tree: usize) -> Result<()> {
        let state = self.trees.get(tree).ok_or(Error::InvalidIndex)?;
        let mut groups: Option<Vec<u32>> = None;
        for part in &state.parts {
            let spec = self.oracles.get(part).ok_or(Error::InvalidConfig)?;
            if spec.answers == 0 {
                continue;
            }
            match &groups {
                Some(seen) if *seen != spec.groups => return Err(Error::StatementMismatch),
                Some(_) => {}
                None => groups = Some(spec.groups.clone()),
            }
        }
        let mut counts = state.counts.lock().map_err(|_| Error::InvalidSessionState)?;
        if state.one_list
            && (groups.as_ref().map_or(false, |groups| groups.len() > 1) || counts.lists != 0)
        {
            return Err(Error::StatementMismatch);
        }
        counts.lists = counts.lists.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok(())
    }

    /// The list at which tree `tree` reads for a caller's list `at`: its own
    /// list when the lengths agree, otherwise the shorter list.
    pub fn list_for(&self, tree: usize, at: &PositionList) -> Result<PositionList> {
        let state = self.trees.get(tree).ok_or(Error::InvalidIndex)?;
        if at.log_n() == state.log_n {
            Ok(at.clone())
        } else if state.log_n < at.log_n() {
            at.short_list(state.log_n)
        } else {
            Err(Error::InvalidIndex)
        }
    }

    /// Prover-side opening count: all touched trees are opened, positions are
    /// added by the actual own-list size (a short list for a shorter tree), and
    /// every part's main/mask hiding limit is checked before the opening is
    /// returned. An error leaves the raised counts in place (fail closed).
    pub fn open_oracles(&self, ids: &[u64], at: &PositionList) -> Result<Opened> {
        if ids.is_empty() {
            return Ok(Opened::default());
        }
        if !self.whole_trees(ids)? {
            return Err(Error::StatementMismatch);
        }
        let trees = self.trees_of(ids)?;
        let mut opened = Vec::with_capacity(trees.len());
        for tree in trees {
            self.count_opening(tree)?;
            let own = self.list_for(tree, at)?;
            let added = u64::try_from(own.size()).map_err(|_| Error::ResourceLimit)?;
            let positions = {
                let state = &self.trees[tree];
                let mut counts = state.counts.lock().map_err(|_| Error::InvalidSessionState)?;
                counts.positions = counts.positions.checked_add(added).ok_or(Error::ResourceLimit)?;
                counts.positions
            };
            for part in &self.trees[tree].parts {
                let spec = self.oracles.get(part).ok_or(Error::InvalidConfig)?;
                if spec.main_rows > 0 && positions > spec.main_hiding {
                    return Err(Error::ResourceLimit);
                }
                if spec.mask_rows > 0 && positions > spec.mask_hiding {
                    return Err(Error::ResourceLimit);
                }
            }
            opened.push(OpenedTree {
                tree,
                positions,
                list: own,
            });
        }
        Ok(Opened { trees: opened })
    }

    /// Verifier-side opening: validate and count the list, but do not maintain
    /// the prover's private positions/hiding ledger.
    pub fn read_opened(&self, ids: &[u64], at: &PositionList) -> Result<ReadOpened> {
        if ids.is_empty() {
            return Ok(ReadOpened::default());
        }
        if !self.whole_trees(ids)? {
            return Err(Error::StatementMismatch);
        }
        let trees = self.trees_of(ids)?;
        let mut opened = Vec::with_capacity(trees.len());
        for tree in trees {
            self.count_opening(tree)?;
            opened.push(ReadOpenedTree {
                tree,
                list: self.list_for(tree, at)?,
            });
        }
        Ok(ReadOpened { trees: opened })
    }

    /// A tree may be released only after every part has been read (its list
    /// count reaches the group count, at least one) and no part is held.
    pub fn release_oracles(&self, ids: &[u64]) -> Result<Vec<usize>> {
        let mut released = Vec::new();
        for tree in self.trees_of(ids)? {
            let counts = self.counts(tree)?;
            let mut done = true;
            for part in &self.trees[tree].parts {
                let spec = self.oracles.get(part).ok_or(Error::InvalidConfig)?;
                let groups = spec.groups.len().max(1) as u64;
                done = done && !self.held.contains(part) && counts.lists >= groups;
            }
            if done {
                released.push(tree);
            }
        }
        Ok(released)
    }

    /// Continue another chain's tree: sizes must agree, one-list trees cannot
    /// be adopted, and both books share the same list/position counters.
    pub fn adopt(&mut self, target: usize, source: &TreeBook, source_tree: usize) -> Result<()> {
        let target_state = self.trees.get(target).ok_or(Error::InvalidIndex)?;
        let source_state = source.trees.get(source_tree).ok_or(Error::InvalidIndex)?;
        if target_state.parts.len() != source_state.parts.len()
            || target_state.one_list
            || source_state.one_list
        {
            return Err(Error::StatementMismatch);
        }
        self.trees[target].counts = Arc::clone(&source.trees[source_tree].counts);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle(id: u64, groups: Vec<u32>) -> OracleSpec {
        OracleSpec {
            id,
            groups,
            answers: 1,
            main_rows: 1,
            mask_rows: 1,
            main_hiding: 4,
            mask_hiding: 4,
        }
    }

    fn book() -> TreeBook {
        TreeBook::new(
            vec![TreeSpec {
                log_n: 3,
                parts: vec![1, 2],
            }],
            vec![oracle(1, vec![7]), oracle(2, vec![7])],
        )
        .unwrap()
    }

    #[test]
    fn position_lists_are_nonempty_and_short_lists_deduplicate() {
        assert!(PositionList::from_draws(Vec::new(), 2).is_err());
        assert!(PositionList::from_draws(vec![4], 2).is_err());
        let list = PositionList::from_draws(vec![4, 0, 4, 1], 3).unwrap();
        assert_eq!(list.size(), 3);
        assert_eq!(list.all(), &[0, 1, 4]);
        assert_eq!(list.draws(), &[4, 0, 4, 1]);
        let short = list.short_list(2).unwrap();
        assert_eq!(short.all(), &[0, 1]);
        assert_eq!(short.draws(), &[0, 0, 0, 1]);
        assert!(list.short_list(4).is_err());
    }

    #[test]
    fn empty_ids_are_a_noop_but_empty_draws_are_not() {
        let book = book();
        let at = PositionList::from_draws(vec![0], 3).unwrap();
        let opened = book.open_oracles(&[], &at).unwrap();
        assert!(opened.trees.is_empty());
        assert_eq!(book.counts(0).unwrap(), TreeCounts::default());
        assert!(book.read_opened(&[], &at).unwrap().trees.is_empty());
        assert!(book.open_oracles(&[1, 2], &at).is_ok());
        assert_eq!(book.counts(0).unwrap().lists, 1);
    }

    #[test]
    fn positions_count_cross_list_repeats_and_short_lists() {
        let book = book();
        let first = PositionList::from_draws(vec![0, 4], 3).unwrap();
        let second = PositionList::from_draws(vec![4, 0], 3).unwrap();
        book.open_oracles(&[1, 2], &first).unwrap();
        assert_eq!(book.counts(0).unwrap(), TreeCounts { lists: 1, positions: 2 });
        book.open_oracles(&[1, 2], &second).unwrap();
        assert_eq!(book.counts(0).unwrap(), TreeCounts { lists: 2, positions: 4 });
        let short_book = TreeBook::new(
            vec![TreeSpec {
                log_n: 2,
                parts: vec![1, 2],
            }],
            vec![oracle(1, vec![7]), oracle(2, vec![7])],
        )
        .unwrap();
        let at = PositionList::from_draws(vec![0, 4, 5], 3).unwrap();
        let opened = short_book.open_oracles(&[1, 2], &at).unwrap();
        assert_eq!(opened.trees[0].list.all(), &[0, 1]);
        assert_eq!(opened.trees[0].positions, 2);
    }

    #[test]
    fn whole_trees_and_group_agreement_are_required() {
        let book = book();
        let at = PositionList::from_draws(vec![0], 3).unwrap();
        assert!(book.open_oracles(&[1], &at).is_err());
        let mixed = TreeBook::new(
            vec![TreeSpec {
                log_n: 3,
                parts: vec![1, 2],
            }],
            vec![oracle(1, vec![7]), oracle(2, vec![8])],
        )
        .unwrap();
        assert!(mixed.open_oracles(&[1, 2], &at).is_err());
        assert_eq!(mixed.counts(0).unwrap().lists, 0);
    }

    #[test]
    fn one_list_trees_reject_second_lists_and_adoption() {
        let mut one = oracle(1, vec![7]);
        one.answers = 0;
        let book = TreeBook::new(
            vec![TreeSpec {
                log_n: 3,
                parts: vec![1],
            }],
            vec![one],
        )
        .unwrap();
        let at = PositionList::from_draws(vec![0], 3).unwrap();
        book.open_oracles(&[1], &at).unwrap();
        assert!(book.open_oracles(&[1], &at).is_err());
        let mut target = TreeBook::new(
            vec![TreeSpec {
                log_n: 3,
                parts: vec![1],
            }],
            vec![oracle(1, vec![7])],
        )
        .unwrap();
        assert!(target.adopt(0, &book, 0).is_err());
    }

    #[test]
    fn hiding_limits_reject_before_output_and_leave_counts_high() {
        let tight = OracleSpec {
            main_hiding: 2,
            mask_hiding: 2,
            ..oracle(1, vec![7])
        };
        let book = TreeBook::new(
            vec![TreeSpec {
                log_n: 3,
                parts: vec![1],
            }],
            vec![tight],
        )
        .unwrap();
        let first = PositionList::from_draws(vec![0, 1], 3).unwrap();
        let second = PositionList::from_draws(vec![2, 3], 3).unwrap();
        assert!(book.open_oracles(&[1], &first).is_ok());
        assert!(book.open_oracles(&[1], &second).is_err());
        assert_eq!(book.counts(0).unwrap(), TreeCounts { lists: 2, positions: 4 });
    }

    #[test]
    fn verifier_reads_count_lists_but_not_prover_positions() {
        let book = book();
        let at = PositionList::from_draws(vec![0, 1], 3).unwrap();
        let read = book.read_opened(&[1, 2], &at).unwrap();
        assert_eq!(read.trees.len(), 1);
        assert_eq!(book.counts(0).unwrap(), TreeCounts { lists: 1, positions: 0 });
        assert!(book.read_opened(&[1], &at).is_err());
    }

    #[test]
    fn adoption_shares_counts_and_release_waits_for_held_and_lists() {
        let source = book();
        let mut target = book();
        let at = PositionList::from_draws(vec![0, 1], 3).unwrap();
        source.open_oracles(&[1, 2], &at).unwrap();
        target.adopt(0, &source, 0).unwrap();
        assert_eq!(target.counts(0).unwrap(), TreeCounts { lists: 1, positions: 2 });
        target.set_held(1, true).unwrap();
        assert!(target.release_oracles(&[1, 2]).unwrap().is_empty());
        target.set_held(1, false).unwrap();
        assert_eq!(target.release_oracles(&[1, 2]).unwrap(), vec![0]);
    }
}
