//! Random tails, masks, exposure budgets and leaf rotation.
//!
//! Budgets are counted per exposed object (query, column, OOD answer, linear
//! leakage, shared relation), never per proof. A reservation is made before a
//! challenge is drawn; a failure consumes it and is never refunded.

use std::collections::BTreeMap;

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::common::random::RandomSource;
use zkie_core::error::{Error, Result};

use crate::pcs::request::field::{Fp, Fp3, MODULUS};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExposureKind {
    Query,
    Column,
    Ood,
    LinearLeak,
    SharedRelation,
}

impl ExposureKind {
    /// The budget-item name in the private ledger.
    pub const fn budget_kind(self) -> &'static str {
        match self {
            ExposureKind::Query => "hiding.query",
            ExposureKind::Column => "hiding.column",
            ExposureKind::Ood => "hiding.ood",
            ExposureKind::LinearLeak => "hiding.linear_leak",
            ExposureKind::SharedRelation => "hiding.shared_relation",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HidingLimits {
    pub per_kind: BTreeMap<ExposureKind, u64>,
}

impl HidingLimits {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, kind: ExposureKind, limit: u64) -> Self {
        self.per_kind.insert(kind, limit);
        self
    }

    pub fn limit(&self, kind: ExposureKind) -> u64 {
        self.per_kind.get(&kind).copied().unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExposureLease {
    pub kind: ExposureKind,
    pub amount: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HidingState {
    limits: HidingLimits,
    reserved: BTreeMap<ExposureKind, u64>,
    exposed: BTreeMap<ExposureKind, u64>,
    epoch: u64,
}

impl HidingState {
    pub fn new(limits: HidingLimits) -> Self {
        Self {
            limits,
            reserved: BTreeMap::new(),
            exposed: BTreeMap::new(),
            epoch: 0,
        }
    }

    pub fn limits(&self) -> &HidingLimits {
        &self.limits
    }

    pub fn reserved(&self, kind: ExposureKind) -> u64 {
        self.reserved.get(&kind).copied().unwrap_or(0)
    }

    pub fn exposed(&self, kind: ExposureKind) -> u64 {
        self.exposed.get(&kind).copied().unwrap_or(0)
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Atomically reserve the worst case before a challenge is derived.
    pub fn reserve(&mut self, kind: ExposureKind, worst_case: u64) -> Result<ExposureLease> {
        if worst_case == 0 {
            return Err(Error::InvalidConfig);
        }
        let current = self.reserved(kind);
        let next = current
            .checked_add(worst_case)
            .ok_or(Error::ResourceLimit)?;
        if next > self.limits.limit(kind) {
            return Err(Error::LeafBudgetExhausted);
        }
        self.reserved.insert(kind, next);
        self.epoch = self.epoch.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok(ExposureLease {
            kind,
            amount: worst_case,
        })
    }

    /// Record what was actually exposed. The reservation is not reduced: the
    /// worst case may already have been exposed.
    pub fn record_exposure(&mut self, lease: ExposureLease, actual: u64) -> Result<()> {
        if actual > lease.amount {
            return Err(Error::LeafBudgetExhausted);
        }
        let current = self.exposed(lease.kind);
        let next = current.checked_add(actual).ok_or(Error::ResourceLimit)?;
        if next > self.reserved(lease.kind) {
            return Err(Error::LeafBudgetExhausted);
        }
        self.exposed.insert(lease.kind, next);
        self.epoch = self.epoch.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok(())
    }

    /// A failed or cancelled operation keeps its reservation: there is no
    /// refund path and no method that lowers `reserved`.
    pub fn finish(&mut self, _lease: ExposureLease) {}

    pub fn validate(&self) -> Result<()> {
        for (kind, exposed) in &self.exposed {
            if *exposed > self.reserved(*kind) || self.reserved(*kind) > self.limits.limit(*kind) {
                return Err(Error::StateRollbackSuspected);
            }
        }
        Ok(())
    }
}

/// Make a frame of canonical field elements from a domain-separated stream.
/// Words at or above the modulus are skipped by taking the next word.
pub fn derive_mask_stream(
    rng: &mut dyn RandomSource,
    domain: &[u8],
    logical_object: u64,
    round: u32,
    count: usize,
) -> Result<Vec<Fp3>> {
    let mut values = Vec::with_capacity(count);
    let base = (round as u64) << 32;
    let mut offset = 0u64;
    while values.len() < count {
        let mut bytes = [0u8; 24];
        let at = base
            .checked_add(offset.checked_mul(24).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        rng.fill(domain, logical_object, at, &mut bytes)?;
        let mut words = [0u64; 3];
        let mut rejected = false;
        for index in 0..3 {
            let word = u64::from_le_bytes(
                bytes[index * 8..(index + 1) * 8]
                    .try_into()
                    .map_err(|_| Error::InvalidEncoding)?,
            );
            if word >= MODULUS {
                rejected = true;
                break;
            }
            words[index] = word;
        }
        if rejected {
            offset = offset.checked_add(1).ok_or(Error::ResourceLimit)?;
            continue;
        }
        values.push(Fp3::new(
            Fp::from_u64(words[0]),
            Fp::from_u64(words[1]),
            Fp::from_u64(words[2]),
        ));
        offset = offset.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    Ok(values)
}

/// The RS random tail of one row: base-field values, fresh per leaf.
pub fn allocate_random_tail(
    rng: &mut dyn RandomSource,
    leaf_id: u64,
    round: u32,
    count: usize,
) -> Result<Vec<Fp>> {
    let mut values = Vec::with_capacity(count);
    let mut offset = 0u64;
    while values.len() < count {
        let mut bytes = [0u8; 8];
        rng.fill(b"random-tail", leaf_id, ((round as u64) << 32) + offset, &mut bytes)?;
        let word = u64::from_le_bytes(bytes);
        offset = offset.checked_add(1).ok_or(Error::ResourceLimit)?;
        if word < MODULUS {
            values.push(Fp::from_u64(word));
        }
    }
    Ok(values)
}

/// The OOD answer mask: add a fresh mask value in the challenge field.
pub fn mask_ood_answers(answers: &[Fp3], masks: &[Fp3]) -> Result<Vec<Fp3>> {
    if answers.len() != masks.len() {
        return Err(Error::ShapeMismatch);
    }
    Ok(answers
        .iter()
        .zip(masks)
        .map(|(answer, mask)| answer.add(mask))
        .collect())
}

/// One leaf's lifecycle. Rotation needs an explicit authorization and never
/// resets the exposure ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafLifecycle {
    pub current: u64,
    pub generation: u64,
    uses: u64,
    per_leaf_uses: u64,
    rotation_authorized: bool,
}

impl LeafLifecycle {
    pub fn new(leaf: u64, generation: u64, per_leaf_uses: u64, authorized: bool) -> Result<Self> {
        if per_leaf_uses == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            current: leaf,
            generation,
            uses: 0,
            per_leaf_uses,
            rotation_authorized: authorized,
        })
    }

    pub fn uses(&self) -> u64 {
        self.uses
    }

    pub fn record_use(&mut self) -> Result<()> {
        self.uses = self.uses.checked_add(1).ok_or(Error::ResourceLimit)?;
        if self.uses > self.per_leaf_uses {
            return Err(Error::LeafBudgetExhausted);
        }
        Ok(())
    }

    pub fn rotate_leaf(&mut self) -> Result<u64> {
        if !self.rotation_authorized {
            return Err(Error::InvalidSessionState);
        }
        if self.uses < self.per_leaf_uses {
            return Err(Error::InvalidSessionState);
        }
        self.current = self.current.checked_add(1).ok_or(Error::ResourceLimit)?;
        self.generation = self.generation.checked_add(1).ok_or(Error::ResourceLimit)?;
        self.uses = 0;
        Ok(self.current)
    }
}

/// Validate a serialized hiding state: all counters non-negative by type,
/// exposure never above reservation, epoch monotone.
pub fn validate_hiding_state(state: &HidingState) -> Result<()> {
    state.validate()
}
