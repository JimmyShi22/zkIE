//! Same-source derivation of the request-level consumer plan.
//!
//! The static layer (model and asset identities, the public reader, the table
//! configuration and the qualification proof reference) lives in the commit
//! artifact. The request layer is derived here from the public request facts
//! and the validated sources: the trusted model plan's step/port/slice and
//! shift data, the same-instance table identity and range, the validated
//! activation/output digests and integer bounds, and the approved later plan.
//! The verifier derives the same plan and compares the canonical digest; it
//! never deserializes a plan and trusts it.

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::proof::consumer_psc::{ConsumerMaskSource, ConsumerObligation, ConsumerTable};
use crate::proof::consumer_request::{
    ConsumerIntegerRanges, ConsumerRequestPlan, ConsumerSharedPlan,
};

/// The public facts of one request that the plan derives from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestFacts {
    pub request: u64,
    /// The same-instance table identity and configuration, checked against the
    /// accepted qualification before this plan is derived.
    pub table: ConsumerTable,
    /// The mask allocation of the trusted model plan.
    pub mask: ConsumerMaskSource,
    /// The approved later plan and its durable budget values.
    pub shared: ConsumerSharedPlan,
    /// The integer bounds of the validated activation and weight ranges.
    pub ranges: ConsumerIntegerRanges,
    /// The digest of the validated request activation values.
    pub activation_digest: Digest,
    /// The digest of the validated public output rows.
    pub outputs_digest: Digest,
    /// The ordered head obligations of the trusted model plan.
    pub obligations: Vec<ConsumerObligation>,
}

impl ConsumerRequestFacts {
    pub fn fault(&self) -> Option<&'static str> {
        if self.request == 0
            || self.activation_digest == Digest::ZERO
            || self.outputs_digest == Digest::ZERO
            || self.obligations.is_empty()
        {
            return Some("a consumer request fact is empty");
        }
        if self.table.identity == Digest::ZERO
            || self.table.configuration == Digest::ZERO
            || self.table.root == Digest::ZERO
        {
            return Some("the consumer table identity is empty");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }
}

/// Derives the request-level consumer plan from the public facts. Every
/// obligation must name the same table as the facts; the registration
/// constructor checks the heads, the slices and the digests.
pub fn derive_consumer_request_plan(facts: &ConsumerRequestFacts) -> Result<ConsumerRequestPlan> {
    facts.check()?;
    for obligation in &facts.obligations {
        if obligation.table != facts.table {
            return Err(Error::StatementMismatch);
        }
    }
    let plan = ConsumerRequestPlan::new(
        facts.request,
        facts.table.clone(),
        facts.mask,
        facts.shared,
        facts.ranges,
        facts.activation_digest,
        facts.outputs_digest,
        facts.obligations.clone(),
    )?;
    plan.fault().map_or(Ok(plan), |_| Err(Error::InvalidConfig))
}

/// The canonical digest of the derived plan's statement, which can be bound
/// into the request statement or the first chain.
pub fn consumer_request_plan_digest(plan: &ConsumerRequestPlan) -> Result<Digest> {
    Ok(digest_bytes(&plan.statement_bytes()))
}

/// Re-derives the plan from the same facts and requires the expected digest.
/// A verifier that holds only the facts and the digest never trusts a
/// serialized plan.
pub fn check_consumer_request_plan(
    facts: &ConsumerRequestFacts,
    expected: &Digest,
) -> Result<ConsumerRequestPlan> {
    let plan = derive_consumer_request_plan(facts)?;
    if *expected == Digest::ZERO || consumer_request_plan_digest(&plan)? != *expected {
        return Err(Error::StatementMismatch);
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(identity: u8) -> ConsumerTable {
        ConsumerTable {
            path: "weights/L0_k_w_i32.bin".to_string(),
            oracle: 30,
            root: Digest([0x11; 32]),
            identity: Digest([identity; 32]),
            configuration: Digest([0x13; 32]),
        }
    }

    fn obligation(table: ConsumerTable, head: u32, step: u32) -> ConsumerObligation {
        ConsumerObligation {
            request: 7,
            source_step: step,
            activation_step: 9,
            head,
            shape: crate::proof::consumer_psc::ConsumerPscShape {
                positions: 3,
                places: 16,
                records: 16,
                slice: (u64::from(head) * 4, (u64::from(head) + 1) * 4),
            },
            shift: 16,
            table,
            mask: ConsumerMaskSource {
                oracle: 60,
                hiding_base: 100,
                shown_slot: 160,
            },
        }
    }

    fn facts() -> ConsumerRequestFacts {
        let table = table(0x12);
        ConsumerRequestFacts {
            request: 7,
            table: table.clone(),
            mask: ConsumerMaskSource {
                oracle: 60,
                hiding_base: 100,
                shown_slot: 160,
            },
            shared: ConsumerSharedPlan {
                m_once: 8,
                m_later: 8,
                one_time_columns: 2,
                later_columns: 2,
                declared_batches: 1,
                table_tail: 4,
            },
            ranges: ConsumerIntegerRanges {
                x_abs_bound: 1 << 20,
                weight_lo: -(1 << 18),
                weight_hi: (1 << 18) - 1,
            },
            activation_digest: Digest([0x41; 32]),
            outputs_digest: Digest([0x42; 32]),
            obligations: vec![obligation(table.clone(), 0, 20), obligation(table, 1, 21)],
        }
    }

    #[test]
    fn the_plan_derives_from_the_public_facts_and_rebuilds_the_same_digest() {
        let facts = facts();
        let plan = derive_consumer_request_plan(&facts).unwrap();
        let digest = consumer_request_plan_digest(&plan).unwrap();
        assert_ne!(digest, Digest::ZERO);
        let again = derive_consumer_request_plan(&facts).unwrap();
        assert_eq!(consumer_request_plan_digest(&again).unwrap(), digest);
        // The verifier rebuilds from the same facts and requires the digest.
        let checked = check_consumer_request_plan(&facts, &digest).unwrap();
        assert_eq!(checked, plan);
        assert!(check_consumer_request_plan(&facts, &Digest([0x55; 32])).is_err());
        // Another activation, output, table identity or obligation changes the
        // derived plan and therefore its digest.
        let mut changed = facts.clone();
        changed.activation_digest = Digest([0x43; 32]);
        assert_ne!(
            consumer_request_plan_digest(&derive_consumer_request_plan(&changed).unwrap()).unwrap(),
            digest
        );
        let mut changed = facts.clone();
        changed.outputs_digest = Digest([0x44; 32]);
        assert_ne!(
            consumer_request_plan_digest(&derive_consumer_request_plan(&changed).unwrap()).unwrap(),
            digest
        );
        let mut changed = facts.clone();
        changed.table = table(0x14);
        for obligation in &mut changed.obligations {
            obligation.table = changed.table.clone();
        }
        assert_ne!(
            consumer_request_plan_digest(&derive_consumer_request_plan(&changed).unwrap()).unwrap(),
            digest
        );
        // A facts table that its obligations do not name is refused.
        let mut bad = facts.clone();
        bad.table = table(0x15);
        assert!(derive_consumer_request_plan(&bad).is_err());
        let mut changed = facts.clone();
        changed.shared.m_later = 7;
        assert_ne!(
            consumer_request_plan_digest(&derive_consumer_request_plan(&changed).unwrap()).unwrap(),
            digest
        );
        // An obligation on another table than the facts is refused.
        let mut changed = facts;
        changed.obligations[0].table = table(0x15);
        assert!(derive_consumer_request_plan(&changed).is_err());
    }
}
