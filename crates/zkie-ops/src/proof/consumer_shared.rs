//! The consumer shared LATER stage's column plan and durable charge.
//!
//! The stage reads a table batch of shared columns per run. The one-time
//! columns and the declared later batch form the instance total; the durable
//! table-life count is charged one fixed batch per run and is never reset by a
//! later run. The charge is the same primitive the table key stage uses.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::read::HidingCharge;

use crate::proof::consumer_request::ConsumerSharedPlan;
use crate::proof::hidden_send_session::{shared_column_charge, HiddenTableProver, ReadLedgerCell};

/// The fixed column geometry of one consumer shared run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerSharedColumns {
    pub one_time_columns: u64,
    pub later_columns: u64,
    pub declared_batches: u32,
    /// The columns one run draws: the one-time batch.
    pub column_draws: usize,
    /// The per-run draw limit; it equals the run's own draw count.
    pub column_budget: usize,
    /// The fixed batch size the durable table-life count reserves.
    pub batch_columns: u64,
    /// The instance total: one-time columns plus every declared later batch.
    pub max_columns: u64,
}

impl ConsumerSharedColumns {
    pub fn new(shared: &ConsumerSharedPlan) -> Result<Self> {
        if shared.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        // The approved later rule: one declared later batch whose columns are
        // one fixed batch, so the instance total is one-time plus that batch.
        // The row counts follow the later rule: a later goal cannot exceed the
        // one-time goal.
        if shared.declared_batches != 1
            || shared.later_columns != shared.one_time_columns
            || shared.m_later == 0
            || shared.m_later > shared.m_once
        {
            return Err(Error::InvalidConfig);
        }
        let max_columns = shared
            .one_time_columns
            .checked_add(
                shared
                    .later_columns
                    .checked_mul(u64::from(shared.declared_batches))
                    .ok_or(Error::ResourceLimit)?,
            )
            .ok_or(Error::ResourceLimit)?;
        if max_columns > shared.table_tail {
            return Err(Error::InvalidConfig);
        }
        let column_draws =
            usize::try_from(shared.one_time_columns).map_err(|_| Error::ResourceLimit)?;
        Ok(Self {
            one_time_columns: shared.one_time_columns,
            later_columns: shared.later_columns,
            declared_batches: shared.declared_batches,
            column_draws,
            column_budget: column_draws,
            batch_columns: shared.one_time_columns,
            max_columns,
        })
    }
}

/// The shared stage's column geometry and its verifier-side read ledger. The
/// ledger's maximum is the instance total; a run draws one batch.
#[derive(Clone, Debug)]
pub struct ConsumerSharedStage {
    pub columns: ConsumerSharedColumns,
    pub ledger: ReadLedgerCell,
}

impl ConsumerSharedStage {
    pub fn new(shared: &ConsumerSharedPlan) -> Result<Self> {
        let columns = ConsumerSharedColumns::new(shared)?;
        let ledger = ReadLedgerCell::new(columns.max_columns);
        Ok(Self { columns, ledger })
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.columns.column_draws == 0
            || self.columns.column_draws > self.columns.column_budget
            || self.columns.max_columns == 0
            || self.ledger.max_reads().ok() != Some(self.columns.max_columns)
        {
            return Some("the consumer shared column plan is not settled");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }
}

/// Charges one shared run: the durable table-life count reserves exactly one
/// fixed batch before any column is read, and the in-memory hiding state
/// begins the run's own column count. A plan whose batch differs from the
/// table's fixed batch is refused, and no charge is returned without a
/// durable reservation.
pub fn consumer_shared_charge(
    stage: &ConsumerSharedStage,
    table: &mut dyn HiddenTableProver,
) -> Result<HidingCharge> {
    stage.check()?;
    shared_column_charge(
        table,
        stage.columns.batch_columns,
        stage.columns.column_draws,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use zkie_engine::pcs::request::read::DurableColumnBudget;

    fn shared(one_time: u64, later: u64, batches: u32, tail: u64) -> ConsumerSharedPlan {
        ConsumerSharedPlan {
            m_once: 8,
            m_later: 8,
            one_time_columns: one_time,
            later_columns: later,
            declared_batches: batches,
            table_tail: tail,
        }
    }

    #[test]
    fn the_columns_follow_the_approved_later_rule() {
        let stage = ConsumerSharedStage::new(&shared(2, 2, 1, 4)).unwrap();
        assert_eq!(stage.columns.column_draws, 2);
        assert_eq!(stage.columns.column_budget, 2);
        assert_eq!(stage.columns.batch_columns, 2);
        assert_eq!(stage.columns.max_columns, 4);
        assert_eq!(stage.ledger.max_reads().unwrap(), 4);
        assert!(stage.fault().is_none());
        // Another later batch count, another batch size, a later goal above
        // the one-time goal and an instance total above the table tail are
        // refused.
        assert!(ConsumerSharedStage::new(&shared(2, 2, 2, 8)).is_err());
        assert!(ConsumerSharedStage::new(&shared(2, 1, 1, 4)).is_err());
        assert!(ConsumerSharedStage::new(&shared(2, 2, 1, 3)).is_err());
        let mut plan = shared(2, 2, 1, 4);
        plan.m_later = 9;
        assert!(ConsumerSharedStage::new(&plan).is_err());
        let mut plan = shared(2, 2, 1, 4);
        plan.one_time_columns = 0;
        assert!(ConsumerSharedStage::new(&plan).is_err());
    }

    #[test]
    fn the_durable_budget_reserves_one_fixed_batch_and_never_resets() {
        let root = std::env::temp_dir().join(format!(
            "zkie-consumer-shared-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let count: PathBuf = root.join("count.bin");
        DurableColumnBudget::initialize(&root, count.clone()).unwrap();
        let budget = DurableColumnBudget::open(&root, count, 2, 4).unwrap();
        assert_eq!(budget.columns(), 2);
        assert_eq!(budget.count().unwrap(), 0);
        budget.reserve().unwrap();
        budget.reserve().unwrap();
        assert_eq!(budget.count().unwrap(), 4);
        // The table-life limit is reached: no further batch is admitted and
        // the count is not reset.
        assert!(budget.reserve().is_err());
        assert_eq!(budget.count().unwrap(), 4);
        // Opening the same count again continues from it.
        let again = DurableColumnBudget::open(&root, root.join("count.bin"), 2, 4).unwrap();
        assert_eq!(again.count().unwrap(), 4);
        assert!(again.reserve().is_err());
        std::fs::remove_dir_all(&root).ok();
    }
}
