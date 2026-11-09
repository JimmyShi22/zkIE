//! Isolated candidate for committed-table test state.
//!
//! This module models the public shape/pricing/standing comparison of a table
//! read. It is a candidate fixture boundary: no key/entry file check is
//! connected, no proof-reported state is trusted, and no cross-process state is
//! persisted. The formal production source of a positive tested state would be
//! the key/entry check paths; those are not implemented here. The reference
//! `UnprovedTable`/`IfTested` conditional interface is also absent, so a
//! missing test is not converted into a yes. A table-test value made by the
//! test-only fixture is an isolated sample and must never be promoted to a
//! production trusted state.

use zkie_core::error::{Error, Result};

use crate::pcs::request::code::CodeParams;
use crate::pcs::request::table::{validate_index_shape, IndexReader, IndexShape, TableDistances};

/// The complete candidate setting whose equality is required when a read says
/// that the table already passed its one-time test. The fields mirror the
/// setting inputs used by the isolated layer; absent fields remain production
/// gaps, not defaults.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableSetting {
    pub field_name: String,
    pub modulus: [u8; 32],
    pub degree: u32,
    pub circle: bool,
    pub f_bytes: u32,
    pub e_bytes: u32,
    pub digest_bytes: u32,
    pub salt_bytes: usize,
    pub nonce_bytes: u32,
    pub position_bytes: u32,
    pub hash_block_bytes: u32,
    pub label_bytes: u32,
    pub chain_extra_bytes: u32,
    pub chain_free_words: u32,
    pub target: u32,
    pub instances: u64,
    pub samples: u32,
    pub saved_coefficient: bool,
    pub work: [[u32; 3]; 6],
}

impl TableSetting {
    /// Full structural equality, not only target and work bits.
    pub fn same_setting(&self, other: &Self) -> bool {
        self == other
    }
}

/// Which test a round runs on a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableTest {
    Later,
    OneTime,
    Link,
}

/// What the plan's error bound was priced for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PricedTable {
    pub code: CodeParams,
    pub v: u32,
    pub answers: u32,
    pub distances: TableDistances,
    pub test: TableTest,
}

/// The candidate priced table for one run shape and test.
pub fn priced_table(shape: &IndexShape, test: TableTest) -> PricedTable {
    PricedTable {
        code: shape.code,
        v: shape.v,
        answers: shape.answers,
        distances: shape.distances,
        test,
    }
}

/// The standing that a caller says a read has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableStanding {
    UnderTest,
    AlreadyTested,
}

/// A candidate tested-table value. Its fields are private and the only
/// constructor is the test-only fixture below; a proof or ordinary caller
/// cannot mint it. A real positive source remains the key/entry check.
#[derive(Clone, Debug)]
pub struct TestedTable {
    reader: IndexReader,
    under: TableSetting,
}

impl TestedTable {
    pub fn reader(&self) -> &IndexReader {
        &self.reader
    }

    pub fn tested_under(&self) -> &TableSetting {
        &self.under
    }

    /// Test-only fixture: deliberately named to prevent it from being read as
    /// a production positive. It is compiled out of normal builds.
    #[cfg(test)]
    pub fn for_an_isolated_test(reader: IndexReader, under: TableSetting) -> Self {
        Self { reader, under }
    }
}

/// A candidate table read: either an untested reader or a tested fixture with
/// its setting and the caller's standing.
#[derive(Clone, Debug)]
pub struct TableRead {
    reader: IndexReader,
    tested_under: Option<TableSetting>,
    standing: TableStanding,
}

impl TableRead {
    pub fn from_untested(reader: IndexReader) -> Self {
        Self {
            reader,
            tested_under: None,
            standing: TableStanding::UnderTest,
        }
    }

    pub fn from_tested(tested: &TestedTable) -> Self {
        Self {
            reader: tested.reader.clone(),
            tested_under: Some(tested.under.clone()),
            standing: TableStanding::AlreadyTested,
        }
    }

    pub fn reader(&self) -> &IndexReader {
        &self.reader
    }

    pub fn standing(&self) -> TableStanding {
        self.standing
    }

    pub fn tested_under(&self) -> Option<&TableSetting> {
        self.tested_under.as_ref()
    }
}

/// Compare a candidate read with an independent trusted run shape, the plan's
/// priced table and the run setting. The run shape's code/v/answers/salt are
/// checked against the reader separately; distances are checked separately and
/// salt is deliberately not part of `PricedTable`.
///
/// ONE_TIME requires an untested standing. LATER requires a tested fixture and
/// full setting equality. LINK leaves the two-sided standing choice to the
/// trusted caller; an AlreadyTested side still needs its tested fixture and
/// setting equality, while an UnderTest side has no tested evidence.
pub fn check_table_read(
    read: &TableRead,
    run_shape: &IndexShape,
    priced: &PricedTable,
    setting: &TableSetting,
) -> Result<()> {
    if read.reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    validate_index_shape(run_shape)?;
    if read.reader.shape.code != run_shape.code
        || read.reader.shape.v != run_shape.v
        || read.reader.shape.answers != run_shape.answers
        || read.reader.shape.salt_bytes != run_shape.salt_bytes
    {
        return Err(Error::ShapeMismatch);
    }
    if read.reader.shape.distances != run_shape.distances {
        return Err(Error::StatementMismatch);
    }
    if priced_table(run_shape, priced.test) != *priced {
        return Err(Error::StatementMismatch);
    }
    match priced.test {
        TableTest::OneTime => {
            if read.standing != TableStanding::UnderTest || read.tested_under.is_some() {
                return Err(Error::StatementMismatch);
            }
        }
        TableTest::Later => {
            let tested = read.tested_under.as_ref().ok_or(Error::StatementMismatch)?;
            if read.standing != TableStanding::AlreadyTested
                || !tested.same_setting(setting)
            {
                return Err(Error::StatementMismatch);
            }
        }
        TableTest::Link => match read.standing {
            TableStanding::AlreadyTested => {
                let tested = read.tested_under.as_ref().ok_or(Error::StatementMismatch)?;
                if !tested.same_setting(setting) {
                    return Err(Error::StatementMismatch);
                }
            }
            TableStanding::UnderTest => {
                if read.tested_under.is_some() {
                    return Err(Error::StatementMismatch);
                }
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::table::{IndexRecord, TableDistances};
    use zkie_core::types::Digest;

    fn setting() -> TableSetting {
        TableSetting {
            field_name: "test-field".to_string(),
            modulus: [7; 32],
            degree: 3,
            circle: false,
            f_bytes: 8,
            e_bytes: 24,
            digest_bytes: 32,
            salt_bytes: 40,
            nonce_bytes: 8,
            position_bytes: 0,
            hash_block_bytes: 128,
            label_bytes: 16,
            chain_extra_bytes: 32,
            chain_free_words: 4,
            target: 128,
            instances: 1,
            samples: 2,
            saved_coefficient: true,
            work: [[0; 3]; 6],
        }
    }

    fn reader() -> IndexReader {
        IndexReader {
            shape: IndexShape {
                code: build_code_plan(4, 8, 1).unwrap(),
                v: 2,
                answers: 2,
                salt_bytes: 40,
                distances: TableDistances {
                    radius: 3,
                    later_radius: 4,
                    m_once: 5,
                    m_later: 6,
                },
            },
            configuration: Digest([7; 32]),
            root: Digest([8; 32]),
            record: IndexRecord {
                y: vec![Default::default(); 8],
            },
        }
    }

    fn priced_for(test: TableTest) -> PricedTable {
        priced_table(&reader().shape, test)
    }

    fn run_shape() -> IndexShape {
        reader().shape
    }

    #[test]
    fn priced_table_binds_all_shape_fields_and_test() {
        let base = priced_for(TableTest::OneTime);
        let shape = reader().shape;
        let mut changed = base.clone();
        changed.code.ell += 1;
        assert_ne!(changed, base);
        changed = base.clone();
        changed.v += 1;
        assert_ne!(changed, base);
        changed = base.clone();
        changed.answers += 1;
        assert_ne!(changed, base);
        changed = base.clone();
        changed.distances.radius += 1;
        assert_ne!(changed, base);
        changed = base.clone();
        changed.test = TableTest::Later;
        assert_ne!(changed, base);
        assert_eq!(priced_table(&shape, TableTest::OneTime), base);
    }

    #[test]
    fn standing_matches_one_time_later_and_link() {
        let set = setting();
        let untested = TableRead::from_untested(reader());
        let tested = TableRead::from_tested(&TestedTable::for_an_isolated_test(
            reader(),
            set.clone(),
        ));

        assert!(check_table_read(&untested, &run_shape(), &priced_for(TableTest::OneTime), &set).is_ok());
        assert!(check_table_read(&tested, &run_shape(), &priced_for(TableTest::OneTime), &set).is_err());
        assert!(check_table_read(&untested, &run_shape(), &priced_for(TableTest::Later), &set).is_err());
        assert!(check_table_read(&tested, &run_shape(), &priced_for(TableTest::Later), &set).is_ok());
        assert!(check_table_read(&untested, &run_shape(), &priced_for(TableTest::Link), &set).is_ok());
        assert!(check_table_read(&tested, &run_shape(), &priced_for(TableTest::Link), &set).is_ok());
        let mut other = set.clone();
        other.target += 1;
        assert!(check_table_read(&tested, &run_shape(), &priced_for(TableTest::Link), &other).is_err());
    }

    #[test]
    fn tested_state_requires_full_setting_equality() {
        let set = setting();
        let tested = TableRead::from_tested(&TestedTable::for_an_isolated_test(
            reader(),
            set.clone(),
        ));
        let mut changed = set.clone();
        changed.target += 1;
        assert!(check_table_read(&tested, &run_shape(), &priced_for(TableTest::Later), &changed).is_err());
        changed = set.clone();
        changed.samples += 1;
        assert!(check_table_read(&tested, &run_shape(), &priced_for(TableTest::Later), &changed).is_err());
        changed = set.clone();
        changed.work[2][1] += 1;
        assert!(check_table_read(&tested, &run_shape(), &priced_for(TableTest::Later), &changed).is_err());
        assert!(set.same_setting(&setting()));
        assert!(!set.same_setting(&changed));
    }

    #[test]
    fn priced_or_shape_mismatch_is_rejected() {
        let set = setting();
        let read = TableRead::from_untested(reader());
        let mut priced = priced_for(TableTest::OneTime);
        priced.distances.later_radius += 1;
        assert!(check_table_read(&read, &run_shape(), &priced, &set).is_err());
        let mut bad_reader = reader();
        bad_reader.shape.code.ell += 1;
        assert!(check_table_read(&TableRead::from_untested(bad_reader), &run_shape(), &priced_for(TableTest::OneTime), &set).is_err());
    }
    #[test]
    fn valid_salt_change_in_reader_is_rejected_by_the_run_shape() {
        let set = setting();
        let mut reader = reader();
        reader.shape.salt_bytes = 48;
        let run = run_shape();
        assert_eq!(run.salt_bytes, 40);
        let priced = priced_table(&run, TableTest::OneTime);
        assert!(check_table_read(&TableRead::from_untested(reader), &run, &priced, &set).is_err());
    }

}
